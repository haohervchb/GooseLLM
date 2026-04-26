# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    """DFlash draft model proposer for speculative decoding.

    Uses non-causal (bidirectional) attention to generate all draft tokens
    in a single forward pass (parallel drafting), conditioned on hidden states
    from the target model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        self.method = "dflash"

        # DFlash: only next_token_ids + mask tokens are query tokens
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        pass

    @property
    @override
    def block_size(self) -> int:
        return self.num_speculative_tokens + 1

    def _init_parallel_drafting_params(self):
        """Initialize DFlash mask token handling for parallel drafting."""
        model_hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        else:
            raise ValueError(
                "For DFlash parallel drafting, the draft model config must have "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        # Initialize hidden state tensor for mask embeddings
        self.parallel_drafting_hidden_state_tensor = torch.empty(
            self.hidden_size, dtype=self.dtype, device=self.device
        )

    @override
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()

        # Combine hidden states for DFlash (auxiliary layer extraction)
        if getattr(self, "eagle3_use_aux_hidden_state", False) and hasattr(
            self.model, "combine_hidden_states"
        ):
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )

        # Run DFlash-specific input setup (non-causal attention)
        num_query, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        # Build attention metadata
        assert self.runner is not None
        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
            for layer_name in self.indexer_layer_names:
                per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._pad_batch_across_dp(
                num_tokens_unpadded=num_query, num_tokens_padded=num_query
            )
        )

        # Pre-insert context KVs directly into cache (DFlash-specific)
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[: common_attn_metadata.num_actual_tokens],
            self._context_slot_mapping_buffer[: common_attn_metadata.num_actual_tokens],
        )

        # Build model inputs
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None
        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions_dflash(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping={name: self._slot_mapping_buffer[:num_input_tokens] for name in self.attn_layer_names},
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                sample_hidden_states = ret_hidden_states[token_indices_to_sample]
            else:
                _, sample_hidden_states = ret_hidden_states[token_indices_to_sample]

        # Greedy sample from mask token positions
        if self.use_local_argmax_reduction:
            draft_token_ids = self.model.get_top_tokens(sample_hidden_states)
        else:
            draft_token_ids = self.model.compute_logits(sample_hidden_states).argmax(
                dim=-1
            )

        # Reshape to [batch_size, num_spec_tokens]
        # draft_token_ids is [batch_size * num_speculative_tokens]
        return draft_token_ids.view(batch_size, self.num_speculative_tokens)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=(
                cad.seq_lens_cpu_upper_bound + num_query_per_req
                if cad.seq_lens_cpu_upper_bound is not None
                else None
            ),
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=False,
        )

        return num_query_total, token_indices_to_sample, new_cad

    def _get_positions_dflash(self, num_tokens: int) -> torch.Tensor:
        """Get positions for DFlash (query tokens only)."""
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    @override
    def _get_slot_mapping(
        self, num_tokens: int, slot_mapping=None
    ) -> dict[str, torch.Tensor]:
        return {name: self._slot_mapping_buffer[:num_tokens] for name in self.attn_layer_names}

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        # DFlash requires non-causal attention
        if common_attn_metadata.causal:
            raise ValueError(
                "DFlash requires non-causal attention (causal=False). "
                "Please use FlashAttention or another backend that supports "
                "non-causal attention for speculative decoding."
            )
        per_group_attn_metadata = []
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = common_attn_metadata
            per_group_attn_metadata.append(common_attn_metadata)
        return per_group_attn_metadata, per_layer_attn_metadata

    @override
    def model_returns_tuple(self) -> bool:
        return False

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for DFlash.

        DFlash only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            dflash_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            dflash_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(dflash_cudagraph_mode)

    def _get_attention_metadata_builder(self):
        assert self.runner is not None
        return self.runner._make_spec_decode_attn_metadata_builder_()

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings=None,
    ) -> None:
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_query_tokens, num_tokens_padded=num_query_tokens
        )
        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        slot_mapping_dict = self._get_slot_mapping(num_input_tokens)

        context_positions = self._context_positions_buffer[:num_tokens]
        context_states = self.hidden_states[:num_tokens]

        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions_dflash(num_input_tokens),
                inputs_embeds=None,
            )

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state
