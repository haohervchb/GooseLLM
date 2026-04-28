# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing import Any
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.worker.utils import bind_kv_cache
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

        self._init_parallel_drafting_params()

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
        self._draft_kv_initialized = False
        self._scratch_block_id = -1

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
        dflash_config = getattr(model_hf_config, "dflash_config", {})
        self.parallel_drafting_token_id = dflash_config.get("mask_token_id", 151643)

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

        # NOTE: DFlash does NOT use combine_hidden_states; the fc layer in the
        # draft model maps concatenated multi-layer features back to hidden_size,
        # but during inference the draft model only returns the final hidden state.
        # The upstream DFlash proposer never calls combine_hidden_states.

        # Run DFlash-specific input setup (non-causal attention)
        if not self._draft_kv_initialized:
            self._allocate_draft_kv_cache()

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

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                num_tokens_unpadded=num_query, num_tokens_padded=num_query
            )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens

        # Pre-insert context KVs directly into cache (DFlash-specific)
        num_context = self._dflash_num_context
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
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

        # DEBUG: detect NaN in draft hidden states
        if torch.isnan(ret_hidden_states).any():
            logger.warning(
                "DFlash propose: ret_hidden_states contains NaN (shape=%s)",
                ret_hidden_states.shape,
            )
        if torch.isinf(ret_hidden_states).any():
            logger.warning(
                "DFlash propose: ret_hidden_states contains Inf (shape=%s)",
                ret_hidden_states.shape,
            )

        if not self.model_returns_tuple():
            sample_hidden_states = ret_hidden_states[token_indices_to_sample]
        else:
            _, sample_hidden_states = ret_hidden_states[token_indices_to_sample]

        # Greedy sample from mask token positions
        if self.use_local_argmax_reduction:
            draft_token_ids = self.model.get_top_tokens(sample_hidden_states)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
            # DEBUG: check for NaN/Inf in draft logits
            if logits is not None:
                has_nan = torch.isnan(logits).any().item()
                has_inf = torch.isinf(logits).any().item()
                if has_nan or has_inf:
                    logger.warning(
                        "DFlash draft logits NaN=%s Inf=%s stats: min=%s max=%s mean=%s",
                        has_nan,
                        has_inf,
                        logits.min().item() if not has_nan else "nan",
                        logits.max().item() if not has_nan else "nan",
                        logits.mean().item() if not has_nan else "nan",
                    )
            draft_token_ids = logits.argmax(dim=-1)

        # DEBUG: log draft token distribution
        unique, counts = torch.unique(draft_token_ids, return_counts=True)
        top_counts, top_idx = torch.topk(counts, min(5, len(counts)))
        logger.info(
            "DFlash draft tokens unique=%d top_ids=%s top_counts=%s",
            len(unique),
            unique[top_idx].tolist(),
            top_counts.tolist(),
        )

        # Reshape to [batch_size, num_spec_tokens]
        # draft_token_ids is [batch_size * num_speculative_tokens]
        return draft_token_ids.view(batch_size, self.num_speculative_tokens)

    def _allocate_draft_kv_cache(self):
        """Allocate and bind KV cache for the draft model's attention layers."""
        if self._draft_kv_initialized:
            return
        
        if self.runner is None or not hasattr(self.runner, "kv_caches") or not self.runner.kv_caches:
            logger.warning("DFlash: target model KV cache not available for binding yet")
            return

        # Find target model's KV cache to get block info
        target_kv_caches = self.runner.kv_caches
        sample_cache = target_kv_caches[0]
        # GooseLLM convention: kv_cache can be a list of tensors
        if isinstance(sample_cache, list):
            sample_cache = sample_cache[0]
        
        target_num_blocks = sample_cache.shape[0]
        # CAP draft blocks to save memory (e.g. 2048 blocks = 32k tokens)
        self.num_draft_blocks = min(target_num_blocks, 2048)
        block_size = sample_cache.shape[2]
        self._scratch_block_id = self.num_draft_blocks - 1
        
        logger.info("DFlash: allocating draft KV cache with %d blocks (capped from %d), block_size=%d", 
                    self.num_draft_blocks, target_num_blocks, block_size)

        # Map layers to their caches for binding
        kv_caches_to_bind = {}
        
        # DFlashQwen3ForCausalLM has self.model (DFlashQwen3Model)
        draft_model = self.model
        if hasattr(draft_model, "model"):
            draft_model = draft_model.model
            
        for layer in draft_model.layers:
            # Each layer is a DFlashQwen3DecoderLayer
            attn_layer = layer.self_attn
            layer_name = attn_layer.layer_name
            num_kv_heads = attn_layer.num_kv_heads
            head_dim = attn_layer.head_dim
            
            # Shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
            shape = (self.num_draft_blocks, 2, block_size, num_kv_heads, head_dim)
            # Allocate (use zeros to avoid NaNs if uninitialized parts are read)
            kv_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
            kv_caches_to_bind[layer_name] = kv_cache

        # Bind to draft model's attention layers
        # forward_context here is just a dict to satisfy bind_kv_cache's signature
        # as it expects objects with a .kv_cache attribute.
        dummy_forward_context = {}
        for layer in draft_model.layers:
            dummy_forward_context[layer.self_attn.layer_name] = layer.self_attn.attn

        # bind_kv_cache(kv_caches_dict, forward_context_dict, runner_kv_caches_list)
        # We don't want to add them to the runner's list, so we pass a dummy list.
        dummy_runner_list = []
        bind_kv_cache(kv_caches_to_bind, dummy_forward_context, dummy_runner_list)
        
        # Also need to call _build_fused_kv_buffers on the model so it can
        # find the newly bound kv_cache during precompute.
        if hasattr(draft_model, "_build_fused_kv_buffers"):
            draft_model._build_fused_kv_buffers()

        self._draft_kv_initialized = True
        logger.info("DFlash: draft KV cache allocated and bound successfully")

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

        # Create a draft-specific block table by aliasing the target model's blocks
        # to fit in the smaller draft KV cache.
        draft_block_table = cad.block_table_tensor % self.num_draft_blocks

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
            block_table_ptr=draft_block_table,
            block_table_stride=draft_block_table.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            scratch_block_id=self._scratch_block_id,
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
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=draft_block_table,
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

        DFlash draft model is very small (5 layers); CUDA graphs offer
        little benefit and can trigger custom-all-reduce IPC issues when
        captured inside the target model's graph capture context.
        We force NONE to keep the draft model on the eager path.
        """
        self.cudagraph_dispatcher.initialize_cudagraph_keys(CUDAGraphMode.NONE)

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.compilation.backends import set_model_tag

        draft_model_config = self.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
            ).keys()
        )

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        draft_attn_layer_names = (
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
            ).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)
        self.indexer_layer_names = []

        # Share embed_tokens with the target model if the draft checkpoint
        # does not contain its own embedding weights.
        if get_pp_group().world_size == 1:
            if isinstance(target_model, SupportsMultiModal):
                target_language_model = target_model.get_language_model()
            else:
                target_language_model = target_model
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is not None:
                if hasattr(inner_model, "embed_tokens"):
                    target_embed_tokens = inner_model.embed_tokens
                elif hasattr(inner_model, "embedding"):
                    target_embed_tokens = inner_model.embedding
                else:
                    target_embed_tokens = None

                has_own_embed = getattr(self.model, "has_own_embed_tokens", False)
                logger.info(
                    "DFlash load_model: target_embed_tokens=%s has_own_embed_tokens=%s",
                    target_embed_tokens is not None,
                    has_own_embed,
                )
                if target_embed_tokens is not None and not has_own_embed:
                    logger.info(
                        "DFlash draft model does not have its own embed_tokens. "
                        "Sharing target model embedding weights with the draft model."
                    )
                    if hasattr(self.model.model, "embed_tokens"):
                        del self.model.model.embed_tokens
                    self.model.model.embed_tokens = target_embed_tokens
                elif target_embed_tokens is not None:
                    logger.info(
                        "DFlash draft model has its own embed_tokens. Keeping separate."
                    )
                else:
                    logger.warning(
                        "DFlash draft model: target_embed_tokens not found."
                    )

            # Share lm_head with the target model if the draft checkpoint
            # does not contain its own lm_head weights.
            target_lm_head = getattr(target_language_model, "lm_head", None)
            has_own_lm_head = getattr(self.model, "has_own_lm_head", False)
            logger.info(
                "DFlash load_model: target_lm_head=%s has_own_lm_head=%s",
                target_lm_head is not None,
                has_own_lm_head,
            )
            if target_lm_head is not None and not has_own_lm_head:
                logger.info(
                    "DFlash draft model does not have its own lm_head. "
                    "Sharing target model lm_head weights with the draft model."
                )
                if hasattr(self.model, "lm_head"):
                    del self.model.lm_head
                self.model.lm_head = target_lm_head
            elif target_lm_head is not None:
                logger.info(
                    "DFlash draft model has its own lm_head. Keeping separate."
                )

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for DFlash layers.
        Returns the first builder that has any of the DFlash layer names.
        """
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        builder = None
        chosen_layer = self.attn_layer_names[0]

        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            f"No attention metadata builder found for DFlash layer: {chosen_layer}"
        )
        return builder

    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings=None,
    ) -> None:
        """Warm-up the draft model to trigger torch.compile tracing.

        DFlash only needs one forward pass due to parallel drafting.
        We deliberately avoid CUDA graphs for the draft model to prevent
        nested-capture issues with custom all-reduce.
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_query_tokens, num_tokens_padded=num_query_tokens
        )

        # Force eager execution for the draft model warm-up
        cudagraph_runtime_mode = CUDAGraphMode.NONE
        num_input_tokens = num_tokens_dp_padded
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        slot_mapping_dict = self._get_slot_mapping(num_input_tokens)

        # Use dummy buffers for context states; no actual KV write happens
        # because precompute_and_store_context_kv returns early when
        # context_slot_mapping is None.
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
