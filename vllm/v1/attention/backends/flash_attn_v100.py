# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Prefill uses the dense Flash V100 kernel for strict no-prefix cases.
Decode falls back to Triton attention (ai-bond does not provide paged decode).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_warned_prefill_fallback = False
_warned_feature_fallback = False
_warned_decode_fallback = False
_warned_missing_flash_ops = False
_logged_prefill_flash = False


def _iter_flash_attn_v100_roots():
    repo_root = Path(__file__).resolve().parents[4]
    env_root = os.environ.get("FLASH_ATTN_V100_DIR")
    candidate_roots = []
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    candidate_roots.extend([
        repo_root.parent / "flash-attention-v100-ai-bond",
        repo_root.parent / "flash-attention-v100",
        repo_root / "flash-attention-v100-ai-bond",
        repo_root / "flash-attention-v100",
        Path.home() / "flash-attention-v100-ai-bond",
        Path.home() / "flash-attention-v100",
    ])

    seen = set()
    for root in candidate_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        yield root


def _ensure_flash_attn_v100_on_path() -> bool:
    for root in _iter_flash_attn_v100_roots():
        if not (root / "flash_attn_v100" / "__init__.py").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return True
    return False


def _import_flash_attn_v100_module():
    try:
        import flash_attn_v100 as flash_attn_v100_mod

        return flash_attn_v100_mod
    except ImportError:
        if not _ensure_flash_attn_v100_on_path():
            return None
        try:
            import flash_attn_v100 as flash_attn_v100_mod

            return flash_attn_v100_mod
        except ImportError:
            return None


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func
    flash_attn_v100_mod = _import_flash_attn_v100_module()
    if flash_attn_v100_mod is not None:
        if _flash_attn_func is None:
            _flash_attn_func = getattr(flash_attn_v100_mod, "flash_attn_func", None)
    return _flash_attn_func


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with strict fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attn_func = _get_flash_ops()
        self.use_flash_v100 = self.flash_attn_func is not None

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and not self.kv_cache_dtype.startswith("fp8")
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use dense Flash V100 only when there is no prefix context.
        - Decode: always falls back to Triton (ai-bond has no paged decode kernel).
        """
        global _warned_prefill_fallback, _warned_feature_fallback
        global _warned_decode_fallback, _warned_missing_flash_ops
        global _logged_prefill_flash

        if attn_metadata is None:
            assert output is not None
            return output.fill_(0)

        if not self.use_flash_v100 and not _warned_missing_flash_ops:
            logger.warning(
                "FLASH_ATTN_V100 backend selected, but optional module "
                "'flash_attn_v100' is unavailable. Falling back to Triton "
                "attention paths."
            )
            _warned_missing_flash_ops = True

        if not self._supports_flash_v100_path():
            if self.use_flash_v100 and not _warned_feature_fallback:
                logger.warning(
                    "FLASH_ATTN_V100 fallback to Triton due to unsupported "
                    "attention features (alibi/softcap/sliding window/fp8/etc)."
                )
                _warned_feature_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()

        if is_prefill:
            if is_capturing:
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback during CUDA graph "
                        "capture. Using Triton path for capture safety."
                    )
                    _warned_prefill_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if key.shape[1] == 0 or query.shape[1] % key.shape[1] != 0:
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback: unsupported Q/KV head "
                        "layout for grouped attention. Using Triton for correctness."
                    )
                    _warned_prefill_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if _has_prefix_context(attn_metadata):
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback: detected prefix/chunked "
                        "prefill (seq_len > query_len). Using Triton for correctness."
                    )
                    _warned_prefill_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            return self._flash_v100_prefill(query, key, value, attn_metadata, output)

        if not _warned_decode_fallback:
            logger.info(
                "FLASH_ATTN_V100 decode path: using Triton (ai-bond has no paged decode kernel)."
            )
            _warned_decode_fallback = True
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def _flash_v100_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for no-prefix case (query_len == seq_len per sequence).

        Batches all sequences into a single kernel call using (B, M, H, D) format.
        The ai-bond kernel handles GQA/MQA internally — no repeat_interleave needed.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        num_heads_q = query.shape[1]
        num_heads_kv = key.shape[1]
        head_dim = query.shape[2]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu if query_start_loc_cpu is not None else attn_metadata.query_start_loc
        )

        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens_tensor = (
            seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        )
        seq_lens = [int(seq_lens_tensor[i].item()) for i in range(len(seq_lens_tensor))]

        num_seqs = len(seq_lens)
        max_seq_len = max(seq_lens)

        q_padded = torch.zeros(
            (num_seqs, max_seq_len, num_heads_q, head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        k_padded = torch.zeros(
            (num_seqs, max_seq_len, num_heads_kv, head_dim),
            dtype=key.dtype,
            device=key.device,
        )
        v_padded = torch.zeros(
            (num_seqs, max_seq_len, num_heads_kv, head_dim),
            dtype=value.dtype,
            device=value.device,
        )

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            length = end - start
            q_padded[i, :length] = query[start:end]
            k_padded[i, :length] = key[start:end]
            v_padded[i, :length] = value[start:end]

        out_padded = self.flash_attn_func(
            q_padded,
            k_padded,
            v_padded,
            causal=True,
            softmax_scale=self.scale,
        )

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            length = end - start
            out_view[start:end] = out_padded[i, :length]

        return output


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Use HND physical layout for V100 decode. The semantic shape stays
        # [num_blocks, 2, block_size, num_kv_heads, head_size], but the raw
        # allocation is [num_blocks, 2, num_kv_heads, block_size, head_size],
        # which gives decode a better access pattern while preserving the
        # existing cache update and prefill interfaces.
        logger.info_once(
            "FLASH_ATTN_V100 using HND physical KV cache layout for decode.")
        if include_num_layers_dimension:
            return (1, 0, 2, 4, 3, 5)
        return (0, 1, 3, 2, 4)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Flash Attention V100 requires head_dim % 8 == 0.
        return [64, 80, 96, 112, 128, 256]
