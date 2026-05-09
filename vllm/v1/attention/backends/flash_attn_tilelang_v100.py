# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang-optimized FlashAttention V100 backend for SM70.

Uses tilelang_fa_v100 JIT-compiled kernels for paged prefill.
Decode falls back to Triton (or SM70 decode kernel if FA-V100 is also available).
Same metadata builder as FLASH_ATTN_V100. No breaking changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn_v100 import (
    FlashAttnV100MetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

# Auto-discover tilelang + tilelang-fa-v100 from 3rdparty submodules
# (installed via cmake build, not pip, so won't be on sys.path by default)
_repo_root = Path(__file__).resolve().parents[4]
_tl_3rd = _repo_root / "3rdparty"
for _p in [_tl_3rd / "tilelang", _tl_3rd / "tilelang-fa-v100"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = init_logger(__name__)

_tilelang_paged_forward = None
_warned_missing = False
_warned_prefill = False
_logged_prefill = False


def _get_tilelang_ops():
    global _tilelang_paged_forward
    if _tilelang_paged_forward is not None:
        return _tilelang_paged_forward
    try:
        import tilelang_fa_v100
        _tilelang_paged_forward = getattr(tilelang_fa_v100, "tilelang_paged_forward", None)
    except ImportError:
        pass
    return _tilelang_paged_forward


class FlashAttnTileLangV100Impl(TritonAttentionImpl):
    """TileLang paged prefill, Triton fallback for decode and edge cases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tilelang_paged = _get_tilelang_ops()
        self.use_tilelang_paged = self.tilelang_paged is not None
        self._tilelang_paged_ready = False

    def _ensure_paged_ready(self):
        if self._tilelang_paged_ready:
            return True
        if not self.use_tilelang_paged:
            return False
        if self.attn_type != AttentionType.DECODER:
            return False
        if self.alibi_slopes is not None:
            return False
        if self.logits_soft_cap != 0:
            return False
        if self.sinks is not None:
            return False
        if self.sliding_window != (-1, -1):
            return False
        if self.kv_cache_dtype.startswith("fp8"):
            return False
        self._tilelang_paged_ready = True
        return True

    def _tilelang_paged_prefill(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output):
        global _warned_prefill

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(1)
        k_cache = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
        v_cache = value_cache if value_cache.is_contiguous() else value_cache.contiguous()

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table

        if attn_metadata.prefix_kv_lens is not None:
            prefix_kv_lens = attn_metadata.prefix_kv_lens
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            prefix_kv_lens = seq_lens - query_lens
            prefix_kv_lens = torch.clamp(prefix_kv_lens, min=0)

        torch.cuda.synchronize()
        causal = getattr(attn_metadata, "causal", True)

        _, softmax_lse = self.tilelang_paged(
            query, k_cache, v_cache, block_table, seq_lens,
            query_start_loc, prefix_kv_lens,
            out=out_view, block_size=k_cache.shape[1],
            softmax_scale=self.scale, causal=causal,
            num_kv_heads=key.shape[1],
        )

        if torch.isnan(output).any():
            if not _warned_prefill:
                logger.warning("FLASH_ATTN_TILELANG_V100 NaN, falling back to Triton.")
                _warned_prefill = True
            return TritonAttentionImpl.forward(
                self, layer, query, key, value, kv_cache,
                attn_metadata, output,
            )
        return softmax_lse

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        global _warned_missing, _warned_prefill, _logged_prefill

        if attn_metadata is None:
            assert output is not None
            return output.fill_(0)

        if not self.use_tilelang_paged and not _warned_missing:
            logger.warning(
                "FLASH_ATTN_TILELANG_V100: tilelang_fa_v100 not installed. "
                "Falling back to Triton."
            )
            _warned_missing = True

        causal = getattr(attn_metadata, "causal", True)

        if not self._supports_path():
            if not causal:
                raise RuntimeError(
                    "FLASH_ATTN_TILELANG_V100 cannot fall back for non-causal. "
                    "Ensure no alibi/softcap/sliding_window/fp8."
                )
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale, output_block_scale)

        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()

        if is_prefill:
            if is_capturing:
                if not causal:
                    raise RuntimeError("FLASH_ATTN_TILELANG_V100: non-causal prefill during CUDA graph capture.")
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, output_block_scale)
            if not self._ensure_paged_ready():
                if not causal:
                    raise RuntimeError("FLASH_ATTN_TILELANG_V100 paged kernel not ready for non-causal.")
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, output_block_scale)
            if not _logged_prefill:
                logger.info("FLASH_ATTN_TILELANG_V100 paged prefill path active.")
                _logged_prefill = True
            return self._tilelang_paged_prefill(
                layer, query, key, value, kv_cache, attn_metadata, output)

        # Decode: Triton path
        return super().forward(layer, query, key, value, kv_cache,
                               attn_metadata, output, output_scale, output_block_scale)

    def _supports_path(self):
        return (
            self.use_tilelang_paged
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and not self.kv_cache_dtype.startswith("fp8")
        )


class FlashAttnTileLangV100Backend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_TILELANG_V100"

    @staticmethod
    def get_impl_cls():
        return FlashAttnTileLangV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)
