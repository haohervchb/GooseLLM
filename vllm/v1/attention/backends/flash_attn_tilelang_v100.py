# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang-optimized FlashAttention V100 backend for SM70.
 
 Uses tilelang_fa_v100 JIT-compiled kernels for paged prefill AND decode.
 Prefill: direct paged kernel call.
 Decode:  pads single Q token to block_M=32 rows, calls same paged kernel,
          takes only row 0 of output.
 Same metadata builder as FLASH_ATTN_V100. No breaking changes.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Ensure local patched tilelang is found BEFORE site-packages
_tl_home = Path.home() / "tilelang"
if _tl_home.exists() and str(_tl_home) not in sys.path:
    sys.path.insert(0, str(_tl_home))

# Suppress noisy TVM FFI warnings
warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

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

# Auto-discover tilelang-fa-v100 from 3rdparty submodule
_repo_root = Path(__file__).resolve().parents[4]
_tl_3rd_fa = _repo_root / "3rdparty" / "tilelang-fa-v100"
if _tl_3rd_fa.exists() and str(_tl_3rd_fa) not in sys.path:
    sys.path.insert(0, str(_tl_3rd_fa))

logger = init_logger(__name__)

_tilelang_paged_forward = None
_tilelang_decode_forward = None
_warned_missing = False
_warned_prefill = False
_logged_prefill = False
_logged_decode = False


def _get_tilelang_ops():
    global _tilelang_paged_forward, _tilelang_decode_forward
    if _tilelang_paged_forward is None or _tilelang_decode_forward is None:
        try:
            import tilelang_fa_v100
            if _tilelang_paged_forward is None:
                _tilelang_paged_forward = getattr(tilelang_fa_v100, "tilelang_paged_forward", None)
            if _tilelang_decode_forward is None:
                _tilelang_decode_forward = getattr(tilelang_fa_v100, "tilelang_decode_forward", None)
        except ImportError:
            pass
    return _tilelang_paged_forward, _tilelang_decode_forward


class FlashAttnTileLangV100Impl(TritonAttentionImpl):
    """TileLang paged prefill and decode, Triton fallback for edge cases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tilelang_paged, self.tilelang_decode = _get_tilelang_ops()
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

        # Synchronize only outside CUDA graph capture
        if not (query.is_cuda and torch.cuda.is_current_stream_capturing()):
            torch.cuda.synchronize()
        causal = getattr(attn_metadata, "causal", True)

        # TileLang kernel requires page_block_size=16. Hybrid models (e.g.
        # Qwen3.5-122B-A10B) may use larger block_size (1056) to align
        # attention and Mamba page sizes. Reshape on the fly.
        actual_block_size = k_cache.shape[1]
        block_size = 16
        if actual_block_size != block_size:
            factor = actual_block_size // block_size
            k_cache = k_cache.reshape(-1, block_size, k_cache.shape[2],
                                      k_cache.shape[3])
            v_cache = v_cache.reshape(-1, block_size, v_cache.shape[2],
                                      v_cache.shape[3])
            B, M = block_table.shape
            arange = torch.arange(factor, device=block_table.device,
                                  dtype=block_table.dtype)
            block_table = (
                block_table.unsqueeze(-1) * factor + arange
            ).reshape(B, M * factor)

        _, softmax_lse = self.tilelang_paged(
            query, k_cache, v_cache, block_table, seq_lens,
            query_start_loc, prefix_kv_lens,
            out=out_view, block_size=block_size,
            softmax_scale=self.scale, causal=causal,
            num_kv_heads=key.shape[1],
        )

        if not (query.is_cuda and torch.cuda.is_current_stream_capturing()):
            if torch.isnan(output).any():
                if not _warned_prefill:
                    logger.warning("FLASH_ATTN_TILELANG_V100 NaN, falling back to Triton.")
                    _warned_prefill = True
                return TritonAttentionImpl.forward(
                    self, layer, query, key, value, kv_cache,
                    attn_metadata, output,
                )
        return output

    def _tilelang_decode(self, layer, query, key, value, kv_cache,
                         attn_metadata, output):
        """Decode: shared-memory softmax kernel (avoids 1D fragment layout conflict)."""
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()

        if not is_capturing:
            if kv_cache.numel() == 0:
                return output.fill_(0)
            if attn_metadata.seq_lens.max().item() == 0:
                return output.fill_(0)

        query_flat = query[:attn_metadata.num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(1)
        k_cache = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
        v_cache = value_cache if value_cache.is_contiguous() else value_cache.contiguous()

        result = self.tilelang_decode(
            query_flat, k_cache, v_cache, attn_metadata.block_table, attn_metadata.seq_lens,
            block_size=k_cache.shape[1],
            num_kv_heads=key.shape[1],
            softmax_scale=self.scale,
        )
        output[:attn_metadata.num_actual_tokens].copy_(result)

        if not is_capturing and torch.isnan(output).any():
            logger.warning("FLASH_ATTN_TILELANG_V100 decode NaN, falling back to Triton.")
            return TritonAttentionImpl.forward(
                self, layer, query, key, value, kv_cache,
                attn_metadata, output,
            )
        return output

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        global _warned_missing, _warned_prefill, _logged_prefill, _logged_decode

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
            if not self._ensure_paged_ready():
                if not causal:
                    raise RuntimeError("FLASH_ATTN_TILELANG_V100 paged kernel not ready for non-causal.")
                return super().forward(layer, query, key, value, kv_cache,
                                       attn_metadata, output, output_scale, output_block_scale)
            if not _logged_prefill and not is_capturing:
                logger.info("FLASH_ATTN_TILELANG_V100 paged prefill path active.")
                _logged_prefill = True
            return self._tilelang_paged_prefill(
                layer, query, key, value, kv_cache, attn_metadata, output)

        # Decode: Triton (faster than tilelang decode which wastes 15/16 MMA on padded rows)
        return super().forward(layer, query, key, value, kv_cache,
                               attn_metadata, output, output_scale, output_block_scale)

        # The tilelang decode kernel (shared-memory softmax) is available but slower
        # for single-token decode due to SM70 MMA requiring block_M >= 16.
        # Kept for reference in _tilelang_decode.

    def _supports_path(self):
        return (
            self.use_tilelang_paged
            and self.tilelang_decode is not None
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and not self.kv_cache_dtype.startswith("fp8")
            and self.sliding_window == (-1, -1)
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
