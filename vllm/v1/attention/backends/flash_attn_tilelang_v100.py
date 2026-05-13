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
_logged_mixed = False
_diag_done = False


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
        if self.sliding_window != (-1, -1):
            return False
        if self.kv_cache_dtype.startswith("fp8"):
            return False
        self._tilelang_paged_ready = True
        return True

    def _tilelang_paged_prefill(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output):
        global _warned_prefill, _diag_done

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

        causal = getattr(attn_metadata, "causal", True)

        # TileLang kernel requires page_block_size=16. Hybrid models (e.g.
        # Qwen3.5-122B-A10B) may use larger block_size (1056) to align
        # attention and Mamba page sizes. Reshape on the fly: each 1056-token
        # page becomes 66 sub-pages of 16 tokens.
        actual_block_size = k_cache.shape[1]
        block_size = 16
        if actual_block_size != block_size:
            factor = actual_block_size // block_size
            # Reshape K/V: [N, 1056, Hkv, D] -> [N*66, 16, Hkv, D]
            k_cache = k_cache.reshape(-1, block_size, k_cache.shape[2],
                                      k_cache.shape[3])
            v_cache = v_cache.reshape(-1, block_size, v_cache.shape[2],
                                      v_cache.shape[3])
            # Expand block_table: [B, M] -> [B, M*66]
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
            if not _diag_done:
                _diag_done = True
                # One-shot correctness check: compare with Triton
                try:
                    triton_out = torch.zeros_like(output)
                    TritonAttentionImpl.forward(
                        self, layer, query, key, value, kv_cache,
                        attn_metadata, triton_out,
                    )
                    tl_slice = output[:num_actual_tokens]
                    tr_slice = triton_out[:num_actual_tokens]
                    diff = (tl_slice - tr_slice).abs()
                    logger.info(
                        "TILELANG vs Triton: max_diff=%.6f, mean_diff=%.6f, "
                        "tl_absmax=%.4f, tr_absmax=%.4f, any_nan_tl=%s, any_nan_tr=%s",
                        diff.max().item(), diff.mean().item(),
                        tl_slice.abs().max().item(), tr_slice.abs().max().item(),
                        torch.isnan(tl_slice).any().item(),
                        torch.isnan(tr_slice).any().item(),
                    )
                except Exception as e:
                    logger.warning("TILELANG comparison failed: %s", e)

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
            block_size=16,
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

    def _tilelang_mixed_forward(self, layer, query, key, value, kv_cache,
                                 attn_metadata, output):
        """Mixed prefill+decode: prefill via TileLang paged, decode via Triton."""
        qsl = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        query_lens = qsl[1:] - qsl[:-1]
        is_pf = query_lens > 1
        pf_idx = torch.where(is_pf)[0]
        dec_idx = torch.where(~is_pf)[0]

        # Decode elements via Triton
        if len(dec_idx) > 0:
            dec_total = len(dec_idx)
            dec_starts = qsl[dec_idx]

            dec_query = query.new_empty(dec_total, *query.shape[1:])
            for i, s in enumerate(dec_starts):
                dec_query[i] = query[s]

            dec_meta = TritonAttentionMetadata(
                num_actual_tokens=dec_total,
                max_query_len=1,
                query_start_loc=torch.arange(
                    dec_total + 1, dtype=qsl.dtype, device=qsl.device),
                max_seq_len=int(seq_lens[dec_idx].max().item()),
                seq_lens=seq_lens[dec_idx],
                block_table=block_table[dec_idx],
                slot_mapping=attn_metadata.slot_mapping,
                seq_threshold_3D=attn_metadata.seq_threshold_3D,
                num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                softmax_segm_output=attn_metadata.softmax_segm_output,
                softmax_segm_max=attn_metadata.softmax_segm_max,
                softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )

            dec_out = torch.empty_like(dec_query)
            super().forward(layer, dec_query, key, value, kv_cache,
                            dec_meta, dec_out)

            for i, idx in enumerate(dec_idx.tolist()):
                output[qsl[idx]:qsl[idx + 1]] = dec_out[i:i + 1]

        # Prefill elements via TileLang
        if len(pf_idx) > 0:
            pf_starts = qsl[pf_idx]
            pf_ends = qsl[pf_idx + 1]
            pf_lens = pf_ends - pf_starts
            pf_total = pf_lens.sum().item()

            pf_query = query.new_empty(pf_total, *query.shape[1:])
            offset = 0
            for i, idx in enumerate(pf_idx.tolist()):
                n = (qsl[idx + 1] - qsl[idx]).item()
                pf_query[offset:offset + n] = query[
                    qsl[idx]:qsl[idx + 1]]
                offset += n

            pf_qsl = torch.zeros(len(pf_idx) + 1, dtype=qsl.dtype,
                                 device=qsl.device)
            pf_qsl[1:] = pf_lens.cumsum(dim=0)

            pf_meta = TritonAttentionMetadata(
                num_actual_tokens=pf_total,
                max_query_len=int(pf_lens.max().item()),
                query_start_loc=pf_qsl,
                max_seq_len=attn_metadata.max_seq_len,
                seq_lens=seq_lens[pf_idx],
                block_table=block_table[pf_idx],
                slot_mapping=attn_metadata.slot_mapping,
                seq_threshold_3D=attn_metadata.seq_threshold_3D,
                num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                softmax_segm_output=attn_metadata.softmax_segm_output,
                softmax_segm_max=attn_metadata.softmax_segm_max,
                softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )
            pf_meta.causal = getattr(attn_metadata, "causal", True)

            pf_out = torch.empty_like(pf_query)
            self._tilelang_paged_prefill(layer, pf_query, key, value,
                                         kv_cache, pf_meta, pf_out)

            offset = 0
            for idx in pf_idx.tolist():
                n = (qsl[idx + 1] - qsl[idx]).item()
                output[qsl[idx]:qsl[idx + 1]] = pf_out[offset:offset + n]
                offset += n

        return output

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        global _warned_missing, _warned_prefill, _logged_prefill, _logged_decode, _logged_mixed

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

            # Mixed batch: dispatch prefill and decode separately
            if not is_capturing:
                query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
                has_decode = (query_lens == 1).any().item()
                if has_decode:
                    if not _logged_mixed:
                        logger.info(
                            "FLASH_ATTN_TILELANG_V100 mixed batch: prefill via "
                            "TileLang, decode via Triton.")
                        _logged_mixed = True
                    return self._tilelang_mixed_forward(
                        layer, query, key, value, kv_cache,
                        attn_metadata, output)

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
