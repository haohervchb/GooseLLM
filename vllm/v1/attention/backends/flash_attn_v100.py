# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Prefill uses the Flash V100 paged kernel (block-table KV cache).
Decode uses SM70 decode kernel or Triton attention.

When this backend is active, the model runner automatically increases
max_num_tokens from the scheduler default (typically 8192-16384) to 65536,
reducing the number of chunks for long prompts without sacrificing
decode throughput or VRAM. The FA2 paged kernel handles prefix/chunked
prefill efficiently per-sequence.
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
_flash_attn_paged = None
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
    # In-tree path (unified repo) — preferred
    candidate_roots.append(repo_root / "csrc" / "flash_attention_v100")
    # Legacy out-of-tree paths (backward compatibility)
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
    global _flash_attn_func, _flash_attn_paged
    flash_attn_v100_mod = _import_flash_attn_v100_module()
    if flash_attn_v100_mod is not None:
        if _flash_attn_func is None:
            _flash_attn_func = getattr(flash_attn_v100_mod, "flash_attn_func", None)
        if _flash_attn_paged is None:
            _flash_attn_paged = getattr(
                flash_attn_v100_mod, "flash_attn_paged_forward", None
            )
    return _flash_attn_func, _flash_attn_paged


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the paged prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with paged prefill support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attn_func, self.flash_attn_paged = _get_flash_ops()
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_paged = self.flash_attn_paged is not None
        self._flash_attn_paged_ready = False

    def _ensure_paged_ready(self):
        """Validate paged kernel is ready (head dim, device, etc.)."""
        if self._flash_attn_paged_ready:
            return True
        if not self.use_flash_paged:
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
        # NOTE: the paged kernel grid is (q_tiles, seqs, num_heads); each block
        # handles one head independently, so any positive num_heads works.
        self._flash_attn_paged_ready = True
        return True

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

        - Prefill: use Flash V100 paged kernel (handles prefix/chunked).
        - Decode: uses SM70 decode kernel (via Triton super).
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
            if not self._ensure_paged_ready():
                if self.use_flash_paged and not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback: paged kernel not ready "
                        "(head dim, device, or feature mismatch). Using Triton."
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
                    "FLASH_ATTN_V100 paged prefill path active (handles prefix/chunked)."
                )
                _logged_prefill_flash = True
            return self._flash_v100_paged_prefill(
                query, key, value, kv_cache, attn_metadata, output
            )

        if not _warned_decode_fallback:
            logger.info(
                "FLASH_ATTN_V100 decode path: using SM70/Triton kernels."
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

    def _flash_v100_paged_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path using FA2 paged kernel with block-table KV cache lookup.

        Handles both pure prefill (query_len == seq_len) and chunked/prefix
        prefill (query_len < seq_len) via the prefix_kv_lens parameter.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        num_heads_q = query.shape[1]
        num_heads_kv = key.shape[1]
        head_dim = query.shape[2]

        # Extract K/V from paged KV cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
        key_cache, value_cache = kv_cache.unbind(1)

        # Only copy if non-contiguous (vLLM cache is usually already contiguous)
        k_cache = key_cache if key_cache.is_contiguous() else key_cache.contiguous()
        v_cache = value_cache if value_cache.is_contiguous() else value_cache.contiguous()

        # Get metadata
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table

        # Prefix lengths — needed for chunked prefill (query starts partway through KV)
        # For pure prefill, all zeros. For chunked, uses prefix_kv_lens from metadata.
        if attn_metadata.prefix_kv_lens is not None:
            prefix_kv_lens = attn_metadata.prefix_kv_lens
        else:
            # Compute prefix KV lengths for chunked prefill:
            # prefix = seq_len - query_len
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            prefix_kv_lens = seq_lens - query_lens
            prefix_kv_lens = torch.clamp(prefix_kv_lens, min=0)

        block_size = k_cache.shape[1]
        softmax_scale = self.scale

        # Call paged kernel with native GQA support.
        # K/V cache keep their original num_kv_heads; kernel computes kv_head_id internally.
        _unused_out, softmax_lse = self.flash_attn_paged(
            query,
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            query_start_loc,
            prefix_kv_lens,
            out=out_view,
            block_size=block_size,
            softmax_scale=softmax_scale,
            causal=True,
            num_kv_heads=num_heads_kv,
        )

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
        # DISABLED: HND (head-major) layout caused 20% decode slowdown and
        # serious decay with prompt length. Triton decode loads tokens within
        # a block contiguously; HND scatters them by num_kv_heads. The paged
        # prefill kernel also had to copy non-contiguous KV cache every call.
        # Reverting to default block-major layout until a custom SM70 decode
        # kernel is actually deployed and proven faster.
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Flash Attention V100 requires head_dim % 8 == 0.
        return [64, 80, 96, 112, 128, 256]
