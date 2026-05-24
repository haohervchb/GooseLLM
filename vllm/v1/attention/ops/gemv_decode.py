# SPDX-License-Identifier: Apache-2.0
"""CUDA GEMV decode kernel — 4-warp, segment-split for SM occupancy."""

from __future__ import annotations

from pathlib import Path
import torch
from vllm.logger import init_logger

_ROOT = Path(__file__).resolve().parents[4]
_SO_PATH = _ROOT / "build_gemv" / "libgemv_decode.so"

logger = init_logger(__name__)
_loaded = False

# Segment buffers — allocated once per (heads, dim) config, reused across layers
_seg_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

_NUM_SEGMENTS = 128  # fills 80 V100 SMs


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    if not _SO_PATH.exists():
        raise RuntimeError(f"GEMV decode kernel not found at {_SO_PATH}")
    torch.ops.load_library(str(_SO_PATH))
    _loaded = True
    logger.info("GEMV decode CUDA kernel loaded")


def _get_seg_buffers(
    num_seqs: int, num_heads: int, dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (num_seqs, num_heads, dim)
    if key not in _seg_buffers:
        n = num_seqs * num_heads * _NUM_SEGMENTS
        sm = torch.empty(n, dtype=torch.float32, device=device)
        ss = torch.empty(n, dtype=torch.float32, device=device)
        so = torch.empty(n, dim, dtype=torch.float16, device=device)
        _seg_buffers[key] = (sm, ss, so)
        logger.info(
            "GEMV segment buffers: %d seqs × %d heads × %d segs × %d dim = %.1f MB",
            num_seqs, num_heads, _NUM_SEGMENTS, dim,
            (sm.numel() * 4 + ss.numel() * 4 + so.numel() * 2) / 1e6,
        )
    return _seg_buffers[key]


def gemv_paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_pages: int,
) -> None:
    _ensure_loaded()
    sm, ss, so = _get_seg_buffers(query.shape[0], query.shape[1], query.shape[2], output.device)
    torch.ops.gemv_decode_ops.segment_gemv_decode(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, num_pages, _NUM_SEGMENTS, sm, ss, so,
    )
