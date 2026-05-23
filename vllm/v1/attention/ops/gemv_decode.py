# SPDX-License-Identifier: Apache-2.0
"""CUDA GEMV decode kernel — 32-thread warp-shuffle, zero syncthreads."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from vllm.logger import init_logger

_ROOT = Path(__file__).resolve().parents[4]
_SO_PATH = _ROOT / "build_gemv" / "libgemv_decode.so"

logger = init_logger(__name__)

_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    if not _SO_PATH.exists():
        raise RuntimeError(
            f"GEMV decode kernel not found at {_SO_PATH}. "
            f"Build it with: cd build_gemv && bash build.sh"
        )
    torch.ops.load_library(str(_SO_PATH))
    _loaded = True
    logger.info("GEMV decode CUDA kernel loaded from %s", _SO_PATH)


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
    torch.ops.gemv_decode_ops.gemv_decode(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, num_pages,
    )
