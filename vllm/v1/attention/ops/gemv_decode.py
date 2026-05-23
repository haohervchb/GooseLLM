# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock

from vllm.logger import init_logger

_ROOT_DIR = Path(__file__).resolve().parents[4]
_EXT_NAME = "vllm_gemv_decode_ext"
_BUILD_DIR = _ROOT_DIR / ".cache" / _EXT_NAME
_LOCK_PATH = _BUILD_DIR / ".build.lock"

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _load_standalone_ext():
    from torch.utils.cpp_extension import load

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    verbose = os.getenv("VLLM_GEMV_DECODE_VERBOSE", "0") == "1"

    sources = [
        str(_ROOT_DIR / "csrc/attention/gemv_decode_bindings.cpp"),
        str(_ROOT_DIR / "csrc/attention/gemv_decode.cu"),
    ]

    extra_include_paths = [str(_ROOT_DIR / "csrc")]

    extra_cflags = ["-O3"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_70,code=sm_70",
    ]

    with FileLock(str(_LOCK_PATH)):
        try:
            return load(
                name=_EXT_NAME,
                sources=sources,
                extra_include_paths=extra_include_paths,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                build_directory=str(_BUILD_DIR),
                verbose=verbose,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build/load GEMV decode extension. "
                "Set VLLM_GEMV_DECODE_VERBOSE=1 for compile logs."
            ) from exc


class GemvDecodeOp(torch.autograd.Function):
    """Autograd wrapper — makes pybind11 kernel graph-capturable."""

    @staticmethod
    def forward(ctx, output, query, key_cache, value_cache,
                num_kv_heads_t, scale_t, block_tables, seq_lens,
                block_size_t, num_pages_t):
        ext = _load_standalone_ext()
        ext.gemv_paged_decode_attention(
            output, query, key_cache, value_cache,
            int(num_kv_heads_t.item()), float(scale_t.item()),
            block_tables, seq_lens,
            int(block_size_t.item()), int(num_pages_t.item()),
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (None,) * 10


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
    GemvDecodeOp.apply(
        output, query, key_cache, value_cache,
        torch.tensor(num_kv_heads, device=output.device),
        torch.tensor(scale, device=output.device),
        block_tables, seq_lens,
        torch.tensor(block_size, device=output.device),
        torch.tensor(num_pages, device=output.device),
    )
