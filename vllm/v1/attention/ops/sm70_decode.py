# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock

from vllm.logger import init_logger
from vllm.platforms import current_platform

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops


_ROOT_DIR = Path(__file__).resolve().parents[4]
_EXT_NAME = "vllm_sm70_decode_ext_v3"
_BUILD_DIR = _ROOT_DIR / ".cache" / _EXT_NAME
_LOCK_PATH = _BUILD_DIR / ".build.lock"

logger = init_logger(__name__)
_logged_first_decode_call = False


def _has_builtin_op() -> bool:
    return current_platform.is_cuda_alike() and hasattr(
        ops, "sm70_paged_decode_attention"
    )


@lru_cache(maxsize=1)
def _load_standalone_ext():
    from torch.utils.cpp_extension import load

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    verbose = os.getenv("VLLM_SM70_DECODE_VERBOSE", "0") == "1"

    sources = [
        str(_ROOT_DIR / "csrc/attention/turbomind_sm70_decode_bindings.cpp"),
        str(_ROOT_DIR / "csrc/attention/turbomind_sm70_decode.cu"),
    ]

    extra_include_paths = [
        str(_ROOT_DIR),
        str(_ROOT_DIR / "csrc"),
    ]

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
        except Exception as exc:  # pragma: no cover - startup-only failure path
            raise RuntimeError(
                "Failed to build/load standalone SM70 decode extension. "
                "Set VLLM_SM70_DECODE_VERBOSE=1 for compile logs."
            ) from exc


def ensure_sm70_paged_decode_available() -> None:
    if _has_builtin_op():
        return
    _load_standalone_ext()


def get_sm70_decode_impl_label() -> str:
    return "builtin" if _has_builtin_op() else "standalone"


def sm70_paged_decode_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
) -> None:
    global _logged_first_decode_call
    if _has_builtin_op():
        impl = ops.sm70_paged_decode_attention
    else:
        impl = _load_standalone_ext().sm70_paged_decode_attention

    if not _logged_first_decode_call:
        logger.info_once(
            "Invoking SM70 standalone decode kernel: query=%s key_cache=%s "
            "value_cache=%s block_size=%d max_seq_len=%d",
            tuple(query.shape),
            tuple(key_cache.shape),
            tuple(value_cache.shape),
            block_size,
            max_seq_len,
        )
        _logged_first_decode_call = True

    impl(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
    )
