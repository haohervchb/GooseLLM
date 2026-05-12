# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tilelang-accelerated chunk_scaled_dot_kkt_fwd for V100/SM70.

Supports both fixed-length and variable-length (cu_seqlens) inputs.
"""

import math
import torch
import triton

import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy

from vllm.logger import init_logger
logger = init_logger(__name__)
_logged_once = False


def _kkt_factory(B, S, H, Hg, Kdim, BT):
    """Fixed-length factory. BK/threads/num_stages are LOCAL, not params."""
    BK = 16
    threads = 128
    num_stages = 0
    NT = S // BT

    @T.prim_func
    def main(
        k: T.Tensor([B, S, Hg, Kdim], T.float16),
        beta: T.Tensor([B, S, H], T.float16),
        g: T.Tensor([B, S, H], T.float16),
        A: T.Tensor([B, S, H, BT], T.float32),
    ):
        with T.Kernel(NT, H, B, threads=threads) as (bx, by, bz):
            sh = T.alloc_shared([BT, BK], T.float16)
            sc = T.alloc_shared([BT, BK], T.float16)
            bs = T.alloc_shared([BT], T.float16)
            gs = T.alloc_shared([BT], T.float16)
            ac = T.alloc_fragment([BT, BT], T.float32)
            T.fill(ac, 0)

            cs = bz * S + bx * BT
            for i in T.Parallel(BT):
                bs[i] = T.cast(beta[bz, cs + i, by], T.float16)
                gs[i] = T.cast(g[bz, cs + i, by], T.float16)

            hk = by // (H // Hg) if Hg > 0 else 0

            for ik in T.Pipelined(T.ceildiv(Kdim, BK)):
                ko = ik * BK
                for i, j in T.Parallel(BT, BK):
                    kv = k[bz, cs + i, hk, ko + j]
                    sh[i, j] = kv * bs[i]
                    sc[i, j] = kv
                T.gemm(sh, sc, ac, transpose_B=True, policy=GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BT):
                v = ac[i, j] * T.exp(gs[i] - gs[j])
                A[bz, cs + i, by, j] = T.cast(v, T.float32) if j < i else T.cast(0, T.float32)
    return main


def _kkt_factory_varlen(H, Hg, Kdim, BT):
    """Variable-length factory. BK/threads/num_stages are LOCAL, not params.
    k has shape (1, T_total, Hg, Kdim) where T_total varies.
    """
    BK = 16
    threads = 128
    num_stages = 0
    TT = T.dynamic("TT")
    NC = T.dynamic("NC")
    NT = T.dynamic("NT")

    @T.prim_func
    def main(
        k: T.Tensor([1, TT, Hg, Kdim], T.float16),
        beta: T.Tensor([1, TT, H], T.float16),
        g: T.Tensor([1, TT, H], T.float16),
        cu_seqlens: T.Tensor([NC], T.int32),
        chunk_indices: T.Tensor([NT, 2], T.int32),
        A: T.Tensor([1, TT, H, BT], T.float32),
    ):
        with T.Kernel(NT, H, 1, threads=threads) as (bx, by, bz):
            sh = T.alloc_shared([BT, BK], T.float16)
            sc = T.alloc_shared([BT, BK], T.float16)
            bs = T.alloc_shared([BT], T.float16)
            gs = T.alloc_shared([BT], T.float16)
            ac = T.alloc_fragment([BT, BT], T.float32)
            T.fill(ac, 0)

            # Load chunk_idx, seq_idx from chunk_indices
            i_n = chunk_indices[bx, 0]
            i_t = chunk_indices[bx, 1]
            # Compute bos/eos from cu_seqlens
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            cs = bos + i_t * BT
            T_tokens = eos - bos

            hk = by // (H // Hg) if Hg > 0 else 0

            for i in T.Parallel(BT):
                valid = cs + i < eos
                bs[i] = T.if_then_else(valid, T.cast(beta[0, cs + i, by], T.float16), T.cast(0, T.float16))
                gs[i] = T.if_then_else(valid, T.cast(g[0, cs + i, by], T.float16), T.cast(-100, T.float16))

            for ik in T.Pipelined(T.ceildiv(Kdim, BK)):
                ko = ik * BK
                for i, j in T.Parallel(BT, BK):
                    kv = T.if_then_else(cs + i < eos, k[0, cs + i, hk, ko + j], T.cast(0, T.float16))
                    sh[i, j] = kv * bs[i]
                    sc[i, j] = kv
                T.gemm(sh, sc, ac, transpose_B=True, policy=GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BT, BT):
                v = ac[i, j] * T.exp(gs[i] - gs[j])
                if j < i and cs + i < eos:
                    A[0, cs + i, by, j] = T.cast(v, T.float32)
                elif cs + i < eos:
                    A[0, cs + i, by, j] = T.cast(0, T.float32)
    return main


# Module-level jit wrappers
_JIT_FIXED = tilelang.jit(out_idx=[3])(_kkt_factory)
_JIT_VARLEN = tilelang.jit(out_idx=[5])(_kkt_factory_varlen)

_KERNEL_CACHE_FIXED = {}
_KERNEL_CACHE_VARLEN = {}


def _get_kkt_fixed(B, T, H, Hg, K, BT):
    key = (B, T, H, Hg, K, BT)
    if key not in _KERNEL_CACHE_FIXED:
        _KERNEL_CACHE_FIXED[key] = _JIT_FIXED(B, T, H, Hg, K, BT)
    return _KERNEL_CACHE_FIXED[key]


def _get_kkt_varlen(H, Hg, K, BT):
    key = (H, Hg, K, BT)
    if key not in _KERNEL_CACHE_VARLEN:
        _KERNEL_CACHE_VARLEN[key] = _JIT_VARLEN(H, Hg, K, BT)
    return _KERNEL_CACHE_VARLEN[key]


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    global _logged_once
    if not _logged_once:
        logger.info("Using Tilelang-accelerated chunk_scaled_dot_kkt_fwd (FLA_USE_TILELANG=1)")
        _logged_once = True

    B, TT, Hg, Kd = k.shape
    H = beta.shape[-1]
    BT = chunk_size

    # Cast inputs to fp16 for the Tilelang kernel (production may pass fp32)
    k = k.to(torch.float16)
    if beta is not None:
        beta = beta.to(torch.float16)
    if g is not None:
        g = g.to(torch.float16)

    if cu_seqlens is not None:
        from ..index import prepare_chunk_indices
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        fn = _get_kkt_varlen(H, Hg, Kd, BT)
        A_out = fn(k, beta, g, cu_seqlens, chunk_indices)
        A = torch.empty(B, TT, H, BT, device=k.device, dtype=output_dtype)
        A.copy_(A_out)
        return A

    assert TT % BT == 0, f"T={TT} must be divisible by BT={BT} for Tilelang KKT"
    A = torch.empty(B, TT, H, BT, device=k.device, dtype=output_dtype)
    fn = _get_kkt_fixed(B, TT, H, Hg, Kd, BT)
    A_out = fn(k, beta, g)
    A.copy_(A_out)
    return A
