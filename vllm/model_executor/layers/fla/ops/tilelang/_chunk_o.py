# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tilelang chunk_fwd_o — falls back to Triton transparently.

Note: The Triton chunk_fwd_o has a known incompatibility with Triton 3.5.1
(tl.exp does not accept fp16). This is a pre-existing issue in the FLA code
unrelated to Tilelang. When FLA_USE_TILELANG=1, this fallback is used.
"""

import os as _os
import torch


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    old = _os.environ.pop("FLA_USE_TILELANG", None)
    try:
        from ..chunk_o import chunk_fwd_o as _tri
    finally:
        if old is not None:
            _os.environ["FLA_USE_TILELANG"] = old
    return _tri(
        q=q, k=k, v=v, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size,
    )
