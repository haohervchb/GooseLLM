# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import warnings

warnings.filterwarnings("ignore", message="Field.*duplicates an ancestor field")
warnings.filterwarnings("ignore", message=".*GemmSPWarpPolicy.*")

_USE_TILELANG = os.environ.get("FLA_USE_TILELANG", "0") == "1"
_chunk_scaled_dot_kkt_fwd = None
_chunk_fwd_o = None


def _ensure_imported():
    global _chunk_scaled_dot_kkt_fwd, _chunk_fwd_o
    if _chunk_scaled_dot_kkt_fwd is not None:
        return True
    if not _USE_TILELANG:
        return False
    try:
        from ._kkt import chunk_scaled_dot_kkt_fwd as _f1
        # chunk_fwd_o always falls back to Triton (Tilelang 0.1.9 GEMM shape issue)
        from ._chunk_o import chunk_fwd_o as _f2
        _chunk_scaled_dot_kkt_fwd = _f1
        _chunk_fwd_o = _f2
        return True
    except Exception:
        return False


def chunk_scaled_dot_kkt_fwd(*args, **kwargs):
    if _ensure_imported():
        return _chunk_scaled_dot_kkt_fwd(*args, **kwargs)
    raise ImportError(
        "Tilelang kernel not available. "
        "Set FLA_USE_TILELANG=1 and ensure tilelang is installed."
    )


def chunk_fwd_o(*args, **kwargs):
    if _ensure_imported():
        return _chunk_fwd_o(*args, **kwargs)
    raise ImportError(
        "Tilelang kernel not available. "
        "Set FLA_USE_TILELANG=1 and ensure tilelang is installed."
    )
