# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

MAX_SMEM = 86000  # V100 shared memory limit


def smem_kkt(BT, BK):
    """Shared memory for KKT: 2xK_tile + beta + g."""
    return (BT * BK * 2 + BT * BK * 2 + BT * 2 + BT * 2)


def smem_chko(BT, BK, BV):
    """Shared memory for chunk_fwd_o: Q+K+h+V + A16_sh + g."""
    return (
        BT * BK * 2
        + BT * BK * 2
        + BK * BV * 2
        + BT * BV * 2
        + BT * BT * 2
        + BT * 2
    )
