# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ._utils import smem_kkt, MAX_SMEM


def kkt_configs(*args, **kwargs):
    BT = args[0] if args else kwargs.get("BT", 64)
    K = args[1] if len(args) > 1 else kwargs.get("K", 128)
    configs = []
    BK_vals = [16, 32, 64, 128]
    for BK in BK_vals:
        if BK > K:
            continue
        if smem_kkt(BT, BK) > MAX_SMEM:
            continue
        for threads in [64, 128, 256]:
            if threads < BT:
                continue
            for num_stages in [0, 1]:
                configs.append(dict(BK=BK, threads=threads, num_stages=num_stages))
    if not configs:
        configs.append(dict(BK=min(16, K), threads=128, num_stages=0))
    return configs
