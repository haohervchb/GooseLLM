#!/usr/bin/env python3
"""Validate GEMV CUDA decode kernel against manual PyTorch reference."""

import math
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent.parent

_ext = load(
    name='vllm_gemv_decode_ext',
    sources=[
        str(ROOT / 'csrc/attention/gemv_decode_bindings.cpp'),
        str(ROOT / 'csrc/attention/gemv_decode.cu'),
    ],
    extra_include_paths=[str(ROOT / 'csrc')],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_70,code=sm_70'],
)


def make_inputs(batch, num_heads, num_kv_heads, dim, seq_lens, block_size=16):
    max_blocks = max((sl + block_size - 1) // block_size for sl in seq_lens) + 2
    total_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 4
    num_pages = max(total_blocks, max_blocks * batch)

    device = "cuda"
    q = torch.randn(batch, num_heads, dim, device=device, dtype=torch.float16)
    kc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)
    vc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)

    block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
    offset = 0
    for i, sl in enumerate(seq_lens):
        n_blocks = (sl + block_size - 1) // block_size
        block_table[i, :n_blocks] = torch.arange(
            offset, offset + n_blocks, dtype=torch.int32, device=device
        )
        offset += n_blocks

    sl_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return q, kc, vc, block_table, sl_tensor, num_pages


def run_gemv(q, kc, vc, block_table, seq_lens, num_pages):
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    scale = dim ** -0.5
    out = torch.empty(batch, heads, dim, dtype=torch.float16, device=q.device)
    _ext.gemv_paged_decode_attention(out, q, kc, vc, heads_kv, scale, block_table, seq_lens, block_size, num_pages)
    return out


def run_manual(q, kc, vc, block_table, seq_lens):
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    scale = dim ** -0.5
    out = torch.zeros(batch, heads, dim, dtype=torch.float32, device="cpu")
    for bz in range(batch):
        sl = int(seq_lens[bz].item())
        if sl == 0: continue
        for bx in range(heads):
            kvh = bx // (heads // heads_kv)
            Kf = torch.zeros(sl, dim, dtype=torch.float32)
            Vf = torch.zeros(sl, dim, dtype=torch.float32)
            for t in range(sl):
                pi, ti = t // block_size, t % block_size
                ph = int(block_table[bz, pi].item())
                Kf[t] = kc[ph, ti, kvh].cpu().float()
                Vf[t] = vc[ph, ti, kvh].cpu().float()
            qt = q[bz, bx].cpu().float()
            acc = torch.zeros(dim, dtype=torch.float32)
            m = float("-inf"); l = 0.0
            for k in range(0, sl, 16):
                end = min(k+16, sl)
                Kt, Vt = Kf[k:end], Vf[k:end]
                scores = qt @ Kt.T
                for j in range(len(scores)): scores[j] *= scale
                m_cur = float(scores.max().item())
                om, nm = m, max(m, m_cur)
                sf = math.exp(om - nm) if om != float("-inf") else 0.0
                acc *= sf
                ts = 0.0
                probs = torch.empty(len(scores), dtype=torch.float32)
                for j in range(len(scores)):
                    p = math.exp(float(scores[j].item()) - nm)
                    probs[j] = p; ts += p
                l = l * sf + ts; m = nm
                acc += probs @ Vt
            if l > 0: out[bz, bx] = acc / l
    return out.to(dtype=torch.float16, device=q.device)


def validate(name, batch, heads, kv_heads, dim, seq_lens):
    print(f"  {name} ({batch}x{heads}/{kv_heads}x{dim} seq={seq_lens})")
    q, kc, vc, bt, sl, np = make_inputs(batch, heads, kv_heads, dim, seq_lens)
    o_cuda = run_gemv(q, kc, vc, bt, sl, np)
    o_ref = run_manual(q, kc, vc, bt, sl)
    d = (o_cuda.float() - o_ref.float()).abs()
    ok = d.max().item() < 0.05 and not torch.isnan(o_cuda).any()
    print(f"    max={d.max().item():.6f} mean={d.mean().item():.6f} → {'PASS' if ok else 'FAIL'}")
    return ok


results = []
results.append(validate("[1] Partial tile", 2, 4, 1, 256, [20, 17]))
results.append(validate("[2] Single token", 1, 4, 1, 256, [1]))
results.append(validate("[3] Exact tile", 2, 4, 1, 256, [16, 16]))
results.append(validate("[4] Empty KV", 1, 4, 1, 256, [0]))
results.append(validate("[5] Long seq", 1, 4, 1, 256, [512]))
results.append(validate("[6] GQA=4", 2, 8, 2, 256, [32, 15]))
print(f"\n{sum(results)}/{len(results)} passed" + (" - ALL PASSED" if all(results) else ""))
