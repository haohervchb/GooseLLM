#!/usr/bin/env python3
"""Validate GEMV CUDA decode kernel against manual PyTorch reference.

No vLLM server, no model — pure kernel math validation.
"""

import math
import torch
import sys
from pathlib import Path

# Ensure vLLM is importable
_goose = Path(__file__).resolve().parents[2]
if str(_goose) not in sys.path:
    sys.path.insert(0, str(_goose))


def _load_gemv_kernel():
    from vllm.v1.attention.ops.gemv_decode import (
        ensure_gemv_paged_decode_available,
        gemv_paged_decode_attention,
    )
    ensure_gemv_paged_decode_available()
    return gemv_paged_decode_attention


def make_inputs(batch, num_heads, num_kv_heads, dim, seq_lens, block_size=16):
    max_blocks = max((sl + block_size - 1) // block_size for sl in seq_lens) + 2
    total_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 4
    num_pages = max(total_blocks, max_blocks * batch)

    device = "cuda"
    q = torch.randn(batch, num_heads, dim, device=device, dtype=torch.float16)
    kc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)
    vc = torch.randn(num_pages, block_size, num_kv_heads, dim, device=device, dtype=torch.float16)

    block_table = torch.full(
        (batch, max_blocks), -1, dtype=torch.int32, device=device
    )
    offset = 0
    for i, sl in enumerate(seq_lens):
        n_blocks = (sl + block_size - 1) // block_size
        block_table[i, :n_blocks] = torch.arange(
            offset, offset + n_blocks, dtype=torch.int32, device=device
        )
        offset += n_blocks

    sl_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return q, kc, vc, block_table, sl_tensor, num_pages


def run_gemv_cuda(q, kc, vc, block_table, seq_lens, num_pages):
    kernel = _load_gemv_kernel()
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    scale = dim ** -0.5

    out = torch.empty(batch, heads, dim, dtype=torch.float16, device=q.device)
    kernel(
        out, q, kc, vc,
        heads_kv, scale,
        block_table, seq_lens,
        block_size, num_pages,
    )
    return out


def run_manual_reference(q, kc, vc, block_table, seq_lens):
    batch, heads, dim = q.shape
    heads_kv = kc.shape[2]
    block_size = kc.shape[1]
    scale = dim ** -0.5

    out = torch.zeros(batch, heads, dim, dtype=torch.float32, device="cpu")

    for bz in range(batch):
        sl = int(seq_lens[bz].item())
        if sl == 0:
            continue
        for bx in range(heads):
            kvh = bx // (heads // heads_kv)
            K_full = torch.zeros(sl, dim, dtype=torch.float32)
            V_full = torch.zeros(sl, dim, dtype=torch.float32)
            for t in range(sl):
                page_idx = t // block_size
                token_in_page = t % block_size
                ph = int(block_table[bz, page_idx].item())
                K_full[t] = kc[ph, token_in_page, kvh].cpu().float()
                V_full[t] = vc[ph, token_in_page, kvh].cpu().float()

            q_token = q[bz, bx].cpu().float()
            acc = torch.zeros(dim, dtype=torch.float32)
            m = float("-inf")
            l = 0.0

            for k in range(0, sl, 16):
                end = min(k + 16, sl)
                K_tile = K_full[k:end]
                V_tile = V_full[k:end]

                scores = q_token @ K_tile.T
                for j in range(len(scores)):
                    scores[j] *= scale

                m_cur = float(scores.max().item())
                old_m = m
                new_m = max(old_m, m_cur)
                sf = math.exp(old_m - new_m) if old_m != float("-inf") else 0.0

                acc *= sf
                tile_sum = 0.0
                probs = torch.empty(len(scores), dtype=torch.float32)
                for j in range(len(scores)):
                    p = math.exp(float(scores[j].item()) - new_m)
                    probs[j] = p
                    tile_sum += p
                l = l * sf + tile_sum
                m = new_m

                acc += probs @ V_tile

            if l > 0:
                out[bz, bx] = acc / l

    return out.to(dtype=torch.float16, device=q.device)


def validate_case(name, batch, heads, kv_heads, dim, seq_lens):
    print(f"  {name} (batch={batch}, heads={heads}/{kv_heads}, dim={dim}, "
          f"seq_lens={seq_lens})")
    q, kc, vc, bt, sl, npages = make_inputs(batch, heads, kv_heads, dim, seq_lens)

    output_cuda = run_gemv_cuda(q, kc, vc, bt, sl, npages)
    output_ref = run_manual_reference(q, kc, vc, bt, sl)

    diff = (output_cuda.float() - output_ref.float()).abs()
    cuda_nan = torch.isnan(output_cuda).any().item()
    ref_nan = torch.isnan(output_ref).any().item()

    passed = diff.max().item() < 0.05 and not cuda_nan and not ref_nan
    status = "PASS" if passed else "FAIL"
    print(f"    max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f} "
          f"cuda_nan={cuda_nan} ref_nan={ref_nan} → {status}")
    return passed


def main():
    print("=" * 60)
    print("GEMV CUDA Decode Kernel Validation (vs manual reference)")
    print("=" * 60)

    results = []
    results.append(validate_case("[1] Partial last tile", 2, 4, 1, 256, [20, 17]))
    results.append(validate_case("[2] Single token", 1, 4, 1, 256, [1]))
    results.append(validate_case("[3] Exact tile", 2, 4, 1, 256, [16, 16]))
    results.append(validate_case("[4] Empty KV", 1, 4, 1, 256, [0]))
    results.append(validate_case("[5] Long sequence", 1, 4, 1, 256, [512]))
    results.append(validate_case("[6] GQA ratio=4", 2, 8, 2, 256, [32, 15]))

    n_pass = sum(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{len(results)} passed")
    if n_pass == len(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    return n_pass == len(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
