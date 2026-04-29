#!/usr/bin/env python3
"""
Comprehensive test suite for FlashAttention-2 Paged Prefill kernel (SM70/V100).

Tests correctness, NaN/inf safety, GQA, chunked prefill, multi-sequence batching,
block-table layouts, and the exact vLLM integration pipeline (write-then-read).

Usage:
    python test_paged_prefill.py [--quick] [--verbose]
"""

import gc
import math
import sys
import time
import argparse
from collections import defaultdict

import torch

try:
    import flash_attn_v100_cuda
except ImportError:
    print("flash_attn_v100_cuda not found.")
    print("Run: cd csrc/flash_attention_v100 && python setup.py build_ext --inplace")
    sys.exit(1)

# ============================================================================
# REFERENCE IMPLEMENTATION
# ============================================================================

def paged_prefill_reference(q, k_cache, v_cache, block_table, seq_lens,
                            query_start_loc, prefix_kv_lens, block_size,
                            softmax_scale, causal, num_kv_heads):
    """
    Ground truth: reconstruct logical K/V from paged cache, run fp32 attention.
    Handles GQA by expanding K/V to match Q heads.

    Returns: torch.Tensor [num_tokens, num_heads, D] in q.dtype
    """
    num_tokens, num_heads, D = q.shape
    # Support both tensor and list inputs
    if isinstance(seq_lens, torch.Tensor):
        num_seqs = seq_lens.shape[0]
        seq_lens_list = [int(seq_lens[i].item()) for i in range(num_seqs)]
        prefix_kv_lens_list = [int(prefix_kv_lens[i].item()) for i in range(num_seqs)]
        qsl_list = [int(query_start_loc[i].item()) for i in range(query_start_loc.shape[0])]
    else:
        num_seqs = len(seq_lens)
        seq_lens_list = [int(s) for s in seq_lens]
        prefix_kv_lens_list = [int(p) for p in prefix_kv_lens]
        qsl_list = [int(s) for s in query_start_loc]
    group_size = num_heads // num_kv_heads
    device = q.device
    out = torch.zeros(num_tokens, num_heads, D, dtype=torch.float32, device=device)

    for seq in range(num_seqs):
        sl = seq_lens_list[seq]
        prefix = prefix_kv_lens_list[seq]
        q_start = qsl_list[seq]
        q_end = qsl_list[seq + 1]
        q_len = q_end - q_start
        if q_len <= 0 or sl <= 0:
            continue

        # Reconstruct logical K/V from paged cache
        k_seq = torch.zeros(sl, num_kv_heads, D, dtype=torch.float32, device=device)
        v_seq = torch.zeros(sl, num_kv_heads, D, dtype=torch.float32, device=device)
        for pos in range(sl):
            blk = pos // block_size
            off = pos % block_size
            phys = int(block_table[seq, blk].item())
            if phys >= 0:
                k_seq[pos] = k_cache[phys, off].float()
                v_seq[pos] = v_cache[phys, off].float()

        q_seq = q[q_start:q_end].float()  # [q_len, num_heads, D]

        # Attention per head with GQA
        for h in range(num_heads):
            kv_h = h // group_size
            scores = q_seq[:, h, :] @ k_seq[:, kv_h, :].T  # [q_len, sl]
            scores *= softmax_scale

            if causal:
                for i in range(q_len):
                    scores[i, prefix + i + 1:] = float('-inf')

            scores_max = scores.max(dim=-1, keepdim=True).values
            scores = scores - scores_max
            exp_scores = torch.exp(torch.clamp(scores, min=-80.0, max=80.0))
            probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

            out[q_start:q_end, h, :] += probs @ v_seq[:, kv_h, :]

    return out.to(dtype=q.dtype)


def dense_fa2_reference(q, k, v, softmax_scale, causal):
    """
    Run dense FA2 kernel on logically contiguous Q, K, V.
    Returns: torch.Tensor in q.dtype
    """
    # convert to [B, H, M, D] layout for the kernel
    q_fa2 = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, M, D]
    k_fa2 = k.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, N, D]
    v_fa2 = v.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, N, D]

    out = flash_attn_v100_cuda.fwd(
        q_fa2, k_fa2, v_fa2,
        None, None, 0.0, softmax_scale, causal, -1, -1, 0.0, False, None
    )[0]
    # Convert back: [1, H, M, D] -> [M, H, D]
    return out.permute(0, 2, 1, 3).squeeze(0).contiguous()


# ============================================================================
# HELPERS
# ============================================================================

def make_block_table(num_seqs, seq_lens, block_size, start_block=0):
    """Create block_table [num_seqs, max_blocks] filled with physical block IDs."""
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device='cuda')
    phys = start_block
    for i, sl in enumerate(seq_lens):
        nb = (sl + block_size - 1) // block_size
        for b in range(nb):
            block_table[i, b] = phys + b
        phys += nb
    return block_table, phys


def make_custom_block_table(num_seqs, seq_lens, block_size, assignments):
    """
    Create block_table with custom physical block assignments.
    assignments: list of lists, e.g. [[3,7,1], [5,2]] for 2 seqs.
    """
    max_blocks = max(len(a) for a in assignments)
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device='cuda')
    for i, blocks in enumerate(assignments):
        for b, phys in enumerate(blocks):
            block_table[i, b] = phys
    return block_table


def populate_kv_cache_from_dense(k_cache, v_cache, block_table, seq_lens,
                                  k_dense, v_dense, block_size):
    """Populate paged caches from dense sequential K/V tensors."""
    offset = 0
    for seq in range(len(seq_lens)):
        sl = seq_lens[seq]
        for pos in range(sl):
            blk = pos // block_size
            off = pos % block_size
            phys = int(block_table[seq, blk].item())
            if phys >= 0:
                k_cache[phys, off] = k_dense[offset + pos]
                v_cache[phys, off] = v_dense[offset + pos]
        offset += sl


def make_test_inputs(num_seqs, query_lens, seq_lens, num_heads, num_kv_heads,
                     head_dim, block_size, block_start=0, seed=None):
    """
    Create the full set of inputs for paged_fwd.
    Returns dict with all tensors + metadata.
    """
    if seed is not None:
        torch.manual_seed(seed)

    total_query = sum(query_lens)
    total_seq = sum(seq_lens)
    device = 'cuda'

    # Create dense data first
    q = torch.randn(total_query, num_heads, head_dim, dtype=torch.float16, device=device)
    k_dense = torch.randn(total_seq, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v_dense = torch.randn(total_seq, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    # Block table
    block_table, num_blocks = make_block_table(num_seqs, seq_lens, block_size, block_start)

    # Paged caches
    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float16, device=device)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float16, device=device)
    populate_kv_cache_from_dense(k_cache, v_cache, block_table, seq_lens,
                                  k_dense, v_dense, block_size)

    # Metadata tensors
    query_start_loc = torch.tensor(
        [0] + list(torch.tensor(query_lens, dtype=torch.int32).cumsum(0).tolist()),
        dtype=torch.int32, device=device
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    prefix_kv_lens_list = [sl - ql for sl, ql in zip(seq_lens, query_lens)]
    prefix_kv_lens_t = torch.tensor(prefix_kv_lens_list, dtype=torch.int32, device=device)

    return {
        'q': q,
        'k_cache': k_cache,
        'v_cache': v_cache,
        'k_dense': k_dense,
        'v_dense': v_dense,
        'block_table': block_table,
        'seq_lens': seq_lens_t,
        'query_start_loc': query_start_loc,
        'prefix_kv_lens': prefix_kv_lens_t,
        'num_blocks': num_blocks,
        'num_seqs': num_seqs,
        'total_query': total_query,
        'total_seq': total_seq,
    }


def run_paged_fwd(inputs, softmax_scale, causal, block_size=None):
    """Run the paged prefill kernel. Returns (output, softmax_lse)."""
    bs = block_size if block_size is not None else inputs.get('block_size', 16)
    out = torch.empty_like(inputs['q'])
    result = flash_attn_v100_cuda.paged_fwd(
        inputs['q'],
        inputs['k_cache'],
        inputs['v_cache'],
        inputs['block_table'],
        inputs['seq_lens'],
        inputs['query_start_loc'],
        inputs['prefix_kv_lens'],
        out,
        inputs['k_cache'].shape[2],  # num_kv_heads
        bs,
        float(softmax_scale),
        causal
    )
    return result[0], result[1]


def check_finite(tensor, name):
    """Return (ok: bool, nan_count, inf_count)."""
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    return (n_nan == 0 and n_inf == 0), n_nan, n_inf


def compute_error_metrics(out_kernel, out_ref):
    """Compute error metrics between kernel output and reference."""
    out_k_f32 = out_kernel.float()
    out_r_f32 = out_ref.float()

    abs_diff = (out_k_f32 - out_r_f32).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    # Relative error
    denom = out_r_f32.abs() + 1e-12
    rel_err = (abs_diff / denom).max().item()

    # Cosine similarity
    flat_k = out_k_f32.reshape(-1)
    flat_r = out_r_f32.reshape(-1)
    cos_sim = (flat_k @ flat_r) / (flat_k.norm() * flat_r.norm() + 1e-12)
    cos_sim = cos_sim.item()

    return max_abs_err, mean_abs_err, rel_err, cos_sim


# ============================================================================
# TEST RUNNER
# ============================================================================

_results = []


def record(name, passed, message="", details=None):
    status = "PASS" if passed else "FAIL"
    _results.append({
        'name': name, 'passed': passed, 'message': message,
        'details': details or {}
    })
    marker = "✅" if passed else "❌"
    detail_str = f"  {message}" if message else ""
    print(f"  {marker} {name}{detail_str}")
    return passed


def print_summary():
    passed = sum(1 for r in _results if r['passed'])
    failed = sum(1 for r in _results if not r['passed'])
    total = len(_results)
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print(f"\nFAILED TESTS:")
        for r in _results:
            if not r['passed']:
                print(f"  ❌ {r['name']}: {r['message']}")
    print(f"{'='*70}")


# ============================================================================
# SELF-VALIDATION TESTS
# ============================================================================

def test_self_validation():
    print("\n--- Self-Validation ---")

    # S1: Reference GQA vs MHA-equivalent
    inp = make_test_inputs(1, [8], [8], num_heads=4, num_kv_heads=1,
                           head_dim=64, block_size=16, seed=42)
    # Create MHA-equivalent: H=4, KV_H=4, GQA reference with group_size=1
    inp_mha = make_test_inputs(1, [8], [8], num_heads=4, num_kv_heads=4,
                                head_dim=64, block_size=16, seed=42)
    # Copy same K/V data across KV heads for fairness
    for h in range(4):
        inp['k_dense'][:, 0:1, :] = inp_mha['k_dense'][:, h:h+1, :]
        inp['v_dense'][:, 0:1, :] = inp_mha['v_dense'][:, h:h+1, :]
    # Re-populate KV cache
    populate_kv_cache_from_dense(inp['k_cache'], inp['v_cache'],
                                  inp['block_table'], [8],
                                  inp['k_dense'], inp['v_dense'], 16)
    populate_kv_cache_from_dense(inp_mha['k_cache'], inp_mha['v_cache'],
                                  inp_mha['block_table'], [8],
                                  inp_mha['k_dense'], inp_mha['v_dense'], 16)

    out_gqa = paged_prefill_reference(inp['q'], inp['k_cache'], inp['v_cache'],
                                       inp['block_table'], [8],
                                       inp['query_start_loc'],
                                       inp['prefix_kv_lens'],
                                       16, 1.0 / math.sqrt(64), True, 1)
    out_mha = paged_prefill_reference(inp_mha['q'], inp_mha['k_cache'],
                                       inp_mha['v_cache'], inp_mha['block_table'],
                                       [8], inp_mha['query_start_loc'],
                                       inp_mha['prefix_kv_lens'],
                                       16, 1.0 / math.sqrt(64), True, 4)
    ok_s1a = check_finite(out_gqa, "ref_gqa")[0] and check_finite(out_mha, "ref_mha")[0]
    record("S1a: GQA reference produces finite output", ok_s1a)

    # S2: Causal correctness
    inp2 = make_test_inputs(1, [16], [16], 4, 4, 64, 16, seed=123)
    out_causal = paged_prefill_reference(inp2['q'], inp2['k_cache'], inp2['v_cache'],
                                          inp2['block_table'], [16],
                                          inp2['query_start_loc'],
                                          inp2['prefix_kv_lens'],
                                          16, 1.0 / math.sqrt(64), True, 4)
    out_noncausal = paged_prefill_reference(inp2['q'], inp2['k_cache'], inp2['v_cache'],
                                             inp2['block_table'], [16],
                                             inp2['query_start_loc'],
                                             inp2['prefix_kv_lens'],
                                             16, 1.0 / math.sqrt(64), False, 4)
    # Causal: each row i attends only to positions 0..i
    # Non-causal: all rows attend to all positions
    # Row 0: causal sees 1 token, non-causal sees N tokens → DIFFER
    # Last row (N-1): causal sees N tokens, non-causal sees N tokens → SAME
    diff = (out_causal - out_noncausal).abs()
    first_row_diff = diff[0].max().item()
    last_row_diff = diff[-1].max().item()
    ok = first_row_diff > 1e-3 and last_row_diff < 1e-5
    record("S2: Causal mask affects output correctly", ok,
           f"first_row_diff={first_row_diff:.2e}, last_row_diff={last_row_diff:.2e}")

    # S3: Deterministic reference
    inp3 = make_test_inputs(1, [32], [32], 8, 2, 64, 16, seed=99)
    out_a = paged_prefill_reference(inp3['q'], inp3['k_cache'], inp3['v_cache'],
                                     inp3['block_table'], [32], inp3['query_start_loc'],
                                     inp3['prefix_kv_lens'], 16, 1.0/math.sqrt(64), True, 2)
    out_b = paged_prefill_reference(inp3['q'], inp3['k_cache'], inp3['v_cache'],
                                     inp3['block_table'], [32], inp3['query_start_loc'],
                                     inp3['prefix_kv_lens'], 16, 1.0/math.sqrt(64), True, 2)
    max_diff = (out_a.float() - out_b.float()).abs().max().item()
    record("S3: Reference is deterministic", max_diff == 0.0,
           f"max_diff={max_diff:.2e}")


# ============================================================================
# CORRECTNESS TESTS
# ============================================================================

def run_correctness_test(name, num_seqs, query_lens, seq_lens, num_heads, num_kv_heads,
                         head_dim, block_size, causal, seed=42, tolerance=None):
    """Run a single correctness test: paged kernel vs reference."""
    inp = make_test_inputs(num_seqs, query_lens, seq_lens, num_heads, num_kv_heads,
                           head_dim, block_size, seed=seed)
    scale = 1.0 / math.sqrt(head_dim)

    # Run kernel
    out_kernel, lse = run_paged_fwd(inp, scale, causal, block_size)

    # Check finite
    ok_f, nan, inf = check_finite(out_kernel, "kernel")
    if not ok_f:
        return record(name, False, f"NaN={nan}, Inf={inf} in kernel output")

    # Run reference
    ref_kv = num_kv_heads
    out_ref = paged_prefill_reference(inp['q'], inp['k_cache'], inp['v_cache'],
                                       inp['block_table'], inp['seq_lens'],
                                       inp['query_start_loc'], inp['prefix_kv_lens'],
                                       block_size, scale, causal, ref_kv)

    ok_r, nan_r, inf_r = check_finite(out_ref, "ref")
    if not ok_r:
        return record(name, False, f"NaN={nan_r}, Inf={inf_r} in reference (BUG IN REFERENCE)")

    max_err, mean_err, rel_err, cos_sim = compute_error_metrics(out_kernel, out_ref)

    if tolerance is None:
        # Default: max error <= 2.2e-3 (fp16 ULP = 9.77e-04, allow 2-3 ULP)
        passed = max_err <= 2.2e-3 and cos_sim > 0.999
    else:
        passed = max_err <= tolerance and cos_sim > 0.999

    return record(name, passed,
                  f"max_err={max_err:.2e}, cos={cos_sim:.6f}, nan=0")


def test_correctness_minimal():
    print("\n--- Minimal Correctness ---")
    run_correctness_test("C1: 1x1 D=64 causal", 1, [1], [1], 1, 1, 64, 16, True)
    run_correctness_test("C2: 1x1 D=64 non-causal", 1, [1], [1], 1, 1, 64, 16, False)
    run_correctness_test("C3: 1x1 D=128 causal", 1, [1], [1], 1, 1, 128, 16, True)
    run_correctness_test("C4: 1x1 D=256 causal", 1, [1], [1], 1, 1, 256, 16, True)
    run_correctness_test("C5: 1x4 D=64 MHA", 1, [1], [4], 4, 4, 64, 16, True)
    run_correctness_test("C6: 1x8 D=64 GQA 4:1", 1, [1], [8], 8, 2, 64, 16, True)
    run_correctness_test("C7: 8x8 D=64 MHA", 1, [8], [8], 4, 4, 64, 16, True)
    run_correctness_test("C8: 16x32 D=64 MHA", 1, [16], [32], 4, 4, 64, 16, True)
    run_correctness_test("C9: 32x64 D=128 MHA", 1, [32], [64], 8, 8, 128, 16, True)
    run_correctness_test("C10: 16x32 D=256 MHA", 1, [16], [32], 4, 4, 256, 16, True)


def test_correctness_tile_boundaries():
    print("\n--- Tile Boundary Correctness ---")
    # D=64: BLOCK_M=64, BLOCK_N=128
    run_correctness_test("T1: D=64 64x64 causal", 1, [64], [64], 1, 1, 64, 16, True)
    run_correctness_test("T2: D=64 64x128 causal", 1, [64], [128], 1, 1, 64, 16, True)
    run_correctness_test("T3: D=64 65x65 causal", 1, [65], [65], 1, 1, 64, 16, True)
    run_correctness_test("T4: D=64 64x129 causal", 1, [64], [129], 1, 1, 64, 16, True)
    run_correctness_test("T5: D=64 128x256 causal", 1, [128], [256], 1, 1, 64, 16, True)
    # D=128: BLOCK_M=32, BLOCK_N=176
    run_correctness_test("T6: D=128 32x176 causal", 1, [32], [176], 1, 1, 128, 16, True)
    run_correctness_test("T7: D=128 33x177 causal", 1, [33], [177], 1, 1, 128, 16, True)
    # D=256: BLOCK_M=32, BLOCK_N=64
    run_correctness_test("T8: D=256 32x64 causal", 1, [32], [64], 1, 1, 256, 16, True)
    run_correctness_test("T9: D=256 33x65 causal", 1, [33], [65], 1, 1, 256, 16, True)


def test_correctness_block_table():
    print("\n--- Block Table Layout ---")
    # B1: Single aligned block
    inp = make_test_inputs(1, [16], [16], 4, 4, 64, 16, seed=1)
    scale = 1.0 / math.sqrt(64)
    out_kernel, _ = run_paged_fwd(inp, scale, True)
    out_ref = paged_prefill_reference(inp['q'], inp['k_cache'], inp['v_cache'],
                                       inp['block_table'], inp['seq_lens'],
                                       inp['query_start_loc'], inp['prefix_kv_lens'],
                                       16, scale, True, 4)
    max_err, _, _, cos = compute_error_metrics(out_kernel, out_ref)
    record("B1: Single block [0]", max_err <= 2.2e-3 and cos > 0.999,
           f"max_err={max_err:.2e}, cos={cos:.6f}")

    # B2: Non-zero block start
    inp2 = make_test_inputs(1, [16], [16], 4, 4, 64, 16, block_start=5, seed=2)
    out_kernel2, _ = run_paged_fwd(inp2, scale, True)
    out_ref2 = paged_prefill_reference(inp2['q'], inp2['k_cache'], inp2['v_cache'],
                                        inp2['block_table'], inp2['seq_lens'],
                                        inp2['query_start_loc'], inp2['prefix_kv_lens'],
                                        16, scale, True, 4)
    max_err2, _, _, cos2 = compute_error_metrics(out_kernel2, out_ref2)
    record("B2: Block start at 5", max_err2 <= 2.2e-3 and cos2 > 0.999,
           f"max_err={max_err2:.2e}, cos={cos2:.6f}")

    # B3: Non-contiguous blocks
    custom_bt = make_custom_block_table(1, [48], 16, [[3, 7, 1]])
    inp3 = make_test_inputs(1, [48], [48], 4, 4, 64, 16, seed=3)
    inp3['block_table'] = custom_bt
    # Need a bigger cache
    kc_new = torch.zeros(8, 16, 4, 64, dtype=torch.float16, device='cuda')
    vc_new = torch.zeros(8, 16, 4, 64, dtype=torch.float16, device='cuda')
    populate_kv_cache_from_dense(kc_new, vc_new, custom_bt, [48],
                                  inp3['k_dense'], inp3['v_dense'], 16)
    inp3['k_cache'] = kc_new
    inp3['v_cache'] = vc_new
    out_kernel3, _ = run_paged_fwd(inp3, scale, True)
    out_ref3 = paged_prefill_reference(inp3['q'], inp3['k_cache'], inp3['v_cache'],
                                        inp3['block_table'], inp3['seq_lens'],
                                        inp3['query_start_loc'], inp3['prefix_kv_lens'],
                                        16, scale, True, 4)
    max_err3, _, _, cos3 = compute_error_metrics(out_kernel3, out_ref3)
    record("B3: Non-contiguous blocks [3,7,1]", max_err3 <= 2.2e-3 and cos3 > 0.999,
           f"max_err={max_err3:.2e}, cos={cos3:.6f}")

    # B4: block_size=32
    inp4 = make_test_inputs(1, [32], [32], 4, 4, 64, 32, seed=4)
    inp4['block_size'] = 32
    out_kernel4, _ = run_paged_fwd(inp4, scale, True, block_size=32)
    out_ref4 = paged_prefill_reference(inp4['q'], inp4['k_cache'], inp4['v_cache'],
                                        inp4['block_table'], inp4['seq_lens'],
                                        inp4['query_start_loc'], inp4['prefix_kv_lens'],
                                        32, scale, True, 4)
    max_err4, _, _, cos4 = compute_error_metrics(out_kernel4, out_ref4)
    record("B4: block_size=32", max_err4 <= 2.2e-3 and cos4 > 0.999,
           f"max_err={max_err4:.2e}, cos={cos4:.6f}")

    # B5: Multi-seq with interleaved blocks
    custom_bt5 = make_custom_block_table(3, [16, 32, 8], 16,
                                          [[0, 3], [1, 4], [2]])
    inp5 = make_test_inputs(3, [16, 32, 8], [16, 32, 8], 4, 4, 64, 16, seed=5)
    inp5['block_table'] = custom_bt5
    kc5 = torch.zeros(5, 16, 4, 64, dtype=torch.float16, device='cuda')
    vc5 = torch.zeros(5, 16, 4, 64, dtype=torch.float16, device='cuda')
    populate_kv_cache_from_dense(kc5, vc5, custom_bt5, [16, 32, 8],
                                  inp5['k_dense'], inp5['v_dense'], 16)
    inp5['k_cache'] = kc5
    inp5['v_cache'] = vc5
    out_kernel5, _ = run_paged_fwd(inp5, scale, True)
    out_ref5 = paged_prefill_reference(inp5['q'], inp5['k_cache'], inp5['v_cache'],
                                        inp5['block_table'], inp5['seq_lens'],
                                        inp5['query_start_loc'], inp5['prefix_kv_lens'],
                                        16, scale, True, 4)
    max_err5, _, _, cos5 = compute_error_metrics(out_kernel5, out_ref5)
    record("B5: Multi-seq interleaved blocks", max_err5 <= 2.2e-3 and cos5 > 0.999,
           f"max_err={max_err5:.2e}, cos={cos5:.6f}")


# ============================================================================
# GQA TESTS
# ============================================================================

def test_gqa():
    print("\n--- GQA Ratios ---")
    run_correctness_test("G1: H=2 KV_H=2 (MHA)", 1, [16], [32], 2, 2, 64, 16, True)
    run_correctness_test("G2: H=4 KV_H=2 (GQA 2:1)", 1, [32], [64], 4, 2, 64, 16, True)
    run_correctness_test("G3: H=14 KV_H=2 (Qwen 0.5B)", 1, [32], [128], 14, 2, 64, 16, True)
    run_correctness_test("G4: H=14 KV_H=2 short prefill", 1, [8], [16], 14, 2, 64, 16, True)
    run_correctness_test("G5: H=14 KV_H=2 long prefill", 1, [64], [256], 14, 2, 64, 16, True)
    run_correctness_test("G6: H=32 KV_H=8 D=128", 1, [64], [128], 32, 8, 128, 16, True)
    run_correctness_test("G7: H=32 KV_H=2 D=64", 1, [16], [32], 32, 2, 64, 16, True)
    run_correctness_test("G8: H=40 KV_H=5 D=128", 1, [8], [16], 40, 5, 128, 16, True)


# ============================================================================
# CHUNKED / PREFIX PREFiLL TESTS
# ============================================================================

def test_prefix_prefill():
    print("\n--- Chunked / Prefix Prefill ---")
    run_correctness_test("P1: Prefix=16 query=16 causal", 1, [16], [32], 4, 4, 64, 16, True)
    run_correctness_test("P2: Prefix=16 query=16 non-causal", 1, [16], [32], 4, 4, 64, 16, False)
    run_correctness_test("P3: Long prefix (56)", 1, [8], [64], 4, 4, 64, 16, True)
    run_correctness_test("P4: Tiny prefix (1)", 1, [32], [33], 4, 4, 64, 16, True)
    run_correctness_test("P5: D=128 prefix", 1, [64], [128], 8, 8, 128, 16, True)
    run_correctness_test("P6: D=256 prefix", 1, [32], [64], 4, 4, 256, 16, True)
    run_correctness_test("P7: Single-token decode-like", 1, [1], [65], 4, 4, 64, 16, True)


# ============================================================================
# MULTI-SEQUENCE BATCHED TESTS
# ============================================================================

def test_multi_sequence():
    print("\n--- Multi-Sequence Batched ---")
    run_correctness_test("M1: 2 seqs [1,16]", 2, [1, 16], [1, 16], 4, 4, 64, 16, True)
    run_correctness_test("M2: 4 seqs [1,16,5,16]", 4, [1, 16, 5, 16], [1, 16, 5, 16], 32, 4, 128, 16, True)
    run_correctness_test("M3: 8 seqs uniform", 8, [8,8,8,8,8,8,8,8], [8,8,8,8,8,8,8,8], 16, 16, 64, 16, True)
    run_correctness_test("M4: 4 seqs varying", 4, [64,128,32,96], [64,128,32,96], 8, 2, 128, 16, True)
    run_correctness_test("M5: 4 seqs with GQA", 4, [64,128,32,96], [64,128,32,96], 32, 8, 128, 16, True)
    run_correctness_test("M6: 4 seqs large", 4, [100,200,50,150], [100,200,50,150], 4, 4, 64, 16, True)


# ============================================================================
# NaN / INF SAFETY TESTS (CRITICAL)
# ============================================================================

def test_nan_safety():
    print("\n--- NaN / Inf Safety ---")

    # N1: All Q=0, random K/V
    inp = make_test_inputs(1, [32], [32], 8, 2, 64, 16, seed=10)
    inp['q'].zero_()
    scale = 1.0 / math.sqrt(64)
    out, _ = run_paged_fwd(inp, scale, True)
    ok, nan, inf = check_finite(out, "N1")
    record("N1: All-zero Q, random K/V", ok, f"NaN={nan}, Inf={inf}")

    # N2: All Q=0, all K/V=0
    inp2 = make_test_inputs(1, [32], [32], 8, 2, 64, 16, seed=11)
    inp2['q'].zero_()
    inp2['k_cache'].zero_()
    inp2['v_cache'].zero_()
    out2, _ = run_paged_fwd(inp2, scale, True)
    ok2, nan2, inf2 = check_finite(out2, "N2")
    record("N2: All zeros", ok2, f"NaN={nan2}, Inf={inf2}")

    # N3: Random Q, all K/V=0
    inp3 = make_test_inputs(1, [32], [32], 8, 2, 64, 16, seed=12)
    inp3['k_cache'].zero_()
    inp3['v_cache'].zero_()
    out3, _ = run_paged_fwd(inp3, scale, True)
    ok3, nan3, inf3 = check_finite(out3, "N3")
    # Also check: with all-zero KV, attention output should be zero
    max_val = out3.float().abs().max().item()
    record("N3: Random Q, zero K/V", ok3 and max_val < 1e-6,
           f"NaN={nan3}, Inf={inf3}, max_output={max_val:.2e}")

    # N4: UNINITIALIZED KV cache (torch.empty - simulates fresh allocator memory)
    total_q = 32
    num_heads = 8
    num_kv_heads = 2
    D = 64
    bs = 16
    num_seqs = 1
    q4 = torch.randn(total_q, num_heads, D, dtype=torch.float16, device='cuda')
    kc4 = torch.empty(8, bs, num_kv_heads, D, dtype=torch.float16, device='cuda')
    vc4 = torch.empty(8, bs, num_kv_heads, D, dtype=torch.float16, device='cuda')
    bt4 = make_block_table(num_seqs, [32], bs)[0]
    qsl4 = torch.tensor([0, 32], dtype=torch.int32, device='cuda')
    sl4 = torch.tensor([32], dtype=torch.int32, device='cuda')
    pkl4 = torch.tensor([0], dtype=torch.int32, device='cuda')
    out4 = torch.empty_like(q4)
    flash_attn_v100_cuda.paged_fwd(
        q4, kc4, vc4, bt4, sl4, qsl4, pkl4, out4,
        num_kv_heads, bs, float(scale), True
    )
    ok4, nan4, inf4 = check_finite(out4, "N4")
    record("N4: torch.empty KV cache (uninit memory)", ok4,
           f"NaN={nan4}, Inf={inf4}")

    # N5: Uninitialized cache + partial write
    q5 = torch.randn(total_q, num_heads, D, dtype=torch.float16, device='cuda')
    kc5 = torch.empty(8, bs, num_kv_heads, D, dtype=torch.float16, device='cuda')
    vc5 = torch.empty(8, bs, num_kv_heads, D, dtype=torch.float16, device='cuda')
    # Write valid data to first 2 blocks only (positions 0-31), rest uninitialized
    bt5, _ = make_block_table(num_seqs, [32], bs)
    kd5 = torch.randn(32, num_kv_heads, D, dtype=torch.float16, device='cuda')
    vd5 = torch.randn(32, num_kv_heads, D, dtype=torch.float16, device='cuda')
    # Write to blocks 0,1 (covering positions 0-31)
    for pos in range(32):
        blk = pos // bs
        off = pos % bs
        phys = bt5[0, blk].item()
        if phys >= 0:
            kc5[phys, off] = kd5[pos]
            vc5[phys, off] = vd5[pos]
    out5 = torch.empty_like(q5)
    flash_attn_v100_cuda.paged_fwd(
        q5, kc5, vc5, bt5, sl4, qsl4, pkl4, out5,
        num_kv_heads, bs, float(scale), True
    )
    ok5, nan5, inf5 = check_finite(out5, "N5")
    record("N5: Partial write + uninit cache", ok5,
           f"NaN={nan5}, Inf={inf5}")

    # N6: Extreme fp16 max values in Q
    inp6 = make_test_inputs(1, [16], [16], 4, 4, 64, 16, seed=13)
    inp6['q'].fill_(65504.0)  # fp16 max
    out6, _ = run_paged_fwd(inp6, scale, True)
    ok6, nan6, inf6 = check_finite(out6, "N6")
    record("N6: Q = fp16 max (65504)", ok6, f"NaN={nan6}, Inf={inf6}")

    # N7: Extreme fp16 max values in K/V
    inp7 = make_test_inputs(1, [16], [16], 4, 4, 64, 16, seed=14)
    inp7['k_cache'].fill_(65504.0)
    inp7['v_cache'].fill_(65504.0)
    out7, _ = run_paged_fwd(inp7, scale, True)
    ok7, nan7, inf7 = check_finite(out7, "N7")
    record("N7: K/V = fp16 max (65504)", ok7, f"NaN={nan7}, Inf={inf7}")

    # N8: All constant 1.0
    inp8 = make_test_inputs(1, [16], [16], 4, 4, 64, 16, seed=15)
    inp8['q'].fill_(1.0)
    inp8['k_cache'].fill_(1.0)
    inp8['v_cache'].fill_(1.0)
    out8, _ = run_paged_fwd(inp8, scale, True)
    ok8, nan8, inf8 = check_finite(out8, "N8")
    record("N8: All constant 1.0", ok8, f"NaN={nan8}, Inf={inf8}")

    # N9: Repeated calls with same inputs must be deterministic
    inp9 = make_test_inputs(1, [32], [32], 8, 2, 64, 16, seed=16)
    out9a, _ = run_paged_fwd(inp9, scale, True)
    out9b, _ = run_paged_fwd(inp9, scale, True)
    out9c, _ = run_paged_fwd(inp9, scale, True)
    diff_ab = (out9a.float() - out9b.float()).abs().max().item()
    diff_bc = (out9b.float() - out9c.float()).abs().max().item()
    ok9 = diff_ab == 0.0 and diff_bc == 0.0
    record("N9: Deterministic across calls", ok9,
           f"diff_ab={diff_ab:.2e}, diff_bc={diff_bc:.2e}")


# ============================================================================
# CROSS-KERNEL VALIDATION (Paged == Dense FA2)
# ============================================================================

def test_cross_kernel():
    print("\n--- Cross-Kernel Validation (Paged vs Dense FA2) ---")

    def cross_kernel_test(name, M, N, D, H, KV_H, causal, seed=100, tol=1e-4):
        bs = 16
        torch.manual_seed(seed)

        # Create dense data
        q = torch.randn(M, H, D, dtype=torch.float16, device='cuda')
        k = torch.randn(N, KV_H, D, dtype=torch.float16, device='cuda')
        v = torch.randn(N, KV_H, D, dtype=torch.float16, device='cuda')

        scale = 1.0 / math.sqrt(D)

        # Run dense FA2
        # Need to handle GQA: the dense kernel expects same num_heads for Q and K/V
        if H == KV_H:
            out_dense = dense_fa2_reference(q, k, v, scale, causal)
        else:
            # Dense FA2 doesn't support GQA directly. Skip comparison for GQA cases.
            # Instead compare paged kernel against reference only.
            pass

        # Build paged cache
        num_blocks = (N + bs - 1) // bs
        k_cache = torch.zeros(num_blocks, bs, KV_H, D, dtype=torch.float16, device='cuda')
        v_cache = torch.zeros(num_blocks, bs, KV_H, D, dtype=torch.float16, device='cuda')
        for pos in range(N):
            blk = pos // bs
            off = pos % bs
            k_cache[blk, off] = k[pos]
            v_cache[blk, off] = v[pos]

        bt = torch.tensor([[i for i in range(num_blocks)]], dtype=torch.int32, device='cuda')
        qsl = torch.tensor([0, M], dtype=torch.int32, device='cuda')
        sl = torch.tensor([N], dtype=torch.int32, device='cuda')
        pkl = torch.tensor([N - M], dtype=torch.int32, device='cuda')
        out_paged = torch.empty_like(q)

        flash_attn_v100_cuda.paged_fwd(
            q, k_cache, v_cache, bt, sl, qsl, pkl, out_paged,
            KV_H, bs, float(scale), causal
        )

        ok, nan, inf = check_finite(out_paged, name)
        if not ok:
            return record(name, False, f"NaN={nan}, Inf={inf}")

        if H == KV_H:
            max_diff = (out_paged.float() - out_dense.float()).abs().max().item()
            passed = max_diff <= tol
            return record(name, passed,
                          f"paged_vs_dense max_diff={max_diff:.2e}")
        else:
            # Just verify finite output via reference
            out_ref = paged_prefill_reference(q, k_cache, v_cache, bt, sl, qsl, pkl,
                                                bs, scale, causal, KV_H)
            max_err, _, _, cos = compute_error_metrics(out_paged, out_ref)
            return record(name, max_err <= 2.2e-3 and cos > 0.999,
                          f"max_err={max_err:.2e}, cos={cos:.6f} (ref only, GQA)")

    cross_kernel_test("X1: M=16 N=32 D=64 MHA causal", 16, 32, 64, 8, 8, True)
    cross_kernel_test("X2: M=64 N=128 D=64 MHA causal", 64, 128, 64, 8, 8, True)
    cross_kernel_test("X3: M=64 N=128 D=64 MHA non-causal", 64, 128, 64, 8, 8, False)
    cross_kernel_test("X4: M=16 N=32 D=64 GQA 4:1", 16, 32, 64, 8, 2, True)
    cross_kernel_test("X5: M=32 N=64 D=128 MHA causal", 32, 64, 128, 4, 4, True)
    cross_kernel_test("X6: M=32 N=64 D=128 GQA 4:1", 32, 64, 128, 16, 4, True)
    cross_kernel_test("X7: M=16 N=32 D=256 MHA causal", 16, 32, 256, 4, 4, True)


# ============================================================================
# WRITE-THEN-READ PIPELINE TESTS
# ============================================================================

def test_pipeline_roundtrip():
    """
    Simulates vLLM pipeline:
    1. do_kv_cache_update (triton_reshape_and_cache_flash) writes K/V to cache
    2. paged_fwd reads from cache
    3. Verify output matches direct attention on original K/V
    """
    print("\n--- Write-Then-Read Pipeline ---")

    try:
        from vllm.v1.attention.ops.triton_unified_attention import triton_reshape_and_cache_flash
        _has_triton_cache = True
    except ImportError:
        print("  ⚠️  triton_reshape_and_cache_flash not available, skipping pipeline tests")
        return

    bs = 16
    D = 64
    H = 14
    KV_H = 2
    seq_len = 32
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(200)

    # Original K/V
    k_orig = torch.randn(seq_len, KV_H, D, dtype=torch.float16, device='cuda')
    v_orig = torch.randn(seq_len, KV_H, D, dtype=torch.float16, device='cuda')
    q_orig = torch.randn(seq_len, H, D, dtype=torch.float16, device='cuda')

    # Build vLLM-style KV cache: [2, num_blocks, block_size, num_kv_heads, D]
    num_blocks = (seq_len + bs - 1) // bs
    kv_cache = torch.zeros(2, num_blocks, bs, KV_H, D, dtype=torch.float16, device='cuda')

    # Create slot_mapping
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device='cuda')

    # Step 1: Write K/V to cache using Triton reshape_and_cache
    key_cache, value_cache = kv_cache.unbind(0)
    triton_reshape_and_cache_flash(
        k_orig, v_orig,
        key_cache, value_cache,
        slot_mapping,
        "auto",
        torch.tensor(1.0, device='cuda'),
        torch.tensor(1.0, device='cuda'),
    )
    torch.cuda.synchronize()

    # Step 2: Reconstruct block_table matching slot_mapping
    bt = torch.zeros(1, num_blocks, dtype=torch.int32, device='cuda')
    for pos in range(seq_len):
        slot = slot_mapping[pos].item()
        phys_blk = slot // bs
        bt[0, pos // bs] = phys_blk

    qsl = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
    sl = torch.tensor([seq_len], dtype=torch.int32, device='cuda')
    pkl = torch.tensor([0], dtype=torch.int32, device='cuda')

    # Step 3: Run paged_fwd
    out_paged = torch.empty_like(q_orig)
    flash_attn_v100_cuda.paged_fwd(
        q_orig, key_cache, value_cache, bt, sl, qsl, pkl, out_paged,
        KV_H, bs, float(scale), True
    )
    torch.cuda.synchronize()

    # Verify
    ok, nan, inf = check_finite(out_paged, "R1_out")
    out_ref = paged_prefill_reference(q_orig, key_cache, value_cache, bt, sl, qsl, pkl,
                                        bs, scale, True, KV_H)
    max_err, _, _, cos = compute_error_metrics(out_paged, out_ref)
    passed = ok and max_err <= 2.2e-3 and cos > 0.999
    record("R1: Pipeline write-then-read (Qwen config)", passed,
           f"NaN={nan}, Inf={inf}, max_err={max_err:.2e}, cos={cos:.6f}")

    # R2: Non-contiguous slot_mapping
    # Write to sparse slots
    slot_mapping2 = torch.tensor([10, 20, 30, 5, 15, 25, 0, 1, 2, 3,
                                   4, 6, 7, 8, 9, 11, 12, 13, 14, 16,
                                   17, 18, 19, 21, 22, 23, 24, 26, 27, 28,
                                   29, 31], dtype=torch.int64, device='cuda')
    seq_len2 = 32
    k2 = torch.randn(seq_len2, KV_H, D, dtype=torch.float16, device='cuda')
    v2 = torch.randn(seq_len2, KV_H, D, dtype=torch.float16, device='cuda')
    q2 = torch.randn(seq_len2, H, D, dtype=torch.float16, device='cuda')

    num_blocks2 = 3  # 32/16=2, but sparse slots need more
    kv_cache2 = torch.zeros(2, num_blocks2, bs, KV_H, D, dtype=torch.float16, device='cuda')
    kc2, vc2 = kv_cache2.unbind(0)
    triton_reshape_and_cache_flash(k2, v2, kc2, vc2, slot_mapping2,
                                     "auto",
                                     torch.tensor(1.0, device='cuda'),
                                     torch.tensor(1.0, device='cuda'))
    torch.cuda.synchronize()

    # Build block_table from slot_mapping2
    bt2 = torch.full((1, num_blocks2), -1, dtype=torch.int32, device='cuda')
    for pos in range(seq_len2):
        slot = slot_mapping2[pos].item()
        phys_blk = slot // bs
        logical_blk = pos // bs
        if bt2[0, logical_blk] < 0:
            bt2[0, logical_blk] = phys_blk

    # Run paged_fwd (need to handle that some blocks might not be in bt2)
    out_paged2 = torch.empty_like(q2)
    flash_attn_v100_cuda.paged_fwd(
        q2, kc2, vc2, bt2, sl, qsl, pkl, out_paged2,
        KV_H, bs, float(scale), True
    )
    ok2, nan2, inf2 = check_finite(out_paged2, "R2_out")
    record("R2: Pipeline non-contiguous slot mapping", ok2,
           f"NaN={nan2}, Inf={inf2}")


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_edge_cases():
    print("\n--- Edge Cases ---")

    # E1: Zero query length (should return early)
    bs = 16
    D = 64
    inp = make_test_inputs(1, [0], [64], 4, 4, D, bs, seed=300)
    scale = 1.0 / math.sqrt(D)
    try:
        out, _ = run_paged_fwd(inp, scale, True)
        ok, nan, inf = check_finite(out, "E1")
        # All-zero output expected since zero query length
        max_val = out.float().abs().max().item() if out.numel() > 0 else 0.0
        # The kernel grid uses max_q_tiles = (num_tokens + BLOCK_M - 1) / BLOCK_M
        # with num_tokens=0, this becomes 0, so no blocks launch
        # But the launcher checks max_q_tiles > 0, so it should error
        # Actually the launcher does: TORCH_CHECK(max_q_tiles > 0 && max_q_tiles <= 2048)
        # Let's see what actually happens
        record("E1: Zero query length", ok, f"NaN={nan}, Inf={inf}, max_val={max_val:.2e}")
    except RuntimeError as e:
        # Expected: max_q_tiles == 0 causes assertion
        record("E1: Zero query length", True,
               f"Correctly rejected (max_q_tiles=0): {str(e)[:80]}")

    # E2: Zero KV length (seq_len=0)
    inp2 = make_test_inputs(1, [0], [0], 4, 4, D, bs, seed=301)
    try:
        out2, _ = run_paged_fwd(inp2, scale, True)
        ok2, _, _ = check_finite(out2, "E2")
        record("E2: Zero KV length", ok2)
    except RuntimeError as e:
        record("E2: Zero KV length", True,
               f"Correctly handled: {str(e)[:80]}")

    # E3: Single token, single seq (decode-like but through paged kernel)
    run_correctness_test("E3: Decode-like 1x1", 1, [1], [1], 14, 2, 64, 16, True)

    # E4: Very long KV sequence
    run_correctness_test("E4: Long KV (N=512)", 1, [128], [512], 4, 4, 64, 16, True)

    # E5: Very long KV with D=128
    run_correctness_test("E5: Long KV D=128 (N=512)", 1, [64], [512], 4, 4, 128, 16, True)

    # E6: Query longer than tile but shorter than KV (M=80, N=256)
    run_correctness_test("E6: M=80 N=256 D=64", 1, [80], [256], 4, 4, 64, 16, True)


# ============================================================================
# FIRST-RUN REPRODUCTION TEST
# ============================================================================

def test_first_run_isolation():
    """
    Test designed to reproduce the '!!!' bug:
    1. Use torch.empty (simulates uninitialized allocator memory)
    2. Run kernel ONCE (no warmup)
    3. Check output is finite AND correct
    4. Check that output is NOT all-constant (actual useful result)

    This is the closest we can get to the real first-prompt scenario
    without spinning up a full vLLM server.
    """
    print("\n--- First-Run Isolation Test (!!! bug reproduction) ---")

    D = 64
    H = 14  # Qwen 0.5B config
    KV_H = 2
    bs = 16
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(999)

    # Prefill: 64 token prompt
    seq_len = 64
    num_tokens = seq_len

    # Create realistic Q from random normal
    q = torch.randn(num_tokens, H, D, dtype=torch.float16, device='cuda')

    # Create paged K/V cache from random normal
    num_blocks = (seq_len + bs - 1) // bs  # = 4
    k_dense = torch.randn(seq_len, KV_H, D, dtype=torch.float16, device='cuda')
    v_dense = torch.randn(seq_len, KV_H, D, dtype=torch.float16, device='cuda')

    # Use torch.empty to simulate uninitialized memory
    k_cache = torch.empty(num_blocks, bs, KV_H, D, dtype=torch.float16, device='cuda')
    v_cache = torch.empty(num_blocks, bs, KV_H, D, dtype=torch.float16, device='cuda')

    # Then write ONLY the valid data (as do_kv_cache_update would)
    for pos in range(seq_len):
        blk = pos // bs
        off = pos % bs
        k_cache[blk, off] = k_dense[pos]
        v_cache[blk, off] = v_dense[pos]

    # Block table
    bt = make_block_table(1, [seq_len], bs)[0]
    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda')
    sl = torch.tensor([seq_len], dtype=torch.int32, device='cuda')
    pkl = torch.tensor([0], dtype=torch.int32, device='cuda')

    # Reference output first
    out_ref = paged_prefill_reference(q, k_cache, v_cache, bt, sl, qsl, pkl,
                                        bs, scale, True, KV_H)

    # NOW: fresh run with no prior kernel launches on this path
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    out_kernel = torch.empty_like(q)
    flash_attn_v100_cuda.paged_fwd(
        q, k_cache, v_cache, bt, sl, qsl, pkl, out_kernel,
        KV_H, bs, float(scale), True
    )
    torch.cuda.synchronize()

    # Check 1: finite output
    ok_f, nan_f, inf_f = check_finite(out_kernel, "first_run")
    if not ok_f:
        record("FR1: First-run finite output", False,
               f"NaN={nan_f}, Inf={inf_f} ❌ BUG REPRODUCED")
        # Print first few values for debugging
        print(f"    Kernel output sample (first 5 tokens, head 0):")
        print(f"    {out_kernel[:5, 0, :4]}")
        print(f"    Reference output sample:")
        print(f"    {out_ref[:5, 0, :4]}")
        return
    record("FR1: First-run finite output", True)

    # Check 2: correctness
    max_err, mean_err, rel_err, cos = compute_error_metrics(out_kernel, out_ref)
    passed = max_err <= 2.2e-3 and cos > 0.999
    record("FR2: First-run correctness", passed,
           f"max_err={max_err:.2e}, mean_err={mean_err:.2e}, cos={cos:.6f}")

    # Check 3: output is NOT trivially zero/constant
    out_max = out_kernel.float().abs().max().item()
    out_min = out_kernel.float().abs().min().item()
    out_mean = out_kernel.float().abs().mean().item()
    not_degenerate = out_max > 1e-6
    record("FR3: First-run non-degenerate output", not_degenerate,
           f"max|val|={out_max:.4f}, mean|val|={out_mean:.4f}")

    # Check 4: second run produces identical result
    out_kernel2 = torch.empty_like(q)
    flash_attn_v100_cuda.paged_fwd(
        q, k_cache, v_cache, bt, sl, qsl, pkl, out_kernel2,
        KV_H, bs, float(scale), True
    )
    torch.cuda.synchronize()
    diff_12 = (out_kernel.float() - out_kernel2.float()).abs().max().item()
    record("FR4: Second run identical to first", diff_12 == 0.0,
           f"diff_1-2={diff_12:.2e}")


# ============================================================================
# TOLERANCE BENCHMARKS
# ============================================================================

def test_tolerance_benchmarks():
    print("\n--- Tolerance Benchmarks ---")
    # Run several varied configs and check all pass tight tolerance
    configs = [
        ("L1: M=64 N=128 D=64 MHA", [64], [128], 8, 8, 64),
        ("L2: M=64 N=128 D=64 GQA", [64], [128], 16, 4, 64),
        ("L3: M=32 N=128 D=128 MHA", [32], [128], 8, 8, 128),
        ("L4: M=32 N=64 D=256 MHA", [32], [64], 4, 4, 256),
        ("L5: M=32 N=256 D=64 Qwen", [32], [256], 14, 2, 64),
    ]
    all_ok = True
    for name, ql, sl, H, KV_H, D in configs:
        inp = make_test_inputs(1, ql, sl, H, KV_H, D, 16, seed=400)
        scale = 1.0 / math.sqrt(D)
        out_k, _ = run_paged_fwd(inp, scale, True)
        out_r = paged_prefill_reference(inp['q'], inp['k_cache'], inp['v_cache'],
                                         inp['block_table'], inp['seq_lens'],
                                         inp['query_start_loc'], inp['prefix_kv_lens'],
                                         16, scale, True, KV_H)
        max_err, mean_err, rel_err, cos = compute_error_metrics(out_k, out_r)
        ok = max_err <= 2.2e-3 and cos > 0.999
        record(name, ok, f"max={max_err:.2e}, mean={mean_err:.2e}, cos={cos:.6f}")
        if not ok:
            all_ok = False
    return all_ok


# ============================================================================
# QWEN MODEL-SPECIFIC REPRODUCTION
# ============================================================================

def test_qwen_config():
    """Tests using exact Qwen 2.5-0.5B config: D=64, H=14, KV_H=2."""
    print("\n--- Qwen 0.5B Config Reproduction ---")
    run_correctness_test("Q1: 32-token Qwen prefill", 1, [32], [32], 14, 2, 64, 16, True)
    run_correctness_test("Q2: 128-token Qwen prefill", 1, [128], [128], 14, 2, 64, 16, True)
    run_correctness_test("Q3: 512-token Qwen prefill", 1, [512], [512], 14, 2, 64, 16, True)
    run_correctness_test("Q4: Qwen batch [32,64,16,128]", 4,
                         [32, 64, 16, 128], [32, 64, 16, 128], 14, 2, 64, 16, True)
    run_correctness_test("Q5: Qwen 64 tokens non-causal", 1, [64], [64], 14, 2, 64, 16, False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test FlashAttention-2 Paged Prefill kernel")
    parser.add_argument("--quick", action="store_true", help="Run only critical tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-long", action="store_true", help="Skip long-running tests")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    cap = torch.cuda.get_device_capability()
    if cap < (7, 0):
        print(f"Skipping: device capability {cap} < (7,0). Volta V100 required.")
        sys.exit(1)

    print(f"Running on {torch.cuda.get_device_name()} (capability {cap})")
    print(f"flash_attn_v100_cuda available: {flash_attn_v100_cuda is not None}")

    # Run tests
    test_self_validation()
    test_correctness_minimal()
    test_correctness_tile_boundaries()
    test_correctness_block_table()
    test_gqa()
    test_prefix_prefill()
    test_multi_sequence()
    test_edge_cases()

    if not args.skip_long:
        test_cross_kernel()
        test_pipeline_roundtrip()

    # NaN safety - always runs (critical)
    test_nan_safety()
    test_first_run_isolation()

    if not args.quick:
        test_tolerance_benchmarks()
        test_qwen_config()

    print_summary()

    # Exit code
    if any(not r['passed'] for r in _results):
        sys.exit(1)


if __name__ == "__main__":
    main()
