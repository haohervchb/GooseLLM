#!/usr/bin/env python3
"""
Comprehensive test suite for GatedDeltaNet (GDN) attention kernels.

Targets the 27B dense model config: K=128, V=128, H=24, chunk_size=64.
Tests both fused_recurrent (decode) and chunk_gated_delta_rule (prefill).

Usage:
    python test_gdn_prefill.py [--quick] [--verbose]
"""

import gc
import math
import sys
import time
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

try:
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.models.qwen3_next import fused_gdn_gating
    _HAS_GDN = True
except ImportError as e:
    print(f"GDN kernels not available: {e}")
    print("Run: pip install -e . first")
    _HAS_GDN = False

# ============================================================================
# REFERENCE IMPLEMENTATION
# ============================================================================

def gdn_reference_sequential(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H]  — log-space gate (raw from fused_gdn_gating)
    beta: torch.Tensor,  # [B, T, H]  — sigmoid output in [0,1]
    scale: float,
    initial_state: torch.Tensor | None = None,  # [B, H, V, K]
    use_qk_l2norm: bool = True,
):
    """
    Sequential (token-by-token) Gated DeltaNet reference.
    Matches the fused_recurrent kernel behavior.

    Args:
        q: queries   [B, T, H, K]
        k: keys      [B, T, H, K]
        v: values    [B, T, H, V]
        g: log-gate  [B, T, H] (from fused_gdn_gating, will be exp'd)
        beta: gating [B, T, H] (sigmoid, in [0,1])
        scale: attention scale
        initial_state: [B, H, V, K] or None (zeros)
        use_qk_l2norm: L2 normalize q and k
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    device = q.device
    dtype_out = q.dtype

    if initial_state is None:
        h = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
    else:
        h = initial_state.float().clone()

    q = q.float()
    k = k.float()
    v = v.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=device)

    for t in range(T):
        b_q = q[:, t]  # [B, H, K]
        b_k = k[:, t]  # [B, H, K]
        b_v = v[:, t]  # [B, H, V]
        b_g = g[:, t]  # [B, H]
        b_beta = beta[:, t]  # [B, H]

        if use_qk_l2norm:
            b_q = b_q / (torch.sqrt((b_q * b_q).sum(dim=-1, keepdim=True)) + 1e-6)
            b_k = b_k / (torch.sqrt((b_k * b_k).sum(dim=-1, keepdim=True)) + 1e-6)

        b_q = b_q * scale

        # Apply forget gate: h = exp(g) * h
        h = h * torch.exp(b_g)[:, :, None, None]  # [B, H, V, K]

        # Compute error: r = v - h @ k^T  (per-head matmul)
        # h: [B, H, V, K], k: [B, H, K] -> [B, H, V]
        hk = torch.einsum('bhvk,bhk->bhv', h, b_k)
        r = b_v - hk  # [B, H, V]

        # Apply beta: r *= beta
        r = r * b_beta[:, :, None]  # [B, H, V]

        # Update hidden state: h += r ⊗ k
        h = h + torch.einsum('bhv,bhk->bhvk', r, b_k)

        # Output: o = h @ q^T
        b_o = torch.einsum('bhvk,bhk->bhv', h, b_q)
        o[:, t] = b_o

    return o.to(dtype_out)


# ============================================================================
# HELPERS
# ============================================================================

def make_gdn_inputs(B, T, H, V, K, seed=None, device='cuda', use_l2norm=True):
    """Create random inputs for GDN kernel testing. Uses Qwen 27B params."""
    if seed is not None:
        torch.manual_seed(seed)

    q = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, V, dtype=torch.float16, device=device)

    if use_l2norm:
        q = F.normalize(q.float(), p=2, dim=-1).to(torch.float16)
        k = F.normalize(k.float(), p=2, dim=-1).to(torch.float16)

    # Gate parameters (matching Qwen's fused_gdn_gating)
    a = torch.randn(B * T, H, dtype=torch.float16, device=device) * 0.1
    b = torch.randn(B * T, H, dtype=torch.float16, device=device) * 0.1
    dt_bias = torch.randn(H, dtype=torch.float16, device=device) * 0.1
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.1 - 3.0  # A_log ~ -3

    g_raw, beta_raw = fused_gdn_gating(A_log, a, b, dt_bias)
    # g_raw: [1, B*T, H] -> [B, T, H]
    g = g_raw.squeeze(0).reshape(B, T, H)
    beta = beta_raw.squeeze(0).reshape(B, T, H)

    scale = K ** -0.5

    return {
        'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta,
        'scale': scale, 'B': B, 'T': T, 'H': H, 'V': V, 'K': K,
    }


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
# TEST: FUSED RECURRENT (DECODE KERNEL)
# ============================================================================

def test_decode_correctness():
    print("\n--- Decode: Fused Recurrent Correctness ---")

    H, V, K = 24, 128, 128  # Qwen 27B config: head_dim=256 split as K=128, V=128

    def run_decode_test(name, B, T, seed=42):
        inp = make_gdn_inputs(B, T, H, V, K, seed=seed)

        # Run fused_recurrent kernel (decode path)
        o_kernel, _ = fused_recurrent_gated_delta_rule(
            q=inp['q'],
            k=inp['k'],
            v=inp['v'],
            g=inp['g'],
            beta=inp['beta'],
            initial_state=torch.zeros(B, H, V, K, dtype=torch.float32, device='cuda'),
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        ok, nan, inf = check_finite(o_kernel, name)
        if not ok:
            return record(name, False, f"NaN={nan}, Inf={inf}")

        # Reference
        o_ref = gdn_reference_sequential(
            inp['q'], inp['k'], inp['v'], inp['g'], inp['beta'],
            inp['scale'],
            initial_state=torch.zeros(B, H, V, K, dtype=torch.float32, device='cuda'),
            use_qk_l2norm=True,
        )

        max_err, mean_err, rel_err, cos = compute_error_metrics(o_kernel, o_ref)
        passed = max_err <= 5e-3 and cos > 0.99
        return record(name, passed,
                      f"max_err={max_err:.2e}, cos={cos:.6f}")

    # Single token (decode)
    run_decode_test("D1: T=1 B=1", 1, 1)
    run_decode_test("D2: T=1 B=4", 4, 1)

    # Multi-token (small batch prefill via decode path)
    run_decode_test("D3: T=2 B=1", 1, 2)
    run_decode_test("D4: T=4 B=1", 1, 4)
    run_decode_test("D5: T=8 B=1", 1, 8)
    run_decode_test("D6: T=16 B=1", 1, 16)
    run_decode_test("D7: T=32 B=1", 1, 32)
    run_decode_test("D8: T=1 B=8", 8, 1)


# ============================================================================
# TEST: CHUNKED PREFiLL KERNEL
# ============================================================================

def test_chunked_prefill_correctness():
    print("\n--- Prefill: Chunked DeltaRule Correctness ---")

    H, V, K = 24, 128, 128

    def run_prefill_test(name, B, T, seed=42, use_varlen=False):
        inp = make_gdn_inputs(B, T, H, V, K, seed=seed)

        if use_varlen:
            # Flatten to [1, B*T, ...] with cu_seqlens
            q = inp['q'].reshape(1, B * T, H, K).contiguous()
            k = inp['k'].reshape(1, B * T, H, K).contiguous()
            v = inp['v'].reshape(1, B * T, H, V).contiguous()
            g = inp['g'].reshape(1, B * T, H).contiguous()
            beta = inp['beta'].reshape(1, B * T, H).contiguous()
            cu_seqlens = torch.tensor(
                [i * T for i in range(B + 1)], dtype=torch.int64, device='cuda'
            )
            initial_state = torch.zeros(B, H, V, K, dtype=torch.float32, device='cuda')
        else:
            q, k, v, g, beta = inp['q'], inp['k'], inp['v'], inp['g'], inp['beta']
            cu_seqlens = None
            initial_state = torch.zeros(B, H, V, K, dtype=torch.float32, device='cuda')

        # Run chunked prefill kernel
        o_kernel, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=inp['scale'],
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        if use_varlen:
            o_kernel = o_kernel.reshape(B, T, H, V)

        ok, nan, inf = check_finite(o_kernel, name)
        if not ok:
            return record(name, False, f"NaN={nan}, Inf={inf}")

        # Reference (sequential)
        o_ref = gdn_reference_sequential(
            inp['q'], inp['k'], inp['v'], inp['g'], inp['beta'],
            inp['scale'],
            initial_state=torch.zeros(B, H, V, K, dtype=torch.float32, device='cuda'),
            use_qk_l2norm=True,
        )

        max_err, mean_err, rel_err, cos = compute_error_metrics(o_kernel, o_ref)
        passed = max_err <= 5e-3 and cos > 0.99
        return record(name, passed,
                      f"max_err={max_err:.2e}, cos={cos:.6f}")

    # Small sequences — critical for finding NaN threshold
    run_prefill_test("P1: T=1 B=1", 1, 1)
    run_prefill_test("P2: T=2 B=1", 1, 2)
    run_prefill_test("P3: T=3 B=1", 1, 3)
    run_prefill_test("P4: T=4 B=1", 1, 4)
    run_prefill_test("P5: T=5 B=1", 1, 5)
    run_prefill_test("P6: T=8 B=1", 1, 8)
    run_prefill_test("P7: T=16 B=1", 1, 16)

    # Chunk size boundaries (chunk_size=64)
    run_prefill_test("P8: T=63 B=1", 1, 63)
    run_prefill_test("P9: T=64 B=1", 1, 64)
    run_prefill_test("P10: T=65 B=1", 1, 65)
    run_prefill_test("P11: T=128 B=1", 1, 128)

    # Multi-batch
    run_prefill_test("P12: T=16 B=2", 2, 16)
    run_prefill_test("P13: T=32 B=4", 4, 32)

    # Variable-length (cu_seqlens)
    run_prefill_test("P14: T=16 B=4 varlen", 4, 16, use_varlen=True)
    run_prefill_test("P15: T=64 B=2 varlen", 2, 64, use_varlen=True)


# ============================================================================
# TEST: NaN / INF SAFETY
# ============================================================================

def test_gdn_nan_safety():
    print("\n--- GDN NaN / Inf Safety ---")

    H, V, K = 24, 128, 128
    T = 16
    B = 2
    device = 'cuda'

    def make_custom(B, T, H, V, K, seed=42):
        inp = make_gdn_inputs(B, T, H, V, K, seed=seed)
        initial_state = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)
        return inp, initial_state

    # N1: All zeros
    inp, init = make_custom(1, T, H, V, K, 101)
    for k in ['q', 'k', 'v', 'g', 'beta']:
        inp[k].zero_()
    o, _ = chunk_gated_delta_rule(
        q=inp['q'], k=inp['k'], v=inp['v'], g=inp['g'], beta=inp['beta'],
        scale=inp['scale'], initial_state=init, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    ok, nan, inf = check_finite(o, "N1")
    record("N1: All-zero inputs prefill", ok, f"NaN={nan}, Inf={inf}")

    # N2: All zeros decode
    inp2, init2 = make_custom(1, 1, H, V, K, 102)
    for k in ['q', 'k', 'v', 'g', 'beta']:
        inp2[k].zero_()
    o2, _ = fused_recurrent_gated_delta_rule(
        q=inp2['q'], k=inp2['k'], v=inp2['v'], g=inp2['g'], beta=inp2['beta'],
        initial_state=init2, inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    ok2, nan2, inf2 = check_finite(o2, "N2")
    record("N2: All-zero inputs decode", ok2, f"NaN={nan2}, Inf={inf2}")

    # N3: Uninitialized initial state (torch.empty)
    for seed in [200, 201, 202, 203, 204]:
        inp3, _ = make_custom(1, T, H, V, K, seed)
        init3 = torch.empty(1, H, V, K, dtype=torch.float32, device=device)
        o3, _ = chunk_gated_delta_rule(
            q=inp3['q'], k=inp3['k'], v=inp3['v'], g=inp3['g'], beta=inp3['beta'],
            scale=inp3['scale'], initial_state=init3, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        ok3, nan3, inf3 = check_finite(o3, f"N3_s{seed}")
        if not ok3:
            break
    record("N3: torch.empty initial state (5 seeds)", ok3,
           f"NaN={nan3}, Inf={inf3} (last seed)")

    # N4: Extreme input values
    inp4, init4 = make_custom(1, T, H, V, K, 205)
    inp4['q'].fill_(100.0)
    inp4['k'].fill_(100.0)
    inp4['v'].fill_(100.0)
    o4, _ = chunk_gated_delta_rule(
        q=inp4['q'], k=inp4['k'], v=inp4['v'], g=inp4['g'], beta=inp4['beta'],
        scale=inp4['scale'], initial_state=init4, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    ok4, nan4, inf4 = check_finite(o4, "N4")
    record("N4: Extreme values (100.0) prefill", ok4, f"NaN={nan4}, Inf={inf4}")

    # N5: Very negative gate (large negative g -> exp(g) ≈ 0)
    inp5, init5 = make_custom(1, T, H, V, K, 206)
    inp5['g'].fill_(-100.0)
    o5, _ = chunk_gated_delta_rule(
        q=inp5['q'], k=inp5['k'], v=inp5['v'], g=inp5['g'], beta=inp5['beta'],
        scale=inp5['scale'], initial_state=init5, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    ok5, nan5, inf5 = check_finite(o5, "N5")
    record("N5: Very negative gates (-100.0) prefill", ok5,
           f"NaN={nan5}, Inf={inf5}")

    # N6: Very positive gate (gate > 0 -> exp(g) > 1)
    inp6, init6 = make_custom(1, T, H, V, K, 207)
    inp6['g'].fill_(10.0)
    o6, _ = chunk_gated_delta_rule(
        q=inp6['q'], k=inp6['k'], v=inp6['v'], g=inp6['g'], beta=inp6['beta'],
        scale=inp6['scale'], initial_state=init6, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    ok6, nan6, inf6 = check_finite(o6, "N6")
    record("N6: Very positive gates (10.0) prefill", ok6,
           f"NaN={nan6}, Inf={inf6}")

    # N7: Determinism — same inputs → same output
    inp7, init7 = make_custom(1, 32, H, V, K, 300)
    o7a, _ = chunk_gated_delta_rule(
        q=inp7['q'], k=inp7['k'], v=inp7['v'], g=inp7['g'], beta=inp7['beta'],
        scale=inp7['scale'], initial_state=init7, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    o7b, _ = chunk_gated_delta_rule(
        q=inp7['q'], k=inp7['k'], v=inp7['v'], g=inp7['g'], beta=inp7['beta'],
        scale=inp7['scale'], initial_state=init7, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    diff = (o7a.float() - o7b.float()).abs().max().item()
    record("N7: Deterministic prefill calls", diff == 0.0,
           f"diff_1-2={diff:.2e}")


# ============================================================================
# TEST: CHUNKED vs FUSED RECURRENT CONSISTENCY
# ============================================================================

def test_chunked_vs_fused():
    """
    The chunked prefill and fused recurrent (decode) kernels should produce
    the same output when the initial state is the same and sequence is identical.
    """
    print("\n--- Prefill vs Decode Consistency ---")

    H, V, K = 24, 128, 128
    B = 1
    device = 'cuda'

    for T in [1, 2, 4, 8, 16, 32]:
        inp = make_gdn_inputs(B, T, H, V, K, seed=400 + T)
        initial_state = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)

        o_prefill, _ = chunk_gated_delta_rule(
            q=inp['q'], k=inp['k'], v=inp['v'], g=inp['g'], beta=inp['beta'],
            scale=inp['scale'], initial_state=initial_state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        o_decode, _ = fused_recurrent_gated_delta_rule(
            q=inp['q'], k=inp['k'], v=inp['v'], g=inp['g'], beta=inp['beta'],
            initial_state=initial_state,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        ok_p, nan_p, inf_p = check_finite(o_prefill, f"C_T{T}_prefill")
        ok_d, nan_d, inf_d = check_finite(o_decode, f"C_T{T}_decode")
        if not (ok_p and ok_d):
            record(f"C1: T={T} prefill==decode", False,
                   f"prefill NaN={nan_p} Inf={inf_p}, decode NaN={nan_d} Inf={inf_d}")
            continue

        max_err, _, _, cos = compute_error_metrics(o_prefill, o_decode)
        record(f"C1: T={T} prefill==decode", max_err <= 1e-3 and cos > 0.999,
               f"max_err={max_err:.2e}, cos={cos:.6f}")


# ============================================================================
# TEST: CHUNK SIZE BOUNDARIES
# ============================================================================

def test_chunk_size_boundaries():
    print("\n--- Chunk Size Boundaries ---")
    H, V, K = 24, 128, 128
    B = 1
    device = 'cuda'

    # chunk_size=64 is hardcoded; test boundaries
    # The BT (block time) is computed as min(64, next_power_of_2(T))
    for T in [1, 2, 4, 8, 16, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256]:
        inp = make_gdn_inputs(B, T, H, V, K, seed=T)
        init = torch.zeros(B, H, V, K, dtype=torch.float32, device=device)

        o_kernel, _ = chunk_gated_delta_rule(
            q=inp['q'], k=inp['k'], v=inp['v'], g=inp['g'], beta=inp['beta'],
            scale=inp['scale'], initial_state=init,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        ok, nan, inf = check_finite(o_kernel, f"T={T}")
        if not ok:
            record(f"B1: T={T} finite", False, f"NaN={nan}, Inf={inf}")
            continue

        o_ref = gdn_reference_sequential(
            inp['q'], inp['k'], inp['v'], inp['g'], inp['beta'],
            inp['scale'],
            initial_state=torch.zeros(B, H, V, K, dtype=torch.float32, device=device),
            use_qk_l2norm=True,
        )
        max_err, _, _, cos = compute_error_metrics(o_kernel, o_ref)
        record(f"B1: T={T} correct", max_err <= 5e-3 and cos > 0.99,
               f"max_err={max_err:.2e}, cos={cos:.6f}, nan=0")


# ============================================================================
# TEST: HEAD DIMENSION VARIANTS
# ============================================================================

def test_head_dim_variants():
    print("\n--- Head Dimension Variants ---")

    for name, H, V, K in [
        ("V1: H=8 K=64 V=64", 8, 64, 64),
        ("V2: H=16 K=128 V=128", 16, 128, 128),
        ("V3: H=24 K=128 V=128", 24, 128, 128),  # Qwen 27B
        ("V4: H=32 K=128 V=128", 32, 128, 128),
        ("V5: H=24 K=64 V=192", 24, 64, 192),  # Uneven K/V
    ]:
        inp = make_gdn_inputs(1, 16, H, V, K, seed=500)
        init = torch.zeros(1, H, V, K, dtype=torch.float32, device='cuda')

        o, _ = chunk_gated_delta_rule(
            q=inp['q'], k=inp['k'], v=inp['v'], g=inp['g'], beta=inp['beta'],
            scale=inp['scale'], initial_state=init,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        ok, nan, inf = check_finite(o, name)
        if not ok:
            record(name, False, f"NaN={nan}, Inf={inf}")
            continue

        o_ref = gdn_reference_sequential(
            inp['q'], inp['k'], inp['v'], inp['g'], inp['beta'],
            inp['scale'],
            initial_state=torch.zeros(1, H, V, K, dtype=torch.float32, device='cuda'),
            use_qk_l2norm=True,
        )
        max_err, _, _, cos = compute_error_metrics(o, o_ref)
        record(name, max_err <= 5e-3 and cos > 0.99,
               f"max_err={max_err:.2e}, cos={cos:.6f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test GDN attention kernels")
    parser.add_argument("--quick", action="store_true", help="Run only critical tests")
    parser.add_argument("--skip-long", action="store_true", help="Skip slow tests")
    args = parser.parse_args()

    if not _HAS_GDN:
        print("GDN kernels not importable — run 'pip install -e .' first")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    cap = torch.cuda.get_device_capability()
    print(f"Running on {torch.cuda.get_device_name()} (capability {cap})")
    print(f"GDN kernels: available")

    # Run tests
    test_decode_correctness()

    if not args.skip_long:
        test_chunked_prefill_correctness()

    if not args.skip_long:
        test_chunked_vs_fused()

    if not args.quick:
        test_chunk_size_boundaries()

    test_gdn_nan_safety()

    if not args.quick:
        test_head_dim_variants()

    print_summary()

    if any(not r['passed'] for r in _results):
        sys.exit(1)


if __name__ == "__main__":
    main()
