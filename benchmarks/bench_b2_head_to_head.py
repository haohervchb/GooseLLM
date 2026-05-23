#!/usr/bin/env python3
"""B2: Production attention vs TileLang decode kernel — direct comparison.
No model load. Just imports both implementations, creates representative tensors,
and times them at varying batch sizes.
"""
import sys, os, time, copy, torch
import warnings
warnings.filterwarnings("ignore")

# ── Add TileLang to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "3rdparty", "tilelang-fa-v100"))
import tilelang_fa_v100
tilelang_decode = tilelang_fa_v100.tilelang_decode_forward

# ── Benchmark config (matches your real models) ──
HEAD_DIM = 256
NUM_KV_HEADS = 2
NUM_HEADS_Q = 16
BLOCK_SIZE = 16
SEQ_LEN = 1000
MAX_BLOCKS = 256

def make_tensors(bsz):
    nb = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    q = torch.randn(bsz, NUM_HEADS_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
    k4 = torch.randn(nb, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    v4 = torch.randn_like(k4)
    bt = torch.full((bsz, MAX_BLOCKS), -1, dtype=torch.int32, device="cuda")
    for i in range(bsz):
        bt[i, :nb] = torch.arange(nb, dtype=torch.int32, device="cuda")
    sl = torch.full((bsz,), SEQ_LEN, dtype=torch.int32, device="cuda")
    return q, k4, v4, bt, sl

# ═══════════════════════════════════════════════════════════════
# PATH A: TileLang decode kernel
# ═══════════════════════════════════════════════════════════════

def bench_tilelang(bsz_list):
    results = {}
    for bsz in bsz_list:
        q, k4, v4, bt, sl = make_tensors(bsz)
        # Warmup
        for _ in range(10):
            tilelang_decode(q, k4, v4, bt, sl, block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            tilelang_decode(q, k4, v4, bt, sl, block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS)
        e.record(); torch.cuda.synchronize()
        results[bsz] = s.elapsed_time(e) / 100 * 1000
    return results

# ═══════════════════════════════════════════════════════════════
# PATH B: Production attention (TritonAttentionImpl.decode path) 
# ═══════════════════════════════════════════════════════════════

def bench_production_decode(bsz_list):
    """Call the actual production decode attention kernel.
    
    FlashAttnTileLangV100Impl.forward() for decode falls to:
      super().forward() → TritonAttentionImpl.forward()
      → sm70_paged_decode_attention() or unified_attention()
    
    We replicate the exact call path.
    """
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    results = {}
    for bsz in bsz_list:
        q, k4, v4, bt, sl = make_tensors(bsz)
        
        # unified_attention expects 4D paged K/V: [blocks, block_size, num_kv_heads, head_dim]
        # q is [batch, num_heads, head_dim]
        out = torch.empty(bsz, NUM_HEADS_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
        
        # Build args matching TritonAttentionImpl.forward() → unified_attention() call
        cu_seqlens_q = torch.arange(bsz + 1, dtype=torch.int32, device="cuda")
        max_seqlen_q = 1  # decode
        seqused_k = sl
        max_seqlen_k = SEQ_LEN
        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)
        causal = True
        window_size = (-1, -1)
        softcap = 0.0
        q_descale = None
        k_descale = None
        v_descale = None
        
        # Warmup
        for _ in range(10):
            unified_attention(
                q, k4, v4, out,
                cu_seqlens_q, max_seqlen_q,
                seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size,
                bt, softcap,
                q_descale, k_descale, v_descale,
            )
        torch.cuda.synchronize()
        
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100):
            unified_attention(
                q, k4, v4, out,
                cu_seqlens_q, max_seqlen_q,
                seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size,
                bt, softcap,
                q_descale, k_descale, v_descale,
            )
        e.record(); torch.cuda.synchronize()
        results[bsz] = s.elapsed_time(e) / 100 * 1000
    return results

# ═══════════════════════════════════════════════════════════════
# PATH C: Production attention via SM70 SIMT decode kernel
# ═══════════════════════════════════════════════════════════════

def bench_sm70_simt_decode(bsz_list):
    """Call the SM70 SIMT paged decode attention kernel directly.
    This is what TritonAttentionImpl uses when available on SM70.
    """
    results = {}
    for bsz in bsz_list:
        q, k4, v4, bt, sl = make_tensors(bsz)
        out = torch.empty(bsz, NUM_HEADS_Q, HEAD_DIM, dtype=torch.float16, device="cuda")
        scale = 1.0 / (HEAD_DIM ** 0.5)
        
        try:
            from vllm.v1.attention.ops.sm70_decode import sm70_paged_decode_attention
            # Warmup
            for _ in range(10):
                sm70_paged_decode_attention(out, q, k4, v4, NUM_KV_HEADS, scale, bt, sl, BLOCK_SIZE, SEQ_LEN)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(100):
                sm70_paged_decode_attention(out, q, k4, v4, NUM_KV_HEADS, scale, bt, sl, BLOCK_SIZE, SEQ_LEN)
            e.record(); torch.cuda.synchronize()
            results[bsz] = s.elapsed_time(e) / 100 * 1000
        except Exception as err:
            results[bsz] = f"CRASH: {type(err).__name__}"
    return results

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("B2: TileLang vs Production Attention — Head-to-head")
    print(f"Config: head_dim={HEAD_DIM}, kv_heads={NUM_KV_HEADS}, q_heads={NUM_HEADS_Q}")
    print("=" * 70)

    bsz_list = [1, 2, 4, 8, 16, 32]

    print("\nWarming up TileLang...")
    tl = bench_tilelang(bsz_list)

    print("Warming up Production (Triton unified_attention)...")
    triton_decode = bench_production_decode(bsz_list)

    print("Testing Production (SM70 SIMT decode)...")
    sm70 = bench_sm70_simt_decode(bsz_list)

    # ── Print results ──
    print(f"\n{'bsz':>5s}  {'TileLang(μs)':>14s}  {'Triton(μs)':>14s}  {'SM70(μs)':>14s}  "
          f"{'TileLang/Tr':>12s}  {'Tr faster?':>12s}")
    print("-" * 76)
    for bsz in bsz_list:
        tl_str = f"{tl[bsz]:>10.1f}" if isinstance(tl[bsz], float) else f"{str(tl[bsz]):>10s}"
        tr_str = f"{triton_decode[bsz]:>10.1f}" if isinstance(triton_decode[bsz], float) else f"{str(triton_decode[bsz]):>10s}"
        sm_str = f"{sm70[bsz]:>10.1f}" if isinstance(sm70[bsz], float) else f"{str(sm70[bsz]):>10s}"
        
        if isinstance(tl[bsz], float) and isinstance(triton_decode[bsz], float):
            ratio = tl[bsz] / triton_decode[bsz]
            faster = "Triton" if triton_decode[bsz] < tl[bsz] else "TileLang"
            print(f"  {bsz:>3d}  {tl_str:>14s}  {tr_str:>14s}  {sm_str:>14s}  "
                  f"{ratio:>10.2f}x  {faster:>12s}")
        else:
            print(f"  {bsz:>3d}  {tl_str:>14s}  {tr_str:>14s}  {sm_str:>14s}  "
                  f"{'N/A':>10s}  {'N/A':>12s}")
    
    # ── Per-token efficiency ──
    print(f"\n{'bsz':>5s}  {'TL μs/tok':>12s}  {'Tr μs/tok':>12s}  ")
    print("-" * 36)
    for bsz in bsz_list:
        if isinstance(tl[bsz], float) and isinstance(triton_decode[bsz], float):
            print(f"  {bsz:>3d}  {tl[bsz]/bsz:>10.1f}    {triton_decode[bsz]/bsz:>10.1f}")
    
    print("\nCONCLUSION:")
    if all(isinstance(tl[b], float) for b in bsz_list) and all(isinstance(triton_decode[b], float) for b in bsz_list):
        # Find crossover point
        crossover = None
        for bsz in bsz_list:
            if tl[bsz] < triton_decode[bsz]:
                crossover = bsz
                break
        if crossover:
            print(f"  TileLang decode beats production Triton at bsz >= {crossover}")
        else:
            print(f"  Production Triton is faster at all batch sizes tested")
        
        # Overall speedup at max batch
        bsz_max = max(bsz_list)
        if tl[bsz_max] < triton_decode[bsz_max]:
            spd = triton_decode[bsz_max] / tl[bsz_max]
            print(f"  At bsz={bsz_max}: TileLang is {spd:.2f}x faster ({triton_decode[bsz_max]:.0f} → {tl[bsz_max]:.0f} μs)")
        else:
            spd = tl[bsz_max] / triton_decode[bsz_max]
            print(f"  At bsz={bsz_max}: Triton is {spd:.2f}x faster ({tl[bsz_max]:.0f} → {triton_decode[bsz_max]:.0f} μs)")
    else:
        print("  See above for partial results")
    
    print()
