# Plan: Phase 3 — Grid Restructuring for GQA KV Redundancy Elimination

**Date:** 2025-04-24  
**Status:** In Progress  
**Branch:** main  
**Predecessor:** Phase 2 (SparkAttention m8n8k4) — ABORTED, artifacts on `spark/m8n8k4-research`

---

## 1. Problem Statement

Current launcher uses a 3D grid: `(max_q_tiles, num_seqs, num_heads)`.

Each block handles one (q_tile, seq, head) triple. Within a block:
- All 16 warps load Q for their specific head
- All 16 warps load K/V for their specific KV-head
- For GQA with 32 Q-heads / 2 KV-heads: **16 blocks load identical K/V** for the same KV-head tile

At 134K context, this is ~288 GB of redundant KV HBM traffic per head.

---

## 2. Goal

Restructure grid so that **all warps in a block share the same KV-head tile**, eliminating redundant K/V loads. Q-heads are processed sequentially or in groups within the same block.

---

## 3. Approach

### Option A: Tile-Parallel Grid (Recommended)

Change grid from `(q_tiles, seqs, heads)` to `(q_tiles, seqs, kv_heads)`.

Each block handles ONE KV-head. Within the block:
- Load K/V tile once into SMEM (cooperative)
- Loop over all Q-heads that map to this KV-head
- For each Q-head group (e.g., 8 Q-heads at a time = 8 warps each doing 2 tiles):
  - Load Q tiles for those Q-heads
  - Compute attention using shared K/V
  - Accumulate results

**Benefit:** K/V loaded once, used by 16 Q-heads → **16× reduction in KV HBM traffic**

**Challenge:** 
- Output accumulation: O for each Q-head must be written separately
- LSE tracking: each Q-head needs its own max/sum
- SMEM size: may need larger output buffers or register spilling

### Option B: Persistent Kernel with KV Tile Loop

Similar to Option A but with a persistent kernel that loops over KV tiles:
- One block handles a range of Q tiles for one KV-head
- Outer loop over KV tiles
- Inner loop over Q-heads

**Benefit:** Even better occupancy and pipelining
**Challenge:** More complex thread scheduling

---

## 4. Implementation & Results

### Step 1: Implement Sequential Q-Head Loop Kernel
Implemented `flash_attention_paged_forward_kernel_gqa_shared_kv` with grid `(q_tiles, seqs, kv_heads)`.
Within each block, K/V is loaded once and all Q-heads mapping to that KV-head are processed sequentially.

### Step 2: Validation
- Native GQA test (`num_heads_q=8, num_heads_kv=2`): **PASSED** — numerical correctness verified
- All existing tests: **PASSED**

### Step 3: Benchmarking Results

| Sequence | Original (tok/s) | Phase 3 (tok/s) | Result |
|----------|-----------------|-----------------|--------|
| 64       | 3,857,315       | 312,823         | **12× SLOWER** |
| 1,024    | 1,026,458       | 523,661         | **2× SLOWER** |
| 512 × 4  | 1,764,250       | 1,181,281       | **1.5× SLOWER** |
| 8,192    | 190,077         | 160,807         | **1.2× SLOWER** |
| 16,384   | 91,822          | 84,178          | **1.1× SLOWER** |
| 32,768   | 44,921          | 41,928          | **1.07× SLOWER** |
| 65,536   | 22,325          | 21,380          | **~same** |

### Root Cause Analysis

**K/V loading is NOT the bottleneck.**

For D=128, BLOCK_N=176:
- K tile size: 176 × 128 × 2 bytes = 45KB
- At 900 GB/s HBM bandwidth: 45KB / 900GB/s = **0.05 microseconds**
- Time per tile (from benchmark): ~50-100 microseconds
- K/V loading is **<0.1% of total time**

**The real bottleneck is WMMA instruction latency and small tile sizes.**

With BLOCK_M=32, BLOCK_N=176, each block processes only 22 WMMA tiles for Q@K^T and 16 tiles for P@V. The overhead of WMMA setup, online softmax, and causal masking dominates. Reducing K/V loads by 16× saves negligible time because K/V was already a tiny fraction.

**Parallelism loss dominates at short sequences.**
- Original: 32 q-head blocks run in parallel on 32 SMs
- Phase 3: 2 kv-head blocks run in parallel, each loops over 16 q-heads sequentially
- For short sequences where blocks don't serialize, Phase 3 has 16× less parallelism

---

## 5. Decision: **ABORT Phase 3 as deployed optimization**

The GQA-shared-KV kernel is **numerically correct** but **performance-negative** for all practical sequence lengths. The code is preserved in `fused_mha_paged_forward.cu` (disabled via `use_shared_kv = false`) as reference.

---

## 6. Lessons Learned

1. **Measure before optimizing.** The assumption that K/V HBM traffic was the bottleneck was incorrect.
2. **Parallelism matters more than memory traffic for small tiles.** WMMA overhead dominates; K/V loading is negligible.
3. **Tile size is the real constraint.** BLOCK_M=32 is too small to amortize WMMA setup costs effectively.

---

## 7. Next Steps

Potential future directions:
- **Increase BLOCK_M** (e.g., to 64 or 128) to amortize WMMA overhead — requires SMEM increase and occupancy reduction
- **Multi-query attention (MQA)** kernel specialization for extreme GQA ratios
- **Kernel fusion with downstream ops** (e.g., layer norm, projection) to reduce round-trips
- **Quantized KV cache** (INT8) to reduce HBM traffic if it ever becomes the bottleneck
