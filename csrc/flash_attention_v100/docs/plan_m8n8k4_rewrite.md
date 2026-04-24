# Plan: Raw Volta mma.sync.m8n8k4 Rewrite (Phase 2C/2D)

**Date:** 2025-04-24  
**Objective:** Replace WMMA `m16n16k16` with raw PTX `mma.sync.m8n8k4` to enable register-level C→A layout transform, eliminating SMEM round-trip for P matrix.

---

## 1. Background

Current WMMA `m16n16k16` path requires writing P to SMEM between Q@K^T and P@V. SparkAttention paper shows that Volta's native `m8n8k4` supports a low-cost `shfl_xor_sync(2)` transform from FP32 accumulator to f16 A-fragment, avoiding SMEM entirely.

Our empirical probes confirmed that WMMA `m16n16k16` C→A transform requires **64+ shuffle ops per warp** (too expensive). The raw `m8n8k4` path needs only **~8 shuffle ops per thread** (documented in PTX ISA §9.7.14.5.1 and SparkAttention Fig 8b).

---

## 2. Technical Foundation

### Volta `mma.sync.m8n8k4` mechanics
- One warp (32 threads) executes **4 concurrent** `m8n8k4` ops (quadpairs)
- Quadpair groups: lanes {0-3,16-19}, {4-7,20-23}, {8-11,24-27}, {12-15,28-31}
- Each quadpair computes one **8×8 output tile** from A(8×4) × B(4×8)
- Per-thread fragments: **4 f16** for A (2 `f16x2` regs), **4 f16** for B, **8 f32** for C/D

### Throughput equivalence
For D=128, both approaches need **8 MMA instructions** per 16×16 output area:
- WMMA `m16n16k16`: 8 instr × 4,096 MACs = 32,768 MACs
- Raw `m8n8k4` (4 concurrent): 8 instr × 4,096 MACs = 32,768 MACs

**Compute throughput is identical.** The win must come from I/O reduction.

### C→A layout transform
- FP32 accumulator C (8 f32/thread) → f16 A-fragment (4 f16/thread)
- Pattern: `shfl_xor_sync(0xFF, val, 2)` between thread pairs
- Thread pairs: (0↔2), (1↔3), (16↔18), (17↔19), etc.
- Cost: ~8 shuffle ops + 8 f32→f16 conversions per thread per 8×8 tile

---

## 3. Priority

**D=128 and D=256 are CRITICAL for LLM workloads.** D=64 is secondary.

| Head Dim | Priority | Status |
|----------|----------|--------|
| D=128    | P0       | Primary target (Qwen3.5-122B-A10B-AWQ) |
| D=256    | P0       | Secondary target (larger models) |
| D=64     | P1       | Only if time permits |

---

## 4. Implementation Steps

### Step 1: Raw m8n8k4 Backend Header (`04_mma_m8n8k4.cuh`)
- Define `MmaM8N8K4` struct with fragment types
- Implement `mma_m8n8k4_sync()` inline PTX wrapper
- Implement SMEM→register load functions for A and B
- Implement register→SMEM store for D (final output only)

**Effort:** ~150 lines, ~4 hours  
**Risk:** Low — docs/volta.md has exact PTX syntax

### Step 2: Empirical Fragment Layout Verification
- Probe kernel to verify C-fragment layout matches PTX ISA formulas
- Probe kernel to verify A-fragment layout
- Verify 4 quadpairs produce 2×2 grid of 8×8 tiles = 16×16 total

**Effort:** ~100 lines throwaway code, ~4 hours  
**Risk:** Low — PTX ISA gives exact formulas

### Step 3: FP32 C→A Layout Transform (`volta_layout_transform_c_to_a`)
- Per-quadpair transform: 8 f32 → 4 f16 via `shfl_xor_sync(2)`
- Pack converted f16 into 2 `uint32_t` f16x2 registers
- Pattern derived from PTX ISA + SparkAttention paper

**Effort:** ~80 lines, ~6 hours  
**Risk:** MEDIUM — need empirical verification

### Step 4: Kernel Rewrite (`fused_mha_paged_forward_m8n8k4.cu`)
- New kernel `flash_attention_paged_forward_kernel_m8n8k4<D, IS_CAUSAL>`
- BLOCK_M/BLOCK_N multiples of 8; one warp handles 16×16 output area
- Inner loop per KV tile:
  1. Cooperative K load to SMEM
  2. Warp loads Q slice to registers
  3. Issue `mma_m8n8k4_sync` for Q@K^T
  4. Online softmax in registers
  5. Causal mask at 8×8 granularity
  6. **C→A transform via shuffles**
  7. Issue `mma_m8n8k4_sync` for P@V
  8. Scale and accumulate to O

**Effort:** ~400 lines, ~16 hours  
**Risk:** HIGH — complete rewrite

### Step 5: Integration & Fallback
- Launcher dispatches via new boolean flag
- Keep original WMMA kernel as fallback
- Compile-time switch for development

**Effort:** ~30 lines, ~2 hours  
**Risk:** Low

### Step 6: Validation
- Numerical exactness vs PyTorch reference
- Causal masking correctness
- Paged KV cache with block tables
- GQA with num_heads_q > num_heads_kv
- Chunked prefill scenarios

**Effort:** ~8 hours  
**Risk:** Medium

### Step 7: Benchmarking
- Short (64), medium (1K-8K), long (32K-65K)
- Compare: original WMMA, pragmatic Spark, raw m8n8k4, PyTorch

**Effort:** ~4 hours  
**Risk:** HIGH

---

## 5. Decision Gates

### Gate 1 (Day 1): Empirical Probe Results
- Probe C and A fragment layouts for m8n8k4
- Measure raw `shfl_xor_sync(2)` latency on V100
- **Go/No-Go:** If shuffle latency > ~20 cycles/thread, ABORT

### Gate 2 (Day 2): Transform Verification
- Micro-kernel: Q@K^T → C→A → P@V for one 16×16 tile
- Compare vs reference (cuBLAS or WMMA)
- **Go/No-Go:** If error > 0.1% or inf/nan unresolvable in 1 day, ABORT

### Gate 3 (Day 3): Full Kernel Benchmark
- Complete D=128 kernel rewrite
- Benchmark vs original at seq=1K, 8K, 32K
- **Go/No-Go:** If speedup < 5% at any length, ABORT → Phase 3

---

## 6. Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| C→A transform doesn't eliminate enough SMEM traffic | Medium | High | Gate 1 probe |
| m8n8k4 kernel slower due to more k-tiles | Low | High | Proven throughput-equivalent |
| PTX fragment layout formulas misinterpreted | Medium | High | Gate 2 empirical verification |
| Causal masking at 8×8 granularity complex/error-prone | Medium | High | Unit test early |
| Register pressure causes spills | Medium | Medium | Check PTXAS output |
| GQA interaction breaks | Medium | High | Test GQA early |
| Week+ work with no speedup | Medium | Critical | Time-box to 3 days max |

---

## 7. Effort Estimate

| Step | Lines | Time |
|------|-------|------|
| Backend header | 150 | 4h |
| Empirical probes | 100 | 4h |
| C→A transform | 80 | 6h |
| Kernel rewrite | 400 | 16h |
| Integration | 30 | 2h |
| Validation | — | 8h |
| Benchmark | — | 4h |
| **Total** | **~760** | **~44h (5-6 dev days)** |

---

## 8. Fallback Strategy

If any gate fails, immediately pivot to **Phase 3: Grid Restructuring for GQA KV Redundancy Elimination**. The m8n8k4 code remains in repository (disabled) as reference.

---

## 9. Results & Decision (2025-04-24)

### What Was Proven

| Test | Result |
|------|--------|
| Basic `mma.m8n8k4` functionality | **PASSED** — all 256 outputs correct |
| C-fragment layout (empirical) | **VERIFIED** — matches PTX ISA §9.7.14.5.1 exactly |
| C→A transform correctness | **VERIFIED** — all 32 lanes produce correct A-fragments |
| Micro-benchmark (16×16 tile, D=128) | **1.21× speedup** for m8n8k4 vs WMMA per tile |
| `shfl_xor_sync(2)` latency | **20 cycles** per shuffle |

### The Fatal Problem: Tile Granularity for Inference

The micro-benchmark speedup is **real but irrelevant** at full-kernel scale.

**WMMA m16n16k16** (current):
- One instruction = one 16×16 tile = 256 output elements
- For D=128, BLOCK_M=32, BLOCK_N=176: ~11 MMA instructions per warp

**Raw m8n8k4** (proposed):
- One instruction = 4 concurrent 8×8 tiles = 256 output elements
- But the 4 quadpairs compute the **same** tile; to cover different tiles requires separate instructions
- For same BLOCK_M/BLOCK_N: **~176 MMA instructions per warp** (16× more)

At ~7 cycles per m8n8k4 instruction vs ~40 cycles per WMMA instruction (amortized), the full-kernel projection is:

| Path | Instructions/warp | Est. cycles/warp | Relative |
|------|-------------------|------------------|----------|
| WMMA | ~11 | ~4,300 | 1.0× |
| m8n8k4 | ~176 | ~14,300 | **3.3× SLOWER** |

The 1.2× SMEM savings per tile is completely overwhelmed by the 16× instruction overhead.

### Why SparkAttention Still Wins for Training

SparkAttention uses m8n8k4 for **training** with large **batch sizes**. In training, the batch dimension provides parallelism — many independent tiles execute in parallel across warps. In **inference** with paged attention, we have **long sequences** and small batches. WMMA's larger 16×16 tiles amortize instruction overhead much better for this workload.

### Decision: **ABORT m8n8k4 rewrite. Proceed to Phase 3.**

Phase 3 (grid restructuring for GQA KV redundancy) targets the **actual bottleneck**: 16× redundant KV HBM loads per tile. Eliminating this would save ~288 GB of HBM traffic per head at 134K context — orders of magnitude more impactful than any SMEM optimization.

### Artifacts Preserved

- `include/04_mma_m8n8k4.cuh` — raw m8n8k4 PTX backend
- `docs/volta.md` — Volta inline PTX documentation
- Probe sources: `probe_m8n8k4.cu`, `probe_wmma_layout*.cu`, `bench_m8n8k4*.cu`

These are preserved on branch `spark/m8n8k4-research` for future reference.

---

## 10. Notes

- D=128 and D=256 are P0. D=64 is P1 (only if time permits).
- Temporary debug `printf` instrumentation allowed during development; must be removed before commit.
- Original WMMA kernel must remain as runtime-selectable fallback.
- All changes in `flash-attention-v100-ai-bond` repo; wrapper in `1Cat-vLLM` repo unchanged.
