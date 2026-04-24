// ======================================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
// FlashAttention-2 Paged Forward for V100 (SM70)
// 
// Supports KV cache stored in page blocks (block table), same FA2 WMMA pipeline
// as fused_mha_forward.cu.
//
// Input layout (vLLM format):
//   Q:            [num_tokens, num_heads, head_dim] — dense, concatenated across seqs
//   k_cache:      [num_blocks, block_size, num_kv_heads, head_dim] — paged KV-K
//   v_cache:      [num_blocks, block_size, num_kv_heads, head_dim] — paged KV-V  
//   block_table:  [num_seqs, max_blocks_per_seq] — int32 block indices
//   seq_lens:     [num_seqs] — total sequence length (int32)
//   query_start_loc: [num_seqs + 1] — cumulative query positions in Q (int32)
//   prefix_kv_lens:  [num_seqs] — KV positions where query starts in sequence (int32)
//   block_size:   tokens per block (typically 16)
//   out:          [num_tokens, num_heads, head_dim] — output
//   softmax_lse:  [num_heads, num_tokens] — log-sum-exp values
//
// Execution:
//   Grid: (max_q_tiles, num_seqs, num_heads) — one block per (Q-tile, seq, head)
//   Native GQA: each Q-head block loads K/V using kv_head_id = head_id / group_size
//   so concurrent Q-head blocks mapping to the same KV-head share L2 cache.
//   Uses existing WMMA_GEMM_* templates for the compute pipeline.
//
// Only forward pass is needed (inference-only for vLLM).
// ======================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <algorithm>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "00_volta_const.cuh"
#include "01_forward_config.cuh"
#include "02_wmma.cuh"

// --------------------------------------------------------------------------
// FORWARD KERNEL — paged KV cache with prefix support and native GQA
// Grid: (max_q_tiles, num_seqs, num_heads) — 3D
// Each block processes one (Q-tile, seq, head) tile.
// Q is 3D flat [num_tokens, num_heads, D] indexed via query_start_loc
// K/V are paged [num_blocks, block_size, num_kv_heads, D] indexed via block_table
// --------------------------------------------------------------------------
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(512, 2)
flash_attention_paged_forward_kernel(
    const uint64_t q_base_addr,       // Q: [num_tokens, num_heads, D] — flat 3D
    const uint64_t k_cache_addr,      // KV cache K: [num_blocks, block_size, num_kv_heads, D]
    const uint64_t v_cache_addr,      // KV cache V: [num_blocks, block_size, num_kv_heads, D]
    const uint64_t out_base_addr,     // Out: [num_tokens, num_heads, D]
    const int* __restrict__ block_table,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,     // [num_seqs] — total sequence lengths
    const int* __restrict__ query_start_loc, // [num_seqs + 1] — Q positions
    const int* __restrict__ prefix_kv_lens,  // [num_seqs] — KV offset per sequence
    float* __restrict__ softmax_lse,      // [num_heads, num_tokens]
    const int num_seqs,
    const int max_blocks_per_seq,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int num_tokens,
    const float softmax_scale)
{
    using Config = KernelConfig<D>;
    
    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int TPB     = 512;
    constexpr int TPR     = TPB / BLOCK_M;
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int D_STRIDE = Config::D_STRIDE;
    constexpr int N_STRIDE = Config::N_STRIDE;
    constexpr int D_STRIDE_U4 = (D_STRIDE + 7) >> 3;

    // ---- Identify work ----
    const int q_tile = blockIdx.x;
    const int seq    = blockIdx.y;
    const int head_id = blockIdx.z;

    if (seq >= num_seqs) return;
    if (head_id >= num_heads) return;

    const int seq_len = seq_lens[seq];
    if (seq_len <= 0) return;

    // Prefix length: KV positions before the query starts
    const int prefix_len = prefix_kv_lens[seq];
    // Query length: how many tokens we need to process Q for this sequence
    const int query_len = seq_len - prefix_len;
    // Q buffer start for this sequence
    const int q_abs = query_start_loc[seq];

    if (query_len <= 0) return;
    if (q_abs + q_tile * BLOCK_M >= q_abs + query_len) return;
    
    const int start_q_local = q_tile * BLOCK_M;  // position within query buffer
    const int valid_q = min(BLOCK_M, query_len - start_q_local);

    // Number of KV tiles = ceil(seq_len / BLOCK_N)
    int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    // Causal: restrict KV tiles based on global KV position
    if constexpr (IS_CAUSAL) {
        const int max_kv_for_q = prefix_len + start_q_local + valid_q - 1;
        if (max_kv_for_q < 0) { num_kv_tiles = 0; }
        else { num_kv_tiles = min(num_kv_tiles, (max_kv_for_q + 1 + BLOCK_N - 1) / BLOCK_N); }
    }

    // ---- Thread IDs ----
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // Sequence-specific block table pointer
    const int* bt = block_table + seq * max_blocks_per_seq;

    // ---- Shared memory ----
    extern __shared__ char smem_raw[];
    
    // Reinterpret parsed into struct fields
    auto& _s = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);
    __half* sQ   = _s.q;
    __half* sK   = _s.reuse_kv.k;
    __half* sV   = _s.reuse_kv.v;
    float*  sS   = _s.reuse_sp.s;
    __half* sP   = _s.reuse_sp.p;
    float*  sO   = _s.o;
    float*  sMax = _s.row_max;
    float*  sSum = _s.row_sum;

    // Row stride for Q/output vs KV cache (native GQA)
    const uint64_t q_row_stride  = static_cast<uint64_t>(num_heads)    * D * 2;
    const uint64_t kv_row_stride = static_cast<uint64_t>(num_kv_heads) * D * 2;

    // Head offsets
    const uint64_t q_head_offset  = static_cast<uint64_t>(head_id) * D * 2;
    const int group_size = num_heads / num_kv_heads;
    const int kv_head_id = head_id / group_size;
    const uint64_t kv_head_offset = static_cast<uint64_t>(kv_head_id) * D * 2;

    // Global Q start position for this tile
    const int global_q_start = q_abs + start_q_local;

    // Zero sO, sMax, sSum for this block
    const int o_elems = BLOCK_M * D_STRIDE;
    for (int i = tid; i < o_elems; i += TPB) {
        sO[i] = 0.0f;
    }
    if (tid < BLOCK_M) {
        sMax[tid] = NEG_INF;
        sSum[tid] = 1.0f;
    }
    __syncthreads();

    // ---- Load Q tile from flat 3D buffer for this head ----
    {
        int q_u4_per_row = (D + 7) >> 3;
        const int q_total = valid_q * q_u4_per_row;
        uint32_t q_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
        
        for (int i = tid; i < q_total; i += TPB) {
            const int r = i / q_u4_per_row;
            const int c = i % q_u4_per_row;
            const int abs_r = global_q_start + r;
            
            if (abs_r >= global_q_start + valid_q) {
                uint32_t d = q_dst + static_cast<uint32_t>(r * D_STRIDE_U4 + c) * 16;
                asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                             :: "r"(d) : "memory");
                continue;
            }
            
            uint64_t src = q_base_addr 
                         + static_cast<uint64_t>(abs_r) * q_row_stride
                         + q_head_offset
                         + static_cast<uint64_t>(c) * 16;
            
            uint32_t d = q_dst + static_cast<uint32_t>(r * D_STRIDE_U4 + c) * 16;
            
            uint32_t r0, r1, r2, r3;
            __asm__ volatile(
                "{\n  .reg .pred p;\n"
                "  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                "}\n"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                : "l"(src), "r"(d) : "memory");
        }
    }

    __syncthreads();

    // ---- Save Q pointer for WMMA ----
    __half* qs = sQ;
    float*  ss = sS;
    
    // ---- Main loop: iterate over KV tiles ----
    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const int kv_start = tile * BLOCK_N;
        if (kv_start >= seq_len) break;
        const int kv_valid = min(BLOCK_N, seq_len - kv_start);

        // Causal early skip
        if constexpr (IS_CAUSAL) {
            if (kv_start >= prefix_len + start_q_local + valid_q) continue;
        }

        // ---- Load K tile from paged cache for this KV head ----
        {
            constexpr int k_u4 = D_STRIDE_U4;
            const int k_total = kv_valid * k_u4;
            uint32_t k_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
            if (k_total == 0) {
                __syncthreads();
                continue;
            }

            for (int i = tid; i < k_total; i += TPB) {
                const int tok = i / k_u4;
                const int u4c = i % k_u4;

                int pos = kv_start + tok;
                if (pos >= seq_len) {
                    uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                int blk = pos / block_size;
                int off = pos % block_size;
                if (blk >= max_blocks_per_seq) {
                    uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                int phys = bt[blk];
                if (phys < 0) {
                    uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                uint64_t src = k_cache_addr
                             + static_cast<uint64_t>(phys) * block_size * kv_row_stride
                             + static_cast<uint64_t>(off) * kv_row_stride
                             + kv_head_offset
                             + static_cast<uint64_t>(u4c) * 16;

                uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;

                uint32_t r0, r1, r2, r3;
                __asm__ volatile(
                    "{\n  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                    "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                    "}\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "l"(src), "r"(d) : "memory");
            }
        }

        __syncthreads();

        // ---- S = Q @ K^T ----
        WMMA_GEMM_SCORES<GemmType::sQ_KT, D, IS_CAUSAL, BLOCK_M, BLOCK_N, 
                         D_STRIDE, N_STRIDE, WARPS_PER_BLOCK>(
            qs, sK, ss,
            valid_q, kv_valid,
            prefix_len + start_q_local, kv_start,
            softmax_scale,
            warp_id, lane_id
        );

        __syncthreads();

        // ---- Online softmax + O scaling ----
        ONLINE_SOFTMAX<BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, TPR>(
            sS, sP, sO,
            sMax, sSum,
            valid_q, kv_valid,
            tid, tile
        );

        __syncthreads();

        // ---- Load V tile from paged cache for this KV head ----
        {
            constexpr int v_u4 = D_STRIDE_U4;
            const int v_total = kv_valid * v_u4;
            uint32_t v_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
            if (v_total == 0) {
                __syncthreads();
                continue;
            }

            for (int i = tid; i < v_total; i += TPB) {
                const int tok = i / v_u4;
                const int u4c = i % v_u4;

                int pos = kv_start + tok;
                if (pos >= seq_len) {
                    uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                int blk = pos / block_size;
                int off = pos % block_size;
                if (blk >= max_blocks_per_seq) {
                    uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                int phys = bt[blk];
                if (phys < 0) {
                    uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};"
                                 :: "r"(d) : "memory");
                    continue;
                }

                uint64_t src = v_cache_addr
                             + static_cast<uint64_t>(phys) * block_size * kv_row_stride
                             + static_cast<uint64_t>(off) * kv_row_stride
                             + kv_head_offset
                             + static_cast<uint64_t>(u4c) * 16;

                uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;

                uint32_t r0, r1, r2, r3;
                __asm__ volatile(
                    "{\n  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                    "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                    "}\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "l"(src), "r"(d) : "memory");
            }
        }

        __syncthreads();

        // ---- dO += P @ V ----
        WMMA_GEMM_GRADIENTS<GemmType::dO_PV, D, BLOCK_M, BLOCK_N, 
                            N_STRIDE, D_STRIDE, WARPS_PER_BLOCK>(
            sP, sV, sO,
            valid_q, kv_valid,
            warp_id, lane_id
        );

        __syncthreads();
    }

    // ---- Epilogue: store normalized output to flat 3D output for this head ----
    {
        constexpr int global_chunks = D >> 2;
        const int total_iters = valid_q * global_chunks;
        for (int i = tid; i < total_iters; i += TPB) {
            const int r = i / global_chunks;
            const int col = (i % global_chunks) << 2;

            float norm = __frcp_rn(fmaxf(sSum[r], 1e-24f));
            
            const float4 v = *reinterpret_cast<const float4*>(sO + r * D_STRIDE + col);
            __half2 h0 = __float22half2_rn(make_float2(v.x * norm, v.y * norm));
            __half2 h1 = __float22half2_rn(make_float2(v.z * norm, v.w * norm));
            
            ushort v0 = __half_as_ushort(h0.x), v1 = __half_as_ushort(h0.y);
            ushort v2 = __half_as_ushort(h1.x), v3 = __half_as_ushort(h1.y);
            
            uint64_t addr = out_base_addr
                          + static_cast<uint64_t>(global_q_start + r) * q_row_stride
                          + q_head_offset
                          + static_cast<uint64_t>(col) * 2;
            __asm__ volatile(
                "st.global.v4.u16 [%0], {%1,%2,%3,%4};"
                : : "l"(addr), "h"(v0), "h"(v1), "h"(v2), "h"(v3)
                : "memory");
        }
    }

    // ---- Write LSE to flat output for this head ----
    if (tid < valid_q) {
        const float sum = fmaxf(sSum[tid], 1e-24f);
        int lse_idx = head_id * num_tokens + global_q_start + tid;
        softmax_lse[lse_idx] = sMax[tid] + logf(sum);
    }
}

// --------------------------------------------------------------------------
// PHASE 3: GQA-OPTIMIZED KERNEL — shares K/V across Q-heads within a block
//
// Grid: (max_q_tiles, num_seqs, num_kv_heads) — one block per (Q-tile, seq, KV-head)
// Each block loads K/V once, then loops over all Q-heads mapping to that KV-head.
// This eliminates redundant K/V HBM loads for GQA (e.g., 32 Q-heads / 2 KV-heads
// saves 16x K/V traffic).
// --------------------------------------------------------------------------
template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(512, 2)
flash_attention_paged_forward_kernel_gqa_shared_kv(
    const uint64_t q_base_addr,
    const uint64_t k_cache_addr,
    const uint64_t v_cache_addr,
    const uint64_t out_base_addr,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ prefix_kv_lens,
    float* __restrict__ softmax_lse,
    const int num_seqs,
    const int max_blocks_per_seq,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int num_tokens,
    const float softmax_scale)
{
    using Config = KernelConfig<D>;

    constexpr int BLOCK_M = Config::BLOCK_M;
    constexpr int BLOCK_N = Config::BLOCK_N;
    constexpr int TPB     = 512;
    constexpr int TPR     = TPB / BLOCK_M;
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int D_STRIDE = Config::D_STRIDE;
    constexpr int N_STRIDE = Config::N_STRIDE;
    constexpr int D_STRIDE_U4 = (D_STRIDE + 7) >> 3;

    // ---- Identify work ----
    const int q_tile    = blockIdx.x;
    const int seq       = blockIdx.y;
    const int kv_head_id = blockIdx.z;

    if (seq >= num_seqs) return;
    if (kv_head_id >= num_kv_heads) return;

    const int seq_len = seq_lens[seq];
    if (seq_len <= 0) return;

    const int prefix_len = prefix_kv_lens[seq];
    const int query_len = seq_len - prefix_len;
    const int q_abs = query_start_loc[seq];

    if (query_len <= 0) return;
    if (q_abs + q_tile * BLOCK_M >= q_abs + query_len) return;

    const int start_q_local = q_tile * BLOCK_M;
    const int valid_q = min(BLOCK_M, query_len - start_q_local);

    int num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    if constexpr (IS_CAUSAL) {
        const int max_kv_for_q = prefix_len + start_q_local + valid_q - 1;
        if (max_kv_for_q < 0) { num_kv_tiles = 0; }
        else { num_kv_tiles = min(num_kv_tiles, (max_kv_for_q + 1 + BLOCK_N - 1) / BLOCK_N); }
    }

    // ---- Thread IDs ----
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int* bt = block_table + seq * max_blocks_per_seq;

    // ---- Shared memory ----
    extern __shared__ char smem_raw[];
    auto& _s = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw);
    __half* sQ = _s.q;
    __half* sK = _s.reuse_kv.k;
    __half* sV = _s.reuse_kv.v;
    float*  sS = _s.reuse_sp.s;
    __half* sP = _s.reuse_sp.p;
    float*  sO = _s.o;
    float*  sMax = _s.row_max;
    float*  sSum = _s.row_sum;

    const uint64_t q_row_stride  = static_cast<uint64_t>(num_heads)    * D * 2;
    const uint64_t kv_row_stride = static_cast<uint64_t>(num_kv_heads) * D * 2;

    const int group_size = num_heads / num_kv_heads;
    const uint64_t kv_head_offset = static_cast<uint64_t>(kv_head_id) * D * 2;

    const int global_q_start = q_abs + start_q_local;

    // ---- Loop over all Q-heads that map to this KV-head ----
    for (int q_head_offset = 0; q_head_offset < group_size; ++q_head_offset) {
        const int head_id = kv_head_id * group_size + q_head_offset;
        const uint64_t q_head_offset_bytes = static_cast<uint64_t>(head_id) * D * 2;

        // Zero sO, sMax, sSum for this Q-head
        const int o_elems = BLOCK_M * D_STRIDE;
        for (int i = tid; i < o_elems; i += TPB) sO[i] = 0.0f;
        if (tid < BLOCK_M) {
            sMax[tid] = NEG_INF;
            sSum[tid] = 1.0f;
        }
        __syncthreads();

        // Load Q tile for this Q-head
        {
            int q_u4_per_row = (D + 7) >> 3;
            const int q_total = valid_q * q_u4_per_row;
            uint32_t q_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
            for (int i = tid; i < q_total; i += TPB) {
                const int r = i / q_u4_per_row;
                const int c = i % q_u4_per_row;
                const int abs_r = global_q_start + r;
                if (abs_r >= global_q_start + valid_q) {
                    uint32_t d = q_dst + static_cast<uint32_t>(r * D_STRIDE_U4 + c) * 16;
                    asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                    continue;
                }
                uint64_t src = q_base_addr
                             + static_cast<uint64_t>(abs_r) * q_row_stride
                             + q_head_offset_bytes
                             + static_cast<uint64_t>(c) * 16;
                uint32_t d = q_dst + static_cast<uint32_t>(r * D_STRIDE_U4 + c) * 16;
                uint32_t r0, r1, r2, r3;
                __asm__ volatile(
                    "{\n  .reg .pred p;\n"
                    "  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                    "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                    "}\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "l"(src), "r"(d) : "memory");
            }
        }
        __syncthreads();

        __half* qs = sQ;
        float*  ss = sS;

        // ---- Main loop: iterate over KV tiles ----
        for (int tile = 0; tile < num_kv_tiles; ++tile) {
            const int kv_start = tile * BLOCK_N;
            if (kv_start >= seq_len) break;
            const int kv_valid = min(BLOCK_N, seq_len - kv_start);

            if constexpr (IS_CAUSAL) {
                if (kv_start >= prefix_len + start_q_local + valid_q) continue;
            }

            // Load K tile from paged cache (shared across all Q-heads in this block)
            {
                constexpr int k_u4 = D_STRIDE_U4;
                const int k_total = kv_valid * k_u4;
                uint32_t k_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
                if (k_total == 0) { __syncthreads(); continue; }
                for (int i = tid; i < k_total; i += TPB) {
                    const int tok = i / k_u4;
                    const int u4c = i % k_u4;
                    int pos = kv_start + tok;
                    if (pos >= seq_len) {
                        uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    int blk = pos / block_size;
                    int off = pos % block_size;
                    if (blk >= max_blocks_per_seq) {
                        uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    int phys = bt[blk];
                    if (phys < 0) {
                        uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    uint64_t src = k_cache_addr
                                 + static_cast<uint64_t>(phys) * block_size * kv_row_stride
                                 + static_cast<uint64_t>(off) * kv_row_stride
                                 + kv_head_offset
                                 + static_cast<uint64_t>(u4c) * 16;
                    uint32_t d = k_dst + static_cast<uint32_t>(tok * k_u4 + u4c) * 16;
                    uint32_t r0, r1, r2, r3;
                    __asm__ volatile(
                        "{\n  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                        "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                        "}\n"
                        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                        : "l"(src), "r"(d) : "memory");
                }
            }
            __syncthreads();

            // S = Q @ K^T
            WMMA_GEMM_SCORES<GemmType::sQ_KT, D, IS_CAUSAL, BLOCK_M, BLOCK_N,
                             D_STRIDE, N_STRIDE, WARPS_PER_BLOCK>(
                qs, sK, ss,
                valid_q, kv_valid,
                prefix_len + start_q_local, kv_start,
                softmax_scale,
                warp_id, lane_id
            );
            __syncthreads();

            // Online softmax
            ONLINE_SOFTMAX<BLOCK_M, BLOCK_N, N_STRIDE, D_STRIDE, TPR>(
                sS, sP, sO,
                sMax, sSum,
                valid_q, kv_valid,
                tid, tile
            );
            __syncthreads();

            // Load V tile from paged cache (shared across all Q-heads in this block)
            {
                constexpr int v_u4 = D_STRIDE_U4;
                const int v_total = kv_valid * v_u4;
                uint32_t v_dst = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
                if (v_total == 0) { __syncthreads(); continue; }
                for (int i = tid; i < v_total; i += TPB) {
                    const int tok = i / v_u4;
                    const int u4c = i % v_u4;
                    int pos = kv_start + tok;
                    if (pos >= seq_len) {
                        uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    int blk = pos / block_size;
                    int off = pos % block_size;
                    if (blk >= max_blocks_per_seq) {
                        uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    int phys = bt[blk];
                    if (phys < 0) {
                        uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                        asm volatile("st.shared.v4.u32 [%0], {0,0,0,0};" :: "r"(d) : "memory");
                        continue;
                    }
                    uint64_t src = v_cache_addr
                                 + static_cast<uint64_t>(phys) * block_size * kv_row_stride
                                 + static_cast<uint64_t>(off) * kv_row_stride
                                 + kv_head_offset
                                 + static_cast<uint64_t>(u4c) * 16;
                    uint32_t d = v_dst + static_cast<uint32_t>(tok * v_u4 + u4c) * 16;
                    uint32_t r0, r1, r2, r3;
                    __asm__ volatile(
                        "{\n  ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                        "  st.shared.v4.u32 [%5], {%0,%1,%2,%3};\n"
                        "}\n"
                        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                        : "l"(src), "r"(d) : "memory");
                }
            }
            __syncthreads();

            // dO += P @ V
            WMMA_GEMM_GRADIENTS<GemmType::dO_PV, D, BLOCK_M, BLOCK_N,
                                N_STRIDE, D_STRIDE, WARPS_PER_BLOCK>(
                sP, sV, sO,
                valid_q, kv_valid,
                warp_id, lane_id
            );
            __syncthreads();
        }

        // Epilogue: store normalized output for this Q-head
        {
            constexpr int global_chunks = D >> 2;
            const int total_iters = valid_q * global_chunks;
            for (int i = tid; i < total_iters; i += TPB) {
                const int r = i / global_chunks;
                const int col = (i % global_chunks) << 2;
                float norm = __frcp_rn(fmaxf(sSum[r], 1e-24f));
                const float4 v = *reinterpret_cast<const float4*>(sO + r * D_STRIDE + col);
                __half2 h0 = __float22half2_rn(make_float2(v.x * norm, v.y * norm));
                __half2 h1 = __float22half2_rn(make_float2(v.z * norm, v.w * norm));
                ushort v0 = __half_as_ushort(h0.x), v1 = __half_as_ushort(h0.y);
                ushort v2 = __half_as_ushort(h1.x), v3 = __half_as_ushort(h1.y);
                uint64_t addr = out_base_addr
                              + static_cast<uint64_t>(global_q_start + r) * q_row_stride
                              + q_head_offset_bytes
                              + static_cast<uint64_t>(col) * 2;
                __asm__ volatile(
                    "st.global.v4.u16 [%0], {%1,%2,%3,%4};"
                    : : "l"(addr), "h"(v0), "h"(v1), "h"(v2), "h"(v3)
                    : "memory");
            }
        }

        // Write LSE for this Q-head
        if (tid < valid_q) {
            const float sum = fmaxf(sSum[tid], 1e-24f);
            int lse_idx = head_id * num_tokens + global_q_start + tid;
            softmax_lse[lse_idx] = sMax[tid] + logf(sum);
        }
        __syncthreads();  // Ensure SMEM is ready for next Q-head
    }
}

// --------------------------------------------------------------------------
// LAUNCHER — supports 3D flat Q with prefix_kv_lens and native GQA
// --------------------------------------------------------------------------
template<int D>
void launcher_flash_attention_paged_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& prefix_kv_lens,
    torch::Tensor& Out,
    torch::Tensor& softmax_lse,
    int num_kv_heads,
    int block_size,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream)
{
    using Config = KernelConfig<D>;
    
    TORCH_CHECK(Q.dim() == 3 && Q.size(2) == D, "Q must be [num_tokens, num_heads, ", D, "]");
    TORCH_CHECK(K_cache.dim() == 4 && K_cache.size(2) == num_kv_heads, "K_cache must be [num_blocks, block_size, num_kv_heads, ", D, "]");
    TORCH_CHECK(V_cache.dim() == 4 && V_cache.size(2) == num_kv_heads, "V_cache must be [num_blocks, block_size, num_kv_heads, ", D, "]");
    
    const int num_tokens = Q.size(0);
    const int num_heads = Q.size(1);
    
    const int num_seqs = seq_lens.size(0);
    const int max_bt = block_table.size(1);
    
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
    TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == num_seqs + 1,
                "query_start_loc must be [num_seqs + 1]");
    TORCH_CHECK(prefix_kv_lens.dim() == 1 && prefix_kv_lens.size(0) == num_seqs,
                "prefix_kv_lens must be [num_seqs]");
    
    // Compute max query tiles using num_tokens as a safe upper bound.
    const int max_q_tiles = (num_tokens + Config::BLOCK_M - 1) / Config::BLOCK_M;
    
    TORCH_CHECK(max_q_tiles > 0 && max_q_tiles <= 2048, "max_q_tiles out of range");
    
    // Grid: (max_q_tiles, num_seqs, num_heads) — 3D grid, native GQA
    const dim3 grid(max_q_tiles, num_seqs, num_heads);
    const dim3 block(Config::THREADS_PER_BLOCK);
    const size_t smem = Config::TOTAL_SMEM;
    
    TORCH_CHECK(smem <= MAX_SMEM_PER_SM,
                "Shared memory exceeds 96KB for Paged Forward: ", smem);

    uint64_t out_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Out.data_ptr()));

    // Phase 3: GQA-shared-KV kernel is implemented but disabled by default.
    // Benchmarks showed it is slower for short sequences and only breaks even
    // at very long sequences, because K/V loading is not the actual bottleneck.
    // The kernel code is kept for reference but not used.
    const bool use_shared_kv = false;

    if (use_shared_kv) {
        // Grid over KV-heads instead of Q-heads; block loops over Q-heads
        const dim3 grid_kv(max_q_tiles, num_seqs, num_kv_heads);
        if (is_causal) {
            auto kernel = flash_attention_paged_forward_kernel_gqa_shared_kv<D, true>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kernel<<<grid_kv, block, smem, stream>>>(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Q.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(K_cache.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(V_cache.data_ptr())),
                out_addr,
                block_table.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                query_start_loc.data_ptr<int>(),
                prefix_kv_lens.data_ptr<int>(),
                softmax_lse.data_ptr<float>(),
                num_seqs, max_bt,
                block_size, num_heads, num_kv_heads, num_tokens, softmax_scale);
        } else {
            auto kernel = flash_attention_paged_forward_kernel_gqa_shared_kv<D, false>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kernel<<<grid_kv, block, smem, stream>>>(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Q.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(K_cache.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(V_cache.data_ptr())),
                out_addr,
                block_table.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                query_start_loc.data_ptr<int>(),
                prefix_kv_lens.data_ptr<int>(),
                softmax_lse.data_ptr<float>(),
                num_seqs, max_bt,
                block_size, num_heads, num_kv_heads, num_tokens, softmax_scale);
        }
    } else {
        // Original path: one block per (q_tile, seq, q_head)
        if (is_causal) {
            auto kernel = flash_attention_paged_forward_kernel<D, true>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kernel<<<grid, block, smem, stream>>>(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Q.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(K_cache.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(V_cache.data_ptr())),
                out_addr,
                block_table.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                query_start_loc.data_ptr<int>(),
                prefix_kv_lens.data_ptr<int>(),
                softmax_lse.data_ptr<float>(),
                num_seqs, max_bt,
                block_size, num_heads, num_kv_heads, num_tokens, softmax_scale);
        } else {
            auto kernel = flash_attention_paged_forward_kernel<D, false>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            kernel<<<grid, block, smem, stream>>>(
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Q.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(K_cache.data_ptr())),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(V_cache.data_ptr())),
                out_addr,
                block_table.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                query_start_loc.data_ptr<int>(),
                prefix_kv_lens.data_ptr<int>(),
                softmax_lse.data_ptr<float>(),
                num_seqs, max_bt,
                block_size, num_heads, num_kv_heads, num_tokens, softmax_scale);
        }
    }
}

// --------------------------------------------------------------------------
// ENTRY POINT — matches vLLM-style interface
// --------------------------------------------------------------------------
std::vector<at::Tensor> flash_attention_paged_forward(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& query_start_loc,
    const at::Tensor& prefix_kv_lens,
    at::Tensor& out,
    const int num_kv_heads,
    const int block_size,
    const float softmax_scale,
    bool is_causal)
{
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k_cache.dtype() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.dtype() == torch::kFloat16, "v_cache must be fp16");
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda());
    TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dim");

    const auto q_sizes = q.sizes();
    const int num_tokens = q_sizes[0];
    const int num_heads = q_sizes[1];
    const int D = q_sizes[2];
    
    TORCH_CHECK(D == 64 || D == 128 || D == 256, "D must be 64, 128, or 256 for paged kernel");
    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    
    const int num_seqs = seq_lens.size(0);
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == num_seqs,
                "block_table must be [num_seqs, max_blocks]");
    TORCH_CHECK(prefix_kv_lens.size(0) == num_seqs,
                "prefix_kv_lens must be [num_seqs]");
    
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    TORCH_CHECK(out.size(0) == num_tokens, "out num_tokens must match q");
    
    auto softmax_lse = torch::empty({num_heads, num_tokens}, q.options().dtype(torch::kFloat32));
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    switch (D) {
        case 64:
            launcher_flash_attention_paged_forward<64>(
                q, k_cache, v_cache, block_table, seq_lens, query_start_loc, 
                prefix_kv_lens, out, softmax_lse, num_kv_heads, block_size, softmax_scale, is_causal, stream);
            break;
        case 128:
            launcher_flash_attention_paged_forward<128>(
                q, k_cache, v_cache, block_table, seq_lens, query_start_loc,
                prefix_kv_lens, out, softmax_lse, num_kv_heads, block_size, softmax_scale, is_causal, stream);
            break;
        case 256:
            launcher_flash_attention_paged_forward<256>(
                q, k_cache, v_cache, block_table, seq_lens, query_start_loc,
                prefix_kv_lens, out, softmax_lse, num_kv_heads, block_size, softmax_scale, is_causal, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported head dim: ", D);
    }
    
    return {out, softmax_lse};
}
