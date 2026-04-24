#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fused_mma.h"
#include "00_volta_const.cuh"

// ============================================================================
// SparkAttention-style register-level pipeline for Volta (SM70)
//
// Phase 2A/2B: Switch WMMA backend to fused_mma.h, keep accumulator in
// registers after Q@K^T, compute softmax in registers, write P directly
// to SMEM (still needs SMEM for P@V A-operand until register shuffle).
//
// Tile assignment is restructured by row group so each warp only touches
// rows within its group, enabling warp-local max/sum reduction.
// ============================================================================

namespace spark {

using namespace volta;

// ============================================================================
// Row-group tile assignment for the Spark path.
// Each warp gets tiles from ONE row group (16 rows with m16n16k16).
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int WARPS_PER_BLOCK>
struct SparkTileAssignment {
    static constexpr int WMMA_TILE_M = 16;
    static constexpr int num_row_groups = (BLOCK_M + WMMA_TILE_M - 1) / WMMA_TILE_M;
    static constexpr int warps_per_group = WARPS_PER_BLOCK / num_row_groups;
    static constexpr int num_tiles_n = (BLOCK_N + WMMA_TILE_M - 1) / WMMA_TILE_M;
    static constexpr int tiles_per_warp = (num_tiles_n + warps_per_group - 1) / warps_per_group;
    static constexpr int total_tiles = num_row_groups * num_tiles_n;

    __device__ __forceinline__ static void get_tile(int warp_id, int tile_local,
                                                      int& tile_m, int& tile_n,
                                                      int& group_id, int& warp_in_group) {
        group_id = warp_id / warps_per_group;
        warp_in_group = warp_id % warps_per_group;
        tile_m = group_id * WMMA_TILE_M;
        const int tile_n_idx = warp_in_group * tiles_per_warp + tile_local;
        tile_n = tile_n_idx * WMMA_TILE_M;
    }
};

// Generic row-group tile assignment with explicit N dimension (for P@V)
template<int BLOCK_M, int N_DIM, int WARPS_PER_BLOCK>
struct SparkTileAssignmentN {
    static constexpr int WMMA_TILE_M = 16;
    static constexpr int num_row_groups = (BLOCK_M + WMMA_TILE_M - 1) / WMMA_TILE_M;
    static constexpr int warps_per_group = WARPS_PER_BLOCK / num_row_groups;
    static constexpr int num_tiles_n = (N_DIM + WMMA_TILE_M - 1) / WMMA_TILE_M;
    static constexpr int tiles_per_warp = (num_tiles_n + warps_per_group - 1) / warps_per_group;
    static constexpr int total_tiles = num_row_groups * num_tiles_n;

    __device__ __forceinline__ static void get_tile(int warp_id, int tile_local,
                                                      int& tile_m, int& tile_n,
                                                      int& group_id, int& warp_in_group) {
        group_id = warp_id / warps_per_group;
        warp_in_group = warp_id % warps_per_group;
        tile_m = group_id * WMMA_TILE_M;
        const int tile_n_idx = warp_in_group * tiles_per_warp + tile_local;
        tile_n = tile_n_idx * WMMA_TILE_M;
    }
};

// ============================================================================
// Utility: Get (row, col) for element i of m16n16k16 accumulator fragment
// ============================================================================
__device__ __forceinline__ void get_acc_row_col(int lane_id, int i, int& row, int& col) {
    row = (lane_id & 0b1) + ((lane_id >> 2) & 0b1) * 8 + ((lane_id >> 4) & 0b1) * 4 + ((i >> 1) & 0b1) * 2;
    col = ((lane_id >> 1) & 0b1) * 2 + ((lane_id >> 3) & 0b1) * 8 + (i & 0b1) + ((i >> 2) & 0b1) * 4;
}

// ============================================================================
// SPARK_GEMM_QKT: Q@K^T with register accumulator output
// Uses row-group tile assignment.
// ============================================================================
template<bool IS_CAUSAL, int BLOCK_M, int BLOCK_N, int D, int D_STRIDE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void SPARK_GEMM_QKT(
    const __half* __restrict__ sQ,
    const __half* __restrict__ sK,
    fragment<accumulator, 16, 16, 16, float>* acc_frags,
    int VALID_M, int VALID_N,
    int GLOBAL_M, int GLOBAL_N,
    float SOFTMAX_SCALE,
    int WARP_ID, int LANE_ID
) {
    using TA = SparkTileAssignment<BLOCK_M, BLOCK_N, WARPS_PER_BLOCK>;
    constexpr int num_tiles_k = (D + 16 - 1) / 16;

    #pragma unroll
    for (int tile_local = 0; tile_local < TA::tiles_per_warp; ++tile_local) {
        int tile_m, tile_n, group_id, warp_in_group;
        TA::get_tile(WARP_ID, tile_local, tile_m, tile_n, group_id, warp_in_group);

        if (tile_n >= BLOCK_N || tile_n >= VALID_N) continue;
        if (tile_m >= VALID_M) continue;

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * 16;
            if (k_offset >= D) break;
            load_matrix_sync(a_frag, sQ + tile_m * D_STRIDE + k_offset, D_STRIDE);
            load_matrix_sync(b_frag, sK + tile_n * D_STRIDE + k_offset, D_STRIDE);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Causal mask + scale in registers
        if constexpr (IS_CAUSAL) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int row, col;
                get_acc_row_col(LANE_ID, i, row, col);
                int g_m = GLOBAL_M + tile_m + row;
                int g_n = GLOBAL_N + tile_n + col;
                bool in_bounds = (g_m < GLOBAL_M + VALID_M) && (g_n < GLOBAL_N + VALID_N);
                c_frag.x[i] = in_bounds
                    ? ((g_n > g_m) ? NEG_INF : c_frag.x[i] * SOFTMAX_SCALE)
                    : NEG_INF;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                c_frag.x[i] *= SOFTMAX_SCALE;
            }
        }

        acc_frags[tile_local] = c_frag;
    }
}

// ============================================================================
// SPARK_WARP_REDUCE_MAX: Reduce a float across a warp using shfl_xor
// ============================================================================
__device__ __forceinline__ float SPARK_WARP_REDUCE_MAX(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, offset));
    }
    return v;
}

// ============================================================================
// SPARK_WARP_REDUCE_SUM: Reduce a float across a warp using shfl_xor
// ============================================================================
__device__ __forceinline__ float SPARK_WARP_REDUCE_SUM(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
    }
    return v;
}

// ============================================================================
// SPARK_COMPUTE_TILE_STATS: For one accumulator tile, compute warp-reduced
// row max and row sum.  Writes to sPartialMax/sPartialSum SMEM buffers.
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int WARPS_PER_BLOCK>
__device__ __forceinline__ void SPARK_COMPUTE_TILE_STATS(
    const fragment<accumulator, 16, 16, 16, float>& acc_frag,
    int tile_m, int tile_n,
    int VALID_M, int VALID_N,
    int GLOBAL_M, int GLOBAL_N,
    int LANE_ID, int WARP_ID,
    float* sPartialMax,   // [WARPS_PER_BLOCK][16]
    float* sPartialSum    // [WARPS_PER_BLOCK][16]
) {
    float local_max[16];
    float local_sum[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) { local_max[r] = NEG_INF; local_sum[r] = 0.0f; }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row, col;
        get_acc_row_col(LANE_ID, i, row, col);
        int g_m = GLOBAL_M + tile_m + row;
        int g_n = GLOBAL_N + tile_n + col;
        if (g_m < GLOBAL_M + VALID_M && g_n < GLOBAL_N + VALID_N) {
            local_max[row] = fmaxf(local_max[row], acc_frag.x[i]);
        }
    }

    // Warp reduce and accumulate across tiles for this warp (lane 0 writes)
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        local_max[r] = SPARK_WARP_REDUCE_MAX(local_max[r]);
        if (LANE_ID == 0) {
            int idx = WARP_ID * 16 + r;
            sPartialMax[idx] = fmaxf(sPartialMax[idx], local_max[r]);
        }
    }
}

// ============================================================================
// SPARK_WRITE_P_AND_SUM: After global max is known, compute P = exp(S-max),
// accumulate sum, and write P to SMEM.  Also returns warp-reduced sum.
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int N_STRIDE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void SPARK_WRITE_P_AND_SUM(
    const fragment<accumulator, 16, 16, 16, float>& acc_frag,
    int tile_m, int tile_n,
    int VALID_M, int VALID_N,
    int GLOBAL_M, int GLOBAL_N,
    int LANE_ID, int WARP_ID,
    const float* row_max,  // [BLOCK_M] — current (new) global max per row
    __half* sP,            // output P matrix
    float* sPartialSum     // [WARPS_PER_BLOCK][16] — accumulated warp-local sums
) {
    float local_sum[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) local_sum[r] = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row, col;
        get_acc_row_col(LANE_ID, i, row, col);
        int g_m = GLOBAL_M + tile_m + row;
        int g_n = GLOBAL_N + tile_n + col;
        int smem_idx = (tile_m + row) * N_STRIDE + (tile_n + col);
        if (g_m < GLOBAL_M + VALID_M && g_n < GLOBAL_N + VALID_N) {
            float gmax = row_max[g_m - GLOBAL_M];
            float s = acc_frag.x[i];
            float p = (s == NEG_INF) ? 0.0f : __expf(fmaxf(s - gmax, -80.0f));
            local_sum[row] += p;
            sP[smem_idx] = __float2half_rn(p);
        } else {
            local_sum[row] += 0.0f;
            sP[smem_idx] = __float2half_rn(0.0f);
        }
    }

    // Warp reduce and accumulate across tiles for this warp (lane 0 writes)
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        float sum = SPARK_WARP_REDUCE_SUM(local_sum[r]);
        if (LANE_ID == 0) {
            int idx = WARP_ID * 16 + r;
            sPartialSum[idx] += sum;
        }
    }
}

// ============================================================================
// SPARK_GEMM_PV: P@V using fused_mma.h (same structure as 02_wmma.cuh)
// Uses row-group tile assignment.
// ============================================================================
template<int D, int BLOCK_M, int BLOCK_N, int D_STRIDE, int N_STRIDE, int WARPS_PER_BLOCK>
__device__ __forceinline__ void SPARK_GEMM_PV(
    const __half* __restrict__ sP,
    const __half* __restrict__ sV,
    float* __restrict__ sO,
    int VALID_M, int VALID_K,
    int WARP_ID, int LANE_ID
) {
    using TA = SparkTileAssignmentN<BLOCK_M, D, WARPS_PER_BLOCK>;
    constexpr int num_tiles_k = (BLOCK_N + 16 - 1) / 16;

    #pragma unroll
    for (int tile_local = 0; tile_local < TA::tiles_per_warp; ++tile_local) {
        int tile_m, tile_n, group_id, warp_in_group;
        TA::get_tile(WARP_ID, tile_local, tile_m, tile_n, group_id, warp_in_group);

        if (tile_n >= D || tile_m >= VALID_M) continue;

        fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;

        load_matrix_sync(c_frag, sO + tile_m * D_STRIDE + tile_n, D_STRIDE, mem_row_major);

        #pragma unroll
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            const int k_offset = k_tile * 16;
            if (k_offset >= VALID_K) break;
            load_matrix_sync(a_frag, sP + tile_m * N_STRIDE + k_offset, N_STRIDE);
            load_matrix_sync(b_frag, sV + k_offset * D_STRIDE + tile_n, D_STRIDE);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        store_matrix_sync(sO + tile_m * D_STRIDE + tile_n, c_frag, D_STRIDE, mem_row_major);
    }
}

} // namespace spark
