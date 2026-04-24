#pragma once

// ============================================================================
// Raw Volta m8n8k4 MMA backend (sm_70)
//
// Replaces WMMA m16n16k16 macro-layer with direct PTX mma.sync.m8n8k4.
// One warp executes 4 concurrent m8n8k4 ops (quadpairs):
//   Group 0: lanes 0-3, 16-19
//   Group 1: lanes 4-7, 20-23
//   Group 2: lanes 8-11, 24-27
//   Group 3: lanes 12-15, 28-31
//
// Each quadpair computes one 8x8 output tile from A(8x4) x B(4x8).
//
// Fragment sizes per thread (PTX ISA 9.2 §9.7.14.5.1):
//   A: 2 x uint32_t (packed f16x2) = 4 f16 elements
//   B: 2 x uint32_t (packed f16x2) = 4 f16 elements
//   C/D: 8 x float = 8 f32 elements
// ============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

namespace volta_m8n8k4 {

// ---------------------------------------------------------------------------
// Fragment types
// ---------------------------------------------------------------------------
struct A_frag { uint32_t x[2]; };
struct B_frag { uint32_t x[2]; };
struct C_frag { float    x[8]; };

// ---------------------------------------------------------------------------
// Inline PTX: mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
//
// D = A * B + C
// A: row-major f16, B: col-major f16, C/D: f32
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_sync_m8n8k4(
    C_frag& d,
    const A_frag& a,
    const B_frag& b,
    const C_frag& c)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 \n\t"
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11}, "
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3]),
          "+f"(d.x[4]), "+f"(d.x[5]), "+f"(d.x[6]), "+f"(d.x[7])
        : "r"(a.x[0]), "r"(a.x[1]),
          "r"(b.x[0]), "r"(b.x[1]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7])
    );
}

// ---------------------------------------------------------------------------
// Load A from shared memory (row-major) to registers
//
// Each thread needs 4 f16 elements from A.
// For row-major A(8xK), thread layout (per quadpair):
//   row = lane_id % 4            [if lane_id < 16]
//         (lane_id % 4) + 4      [otherwise]
//   col = i (i = 0..3 for the 4 elements)
//
// We load 2 f16x2 vectors = 4 f16 per thread.
// The two f16x2 are at columns (k_offset + 0, k_offset + 2).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_a_from_smem(
    A_frag& frag,
    const __half* smem_a,      // base pointer to A tile in SMEM
    int row,                    // thread's row in 8x8 tile (0..7)
    int k_offset,               // starting K index (0, 2, 4, ...)
    int ldm)                    // leading dimension (stride)
{
    // Load two f16x2 values: columns k_offset and k_offset+2
    // Each f16x2 is 4 bytes, so we can use a 32-bit load
    const uint32_t* src0 = reinterpret_cast<const uint32_t*>(
        smem_a + row * ldm + k_offset);
    const uint32_t* src1 = reinterpret_cast<const uint32_t*>(
        smem_a + row * ldm + k_offset + 2);
    frag.x[0] = *src0;
    frag.x[1] = *src1;
}

// ---------------------------------------------------------------------------
// Load B from shared memory (col-major) to registers
//
// For col-major B(Kx8), thread layout (per quadpair):
//   row = i (i = 0..3)
//   col = lane_id % 4           [if lane_id < 16]
//         (lane_id % 4) + 4     [otherwise]
//
// We load 2 f16x2 vectors = 4 f16 per thread.
// The two f16x2 are at rows (k_offset + 0, k_offset + 2).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_b_from_smem(
    B_frag& frag,
    const __half* smem_b,      // base pointer to B tile in SMEM
    int col,                    // thread's col in 8x8 tile (0..7)
    int k_offset,               // starting K index (0, 2, 4, ...)
    int ldm)                    // leading dimension (stride)
{
    // For col-major, B[k, col] = smem_b[k * ldm + col]
    // Load two f16x2 values: rows k_offset and k_offset+2
    const uint32_t* src0 = reinterpret_cast<const uint32_t*>(
        smem_b + k_offset * ldm + col);
    const uint32_t* src1 = reinterpret_cast<const uint32_t*>(
        smem_b + (k_offset + 2) * ldm + col);
    frag.x[0] = *src0;
    frag.x[1] = *src1;
}

// ---------------------------------------------------------------------------
// Store C/D to shared memory (row-major)
//
// Each thread writes 8 f32 elements = 2 rows × 4 cols.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void store_c_to_smem(
    const C_frag& frag,
    float* smem_c,              // base pointer to C tile in SMEM
    int row0,                   // first row this thread writes (0..7)
    int row1,                   // second row this thread writes (0..7)
    int col0,                   // starting col (0 or 4)
    int ldm)                    // leading dimension (stride)
{
    // frag layout (from PTX ISA):
    // row0 gets cols col0, col0+1, col0+2, col0+3
    // row1 gets cols col0, col0+1, col0+2, col0+3
    // Actually the exact mapping depends on lane_id; this is a helper
    // that stores in row-major order based on thread position.
    //
    // For precise store, caller must know exact (row, col) per element.
    // This function is a generic helper; specialized versions may be needed.

    // Store 4 floats for row0
    float* dst0 = smem_c + row0 * ldm + col0;
    dst0[0] = frag.x[0];
    dst0[1] = frag.x[1];
    dst0[2] = frag.x[4];  // Note: layout is interleaved
    dst0[3] = frag.x[5];

    // Store 4 floats for row1
    float* dst1 = smem_c + row1 * ldm + col0;
    dst1[0] = frag.x[2];
    dst1[1] = frag.x[3];
    dst1[2] = frag.x[6];
    dst1[3] = frag.x[7];
}

// ---------------------------------------------------------------------------
// Zero-initialize C fragment
// ---------------------------------------------------------------------------
__device__ __forceinline__ void zero_c_frag(C_frag& frag) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) frag.x[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Thread position helpers for m8n8k4
//
// Returns the thread's role within its quadpair.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int get_lane_id_m8n8k4() {
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ int get_quadpair_id(int lane_id) {
    // Quadpair: groups of 8 threads
    return lane_id / 4;  // 0,1,2,3,4,5,6,7 → but with high/low split
}

__device__ __forceinline__ void get_m8n8k4_thread_rows(
    int lane_id, int& row0, int& row1)
{
    // From PTX ISA §9.7.14.5.1 (f32 ctype):
    // row = (laneid & 1) + (i & 2)  [with +4 offset if laneid >= 16]
    // For i=0,1: row0 = (laneid & 1) + 0  or  + 2
    // Actually let's use the simpler formula:
    // For a given lane, it holds two rows.
    // Low group (0-15): rows are (lane%4) and (lane%4)+2? No...
    //
    // PTX ISA says:
    // row = X  where X = (laneid & 0b1) + (i & 0b10)
    // For i=0: row = (laneid & 1) + 0
    // For i=1: row = (laneid & 1) + 2
    // But wait, that's only 2 rows per thread, and there are 8 rows total...
    //
    // Actually re-reading: for f32 ctype, each thread holds 8 f32 elements.
    // The formula gives row for each ci (i=0..7).
    // row = (laneid & 1) + (i & 2)  [with +4 if laneid >= 16]
    // For i in {0,1,2,3,4,5,6,7}:
    //   i=0,1: row = (laneid&1) + 0
    //   i=2,3: row = (laneid&1) + 2
    //   i=4,5: row = (laneid&1) + 0  ??? That can't be right...
    //
    // Wait, I think I misread. Let me re-examine the probe output.
    // From our WMMA probe, the C layout for m16n16k16 was complex.
    // For m8n8k4, the PTX ISA gives separate formulas.
    //
    // Actually the PTX ISA formula is:
    // row = X  if laneid < 16,  X+4 otherwise
    //   where X = (laneid & 0b1) + (i & 0b10)
    // col = (i & 0b100) + (laneid & 0b10) + (i & 0b1)
    //
    // For i=0..7 in a single m8n8k4 computation:
    // The 8 threads in a quadpair collectively hold 64 f32 values = 8x8 matrix.
    // Each thread holds 8 f32 values.
    //
    // Let's derive for laneid=0 (low group, computation 1):
    // i=0: row=(0&1)+(0&2)=0, col=(0&4)+(0&2)+(0&1)=0 → (0,0)
    // i=1: row=0, col=1 → (0,1)
    // i=2: row=2, col=0 → (2,0)
    // i=3: row=2, col=1 → (2,1)
    // i=4: row=0, col=4 → (0,4)
    // i=5: row=0, col=5 → (0,5)
    // i=6: row=2, col=4 → (2,4)
    // i=7: row=2, col=5 → (2,5)
    //
    // For laneid=1:
    // i=0: row=1, col=0 → (1,0)
    // i=1: row=1, col=1 → (1,1)
    // ...
    //
    // For laneid=2:
    // i=0: row=0, col=2 → (0,2)
    // i=1: row=0, col=3 → (0,3)
    // ...
    //
    // For laneid=16 (high group, computation 1):
    // i=0: row=4, col=0 → (4,0)
    // ...
    //
    // OK so each thread holds:
    //   2 rows × 4 cols (for low group: rows 0,2 or 1,3 or 4,6 or 5,7)
    //   cols are 0,1,4,5 or 2,3,6,7 depending on lane
    //
    // This means:
    // lane 0: rows {0,2}, cols {0,1,4,5}
    // lane 1: rows {1,3}, cols {0,1,4,5}
    // lane 2: rows {0,2}, cols {2,3,6,7}
    // lane 3: rows {1,3}, cols {2,3,6,7}
    // lane 16: rows {4,6}, cols {0,1,4,5}
    // lane 17: rows {5,7}, cols {0,1,4,5}
    // lane 18: rows {4,6}, cols {2,3,6,7}
    // lane 19: rows {5,7}, cols {2,3,6,7}
    //
    // And similarly for lanes 4-7, 20-23 (computation 2) etc.
    //
    // For the actual m8n8k4, the computation 2 would map to rows 0-7, cols 0-7
    // but within a different quadpair group.
    //
    // Actually, I think each quadpair computes an INDEPENDENT 8x8 tile.
    // So quadpair 0 (lanes 0-3,16-19) computes tile at some (tile_m, tile_n).
    // Quadpair 1 (lanes 4-7,20-23) computes a different tile.
    //
    // So for our kernel, we treat each quadpair as computing one 8x8 output.
    // Within a quadpair:
    //   lane 0: elements at (0,0),(0,1),(2,0),(2,1),(0,4),(0,5),(2,4),(2,5)
    //   etc.
    //
    // The important thing for C→A transform:
    // A-fragment (row-major) for the same quadpair:
    //   row = laneid % 4          [if laneid < 16 within the quadpair]
    //   col = i  (i = 0..3)
    //
    // Wait, the quadpair has lanes {0,1,2,3,16,17,18,19}.
    // For A-fragment row-major:
    // lane 0: row 0, cols 0,1,2,3
    // lane 1: row 1, cols 0,1,2,3
    // lane 2: row 2, cols 0,1,2,3
    // lane 3: row 3, cols 0,1,2,3
    // lane 16: row 4, cols 0,1,2,3
    // lane 17: row 5, cols 0,1,2,3
    // lane 18: row 6, cols 0,1,2,3
    // lane 19: row 7, cols 0,1,2,3
    //
    // So within a quadpair, each thread holds exactly one COMPLETE row of A!
    // That's why the C→A transform only needs shuffles WITHIN the quadpair.

    // For now, this function is a placeholder. We'll implement exact mapping
    // after empirical verification.
    row0 = (lane_id & 1) + ((lane_id < 16) ? 0 : 4);
    row1 = row0 + 2;
}

} // namespace volta_m8n8k4
