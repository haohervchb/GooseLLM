#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace vllm {
namespace {

__global__ void gemv_decode_kernel(
    half* __restrict__ out,
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int64_t max_blocks_per_seq,
    int64_t block_size,
    int64_t num_pages,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t out_stride_0,
    int64_t out_stride_1,
    int64_t k_stride_0,
    int64_t k_stride_1,
    int64_t k_stride_2,
    int64_t k_stride_3,
    int64_t v_stride_0,
    int64_t v_stride_1,
    int64_t v_stride_2,
    int64_t v_stride_3) {

  int head_idx = blockIdx.x;
  int seq_idx = blockIdx.y;
  int lane = threadIdx.x;

  if (head_idx >= num_heads) return;

  int seq_len = seq_lens[seq_idx];
  half* out_ptr = out + seq_idx * out_stride_0 + head_idx * out_stride_1;

  if (seq_len <= 0) {
    // Zero out for empty sequences
    for (int d = lane; d < 256; d += 32) out_ptr[d] = __float2half(0.0f);
    return;
  }

  int kv_head = head_idx / (num_heads / num_kv_heads);
  const half* q_ptr = query + seq_idx * q_stride_0 + head_idx * q_stride_1;

  // Accumulators — each lane handles 8 output dims (256/32)
  float acc[8];
  for (int i = 0; i < 8; i++) acc[i] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int* bt_row = block_tables + seq_idx * max_blocks_per_seq;
  int num_blocks = (seq_len + block_size - 1) / block_size;
  if (num_blocks > max_blocks_per_seq) num_blocks = max_blocks_per_seq;

  for (int blk = 0; blk < num_blocks; blk++) {
    int phys = bt_row[blk];
    bool valid = (phys >= 0 && phys < num_pages);
    int token_start = blk * block_size;
    int tokens = (block_size < seq_len - token_start) ? block_size : (seq_len - token_start);

    // ── QK^T: warp-shuffle dot products ──
    float scores[16];
    float max_score = -INFINITY;

    for (int t = 0; t < tokens; t++) {
      // Each lane computes partial dot product
      float dot = 0.0f;
      if (valid) {
        int64_t k_base = (int64_t)phys * k_stride_0 + (int64_t)kv_head * k_stride_2;
        const half* k_ptr = key_cache + k_base + t * k_stride_1;
        for (int d = lane; d < 256; d += 32) {
          dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d * k_stride_3]);
        }
      }
      // Warp shuffle sum
#pragma unroll
      for (int mask = 16; mask >= 1; mask /= 2) {
        dot += __shfl_xor_sync(0xffffffff, dot, mask);
      }
      if (lane == 0) {
        scores[t] = dot * scale;
        if (scores[t] > max_score) max_score = scores[t];
      }
    }

    // ── Online softmax (lane 0 only) ──
    float old_m = m;
    float new_m = fmaxf(old_m, max_score);
    float sf = (old_m == -INFINITY) ? 0.0f : __expf(old_m - new_m);
    float tile_sum = 0.0f;
    float probs[16];

    if (lane == 0) {
      for (int t = 0; t < tokens; t++) {
        float p = __expf(scores[t] - new_m);
        probs[t] = p;
        tile_sum += p;
      }
    }

    // Broadcast probs and state to all lanes via shared memory trick: using sf register
    // Lane 0 writes, other lanes read via __shfl_sync
#pragma unroll
    for (int t = 0; t < 16; t++) {
      probs[t] = __shfl_sync(0xffffffff, probs[t], 0);
    }
    tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);
    new_m = __shfl_sync(0xffffffff, new_m, 0);
    sf = __shfl_sync(0xffffffff, sf, 0);

    m = new_m;
    // Rescale accumulator
    for (int i = 0; i < 8; i++) acc[i] *= sf;
    l = l * sf + tile_sum;

    // ── PV: weighted sum ──
    for (int t = 0; t < tokens; t++) {
      if (valid) {
        int64_t v_base = (int64_t)phys * v_stride_0 + (int64_t)kv_head * v_stride_2;
        const half* v_ptr = value_cache + v_base + t * v_stride_1;
        for (int i = 0; i < 8; i++) {
          int d = lane + i * 32;
          if (d < 256) {
            acc[i] += probs[t] * __half2float(v_ptr[d * v_stride_3]);
          }
        }
      }
    }
  }

  // ── Output ──
  if (l > 0.0f) {
    float inv_l = 1.0f / (l + 1e-6f);
    for (int i = 0; i < 8; i++) {
      int d = lane + i * 32;
      if (d < 256) {
        out_ptr[d] = __float2half(acc[i] * inv_l);
      }
    }
  }
}

void launch_gemv(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t num_pages) {

  const int64_t num_seqs = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t max_blocks_per_seq = block_tables.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(32, 1, 1);  // 1 warp

  gemv_decode_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(out.data_ptr()),
      reinterpret_cast<const half*>(query.data_ptr()),
      reinterpret_cast<const half*>(key_cache.data_ptr()),
      reinterpret_cast<const half*>(value_cache.data_ptr()),
      num_heads,
      num_kv_heads,
      static_cast<float>(scale),
      block_tables.data_ptr<int>(),
      seq_lens.data_ptr<int>(),
      max_blocks_per_seq,
      block_size,
      num_pages,
      query.stride(0), query.stride(1),
      out.stride(0), out.stride(1),
      key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), key_cache.stride(3),
      value_cache.stride(0), value_cache.stride(1),
      value_cache.stride(2), value_cache.stride(3));
}

}  // namespace

void gemv_paged_decode_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t num_pages) {
  launch_gemv(out, query, key_cache, value_cache,
              num_kv_heads, scale, block_tables, seq_lens,
              block_size, num_pages);
}

}  // namespace vllm
