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
    int64_t q_stride_0, int64_t q_stride_1,
    int64_t out_stride_0, int64_t out_stride_1,
    int64_t k_stride_0, int64_t k_stride_1,
    int64_t k_stride_2, int64_t k_stride_3,
    int64_t v_stride_0, int64_t v_stride_1,
    int64_t v_stride_2, int64_t v_stride_3) {

  int head_idx = blockIdx.x;
  int seq_idx = blockIdx.y;
  int lane = threadIdx.x;
  int token_group = lane >> 1;
  int sub_lane = lane & 1;

  if (head_idx >= num_heads) return;

  int seq_len = seq_lens[seq_idx];
  half* out_ptr = out + seq_idx * out_stride_0 + head_idx * out_stride_1;

  if (seq_len <= 0) {
    for (int d = lane; d < 256; d += 32) out_ptr[d] = __float2half(0.0f);
    return;
  }

  int kv_head = head_idx / (num_heads / num_kv_heads);
  const half* q_ptr = query + seq_idx * q_stride_0 + head_idx * q_stride_1;

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  const int* bt_row = block_tables + seq_idx * max_blocks_per_seq;
  int num_blocks = (seq_len + block_size - 1) / block_size;
  if (num_blocks > max_blocks_per_seq) num_blocks = max_blocks_per_seq;

  for (int blk = 0; blk < num_blocks; blk++) {
    int phys = bt_row[blk];
    bool valid = (phys >= 0 && phys < num_pages);
    int tokens = (block_size < seq_len - blk * block_size) ? block_size : (seq_len - blk * block_size);

    // ═══ PHASE 1: QK^T — parallel across 16 tokens, 2 lanes per token ═══
    float dot = 0.0f;
    if (token_group < tokens && valid) {
      int64_t k_base = (int64_t)phys * k_stride_0 + (int64_t)kv_head * k_stride_2;
      const half* k_ptr = key_cache + k_base + token_group * k_stride_1;
      for (int d = sub_lane; d < 256; d += 2) {
        dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d * k_stride_3]);
      }
    }
    dot += __shfl_xor_sync(0xffffffff, dot, 1);
    float score = (token_group < tokens) ? dot * scale : -INFINITY;

    // ═══ PHASE 2: Lane 0 gathers all 16 scores ═══
    float all_scores[16];
#pragma unroll
    for (int t = 0; t < 16; t++) {
      all_scores[t] = __shfl_sync(0xffffffff, score, t * 2);
    }

    // ═══ PHASE 3: Lane 0 computes softmax + broadcasts ═══
    float sf = 0.0f;
    float tile_sum = 0.0f;
    float probs[16];

    if (lane == 0) {
      float m_cur = all_scores[0];
#pragma unroll
      for (int t = 1; t < tokens; t++) {
        if (all_scores[t] > m_cur) m_cur = all_scores[t];
      }
      if (m_cur == -INFINITY) m_cur = 0.0f;

      float old_m = m;
      float new_m = fmaxf(old_m, m_cur);
      sf = (old_m == -INFINITY) ? 0.0f : __expf(old_m - new_m);

      for (int i = 0; i < 8; i++) acc[i] *= sf;

      float ts = 0.0f;
#pragma unroll
      for (int t = 0; t < tokens; t++) {
        float p = __expf(all_scores[t] - new_m);
        probs[t] = p;
        ts += p;
      }
      tile_sum = ts;
      l = l * sf + ts;
      m = new_m;
    }

    // Broadcast state from lane 0 to all lanes
    sf    = __shfl_sync(0xffffffff, sf, 0);
    m     = __shfl_sync(0xffffffff, m, 0);
    l     = __shfl_sync(0xffffffff, l, 0);
#pragma unroll
    for (int t = 0; t < 16; t++) {
      probs[t] = __shfl_sync(0xffffffff, probs[t], 0);
    }

    // ═══ PHASE 4: All lanes rescale acc ═══
    if (lane != 0) {
      for (int i = 0; i < 8; i++) acc[i] *= sf;
    }

    // ═══ PHASE 5: PV — accumulate weighted V ═══
    if (valid) {
      for (int t = 0; t < tokens; t++) {
        int64_t v_base = (int64_t)phys * v_stride_0 + (int64_t)kv_head * v_stride_2;
        const half* v_ptr = value_cache + v_base + t * v_stride_1;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int d = lane + i * 32;
          if (d < 256) {
            acc[i] += probs[t] * __half2float(v_ptr[d * v_stride_3]);
          }
        }
      }
    }
  }

  // ═══ OUTPUT ═══
  if (l > 0.0f) {
    float inv_l = 1.0f / (l + 1e-6f);
#pragma unroll
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
  dim3 block(32, 1, 1);

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
