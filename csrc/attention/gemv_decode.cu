#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace vllm {
namespace {

__global__ void gemv_decode_kernel(
    half* __restrict__ out, const half* __restrict__ query,
    const half* __restrict__ key_cache, const half* __restrict__ value_cache,
    int64_t num_heads, int64_t num_kv_heads, float scale,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int64_t max_blocks_per_seq, int64_t block_size, int64_t num_pages,
    int64_t q_stride_0, int64_t q_stride_1, int64_t out_stride_0,
    int64_t out_stride_1, int64_t k_stride_0, int64_t k_stride_1,
    int64_t k_stride_2, int64_t k_stride_3, int64_t v_stride_0,
    int64_t v_stride_1, int64_t v_stride_2, int64_t v_stride_3) {

  int group_idx = blockIdx.x; int seq_idx = blockIdx.y;
  int lane = threadIdx.x;
  int token_group = lane >> 1; int sub_lane = lane & 1;
  int seq_len = seq_lens[seq_idx];
  int heads_per_kv = num_heads / num_kv_heads;

  for (int h = 0; h < 4; h++) {
    int head_idx = group_idx * 4 + h;
    if (head_idx >= num_heads) continue;

    half* out_ptr = out + seq_idx * out_stride_0 + head_idx * out_stride_1;
    if (seq_len <= 0) {
      for (int d = lane; d < 256; d += 32) out_ptr[d] = __float2half(0.0f);
      continue;
    }

    int kv_head = head_idx / heads_per_kv;
    const half* q_ptr = query + seq_idx * q_stride_0 + head_idx * q_stride_1;
    float acc[8]; for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY; float l = 0.0f;
    const int* bt_row = block_tables + seq_idx * max_blocks_per_seq;
    int nb = (seq_len + block_size - 1) / block_size;
    if (nb > max_blocks_per_seq) nb = max_blocks_per_seq;

    for (int blk = 0; blk < nb; blk++) {
      int phys = bt_row[blk]; bool valid = (phys >= 0 && phys < num_pages);
      int tok = block_size; int rem = seq_len - blk * block_size;
      if (tok > rem) tok = rem;

      float dot = 0.0f;
      if (token_group < tok && valid) {
        int64_t kb = (int64_t)phys * k_stride_0 + (int64_t)kv_head * k_stride_2;
        const half* kp = key_cache + kb + token_group * k_stride_1;
        for (int d = sub_lane; d < 256; d += 2)
          dot += __half2float(q_ptr[d]) * __half2float(kp[d * k_stride_3]);
      }
      dot += __shfl_xor_sync(0xffffffff, dot, 1);
      float sc = (token_group < tok) ? dot * scale : -INFINITY;

      float all_scores[16];
#pragma unroll
      for (int t = 0; t < 16; t++) all_scores[t] = __shfl_sync(0xffffffff, sc, t * 2);

      float sf = 0.0f; float probs[16];
      if (lane == 0) {
        float mc = all_scores[0];
#pragma unroll
        for (int t = 1; t < tok; t++) if (all_scores[t] > mc) mc = all_scores[t];
        if (mc == -INFINITY) mc = 0.0f;
        float om = m; float nm = fmaxf(om, mc);
        sf = (om == -INFINITY) ? 0.0f : __expf(om - nm);
        for (int i = 0; i < 8; i++) acc[i] *= sf;
        float ts = 0.0f;
#pragma unroll
        for (int t = 0; t < tok; t++) {
          float p = __expf(all_scores[t] - nm); probs[t] = p; ts += p;
        }
        l = l * sf + ts; m = nm;
      }

      sf = __shfl_sync(0xffffffff, sf, 0);
      m  = __shfl_sync(0xffffffff, m,  0);
      l  = __shfl_sync(0xffffffff, l,  0);
#pragma unroll
      for (int t = 0; t < 16; t++) probs[t] = __shfl_sync(0xffffffff, probs[t], 0);

      if (lane != 0)
        for (int i = 0; i < 8; i++) acc[i] *= sf;

      if (valid)
        for (int t = 0; t < tok; t++) {
          int64_t vb = (int64_t)phys * v_stride_0 + (int64_t)kv_head * v_stride_2;
          const half* vp = value_cache + vb + t * v_stride_1;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            int d = lane + i * 32;
            if (d < 256) acc[i] += probs[t] * __half2float(vp[d * v_stride_3]);
          }
        }
    }

    if (l > 0.0f) {
      float inv_l = 1.0f / (l + 1e-6f);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int d = lane + i * 32;
        if (d < 256) out_ptr[d] = __float2half(acc[i] * inv_l);
      }
    }
  }
}

void launch_gemv(torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t nkv, double scale, torch::Tensor& bt,
    torch::Tensor& sl, int64_t bs, int64_t np) {
  const int64_t ns = query.size(0); const int64_t nh = query.size(1);
  const int64_t mb = bt.size(1);
  const at::cuda::OptionalCUDAGuard g(device_of(query));
  dim3 grid((nh + 3) / 4, ns, 1); dim3 block(32, 1, 1);
  gemv_decode_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr()), reinterpret_cast<const half*>(query.data_ptr()),
      reinterpret_cast<const half*>(key_cache.data_ptr()), reinterpret_cast<const half*>(value_cache.data_ptr()),
      nh, nkv, (float)scale, bt.data_ptr<int>(), sl.data_ptr<int>(), mb, bs, np,
      query.stride(0), query.stride(1), out.stride(0), out.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3));
}

}  // namespace

void gemv_paged_decode_attention(torch::Tensor& out, torch::Tensor& query,
    torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t nkv, double scale,
    torch::Tensor& bt, torch::Tensor& sl, int64_t bs, int64_t np) {
  launch_gemv(out, query, key_cache, value_cache, nkv, scale, bt, sl, bs, np);
}

}  // namespace vllm
