#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace vllm {
namespace {

template <int CTA_H>
__global__ void gemv_decode_kernel(
    half* __restrict__ out, const half* __restrict__ query,
    const half* __restrict__ key_cache, const half* __restrict__ value_cache,
    int64_t num_heads, int64_t num_kv_heads, float scale,
    const int* __restrict__ block_tables, const int* __restrict__ seq_lens,
    int64_t max_blocks_per_seq, int64_t page_block_size, int64_t num_pages,
    int64_t q_stride_0, int64_t q_stride_1, int64_t out_stride_0,
    int64_t out_stride_1, int64_t k_stride_0, int64_t k_stride_1,
    int64_t k_stride_2, int64_t k_stride_3, int64_t v_stride_0,
    int64_t v_stride_1, int64_t v_stride_2, int64_t v_stride_3) {
  static_assert(CTA_H >= 1 && CTA_H <= 4);
  constexpr int TILE_SIZE = 16;

  int cta_idx = blockIdx.x, seq_idx = blockIdx.y;
  int warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
  int token_grp = lane >> 1, sub_lane = lane & 1;
  int q_per_kv = num_heads / num_kv_heads;
  int q_groups  = (q_per_kv + CTA_H - 1) / CTA_H;
  int kv_head   = cta_idx / q_groups;
  int q_group   = cta_idx % q_groups;
  int head_idx  = kv_head * q_per_kv + q_group * CTA_H + warp_id;
  bool valid    = (warp_id < CTA_H) && (head_idx < num_heads);
  if (!valid) return;

  int seq_len = seq_lens[seq_idx];
  half* out_ptr = out + seq_idx * out_stride_0 + head_idx * out_stride_1;
  if (seq_len <= 0) {
    for (int d = lane; d < 256; d += 32) out_ptr[d] = __float2half(0.0f);
    return;
  }

  // Sub-tile factor: how many 16-token tiles per physical 528-token page
  int sub_tiles_per_page = page_block_size / TILE_SIZE; // 528/16 = 33
  const half* q_ptr = query + seq_idx * q_stride_0 + head_idx * q_stride_1;
  float acc[8]; for (int i = 0; i < 8; i++) acc[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  const int* bt_row = block_tables + seq_idx * max_blocks_per_seq;
  int total_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < total_tiles; tile++) {
    // Map logical 16-token tile → physical 528-token page + intra-page offset
    int page_idx  = tile / sub_tiles_per_page;
    int intra_off = (tile % sub_tiles_per_page) * TILE_SIZE;
    int phys      = bt_row[page_idx];
    bool page_ok  = (phys >= 0 && phys < num_pages);
    int tok       = TILE_SIZE;
    int rem       = seq_len - tile * TILE_SIZE;
    if (tok > rem) tok = rem;

    // QK^T: parallel across 16 tokens, 2 lanes per token
    float dot = 0.0f;
    if (token_grp < tok && page_ok) {
      int64_t kb = (int64_t)phys * k_stride_0 + (int64_t)kv_head * k_stride_2;
      const half* kp = key_cache + kb + (intra_off + token_grp) * k_stride_1;
      for (int d = sub_lane; d < 256; d += 2)
        dot += __half2float(q_ptr[d]) * __half2float(kp[d * k_stride_3]);
    }
    dot += __shfl_xor_sync(0xffffffff, dot, 1);
    float score = (token_grp < tok) ? dot * scale : -INFINITY;

    float all_scores[16];
#pragma unroll
    for (int t = 0; t < 16; t++)
      all_scores[t] = __shfl_sync(0xffffffff, score, t * 2);

    float sf = 0.0f, probs[16];
    if (lane == 0) {
      float mc = all_scores[0];
#pragma unroll
      for (int t = 1; t < tok; t++) if (all_scores[t] > mc) mc = all_scores[t];
      if (mc == -INFINITY) mc = 0.0f;
      float om = m, nm = fmaxf(om, mc);
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
    for (int t = 0; t < 16; t++)
      probs[t] = __shfl_sync(0xffffffff, probs[t], 0);

    if (lane != 0)
      for (int i = 0; i < 8; i++) acc[i] *= sf;

    // PV
    if (page_ok) {
      for (int t = 0; t < tok; t++) {
        int64_t vb = (int64_t)phys * v_stride_0 + (int64_t)kv_head * v_stride_2;
        const half* vp = value_cache + vb + (intra_off + t) * v_stride_1;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int d = lane + i * 32;
          if (d < 256) acc[i] += probs[t] * __half2float(vp[d * v_stride_3]);
        }
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

void launch_gemv(torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t nkv, double scale, torch::Tensor& bt,
    torch::Tensor& sl, int64_t bs, int64_t np) {
  const int64_t ns = query.size(0), nh = query.size(1);
  int64_t qpk = nh / nkv;
  const int64_t mb = bt.size(1);
  const at::cuda::OptionalCUDAGuard g(device_of(query));

  dim3 grid(nkv * ((qpk + 3) / 4), ns, 1);
  dim3 block(128, 1, 1);

  gemv_decode_kernel<4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
