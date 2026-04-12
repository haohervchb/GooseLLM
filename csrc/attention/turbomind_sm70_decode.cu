#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_fp16.h>

namespace vllm {

namespace {

template <typename T>
__device__ __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <int HEAD_SIZE, int BLOCK_SIZE, int CTA_H>
__global__ void sm70_paged_decode_kernel_fp16(
    half* __restrict__ out,
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t q_groups_per_kv,
    float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int64_t max_num_blocks_per_seq,
    int64_t block_size,
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
  static_assert(BLOCK_SIZE == 16 || BLOCK_SIZE == 32,
                "Supported block sizes are 16 and 32");
  static_assert(CTA_H >= 1 && CTA_H <= 4, "CTA_H must be in [1, 4]");

  constexpr int WARP = 32;
  constexpr int NUM_THREADS = CTA_H * WARP;
  constexpr int ROWS_PER_THREAD = (HEAD_SIZE + WARP - 1) / WARP;
  constexpr int THREAD_GROUP_SIZE = BLOCK_SIZE >= WARP ? 1 : (WARP / BLOCK_SIZE);

  const int seq_idx = blockIdx.y;
  const int cta_idx = blockIdx.x;
  const int warp_idx = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;

  const int seq_len = seq_lens[seq_idx];
  if (seq_len <= 0) {
    return;
  }

  const int kv_head_idx = cta_idx / q_groups_per_kv;
  const int q_group_idx = cta_idx % q_groups_per_kv;
  const int q_head_idx = kv_head_idx * num_queries_per_kv + q_group_idx * CTA_H + warp_idx;
  const bool q_head_valid = (warp_idx < CTA_H) && (q_head_idx < num_heads);

  extern __shared__ char smem_raw[];
  half* q_tile = reinterpret_cast<half*>(smem_raw);
  half* k_tile = q_tile + CTA_H * HEAD_SIZE;
  half* v_tile = k_tile + BLOCK_SIZE * HEAD_SIZE;
  float* probs = reinterpret_cast<float*>(v_tile + BLOCK_SIZE * HEAD_SIZE);
  float* alpha_tile = probs + CTA_H * BLOCK_SIZE;
  float* m_tile = alpha_tile + CTA_H;
  float* l_tile = m_tile + CTA_H;

  if (q_head_valid) {
    const half* q_ptr = query + seq_idx * q_stride_0 + q_head_idx * q_stride_1;
    for (int d = lane; d < HEAD_SIZE; d += WARP) {
      q_tile[warp_idx * HEAD_SIZE + d] = q_ptr[d];
    }
  }
  __syncthreads();

  float m = -INFINITY;
  float l = 0.0f;
  float acc[ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    acc[i] = 0.0f;
  }

  const int num_seq_blocks = static_cast<int>(ceil_div(seq_len, static_cast<int>(block_size)));
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int logical_block_idx = 0; logical_block_idx < num_seq_blocks; ++logical_block_idx) {
    const int physical_block_idx = block_table[logical_block_idx];
    const int token_start = logical_block_idx * static_cast<int>(block_size);
    const int tokens_in_block = min(static_cast<int>(block_size), seq_len - token_start);
    const int tile_elems = BLOCK_SIZE * HEAD_SIZE;

    for (int idx = threadIdx.x; idx < tile_elems; idx += NUM_THREADS) {
      const int token_in_block = idx / HEAD_SIZE;
      const int dim = idx % HEAD_SIZE;
      if (token_in_block < tokens_in_block) {
        k_tile[idx] = *(key_cache + physical_block_idx * k_stride_0
                        + token_in_block * k_stride_1 + kv_head_idx * k_stride_2
                        + dim * k_stride_3);
        v_tile[idx] = *(value_cache + physical_block_idx * v_stride_0
                        + token_in_block * v_stride_1 + kv_head_idx * v_stride_2
                        + dim * v_stride_3);
      } else {
        k_tile[idx] = __float2half(0.0f);
        v_tile[idx] = __float2half(0.0f);
      }
    }
    __syncthreads();

    if (q_head_valid) {
      const int token_lane = lane / THREAD_GROUP_SIZE;
      const int group_offset = lane % THREAD_GROUP_SIZE;

      float score = -INFINITY;
      if (token_lane < tokens_in_block) {
        const half* q_ptr = q_tile + warp_idx * HEAD_SIZE;
        const half* k_ptr = k_tile + token_lane * HEAD_SIZE;
        float partial = 0.0f;
        for (int d = group_offset; d < HEAD_SIZE; d += THREAD_GROUP_SIZE) {
          partial += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
#pragma unroll
        for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
          partial += __shfl_xor_sync(0xffffffff, partial, mask);
        }
        if (group_offset == 0) {
          score = partial * scale;
          probs[warp_idx * BLOCK_SIZE + token_lane] = score;
        }
      }

      __syncwarp();

      if (lane == 0) {
        float new_m = m;
        for (int t = 0; t < tokens_in_block; ++t) {
          new_m = fmaxf(new_m, probs[warp_idx * BLOCK_SIZE + t]);
        }

        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - new_m);
        float tile_sum = 0.0f;
        for (int t = 0; t < tokens_in_block; ++t) {
          float p = expf(probs[warp_idx * BLOCK_SIZE + t] - new_m);
          probs[warp_idx * BLOCK_SIZE + t] = p;
          tile_sum += p;
        }

        alpha_tile[warp_idx] = alpha;
        m_tile[warp_idx] = new_m;
        l_tile[warp_idx] = l * alpha + tile_sum;
      }

      __syncwarp();

      const float alpha = alpha_tile[warp_idx];
      m = m_tile[warp_idx];
      l = l_tile[warp_idx];

#pragma unroll
      for (int row = 0; row < ROWS_PER_THREAD; ++row) {
        const int dim = lane + row * WARP;
        if (dim < HEAD_SIZE) {
          float value = acc[row] * alpha;
#pragma unroll
          for (int t = 0; t < BLOCK_SIZE; ++t) {
            if (t < tokens_in_block) {
              value += probs[warp_idx * BLOCK_SIZE + t]
                       * __half2float(v_tile[t * HEAD_SIZE + dim]);
            }
          }
          acc[row] = value;
        }
      }
    }

    __syncthreads();
  }

  if (q_head_valid) {
    const float inv_l = 1.0f / (l + 1e-6f);
    half* out_ptr = out + seq_idx * out_stride_0 + q_head_idx * out_stride_1;
#pragma unroll
    for (int row = 0; row < ROWS_PER_THREAD; ++row) {
      const int dim = lane + row * WARP;
      if (dim < HEAD_SIZE) {
        out_ptr[dim] = __float2half(acc[row] * inv_l);
      }
    }
  }
}

template <int HEAD_SIZE, int BLOCK_SIZE, int CTA_H>
void launch_sm70_paged_decode_fp16_impl(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size) {
  const int64_t num_seqs = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t max_num_blocks_per_seq = block_tables.size(1);
  const int64_t num_queries_per_kv = num_heads / num_kv_heads;
  const int64_t q_groups_per_kv = ceil_div(num_queries_per_kv, static_cast<int64_t>(CTA_H));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_kv_heads * q_groups_per_kv, num_seqs, 1);
  dim3 block(CTA_H * 32);

  const size_t smem_size =
      sizeof(half) * (CTA_H * HEAD_SIZE + 2 * BLOCK_SIZE * HEAD_SIZE)
      + sizeof(float) * (CTA_H * BLOCK_SIZE + 3 * CTA_H);

  sm70_paged_decode_kernel_fp16<HEAD_SIZE, BLOCK_SIZE, CTA_H>
      <<<grid, block, smem_size, stream>>>(
          reinterpret_cast<half*>(out.data_ptr()),
          reinterpret_cast<const half*>(query.data_ptr()),
          reinterpret_cast<const half*>(key_cache.data_ptr()),
          reinterpret_cast<const half*>(value_cache.data_ptr()),
          num_heads,
          num_kv_heads,
          num_queries_per_kv,
          q_groups_per_kv,
          static_cast<float>(scale),
          block_tables.data_ptr<int>(),
          seq_lens.data_ptr<int>(),
          max_num_blocks_per_seq,
          block_size,
          query.stride(0),
          query.stride(1),
          out.stride(0),
          out.stride(1),
          key_cache.stride(0),
          key_cache.stride(1),
          key_cache.stride(2),
          key_cache.stride(3),
          value_cache.stride(0),
          value_cache.stride(1),
          value_cache.stride(2),
          value_cache.stride(3));
}

template <int HEAD_SIZE, int BLOCK_SIZE>
void launch_sm70_paged_decode_fp16(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size) {
  const int64_t num_heads = query.size(1);
  const int64_t num_queries_per_kv = num_heads / num_kv_heads;
  if (num_queries_per_kv >= 4) {
    launch_sm70_paged_decode_fp16_impl<HEAD_SIZE, BLOCK_SIZE, 4>(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size);
  } else if (num_queries_per_kv >= 2) {
    launch_sm70_paged_decode_fp16_impl<HEAD_SIZE, BLOCK_SIZE, 2>(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size);
  } else {
    launch_sm70_paged_decode_fp16_impl<HEAD_SIZE, BLOCK_SIZE, 1>(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size);
  }
}

}  // namespace

void sm70_paged_decode_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len) {
  TORCH_CHECK(query.dtype() == at::kHalf,
              "SM70 decode kernel currently supports FP16 query only");
  TORCH_CHECK(key_cache.dtype() == at::kHalf,
              "SM70 decode kernel currently supports FP16 KV cache only");
  TORCH_CHECK(value_cache.dtype() == at::kHalf,
              "SM70 decode kernel currently supports FP16 KV cache only");
  TORCH_CHECK(query.dim() == 3, "query must be [num_seqs, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [num_blocks, block_size, num_kv_heads, head_size]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must be [num_blocks, block_size, num_kv_heads, head_size]");
  TORCH_CHECK(block_size == key_cache.size(1), "block_size mismatch for key_cache");
  TORCH_CHECK(block_size == value_cache.size(1), "block_size mismatch for value_cache");
  TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  TORCH_CHECK(query.size(1) % num_kv_heads == 0,
              "num_heads must be divisible by num_kv_heads");

  const int64_t head_size = query.size(2);
  switch (block_size) {
    case 16:
      switch (head_size) {
        case 64:
          launch_sm70_paged_decode_fp16<64, 16>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
        case 128:
          launch_sm70_paged_decode_fp16<128, 16>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
        case 256:
          launch_sm70_paged_decode_fp16<256, 16>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
      }
      break;
    case 32:
      switch (head_size) {
        case 64:
          launch_sm70_paged_decode_fp16<64, 32>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
        case 128:
          launch_sm70_paged_decode_fp16<128, 32>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
        case 256:
          launch_sm70_paged_decode_fp16<256, 32>(
              out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, seq_lens, block_size);
          return;
      }
      break;
  }

  TORCH_CHECK(false,
              "Unsupported SM70 decode configuration: head_size=", head_size,
              ", block_size=", block_size);
}

}  // namespace vllm
