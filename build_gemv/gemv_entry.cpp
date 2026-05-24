#include <torch/library.h>
#include <cstdint>
namespace vllm {
  void gemv_paged_decode_attention(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, double, at::Tensor&, at::Tensor&, int64_t, int64_t);
  void segment_gemv_paged_decode_attention(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, double, at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, at::Tensor&, at::Tensor&, at::Tensor&);
}
TORCH_LIBRARY(gemv_decode_ops, m) {
  m.def("gemv_decode(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, int block_size, int num_pages) -> ()");
  m.impl("gemv_decode", c10::kCUDA, [](at::Tensor out, at::Tensor query, at::Tensor key_cache, at::Tensor value_cache, int64_t nkv, double s, at::Tensor bt, at::Tensor sl, int64_t bs, int64_t np) {
      vllm::gemv_paged_decode_attention(out, query, key_cache, value_cache, nkv, s, bt, sl, bs, np); });

  m.def("segment_gemv_decode(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, int block_size, int num_pages, int num_segments, Tensor seg_max, Tensor seg_sum, Tensor seg_out) -> ()");
  m.impl("segment_gemv_decode", c10::kCUDA, [](at::Tensor out, at::Tensor query, at::Tensor key_cache, at::Tensor value_cache, int64_t nkv, double s, at::Tensor bt, at::Tensor sl, int64_t bs, int64_t np, int64_t ns, at::Tensor sm, at::Tensor ss, at::Tensor so) {
      vllm::segment_gemv_paged_decode_attention(out, query, key_cache, value_cache, nkv, s, bt, sl, bs, np, ns, sm, ss, so); });
}
