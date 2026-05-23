#include <torch/library.h>
#include <cstdint>
namespace vllm { void gemv_paged_decode_attention(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, double, at::Tensor&, at::Tensor&, int64_t, int64_t); }
TORCH_LIBRARY(gemv_decode_ops, m) {
  m.def("gemv_decode(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, int block_size, int num_pages) -> ()");
  m.impl("gemv_decode", c10::kCUDA, [](at::Tensor out, at::Tensor query, at::Tensor key_cache, at::Tensor value_cache, int64_t nkv, double s, at::Tensor bt, at::Tensor sl, int64_t bs, int64_t np) {
      vllm::gemv_paged_decode_attention(out, query, key_cache, value_cache, nkv, s, bt, sl, bs, np); });
}
