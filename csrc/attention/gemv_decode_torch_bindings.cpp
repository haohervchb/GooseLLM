#include <cstdint>
#include <torch/library.h>

namespace vllm {

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
    int64_t num_pages);

}  // namespace vllm

TORCH_LIBRARY(vllm_gemv_decode_ops, m) {
  m.def(
      "gemv_decode(Tensor out, Tensor query, Tensor key_cache, "
      "Tensor value_cache, int num_kv_heads, Scalar scale, "
      "Tensor block_tables, Tensor seq_lens, int block_size, "
      "int num_pages) -> ()",
      [](torch::Tensor& out, torch::Tensor& query,
         torch::Tensor& key_cache, torch::Tensor& value_cache,
         int64_t num_kv_heads, at::Scalar scale,
         torch::Tensor& block_tables, torch::Tensor& seq_lens,
         int64_t block_size, int64_t num_pages) {
        vllm::gemv_paged_decode_attention(
            out, query, key_cache, value_cache,
            num_kv_heads, scale.toDouble(), block_tables,
            seq_lens, block_size, num_pages);
      });
}
