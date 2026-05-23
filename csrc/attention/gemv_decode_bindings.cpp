#include <cstdint>

#include <torch/extension.h>

namespace py = pybind11;

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gemv_paged_decode_attention",
      &vllm::gemv_paged_decode_attention,
      py::arg("out"),
      py::arg("query"),
      py::arg("key_cache"),
      py::arg("value_cache"),
      py::arg("num_kv_heads"),
      py::arg("scale"),
      py::arg("block_tables"),
      py::arg("seq_lens"),
      py::arg("block_size"),
      py::arg("num_pages"));
}
