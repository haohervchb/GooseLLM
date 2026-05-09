/*
 * SM70 FP16 MoE GEMM using TurboMind s884h kernels.
 * Completely self-contained — no AWQ dependency.
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <mutex>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace vllm {
namespace sm70_fp16_moe {

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& k) const {
    return std::hash<int>()(k.device) ^
           (std::hash<cudaStream_t>()(k.stream) << 1);
  }
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::unordered_map<StreamWorkspaceKey, WorkspaceHolder,
                   StreamWorkspaceKeyHash> workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;

WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  StreamWorkspaceKey key{device, stream};
  {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto it = workspace_cache.find(key);
    if (it != workspace_cache.end()) {
      return it->second;
    }
  }

  WorkspaceHolder holder;
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder.barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder.partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  holder.tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder.flags = torch::zeros({1}, int_opts);

  holder.workspace.barriers = holder.barriers.data_ptr();
  holder.workspace.barriers_size = holder.barriers.numel();
  holder.workspace.partials = holder.partials.data_ptr();
  holder.workspace.partials_size = holder.partials.numel();
  holder.workspace.tensormaps = holder.tensormaps.data_ptr();
  holder.workspace.tensormaps_size = holder.tensormaps.numel();
  holder.workspace.flags = holder.flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto [insert_it, _] = workspace_cache.emplace(key, std::move(holder));
  return insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  return *insert_it->second.gemm;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Build StridedPtr arrays for FP16 MoE weights (no scales)
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> sm70_f16_moe_build_strided_ptrs(
    torch::Tensor tm_weights,
    int64_t k_ld,
    int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(),
              "sm70_f16_moe_build_strided_ptrs: weights must be CUDA.");
  TORCH_CHECK(num_experts > 0,
              "sm70_f16_moe_build_strided_ptrs: num_experts must be > 0.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "sm70_f16_moe_build_strided_ptrs: weights dim0 != num_experts.");
  TORCH_CHECK(tm_weights.scalar_type() == torch::kFloat16,
              "sm70_f16_moe_build_strided_ptrs: weights must be float16.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  std::vector<std::pair<void*, int>> w_ptrs;
  w_ptrs.reserve(num_experts);

  const int64_t w_expert_stride =
      tm_weights.stride(0) * tm_weights.element_size();
  char* w_base = static_cast<char*>(tm_weights.data_ptr());

  for (int64_t e = 0; e < num_experts; ++e) {
    w_ptrs.emplace_back(w_base + e * w_expert_stride,
                        static_cast<int>(k_ld));
  }

  void* w_gpu = turbomind::gemm::MakeStridedPtrs(w_ptrs, stream);

  const int64_t buf_bytes = num_experts * 16;
  auto opts = torch::TensorOptions()
                  .device(tm_weights.device())
                  .dtype(torch::kUInt8);

  auto w_tensor = torch::empty({buf_bytes}, opts);
  cudaMemcpyAsync(w_tensor.data_ptr(), w_gpu, buf_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(w_gpu, stream);

  return {w_tensor};
}

// ---------------------------------------------------------------------------
// Batched FP16 MoE GEMM via TurboMind with StridedPtr arrays
// ---------------------------------------------------------------------------

void sm70_f16_moe_gemm_sm70_out(
    torch::Tensor out,
    torch::Tensor sorted_input,
    torch::Tensor expert_offsets,
    torch::Tensor strided_ptrs_w,
    int64_t num_experts,
    int64_t k,
    int64_t n,
    bool gated_silu) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "sm70_f16_moe_gemm_sm70: input must be CUDA float16.");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.scalar_type() == torch::kInt32,
      "sm70_f16_moe_gemm_sm70: expert_offsets must be CUDA int32.");
  TORCH_CHECK(strided_ptrs_w.is_cuda(),
              "sm70_f16_moe_gemm_sm70: strided_ptrs must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "sm70_f16_moe_gemm_sm70: output must be CUDA float16.");
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "sm70_f16_moe_gemm_sm70: invalid dimensions.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t total_tokens = sorted_input.size(0);
  TORCH_CHECK(out.size(0) == total_tokens,
              "sm70_f16_moe_gemm_sm70: output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "sm70_f16_moe_gemm_sm70: output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0,
                "sm70_f16_moe_gemm_sm70: gated_silu requires even n.");
    TORCH_CHECK(out.size(1) == n / 2,
                "sm70_f16_moe_gemm_sm70: gated_silu output cols must be n/2.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "sm70_f16_moe_gemm_sm70: output cols must match n.");
  }

  if (total_tokens == 0) return;

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* conv_w = converters[0];
  TORCH_CHECK(conv_w,
              "sm70_f16_moe_gemm_sm70: no compatible TurboMind converter.");

  turbomind::gemm::MatrixLayout desc_A{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  desc_A.num = static_cast<int>(num_experts);
  desc_A.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::MatrixLayout desc_U{};

  const auto order_w = conv_w->order;
  const bool is_A_w =
      turbomind::gemm::get_operand_tag(conv_w->pack) ==
      turbomind::gemm::OPERAND_A;
  const bool is_B_w = !is_A_w;

  turbomind::gemm::MatrixLayout w_desc{
      turbomind::kHalf, order_w,
      static_cast<int>(n), static_cast<int>(k),
      order_w == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                            : static_cast<int>(n),
  };
  if (is_B_w) {
    std::swap(w_desc.rows, w_desc.cols);
    w_desc.order = ~w_desc.order;
  }

  turbomind::gemm::MatrixLayout desc_B = w_desc;
  desc_B.pack = conv_w->pack;
  if (is_A_w) {
    desc_B = turbomind::gemm::transpose(desc_B);
  }
  desc_B.ld = 0;
  desc_B.num = static_cast<int>(num_experts);

  turbomind::gemm::MatrixLayout desc_D{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  desc_D.num = static_cast<int>(num_experts);
  desc_D.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation op{};
  op.dispatch = turbomind::gemm::DispatchPolicy::kDefault;
  op.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                           : turbomind::gemm::Epilogue::kNone;
  op.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  op.quant_b = {turbomind::gemm::QuantType::kNone, 0};
  op.batch_dim = 0;

  auto& workspace_holder = get_workspace(device, stream);
  auto& gemm = get_gemm(device);

  const int ec = gemm.Run(op, 1.f,
      sorted_input.data_ptr(), desc_A,
      nullptr, desc_U,
      strided_ptrs_w.data_ptr(), desc_B,
      nullptr, desc_U,
      0.f,
      out.data_ptr(), desc_D,
      out.data_ptr(), desc_D,
      workspace_holder.workspace, stream);

  TORCH_CHECK(ec == 0,
              "sm70_f16_moe_gemm_sm70: TurboMind GEMM failed (ec=", ec, ").");
}

}  // namespace sm70_fp16_moe
}  // namespace vllm

// Global-scope entry points (linked from torch_bindings.cpp)
std::vector<torch::Tensor> sm70_f16_moe_build_strided_ptrs(
    torch::Tensor tm_weights, int64_t k_ld, int64_t num_experts) {
  return vllm::sm70_fp16_moe::sm70_f16_moe_build_strided_ptrs(
      tm_weights, k_ld, num_experts);
}

void sm70_f16_moe_gemm_sm70_out(torch::Tensor out,
                                torch::Tensor sorted_input,
                                torch::Tensor expert_offsets,
                                torch::Tensor strided_ptrs_w,
                                int64_t num_experts,
                                int64_t k,
                                int64_t n,
                                bool gated_silu) {
  vllm::sm70_fp16_moe::sm70_f16_moe_gemm_sm70_out(
      out, sorted_input, expert_offsets, strided_ptrs_w,
      num_experts, k, n, gated_silu);
}
