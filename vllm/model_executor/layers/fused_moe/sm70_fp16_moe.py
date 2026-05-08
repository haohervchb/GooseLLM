"""
SM70 FP16 MoE method using TurboMind s884h GEMM kernels.

Converts FP16 MoE weights to TurboMind format and uses batched GEMM
via StridedPtr arrays for zero-sync, CUDA-graph-safe MoE forward.
"""

import os

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

_DEFAULT_PERSISTENT_MAX_TOKENS = 32


def _interleave_output_rows_for_gated_silu(weight: torch.Tensor) -> torch.Tensor:
    half = weight.shape[0] // 2
    first = weight[:half]
    second = weight[half:]
    return torch.stack((first, second), dim=1).reshape(weight.shape)


def _round_up(value: int, align: int) -> int:
    if align <= 0:
        return value
    return ((value + align - 1) // align) * align


def _sm70_fp16_moe_compare_enabled() -> bool:
    return os.getenv("VLLM_SM70_FP16_MOE_VERIFY") == "1"


class SM70FP16MoEMethod(FusedMoEMethodBase):
    """FP16 MoE method for SM70 (V100) using TurboMind GEMM kernels."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update({})
        extra_weight_attrs.pop("intermediate_size_full", None)

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert FP16 weights to TurboMind format and pre-allocate buffers."""
        if not current_platform.is_cuda_alike():
            return

        cap = torch.cuda.get_device_capability(layer.w13_weight.device)
        if cap != (7, 0):
            return

        if layer.w13_weight.dtype != torch.float16:
            return

        num_experts = layer.w13_weight.shape[0]
        hidden_size = layer.w13_weight.shape[2]
        intermediate_size = layer.w13_weight.shape[1] // 2

        n_align = 32
        inter_k = _round_up(intermediate_size, 16)
        dtype = layer.w13_weight.dtype
        device = layer.w13_weight.device

        # Interleave w13 gate/up rows across all experts at once
        half = intermediate_size
        w13_first = layer.w13_weight[:, :half]
        w13_second = layer.w13_weight[:, half:]
        w13_2d = (
            torch.stack((w13_first, w13_second), dim=2)
            .reshape(num_experts, 2 * intermediate_size, hidden_size)
            .reshape(-1, hidden_size)
        )
        w13_n = w13_2d.shape[0]
        w13_n_align = _round_up(w13_n, n_align)
        if w13_n_align != w13_n:
            w13_2d = torch.cat([
                w13_2d,
                torch.zeros(w13_n_align - w13_n, hidden_size,
                            dtype=dtype, device=device)
            ], dim=0)
        r13 = ops.sm70_f16_prepare(w13_2d)
        w13_k_ld = int(r13[1][0].item())
        w13_tm = r13[0]
        if w13_n_align != w13_n:
            w13_tm = w13_tm[:w13_n]
        layer.w13_tm_weight = Parameter(
            w13_tm.reshape(num_experts, 2 * intermediate_size, hidden_size),
            requires_grad=False)
        layer.w13_k_ld_list = [w13_k_ld] * num_experts

        # w2: process all experts at once
        w2_2d = layer.w2_weight.reshape(-1, intermediate_size)
        if inter_k != intermediate_size:
            w2_pad = torch.zeros(
                w2_2d.shape[0], inter_k,
                dtype=dtype, device=device
            )
            w2_pad[:, :intermediate_size] = w2_2d
            w2_2d = w2_pad
        w2_n = w2_2d.shape[0]
        w2_n_align = _round_up(w2_n, n_align)
        if w2_n_align != w2_n:
            w2_2d = torch.cat([
                w2_2d,
                torch.zeros(w2_n_align - w2_n, w2_2d.shape[1],
                            dtype=dtype, device=device)
            ], dim=0)
        r2 = ops.sm70_f16_prepare(w2_2d)
        w2_k_ld = int(r2[1][0].item())
        w2_tm = r2[0]
        if w2_n_align != w2_n:
            w2_tm = w2_tm[:w2_n]
        layer.w2_tm_weight = Parameter(
            w2_tm.reshape(num_experts, hidden_size, intermediate_size),
            requires_grad=False)
        layer.w2_k_ld_list = [w2_k_ld] * num_experts

        layer.sm70_num_experts = num_experts

        # Build StridedPtr arrays for batched GEMM
        w13_k_ld = layer.w13_k_ld_list[0]
        w2_k_ld = layer.w2_k_ld_list[0]
        try:
            w13_ptrs = ops.sm70_f16_moe_build_strided_ptrs(
                layer.w13_tm_weight, w13_k_ld, num_experts)
            w2_ptrs = ops.sm70_f16_moe_build_strided_ptrs(
                layer.w2_tm_weight, w2_k_ld, num_experts)
            layer.w13_strided_ptrs_w = Parameter(
                w13_ptrs[0], requires_grad=False)
            layer.w2_strided_ptrs_w = Parameter(
                w2_ptrs[0], requires_grad=False)
            layer.sm70_batched_ready = True
            logger.info_once(
                "SM70 FP16 MoE: per-expert GEMM (%d experts, "
                "hidden=%d inter=%d)",
                num_experts, hidden_size, intermediate_size,
            )
        except Exception as e:
            layer.sm70_batched_ready = False
            logger.warning(
                "SM70 FP16 MoE: batched GEMM unavailable (%s), "
                "using per-expert loop fallback.",
                e,
            )

        # Dimensions for batched GEMM
        layer.sm70_w13_k_dim = hidden_size
        layer.sm70_w13_n_dim = 2 * intermediate_size
        layer.sm70_w2_k_dim = intermediate_size
        layer.sm70_w2_n_dim = hidden_size
        layer.sm70_intermediate_size = intermediate_size
        layer.sm70_hidden_logical_size = hidden_size

        # Pre-allocate persistent decode workspace
        top_k = self.moe.experts_per_token
        persistent_tokens = _DEFAULT_PERSISTENT_MAX_TOKENS
        max_slots = persistent_tokens * top_k
        device = layer.w13_tm_weight.device

        layer._buf_max_tokens = persistent_tokens
        layer._buf_max_slots = max_slots
        layer._buf_top_k = top_k
        layer._buf_expert_counts = torch.empty(
            num_experts, dtype=torch.int32, device=device)
        layer._buf_expert_offsets = torch.empty(
            num_experts + 1, dtype=torch.int32, device=device)
        layer._buf_expert_offsets64 = torch.empty(
            num_experts + 1, dtype=torch.int64, device=device)
        layer._buf_gate_up = torch.empty(
            max_slots, layer.sm70_w13_n_dim // 2,
            dtype=torch.float16, device=device)
        layer._buf_intermediate = torch.empty(
            max_slots, layer.sm70_w2_k_dim,
            dtype=torch.float16, device=device)
        layer._buf_permuted_input = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device)
        layer._buf_sorted_output = torch.empty(
            max_slots, hidden_size, dtype=torch.float16, device=device)
        layer._buf_inv_permuted_idx = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device)
        layer._buf_topk_ids_i32 = torch.empty(
            persistent_tokens, top_k, dtype=torch.int32, device=device)
        layer._buf_token_expert_indices = torch.arange(
            max_slots, dtype=torch.int32, device=device).view(
                persistent_tokens, top_k)
        layer._buf_permuted_idx = torch.empty(
            max_slots, dtype=torch.int32, device=device)
        layer._buf_m_indices = torch.empty(
            max_slots, dtype=torch.int32, device=device)
        layer._buf_output = torch.empty(
            persistent_tokens, hidden_size, dtype=torch.float16,
            device=device)

        # Free original weights and clear conversion cache
        del layer.w13_weight, layer.w2_weight
        clear_fn = getattr(ops, "sm70_f16_clear_weight_cache", None)
        if clear_fn is not None:
            clear_fn()

    def _get_buffers(self, layer: torch.nn.Module, total_slots: int,
                     num_tokens: int):
        if (total_slots <= layer._buf_max_slots
                and num_tokens <= layer._buf_max_tokens):
            return {
                "output": layer._buf_output[:num_tokens],
                "permuted_input": layer._buf_permuted_input[:total_slots],
                "sorted_output": layer._buf_sorted_output[:total_slots],
                "gate_up": layer._buf_gate_up[:total_slots],
                "intermediate": layer._buf_intermediate[:total_slots],
                "expert_offsets": layer._buf_expert_offsets,
                "expert_offsets64": layer._buf_expert_offsets64,
                "inv_permuted_idx":
                layer._buf_inv_permuted_idx[:num_tokens],
                "topk_ids_i32": layer._buf_topk_ids_i32[:num_tokens],
                "token_expert_indices":
                layer._buf_token_expert_indices[:num_tokens],
                "permuted_idx": layer._buf_permuted_idx[:total_slots],
                "m_indices": layer._buf_m_indices[:total_slots],
            }

        device = layer._buf_output.device
        top_k = layer._buf_top_k
        hidden_size = layer.sm70_hidden_logical_size
        return {
            "output": torch.empty(num_tokens, hidden_size,
                                  dtype=torch.float16, device=device),
            "permuted_input": torch.empty(total_slots, hidden_size,
                                          dtype=torch.float16, device=device),
            "sorted_output": torch.empty(total_slots, hidden_size,
                                         dtype=torch.float16, device=device),
            "gate_up": torch.empty(total_slots, layer.sm70_w13_n_dim // 2,
                                   dtype=torch.float16, device=device),
            "intermediate": torch.empty(total_slots, layer.sm70_w2_k_dim,
                                        dtype=torch.float16, device=device),
            "expert_offsets": torch.empty(layer.sm70_num_experts + 1,
                                          dtype=torch.int32, device=device),
            "expert_offsets64": torch.empty(layer.sm70_num_experts + 1,
                                            dtype=torch.int64, device=device),
            "inv_permuted_idx": torch.empty(num_tokens, top_k,
                                            dtype=torch.int32, device=device),
            "topk_ids_i32": torch.empty(num_tokens, top_k,
                                        dtype=torch.int32, device=device),
            "token_expert_indices": torch.arange(
                total_slots, dtype=torch.int32, device=device).view(
                    num_tokens, top_k),
            "permuted_idx": torch.empty(total_slots, dtype=torch.int32,
                                        device=device),
            "m_indices": torch.empty(total_slots, dtype=torch.int32,
                                     device=device),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(layer, "sm70_batched_ready", False):
            return self._apply_batched(layer, x, topk_weights, topk_ids)
        return self._apply_sorted_loop(layer, x, topk_weights, topk_ids)

    def _permute_tokens_by_expert(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        buffers: dict[str, torch.Tensor],
    ):
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]

        permuted_input = buffers["permuted_input"]
        expert_offsets64 = buffers["expert_offsets64"]
        inv_permuted_idx = buffers["inv_permuted_idx"]
        permuted_idx = buffers["permuted_idx"]
        m_indices = buffers["m_indices"]
        topk_ids_i32 = buffers["topk_ids_i32"]
        token_expert_indices = buffers["token_expert_indices"]

        topk_ids_i32.copy_(topk_ids, non_blocking=True)
        torch.ops._moe_C.moe_permute(
            x,
            topk_ids_i32,
            token_expert_indices,
            None,
            num_experts,
            num_experts,
            top_k,
            None,
            permuted_input,
            expert_offsets64,
            inv_permuted_idx,
            permuted_idx,
            m_indices,
        )
        buffers["expert_offsets"].copy_(expert_offsets64, non_blocking=True)
        return permuted_input, expert_offsets64, inv_permuted_idx

    def _apply_batched(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        buffers = self._get_buffers(layer, total_slots, num_tokens)
        output = buffers["output"]
        output.zero_()
        if total_slots == 0:
            return output

        (permuted_input, expert_offsets64,
         inv_permuted_idx) = self._permute_tokens_by_expert(
            layer, x, topk_ids, num_experts, buffers)
        expert_offsets = buffers["expert_offsets"]
        intermediate = buffers["intermediate"]
        gate_up = buffers["gate_up"]
        sorted_output = buffers["sorted_output"]

        # Batched w13 GEMM (gate+up) with gated SiLU
        ops.sm70_f16_moe_gemm_sm70_out(
            gate_up,
            permuted_input, expert_offsets,
            layer.w13_strided_ptrs_w,
            num_experts, layer.sm70_w13_k_dim,
            layer.sm70_w13_n_dim, True,
        )

        # Store in the intermediate-sized buffer for w2
        intermediate.copy_(gate_up)

        # Batched w2 GEMM (down projection)
        ops.sm70_f16_moe_gemm_sm70_out(
            sorted_output,
            intermediate, expert_offsets,
            layer.w2_strided_ptrs_w,
            num_experts, layer.sm70_w2_k_dim,
            layer.sm70_w2_n_dim, False,
        )

        if _sm70_fp16_moe_compare_enabled():
            ref = self._apply_sorted_loop(layer, x, topk_weights, topk_ids)
            diff = (output - ref).abs().max().item()
            if diff > 0.01:
                logger.warning(
                    "SM70 FP16 MoE verify: batched vs sorted-loop "
                    "max_diff=%.6f layer=%s",
                    diff, getattr(layer, "layer_name", "<unknown>"),
                )

        sorted_output_logical = sorted_output[:, :hidden_size]
        torch.ops._moe_C.moe_unpermute(
            sorted_output_logical,
            topk_weights,
            inv_permuted_idx,
            expert_offsets64,
            top_k,
            output,
        )
        return output

    def _apply_sorted_loop(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = x.shape
        top_k = topk_ids.shape[1]
        num_experts = layer.sm70_num_experts
        total_slots = num_tokens * top_k

        output = torch.zeros(num_tokens, hidden_size, dtype=x.dtype,
                             device=x.device)
        if total_slots == 0:
            return output

        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        token_origin = (
            torch.arange(num_tokens, device=x.device, dtype=torch.int64)
            .unsqueeze(1).expand(num_tokens, top_k).reshape(-1))

        sorted_order = torch.argsort(flat_ids.long(), stable=True)
        sorted_token_origin = token_origin[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_input = x[sorted_token_origin]

        sorted_expert_ids = flat_ids[sorted_order]
        expert_counts = torch.bincount(
            sorted_expert_ids.long(), minlength=num_experts)
        expert_offsets = torch.zeros(
            num_experts + 1, dtype=torch.int64, device=x.device)
        torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])

        h_offsets = expert_offsets.cpu().numpy()
        h_counts = expert_counts.cpu().numpy()

        sorted_output = torch.empty(
            total_slots, hidden_size, dtype=x.dtype, device=x.device)

        for e in range(num_experts):
            if h_counts[e] == 0:
                continue
            start, end = int(h_offsets[e]), int(h_offsets[e + 1])
            expert_input = sorted_input[start:end]

            w13_tm = layer.w13_tm_weight[e]
            w13_k_ld = layer.w13_k_ld_list[e]
            gate_up = torch.empty(
                end - start, layer.sm70_w13_n_dim // 2,
                dtype=x.dtype, device=x.device)
            ops.sm70_f16_gemm_out(
                gate_up, expert_input, w13_tm, w13_k_ld, True,
            )

            w2_tm = layer.w2_tm_weight[e]
            w2_k_ld = layer.w2_k_ld_list[e]
            expert_output = torch.empty(
                end - start, hidden_size, dtype=x.dtype, device=x.device)
            ops.sm70_f16_gemm_out(
                expert_output, gate_up, w2_tm, w2_k_ld, False,
            )
            sorted_output[start:end] = expert_output

        weighted = sorted_output * sorted_weights.unsqueeze(1).to(x.dtype)
        output.index_add_(0, sorted_token_origin, weighted)
        return output

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None
