// ============================================================================
// * Copyright (c) 2025, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ============================================================================
#ifndef FUSED_MHA_H
#define FUSED_MHA_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/extension.h>
#include <ATen/ATen.h>

/**
 * Flash Attention Forward Pass
 * 
 * Computes attention output with memory-efficient implementation.
 * 
 * @param q                   Query tensor [B, H, M, D] (fp16)
 * @param k                   Key tensor [B, H, N, D] (fp16) 
 * @param v                   Value tensor [B, H, N, D] (fp16)
 * @param out_                Optional output tensor [B, H, M, D] (fp16, output)
 * @param alibi_slopes_       Optional ALiBi slopes tensor for relative positioning
 * @param p_dropout           Dropout probability (0.0 for no dropout)
 * @param softmax_scale       Softmax scale (typically 1/sqrt(D))
 * @param is_causal           Enable causal masking (autoregressive)
 * @param window_size_left    Left window size for sliding window attention
 * @param window_size_right   Right window size for sliding window attention  
 * @param softcap             Soft capping value for attention scores
 * @param return_softmax      Whether to return softmax matrix (memory intensive)
 * @param gen_                Optional random generator for dropout
 * @return                    Vector of output tensors [output, softmax_lse, ...]
 */
std::vector<at::Tensor> flash_attention_forward(
    at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    std::optional<at::Tensor>& out_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_
);

/**
 * Flash Attention Backward Pass
 * 
 * Computes gradients dQ, dK, dV given inputs and output gradients.
 * Supports various attention variants including sliding window and ALiBi.
 * 
 * @param dout          Gradient w.r.t. output [B, H, M, D] (fp16)
 * @param q             Query tensor from forward [B, H, M, D] (fp16)
 * @param k             Key tensor from forward [B, H, N, D] (fp16)
 * @param v             Value tensor from forward [B, H, N, D] (fp16)
 * @param out           Output from forward pass [B, H, M, D] (fp16)
 * @param softmax_lse   Softmax logsumexp from forward [B, H, M] (fp32)
 * @param dq_           Optional gradient w.r.t. query [B, H, M, D] (fp16, output)
 * @param dk_           Optional gradient w.r.t. key [B, H, N, D] (fp16, output)
 * @param dv_           Optional gradient w.r.t. value [B, H, N, D] (fp16, output)
 * @param alibi_slopes_ Optional ALiBi slopes tensor (must match forward)
 * @param p_dropout     Dropout probability (must match forward)
 * @param softmax_scale Softmax scale (must match forward)
 * @param is_causal     Causal masking flag (must match forward)
 * @param window_size_left    Left window size (must match forward)
 * @param window_size_right   Right window size (must match forward)
 * @param softcap       Soft capping value (must match forward)
 * @param deterministic Whether to use deterministic backward pass
 * @param gen_          Optional random generator for dropout
 * @param rng_state     Optional RNG state for reproducibility
 * @return              Vector of gradient tensors [dq, dk, dv, ...]
 */
std::vector<at::Tensor> flash_attention_backward(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    std::optional<at::Tensor>& dq_,
    std::optional<at::Tensor>& dk_,
    std::optional<at::Tensor>& dv_,
    std::optional<at::Tensor>& alibi_slopes_,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    std::optional<at::Generator> gen_,
    std::optional<at::Tensor>& rng_state
);

/**
 * Flash Attention Paged Forward Pass
 *
 * Computes attention output with block-table-based KV cache and native GQA support.
 *
 * @param q                   Query tensor [num_tokens, num_heads, D] (fp16)
 * @param k_cache             Key cache [num_blocks, block_size, num_kv_heads, D] (fp16)
 * @param v_cache             Value cache [num_blocks, block_size, num_kv_heads, D] (fp16)
 * @param block_table         Block table [num_seqs, max_blocks_per_seq] (int32)
 * @param seq_lens            Sequence lengths [num_seqs] (int32)
 * @param query_start_loc     Query start positions [num_seqs + 1] (int32)
 * @param prefix_kv_lens      Prefix KV lengths [num_seqs] (int32)
 * @param out                 Output tensor [num_tokens, num_heads, D] (fp16)
 * @param num_kv_heads        Number of KV heads (for GQA)
 * @param block_size          Tokens per block
 * @param softmax_scale       Softmax scale (typically 1/sqrt(D))
 * @param is_causal           Enable causal masking
 * @return                    Vector of output tensors [output, softmax_lse]
 */
std::vector<at::Tensor> flash_attention_paged_forward(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& seq_lens,
    const at::Tensor& query_start_loc,
    const at::Tensor& prefix_kv_lens,
    at::Tensor& out,
    const int num_kv_heads,
    const int block_size,
    const float softmax_scale,
    bool is_causal
);

#endif // FUSED_MHA_H