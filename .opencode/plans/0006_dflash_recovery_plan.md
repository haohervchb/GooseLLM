# DFlash Integration Recovery Plan

## Executive Summary

The DFlash speculative decoding integration in GooseLLM is broken due to a **single critical bug** in `vllm/v1/spec_decode/dflash.py` (line 180): using `common_attn_metadata.num_actual_tokens` (query count) instead of `self._dflash_num_context` (context count) for slicing context buffers. This causes the RoPE operation to fail with a shape mismatch.

## Root Cause Analysis

### The Bug

In `DFlashProposer.propose()`:
```python
# After set_inputs_first_pass(), num_actual_tokens is OVERWRITTEN
# from num_context → num_query_total (e.g., 11 → 16)
num_query, token_indices_to_sample, common_attn_metadata = (
    self.set_inputs_first_pass(...)
)

# BUG: Uses num_query_total (16) instead of num_context (11)
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,  # shape [num_context, hidden_size]
    self._context_positions_buffer[: common_attn_metadata.num_actual_tokens],  # WRONG
    self._context_slot_mapping_buffer[: common_attn_metadata.num_actual_tokens],  # WRONG
)
```

The `set_inputs_first_pass()` method stores the original context count in `self._dflash_num_context` (line 233) but then overwrites `common_attn_metadata.num_actual_tokens` to `num_query_total` (line 290). The downstream `precompute_and_store_context_kv()` expects positions to match the hidden states shape, causing the RoPE error.

### Why One-By-One Fixes Were Wrong

My previous approach fixed symptoms without addressing the architectural mismatch:
1. **Per-layer RoPE fix**: Broke the fused RoPE optimization that was actually correct
2. **kv_cache list handling**: Added a workaround for a non-existent problem
3. **use_aux_hidden_state = False**: Disabled a feature instead of fixing it

The upstream model file (`~/vllm/vllm/model_executor/models/qwen3_dflash.py`) is correct - it just needs the right context count.

## Architecture Differences

GooseLLM uses `eagle.py` as the `SpecDecodeBaseProposer` base class, while upstream vLLM has moved to `llm_base_proposer.py`. Key differences:

| Feature | Upstream (~vllm) | GooseLLM |
|---------|------------------|----------|
| Base class | `llm_base_proposer.py` | `eagle.py` |
| `build_model_inputs_first_pass()` | Yes | No |
| `_determine_batch_execution_and_padding()` | Yes | No |
| `seq_lens_cpu_upper_bound` | Yes | No |
| `propose()` method | Unified in base | Custom in DFlash |

**We cannot simply port upstream files** because GooseLLM's base class is different.

## Comprehensive Fix Plan

### Step 1: Fix the Core Bug (1 line)

**File**: `vllm/v1/spec_decode/dflash.py` line 180

```python
# BEFORE (broken):
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,
    self._context_positions_buffer[: common_attn_metadata.num_actual_tokens],
    self._context_slot_mapping_buffer[: common_attn_metadata.num_actual_tokens],
)

# AFTER (fixed):
num_context = self._dflash_num_context
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,
    self._context_positions_buffer[:num_context],
    self._context_slot_mapping_buffer[:num_context],
)
```

### Step 2: Revert Model File to Upstream

**File**: `vllm/model_executor/models/qwen3_dflash.py`

Replace with the upstream version from `~/vllm/vllm/model_executor/models/qwen3_dflash.py`. The upstream version:
- Uses correct fused RoPE across all layers
- Sets `use_aux_hidden_state = True` by default
- Does NOT have the kv_cache list workaround

### Step 3: Fix use_aux_hidden_state Default

**File**: `vllm/v1/spec_decode/dflash.py`

Change `_get_eagle3_use_aux_hidden_state_from_config()` to return `True` by default (matching upstream):
```python
def _get_eagle3_use_aux_hidden_state_from_config(self):
    use_aux_hidden_state = True  # was False
    ...
```

### Step 4: Verify gpu_model_runner.py Integration

**File**: `vllm/v1/worker/gpu_model_runner.py`

Ensure DFlash is properly integrated:
- Import `DFlashProposer` ✓ (already done)
- Add to `isinstance` checks ✓ (already done)
- Verify `propose_draft_token_ids` calls the drafter correctly

### Step 5: Test with FLASH_ATTN_V100 Backend

Run the exact command the user specified:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_CUSTOM_ALLREDUCE_ALGO=1stage \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 --port 8082 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 15}'
```

### Step 6: If Issues Persist, Debug Further

Potential remaining issues:
- `FLASH_ATTN_V100` backend might not support non-causal attention (DFlash requires `causal=False`)
- The draft model weights might need special handling
- CUDA graph capture might fail with DFlash's custom buffers

## Files to Modify

1. `vllm/v1/spec_decode/dflash.py` - Fix context count bug + revert use_aux_hidden_state
2. `vllm/model_executor/models/qwen3_dflash.py` - Replace with upstream version

## Files Already Correct

- `vllm/v1/spec_decode/eagle.py` - Base class is fine
- `vllm/v1/worker/gpu_model_runner.py` - Integration is fine

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| FLASH_ATTN_V100 doesn't support non-causal | DFlash sets `causal=False` in attention metadata; backend must support it |
| Draft model weight loading fails | Upstream model file handles this correctly |
| CUDA graph issues with custom buffers | `dummy_run` early-returns to avoid this |

## Success Criteria

1. Server starts without errors
2. First chat completion request succeeds
3. Speculative decoding produces draft tokens (check logs for acceptance rate)
