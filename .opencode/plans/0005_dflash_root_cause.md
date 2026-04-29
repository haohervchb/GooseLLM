# DFlash Integration Plan

## Root Cause Analysis

The error `RuntimeError: query, key and positions must have the same number of tokens` is caused by a single bug in `dflash.py`.

### The Bug

In `DFlashProposer.propose()` at line 180:
```python
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,
    self._context_positions_buffer[: common_attn_metadata.num_actual_tokens],  # BUG!
    self._context_slot_mapping_buffer[: common_attn_metadata.num_actual_tokens],
)
```

`common_attn_metadata.num_actual_tokens` is set **after** `set_inputs_first_pass()` to `num_query_total` (the query count = batch_size * (1+spec_tokens), e.g., 16), NOT the context count.

But `context_positions` and `context_slot_mapping` must be sliced by the **context count** (num_context), because `self._dflash_hidden_states` has shape `[num_context, hidden_size]`, and the fused RoPE operation in `precompute_and_store_context_kv` expects positions to match `num_context`.

When `num_context` (e.g., 11) ≠ `common_attn_metadata.num_actual_tokens` (e.g., 16), the RoPE operation fails because the K tensor has `L * num_context` elements but positions_repeated has `L * num_actual_tokens` elements.

### The Fix (1 line change)

```python
# Use self._dflash_num_context (saved during set_inputs_first_pass)
num_context = self._dflash_num_context
self.model.precompute_and_store_context_kv(
    self._dflash_hidden_states,
    self._context_positions_buffer[:num_context],
    self._context_slot_mapping_buffer[:num_context],
)
```

## Files Changed by Our Fixing Session

The following files were already changed and need to be committed:

### 1. vllm/v1/spec_decode/dflash.py
- Added `self.indexer_layer_names = []` in `load_model()`
- Added `AttentionMetadataBuilder` import
- Added `self._init_parallel_drafting_params()` call in `__init__`
- Replaced broken `_get_attention_metadata_builder()` with working version

### 2. vllm/model_executor/models/qwen3_dflash.py  
- Already reverted to upstream version from ~/vllm

### 3. vllm/v1/spec_decode/eagle.py (from earlier commits)
- Added `DFlashProposer` to assertions in `sample_tokens`

### 4. gpu_model_runner.py (from earlier commits)
- Added `DFlashProposer` to imports and isinstance checks

## Upstream Reference

The upstream source of truth is `/home/rah/vllm/` (vLLM main). Key comparison:

- **`~/vllm/v1/spec_decode/dflash.py`**: Uses `llm_base_proposer.py` instead of `eagle.py` base class
- **`~/vllm/v1/spec_decode/llm_base_proposer.py`**: Contains the unified `propose()` method
- The `build_model_inputs_first_pass()` method in upstream uses `self._dflash_num_context` correctly

The GooseLLM version has diverged from upstream - it uses `eagle.py` as the base class instead of the new `llm_base_proposer.py`. But the core DFlash logic (the one-line bug fix above) is the same.

## Verification

After the fix, test by running:
```bash
# Start server with DFlash speculative decoding
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}' \
    --tensor-parallel-size 4
```
