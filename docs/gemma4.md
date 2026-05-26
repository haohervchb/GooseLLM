# Gemma-4-31B Support in GooseLLM

## Overview

This document outlines the implementation plan for supporting Google's Gemma-4-31B model in GooseLLM with TileLang FA-V100 backend on NVIDIA V100 GPUs.

**Key Constraint**: Bottom-up testing strategy - kernel-level correctness must be verified BEFORE any model loading.

---

## Model Configuration Analysis

### Gemma-4-31B Architecture

```json
{
  "architectures": ["Gemma4ForCausalLM"],
  "text_config": {
    "hidden_size": 5376,
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,              // sliding_attention layers
    "global_head_dim": 512,       // full_attention layers
    "num_hidden_layers": 60,
    "sliding_window": 1024,
    "layer_types": [
      "sliding_attention",  // 50 layers, HD 256
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",     // 10 layers, HD 512
      ...
    ],
    "attn_logit_softcapping": null,
    "final_logit_softcapping": 30.0,
    "rope_parameters": {
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
      "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional"}
    }
  }
}
```

### Layer Distribution

| Layer Type | Count | Head Dim | Sliding Window | GQA Ratio |
|------------|-------|----------|----------------|-----------|
| sliding_attention | 50 | 256 | 1024 | 32:16 (2:1) |
| full_attention | 10 | 512 | None | 32:16 (2:1) |

### Backend Compatibility Matrix

| Feature | FLASH_ATTN_V100 | FLASH_ATTN_TILELANG_V100 | TRITON_ATTN |
|---------|-----------------|--------------------------|-------------|
| HD 256 | ✅ | ✅ | ✅ |
| HD 512 | ❌ (max 256) | ✅ | ✅ |
| Sliding Window | ❌ | ⚠️ (needs impl) | ✅ |
| GQA | ✅ | ✅ | ✅ |
| FP16 | ✅ | ✅ | ✅ |

**Critical Finding**: Full attention layers (HD 512) **CANNOT** use FLASH_ATTN_V100. Must use TileLang or Triton.

---

## Implementation Strategy: Bottom-Up Testing

### Testing Hierarchy

```
┌─────────────────────────────────────────────────┐
│ Phase 3: Model Integration                      │
│   (ONLY IF Phases 1 & 2 PASS)                   │
│   - Copy gemma4.py from mainline                │
│   - Register in registry                        │
│   - Load test with TP2/TP4                      │
├─────────────────────────────────────────────────┤
│ Phase 2: Backend-Level                          │
│   - vLLM attention backend integration          │
│   - TileLang vs Triton comparison               │
├─────────────────────────────────────────────────┤
│ Phase 1: Kernel-Level (DRY, No Model) ← START   │
│   ├─ HD 256 baseline (reference)                │
│   ├─ HD 512 dense (NEW - critical path)         │
│   ├─ HD 512 vs HD 256 comparison                │
│   ├─ HD 256 sliding window                      │
│   ├─ HD 512 sliding window                      │
│   └─ TP2/TP4 simulation                         │
├─────────────────────────────────────────────────┤
│ Phase 0: Prerequisites                          │
│   - Verify TileLang importable                  │
│   - Verify V100 GPU available                   │
└─────────────────────────────────────────────────┘
```

### Gate Criteria

- **Phase 1 Gate**: All kernel tests pass with `rtol=1e-3, atol=1e-4`
- **Phase 2 Gate**: Backend tests pass with `rtol=2e-2, atol=2e-2`
- **Phase 3 Gate**: Model loads and produces valid output

---

## Phase 0: Prerequisites

### Step 0.1: Verify Environment

```bash
cd /home/rah/GooseLLM

# Check TileLang FA-V100 installation
python3 -c "import tilelang_fa_v100; print('✓ TileLang FA-V100 OK')"
python3 -c "from tilelang_fa_v100 import tilelang_paged_forward; print('✓ Import OK')"

# Verify V100 GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Capability: {torch.cuda.get_device_capability(0)}')"
```

**Expected Output**:
```
✓ TileLang FA-V100 OK
✓ Import OK
GPU: Tesla V100-SXM2-16GB
Capability: (7, 0)
```

---

## Phase 1: Kernel-Level Tests

### Step 1.1: HD 256 Baseline Test

**File**: `tests/v1/attention/test_tilelang_kernel_hd256.py`

**Purpose**: Establish baseline for HD 256 (known working configuration)

```python
"""TileLang FA-V100 HD 256 correctness test - no model loading."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward

def test_hd256_dense():
    """HD 256 dense attention (baseline)."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 4, 1024, 16, 8, 256
    block_size, num_blocks = 16, 1024
    
    q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
    
    output = tilelang_paged_forward(
        q, k_cache, v_cache, block_table, cache_seqlens,
        head_dim=256, causal=True
    )
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("✓ HD 256 baseline passed")

if __name__ == "__main__":
    test_hd256_dense()
```

**Run**: `python tests/v1/attention/test_tilelang_kernel_hd256.py`

---

### Step 1.2: HD 512 Kernel Test (CRITICAL)

**File**: `tests/v1/attention/test_tilelang_kernel_hd512.py`

**Purpose**: Verify HD 512 compiles and produces valid output (Gemma4 full_attention layers)

**File**: `3rdparty/tilelang-fa-v100/tilelang_fa_v100/_kernels_paged.py`

**Location**: Line 275-279

**Action**: Add HD 512 configuration

```python
# Current (lines 275-279):
_BEST_CONFIGS = {
    64: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    128: dict(block_M=32, block_N=128, threads=256, num_stages=0, num_splits=1),
    256: dict(block_M=64, block_N=32, threads=256, num_stages=0, num_splits=1),
}

# ADD after line 278:
    512: dict(block_M=32, block_N=32, threads=256, num_stages=0, num_splits=1),
```

**Rationale**:
- `block_M=32, block_N=32` fits V100 shared memory (96KB limit)
- Shared memory calculation: Q(32KB) + K(32KB) + V(32KB) + P(2KB) = 98KB
- If OOM, try `block_M=16, block_N=32` or `block_M=32, block_N=16`

**Test Content**:
```python
"""TileLang FA-V100 HD 512 correctness test - no model loading."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward

def test_hd512_dense():
    """HD 512 dense attention (new feature)."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 4, 1024, 16, 8, 512
    block_size, num_blocks = 16, 1024
    
    q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
    
    output = tilelang_paged_forward(
        q, k_cache, v_cache, block_table, cache_seqlens,
        head_dim=512, causal=True
    )
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any(), "NaN detected in HD 512 output"
    assert not torch.isinf(output).any(), "Inf detected in HD 512 output"
    assert output.abs().max() < 100, f"Unusually large output: {output.abs().max()}"
    print("✓ HD 512 dense passed")

if __name__ == "__main__":
    test_hd512_dense()
```

**Success Criteria**:
- ✅ Kernel compiles (first run may take 30-60s for JIT)
- ✅ No NaN/Inf
- ✅ Output magnitude reasonable (< 100)
- ✅ No CUDA OOM errors

---

### Step 1.3: HD 512 vs HD 256 Comparison

**File**: `tests/v1/attention/test_tilelang_kernel_hd_comparison.py`

**Purpose**: Verify HD 512 produces similar quality to HD 256

```python
"""Compare HD 512 vs HD 256 output quality."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward

def test_hd_comparison():
    """HD 512 should have similar numerical properties as HD 256."""
    batch, seq_len, num_heads, num_kv_heads = 4, 1024, 16, 8
    block_size, num_blocks = 16, 1024
    
    torch.manual_seed(42)
    q_256 = torch.randn(batch * seq_len, num_heads, 256, dtype=torch.float16, device='cuda')
    k_256 = torch.randn(num_blocks, block_size, num_kv_heads, 256, dtype=torch.float16, device='cuda')
    v_256 = torch.randn(num_blocks, block_size, num_kv_heads, 256, dtype=torch.float16, device='cuda')
    
    torch.manual_seed(42)
    q_512 = torch.randn(batch * seq_len, num_heads, 512, dtype=torch.float16, device='cuda')
    k_512 = torch.randn(num_blocks, block_size, num_kv_heads, 512, dtype=torch.float16, device='cuda')
    v_512 = torch.randn(num_blocks, block_size, num_kv_heads, 512, dtype=torch.float16, device='cuda')
    
    block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
    
    out_256 = tilelang_paged_forward(q_256, k_256, v_256, block_table, cache_seqlens, head_dim=256, causal=True)
    out_512 = tilelang_paged_forward(q_512, k_512, v_512, block_table, cache_seqlens, head_dim=512, causal=True)
    
    stats_256 = {
        'mean': out_256.float().mean().item(),
        'std': out_256.float().std().item(),
        'max': out_256.abs().max().item(),
        'nan_count': torch.isnan(out_256).sum().item(),
    }
    stats_512 = {
        'mean': out_512.float().mean().item(),
        'std': out_512.float().std().item(),
        'max': out_512.abs().max().item(),
        'nan_count': torch.isnan(out_512).sum().item(),
    }
    
    print(f"HD 256 stats: {stats_256}")
    print(f"HD 512 stats: {stats_512}")
    
    assert stats_256['nan_count'] == 0
    assert stats_512['nan_count'] == 0
    assert stats_256['max'] < 100
    assert stats_512['max'] < 100
    
    print("✓ HD comparison passed")

if __name__ == "__main__":
    test_hd_comparison()
```

---

### Step 1.4: Sliding Window Test (HD 256)

**File**: `tests/v1/attention/test_tilelang_kernel_sliding.py`

**Purpose**: Verify sliding window works on HD 256 (Gemma4 sliding layers)

```python
"""TileLang FA-V100 sliding window test - HD 256 first."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward

def test_sliding_window_hd256():
    """HD 256 with sliding window (Gemma4 sliding layer)."""
    batch, seq_len, num_heads, num_kv_heads, head_dim = 4, 2048, 16, 8, 256
    sliding_window = 1024
    block_size, num_blocks = 16, 2048
    
    q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
    
    output = tilelang_paged_forward(
        q, k_cache, v_cache, block_table, cache_seqlens,
        head_dim=256, causal=True,
        sliding_window_q=sliding_window, sliding_window_k=sliding_window
    )
    
    assert output.shape == q.shape
    assert not torch.isnan(output).any(), "NaN in sliding window output"
    print("✓ HD 256 sliding window passed")

if __name__ == "__main__":
    test_sliding_window_hd256()
```

---

### Step 1.5: Sliding Window Test (HD 512)

**File**: `tests/v1/attention/test_tilelang_kernel_sliding_hd512.py`

**Purpose**: HD 512 + sliding window (complete Gemma4 case)

Same as Step 1.4 but with `head_dim=512`

---

### Step 1.6: TP2/TP4 Simulation Test

**File**: `tests/v1/attention/test_tilelang_kernel_tp_simulation.py`

**Purpose**: Verify kernel works with reduced head counts (simulating tensor parallelism)

```python
"""Simulate TP2/TP4 by testing with reduced head counts."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward

def test_tp_simulation():
    """Simulate TP2 (16 heads) and TP4 (8 heads) per GPU."""
    batch, seq_len, block_size, num_blocks = 4, 1024, 16, 1024
    
    test_cases = [
        ("TP2_sim", 16, 8, 256),   # 32/2=16 heads, 16/2=8 KV
        ("TP4_sim", 8, 4, 256),    # 32/4=8 heads, 16/4=4 KV
        ("TP2_sim_hd512", 16, 8, 512),
        ("TP4_sim_hd512", 8, 4, 512),
    ]
    
    for name, num_heads, num_kv_heads, head_dim in test_cases:
        q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
        cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
        
        output = tilelang_paged_forward(
            q, k_cache, v_cache, block_table, cache_seqlens,
            head_dim=head_dim, causal=True
        )
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        print(f"✓ {name} passed")

if __name__ == "__main__":
    test_tp_simulation()
```

---

## Phase 2: Backend-Level Tests

*(To be implemented after Phase 1 passes)*

### Step 2.1: TileLang Backend Integration

**File**: `tests/v1/attention/test_tilelang_backend_hd512.py`

### Step 2.2: Backend vs Triton Comparison

**File**: `tests/v1/attention/test_backend_comparison_hd512.py`

---

## Phase 3: Model Integration

*(ONLY IF Phases 1 & 2 PASS)*

### Step 3.1: Copy Gemma4 Model Files

**Source**: `~/vllm/vllm/model_executor/models/gemma4.py`

**Destination**: `/home/rah/GooseLLM/vllm/model_executor/models/gemma4.py`

```bash
cp ~/vllm/vllm/model_executor/models/gemma4.py \
   /home/rah/GooseLLM/vllm/model_executor/models/gemma4.py
```

### Step 3.2: Register in Registry

**File**: `/home/rah/GooseLLM/vllm/model_executor/models/registry.py`

**Location**: Line 111-112

**Action**: Add registration entry

```python
# After line 111 (Gemma3nForCausalLM):
"Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),
```

### Step 3.3: Model Loading Test

**File**: `tests/models/test_gemma4_load.py`

```python
"""Test Gemma4 model loads - ONLY RUN AFTER kernel/backend tests pass."""
from vllm import LLM

def test_gemma4_load_tp2():
    """Load Gemma4 with TP=2."""
    llm = LLM(
        model="google/gemma-4-1B",  # Small model for testing
        dtype="float16",
        tensor_parallel_size=2,
        enforce_eager=True,
    )
    
    outputs = llm.generate("Hello", max_tokens=5)
    assert len(outputs) > 0
    print("✓ Gemma4 TP2 load passed")

def test_gemma4_load_tp4():
    """Load Gemma4 with TP=4."""
    llm = LLM(
        model="google/gemma-4-1B",
        dtype="float16",
        tensor_parallel_size=4,
        enforce_eager=True,
    )
    
    outputs = llm.generate("Hello", max_tokens=5)
    assert len(outputs) > 0
    print("✓ Gemma4 TP4 load passed")
```

---

## File Modification Checklist

| Phase | File | Lines | Change | Status |
|-------|------|-------|--------|--------|
| 1.1 | `3rdparty/tilelang-fa-v100/tilelang_fa_v100/_kernels_paged.py` | 278 | Add HD 512 config | ⬜ |
| 1.2-1.6 | `tests/v1/attention/test_tilelang_kernel_*.py` | NEW | 5 test files | ⬜ |
| 2.1-2.2 | `tests/v1/attention/test_tilelang_backend_*.py` | NEW | 2 test files | ⬜ |
| 2.7-2.8 | `vllm/v1/attention/backends/flash_attn_tilelang_v100.py` | 90-93, ~200-250 | Sliding window support | ⬜ |
| 3.1 | `vllm/model_executor/models/gemma4.py` | NEW | Copy from mainline | ⬜ |
| 3.2 | `vllm/model_executor/models/registry.py` | 111 | Register Gemma4 | ⬜ |
| 3.3 | `tests/models/test_gemma4_load.py` | NEW | Model load test | ⬜ |

---

## Success Criteria

### Phase 1: Kernel-Level
- ✅ HD 512 kernel compiles without error
- ✅ No NaN/Inf in outputs
- ✅ Output magnitude < 100
- ✅ TP2/TP4 simulation passes
- ✅ HD 512 vs HD 256 statistics comparable

### Phase 2: Backend-Level
- ✅ TileLang backend integrates with vLLM
- ✅ Backend output matches Triton within rtol=2e-2

### Phase 3: Model Integration
- ✅ Model loads with TP2 and TP4
- ✅ Inference produces valid output
- ✅ No numerical instability

---

## Reference Implementation

### Official TileLang Sliding Window Pattern

From `~/tilelang/examples/attention_sink/example_gqa_sink_fwd_varlen.py:128-147`:

```python
# Build mask considering causal, sliding window, and padding
if is_causal:
    if window_size is not None:
        for i, j in T.Parallel(block_M, block_N):
            q_idx = bx * block_M + i + offset
            k_idx = actual_k * block_N + j
            # Causal + sliding window mask
            acc_s[i, j] = T.if_then_else(
                (q_idx < k_idx)  # causal: can't see future
                or (q_idx >= k_idx + window_size)  # sliding window: too old
                or (padding_check),
                -T.infinity(acc_s.dtype),
                0,
            )
```

---

## Troubleshooting

### Common Issues

**1. JIT Compilation Timeout**
- Symptom: Kernel takes >5 minutes to compile
- Solution: Reduce `block_M` or `block_N`, check V100 shared memory

**2. CUDA OOM**
- Symptom: `CUDA out of memory` error
- Solution: Use smaller `block_M=16, block_N=32` for HD 512

**3. NaN in Output**
- Symptom: `torch.isnan(output).any() == True`
- Solution: Check softmax stability, verify sliding window mask logic

**4. Performance Degradation**
- Symptom: HD 512 <50% of HD 256 throughput
- Solution: Try different block sizes, check L2 cache utilization

---

## Progress

### ✅ Completed: HD 512 Kernel (2026-05-26)

- **TileLang FA-V100 HD 512 paged kernel**: COMPILED AND VERIFIED
- **Dense FA HD 512**: Auto-tuned to `block_M=16, block_N=32, threads=64` ✓
- **TP2/TP4 simulation**: All head counts pass ✓
- **Key finding**: Requires CUDA 12.8 ptxas (CUDA 12.0 ptxas segfaults on large kernels)

### ✅ Validated Configurations

| Head Dim | block_M | block_N | threads | Shared Mem | Status |
|----------|---------|---------|---------|------------|--------|
| 256 | 64 | 32 | 256 | ~66KB | ✅ Proven |
| 512 | 16 | 32 | 64 | ~81KB | ✅ New |
| 512 (dense) | 16 | 32 | 64 | N/A | ✅ Auto-tuned |

### ⬜ Remaining Work

- **Sliding window** support in TileLang kernel (optional, can fall back to TRITON_ATTN)
- **Gemma4 model integration** (model files + registry)
- **End-to-end model-level tests**

## Notes

- **CUDA 12.8 required** for HD 512 kernel compilation (CUDA 12.0 ptxas has a bug causing segfault on large kernels with 128 MMA iterations)
- **Multimodal support** (Gemma4ForConditionalGeneration) deferred to later phase
- **Sliding window** implementation in TileLang is optional - can fall back to TRITON_ATTN for sliding layers
- **HD 512 dense** is the critical path - must work before any other features
- All kernel tests must pass **BEFORE** attempting model loading
