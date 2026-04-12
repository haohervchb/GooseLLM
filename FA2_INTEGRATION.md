# 1Cat-vLLM with FlashAttention-2 for V100 (SM70)

## Overview

This document details the modifications made to 1Cat-vLLM v0.0.2 to integrate FlashAttention-2 for NVIDIA V100 (SM70) GPUs, achieving ~50% improvement in prefill speed.

## Prerequisites

### System Requirements
- NVIDIA V100 GPU (SM70 architecture)
- CUDA Toolkit installed
- Python 3.10+ environment with PyTorch

### CUDA Version Compatibility

**IMPORTANT:** flash-attention-v100 requires CUDA version compatibility between:
- PyTorch's built-in CUDA version
- System CUDA toolkit (nvcc)
- The CUDA_HOME used during build

Check your current setup:
```bash
# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# Check system nvcc version
nvcc --version

# Check CUDA_HOME
echo $CUDA_HOME
```

---

## Complete Installation Guide

### Step 1: Check Available CUDA Versions

```bash
# List available CUDA installations
ls -la /usr/local/ | grep cuda

# Check CUDA 12.8 availability (recommended)
cat /usr/local/cuda-12.8/version.json
```

The system should have CUDA 12.8 installed at `/usr/local/cuda-12.8`.

### Step 2: Clone flash-attention-v100 Repository

```bash
cd /home/rah
git clone https://github.com/ai-bond/flash-attention-v100.git
cd flash-attention-v100
```

### Step 3: Verify PyTorch Installation

```bash
# Activate your 1Cat-vLLM environment
conda activate 1Cat-vLLM-0.0.2

# Verify PyTorch and CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Example output:
# PyTorch: 2.9.1+cu128
# CUDA: 12.8
```

### Step 4: Build flash-attention-v100 from Source

**Critical:** Set `CUDA_HOME` to match your PyTorch's CUDA version:

```bash
# Activate environment
conda activate 1Cat-vLLM-0.0.2

# Set CUDA_HOME to match PyTorch's built-in CUDA
export CUDA_HOME=/usr/local/cuda-12.8

# Navigate to cloned repo
cd /home/rah/flash-attention-v100

# Build and install (--no-build-isolation is critical!)
pip install . --no-build-isolation

# Or for editable install:
# pip install -e . --no-build-isolation
```

#### Build Options

**Option A: Wheel Installation (Recommended for Deployment)**
```bash
pip install . --no-build-isolation
```

**Option B: Editable Installation (Recommended for Development)**
```bash
pip install -e . --no-build-isolation
```

**Option C: Install from Wheel File (if pre-built available)**
```bash
pip install flash-attention-v100-26.2-cp312-cp312-linux_x86_64.whl
```

### Step 5: Verify Installation

```bash
# Activate environment
conda activate 1Cat-vLLM-0.0.2

# Test import
python -c "from flash_attn_v100 import flash_attn_func; print('flash_attn_v100 installed successfully')"

# Test basic functionality
python -c "
import torch
from flash_attn_v100 import flash_attn_func

# Create test tensors
B, N, H, D = 2, 128, 4, 64
q = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, N, H, D, device='cuda', dtype=torch.float16)

# Run FA2
out = flash_attn_func(q, k, v, causal=True)
print(f'Success! Output shape: {out.shape}')
"
```

### Step 6: Copy Modified Backend to Installed vLLM

The modified `flash_attn_v100.py` is located at:
```
vllm_src_backup/v1/attention/backends/flash_attn_v100.py
```

**For Development (using vllm_src_backup):**
No additional steps needed if using the source from `vllm_src_backup/`.

**For Installed vLLM Package:**
```bash
# Activate environment
conda activate 1Cat-vLLM-0.0.2

# Copy modified backend to installed location
cp /home/rah/1Cat-vLLM/vllm_src_backup/v1/attention/backends/flash_attn_v100.py \
   $CONDA_PREFIX/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn_v100.py

# Verify
python -c "
from vllm.v1.attention.backends.flash_attn_v100 import _get_flash_ops
fa_func, fa_decode = _get_flash_ops()
print(f'flash_attn_func: {fa_func is not None}')
print(f'flash_attn_decode_paged: {fa_decode is not None}')
"
```

---

## Modifications Made to Source Code

### File: `vllm_src_backup/v1/attention/backends/flash_attn_v100.py`

#### Change 1: Updated Module Documentation

```python
# Added Requirements and Limitations to docstring
"""Flash Attention V100 backend for SM70.

Prefill uses the dense Flash V100 kernel (from ai-bond/flash-attention-v100 fork).
Decode uses flash_attn_decode_paged if available, otherwise falls back to Triton.

Requirements:
- flash_attn_v100 package (from ai-bond fork: pip install flash-attention-v100)
- For decode acceleration: flash_attn_decode_paged (optional)
- For paged KV cache extraction: paged_kv_utils (optional)

Limitations:
- Does NOT support FP8 KV cache
"""
```

#### Change 2: Modified `_get_flash_ops()` Function

**Before (original 1Cat-vLLM):**
```python
def _get_flash_ops():
    global _flash_attn_func, _flash_attn_decode_paged
    if _flash_attn_func is None or _flash_attn_decode_paged is None:
        try:
            from flash_attn_v100 import flash_attn_decode_paged, flash_attn_func
            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
    return _flash_attn_func, _flash_attn_decode_paged
```

**After (modified):**
```python
def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available.
    
    Modified to only require flash_attn_func (from ai-bond fork).
    flash_attn_decode_paged is optional - falls back to Triton if unavailable.
    """
    global _flash_attn_func, _flash_attn_decode_paged
    if _flash_attn_func is None:
        try:
            from flash_attn_v100 import flash_attn_func
            _flash_attn_func = flash_attn_func
        except ImportError:
            _flash_attn_func = None
    
    # flash_attn_decode_paged is optional - try to load but don't require
    if _flash_attn_decode_paged is None:
        try:
            from flash_attn_v100 import flash_attn_decode_paged
            _flash_attn_decode_paged = flash_attn_decode_paged
        except (ImportError, AttributeError):
            _flash_attn_decode_paged = None
            
    return _flash_attn_func, _flash_attn_decode_paged
```

**Why:** The ai-bond fork only provides `flash_attn_func`, not `flash_attn_decode_paged`. Decode falls back to Triton.

#### Change 3: Modified `_flash_v100_prefill()` Function

Added GQA/MQA support by expanding K/V heads:

```python
def _flash_v100_prefill(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: TritonAttentionMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Prefill path for no-prefix case (query_len == seq_len per sequence).
    
    Supports GQA/MQA by expanding K/V heads to match Q heads.
    """
    # ... existing code trimming ...
    
    # Check for GQA/MQA - need to expand K/V heads
    num_heads_q = query.shape[1]  # [tokens, num_heads_q, head_dim]
    num_heads_kv = key.shape[1]   # [tokens, num_kv_heads, head_dim]
    needs_expansion = num_heads_q != num_heads_kv

    for i in range(num_seqs):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        if end <= start:
            continue

        q_seq = query[start:end].unsqueeze(0)  # [1, seq, num_heads_q, head_dim]
        k_seq = key[start:end].unsqueeze(0)    # [1, seq, num_kv_heads, head_dim]
        v_seq = value[start:end].unsqueeze(0)  # [1, seq, num_kv_heads, head_dim]

        # Expand K/V to match Q heads for GQA/MQA
        if needs_expansion:
            repeat_factor = num_heads_q // num_heads_kv
            k_seq = k_seq.repeat_interleave(repeat_factor, dim=2)
            v_seq = v_seq.repeat_interleave(repeat_factor, dim=2)

        out_seq = self.flash_attn_func(
            q_seq,
            k_seq,
            v_seq,
            causal=True,
            softmax_scale=self.scale,
        )
        out_view[start:end].copy_(out_seq.squeeze(0))

    return output
```

**Why:** The ai-bond fork doesn't natively support GQA/MQA. We expand K/V heads to match Q heads before calling FA2.

---

## Usage

### Recommended Parameters

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model QuantTrio/Qwen3.5-122B-A10B-AWQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 262144 \
  --tensor-parallel-size 4 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --skip-mm-profiling \
  --attention-backend FLASH_ATTN_V100 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 \
  --port 8082 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Key Parameters for Prefill Performance

| Parameter | Recommended Value | Purpose |
|-----------|------------------|---------|
| `max-num-batched-tokens` | 8192-16384 | Batch more tokens for FA2 efficiency |
| `max-num-seqs` | 1-4 | Concurrency; 1 is optimal for pure prefill |
| `attention-backend` | FLASH_ATTN_V100 | Enable FA2 for prefill |

---

## Performance Results

### Prefill Speed Improvement
- **Before:** ~3000 tokens/s
- **After:** ~4500-5000 tokens/s
- **Improvement:** ~50%

### Model Tested
- Qwen3.5-122B-A10B-AWQ (GQA: 8 Q heads, 1 KV head)

---

## Known Limitations

1. **Decode Performance:** Decode falls back to Triton (no `flash_attn_decode_paged` kernel)
2. **FP8 KV Cache:** Not supported with this backend
3. **High `max-num-batched-tokens`:** May hurt decode performance with concurrent requests
4. **GQA/MQA:** Supported via K/V expansion (adds overhead)

---

## Troubleshooting

### Error: "n_heads mismatch: q has X, k has 1"

This error occurs when using GQA/MQA models. **Fixed** in the modified backend - it now expands K/V heads automatically.

### Error: "CUDA version mismatch"

```
RuntimeError: CUDA version mismatch. Detected X, PyTorch built with Y
```

**Solution:** Set `CUDA_HOME` to match PyTorch's CUDA version:
```bash
export CUDA_HOME=/usr/local/cuda-12.8  # or appropriate version
pip install . --no-build-isolation
```

### Error: "flash_attn_v100 module not found"

```bash
# Verify installation
pip show flash-attn-v100

# If not installed, install it
pip install /path/to/flash-attention-v100 --no-build-isolation
```

### Error: "FLASH_ATTN_V100 fallback to Triton"

**This is expected** if `flash_attn_decode_paged` is unavailable. The backend will use Triton for decode.

To check which paths are active:
```bash
# Look for these log messages:
# "FLASH_ATTN_V100 prefill path active" - FA2 being used for prefill
# "FLASH_ATTN_V100 decode fallback to Triton" - Using Triton for decode
```

---

## Future Improvements

1. **Implement `flash_attn_decode_paged`:** Currently unavailable in ai-bond fork
2. **Optimize GQA Expansion:** Avoid repeated expansion for decode
3. **Fused Rotary Embedding:** Combine with FA2 to reduce kernel launches
4. **Better Decode Path:** Separate attention backend for decode vs prefill

---

## Credits

- **FlashAttention-2 for V100:** https://github.com/ai-bond/flash-attention-v100 by D.Skryabin
- **Original FlashAttention:** Tri Dao et al.
- **1Cat-vLLM:** https://github.com/1bitasia/1Cat-vLLM

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.0.2 | Original | 1Cat-vLLM base with TurboMind AWQ support |
| 0.0.2+FA2 | Current | Added FlashAttention-2 for V100 prefill |
