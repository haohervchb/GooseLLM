# GooseLLM Agent Guide

## One-Line Build

```bash
pip install -e .        # Builds vLLM + SM70 FA2 kernels automatically
```

The SM70 FlashAttention kernel is built as a post-step during `pip install -e .`.
If it fails (e.g., CUDA not available during build), you can build it manually:

```bash
cd csrc/flash_attention_v100
python setup.py build_ext --inplace
```

**Note:** Requires `CUDA_HOME` pointing to CUDA 12.8+ (CUDA 12.0 `ptxas` segfaults on large SM70 kernels).
Auto-set via conda env activation in the `goosellm` environment.

## Kernel Code Location

- **Production kernels**: `csrc/flash_attention_v100/kernel/`
  - `fused_mha_forward.cu` — dense prefill
  - `fused_mha_paged_forward.cu` — paged prefill (block-table KV cache)
  - `fused_mha_backward.cu` — training backward pass
  - `fused_mha_api.cpp` — pybind11 entry points
- **Headers**: `csrc/flash_attention_v100/include/`
- **Python package**: `csrc/flash_attention_v100/flash_attn_v100/`
- **Tests**: `csrc/flash_attention_v100/test.py`

## Research Artifacts

- `research/spark-m8n8k4/03_wmma_spark.cuh` — SparkAttention pragmatic WMMA path (disabled)
- `research/spark-m8n8k4/04_mma_m8n8k4.cuh` — Raw PTX m8n8k4 backend (aborted)

## Backend Activation

The `FLASH_ATTN_V100` attention backend is selected automatically when:
- `--attention-backend FLASH_ATTN_V100` is passed
- `flash_attn_v100_cuda` module is importable
- Model config passes readiness checks (no alibi/softcap/sliding_window/fp8)

The `FLASH_ATTN_TILELANG_V100` attention backend is selected automatically when:
- `--attention-backend FLASH_ATTN_TILELANG_V100` is passed
- `tilelang_fa_v100` module is importable (from `3rdparty/tilelang-fa-v100/`)
- `tilelang` package is installed
- Model config passes the same readiness checks as FLASH_ATTN_V100
- On SM70, `FLASH_ATTN_TILELANG_V100` is the highest-priority backend

### TileLang FA-V100 Kernel Config

Kernels are JIT-compiled via TileLang. Config per head_dim:

| dim | block_M | block_N | threads | notes |
|-----|---------|---------|---------|-------|
| 64  | 32      | 128     | 256     |       |
| 128 | 32      | 128     | 256     |       |
| 256 | 64      | 32      | 256     | 1.3-1.5x faster than M32 N64 |
| 512 | 32      | 32      | 128     | Gemma4 full_attention; 4 warps with KV-union optimization |

## Branch Policy

- `main` — release branch (merge v100-fa2-paged-prefill work here)
- `sm70-decode-optimization` — decode prototype (preserved)
- `marlin-v100-integration` — Marlin AWQ work (preserved)
- `turboquant-v100-work` — TurboQuant work (preserved)

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `FLASH_ATTN_V100_DIR` | Override kernel source location |
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | `1stage` or `2stage` for decode |
| `NCCL_P2P_LEVEL=NVL` | Force NVLink for NCCL |
| `VLLM_DISABLE_PYNCCL=1` | Disable pynccl (not recommended) |
| `VLLM_USE_SM70_DECODE=0` | Disable SM70 decode kernel (default: enabled) |
| `VLLM_DEBUG_CHECK_NAN=1` | Enable NaN/Inf checks in model runner hot path (default: off) |
| `--disable-custom-all-reduce` | Disable custom AR (not recommended) |
| `CUDA_HOME` | Required for TileLang kernel compilation; use CUDA 12.8+ for HD 512 support. CUDA 12.0 `ptxas` segfaults on large SM70 kernels. Auto-set via conda env activation in the `goosellm` environment. |

## Known Limitations

- **Spark/m8n8k4 paths**: Disabled by default (performance-negative).
- **GQA-shared-KV grid**: Implemented but disabled (performance-negative).
- **Gemma4-31B (FP16)**: Supported (text-only, TP4 on 4× V100-32GB). TileLang FA-V100 backend active for prefill, Triton for decode. Concurrent requests work.
- **Gemma4-31B (quantized)**: AWQ/FP8 variants not yet tested.

## Serving Gemma4-31B

```bash
NCCL_P2P_LEVEL=NVL VLLM_CUSTOM_ALLREDUCE_ALGO=2stage FLA_USE_TILELANG=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31B \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --attention-backend FLASH_ATTN_TILELANG_V100 \
    --compilation-config '{"cudagraph_mode":"full_and_piecewise"}' \
    --chat-template examples/tool_chat_template_gemma4.jinja \
    --host 0.0.0.0 \
    --port 8082 \
    --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}'
```

### Flag Explanation

| Flag | Purpose |
|------|---------|
| `--attention-backend FLASH_ATTN_TILELANG_V100` | TileLang FlashAttention on V100 (paged prefill) |
| `--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'` | Text-only mode; skips vision/audio encoder loading |
| `--chat-template examples/tool_chat_template_gemma4.jinja` | Required - tokenizer has no default chat template |
| `--tensor-parallel-size 4` | Shards model across all 4 GPUs |
| `--compilation-config '{"cudagraph_mode":"full_and_piecewise"}'` | CUDA graph capture for decode |
| `--max-model-len 262144` | Full 256K context (adjust down for memory savings) |
