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

## Known Limitations

- **Spark/m8n8k4 paths**: Disabled by default (performance-negative).
- **GQA-shared-KV grid**: Implemented but disabled (performance-negative).
