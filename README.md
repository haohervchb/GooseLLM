# GooseLLM — vLLM for NVIDIA V100 (SM70)

High-throughput LLM inference on Tesla V100 GPUs with custom FlashAttention-2 kernel.

## Quick Start

### Local Build (One-Line)

```bash
git clone https://github.com/haohervchb/GooseLLM.git
cd GooseLLM

# Build vLLM + SM70 kernel (uses all CPU threads automatically)
MAX_JOBS=$(nproc) NVCC_THREADS=4 pip install -e . --no-build-isolation
```

If vLLM's CMake build fails (optional extensions), the kernel is still usable:

```bash
# Build only the SM70 kernel
cd csrc/flash_attention_v100
MAX_JOBS=$(nproc) python setup.py build_ext --inplace
cd ../..

# Install vLLM Python package without C++ extensions
pip install -e . --no-build-isolation
```

### Docker Build

```bash
docker build \
  -f docker/Dockerfile.sm70-build \
  -t goosellm:sm70 \
  .
```

**Override CPU/thread count at build time:**

```bash
docker build \
  -f docker/Dockerfile.sm70-build \
  --build-arg MAX_JOBS=16 \
  --build-arg NVCC_THREADS=4 \
  -t goosellm:sm70 \
  .
```

### Run Server

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_CUSTOM_ALLREDUCE_ALGO=1stage \
  -e NCCL_P2P_LEVEL=NVL \
  goosellm:sm70 \
  python -m vllm.entrypoints.openai.api_server \
    --model QuantTrio/Qwen3.5-122B-A10B-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --tensor-parallel-size 4 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 16384 \
    --skip-mm-profiling \
    --attention-backend FLASH_ATTN_V100 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
    --host 0.0.0.0 \
    --port 8000
```

## Performance Tuning

| Environment Variable | Recommended Value | Effect |
|---------------------|-------------------|--------|
| `VLLM_CUSTOM_ALLREDUCE_ALGO` | `1stage` | Faster decode (direct P2P) |
| `NCCL_P2P_LEVEL` | `NVL` | Force NVLink over PCIe |
| `NCCL_MIN_NCHANNELS` | `4` | Better NCCL throughput |
| `NCCL_MAX_NCHANNELS` | `4` | (if not using batch invariance) |

**Do NOT use:**
- `--disable-custom-all-reduce` (disables optimized P2P, hurts decode)
- `VLLM_DISABLE_PYNCCL=1` (unnecessary with custom AR)

## Model Support

| Model | Config | Status |
|-------|--------|--------|
| Qwen3.5-122B-A10B-AWQ | TP=4, D=256 | ✅ Production |
| Qwen3.6-35B-A3B-AWQ | TP=4, D=256 | ✅ Production |
| Qwen3.6-27B (dense) | TP=4, D=256 | ✅ Expected |

## Architecture

- **Kernel**: `csrc/flash_attention_v100/` — FlashAttention-2 paged prefill for SM70
- **Backend**: `vllm/v1/attention/backends/flash_attn_v100.py` — vLLM integration
- **Build**: `setup.py` auto-discovers in-tree kernel; Docker uses all CPU threads

## References

- Original V100 kernel research: [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- Upstream vLLM: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
