# GooseLLM — vLLM for NVIDIA V100 (SM70)

High-throughput LLM inference on Tesla V100 GPUs with custom FlashAttention-2 kernel.

## Quick Start

### Docker Build

```bash
docker build \
  -f docker/Dockerfile.sm70-build \
  -t goosellm:sm70 \
  .
```

### Local Build

```bash
# 1. Create and activate environment
conda create -n goosellm python=3.12 -y
conda activate goosellm

# 2. Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/cuda.txt
python -m pip install 'setuptools>=77.0.3,<81.0.0' 'setuptools_scm>=8' grpcio-tools cmake build

# 3. Set build environment
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=12.8
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=$(nproc)
export NVCC_THREADS=4

# 4. Build SM70 kernel
cd csrc/flash_attention_v100
sed -i 's/if not torch.cuda.is_available():/if False: # if not torch.cuda.is_available():/' setup.py
python setup.py build_ext --inplace
cd ../..

# 5. Build vLLM wheel (matches 1Cat's original process)
rm -rf build vllm.egg-info .deps/*-build .deps/*-subbuild
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.3.dev0 \
  python -m build --wheel --no-isolation --outdir dist-cu128-sm70

# 6. Install
python -m pip install dist-cu128-sm70/*.whl --no-deps
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

**Do NOT use:**
- `--disable-custom-all-reduce` (disables optimized P2P)
- `VLLM_DISABLE_PYNCCL=1` (unnecessary with custom AR)

## Model Support

| Model | Config | Status |
|-------|--------|--------|
| Qwen3.5-122B-A10B-AWQ | TP=4, D=256 | ✅ Production |
| Qwen3.6-35B-A3B-AWQ | TP=4, D=256 | ✅ Production |

## References

- Original V100 kernel research: [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- Upstream vLLM: [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
