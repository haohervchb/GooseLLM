# vLLM SM70 Docker Quickstart

## Build

```bash
docker build \
  -f docker/Dockerfile.sm70-build \
  -t vllm-sm70:latest \
  .
```

## Run

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-sm70:latest \
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
  --compilation-config '{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}' \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

## References

- 1Cat-vLLM: https://github.com/1CatAI/1Cat-vLLM
- ai-bond flash-attention-v100: https://github.com/ai-bond/flash-attention-v100
