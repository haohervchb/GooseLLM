export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --gpu-memory-utilization 0.50 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --port 8083 \
  --speculative-config '{"method": "dflash", "draft_model_config_override": {"architectures": ["DFlashDraftModel"]}, "model": "Qwen/Qwen2.5-0.5B", "num_speculative_tokens": 15}' \
  > server.log 2>&1 &
echo $! > server.pid
