CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 \
  --port 8082 \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 15}' &

SERVER_PID=$!
sleep 60
python test_client.py
kill $SERVER_PID
