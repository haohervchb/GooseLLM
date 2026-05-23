# GooseLLM Request Lifecycle — Absolute Flowmap

## Startup (One-Time)

| Step | File | Lines | What Happens |
|------|------|-------|-------------|
| Arg parse | `vllm/entrypoints/openai/api_server.py` | 493-505 | `FlexibleArgumentParser` parses all CLI flags |
| Async engine build | `vllm/entrypoints/openai/api_server.py` | 450-453 | `AsyncLLM.from_vllm_config()` creates the V1 engine |
| AsyncLLM init | `vllm/v1/engine/async_llm.py` | 67-198 | Creates `InputProcessor`, `OutputProcessor`, spawns `EngineCore` process |
| Tokenizer load | `vllm/renderers/registry.py` | 74-88 | Loads HuggingFace tokenizer via `renderer_from_config()` |
| EngineCore proc spawn | `vllm/v1/engine/core.py` | 737-746 | MPCore starts a new process with ZMQ sockets for IPC |
| Route registration | `vllm/entrypoints/openai/chat_completion/api_router.py` | 34-46, 107-108 | `POST /v1/chat/completions` → `create_chat_completion` |
| Model + kernels init | `vllm/v1/worker/gpu_model_runner.py` | 3433 | GPUWorker loads model weights, JIT-compiles TileLang FA kernels |
| Attention backend select | `vllm/v1/attention/selector.py` | 46-86 | `FLASH_ATTN_TILELANG_V100` resolved (SM70 priority 0, see `vllm/platforms/cuda.py:76-82`) |
| Uvicorn start | `vllm/entrypoints/launcher.py` | 72-83 | HTTP server starts listening on port 8082 |

---

## PHASE 1: HTTP Request → Tokenization

### 1.1 Request Arrives

```
POST /v1/chat/completions
    │
    ▼ vllm/entrypoints/openai/chat_completion/api_router.py:46-58
create_chat_completion(request, raw_request)
    │
    ▼  Middleware: @load_aware_call (utils.py:120), @with_cancellation (utils.py:71)
    │
    ▼ vllm/entrypoints/openai/chat_completion/serving.py:321
OpenAIServingChat.create_chat_completion()
```

### 1.2 Chat Template Rendering + Tokenization

```
vllm/entrypoints/openai/chat_completion/serving.py:333
    │ render_chat_request(request)
    │
    ├── vllm/entrypoints/openai/engine/serving.py:985
    │   _preprocess_chat()
    │   │
    │   ├── :1012  renderer.render_messages_async(messages)
    │   │         → Applies Jinja2 chat template
    │   │         → Converts [{"role":"user","content":"hello"}]
    │   │           to "<|im_start|>user\nhello<|im_end|>..."
    │   │
    │   └── :1015  renderer.tokenize_prompt_async(prompt)
    │             → HuggingFace tokenizer.encode()
    │             → Returns list[int] token IDs
    │
    ▼ serving.py:339-363
request_id = "chatcmpl-{uuid}"
```

### 1.3 Build EngineCoreRequest

```
serving.py:372-429
    │
    ├── sampling_params = request.to_sampling_params(...)    // temp, top_p, max_tokens
    │
    └── engine_request = self.input_processor.process_inputs(...)
        │  vllm/v1/engine/input_processor.py:490
        │  → validates token IDs, structured output, LoRA
        │  → vllm/inputs/preprocess.py:331 — passthrough (tokens already exist)
        │
        ▼  Returns EngineCoreRequest{prompt_token_ids, sampling_params, ...}
```

---

## PHASE 2: Engine Client → Scheduler

### 2.1 AsyncLLM.generate()

```
serving.py:431-441
generator = self.engine_client.generate(engine_request, ...)
    │
    ▼ vllm/v1/engine/async_llm.py:518
AsyncLLM.generate()
    │
    ├── :548  q = await self.add_request(request_id, prompt, params)
    │   │
    │   ├── :367  _run_output_handler()  →  starts background asyncio task
    │   ├── :374  queue = RequestOutputCollector()  →  per-request asyncio queue
    │   └── :380  await self._add_request(request, prompt, None, 0, queue)
    │       │
    │       ├── :407  output_processor.add_request(request, queue)
    │       └── :410  await engine_core.add_request_async(request)
    │           │
    │           ▼ vllm/v1/engine/core_client.py:966-969
    │           AsyncMPClient.add_request_async()
    │           → ZMQ PUSH socket → EngineCore process
    │
    └── :562-573  Consumer loop:
        while not finished:
            out = queue.get()  →  yields RequestOutput
```

### 2.2 EngineCore Receives Request

```
EngineCore process (separate process)
    │
    ▼ vllm/v1/engine/core.py:978-986
run_busy_loop()
    │
    ├── :984  _process_input_queue()
    │   → ZMQ PULL → handle_client_request()
    │   → EngineCoreRequestType.ADD → :1040
    │
    └── :1040-1042  self.add_request(req)
        │
        ▼ :313  self.scheduler.add_request(request)
```

### 2.3 Scheduler: Waiting Queue

```
vllm/v1/core/sched/scheduler.py:1635-1655
Scheduler.add_request()
    │
    ├── :1652  self.waiting.add_request(request)
    │   → FCFS: deque.append()  (request_queue.py:78)
    │   → or PRIORITY: heapq.heappush()  (request_queue.py:144)
    │
    └── :1653  self.requests[req_id] = request
```

### 2.4 Scheduler: schedule() — Waiting → Running

```
vllm/v1/core/sched/scheduler.py:315-839
Scheduler.schedule()
    │
    ├── PHASE 1: Scheduled running requests (:344-511)
    │   For each running req:
    │   ├── Compute num_new_tokens = num_tokens_with_spec - num_computed_tokens
    │   ├── kv_cache_manager.allocate_slots(request, num_new_tokens)  (:422)
    │   │   → Allocates KV cache blocks (or preempts if out of memory)
    │   └── Add to scheduled_running_reqs, deduct token_budget
    │
    ├── PHASE 2: Schedule waiting (NEW) requests (:527-781)
    │   For each waiting request:
    │   ├── :565  Skip if WAITING_FOR_STREAMING_REQ
    │   ├── :591  get_computed_blocks() → detect prefix cache hits
    │   ├── :701  kv_cache_manager.allocate_slots(request, num_new_tokens)
    │   │   → This is where YOUR PROMPT gets KV cache blocks!
    │   │   → If allocation fails → request stays waiting (OOM)
    │   ├── :734  waiting.pop_request() → remove from waiting
    │   ├── :744  running.append(request) → add to running
    │   └── :763  request.status = RUNNING
    │
    └── PHASE 3: Assemble SchedulerOutput (:798-839)
        → Contains: scheduled_new_reqs, scheduled_running_reqs,
          block_tables, num_scheduled_tokens, etc.
```

---

## PHASE 3: Model Forward Pass (The GPU Work)

### 3.1 SchedulerOutput → GPU Inputs

```
vllm/v1/engine/core.py:380-381
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
    │
    ▼ vllm/v1/executor/abstract.py:46-85 (executor dispatch)
    │   For multi-GPU mp: MultiprocExecutor broadcasts to workers
    │   For single-GPU: UniProcExecutor calls GPUWorker directly
    │
    ▼ vllm/v1/worker/gpu_model_runner.py:3433
GPUModelRunner.execute_model()
    │
    ├── :3462  _update_states()         → update persistent batch state
    │   ├── :964   Remove finished requests from input_batch
    │   └── :1005  Add new requests as CachedRequestState
    │
    ├── :3504  _prepare_inputs()        → build GPU tensors
    │   ├── :1562  block_table.commit_block_table(num_reqs)
    │   │          → Copies block tables to GPU
    │   ├── :1566  Build req_indices, positions_np
    │   │          positions[i] = num_computed_tokens[req_i] + token_offset_i
    │   ├── :1593  Gather input_ids from token_ids_cpu via index_select
    │   └── :1655  block_table.compute_slot_mapping(req_indices, positions)
    │              → Maps each token position to KV cache slot
    │
    ├── :3600  _build_attention_metadata()
    │   → CommonAttentionMetadata{
    │       query_start_loc,      // cumulative token counts (e.g., [0, 3000, 3100, 3200])
    │       seq_lens,             // total seq lengths per request
    │       block_table,          // [num_reqs, max_blocks] block IDs
    │       max_query_len         // >1 for prefill, ==1 for decode
    │     }
    │
    ├── :3623  _preprocess()
    │   → Returns: (input_ids, None, positions, intermediate_tensors, {})

    └── :3659  model_output = self._model_forward(input_ids, positions, ...)
```

### 3.2 The Transformer Stack

```
vllm/v1/worker/gpu_model_runner.py:3144-3174
_model_forward()
    → self.model(input_ids, positions, intermediate_tensors, None)
    │
    ▼ vllm/model_executor/models/qwen3_moe.py:772
Qwen3MoeForCausalLM.forward()
    → self.model(input_ids, positions, ...)   // Qwen3MoeModel
    │
    ▼ :471  Qwen3MoeModel.forward()
    │
    ├── :468  self.embed_input_ids()   → VocabParallelEmbedding
    │         input_ids → hidden_states [num_tokens, hidden_dim]
    │         (With TP=4, each GPU holds 1/4 of vocab → all-reduce after)
    │
    └── :490  FOR layer in self.layers (e.g., 48 layers for 35B):
            │
            ▼ :406  Qwen3MoeDecoderLayer.forward()
            │
            ├── :415   hidden_states = self.input_layernorm(hidden_states)
            │
            ├── :416   hidden_states = self.self_attn(positions, hidden_states)
            │          │
            │          │  === SELF-ATTENTION (details below) ===
            │          │
            │          ▼
            │
            ├── :424   hidden_states, residual = self.post_attention_layernorm(...)
            │
            └── :425   hidden_states = self.mlp(hidden_states)
                       │
                       │  === MoE or Dense MLP (details below) ===
                       │
                       ▼
```

### 3.3 ATTENTION: FLASH_ATTN_TILELANG_V100 Path

```
Qwen3MoeAttention.forward()  →  vllm/model_executor/models/qwen3_moe.py:333-351
    │
    ├── Q = self.q_proj(hidden_states)   → [tokens, num_heads * head_size]
    ├── K = self.k_proj(hidden_states)   → [tokens, num_kv_heads * head_size]
    ├── V = self.v_proj(hidden_states)   → [tokens, num_kv_heads * head_size]
    │   (With TP=4: heads sharded across GPUs, each gets num_heads/4)
    │
    └── self.attn(q, k, v)   → vllm/model_executor/layers/attention/attention.py:384
        │
        ├── :431  query = query.view(-1, num_heads, head_size)
        ├── :434  key = key.view(-1, num_kv_heads, head_size)
        └── :436  value = value.view(-1, num_kv_heads, head_size)

        IF use_direct_call (preferred path):
        ├── :440  unified_kv_cache_update(key, value, layer_name)
        │         → stores K,V into paged KV cache
        │
        └── :443  unified_attention_with_output(query, key, value, output, layer_name)
            │
            ▼ :683-710  attention.py
            attn_metadata, self, kv_cache = get_attention_context(layer_name)
            self.impl.forward(query, key, value, kv_cache, attn_metadata, output)
                │
                ▼  vllm/v1/attention/backends/flash_attn_tilelang_v100.py:219
                FlashAttnTileLangV100Impl.forward()
                │
                ├── is_prefill = attn_metadata.max_query_len > 1
                │
                ├── IF PREFILL (your 16K-token prompt):
                │   │
                │   └── :100-185  _tilelang_paged_prefill(query, k_cache, v_cache, ...)
                │       │
                │       ├── :107  Unbind KV cache: k_cache, v_cache
                │       ├── :115-120  Compute prefix_kv_lens per request
                │       │   (handles cached prefixes + new tokens)
                │       │
                │       ├── :128-143  Block size normalization
                │       │   If actual_block_size != 16:
                │       │   → pad/reshape K,V and block_table to align with TileLang kernel
                │       │
                │       └── :145  self.tilelang_paged(query, k_cache, v_cache,
                │                        block_table, seq_lens, query_start_loc,
                │                        prefix_kv_lens, ...)
                │           │
                │           ▼  3rdparty/tilelang-fa-v100/tilelang_fa_v100/
                │           tilelang_paged_forward()
                │           │
                │           │  JIT-compiled TileLang kernel:
                │           │  ┌─────────────────────────────────────────┐
                │           │ │ head_dim=256: block_M=64, block_N=32     │
                │           │ │ head_dim=128: block_M=32, block_N=128    │
                │           │ │ head_dim=64:  block_M=32, block_N=128    │
                │           │ │ 256 threads, SM70 MMA instructions       │
                │           │ └─────────────────────────────────────────┘
                │           │
                │           │  Kernel does:
                │           │  1. For each tile: load Q block, K block from paged cache
                │           │  2. S = Q @ K^T * scale       (MMA m16n8k4)
                │           │  3. causal mask + softmax     (online softmax)
                │           │  4. O += P @ V                (MMA m16n8k4)
                │           │  5. Write output to output tensor
                │           │
                │           └── Returns: output [num_tokens, num_heads, head_size]
                │
                └── IF DECODE (max_query_len == 1, subsequent tokens):
                    │
                    └── :260-261  Falls back to Triton backend
                        super().forward(...)   → TritonAttentionImpl
                        └── Because TileLang MMA needs block_M >= 16,
                            a single decode token wastes 15/16 of MMA units
        │
        └── :522  output = self.o_proj(output)
                  → Linear projection back to hidden_dim
```

### 3.4 MoE PATH (Qwen3 A3B with Expert Parallel)

```
Qwen3MoeDecoderLayer.forward()  →  vllm/model_executor/models/qwen3_moe.py:425
    │  self.mlp(hidden_states)
    │
    ▼  FOR MoE layers: Qwen3MoeSparseMoeBlock.forward()  (:217-248)
    │  FOR dense layers: Qwen3MoeMLP (standard SwiGLU)
    │
    ├── :225  IF sequence_parallel:
    │         hidden_states = sequence_parallel_chunk(hidden_states)
    │
    ├── :229  router_logits, _ = self.gate(hidden_states)
    │         → ReplicatedLinear(hidden_dim, num_experts=128)
    │         → Every GPU has full routing scores
    │         → Shape: [num_tokens, 128]
    │
    ├── :230-231  shared_out, fused_out = self.experts(...)
    │   │
    │   ▼  vllm/model_executor/layers/fused_moe/shared_fused_moe.py:81
    │   SharedFusedMoE.forward()
    │     → FusedMoE.forward()  →  CustomOp.forward()
    │     → forward_cuda() → forward_native()  (layer.py:1589)
    │     → torch.ops.vllm.moe_forward_shared(...)
    │     → forward_impl()  (layer.py:1814)
    │     │
    │     ├── :1862-1898  EP DISPATCH (All2All):
    │     │   dispatch_res = get_ep_group().dispatch_router_logits(
    │     │       hidden_states, router_logits, is_sequence_parallel, ...)
    │     │   │
    │     │   ▼  vllm/distributed/parallel_state.py:1003
    │     │   │  → device_communicator.dispatch_router_logits()
    │     │   │
    │     │   ▼  vllm/distributed/device_communicators/all2all.py:145
    │     │   │  AgRsAll2AllManager.dispatch_router_logits()
    │     │   │  → All-gather: gathers hidden_states + router_logits
    │     │   │    from all EP ranks onto each rank
    │     │   │  → Each GPU now sees all tokens' hidden states + routing scores
    │     │
    │     ├── :1936-1939  ROUTER SELECTION:
    │     │   topk_weights, topk_ids = self.router.select_experts(
    │     │       hidden_states=x_orig, router_logits=router_logits)
    │     │   │
    │     │   ▼  vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:247
    │     │   │  GroupedTopKRouter.select_experts()
    │     │   │  → Top-8 experts selected using Qwen-style grouped routing
    │     │   │  → Returns: topk_weights [num_tokens, 8], topk_ids [num_tokens, 8]
    │     │
    │     ├── :1941-1946  EXPERT COMPUTATION (fused GEMM):
    │     │   final_hidden_states = self.quant_method.apply(
    │     │       layer=self, x=x, topk_weights=topk_weights, topk_ids=topk_ids)
    │     │   │
    │     │   │  → With EP: each GPU has 128/4 = 32 experts locally
    │     │   │  → Only the experts hosted on this GPU are computed
    │     │   │  → Fused grouped GEMM: batches tokens by target expert,
    │     │   │    then runs batched matrix multiplies
    │     │   │  → Weighted sum: output = Σ(topk_weights[i] * expert_i(x))
    │     │
    │     ├── :1948-1966  SHARED EXPERTS (if present):
    │     │   → Runs shared experts on separate CUDA stream
    │     │   → shared_out = shared_expert(x)
    │     │   → combined with fused_out
    │     │
    │     └── :1968-1978  EP COMBINE (Reduce-Scatter):
    │         states = get_ep_group().combine(states, is_sequence_parallel)
    │         │
    │         ▼  vllm/distributed/device_communicators/all2all.py:115
    │         │  AgRsAll2AllManager.combine()
    │         │  → Reduce-scatter: each GPU gathers its experts' partial sums
    │         │    from all EP ranks, sums them, redistributes
    │         │  → Result: correct full output distributed back across GPUs
    │
    ├── :233-234  IF shared_out: final = shared_out + fused_out
    │
    └── :237-245  TP ALL-REDUCE (if TP > 1):
        IF sequence_parallel:
            tensor_model_parallel_all_gather(final_hidden_states, 0)
        ELSE:
            self.experts.maybe_all_reduce_tensor_model_parallel(...)

Output: hidden_states [num_tokens, hidden_dim]
```

### 3.5 Dense MLP (non-MoE layers)

```
Qwen3MoeMLP.forward()
    └── Standard SwiGLU:
        gate = self.gate_proj(x)                   // [tokens, intermediate_size]
        up   = self.up_proj(x)                     // [tokens, intermediate_size]
        hidden = SiLU(gate) * up
        output = self.down_proj(hidden)            // [tokens, hidden_dim]
```

### 3.6 After Final Layer: Compute Logits

```
Back in Qwen3MoeModel.forward()  (:502-508)
    │  After all layers: final norm
    │
    ▼  vllm/v1/worker/gpu_model_runner.py:3706
if not pooling model:
    sample_hidden_states = hidden_states[logits_indices]  // only last token per request
    logits = self.model.compute_logits(sample_hidden_states)
        │
        ▼  vllm/model_executor/models/qwen3_moe.py:784
        logits = self.logits_processor(self.lm_head, hidden_states)
            → self.lm_head = ParallelLMHead  (vocab parallel)
            → Output: [num_reqs, vocab_size]
```

---

## PHASE 4: Sampling & Post-Processing

### 4.1 Sample Tokens on GPU

```
vllm/v1/engine/core.py:387-389
model_output = future.result()
IF model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
        │
        ▼  vllm/v1/worker/gpu_model_runner.py:3754
        sample_tokens()
        │
        ├── :3793  apply_grammar_bitmask()  → structured output constraints
        │
        ├── :3799  self._sample(logits, spec_decode_metadata)
        │   │
        │   ▼  :2972  _sample()
        │   → self.sampler(logits, sampling_metadata)
        │   │  vllm/v1/worker/gpu/sample/sampler.py:57
        │   │  → Apply frequency/presence penalties
        │   │  → Apply temperature scaling
        │   │  → Apply top-k + top-p filtering
        │   │  → Gumbel-max sampling → sampled_token_ids [num_reqs]
        │
        ├── :3801  _update_states_after_model_execute()
        │   → Stores sampled tokens in input_batch for next step
        │
        └── :3875-3890  _bookkeeping_sync()
            → GPU tensor → CPU Python lists
            → Logprobs extraction
            → NaN detection
            │
            ▼  :3908-3920  Build ModelRunnerOutput:
            ModelRunnerOutput{
                req_ids,
                req_id_to_index,
                sampled_token_ids,      // list[list[int]]
                logprobs,
                prompt_logprobs_dict,
            }
```

### 4.2 Scheduler Updates Request State

```
vllm/v1/engine/core.py:394-395
scheduler.update_from_output(scheduler_output, model_output)
    │
    ▼  vllm/v1/core/sched/scheduler.py:1232-1485
    │
    ├── :1274  For each request with scheduled tokens:
    │   ├── :1291  generated_token_ids = sampled_token_ids[req_index]
    │   ├── :1329  _update_request_with_output(request, new_token_ids)
    │   │   → Appends tokens to request.output_token_ids
    │   │   → Checks stop conditions (EOS token, max_tokens, stop strings)
    │   │   → Returns: (filtered_token_ids, stopped: bool)
    │   │
    │   └── :1386-1404  Create EngineCoreOutput:
    │       EngineCoreOutput{
    │           request_id,
    │           new_token_ids,
    │           finish_reason,           // "stop", "length", "abort", or None
    │           new_logprobs,
    │           events,
    │       }
    │
    └── :1454-1457  Group by client_index:
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=[...])}
```

---

## PHASE 5: Output → HTTP Response (Return Path)

### 5.1 EngineCore → Frontend (via ZMQ)

```
vllm/v1/engine/core.py:984-1024
_process_engine_step()
    → outputs, model_executed = self.step_fn()             (:1019)
    → for each output: self.output_queue.put_nowait(output)  (:1021)
        │
        ▼  Background ZMQ output thread: core.py:1179
        process_output_sockets()
        → MsgpackEncoder.encode_into(EngineCoreOutputs)    (:1238)
        → sockets[client_index].send_multipart(buffers)     (:1241)
        │  (Zero-copy: tensor/Numpy buffers preserved)
        │
        ▼  ZMQ transport (PUSH → PULL)
        │
        ▼  vllm/v1/engine/core_client.py:869
AsyncMPClient.process_outputs_socket()  (background asyncio task)
    → frames = await output_socket.recv_multipart(copy=False)  (:886)
    → outputs = decoder.decode(frames)   → EngineCoreOutputs   (:888)
    → outputs_queue.put_nowait(outputs)                        (:895)
```

### 5.2 OutputProcessor: Detokenize + Aggregate

```
vllm/v1/engine/async_llm.py:639-690
output_handler()  →  background asyncio task
    → :643   outputs = await engine_core.get_output_async()
    → :658   output_processor.process_outputs(outputs_slice, timestamp, stats)
        │
        ▼  vllm/v1/engine/output_processor.py:577
        process_outputs()
        │  For each EngineCoreOutput:
        │
        ├── :632  detokenizer.update(new_token_ids, ...)
        │   │  vllm/v1/engine/detokenizer.py
        │   │  → IncrementalDetokenizer accumulates token IDs
        │   │  → Decodes text incrementally (not re-decoding from scratch)
        │   │  → Checks for stop strings
        │
        ├── :641  logprobs_processor.update_from_output(...)
        │
        └── :644-651  req_state.make_request_output()
            │
            ├── :280-301  Handle stream_interval
            │   → Skip output if not enough new tokens (saves IPC)
            │
            └── :328-370  _new_request_output()
                → Constructs RequestOutput{
                    request_id,
                    outputs: [CompletionOutput{
                        text,                          // decoded string
                        token_ids,                     // all token IDs so far
                        logprobs,
                        finish_reason,
                    }],
                    finished: bool,
                    prompt_token_ids,
                    prompt_logprobs,
                  }
            │
            └── :655-657  req_state.queue.put(request_output)
                │  → Pushes to RequestOutputCollector (per-request asyncio queue)
```

### 5.3 AsyncLLM.generate() Consumer

```
vllm/v1/engine/async_llm.py:562-573
Consumer loop (inside AsyncLLM.generate()):
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        yield out   →  yields RequestOutput to HTTP handler
```

### 5.4 Streaming SSE Response

```
vllm/entrypoints/openai/chat_completion/serving.py:624
chat_completion_stream_generator()
    │
    ├── :734  async for res in result_generator:
    │   │  → Each res: RequestOutput with incremental token(s)
    │   │
    │   ├── FIRST iteration (:743-822):
    │   │   → Sends role preamble:
    │   │     data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0}]}
    │   │
    │   └── SUBSEQUENT iterations (:824-1330):
    │       ├── Extract delta_text from output.text (incremental new text)
    │       ├── Create DeltaMessage(content=delta_text)
    │       ├── Create ChatCompletionResponseStreamChoice
    │       │   {index, delta, logprobs, finish_reason}
    │       └── Serialize + yield as SSE:
    │           data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"},"index":0}]}
    │
    └── :1392  Final termination:
        yield "data: [DONE]\n\n"

vllm/entrypoints/openai/chat_completion/api_router.py:73
    → StreamingResponse(content=generator, media_type="text/event-stream")
    → FastAPI/Starlette handles Transfer-Encoding: chunked
```

### 5.5 Non-Streaming JSON Response

```
vllm/entrypoints/openai/chat_completion/serving.py:1394
chat_completion_full_generator()
    │
    ├── :1410  async for res in result_generator:
    │         final_res = res  →  keeps last (has all tokens)
    │
    ├── :1425-1723  Build ChatCompletionResponseChoice per output
    │   ├── token_ids = output.token_ids  (all accumulated)
    │   ├── Apply tool/reasoning parsing
    │   └── Create ChatMessage(role="assistant", content=full_text)
    │
    └── :1740-1770  Build final response:
        ChatCompletionResponse{
            id, created, model,
            choices: [ChatCompletionResponseChoice{index, message, finish_reason}],
            usage: UsageInfo{prompt_tokens, completion_tokens, total_tokens},
        }

vllm/entrypoints/openai/chat_completion/api_router.py:68
    → JSONResponse(content=response.model_dump())
```

---

## Key Architecture Notes for Your Config

| Flag | Effect |
|------|--------|
| `--tensor-parallel-size 4` | Heads + hidden dim sharded across 4 GPUs. All-reduce after each attention/mlp output |
| `--attention-backend FLASH_ATTN_TILELANG_V100` | Prefill uses JIT-compiled TileLang MMA kernels (block_M=64, block_N=32 for dim=256). Decode uses Triton |
| `--enable-expert-parallel` | 128 experts / 4 GPUs = 32 experts local per GPU. All2All dispatch/combine per MoE layer |
| `--max-num-batched-tokens 16384` | Scheduler stops adding new prefill requests when token budget exceeded |
| `--max-num-seqs 4` | Max 4 concurrent requests running at once |
| `--compilation-config cudagraph_mode=full_and_piecewise` | CUDA Graphs capture entire forward pass for replay (eliminates kernel launch overhead) |
| `VLLM_CUSTOM_ALLREDUCE_ALGO=2stage` | Custom 2-stage all-reduce for decode (low latency for small tensors) |
| `NCCL_P2P_LEVEL=NVL` | Forces NVLink for NCCL GPU-to-GPU communication |
