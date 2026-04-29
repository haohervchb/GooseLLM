from vllm import LLM, SamplingParams
from vllm.config import SpeculativeConfig

llm = LLM(
    model="Qwen/Qwen2.5-0.5B",
    tensor_parallel_size=1,
    trust_remote_code=True,
    dtype="float16",
    speculative_model="Qwen/Qwen2.5-0.5B",
    num_speculative_tokens=5,
    use_v2_block_manager=True
)

prompts = ["hello"]
sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")

