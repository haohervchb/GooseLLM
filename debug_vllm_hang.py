import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
import sys

def run_test(enable_chunked_prefill):
    print(f"\n--- Testing with enable_chunked_prefill={enable_chunked_prefill} ---")
    try:
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B",
            tensor_parallel_size=1,
            dtype="float16",
            speculative_model="Qwen/Qwen2.5-0.5B",
            num_speculative_tokens=5,
            enable_chunked_prefill=enable_chunked_prefill,
            max_model_len=4096, # Keep it small for test
            enforce_eager=True,
            gpu_memory_utilization=0.4
        )
        prompts = ["hello"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            print(f"Generated successfully: {repr(output.outputs[0].text)}")
    except Exception as e:
        print(f"Failed with exception: {e}")

if __name__ == "__main__":
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "false"
    run_test(test_mode.lower() == "true")
