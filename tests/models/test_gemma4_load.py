"""Test Gemma4 model loads - ONLY RUN AFTER kernel/backend tests pass."""
from vllm import LLM


def test_gemma4_load_tp2():
    """Load Gemma4 with TP=2."""
    print("Testing Gemma4 model load with TP=2...")
    
    llm = LLM(
        model="google/gemma-4-31B",
        dtype="float16",
        tensor_parallel_size=2,
        enforce_eager=True,
        max_model_len=2048,
    )
    
    outputs = llm.generate("Hello", max_tokens=5)
    assert len(outputs) > 0
    print(f"✓ Gemma4 TP2 load passed")
    print(f"  Generated: {outputs[0].outputs[0].text}")


def test_gemma4_load_tp4():
    """Load Gemma4 with TP=4."""
    print("Testing Gemma4 model load with TP=4...")
    
    llm = LLM(
        model="google/gemma-4-31B",
        dtype="float16",
        tensor_parallel_size=4,
        enforce_eager=True,
        max_model_len=2048,
    )
    
    outputs = llm.generate("Hello", max_tokens=5)
    assert len(outputs) > 0
    print(f"✓ Gemma4 TP4 load passed")
    print(f"  Generated: {outputs[0].outputs[0].text}")


if __name__ == "__main__":
    # Test TP2 first (requires less VRAM)
    test_gemma4_load_tp2()
    
    # Uncomment to test TP4 (requires more VRAM)
    # test_gemma4_load_tp4()
