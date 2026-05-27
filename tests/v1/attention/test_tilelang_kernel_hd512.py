"""TileLang FA-V100 HD 512 correctness test - no model loading.

Critical test for Gemma-4-31B full_attention layers which use head_dim=512.
This test must pass BEFORE attempting any model loading.
"""
import torch
import sys

# Add tilelang-fa-v100 to path
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')

from tilelang_fa_v100 import tilelang_paged_forward


def test_hd512_dense():
    """HD 512 dense attention (new feature for Gemma4 full_attention layers)."""
    # Gemma4 full_attention layer config (scaled down for testing)
    batch = 4
    seq_len = 1024
    num_heads = 16  # 32 / 2 for TP2 simulation
    num_kv_heads = 8  # 16 / 2 for TP2 simulation
    head_dim = 512  # Gemma4 full_attention uses HD 512
    
    block_size = 16
    num_blocks = 1024
    
    print(f"Testing HD 512 dense attention:")
    print(f"  Batch={batch}, SeqLen={seq_len}, Heads={num_heads}, KVHeads={num_kv_heads}, HD={head_dim}")
    
    # Create test tensors
    # Note: q shape is [num_tokens, num_heads, head_dim]
    q = torch.randn(
        batch * seq_len, num_heads, head_dim,
        dtype=torch.float16, device='cuda'
    )
    # k_cache, v_cache shape is [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.float16, device='cuda'
    )
    v_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.float16, device='cuda'
    )
    block_table = torch.arange(
        batch * (seq_len // block_size + 1),
        dtype=torch.int32, device='cuda'
    ).view(batch, -1)
    cache_seqlens = torch.full(
        (batch,), seq_len,
        dtype=torch.int32, device='cuda'
    )
    
    # query_start_loc: cumulative sequence lengths
    query_start_loc = torch.tensor(
        [i * seq_len for i in range(batch + 1)],
        dtype=torch.int32, device='cuda'
    )
    # prefix_kv_lens: prefix length (0 for simple test)
    prefix_kv_lens = torch.zeros(
        (batch,), dtype=torch.int32, device='cuda'
    )
    
    print("  Running TileLang kernel (first run will JIT compile)...")
    
    # Run TileLang paged forward
    output, lse = tilelang_paged_forward(
        q, k_cache, v_cache, block_table, cache_seqlens,
        query_start_loc, prefix_kv_lens,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        causal=True
    )
    
    # Verify output shape
    assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
    print(f"  ✓ Output shape correct: {output.shape}")
    
    # Check for NaN/Inf
    nan_count = torch.isnan(output).sum().item()
    inf_count = torch.isinf(output).sum().item()
    
    assert nan_count == 0, f"NaN detected in output: {nan_count} NaN values"
    print(f"  ✓ No NaN values")
    
    assert inf_count == 0, f"Inf detected in output: {inf_count} Inf values"
    print(f"  ✓ No Inf values")
    
    # Check output magnitude (should be reasonable for attention)
    max_val = output.abs().max().item()
    assert max_val < 100, f"Unusually large output: max={max_val}"
    print(f"  ✓ Output magnitude reasonable: max={max_val:.4f}")
    
    # Check statistics
    mean_val = output.float().mean().item()
    std_val = output.float().std().item()
    print(f"  Statistics: mean={mean_val:.6f}, std={std_val:.6f}")
    
    print("\n✓ HD 512 dense test PASSED")
    return True


if __name__ == "__main__":
    test_hd512_dense()
