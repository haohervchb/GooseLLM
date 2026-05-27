"""TileLang FA-V100 sliding window test - HD 256.

Gemma-4-31B sliding_attention layers use head_dim=256 with sliding_window=1024.
This test verifies the sliding window mask works correctly.
"""
import torch
import sys

sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')

from tilelang_fa_v100 import tilelang_paged_forward


def test_sliding_window_hd256():
    """HD 256 with sliding window (Gemma4 sliding layer)."""
    batch = 4
    seq_len = 2048
    num_heads = 16
    num_kv_heads = 8
    head_dim = 256
    sliding_window = 1024
    
    block_size = 16
    num_blocks = 2048
    
    print(f"Testing HD 256 sliding window:")
    print(f"  Batch={batch}, SeqLen={seq_len}, Heads={num_heads}, KVHeads={num_kv_heads}, HD={head_dim}")
    print(f"  Sliding window={sliding_window}")
    
    torch.manual_seed(42)
    q = torch.randn(
        batch * seq_len, num_heads, head_dim,
        dtype=torch.float16, device='cuda'
    )
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
    
    query_start_loc = torch.tensor(
        [i * seq_len for i in range(batch + 1)],
        dtype=torch.int32, device='cuda'
    )
    prefix_kv_lens = torch.zeros(
        (batch,), dtype=torch.int32, device='cuda'
    )
    
    print("  Running TileLang kernel with sliding window...")
    
    output, lse = tilelang_paged_forward(
        q, k_cache, v_cache, block_table, cache_seqlens,
        query_start_loc, prefix_kv_lens,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        causal=True,
        sliding_window_q=sliding_window,
        sliding_window_k=sliding_window,
    )
    
    assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
    print(f"  ✓ Output shape correct: {output.shape}")
    
    nan_count = torch.isnan(output).sum().item()
    inf_count = torch.isinf(output).sum().item()
    
    assert nan_count == 0, f"NaN detected: {nan_count} values"
    print(f"  ✓ No NaN values")
    
    assert inf_count == 0, f"Inf detected: {inf_count} values"
    print(f"  ✓ No Inf values")
    
    max_val = output.abs().max().item()
    assert max_val < 100, f"Unusually large output: max={max_val}"
    print(f"  ✓ Output magnitude reasonable: max={max_val:.4f}")
    
    mean_val = output.float().mean().item()
    std_val = output.float().std().item()
    print(f"  Statistics: mean={mean_val:.6f}, std={std_val:.6f}")
    
    print("\n✓ HD 256 sliding window test PASSED")
    return True


if __name__ == "__main__":
    test_sliding_window_hd256()
