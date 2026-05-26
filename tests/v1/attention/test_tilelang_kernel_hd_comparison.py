"""HD 512 vs HD 256 comparison and TP2/TP4 simulation tests."""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')
from tilelang_fa_v100 import tilelang_paged_forward


def test_hd_comparison():
    """HD 512 should have similar numerical properties as HD 256."""
    batch, seq_len, num_heads, num_kv_heads = 4, 1024, 16, 8
    block_size, num_blocks = 16, 1024
    
    torch.manual_seed(42)
    q_256 = torch.randn(batch * seq_len, num_heads, 256, dtype=torch.float16, device='cuda')
    k_256 = torch.randn(num_blocks, block_size, num_kv_heads, 256, dtype=torch.float16, device='cuda')
    v_256 = torch.randn(num_blocks, block_size, num_kv_heads, 256, dtype=torch.float16, device='cuda')
    
    torch.manual_seed(42)
    q_512 = torch.randn(batch * seq_len, num_heads, 512, dtype=torch.float16, device='cuda')
    k_512 = torch.randn(num_blocks, block_size, num_kv_heads, 512, dtype=torch.float16, device='cuda')
    v_512 = torch.randn(num_blocks, block_size, num_kv_heads, 512, dtype=torch.float16, device='cuda')
    
    block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
    query_start_loc = torch.tensor([i * seq_len for i in range(batch + 1)], dtype=torch.int32, device='cuda')
    prefix_kv_lens = torch.zeros((batch,), dtype=torch.int32, device='cuda')
    
    out_256, _ = tilelang_paged_forward(q_256, k_256, v_256, block_table, cache_seqlens, query_start_loc, prefix_kv_lens, block_size=16, num_kv_heads=num_kv_heads, causal=True)
    out_512, _ = tilelang_paged_forward(q_512, k_512, v_512, block_table, cache_seqlens, query_start_loc, prefix_kv_lens, block_size=16, num_kv_heads=num_kv_heads, causal=True)
    
    stats_256 = {
        'mean': out_256.float().mean().item(),
        'std': out_256.float().std().item(),
        'max': out_256.abs().max().item(),
    }
    stats_512 = {
        'mean': out_512.float().mean().item(),
        'std': out_512.float().std().item(),
        'max': out_512.abs().max().item(),
    }
    
    assert torch.isnan(out_256).sum() == 0
    assert torch.isnan(out_512).sum() == 0
    assert out_256.abs().max() < 100
    assert out_512.abs().max() < 100
    print("✓ HD comparison: 256 stats={}, 512 stats={}".format(stats_256, stats_512))


def test_tp_simulation():
    """Simulate TP2/TP4 with reduced head counts."""
    batch, seq_len, block_size, num_blocks = 4, 1024, 16, 1024
    query_start_loc = torch.tensor([i * seq_len for i in range(batch + 1)], dtype=torch.int32, device='cuda')
    prefix_kv_lens = torch.zeros((batch,), dtype=torch.int32, device='cuda')
    
    cases = [
        ("TP2_HD512", 16, 8, 512),
        ("TP4_HD512", 8, 4, 512),
    ]
    for name, num_heads, num_kv_heads, head_dim in cases:
        q = torch.randn(batch * seq_len, num_heads, head_dim, dtype=torch.float16, device='cuda')
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
        block_table = torch.arange(batch * (seq_len // block_size + 1), dtype=torch.int32, device='cuda').view(batch, -1)
        cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')
        
        output, lse = tilelang_paged_forward(q, k_cache, v_cache, block_table, cache_seqlens, query_start_loc, prefix_kv_lens, block_size=16, num_kv_heads=num_kv_heads, causal=True)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print(f"✓ {name}: shape={output.shape}")


if __name__ == "__main__":
    test_hd_comparison()
    test_tp_simulation()
