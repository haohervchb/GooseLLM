"""Backend integration test for TileLang FA-V100.

Simple test to verify the backend integrates correctly with the kernel.
"""
import torch
import sys
sys.path.insert(0, '/home/rah/GooseLLM/3rdparty/tilelang-fa-v100')

from tilelang_fa_v100 import tilelang_paged_forward


def test_tilelang_backend_api():
    """Test TileLang backend API with various configurations."""
    print("Testing TileLang backend API...")
    
    test_cases = [
        ("HD 256, no sliding", 256, -1, -1),
        ("HD 256, sliding=1024", 256, 1024, 1024),
        ("HD 512, no sliding", 512, -1, -1),
        ("HD 512, sliding=1024", 512, 1024, 1024),
    ]
    
    batch = 4
    seq_len = 1024
    num_heads = 16
    num_kv_heads = 8
    block_size = 16
    num_blocks = 1024
    
    for name, head_dim, sw_q, sw_k in test_cases:
        print(f"\n  Testing: {name}")
        
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
        
        try:
            output, lse = tilelang_paged_forward(
                q, k_cache, v_cache, block_table, cache_seqlens,
                query_start_loc, prefix_kv_lens,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                causal=True,
                sliding_window_q=sw_q,
                sliding_window_k=sw_k,
            )
            
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            print(f"    ✓ Output shape: {output.shape}, max={output.abs().max().item():.4f}")
            
        except Exception as e:
            print(f"    ✗ Failed: {str(e)[:80]}")
            return False
    
    print("\n✓ TileLang backend API test PASSED")
    return True


if __name__ == "__main__":
    test_tilelang_backend_api()
