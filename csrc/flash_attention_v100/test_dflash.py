import torch
import flash_attn_v100_cuda

def test():
    num_heads = 32
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    
    num_seqs = 4
    # Query lengths: 1, 16, 5, 16
    query_lens_list = [1, 16, 5, 16]
    # Seq lengths: 10, 32, 20, 100
    seq_lens_list = [10, 32, 20, 100]
    
    total_query = sum(query_lens_list)
    
    q = torch.randn(total_query, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k_cache = torch.randn(200, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    v_cache = torch.randn(200, block_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda')
    
    max_seq_len = max(seq_lens_list)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    
    block_table = torch.full((num_seqs, max_blocks), -1, dtype=torch.int32, device='cuda')
    
    for i, seq_len in enumerate(seq_lens_list):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_table[i, :num_blocks] = torch.arange(num_blocks, dtype=torch.int32, device='cuda')
        
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device='cuda')
    
    # [0, 1, 17, 22, 38]
    locs = [0]
    for ql in query_lens_list:
        locs.append(locs[-1] + ql)
    query_start_loc = torch.tensor(locs, dtype=torch.int32, device='cuda')
    
    prefix_kv_lens_list = [sl - ql for sl, ql in zip(seq_lens_list, query_lens_list)]
    prefix_kv_lens = torch.tensor(prefix_kv_lens_list, dtype=torch.int32, device='cuda')
    
    out = torch.empty_like(q)
    
    flash_attn_v100_cuda.paged_fwd(
        q, k_cache, v_cache, block_table, seq_lens, query_start_loc, prefix_kv_lens, out,
        num_kv_heads, block_size, 1.0, False
    )
    flash_attn_v100_cuda.paged_fwd(
        q, k_cache, v_cache, block_table, seq_lens, query_start_loc, prefix_kv_lens, out,
        num_kv_heads, block_size, 1.0, True
    )

if __name__ == "__main__":
    test()
    print("Mixed query batch passed!", flush=True)
