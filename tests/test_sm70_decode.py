#!/usr/bin/env python3
"""Test script for SM70 paged decode attention kernel."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm.platforms import current_platform


def is_sm70():
    cap = current_platform.get_device_capability()
    return cap is not None and cap.major == 7


def test_sm70_decode():
    if not is_sm70():
        print("WARNING: Not running on SM70 GPU, skipping test")
        return

    from vllm.v1.attention.ops.sm70_decode import sm70_paged_decode_attention

    # Test parameters
    num_seqs = 4
    num_heads = 32
    num_kv_heads = 4
    head_size = 128
    block_size = 16
    max_seq_len = 256
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_num_blocks_per_seq * 2

    scale = 1.0 / (head_size**0.5)

    # Create tensors
    torch.manual_seed(42)
    query = torch.randn(num_seqs, num_heads, head_size, dtype=torch.float16, device="cuda")
    output = torch.zeros(num_seqs, num_heads, head_size, dtype=torch.float16, device="cuda")

    # KV cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
    kv_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=torch.float16, device="cuda")
    key_cache, value_cache = kv_cache.unbind(1)

    # Block tables
    block_tables = torch.zeros(num_seqs, max_num_blocks_per_seq, dtype=torch.int32, device="cuda")
    seq_lens = torch.randint(16, max_seq_len, (num_seqs,), dtype=torch.int32, device="cuda")

    # Fill block tables
    for i in range(num_seqs):
        num_blocks_needed = (seq_lens[i].item() + block_size - 1) // block_size
        block_tables[i, :num_blocks_needed] = torch.arange(i * max_num_blocks_per_seq, i * max_num_blocks_per_seq + num_blocks_needed, dtype=torch.int32, device="cuda")

    # Test SM70 kernel
    try:
        sm70_paged_decode_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            scale=scale,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            max_seq_len=max_seq_len,
        )
        print("SM70 decode kernel executed successfully!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"SM70 decode kernel failed: {e}")
        raise


if __name__ == "__main__":
    test_sm70_decode()
