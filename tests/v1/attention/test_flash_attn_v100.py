# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the FLASH_ATTN_V100 (SM70) attention backend.

Validates the FlashAttention V100 paged prefill kernel and decode path
against reference implementations.
"""

import os
from functools import partial

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_backend_includes_kv_cache_update,
    try_get_attention_backend,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec

# Check if we are on a V100 (SM70) GPU
_device_capability = current_platform.get_device_capability()
_IS_SM70 = _device_capability is not None and _device_capability.major == 7

# Re-use the create_and_prepopulate_kv_cache from test_attention_backends
from tests.v1.attention.test_attention_backends import (
    create_and_prepopulate_kv_cache,
    MockAttentionLayer,
)

HEAD_SIZES = [64, 96, 128, 256]
HEAD_SIZES_DECODE = [64, 128, 256]

BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024], query_lens=[1, 1, 1, 1]
    ),
    "medium_prefill": BatchSpec(
        seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]
    ),
    "mixed_medium": BatchSpec(
        seq_lens=[512, 1024, 2048, 512, 1024, 2048],
        query_lens=[1, 1, 1, 7, 7, 7],
    ),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[1024], query_lens=[64]),
    "large_decode": BatchSpec(seq_lens=[2048] * 8, query_lens=[1] * 8),
    "large_prefill": BatchSpec(seq_lens=[4096] * 4, query_lens=[32] * 4),
}


def _convert_dtype_to_torch(dtype):
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.float16
        raise ValueError(f"Unknown dtype: {dtype}")
    return dtype


def _run_flash_v100_backend(
    vllm_config,
    kv_cache_spec: FullAttentionSpec,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_type: AttentionType = AttentionType.DECODER,
) -> torch.Tensor:
    """Run the FLASH_ATTN_V100 backend's forward pass."""
    backend = AttentionBackendEnum.FLASH_ATTN_V100
    builder_cls, impl_cls = try_get_attention_backend(backend)

    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size ** 0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        attn_type=attn_type,
        kv_cache_dtype="auto",
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)

    backend_includes_kv = try_backend_includes_kv_cache_update(backend)
    if not backend_includes_kv:
        impl.do_kv_cache_update(
            mock_layer, key, value, kv_cache, attn_metadata.slot_mapping
        )

    result = impl.forward(
        mock_layer, query, key, value, kv_cache, attn_metadata, output=output
    )
    return result


def _run_triton_backend(
    vllm_config,
    kv_cache_spec: FullAttentionSpec,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_type: AttentionType = AttentionType.DECODER,
) -> torch.Tensor:
    """Run the TRITON_ATTN backend's forward pass as reference."""
    backend = AttentionBackendEnum.TRITON_ATTN
    builder_cls, impl_cls = try_get_attention_backend(backend)

    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size ** 0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        attn_type=attn_type,
        kv_cache_dtype="auto",
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)

    backend_includes_kv = try_backend_includes_kv_cache_update(backend)
    if not backend_includes_kv:
        impl.do_kv_cache_update(
            mock_layer, key, value, kv_cache, attn_metadata.slot_mapping
        )

    result = impl.forward(
        mock_layer, query, key, value, kv_cache, attn_metadata, output=output
    )
    return result


def _create_synthetic_vllm_config(
    max_model_len: int,
    head_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int = 16,
    num_gpu_blocks: int = 8192,
) -> "VllmConfig":
    """Create a VllmConfig with synthetic model config (no HF download needed)."""
    import json
    import tempfile
    import types

    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    # Create a temp directory with a minimal config.json so ModelConfig validates
    tmpdir = tempfile.mkdtemp(prefix="flash_v100_test_")
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": num_q_heads * head_size,
            "num_attention_heads": num_q_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_size,
            "max_position_embeddings": max_model_len,
            "model_type": "llama",
        }, f)

    hidden_size = num_q_heads * head_size
    model_config = ModelConfig(
        model=tmpdir,
        tokenizer=tmpdir,
        trust_remote_code=True,
        dtype=torch.float16,
        seed=0,
        max_model_len=max_model_len,
    )

    # Override the HF config with synthetic values
    model_config.hf_config.update({
        "hidden_size": hidden_size,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_size,
        "max_position_embeddings": max_model_len,
    })
    model_config.hf_text_config.update({
        "hidden_size": hidden_size,
        "num_attention_heads": num_q_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_size,
        "max_position_embeddings": max_model_len,
        "rope_scaling": None,
    })

    # Patch methods the backends expect
    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, i: 0.0, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config
    )

    # Patch methods the backends expect
    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, i: 0.0, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config
    )

    cache_config = CacheConfig(block_size=block_size, cache_dtype="auto", swap_space=0)
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0

    parallel_config = ParallelConfig(tensor_parallel_size=1)
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=False,
    )

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def _test_flash_v100_correctness(
    batch_spec: BatchSpec,
    head_size: int,
    *,
    num_q_heads: int = 8,
    num_kv_heads: int = 8,
    block_size: int = 16,
    attn_type: AttentionType = AttentionType.DECODER,
    atol: float = 2e-2,
    rtol: float = 2e-2,
):
    """Test FLASH_ATTN_V100 produces correct outputs vs TRITON_ATTN reference."""
    set_random_seed(42)

    max_model_len = max(batch_spec.seq_lens)

    vllm_config = _create_synthetic_vllm_config(
        max_model_len=max_model_len,
        head_size=head_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )
    device = torch.device("cuda:0")
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    dtype = torch.float16

    # Generate data for SDPA reference + vLLM backends
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)

        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    if attn_type == AttentionType.ENCODER_ONLY:
        common_attn_metadata.causal = False

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1000,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
    )

    # Run TRITON_ATTN as reference (expects [num_blocks, 2, block_size, num_kv_heads, head_size])
    # create_and_prepopulate_kv_cache returns [2, num_blocks, ...], so transpose
    kv_cache_for_backend = kv_cache.transpose(0, 1).contiguous()

    ref_output = _run_triton_backend(
        vllm_config,
        kv_cache_spec,
        device,
        common_attn_metadata,
        query_vllm,
        key_vllm,
        value_vllm,
        kv_cache_for_backend,
        attn_type=attn_type,
    )

    # Run FLASH_ATTN_V100 (same layout as Triton)
    flash_v100_output = _run_flash_v100_backend(
        vllm_config,
        kv_cache_spec,
        device,
        common_attn_metadata,
        query_vllm,
        key_vllm,
        value_vllm,
        kv_cache_for_backend,
        attn_type=attn_type,
    )

    assert flash_v100_output.shape == ref_output.shape, (
        f"Shape mismatch: FLASH_ATTN_V100 {flash_v100_output.shape} "
        f"vs TRITON {ref_output.shape}"
    )
    assert flash_v100_output.dtype == ref_output.dtype

    assert torch.isfinite(flash_v100_output).all(), (
        "FLASH_ATTN_V100 produced non-finite values"
    )

    torch.testing.assert_close(
        flash_v100_output,
        ref_output,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: (
            f"FLASH_ATTN_V100 output differs from TRITON_ATTN. "
            f"batch={batch_spec.name}, head_dim={head_size}, "
            f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}\n{msg}"
        ),
    )


# -------------------------------------------------------------------------
# Prefill tests
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _IS_SM70, reason="Requires V100 (SM70) GPU")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_prefill", "medium_prefill", "single_prefill", "large_prefill"],
)
def test_prefill_correctness(batch_spec_name: str, head_size: int):
    """Test FLASH_ATTN_V100 prefill correctness vs TRITON_ATTN."""
    batch_spec = BATCH_SPECS[batch_spec_name]
    _test_flash_v100_correctness(
        batch_spec,
        head_size,
        num_q_heads=4,
        num_kv_heads=4,
        attn_type=AttentionType.DECODER,
    )


@pytest.mark.skipif(not _IS_SM70, reason="Requires V100 (SM70) GPU")
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize(
    "batch_spec_name", ["small_prefill", "mixed_medium"]
)
def test_prefill_gqa_correctness(batch_spec_name: str, head_size: int):
    """Test FLASH_ATTN_V100 prefill with GQA (8 Q-heads, 2 KV-heads)."""
    batch_spec = BATCH_SPECS[batch_spec_name]
    _test_flash_v100_correctness(
        batch_spec,
        head_size,
        num_q_heads=8,
        num_kv_heads=2,
        attn_type=AttentionType.DECODER,
    )


# -------------------------------------------------------------------------
# Decode tests
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _IS_SM70, reason="Requires V100 (SM70) GPU")
@pytest.mark.parametrize("head_size", HEAD_SIZES_DECODE)
@pytest.mark.parametrize(
    "batch_spec_name",
    ["small_decode", "single_decode", "medium_decode", "large_decode"],
)
def test_decode_correctness(batch_spec_name: str, head_size: int):
    """Test FLASH_ATTN_V100 decode correctness vs TRITON_ATTN."""
    batch_spec = BATCH_SPECS[batch_spec_name]
    _test_flash_v100_correctness(
        batch_spec,
        head_size,
        num_q_heads=4,
        num_kv_heads=4,
        attn_type=AttentionType.DECODER,
    )


@pytest.mark.skipif(not _IS_SM70, reason="Requires V100 (SM70) GPU")
@pytest.mark.parametrize("head_size", HEAD_SIZES_DECODE)
@pytest.mark.parametrize("batch_spec_name", ["small_decode", "medium_decode"])
def test_decode_gqa_correctness(batch_spec_name: str, head_size: int):
    """Test FLASH_ATTN_V100 decode with GQA."""
    batch_spec = BATCH_SPECS[batch_spec_name]
    _test_flash_v100_correctness(
        batch_spec,
        head_size,
        num_q_heads=8,
        num_kv_heads=2,
        attn_type=AttentionType.DECODER,
    )


# -------------------------------------------------------------------------
# Smoke test (always runs, works on any GPU or CPU)
# -------------------------------------------------------------------------


def test_backend_import_and_registration():
    """Verify FLASH_ATTN_V100 backend is registered and importable."""
    backend = AttentionBackendEnum.FLASH_ATTN_V100
    assert backend.name == "FLASH_ATTN_V100"

    try:
        backend_class = backend.get_class()
        name = backend_class.get_name()
        assert name == "FLASH_ATTN_V100", f"Unexpected name: {name}"
    except ImportError:
        pytest.skip("FLASH_ATTN_V100 not importable (missing flash_attn_v100_cuda)")
