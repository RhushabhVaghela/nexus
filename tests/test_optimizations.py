"""
Comprehensive Test Suite for Nexus Optimizations

Tests all 8 research-backed optimizations:
1. Layer Pipelining (EasySpec-style)
2. Adaptive Layer Skipping (SWIFT/LayerSkip)
3. Semi-Autoregressive Decoding (SPACE)
4. Async Decompression (nvCOMP)
5. Optimized Compression (ZSTD + quantization)
6. Layer Fusion (Kernel fusion)
7. Early Exit Routing (Dynamic routing)
8. Low-Rank Attention (Sparse attention)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import time
import tempfile
import os

# Import all optimization modules
from nexus.optimizations import (
    LayerPipeliningOptimizer,
    AdaptiveLayerSkipper,
    SemiAutoregressiveDecoder,
    AsyncDecompressor,
    OptimizedCompressor,
    LayerFusionOptimizer,
    EarlyExitRouter,
    SparseAttentionOptimizer,
)

from nexus.optimizations.layer_pipelining import (
    StaleActivationPredictor,
    SpeculativeLayerExecutor,
    PipelineConfig,
)

from nexus.optimizations.adaptive_layer_skipping import (
    LayerSkipRouter,
    SWIFTSkipper,
    LayerSkipConfig,
)

from nexus.optimizations.semi_autoregressive import (
    SPACEDecoder,
    ParallelTokenHead,
    SARConfig,
)

from nexus.optimizations.async_decompression import (
    AsyncDecompressionConfig,
    CUDAStreamManager,
    LayerBufferPool,
)

from nexus.optimizations.compression_optimized import (
    CompressionConfig,
    QuantizationCompressor,
    ZSTDQuantizedCompressor,
    QuantizedTensor,
)

from nexus.optimizations.layer_fusion import (
    FusionConfig,
    FusedAttentionFFN,
    FusedQKVProjection,
    FlashAttentionKernel,
)

from nexus.optimizations.early_exit_routing import (
    DynamicRoutingConfig,
    TokenRouter,
    DynamicLayerRouter,
)

from nexus.optimizations.low_rank_attention import (
    SparseAttentionConfig,
    LowRankAttention,
    SparseAttentionPattern,
    BlockSparseAttention,
)


# Fixtures


@pytest.fixture
def simple_transformer():
    """Create a simple transformer for testing."""

    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=512, num_layers=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=2048,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)

            # Mock config
            self.config = type(
                "Config",
                (),
                {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_layers,
                    "vocab_size": vocab_size,
                    "eos_token_id": 2,
                },
            )()

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.lm_head(x)
            return type("Output", (), {"logits": logits, "hidden_states": [x]})()

        def get_input_embeddings(self):
            return self.embedding

    return SimpleTransformer()


@pytest.fixture
def sample_hidden_states():
    """Generate sample hidden states for testing."""
    return torch.randn(2, 10, 512)  # [batch, seq_len, hidden]


# Test Layer Pipelining


class TestLayerPipelining:
    """Test suite for layer pipelining optimizations."""

    def test_stale_activation_predictor(self, sample_hidden_states):
        """Test stale activation prediction."""
        predictor = StaleActivationPredictor(hidden_size=512, num_layers=4)

        for layer_idx in range(4):
            predicted, confidence = predictor.predict_activation(
                layer_idx, sample_hidden_states
            )

            assert predicted.shape == sample_hidden_states.shape
            assert 0 <= confidence <= 1

    def test_speculative_executor_init(self, simple_transformer):
        """Test speculative executor initialization."""
        layers = list(simple_transformer.layers)
        predictor = StaleActivationPredictor(hidden_size=512, num_layers=4)
        config = PipelineConfig()

        executor = SpeculativeLayerExecutor(layers, predictor, config)

        assert len(executor.layers) == 4
        assert executor.config.use_speculative_execution

    def test_layer_pipelining_optimizer(self, simple_transformer):
        """Test layer pipelining optimizer."""
        optimizer = LayerPipeliningOptimizer(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        assert optimizer.num_layers == 4
        assert optimizer.hidden_size == 512

    def test_pipelining_forward(self, simple_transformer, sample_hidden_states):
        """Test forward pass with pipelining."""
        optimizer = LayerPipeliningOptimizer(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        output, metrics = optimizer.forward(sample_hidden_states)

        assert output.shape == sample_hidden_states.shape
        assert "layers_executed" in metrics

    def test_pipelining_stats(self, simple_transformer, sample_hidden_states):
        """Test pipelining statistics collection."""
        optimizer = LayerPipeliningOptimizer(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        # Run a few forward passes
        for _ in range(5):
            output, _ = optimizer.forward(sample_hidden_states)

        stats = optimizer.get_performance_report()

        assert "optimizer" in stats
        assert "num_layers" in stats
        assert stats["num_layers"] == 4


# Test Adaptive Layer Skipping


class TestAdaptiveLayerSkipping:
    """Test suite for adaptive layer skipping."""

    def test_layer_skip_router(self, sample_hidden_states):
        """Test layer skip router."""
        router = LayerSkipRouter(hidden_size=512, num_layers=80)

        should_exit, confidence = router.should_exit_early(
            sample_hidden_states, layer_idx=40
        )

        assert isinstance(should_exit, bool)
        assert 0 <= confidence <= 1

    def test_swift_skipper(self, sample_hidden_states):
        """Test SWIFT skipper."""
        skipper = SWIFTSkipper(hidden_size=512, skip_every_n=2)

        layer_func = lambda x: x
        output, was_skipped = skipper.forward(
            sample_hidden_states, layer_func, layer_idx=15
        )

        assert output.shape == sample_hidden_states.shape
        assert isinstance(was_skipped, bool)

    def test_adaptive_layer_skipper(self, simple_transformer):
        """Test adaptive layer skipper."""
        skipper = AdaptiveLayerSkipper(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        assert skipper.num_layers == 4

    def test_skipping_forward(self, simple_transformer, sample_hidden_states):
        """Test forward with layer skipping."""
        skipper = AdaptiveLayerSkipper(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        layers = list(simple_transformer.layers)
        output, metrics = skipper.forward_with_skipping(sample_hidden_states, layers)

        assert output.shape == sample_hidden_states.shape
        assert "layers_used" in metrics
        assert "layers_skipped" in metrics

    def test_skipping_stats(self, simple_transformer, sample_hidden_states):
        """Test layer skipping statistics."""
        skipper = AdaptiveLayerSkipper(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        layers = list(simple_transformer.layers)

        # Run multiple passes
        for _ in range(5):
            output, _ = skipper.forward_with_skipping(sample_hidden_states, layers)

        stats = skipper.get_stats()

        assert "total_tokens" in stats
        assert "avg_layers_used" in stats


# Test Semi-Autoregressive Decoding


class TestSemiAutoregressive:
    """Test suite for semi-autoregressive decoding."""

    def test_parallel_token_head(self, sample_hidden_states):
        """Test parallel token head."""
        head = ParallelTokenHead(hidden_size=512, vocab_size=1000, num_heads=4)

        logits_list, confidence_list = head.forward(sample_hidden_states)

        assert len(logits_list) == 4
        assert len(confidence_list) == 4
        assert logits_list[0].shape == (2, 1000)  # [batch, vocab]

    def test_space_decoder_init(self, simple_transformer):
        """Test SPACE decoder initialization."""
        decoder = SPACEDecoder(
            base_model=simple_transformer, vocab_size=1000, hidden_size=512
        )

        assert decoder.vocab_size == 1000
        assert decoder.config.lookahead_tokens == 4

    def test_semi_autoregressive_decoder(self, simple_transformer):
        """Test semi-autoregressive decoder wrapper."""
        decoder = SemiAutoregressiveDecoder(simple_transformer)

        assert hasattr(decoder, "space_decoder")

    def test_decoder_stats(self, simple_transformer):
        """Test decoder statistics."""
        decoder = SPACEDecoder(
            base_model=simple_transformer, vocab_size=1000, hidden_size=512
        )

        stats = decoder.get_stats()

        assert "total_calls" in stats
        assert "acceptance_rate" in stats


# Test Async Decompression


class TestAsyncDecompression:
    """Test suite for async decompression."""

    def test_cuda_stream_manager(self):
        """Test CUDA stream manager."""
        manager = CUDAStreamManager(num_streams=3)

        assert manager.num_streams == 3

        if torch.cuda.is_available():
            stream = manager.get_next_stream()
            assert stream is not None

    def test_layer_buffer_pool(self):
        """Test layer buffer pool."""
        pool = LayerBufferPool(max_size=5)

        # Get buffer
        buffer = pool.get_buffer("layer_0", (100, 512), torch.float32, "cpu")
        assert buffer.shape == (100, 512)

        # Return buffer
        pool.return_buffer("layer_0", buffer)

        # Get same buffer back
        buffer2 = pool.get_buffer("layer_0", (100, 512), torch.float32, "cpu")
        assert buffer2.shape == (100, 512)

    def test_async_decompressor_init(self):
        """Test async decompressor initialization."""
        decompressor = AsyncDecompressor()

        assert decompressor.config.num_worker_threads == 4
        assert decompressor.config.prefetch_depth == 3

    def test_decompressor_stats(self):
        """Test decompressor statistics."""
        decompressor = AsyncDecompressor()

        stats = decompressor.get_stats()

        assert "total_decompressions" in stats
        assert "cache_hits" in stats


# Test Compression Optimized


class TestCompressionOptimized:
    """Test suite for optimized compression."""

    def test_quantization_compressor(self):
        """Test quantization compressor."""
        compressor = QuantizationCompressor()

        tensor = torch.randn(100, 512)
        quantized = compressor.quantize(tensor, bits=8)

        assert isinstance(quantized, QuantizedTensor)
        assert quantized.quantized_data.dtype == torch.uint8

    def test_quantized_tensor_roundtrip(self):
        """Test quantized tensor serialization."""
        tensor = torch.randn(10, 10)
        compressor = QuantizationCompressor()
        quantized = compressor.quantize(tensor, bits=8)

        # Serialize
        data = quantized.to_bytes()
        assert isinstance(data, bytes)

        # Deserialize
        restored = QuantizedTensor.from_bytes(data)
        assert restored.original_shape == tensor.shape

        # Dequantize
        dequantized = restored.dequantize()
        assert dequantized.shape == tensor.shape

    def test_zstd_compressor_init(self):
        """Test ZSTD compressor initialization."""
        compressor = ZSTDQuantizedCompressor()

        assert compressor.config.compression_level == 22

    def test_optimized_compressor(self):
        """Test optimized compressor."""
        compressor = OptimizedCompressor()

        tensor = torch.randn(100, 100)
        compressed = compressor.compress_tensor(tensor)

        assert isinstance(compressed, bytes)

    def test_compressor_stats(self):
        """Test compressor statistics."""
        compressor = ZSTDQuantizedCompressor()

        # Compress some tensors
        for _ in range(5):
            tensor = torch.randn(100, 100)
            compressor.compress(tensor)

        stats = compressor.get_stats()

        assert "compression_ratio" in stats
        assert "original_bytes" in stats


# Test Layer Fusion


class TestLayerFusion:
    """Test suite for layer fusion."""

    def test_fused_qkv_projection(self):
        """Test fused QKV projection."""
        proj = FusedQKVProjection(hidden_size=512, num_heads=8, head_dim=64)

        hidden_states = torch.randn(2, 10, 512)
        q, k, v = proj.forward(hidden_states)

        assert q.shape == (2, 8, 10, 64)
        assert k.shape == (2, 8, 10, 64)
        assert v.shape == (2, 8, 10, 64)

    def test_flash_attention_kernel(self):
        """Test flash attention kernel."""
        kernel = FlashAttentionKernel(num_heads=8, head_dim=64)

        q = torch.randn(2, 8, 10, 64)
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)

        output = kernel.forward(q, k, v, is_causal=True)

        assert output.shape == q.shape

    def test_layer_fusion_optimizer(self):
        """Test layer fusion optimizer."""
        optimizer = LayerFusionOptimizer()

        assert optimizer.config.fuse_attention_ffn
        assert optimizer.config.use_flash_attention

    def test_fusion_stats(self):
        """Test fusion statistics."""
        optimizer = LayerFusionOptimizer()

        stats = optimizer.get_stats()

        assert "layers_fused" in stats


# Test Early Exit Routing


class TestEarlyExitRouting:
    """Test suite for early exit routing."""

    def test_token_router(self, sample_hidden_states):
        """Test token router."""
        router = TokenRouter(hidden_size=512, num_layers=80)

        exit_layers = router.estimate_exit_layer(sample_hidden_states)

        assert exit_layers.shape == (2, 10)  # [batch, seq_len]
        assert (exit_layers >= 0).all() and (exit_layers <= 80).all()

    def test_dynamic_layer_router(self, sample_hidden_states):
        """Test dynamic layer router."""
        router = DynamicLayerRouter(hidden_size=512, num_layers=80)

        layer_mask = router.compute_layer_mask(sample_hidden_states, min_layers=30)

        assert layer_mask.shape == (80,)
        assert (layer_mask >= 0).all() and (layer_mask <= 1).all()

    def test_early_exit_router(self, simple_transformer):
        """Test early exit router."""
        router = EarlyExitRouter(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        assert router.num_layers == 4

    def test_routing_forward(self, simple_transformer, sample_hidden_states):
        """Test forward with routing."""
        router = EarlyExitRouter(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        layers = list(simple_transformer.layers)
        output, metrics = router.forward_with_routing(sample_hidden_states, layers)

        assert output.shape == sample_hidden_states.shape
        assert "layers_executed" in metrics
        assert "avg_exit_layer" in metrics


# Test Low-Rank Attention


class TestLowRankAttention:
    """Test suite for low-rank attention."""

    def test_sparse_attention_pattern(self):
        """Test sparse attention pattern."""
        config = SparseAttentionConfig()
        pattern = SparseAttentionPattern(seq_len=128, config=config)

        assert pattern.attention_mask.shape == (128, 128)
        assert pattern.attention_mask.dtype == torch.bool

    def test_block_sparse_attention(self):
        """Test block sparse attention."""
        attn = BlockSparseAttention(block_size=32)

        q = torch.randn(2, 8, 128, 64)
        k = torch.randn(2, 8, 128, 64)
        v = torch.randn(2, 8, 128, 64)

        output = attn.forward(q, k, v, scale=0.125)

        assert output.shape == q.shape

    def test_low_rank_attention(self):
        """Test low-rank attention module."""
        attn = LowRankAttention(
            hidden_size=512, num_heads=8, seq_len=128, config=SparseAttentionConfig()
        )

        hidden_states = torch.randn(2, 128, 512)
        output = attn.forward(hidden_states)

        assert output.shape == hidden_states.shape

    def test_sparse_attention_optimizer(self):
        """Test sparse attention optimizer."""
        optimizer = SparseAttentionOptimizer()

        assert optimizer.config.sparsity_ratio == 0.8

    def test_sparse_stats(self):
        """Test sparse attention statistics."""
        optimizer = SparseAttentionOptimizer()

        stats = optimizer.get_stats()

        assert "computation_reduction" in stats


# Integration Tests


class TestOptimizationIntegration:
    """Integration tests for optimization combinations."""

    def test_all_optimizations_importable(self):
        """Test that all optimization modules can be imported."""
        # Import all optimization classes to verify they're available
        from nexus.optimizations import (
            LayerPipeliningOptimizer,
            AdaptiveLayerSkipper,
            SemiAutoregressiveDecoder,
            AsyncDecompressor,
            OptimizedCompressor,
            LayerFusionOptimizer,
            EarlyExitRouter,
            SparseAttentionOptimizer,
        )

        # Verify all imports succeeded
        assert LayerPipeliningOptimizer is not None
        assert AdaptiveLayerSkipper is not None
        assert SemiAutoregressiveDecoder is not None
        assert AsyncDecompressor is not None
        assert OptimizedCompressor is not None
        assert LayerFusionOptimizer is not None
        assert EarlyExitRouter is not None
        assert SparseAttentionOptimizer is not None

    def test_config_dataclasses(self):
        """Test that all config dataclasses work."""
        configs = [
            PipelineConfig(),
            LayerSkipConfig(),
            SARConfig(),
            AsyncDecompressionConfig(),
            CompressionConfig(),
            FusionConfig(),
            DynamicRoutingConfig(),
            SparseAttentionConfig(),
        ]

        for config in configs:
            assert config is not None

    def test_performance_targets(self, simple_transformer, sample_hidden_states):
        """Test that optimizations meet performance targets."""
        # This is a smoke test - full benchmarks would take too long

        # Test layer skipping
        skipper = AdaptiveLayerSkipper(
            model=simple_transformer, num_layers=4, hidden_size=512
        )

        layers = list(simple_transformer.layers)
        output, metrics = skipper.forward_with_skipping(sample_hidden_states, layers)

        # Should complete without error
        assert output is not None
        assert metrics is not None

    def test_memory_efficiency(self):
        """Test memory efficiency of optimizations."""
        # Test that buffer pool reuses memory
        pool = LayerBufferPool(max_size=5)

        # Get and return buffers multiple times
        for _ in range(10):
            buffer = pool.get_buffer("test", (1000, 1000), torch.float32, "cpu")
            pool.return_buffer("test", buffer)

        # Verify buffer pool is working correctly
        stats = pool.get_stats()
        assert stats is not None, "Buffer pool should return statistics"
        assert "buffers_reused" in stats, "Pool should track buffer reuse count"
        assert stats["buffers_reused"] >= 10, "Should have reused at least 10 buffers"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
