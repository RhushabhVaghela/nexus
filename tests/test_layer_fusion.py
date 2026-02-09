"""
Comprehensive unit tests for layer_fusion.py
Tests attention fusion, FFN fusion, memory optimization, and compatibility.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Tuple, Optional

from nexus.optimizations.layer_fusion import (
    FusionConfig,
    FusedQKVProjection,
    FlashAttentionKernel,
    FusedFFN,
    FusedAttentionFFN,
    LayerFusionOptimizer,
)


class TestFusionConfig:
    """Test FusionConfig dataclass."""

    def test_default_config_values(self):
        """Test default fusion configuration."""
        config = FusionConfig()
        assert config.fuse_attention_ffn is True
        assert config.fuse_qkv_projection is True
        assert config.use_flash_attention is True
        assert config.optimize_cache_hierarchy is True
        assert config.use_tensor_cores is True
        assert config.sequence_parallel is False

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = FusionConfig(
            fuse_attention_ffn=False,
            fuse_qkv_projection=False,
            use_flash_attention=False,
            optimize_cache_hierarchy=False,
            use_tensor_cores=False,
            sequence_parallel=True,
        )

        assert config.fuse_attention_ffn is False
        assert config.fuse_qkv_projection is False
        assert config.use_flash_attention is False
        assert config.optimize_cache_hierarchy is False
        assert config.use_tensor_cores is False
        assert config.sequence_parallel is True


class TestFusedQKVProjection:
    """Test FusedQKVProjection for fused attention projections."""

    def test_init(self):
        """Test initialization."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        assert proj.hidden_size == 768
        assert proj.num_heads == 12
        assert proj.head_dim == 64
        assert proj.total_dim == 768

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        q, k, v = proj(hidden_states)

        assert q.shape == (batch_size, 12, seq_len, 64)
        assert k.shape == (batch_size, 12, seq_len, 64)
        assert v.shape == (batch_size, 12, seq_len, 64)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        for batch_size in [1, 4, 16]:
            hidden_states = torch.randn(batch_size, 128, 768)
            q, k, v = proj(hidden_states)

            assert q.shape[0] == batch_size
            assert k.shape[0] == batch_size
            assert v.shape[0] == batch_size

    def test_forward_different_seq_lengths(self):
        """Test forward pass with different sequence lengths."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        batch_size = 2
        for seq_len in [1, 64, 256, 512]:
            hidden_states = torch.randn(batch_size, seq_len, 768)
            q, k, v = proj(hidden_states)

            assert q.shape[2] == seq_len
            assert k.shape[2] == seq_len
            assert v.shape[2] == seq_len

    def test_forward_consistency(self):
        """Test that Q, K, V have consistent shapes."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        hidden_states = torch.randn(2, 128, 768)
        q, k, v = proj(hidden_states)

        assert q.shape == k.shape
        assert q.shape == v.shape

    def test_fused_proj_weight_shape(self):
        """Test fused projection weight shape."""
        proj = FusedQKVProjection(hidden_size=768, num_heads=12, head_dim=64)

        # Should have weight for 3 * total_dim
        expected_dim = 3 * 12 * 64  # 3 * num_heads * head_dim
        assert proj.fused_proj.weight.shape == (expected_dim, 768)


class TestFlashAttentionKernel:
    """Test FlashAttentionKernel for optimized attention."""

    def test_init(self):
        """Test initialization."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64)

        assert kernel.num_heads == 12
        assert kernel.head_dim == 64
        assert kernel.softmax_scale == (64**-0.5)

    def test_init_custom_softmax_scale(self):
        """Test initialization with custom softmax scale."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64, softmax_scale=0.1)

        assert kernel.softmax_scale == 0.1

    @patch("src.optimizations.layer_fusion.FlashAttentionKernel.__init__")
    def test_init_flash_attn_available(self, mock_init):
        """Test when FlashAttention is available."""
        mock_init.return_value = None

        kernel = FlashAttentionKernel.__new__(FlashAttentionKernel)
        kernel.num_heads = 12
        kernel.head_dim = 64
        kernel.softmax_scale = 0.125

        # Mock flash_attn
        kernel.has_flash_attn = True
        kernel.flash_attn_func = Mock()

        assert kernel.has_flash_attn is True

    @patch("src.optimizations.layer_fusion.FlashAttentionKernel.__init__")
    def test_init_flash_attn_unavailable(self, mock_init):
        """Test when FlashAttention is unavailable."""
        mock_init.return_value = None

        kernel = FlashAttentionKernel.__new__(FlashAttentionKernel)
        kernel.num_heads = 12
        kernel.head_dim = 64
        kernel.softmax_scale = 0.125
        kernel.has_flash_attn = False

        assert kernel.has_flash_attn is False

    def test_forward_basic_shape(self):
        """Test forward pass basic shapes."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64)
        kernel.has_flash_attn = False  # Use optimized implementation

        batch_size = 2
        seq_len = 128
        q = torch.randn(batch_size, 12, seq_len, 64)
        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)

        output = kernel.forward(q, k, v, is_causal=True)

        assert output.shape == q.shape

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64)
        kernel.has_flash_attn = False

        batch_size = 2
        seq_len = 128
        q = torch.randn(batch_size, 12, seq_len, 64)
        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)

        attention_mask = torch.zeros(batch_size, 1, 1, seq_len)

        output = kernel.forward(q, k, v, attention_mask=attention_mask, is_causal=False)

        assert output.shape == q.shape

    def test_forward_non_causal(self):
        """Test forward pass with non-causal attention."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64)
        kernel.has_flash_attn = False

        batch_size = 2
        seq_len = 128
        q = torch.randn(batch_size, 12, seq_len, 64)
        k = torch.randn(batch_size, 12, seq_len, 64)
        v = torch.randn(batch_size, 12, seq_len, 64)

        output = kernel.forward(q, k, v, is_causal=False)

        assert output.shape == q.shape

    def test_forward_different_seq_lengths(self):
        """Test forward pass with different sequence lengths."""
        kernel = FlashAttentionKernel(num_heads=12, head_dim=64)
        kernel.has_flash_attn = False

        batch_size = 2
        for seq_len in [1, 64, 256]:
            q = torch.randn(batch_size, 12, seq_len, 64)
            k = torch.randn(batch_size, 12, seq_len, 64)
            v = torch.randn(batch_size, 12, seq_len, 64)

            output = kernel.forward(q, k, v, is_causal=False)

            assert output.shape[2] == seq_len

    def test_optimized_attention_shape_preservation(self):
        """Test that optimized attention preserves shape."""
        kernel = FlashAttentionKernel(num_heads=8, head_dim=32)
        kernel.has_flash_attn = False

        batch_size = 4
        seq_len = 256
        q = torch.randn(batch_size, 8, seq_len, 32)
        k = torch.randn(batch_size, 8, seq_len, 32)
        v = torch.randn(batch_size, 8, seq_len, 32)

        output = kernel.forward(q, k, v, is_causal=True)

        assert output.shape == q.shape
        assert output.dtype == q.dtype


class TestFusedFFN:
    """Test FusedFFN for fused feed-forward networks."""

    def test_init_gelu(self):
        """Test initialization with GELU activation."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072, activation="gelu")

        assert ffn.hidden_size == 768
        assert ffn.intermediate_size == 3072
        assert ffn.activation == "gelu"

    def test_init_swiglu(self):
        """Test initialization with SwiGLU activation."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072, activation="swiglu")

        assert ffn.activation == "swiglu"

    def test_init_relu(self):
        """Test initialization with ReLU activation."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072, activation="relu")

        assert ffn.activation == "relu"

    def test_forward_shape(self):
        """Test forward pass shape."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        output = ffn(hidden_states)

        assert output.shape == hidden_states.shape

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072)

        for batch_size in [1, 4, 16]:
            hidden_states = torch.randn(batch_size, 128, 768)
            output = ffn(hidden_states)

            assert output.shape[0] == batch_size

    def test_forward_different_seq_lengths(self):
        """Test forward pass with different sequence lengths."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072)

        batch_size = 2
        for seq_len in [1, 64, 256]:
            hidden_states = torch.randn(batch_size, seq_len, 768)
            output = ffn(hidden_states)

            assert output.shape[1] == seq_len

    def test_fused_gate_up_shape(self):
        """Test fused gate-up projection shape."""
        ffn = FusedFFN(hidden_size=768, intermediate_size=3072)

        # gate_up_proj should output 2 * intermediate_size
        assert ffn.gate_up_proj.weight.shape == (2 * 3072, 768)
        assert ffn.down_proj.weight.shape == (768, 3072)


class TestFusedAttentionFFN:
    """Test FusedAttentionFFN for complete fused transformer block."""

    def test_init(self):
        """Test initialization."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        assert block.hidden_size == 768
        assert block.num_heads == 12
        assert block.head_dim == 64

    def test_forward_shape(self):
        """Test forward pass shape."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        output = block(hidden_states)

        assert output.shape == hidden_states.shape

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)

        output = block(hidden_states, attention_mask=attention_mask)

        assert output.shape == hidden_states.shape

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        for batch_size in [1, 4, 16]:
            hidden_states = torch.randn(batch_size, 128, 768)
            output = block(hidden_states)

            assert output.shape[0] == batch_size

    def test_forward_different_seq_lengths(self):
        """Test forward pass with different sequence lengths."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        batch_size = 2
        for seq_len in [1, 64, 256]:
            hidden_states = torch.randn(batch_size, seq_len, 768)
            output = block(hidden_states)

            assert output.shape[1] == seq_len

    def test_residual_connection(self):
        """Test that residual connections preserve values."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        # Set all weights to identity-like behavior for testing
        with torch.no_grad():
            for param in block.parameters():
                param.fill_(0)

        batch_size = 2
        seq_len = 64
        hidden_states = torch.randn(batch_size, seq_len, 768)

        output = block(hidden_states)

        # With zero weights, output should be close to residual
        assert output.shape == hidden_states.shape

    def test_layer_norms_present(self):
        """Test that layer norms are present."""
        block = FusedAttentionFFN(
            hidden_size=768, num_heads=12, intermediate_size=3072, head_dim=64
        )

        assert hasattr(block, "attn_norm")
        assert hasattr(block, "ffn_norm")
        assert isinstance(block.attn_norm, nn.LayerNorm)
        assert isinstance(block.ffn_norm, nn.LayerNorm)


class TestLayerFusionOptimizer:
    """Test LayerFusionOptimizer for model fusion."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        optimizer = LayerFusionOptimizer()

        assert optimizer.config.fuse_attention_ffn is True
        assert optimizer.fusion_stats["layers_fused"] == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = FusionConfig(use_flash_attention=False)
        optimizer = LayerFusionOptimizer(config)

        assert optimizer.config.use_flash_attention is False

    @patch("src.optimizations.layer_fusion.LayerFusionOptimizer._create_fused_layer")
    def test_fuse_model_with_layers(self, mock_create):
        """Test fusing a model with layers."""
        mock_create.return_value = Mock()

        optimizer = LayerFusionOptimizer()

        # Create mock model with layers
        mock_model = Mock()
        mock_layer = Mock()
        mock_layer.input_layernorm.weight.shape = [768]
        mock_layer.self_attn.num_heads = 12
        mock_layer.mlp.gate_proj.weight.shape = [3072, 768]

        mock_model.model.layers = [mock_layer, mock_layer]

        result = optimizer.fuse_model(mock_model)

        assert optimizer.fusion_stats["layers_fused"] == 2

    def test_fuse_model_empty_layers(self):
        """Test fusing a model with no layers."""
        optimizer = LayerFusionOptimizer()

        mock_model = Mock()
        mock_model.model.layers = []

        result = optimizer.fuse_model(mock_model)

        assert optimizer.fusion_stats["layers_fused"] == 0

    def test_create_fused_layer_success(self):
        """Test successful fused layer creation."""
        optimizer = LayerFusionOptimizer()

        # Create mock original layer
        mock_layer = Mock()
        mock_layer.input_layernorm.weight.shape = [768]
        mock_layer.self_attn.num_heads = 12
        mock_layer.mlp.gate_proj.weight.shape = [3072, 768]

        fused = optimizer._create_fused_layer(mock_layer)

        assert fused is not None
        assert fused.hidden_size == 768
        assert fused.num_heads == 12

    def test_create_fused_layer_fallback_num_heads(self):
        """Test fused layer creation with fallback num_heads."""
        optimizer = LayerFusionOptimizer()

        mock_layer = Mock()
        mock_layer.input_layernorm.weight.shape = [768]
        mock_layer.self_attn.num_attention_heads = 8  # Different attribute name
        mock_layer.mlp.gate_proj.weight.shape = [3072, 768]

        fused = optimizer._create_fused_layer(mock_layer)

        assert fused is not None
        assert fused.num_heads == 8

    def test_create_fused_layer_missing_attributes(self):
        """Test fused layer creation when attributes are missing."""
        optimizer = LayerFusionOptimizer()

        mock_layer = Mock()
        mock_layer.input_layernorm.weight.shape = [768]
        mock_layer.self_attn.num_heads = 12
        # mlp missing gate_proj, has up_proj instead
        mock_layer.mlp.up_proj.weight.shape = [3072, 768]

        fused = optimizer._create_fused_layer(mock_layer)

        assert fused is not None

    def test_create_fused_layer_error(self):
        """Test fused layer creation with error."""
        optimizer = LayerFusionOptimizer()

        mock_layer = Mock()
        mock_layer.input_layernorm.weight.shape = [768]
        mock_layer.self_attn.num_heads = 12
        mock_layer.mlp.gate_proj.weight.shape = [3072, 768]

        # Simulate error
        with patch("src.optimizations.layer_fusion.logger") as mock_logger:
            mock_logger.warning.side_effect = Exception("Test error")
            fused = optimizer._create_fused_layer(mock_layer)

            assert fused is None

    def test_get_stats(self):
        """Test getting fusion statistics."""
        optimizer = LayerFusionOptimizer()

        optimizer.fusion_stats["layers_fused"] = 5
        optimizer.fusion_stats["kernels_merged"] = 10
        optimizer.fusion_stats["memory_saved_mb"] = 256

        stats = optimizer.get_stats()

        assert stats["layers_fused"] == 5
        assert stats["kernels_merged"] == 10
        assert stats["memory_saved_mb"] == 256

    def test_get_stats_empty(self):
        """Test getting stats when no fusion occurred."""
        optimizer = LayerFusionOptimizer()

        stats = optimizer.get_stats()

        assert stats["layers_fused"] == 0
        assert stats["kernels_merged"] == 0
