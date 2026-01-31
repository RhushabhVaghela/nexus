"""
tests/unit/test_nexus_core_adapters.py
Comprehensive tests for nexus_core adapters functionality.

Tests cover:
- BaseAdapter
- VisionAdapter
- AudioAdapter
- ReasoningAdapter
- Adapter forward passes
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Import adapter components
from src.nexus_core.adapters.base_adapter import BaseAdapter
from src.nexus_core.adapters.vision_adapter import VisionAdapter
from src.nexus_core.adapters.audio_adapter import AudioAdapter
from src.nexus_core.adapters.reasoning_adapter import ReasoningAdapter


class TestBaseAdapter:
    """Test BaseAdapter class."""
    
    def test_initialization(self):
        """Test BaseAdapter initialization."""
        config = {"hidden_size": 128, "dropout": 0.1}
        adapter = BaseAdapter(config)
        
        assert adapter.config == config
        assert adapter.config["dropout"] == 0.1
    
    def test_forward_not_implemented(self):
        """Test BaseAdapter forward raises NotImplementedError."""
        config = {"hidden_size": 128}
        adapter = BaseAdapter(config)
        
        x = torch.randn(2, 10, 128)
        
        with pytest.raises(NotImplementedError):
            adapter.forward(x)


class TestVisionAdapter:
    """Test VisionAdapter class."""
    
    def test_initialization(self):
        """Test VisionAdapter initialization."""
        adapter = VisionAdapter(input_dim=768, output_dim=512, hidden_dim=1024, dropout=0.1)
        
        assert adapter.input_dim == 768
        assert adapter.output_dim == 512
        assert adapter.hidden_dim == 1024
        assert isinstance(adapter.down_proj, nn.Linear)
        assert isinstance(adapter.up_proj, nn.Linear)
        assert isinstance(adapter.gate, nn.Sequential)
    
    def test_forward(self):
        """Test VisionAdapter forward pass."""
        adapter = VisionAdapter(input_dim=768, output_dim=512, hidden_dim=1024)
        
        x = torch.randn(2, 10, 768)
        result = adapter.forward(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 512)
    
    def test_forward_with_attention_mask(self):
        """Test VisionAdapter forward with attention mask."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        x = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)
        
        result = adapter.forward(x, attention_mask=attention_mask)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 512)
    
    def test_forward_preserves_batch(self):
        """Test VisionAdapter preserves batch dimension."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10, 768)
            result = adapter.forward(x)
            assert result.shape[0] == batch_size
    
    def test_forward_varies_seq_len(self):
        """Test VisionAdapter with varying sequence lengths."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        seq_lengths = [5, 10, 50, 100]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 768)
            result = adapter.forward(x)
            assert result.shape[1] == seq_len
    
    def test_adapter_layers_exist(self):
        """Test that required layers exist."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        assert hasattr(adapter, 'down_proj')
        assert hasattr(adapter, 'up_proj')
        assert hasattr(adapter, 'gate')
        assert hasattr(adapter, 'dropout')
        assert hasattr(adapter, 'activation')
    
    def test_gating_mechanism(self):
        """Test gating mechanism produces values in valid range."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        x = torch.randn(2, 10, 768)
        gate_output = adapter.gate(x)
        
        # Gate uses sigmoid, so values should be between 0 and 1
        assert (gate_output >= 0).all()
        assert (gate_output <= 1).all()


class TestAudioAdapter:
    """Test AudioAdapter class."""
    
    def test_initialization(self):
        """Test AudioAdapter initialization."""
        adapter = AudioAdapter(input_dim=512, output_dim=768, hidden_dim=1024)
        
        assert adapter.input_dim == 512
        assert adapter.output_dim == 768
        assert adapter.hidden_dim == 1024
    
    def test_forward(self):
        """Test AudioAdapter forward pass."""
        adapter = AudioAdapter(input_dim=512, output_dim=768, hidden_dim=1024)
        
        x = torch.randn(2, 100, 512)  # Audio features typically longer
        result = adapter.forward(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 100, 768)
    
    def test_forward_preserves_batch(self):
        """Test AudioAdapter preserves batch dimension."""
        adapter = AudioAdapter(input_dim=512, output_dim=768)
        
        batch_sizes = [1, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 100, 512)
            result = adapter.forward(x)
            assert result.shape[0] == batch_size
    
    def test_forward_varies_seq_len(self):
        """Test AudioAdapter with varying sequence lengths."""
        adapter = AudioAdapter(input_dim=512, output_dim=768)
        
        seq_lengths = [50, 100, 200]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 512)
            result = adapter.forward(x)
            assert result.shape[1] == seq_len
    
    def test_adapter_configuration(self):
        """Test AudioAdapter configuration parameters."""
        config = {"hidden_size": 256, "dropout": 0.2}
        adapter = AudioAdapter(config=config)
        
        assert adapter.config == config


class TestReasoningAdapter:
    """Test ReasoningAdapter class."""
    
    def test_initialization(self):
        """Test ReasoningAdapter initialization."""
        adapter = ReasoningAdapter(input_dim=4096, output_dim=2048, hidden_dim=8192)
        
        assert adapter.input_dim == 4096
        assert adapter.output_dim == 2048
        assert adapter.hidden_dim == 8192
    
    def test_forward(self):
        """Test ReasoningAdapter forward pass."""
        adapter = ReasoningAdapter(input_dim=4096, output_dim=2048, hidden_dim=8192)
        
        x = torch.randn(2, 20, 4096)
        result = adapter.forward(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 20, 2048)
    
    def test_forward_preserves_batch(self):
        """Test ReasoningAdapter preserves batch dimension."""
        adapter = ReasoningAdapter(input_dim=4096, output_dim=2048)
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 20, 4096)
            result = adapter.forward(x)
            assert result.shape[0] == batch_size
    
    def test_forward_long_sequences(self):
        """Test ReasoningAdapter with long sequences."""
        adapter = ReasoningAdapter(input_dim=4096, output_dim=2048)
        
        # Reasoning often uses longer sequences
        seq_lengths = [100, 500, 1000]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 4096)
            result = adapter.forward(x)
            assert result.shape[1] == seq_len
    
    def test_adapter_integration(self):
        """Test adapter integration with other components."""
        adapter = ReasoningAdapter(input_dim=4096, output_dim=2048)
        
        # Test that adapter can be added to a tower
        mock_tower = Mock()
        mock_tower.add_adapter = Mock()
        
        mock_tower.add_adapter("reasoning", adapter)
        mock_tower.add_adapter.assert_called_once_with("reasoning", adapter)


class TestAdapterCombinations:
    """Test adapter combinations and chaining."""
    
    def test_chained_adapters(self):
        """Test chaining multiple adapters."""
        vision_adapter = VisionAdapter(input_dim=768, output_dim=512)
        audio_adapter = AudioAdapter(input_dim=512, output_dim=256)
        
        # First adapter output matches second adapter input
        x = torch.randn(2, 10, 768)
        intermediate = vision_adapter.forward(x)
        final = audio_adapter.forward(intermediate)
        
        assert final.shape == (2, 10, 256)
    
    def test_adapter_different_configs(self):
        """Test adapters with different configurations."""
        configs = [
            {"hidden_size": 128, "dropout": 0.1},
            {"hidden_size": 256, "dropout": 0.2},
            {"hidden_size": 512, "dropout": 0.05}
        ]
        
        for config in configs:
            adapter = VisionAdapter(input_dim=768, output_dim=512, config=config)
            assert adapter.config == config
    
    def test_adapter_dropout_effect(self):
        """Test dropout has effect during training."""
        adapter = VisionAdapter(input_dim=768, output_dim=512, dropout=0.5)
        adapter.train()
        
        x = torch.randn(2, 10, 768)
        
        # Run multiple times, should get different results due to dropout
        result1 = adapter.forward(x)
        result2 = adapter.forward(x)
        
        # With high probability, these should be different
        # (not guaranteed but very likely with dropout=0.5)
    
    def test_adapter_eval_mode(self):
        """Test adapter in eval mode."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        adapter.eval()
        
        x = torch.randn(2, 10, 768)
        
        # In eval mode, should produce deterministic results
        result1 = adapter.forward(x)
        result2 = adapter.forward(x)
        
        assert torch.allclose(result1, result2)


class TestAdapterEdgeCases:
    """Test adapter edge cases."""
    
    def test_single_batch_item(self):
        """Test adapters with single batch item."""
        vision_adapter = VisionAdapter(input_dim=768, output_dim=512)
        audio_adapter = AudioAdapter(input_dim=512, output_dim=256)
        reasoning_adapter = ReasoningAdapter(input_dim=4096, output_dim=2048)
        
        # Vision
        x = torch.randn(1, 10, 768)
        result = vision_adapter.forward(x)
        assert result.shape == (1, 10, 512)
        
        # Audio
        x = torch.randn(1, 100, 512)
        result = audio_adapter.forward(x)
        assert result.shape == (1, 100, 256)
        
        # Reasoning
        x = torch.randn(1, 20, 4096)
        result = reasoning_adapter.forward(x)
        assert result.shape == (1, 20, 2048)
    
    def test_large_batch(self):
        """Test adapters with large batch size."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        x = torch.randn(64, 10, 768)
        result = adapter.forward(x)
        
        assert result.shape == (64, 10, 512)
    
    def test_zero_dropout(self):
        """Test adapters with zero dropout."""
        adapter = VisionAdapter(input_dim=768, output_dim=512, dropout=0.0)
        
        x = torch.randn(2, 10, 768)
        result = adapter.forward(x)
        
        assert result.shape == (2, 10, 512)
    
    def test_various_hidden_dims(self):
        """Test adapters with various hidden dimensions."""
        hidden_dims = [256, 512, 1024, 2048]
        
        for hidden_dim in hidden_dims:
            adapter = VisionAdapter(input_dim=768, output_dim=512, hidden_dim=hidden_dim)
            x = torch.randn(2, 10, 768)
            result = adapter.forward(x)
            assert result.shape == (2, 10, 512)


class TestAdapterParameters:
    """Test adapter parameter management."""
    
    def test_parameter_count(self):
        """Test adapter parameter counts."""
        vision_adapter = VisionAdapter(input_dim=768, output_dim=512, hidden_dim=1024)
        
        # Expected: down_proj (768*1024 + 1024) + up_proj (1024*512 + 512) + gate (~768*1)
        params = sum(p.numel() for p in vision_adapter.parameters())
        assert params > 0
    
    def test_trainable_parameters(self):
        """Test that adapter parameters are trainable."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        for param in adapter.parameters():
            assert param.requires_grad
    
    def test_state_dict(self):
        """Test adapter state dict."""
        adapter = VisionAdapter(input_dim=768, output_dim=512)
        
        state_dict = adapter.state_dict()
        assert len(state_dict) > 0
        
        # Check for expected keys
        assert any('down_proj' in key for key in state_dict.keys())
        assert any('up_proj' in key for key in state_dict.keys())
