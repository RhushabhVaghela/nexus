"""
Unit tests for nexus_core adapters.

Tests the actual adapter implementations:
- BaseAdapter
- VisionAdapter  
- AudioAdapter
- ReasoningAdapter
"""

import pytest
import torch
import torch.nn as nn

# Import adapter components
from src.core.adapters.base import BaseAdapter
from src.core.adapters.vision_adapter import VisionAdapter
from src.core.adapters.audio_adapter import AudioAdapter
from src.core.adapters.reasoning_adapter import ReasoningAdapter


class TestBaseAdapter:
    """Test BaseAdapter class."""
    
    def test_initialization(self):
        """Test BaseAdapter initialization with teacher_dim and student_dim."""
        adapter = BaseAdapter(teacher_dim=1024, student_dim=512)
        
        # BaseAdapter doesn't store dims directly, but has projection layers
        assert hasattr(adapter, 'proj')
        assert hasattr(adapter, 'scale')
        assert hasattr(adapter, 'shift')
        # Check projection dimensions
        assert adapter.proj.in_features == 1024
        assert adapter.proj.out_features == 512
    
    def test_forward(self):
        """Test BaseAdapter forward pass."""
        adapter = BaseAdapter(teacher_dim=1024, student_dim=512)
        
        x = torch.randn(2, 10, 1024)
        result = adapter(x)
        
        assert result.shape == (2, 10, 512)
    
    def test_forward_different_dimensions(self):
        """Test BaseAdapter with different dimensions."""
        test_cases = [
            (768, 384),
            (4096, 2048),
            (512, 256),
        ]
        
        for teacher_dim, student_dim in test_cases:
            adapter = BaseAdapter(teacher_dim=teacher_dim, student_dim=student_dim)
            x = torch.randn(4, 20, teacher_dim)
            result = adapter(x)
            assert result.shape == (4, 20, student_dim)
    
    def test_parameters_trainable(self):
        """Test that adapter parameters are trainable."""
        adapter = BaseAdapter(teacher_dim=1024, student_dim=512)
        
        for param in adapter.parameters():
            assert param.requires_grad
    
    def test_state_dict(self):
        """Test adapter state dict has expected keys."""
        adapter = BaseAdapter(teacher_dim=1024, student_dim=512)
        
        state_dict = adapter.state_dict()
        assert len(state_dict) > 0
        assert any('proj' in key for key in state_dict.keys())


class TestVisionAdapter:
    """Test VisionAdapter class."""
    
    def test_initialization(self):
        """Test VisionAdapter initialization."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        assert adapter.teacher_dim == 768
        assert adapter.student_dim == 512
        assert hasattr(adapter, 'alignment')
        assert hasattr(adapter, 'gate_proj')
    
    def test_forward(self):
        """Test VisionAdapter forward pass."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        x = torch.randn(2, 10, 768)
        result, gate_score = adapter(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 512)
        assert gate_score.shape == (2, 10, 1)
        # Gate uses sigmoid, values should be between 0 and 1
        assert (gate_score >= 0).all()
        assert (gate_score <= 1).all()
    
    def test_forward_varies_seq_len(self):
        """Test VisionAdapter with varying sequence lengths."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        seq_lengths = [5, 10, 50, 100]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 768)
            result, gate_score = adapter(x)
            assert result.shape == (2, seq_len, 512)
    
    def test_forward_preserves_batch(self):
        """Test VisionAdapter preserves batch dimension."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10, 768)
            result, gate_score = adapter(x)
            assert result.shape[0] == batch_size


class TestAudioAdapter:
    """Test AudioAdapter class."""
    
    def test_initialization(self):
        """Test AudioAdapter initialization."""
        adapter = AudioAdapter(teacher_dim=512, student_dim=768)
        
        assert adapter.teacher_dim == 512
        assert adapter.student_dim == 768
        assert hasattr(adapter, 'alignment')
        assert hasattr(adapter, 'gate_proj')
    
    def test_forward(self):
        """Test AudioAdapter forward pass."""
        adapter = AudioAdapter(teacher_dim=512, student_dim=768)
        
        x = torch.randn(2, 100, 512)  # Audio features typically longer
        result, gate_score = adapter(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 100, 768)
        assert gate_score.shape == (2, 100, 1)
    
    def test_forward_preserves_batch(self):
        """Test AudioAdapter preserves batch dimension."""
        adapter = AudioAdapter(teacher_dim=512, student_dim=768)
        
        batch_sizes = [1, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 100, 512)
            result, gate_score = adapter(x)
            assert result.shape[0] == batch_size


class TestReasoningAdapter:
    """Test ReasoningAdapter class."""
    
    def test_initialization(self):
        """Test ReasoningAdapter initialization."""
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        
        assert adapter.teacher_dim == 4096
        assert adapter.student_dim == 2048
        assert hasattr(adapter, 'alignment')
        assert hasattr(adapter, 'gate_proj')
    
    def test_forward(self):
        """Test ReasoningAdapter forward pass."""
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        
        x = torch.randn(2, 20, 4096)
        result, gate_score = adapter(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 20, 2048)
        assert gate_score.shape == (2, 20, 1)
    
    def test_forward_preserves_batch(self):
        """Test ReasoningAdapter preserves batch dimension."""
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 20, 4096)
            result, gate_score = adapter(x)
            assert result.shape[0] == batch_size
    
    def test_forward_long_sequences(self):
        """Test ReasoningAdapter with long sequences."""
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        
        seq_lengths = [100, 500, 1000]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 4096)
            result, gate_score = adapter(x)
            assert result.shape[1] == seq_len


class TestAdapterCombinations:
    """Test adapter combinations and chaining."""
    
    def test_chained_adapters(self):
        """Test chaining multiple adapters."""
        vision_adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        audio_adapter = AudioAdapter(teacher_dim=512, student_dim=256)
        
        # First adapter output matches second adapter input
        x = torch.randn(2, 10, 768)
        intermediate, _ = vision_adapter(x)
        final, _ = audio_adapter(intermediate)
        
        assert final.shape == (2, 10, 256)


class TestAdapterEdgeCases:
    """Test adapter edge cases."""
    
    def test_single_batch_item(self):
        """Test adapters with single batch item."""
        vision_adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        audio_adapter = AudioAdapter(teacher_dim=512, student_dim=256)
        reasoning_adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        
        # Vision
        x = torch.randn(1, 10, 768)
        result, _ = vision_adapter(x)
        assert result.shape == (1, 10, 512)
        
        # Audio
        x = torch.randn(1, 100, 512)
        result, _ = audio_adapter(x)
        assert result.shape == (1, 100, 256)
        
        # Reasoning
        x = torch.randn(1, 20, 4096)
        result, _ = reasoning_adapter(x)
        assert result.shape == (1, 20, 2048)
    
    def test_large_batch(self):
        """Test adapters with large batch size."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        x = torch.randn(64, 10, 768)
        result, _ = adapter(x)
        
        assert result.shape == (64, 10, 512)
    
    def test_parameter_count(self):
        """Test adapter parameter counts are reasonable."""
        vision_adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        params = sum(p.numel() for p in vision_adapter.parameters())
        assert params > 0
    
    def test_trainable_parameters(self):
        """Test that adapter parameters are trainable."""
        adapter = VisionAdapter(teacher_dim=768, student_dim=512)
        
        for param in adapter.parameters():
            assert param.requires_grad
