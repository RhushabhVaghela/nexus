#!/usr/bin/env python3
"""
Unit tests for Ring Attention module.
"""

import pytest
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.ring_attention import (
    RingAttention, RingAttentionConfig, RingAttentionWrapper,
    RingCommunicator, create_ring_attention
)


class TestRingAttentionConfig:
    """Tests for RingAttentionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RingAttentionConfig()
        
        assert config.block_size == 2048
        assert config.max_seq_length == 524288  # 512K
        assert config.world_size == 1
        assert config.use_flash_attention == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RingAttentionConfig(
            block_size=1024,
            max_seq_length=1048576,  # 1M
            num_heads=64,
            head_dim=64,
        )
        
        assert config.block_size == 1024
        assert config.max_seq_length == 1048576
        assert config.num_heads == 64


class TestRingCommunicator:
    """Tests for RingCommunicator."""
    
    def test_initialization(self):
        """Test communicator initialization."""
        config = RingAttentionConfig()
        comm = RingCommunicator(config)
        
        assert comm.rank == 0
        assert comm.world_size == 1
        assert comm._initialized == False
    
    def test_single_gpu_ring_send_recv(self):
        """Test ring communication falls back in single GPU mode."""
        config = RingAttentionConfig(world_size=1)
        comm = RingCommunicator(config)
        
        send_tensor = torch.randn(4, 8)
        recv_tensor = torch.empty_like(send_tensor)
        
        result = comm.ring_send_recv(send_tensor, recv_tensor)
        
        # In single GPU mode, should return send tensor unchanged
        assert torch.equal(result, send_tensor)


class TestRingAttention:
    """Tests for RingAttention module."""
    
    def test_initialization(self):
        """Test ring attention initialization."""
        config = RingAttentionConfig(
            block_size=64,
            num_heads=8,
            head_dim=32,
        )
        ring_attn = RingAttention(config)
        
        assert ring_attn.config == config
        assert ring_attn.scale == 1.0 / (32 ** 0.5)
    
    def test_forward_single_gpu(self):
        """Test forward pass on single GPU."""
        config = RingAttentionConfig(
            block_size=64,
            num_heads=8,
            head_dim=32,
            world_size=1,
        )
        ring_attn = RingAttention(config)
        
        batch_size = 2
        seq_len = 64
        
        q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        k = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        v = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        
        output = ring_attn(q, k, v, is_causal=True)
        
        assert output.shape == (batch_size, seq_len, config.num_heads, config.head_dim)
    
    def test_forward_non_causal(self):
        """Test forward pass without causal masking."""
        config = RingAttentionConfig(
            block_size=32,
            num_heads=4,
            head_dim=16,
            world_size=1,
        )
        ring_attn = RingAttention(config)
        
        batch_size = 1
        seq_len = 32
        
        q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        k = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        v = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        
        output = ring_attn(q, k, v, is_causal=False)
        
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
    
    def test_blockwise_attention(self):
        """Test blockwise attention computation."""
        config = RingAttentionConfig(
            block_size=32,
            num_heads=4,
            head_dim=16,
        )
        ring_attn = RingAttention(config)
        
        batch_size = 1
        seq_len = 32
        
        q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        k = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        v = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        
        output, lse = ring_attn._blockwise_attention(q, k, v, 0, 0, is_causal=True)
        
        assert output.shape == q.shape
        assert lse.shape == (batch_size, seq_len, config.num_heads)
    
    def test_online_softmax_update(self):
        """Test online softmax accumulation."""
        config = RingAttentionConfig(num_heads=4, head_dim=16)
        ring_attn = RingAttention(config)
        
        batch_size = 1
        seq_len = 32
        
        acc_output = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        acc_lse = torch.randn(batch_size, seq_len, config.num_heads)
        new_output = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        new_lse = torch.randn(batch_size, seq_len, config.num_heads)
        
        combined_output, combined_lse = ring_attn._online_softmax_update(
            acc_output, acc_lse, new_output, new_lse
        )
        
        assert combined_output.shape == acc_output.shape
        assert combined_lse.shape == acc_lse.shape


class TestRingAttentionWrapper:
    """Tests for RingAttentionWrapper."""
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        wrapper = RingAttentionWrapper()
        
        assert wrapper.config is not None
        assert wrapper.ring_attn is not None
    
    def test_wrapper_with_custom_config(self):
        """Test wrapper with custom config."""
        config = RingAttentionConfig(max_seq_length=1048576)
        wrapper = RingAttentionWrapper(config)
        
        assert wrapper.config.max_seq_length == 1048576
    
    def test_get_model_kwargs(self):
        """Test getting model kwargs."""
        wrapper = RingAttentionWrapper()
        kwargs = wrapper.get_model_kwargs()
        
        assert "attn_implementation" in kwargs
        assert "ring_config" in kwargs


class TestCreateRingAttention:
    """Tests for factory function."""
    
    def test_create_default(self):
        """Test creating with defaults."""
        wrapper = create_ring_attention()
        
        assert wrapper.config.max_seq_length == 524288
        assert wrapper.config.block_size == 2048
    
    def test_create_custom(self):
        """Test creating with custom params."""
        wrapper = create_ring_attention(
            max_seq_length=1048576,
            block_size=1024,
            num_heads=64,
            head_dim=64,
        )
        
        assert wrapper.config.max_seq_length == 1048576
        assert wrapper.config.block_size == 1024
        assert wrapper.config.num_heads == 64


class TestRingAttentionMemory:
    """Memory usage tests for single GPU compatibility."""
    
    def test_memory_efficiency(self):
        """Test that ring attention works within memory limits."""
        config = RingAttentionConfig(
            block_size=512,
            num_heads=8,
            head_dim=64,
            world_size=1,
        )
        ring_attn = RingAttention(config)
        
        # Simulate moderate sequence (single GPU mode)
        batch_size = 1
        seq_len = 512
        
        q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        k = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        v = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim)
        
        # Should complete without OOM
        output = ring_attn(q, k, v)
        
        assert output is not None
