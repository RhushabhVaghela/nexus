#!/usr/bin/env python3
"""
Unit tests for the Context Extension module.
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.context_extension import (
    ContextExtensionConfig, ScalingType, RoPEScaler,
    ContextExtender, create_context_extender
)


class TestScalingType:
    """Tests for ScalingType enum."""
    
    def test_scaling_types_exist(self):
        """Test all scaling types are defined."""
        assert ScalingType.LINEAR.value == "linear"
        assert ScalingType.DYNAMIC.value == "dynamic"
        assert ScalingType.YARN.value == "yarn"
        assert ScalingType.NTK.value == "ntk"
        assert ScalingType.LONGROPE.value == "longrope"


class TestContextExtensionConfig:
    """Tests for ContextExtensionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContextExtensionConfig()
        
        assert config.target_length == 131072  # 128K
        assert config.original_length == 32768
        assert config.scaling_type == ScalingType.YARN
        assert config.rope_theta == 10000.0
        assert config.use_flash_attention == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextExtensionConfig(
            target_length=262144,  # 256K
            original_length=8192,
            scaling_type=ScalingType.NTK
        )
        
        assert config.target_length == 262144
        assert config.original_length == 8192
        assert config.scaling_type == ScalingType.NTK


class TestRoPEScaler:
    """Tests for RoPEScaler."""
    
    def test_scaling_factor_calculation(self):
        """Test automatic scaling factor calculation."""
        config = ContextExtensionConfig(
            target_length=131072,
            original_length=32768
        )
        scaler = RoPEScaler(config)
        
        assert scaler.scaling_factor == 4.0  # 131072 / 32768
    
    def test_explicit_scaling_factor(self):
        """Test explicit scaling factor."""
        config = ContextExtensionConfig(
            rope_scaling_factor=8.0
        )
        scaler = RoPEScaler(config)
        
        assert scaler.scaling_factor == 8.0
    
    def test_linear_rope_config(self):
        """Test linear RoPE scaling config."""
        config = ContextExtensionConfig(scaling_type=ScalingType.LINEAR)
        scaler = RoPEScaler(config)
        
        rope_config = scaler.get_rope_config()
        
        assert rope_config["type"] == "linear"
        assert "factor" in rope_config
    
    def test_yarn_rope_config(self):
        """Test YaRN RoPE scaling config."""
        config = ContextExtensionConfig(scaling_type=ScalingType.YARN)
        scaler = RoPEScaler(config)
        
        rope_config = scaler.get_rope_config()
        
        assert rope_config["type"] == "yarn"
        assert "factor" in rope_config
        assert "beta_fast" in rope_config
        assert "beta_slow" in rope_config
        assert "attention_factor" in rope_config
    
    def test_ntk_rope_config(self):
        """Test NTK RoPE scaling config."""
        config = ContextExtensionConfig(scaling_type=ScalingType.NTK)
        scaler = RoPEScaler(config)
        
        rope_config = scaler.get_rope_config()
        
        assert rope_config["type"] == "ntk"
        assert "rope_theta" in rope_config
    
    def test_yarn_attention_factor(self):
        """Test YaRN attention factor calculation."""
        config = ContextExtensionConfig(
            target_length=131072,
            original_length=32768,
            scaling_type=ScalingType.YARN
        )
        scaler = RoPEScaler(config)
        
        attention_factor = scaler._compute_yarn_attention_factor()
        
        # For 4x scaling, should be > 1.0
        assert attention_factor > 1.0
        assert attention_factor < 2.0


class TestContextExtender:
    """Tests for ContextExtender."""
    
    def test_extender_initialization(self):
        """Test extender initialization."""
        extender = ContextExtender()
        
        assert extender.config is not None
        assert extender.scaler is not None
    
    def test_extender_with_custom_config(self):
        """Test extender with custom config."""
        config = ContextExtensionConfig(target_length=65536)
        extender = ContextExtender(config)
        
        assert extender.config.target_length == 65536
    
    def test_get_model_kwargs(self):
        """Test getting model loading kwargs."""
        extender = ContextExtender()
        kwargs = extender.get_model_kwargs()
        
        assert "rope_scaling" in kwargs
        assert "max_position_embeddings" in kwargs
        assert kwargs["max_position_embeddings"] == 131072
    
    def test_get_model_kwargs_with_flash_attention(self):
        """Test flash attention in model kwargs."""
        config = ContextExtensionConfig(use_flash_attention=True)
        extender = ContextExtender(config)
        kwargs = extender.get_model_kwargs()
        
        assert kwargs.get("attn_implementation") == "flash_attention_2"
    
    def test_prepare_training_config(self):
        """Test training config preparation."""
        config = ContextExtensionConfig(
            gradient_checkpointing=True,
            use_flash_attention=True,
            use_sliding_window=True,
            sliding_window_size=4096
        )
        extender = ContextExtender(config)
        
        training_config = extender.prepare_training_config()
        
        assert training_config["gradient_checkpointing"] == True
        assert training_config["use_flash_attention"] == True
        assert training_config["sliding_window"] == 4096


class TestCreateContextExtender:
    """Tests for create_context_extender factory function."""
    
    def test_create_default(self):
        """Test creating with defaults."""
        extender = create_context_extender()
        
        assert extender.config.target_length == 131072
        assert extender.config.scaling_type == ScalingType.YARN
    
    def test_create_custom(self):
        """Test creating with custom params."""
        extender = create_context_extender(
            target_length=262144,
            original_length=16384,
            scaling_type="linear"
        )
        
        assert extender.config.target_length == 262144
        assert extender.config.original_length == 16384
        assert extender.config.scaling_type == ScalingType.LINEAR
    
    def test_create_with_kwargs(self):
        """Test creating with additional kwargs."""
        extender = create_context_extender(
            target_length=65536,
            use_sliding_window=True,
            sliding_window_size=2048
        )
        
        assert extender.config.use_sliding_window == True
        assert extender.config.sliding_window_size == 2048


class TestContextExtensionIntegration:
    """Integration tests for context extension."""
    
    def test_full_workflow(self):
        """Test complete context extension workflow."""
        # Create extender
        extender = create_context_extender(
            target_length=131072,
            original_length=32768,
            scaling_type="yarn"
        )
        
        # Get model kwargs
        kwargs = extender.get_model_kwargs()
        
        # Verify config
        assert kwargs["max_position_embeddings"] == 131072
        assert kwargs["rope_scaling"]["type"] == "yarn"
        assert kwargs["rope_scaling"]["factor"] == 4.0
        
        # Get training config
        training_config = extender.prepare_training_config()
        assert training_config["max_length"] == 131072
    
    def test_scaling_types_produce_valid_configs(self):
        """Test all scaling types produce valid configs."""
        for scaling_type in ["linear", "dynamic", "yarn", "ntk"]:
            extender = create_context_extender(
                target_length=65536,
                scaling_type=scaling_type
            )
            
            kwargs = extender.get_model_kwargs()
            
            assert kwargs["rope_scaling"]["type"] == scaling_type
            assert "factor" in kwargs["rope_scaling"] or "rope_theta" in kwargs["rope_scaling"]
