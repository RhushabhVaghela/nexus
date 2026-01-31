"""
Test suite for the MoE (Mixture of Experts) Handler module.

This module tests the MoEHandler and MoEConfig classes which provide
specialized handling for MoE architectures including Mixtral, Qwen2-MoE,
DeepSeek-MoE, and other MoE variants.

Total test cases: ~35
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.nexus_final.sli.moe_handler import MoEHandler, MoEConfig
from src.nexus_final.sli.exceptions import MoEConfigurationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_mixtral_config():
    """Create a mock Mixtral config."""
    config = MagicMock()
    config.model_type = "mixtral"
    config.architectures = ["MixtralForCausalLM"]
    config.num_local_experts = 8
    config.num_experts_per_tok = 2
    return config


@pytest.fixture
def mock_qwen2_moe_config():
    """Create a mock Qwen2-MoE config."""
    config = MagicMock()
    config.model_type = "qwen2_moe"
    config.architectures = ["Qwen2MoeForCausalLM"]
    config.num_experts = 60
    config.num_experts_per_tok = 4
    return config


@pytest.fixture
def mock_deepseek_moe_config():
    """Create a mock DeepSeek-MoE config."""
    config = MagicMock()
    config.model_type = "deepseek"
    config.architectures = ["DeepseekMoeForCausalLM"]
    config.n_routed_experts = 64
    config.n_shared_experts = 2
    config.num_experts_per_tok = 6
    return config


@pytest.fixture
def mock_grok_config():
    """Create a mock Grok config."""
    config = MagicMock()
    config.model_type = "grok"
    config.architectures = ["GrokForCausalLM"]
    config.num_experts = 8
    config.top_k = 2
    return config


@pytest.fixture
def mock_glm4_moe_config():
    """Create a mock GLM4-MoE config."""
    config = MagicMock()
    config.model_type = "glm4_moe"
    config.architectures = ["Glm4MoeForCausalLM"]
    config.num_experts = 8
    config.top_k = 2
    return config


# =============================================================================
# Test MoEConfig Dataclass
# =============================================================================

class TestMoEConfig:
    """Test MoEConfig dataclass."""
    
    def test_moe_config_default_values(self):
        """Test MoEConfig default values."""
        config = MoEConfig()
        
        assert config.num_experts == 8
        assert config.top_k == 2
        assert config.moe_type == "mixtral"
        assert config.has_shared_experts is False
        assert config.num_shared_experts == 0
        assert config.expert_capacity is None
        assert config.router_aux_loss_coef == 0.001
    
    def test_moe_config_custom_values(self):
        """Test MoEConfig with custom values."""
        config = MoEConfig(
            num_experts=16,
            top_k=4,
            moe_type="deepseek",
            has_shared_experts=True,
            num_shared_experts=2,
            expert_capacity=1024,
            router_aux_loss_coef=0.01
        )
        
        assert config.num_experts == 16
        assert config.top_k == 4
        assert config.moe_type == "deepseek"
        assert config.has_shared_experts is True
        assert config.num_shared_experts == 2
        assert config.expert_capacity == 1024
        assert config.router_aux_loss_coef == 0.01
    
    def test_moe_config_validation_top_k_greater_than_experts(self):
        """Test MoEConfig validation when top_k > num_experts."""
        with pytest.raises(MoEConfigurationError) as exc_info:
            MoEConfig(num_experts=4, top_k=8)
        
        assert "top_k (8) cannot be greater than num_experts (4)" in str(exc_info.value)
    
    def test_moe_config_validation_valid(self):
        """Test MoEConfig validation passes with valid config."""
        # Should not raise
        config = MoEConfig(num_experts=8, top_k=8)
        assert config.top_k == 8


# =============================================================================
# Test MoEHandler Initialization
# =============================================================================

class TestMoEHandlerInitialization:
    """Test MoEHandler initialization."""
    
    def test_handler_initializes_with_mixtral_config(self, mock_mixtral_config):
        """Test handler initializes with Mixtral config."""
        handler = MoEHandler(mock_mixtral_config)
        
        assert handler.config is mock_mixtral_config
        assert handler.moe_config.moe_type == "mixtral"
        assert handler.moe_config.num_experts == 8
        assert handler.moe_config.top_k == 2
    
    def test_handler_initializes_with_qwen2_moe_config(self, mock_qwen2_moe_config):
        """Test handler initializes with Qwen2-MoE config."""
        handler = MoEHandler(mock_qwen2_moe_config)
        
        assert handler.moe_config.moe_type == "qwen2_moe"
        assert handler.moe_config.num_experts == 60
        assert handler.moe_config.top_k == 4
    
    def test_handler_initializes_with_deepseek_moe_config(self, mock_deepseek_moe_config):
        """Test handler initializes with DeepSeek-MoE config."""
        handler = MoEHandler(mock_deepseek_moe_config)
        
        assert handler.moe_config.moe_type == "deepseek"
        assert handler.moe_config.num_experts == 64
        assert handler.moe_config.top_k == 6
        assert handler.moe_config.has_shared_experts is True
        assert handler.moe_config.num_shared_experts == 2
    
    def test_handler_initializes_with_grok_config(self, mock_grok_config):
        """Test handler initializes with Grok config."""
        handler = MoEHandler(mock_grok_config)
        
        assert handler.moe_config.moe_type == "grok"
        assert handler.moe_config.num_experts == 8
        assert handler.moe_config.top_k == 2
    
    def test_handler_initializes_with_glm4_moe_config(self, mock_glm4_moe_config):
        """Test handler initializes with GLM4-MoE config."""
        handler = MoEHandler(mock_glm4_moe_config)
        
        assert handler.moe_config.moe_type == "glm4_moe"
        assert handler.moe_config.num_experts == 8
        assert handler.moe_config.top_k == 2


# =============================================================================
# Test MoE Type Detection
# =============================================================================

class TestMoETypeDetection:
    """Test MoE type detection from config."""
    
    def test_detect_mixtral_by_model_type(self):
        """Test Mixtral detection by model_type."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = []
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "mixtral"
    
    def test_detect_mixtral_by_architecture(self):
        """Test Mixtral detection by architecture."""
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "mixtral"
    
    def test_detect_qwen2_moe_by_model_type(self):
        """Test Qwen2-MoE detection by model_type."""
        config = MagicMock()
        config.model_type = "qwen2_moe"
        config.architectures = []
        config.num_experts = 60
        config.num_experts_per_tok = 4
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "qwen2_moe"
    
    def test_detect_deepseek_by_model_type(self):
        """Test DeepSeek detection by model_type."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = []
        config.n_routed_experts = 64
        config.num_experts_per_tok = 6
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "deepseek"
    
    def test_detect_grok_by_model_type(self):
        """Test Grok detection by model_type."""
        config = MagicMock()
        config.model_type = "grok"
        config.architectures = []
        config.num_experts = 8
        config.top_k = 2
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "grok"
    
    def test_detect_glm4_moe_by_model_type(self):
        """Test GLM4-MoE detection by model_type."""
        config = MagicMock()
        config.model_type = "glm4_moe"
        config.architectures = []
        config.num_experts = 8
        config.top_k = 2
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "glm4_moe"
    
    def test_default_to_mixtral_when_unknown(self):
        """Test default to mixtral for unknown MoE types."""
        config = MagicMock()
        config.model_type = "unknown_moe"
        config.architectures = []
        config.num_experts = 8
        config.top_k = 2
        
        handler = MoEHandler(config)
        assert handler.moe_config.moe_type == "mixtral"


# =============================================================================
# Test MoE Layer Detection
# =============================================================================

class TestMoELayerDetection:
    """Test MoE layer detection."""
    
    def test_is_moe_layer_default_true(self, mock_mixtral_config):
        """Test default MoE layer detection (all layers are MoE)."""
        handler = MoEHandler(mock_mixtral_config)
        
        # By default, all layers are MoE
        assert handler.is_moe_layer(0) is True
        assert handler.is_moe_layer(5) is True
        assert handler.is_moe_layer(31) is True
    
    def test_is_moe_layer_with_interval(self):
        """Test MoE layer detection with interval."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        config.moe_layer_interval = 2
        
        handler = MoEHandler(config)
        
        # Every other layer is MoE
        assert handler.is_moe_layer(0) is True
        assert handler.is_moe_layer(1) is False
        assert handler.is_moe_layer(2) is True
        assert handler.is_moe_layer(3) is False
    
    def test_is_moe_layer_with_indices(self):
        """Test MoE layer detection with specific indices."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.n_routed_experts = 64
        config.num_experts_per_tok = 6
        config.moe_layer_indices = [1, 3, 5, 7]
        
        handler = MoEHandler(config)
        
        assert handler.is_moe_layer(0) is False
        assert handler.is_moe_layer(1) is True
        assert handler.is_moe_layer(2) is False
        assert handler.is_moe_layer(3) is True
        assert handler.is_moe_layer(7) is True


# =============================================================================
# Test Expert Weight Pattern Generation
# =============================================================================

class TestExpertWeightPattern:
    """Test expert weight pattern generation."""
    
    def test_get_expert_weight_pattern_mixtral(self, mock_mixtral_config):
        """Test expert weight pattern for Mixtral."""
        handler = MoEHandler(mock_mixtral_config)
        
        pattern = handler.get_expert_weight_pattern(5, 3)
        assert pattern == "model.layers.5.block_sparse_moe.experts.3."
    
    def test_get_expert_weight_pattern_qwen2_moe(self, mock_qwen2_moe_config):
        """Test expert weight pattern for Qwen2-MoE."""
        handler = MoEHandler(mock_qwen2_moe_config)
        
        pattern = handler.get_expert_weight_pattern(7, 5)
        assert pattern == "model.layers.7.mlp.experts.5."
    
    def test_get_expert_weight_pattern_deepseek(self, mock_deepseek_moe_config):
        """Test expert weight pattern for DeepSeek-MoE."""
        handler = MoEHandler(mock_deepseek_moe_config)
        
        pattern = handler.get_expert_weight_pattern(10, 15)
        assert pattern == "model.layers.10.mlp.experts.15."
    
    def test_get_expert_weight_pattern_grok(self, mock_grok_config):
        """Test expert weight pattern for Grok."""
        handler = MoEHandler(mock_grok_config)
        
        pattern = handler.get_expert_weight_pattern(2, 4)
        assert pattern == "model.layers.2.moe_block.experts.4."


# =============================================================================
# Test Router Weight Pattern Generation
# =============================================================================

class TestRouterWeightPattern:
    """Test router/gate weight pattern generation."""
    
    def test_get_router_weight_pattern_mixtral(self, mock_mixtral_config):
        """Test router weight pattern for Mixtral."""
        handler = MoEHandler(mock_mixtral_config)
        
        pattern = handler.get_router_weight_pattern(5)
        assert pattern == "model.layers.5.block_sparse_moe.gate."
    
    def test_get_router_weight_pattern_qwen2_moe(self, mock_qwen2_moe_config):
        """Test router weight pattern for Qwen2-MoE."""
        handler = MoEHandler(mock_qwen2_moe_config)
        
        pattern = handler.get_router_weight_pattern(7)
        assert pattern == "model.layers.7.mlp.gate."
    
    def test_get_router_weight_pattern_deepseek(self, mock_deepseek_moe_config):
        """Test router weight pattern for DeepSeek-MoE."""
        handler = MoEHandler(mock_deepseek_moe_config)
        
        pattern = handler.get_router_weight_pattern(10)
        assert pattern == "model.layers.10.mlp.gate."
    
    def test_get_router_weight_pattern_grok(self, mock_grok_config):
        """Test router weight pattern for Grok."""
        handler = MoEHandler(mock_grok_config)
        
        pattern = handler.get_router_weight_pattern(2)
        assert pattern == "model.layers.2.moe_block.gate."


# =============================================================================
# Test Expert Indices to Load
# =============================================================================

class TestExpertIndicesToLoad:
    """Test expert indices determination."""
    
    def test_get_expert_indices_for_non_moe_layer(self, mock_mixtral_config):
        """Test empty set returned for non-MoE layer."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        config.moe_layer_interval = 2
        
        handler = MoEHandler(config)
        
        # Odd layers are not MoE
        indices = handler.get_expert_indices_to_load(1)
        assert indices == set()
    
    def test_get_expert_indices_loads_all_by_default(self, mock_mixtral_config):
        """Test loading all experts when no routing weights provided."""
        handler = MoEHandler(mock_mixtral_config)
        
        indices = handler.get_expert_indices_to_load(0)
        
        # Should return all expert indices
        assert indices == set(range(8))
    
    def test_get_expert_indices_with_routing_weights(self, mock_mixtral_config):
        """Test loading top-k experts with routing weights."""
        handler = MoEHandler(mock_mixtral_config)
        
        # Create mock routing weights [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        routing_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        indices = handler.get_expert_indices_to_load(0, routing_weights)
        
        # Should return top-2 indices (6, 7)
        assert indices == {6, 7}
    
    def test_get_expert_indices_with_qwen2_moe(self, mock_qwen2_moe_config):
        """Test expert indices with Qwen2-MoE (60 experts, top-4)."""
        handler = MoEHandler(mock_qwen2_moe_config)
        
        indices = handler.get_expert_indices_to_load(0)
        
        assert indices == set(range(60))
    
    def test_get_expert_indices_with_deepseek(self, mock_deepseek_moe_config):
        """Test expert indices with DeepSeek-MoE (64 experts, top-6)."""
        handler = MoEHandler(mock_deepseek_moe_config)
        
        indices = handler.get_expert_indices_to_load(0)
        
        assert indices == set(range(64))


# =============================================================================
# Test Routing Weight Computation
# =============================================================================

class TestRoutingWeightComputation:
    """Test routing weight computation."""
    
    def test_compute_routing_weights_shape(self, mock_mixtral_config):
        """Test routing weights output shape."""
        handler = MoEHandler(mock_mixtral_config)
        
        batch_size = 2
        seq_len = 10
        hidden_dim = 4096
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        router_weights = torch.randn(hidden_dim, 8)  # 8 experts
        
        expert_indices, expert_weights = handler.compute_routing_weights(
            hidden_states, router_weights
        )
        
        # Check shapes
        assert expert_indices.shape == (batch_size, seq_len, 2)  # top_k=2
        assert expert_weights.shape == (batch_size, seq_len, 2)
    
    def test_compute_routing_weights_normalization(self, mock_mixtral_config):
        """Test that expert weights are normalized."""
        handler = MoEHandler(mock_mixtral_config)
        
        hidden_states = torch.randn(1, 1, 4096)
        router_weights = torch.randn(4096, 8)
        
        expert_indices, expert_weights = handler.compute_routing_weights(
            hidden_states, router_weights
        )
        
        # Weights should sum to approximately 1 for each token
        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_compute_routing_weights_top_k_selection(self, mock_mixtral_config):
        """Test that top-k experts are selected."""
        handler = MoEHandler(mock_mixtral_config)
        
        # Use specific hidden states and router weights for deterministic test
        hidden_states = torch.ones(1, 1, 4)
        # Create router weights that will give predictable rankings
        router_weights = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Expert 7 should be highest
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ])
        
        expert_indices, expert_weights = handler.compute_routing_weights(
            hidden_states, router_weights
        )
        
        # Should select experts 7 and 6 (highest logits)
        assert 7 in expert_indices[0, 0]
        assert 6 in expert_indices[0, 0]


# =============================================================================
# Test Expert Weight Loading
# =============================================================================

class TestExpertWeightLoading:
    """Test expert weight loading functionality."""
    
    def test_load_expert_weights(self, mock_mixtral_config):
        """Test loading weights for specific experts."""
        handler = MoEHandler(mock_mixtral_config)
        
        # Mock weight loader function
        def mock_weight_loader(pattern):
            return {f"{pattern}weight": torch.randn(10, 10)}
        
        expert_weights = handler.load_expert_weights(5, {0, 1}, mock_weight_loader)
        
        assert 0 in expert_weights
        assert 1 in expert_weights
        assert len(expert_weights) == 2
    
    def test_load_expert_weights_single_expert(self, mock_mixtral_config):
        """Test loading weights for a single expert."""
        handler = MoEHandler(mock_mixtral_config)
        
        def mock_weight_loader(pattern):
            return {f"{pattern}w1": torch.randn(5, 5)}
        
        expert_weights = handler.load_expert_weights(3, {5}, mock_weight_loader)
        
        assert 5 in expert_weights
        assert len(expert_weights) == 1


# =============================================================================
# Test Model Info Retrieval
# =============================================================================

class TestModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info_mixtral(self, mock_mixtral_config):
        """Test model info for Mixtral."""
        handler = MoEHandler(mock_mixtral_config)
        
        info = handler.get_model_info()
        
        assert info["moe_type"] == "mixtral"
        assert info["num_experts"] == 8
        assert info["top_k"] == 2
        assert info["has_shared_experts"] is False
        assert info["num_shared_experts"] == 0
    
    def test_get_model_info_deepseek(self, mock_deepseek_moe_config):
        """Test model info for DeepSeek-MoE."""
        handler = MoEHandler(mock_deepseek_moe_config)
        
        info = handler.get_model_info()
        
        assert info["moe_type"] == "deepseek"
        assert info["num_experts"] == 64
        assert info["top_k"] == 6
        assert info["has_shared_experts"] is True
        assert info["num_shared_experts"] == 2
    
    def test_get_model_info_returns_dict(self, mock_mixtral_config):
        """Test that get_model_info returns a dictionary."""
        handler = MoEHandler(mock_mixtral_config)
        
        info = handler.get_model_info()
        
        assert isinstance(info, dict)
        assert all(key in info for key in [
            "moe_type", "num_experts", "top_k",
            "has_shared_experts", "num_shared_experts"
        ])


# =============================================================================
# Test MoE Patterns Dictionary
# =============================================================================

class TestMoEPatterns:
    """Test the MOE_PATTERNS dictionary."""
    
    def test_moe_patterns_contains_all_types(self, mock_mixtral_config):
        """Test that MOE_PATTERNS contains all expected types."""
        handler = MoEHandler(mock_mixtral_config)
        
        expected_types = ["mixtral", "qwen2_moe", "deepseek", "grok", "glm4_moe"]
        for moe_type in expected_types:
            assert moe_type in handler.MOE_PATTERNS
    
    def test_moe_patterns_have_required_fields(self, mock_mixtral_config):
        """Test that each pattern has required fields."""
        handler = MoEHandler(mock_mixtral_config)
        
        required_fields = ["model_types", "architectures", "expert_attr", "top_k_attr"]
        
        for moe_type, pattern in handler.MOE_PATTERNS.items():
            for field in required_fields:
                assert field in pattern, f"{moe_type} missing {field}"
    
    def test_deepseek_pattern_has_shared_attr(self, mock_mixtral_config):
        """Test that DeepSeek pattern has shared_attr field."""
        handler = MoEHandler(mock_mixtral_config)
        
        deepseek_pattern = handler.MOE_PATTERNS["deepseek"]
        assert "shared_attr" in deepseek_pattern
        assert deepseek_pattern["shared_attr"] == "n_shared_experts"


# =============================================================================
# Test MoE Configuration Parsing
# =============================================================================

class TestConfigParsing:
    """Test MoE configuration parsing from model config."""
    
    def test_parse_config_extracts_expert_count(self):
        """Test that expert count is extracted correctly."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 16
        config.num_experts_per_tok = 4
        
        handler = MoEHandler(config)
        
        assert handler.moe_config.num_experts == 16
    
    def test_parse_config_extracts_top_k(self):
        """Test that top_k is extracted correctly."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.n_routed_experts = 64
        config.num_experts_per_tok = 6
        
        handler = MoEHandler(config)
        
        assert handler.moe_config.top_k == 6
    
    def test_parse_config_detects_shared_experts(self):
        """Test that shared experts are detected."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.n_routed_experts = 64
        config.n_shared_experts = 2
        config.num_experts_per_tok = 6
        
        handler = MoEHandler(config)
        
        assert handler.moe_config.has_shared_experts is True
        assert handler.moe_config.num_shared_experts == 2
