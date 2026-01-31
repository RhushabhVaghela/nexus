"""
Test suite for the Universal Layer Factory module.

This module tests the UniversalLayerFactory class which provides factory methods
for creating layer instances from any supported architecture family.

Total test cases: ~50
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.nexus_final.sli.layer_factory import UniversalLayerFactory
from src.nexus_final.sli.architecture_registry import (
    ArchitectureRegistry,
    LlamaFamilyHandler,
    GPTFamilyHandler,
    T5FamilyHandler,
)
from src.nexus_final.sli.exceptions import LayerCreationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def factory():
    """Create a UniversalLayerFactory instance."""
    return UniversalLayerFactory()


@pytest.fixture
def mock_llama_config():
    """Create a mock Llama-style config."""
    config = MagicMock()
    config.model_type = "llama"
    config.architectures = ["LlamaForCausalLM"]
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.vocab_size = 32000
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.intermediate_size = 11008
    config.rms_norm_eps = 1e-6
    config.rope_theta = 10000.0
    return config


@pytest.fixture
def mock_gpt2_config():
    """Create a mock GPT-2 style config."""
    config = MagicMock()
    config.model_type = "gpt2"
    config.architectures = ["GPT2LMHeadModel"]
    config.n_layer = 12
    config.n_embd = 768
    config.n_head = 12
    config.vocab_size = 50257
    return config


@pytest.fixture
def mock_t5_config():
    """Create a mock T5-style config."""
    config = MagicMock()
    config.model_type = "t5"
    config.architectures = ["T5ForConditionalGeneration"]
    config.num_hidden_layers = 12
    config.d_model = 768
    config.num_heads = 12
    config.d_ff = 2048
    config.vocab_size = 32128
    config.dropout_rate = 0.1
    return config


@pytest.fixture
def mock_moe_config():
    """Create a mock MoE-style config."""
    config = MagicMock()
    config.model_type = "mixtral"
    config.architectures = ["MixtralForCausalLM"]
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.vocab_size = 32000
    config.num_local_experts = 8
    config.num_experts_per_tok = 2
    return config


# =============================================================================
# Test Factory Initialization
# =============================================================================

class TestFactoryInitialization:
    """Test UniversalLayerFactory initialization."""
    
    def test_factory_initializes_with_default_registry(self):
        """Test factory initializes with default registry."""
        factory = UniversalLayerFactory()
        assert factory.registry is not None
        assert isinstance(factory.registry, ArchitectureRegistry)
    
    def test_factory_initializes_with_custom_registry(self):
        """Test factory initializes with custom registry."""
        custom_registry = MagicMock(spec=ArchitectureRegistry)
        factory = UniversalLayerFactory(registry=custom_registry)
        assert factory.registry is custom_registry
    
    def test_factory_has_empty_layer_cache_on_init(self):
        """Test factory initializes with empty layer cache."""
        factory = UniversalLayerFactory()
        assert isinstance(factory._layer_cache, dict)
        assert len(factory._layer_cache) == 0


# =============================================================================
# Test Create Layer - Success Cases
# =============================================================================

class TestCreateLayerSuccess:
    """Test successful layer creation for various architectures."""
    
    @patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer')
    def test_create_llama_layer(self, mock_layer_class, factory, mock_llama_config):
        """Test creating a Llama decoder layer."""
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer_class.return_value = mock_layer
        
        layer = factory.create_layer(mock_llama_config, layer_idx=0)
        
        mock_layer_class.assert_called_once()
        call_args = mock_layer_class.call_args
        assert call_args[1]['layer_idx'] == 0
        assert layer is mock_layer
    
    @patch('transformers.models.gpt2.modeling_gpt2.GPT2Block')
    def test_create_gpt2_layer(self, mock_layer_class, factory, mock_gpt2_config):
        """Test creating a GPT-2 block."""
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer_class.return_value = mock_layer
        
        layer = factory.create_layer(mock_gpt2_config, layer_idx=0)
        
        mock_layer_class.assert_called_once()
        call_args = mock_layer_class.call_args
        assert call_args[1]['layer_idx'] == 0
        assert layer is mock_layer
    
    @patch('transformers.models.t5.modeling_t5.T5Block')
    def test_create_t5_layer_decoder(self, mock_layer_class, factory, mock_t5_config):
        """Test creating a T5 block for decoder."""
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer_class.return_value = mock_layer
        
        layer = factory.create_layer(mock_t5_config, layer_idx=0, layer_type="decoder")
        
        mock_layer_class.assert_called_once()
        assert layer is mock_layer
    
    @patch('transformers.models.t5.modeling_t5.T5Block')
    def test_create_t5_layer_encoder(self, mock_layer_class, factory, mock_t5_config):
        """Test creating a T5 block for encoder."""
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer_class.return_value = mock_layer
        
        layer = factory.create_layer(mock_t5_config, layer_idx=0, layer_type="encoder")
        
        mock_layer_class.assert_called_once()
        assert layer is mock_layer
    
    @patch('transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer')
    def test_create_mixtral_layer(self, mock_layer_class, factory, mock_moe_config):
        """Test creating a Mixtral decoder layer."""
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer_class.return_value = mock_layer
        
        layer = factory.create_layer(mock_moe_config, layer_idx=5)
        
        mock_layer_class.assert_called_once()
        call_args = mock_layer_class.call_args
        assert call_args[1]['layer_idx'] == 5
        assert layer is mock_layer
    
    def test_create_layer_with_layer_idx(self, factory, mock_llama_config):
        """Test that layer_idx is passed to layer creation."""
        with patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer') as mock_layer:
            mock_layer.return_value = MagicMock(spec=nn.Module)
            factory.create_layer(mock_llama_config, layer_idx=10)
            assert mock_layer.call_args[1]['layer_idx'] == 10
    
    def test_create_layer_with_different_indices(self, factory, mock_llama_config):
        """Test creating layers at different indices."""
        with patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer') as mock_layer:
            mock_layer.return_value = MagicMock(spec=nn.Module)
            
            for idx in [0, 5, 15, 31]:
                factory.create_layer(mock_llama_config, layer_idx=idx)
                assert mock_layer.call_args[1]['layer_idx'] == idx


# =============================================================================
# Test Create Layer - Error Handling
# =============================================================================

class TestCreateLayerErrors:
    """Test error handling during layer creation."""
    
    def test_create_layer_import_error_raises_layer_creation_error(self, factory, mock_llama_config):
        """Test that import errors are converted to LayerCreationError."""
        with patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer', side_effect=ImportError("Module not found")):
            with pytest.raises(LayerCreationError) as exc_info:
                factory.create_layer(mock_llama_config, layer_idx=5)
            
            assert "layer 5" in str(exc_info.value)
            assert "llama" in str(exc_info.value)
    
    def test_create_layer_attribute_error_raises_layer_creation_error(self, factory, mock_llama_config):
        """Test that attribute errors are converted to LayerCreationError."""
        with patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer', side_effect=AttributeError("Missing attribute")):
            with pytest.raises(LayerCreationError) as exc_info:
                factory.create_layer(mock_llama_config, layer_idx=3)
            
            assert "layer 3" in str(exc_info.value)
    
    def test_create_layer_value_error_raises_layer_creation_error(self, factory, mock_llama_config):
        """Test that value errors are converted to LayerCreationError."""
        with patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer', side_effect=ValueError("Invalid value")):
            with pytest.raises(LayerCreationError) as exc_info:
                factory.create_layer(mock_llama_config, layer_idx=0)


# =============================================================================
# Test Get Weight Prefix
# =============================================================================

class TestGetWeightPrefix:
    """Test weight prefix generation for different architectures."""
    
    def test_get_weight_prefix_llama(self, factory, mock_llama_config):
        """Test weight prefix for Llama architecture."""
        prefix = factory.get_weight_prefix(mock_llama_config, layer_idx=5)
        assert prefix == "model.layers.5."
    
    def test_get_weight_prefix_gpt2(self, factory, mock_gpt2_config):
        """Test weight prefix for GPT-2 architecture."""
        prefix = factory.get_weight_prefix(mock_gpt2_config, layer_idx=3)
        assert prefix == "transformer.h.3."
    
    def test_get_weight_prefix_t5_encoder(self, factory, mock_t5_config):
        """Test weight prefix for T5 encoder."""
        prefix = factory.get_weight_prefix(mock_t5_config, layer_idx=2, layer_type="encoder")
        assert prefix == "encoder.block.2."
    
    def test_get_weight_prefix_t5_decoder(self, factory, mock_t5_config):
        """Test weight prefix for T5 decoder."""
        prefix = factory.get_weight_prefix(mock_t5_config, layer_idx=2, layer_type="decoder")
        assert prefix == "decoder.block.2."
    
    def test_get_weight_prefix_different_indices(self, factory, mock_llama_config):
        """Test weight prefix for different layer indices."""
        for idx in [0, 1, 10, 31]:
            prefix = factory.get_weight_prefix(mock_llama_config, layer_idx=idx)
            assert f"model.layers.{idx}." == prefix


# =============================================================================
# Test Get Embedding Info
# =============================================================================

class TestGetEmbeddingInfo:
    """Test embedding information retrieval."""
    
    def test_get_embedding_info_llama(self, factory, mock_llama_config):
        """Test embedding info for Llama architecture."""
        info = factory.get_embedding_info(mock_llama_config)
        
        assert "embedding" in info
        assert "lm_head" in info
        assert info["embedding"] == "model.embed_tokens"
        assert info["lm_head"] == "lm_head"
    
    def test_get_embedding_info_gpt2(self, factory, mock_gpt2_config):
        """Test embedding info for GPT-2 architecture."""
        info = factory.get_embedding_info(mock_gpt2_config)
        
        assert "embedding" in info
        assert "lm_head" in info
        assert info["embedding"] == "transformer.wte"
        assert info["lm_head"] == "lm_head"
    
    def test_get_embedding_info_returns_dict(self, factory, mock_llama_config):
        """Test that get_embedding_info returns a dictionary."""
        info = factory.get_embedding_info(mock_llama_config)
        assert isinstance(info, dict)
        assert len(info) == 2


# =============================================================================
# Test Get Model Info
# =============================================================================

class TestGetModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info_llama(self, factory, mock_llama_config):
        """Test model info for Llama architecture."""
        info = factory.get_model_info(mock_llama_config)
        
        assert info["family_id"] == "llama"
        assert info["family_name"] == "Llama-Based Architectures"
        assert info["num_layers"] == 32
        assert info["hidden_size"] == 4096
        assert info["vocab_size"] == 32000
        assert info["trust_remote_code"] is False
    
    def test_get_model_info_gpt2(self, factory, mock_gpt2_config):
        """Test model info for GPT-2 architecture."""
        info = factory.get_model_info(mock_gpt2_config)
        
        assert info["family_id"] == "gpt"
        assert info["num_layers"] == 12
        assert info["hidden_size"] == 768
    
    def test_get_model_info_contains_all_fields(self, factory, mock_llama_config):
        """Test that model info contains all expected fields."""
        info = factory.get_model_info(mock_llama_config)
        
        required_fields = [
            "family_id", "family_name", "num_layers",
            "hidden_size", "vocab_size", "trust_remote_code"
        ]
        for field in required_fields:
            assert field in info, f"Missing field: {field}"


# =============================================================================
# Test Is MoE Model Detection
# =============================================================================

class TestIsMoEModel:
    """Test MoE model detection."""
    
    def test_is_moe_model_true_for_mixtral(self, factory, mock_moe_config):
        """Test MoE detection for Mixtral config."""
        assert factory.is_moe_model(mock_moe_config) is True
    
    def test_is_moe_model_false_for_llama(self, factory, mock_llama_config):
        """Test non-MoE detection for Llama config."""
        assert factory.is_moe_model(mock_llama_config) is False
    
    def test_is_moe_model_by_n_routed_experts(self, factory):
        """Test MoE detection via n_routed_experts attribute."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.n_routed_experts = 64
        
        assert factory.is_moe_model(config) is True
    
    def test_is_moe_model_by_moe_intermediate_size(self, factory):
        """Test MoE detection via moe_intermediate_size attribute."""
        config = MagicMock()
        config.model_type = "custom_moe"
        config.architectures = ["CustomMoeForCausalLM"]
        config.moe_intermediate_size = 2048
        
        assert factory.is_moe_model(config) is True
    
    def test_is_moe_model_returns_false_for_non_moe(self, factory):
        """Test non-MoE detection."""
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        
        assert factory.is_moe_model(config) is False


# =============================================================================
# Test Layer Creation for All Architecture Families
# =============================================================================

class TestLayerCreationAllFamilies:
    """Test layer creation for all 12 architecture families."""
    
    @patch('transformers.models.llama.modeling_llama.LlamaDecoderLayer')
    def test_create_layer_llama_family(self, mock_layer, factory):
        """Test layer creation for Llama family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer')
    def test_create_layer_qwen_family(self, mock_layer, factory):
        """Test layer creation for Qwen family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "qwen2"
        config.architectures = ["Qwen2ForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.gpt2.modeling_gpt2.GPT2Block')
    def test_create_layer_gpt_family(self, mock_layer, factory):
        """Test layer creation for GPT family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.t5.modeling_t5.T5Block')
    def test_create_layer_t5_family(self, mock_layer, factory):
        """Test layer creation for T5 family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "t5"
        config.architectures = ["T5ForConditionalGeneration"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.bloom.modeling_bloom.BloomBlock')
    def test_create_layer_bloom_family(self, mock_layer, factory):
        """Test layer creation for BLOOM family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "bloom"
        config.architectures = ["BloomForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.opt.modeling_opt.OPTDecoderLayer')
    def test_create_layer_opt_family(self, mock_layer, factory):
        """Test layer creation for OPT family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "opt"
        config.architectures = ["OPTForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.mamba.modeling_mamba.MambaBlock')
    def test_create_layer_mamba_family(self, mock_layer, factory):
        """Test layer creation for Mamba family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "mamba"
        config.architectures = ["MambaForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer')
    def test_create_layer_moe_family(self, mock_layer, factory):
        """Test layer creation for MoE family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.phi.modeling_phi.PhiDecoderLayer')
    def test_create_layer_phi_family(self, mock_layer, factory):
        """Test layer creation for Phi family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "phi"
        config.architectures = ["PhiForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()
    
    @patch('transformers.models.gemma.modeling_gemma.GemmaDecoderLayer')
    def test_create_layer_gemma_family(self, mock_layer, factory):
        """Test layer creation for Gemma family."""
        mock_layer.return_value = MagicMock(spec=nn.Module)
        config = MagicMock()
        config.model_type = "gemma"
        config.architectures = ["GemmaForCausalLM"]
        
        layer = factory.create_layer(config, 0)
        assert layer is not None
        mock_layer.assert_called_once()


# =============================================================================
# Test Layer Caching
# =============================================================================

class TestLayerCaching:
    """Test layer caching functionality."""
    
    def test_layer_cache_exists(self, factory):
        """Test that factory has a layer cache."""
        assert hasattr(factory, '_layer_cache')
        assert isinstance(factory._layer_cache, dict)


# =============================================================================
# Test Unsupported Architecture Handling
# =============================================================================

class TestUnsupportedArchitecture:
    """Test handling of unsupported architectures."""
    
    def test_create_layer_unsupported_raises_error(self, factory):
        """Test that unsupported architectures raise appropriate error."""
        config = MagicMock()
        config.model_type = "unsupported_xyz"
        config.architectures = ["UnsupportedModel"]
        
        with pytest.raises(Exception) as exc_info:
            factory.create_layer(config, 0)
        
        # Should raise UnsupportedArchitectureError from registry
        assert "unsupported" in str(exc_info.value).lower() or "UnsupportedArchitectureError" in str(type(exc_info.value))


# =============================================================================
# Test Family-Specific Weight Prefixes
# =============================================================================

class TestFamilySpecificPrefixes:
    """Test weight prefix generation for different families."""
    
    def test_prefix_llama(self, factory):
        """Test Llama weight prefix."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        
        prefix = factory.get_weight_prefix(config, 5)
        assert prefix == "model.layers.5."
    
    def test_prefix_gpt(self, factory):
        """Test GPT weight prefix."""
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        
        prefix = factory.get_weight_prefix(config, 5)
        assert prefix == "transformer.h.5."
    
    def test_prefix_bloom(self, factory):
        """Test BLOOM weight prefix."""
        config = MagicMock()
        config.model_type = "bloom"
        config.architectures = ["BloomForCausalLM"]
        
        prefix = factory.get_weight_prefix(config, 5)
        assert prefix == "transformer.h.5."
    
    def test_prefix_opt(self, factory):
        """Test OPT weight prefix."""
        config = MagicMock()
        config.model_type = "opt"
        config.architectures = ["OPTForCausalLM"]
        
        prefix = factory.get_weight_prefix(config, 5)
        assert prefix == "model.decoder.layers.5."
    
    def test_prefix_mamba(self, factory):
        """Test Mamba weight prefix."""
        config = MagicMock()
        config.model_type = "mamba"
        config.architectures = ["MambaForCausalLM"]
        
        prefix = factory.get_weight_prefix(config, 5)
        assert prefix == "backbone.layers.5."
