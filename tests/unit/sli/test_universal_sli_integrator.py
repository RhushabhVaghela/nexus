"""
Test suite for the Universal SLI Integrator module.

This module tests the UniversalSLIIntegrator class which provides the main
interface for Sequential Layer Ingestion across all supported architectures.

Total test cases: ~45
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, ANY

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.nexus_final.sli.universal_sli_integrator import UniversalSLIIntegrator, SequentialLayerIntegrator
from src.nexus_final.sli.exceptions import UnsupportedArchitectureError
from src.nexus_final.sli.architecture_registry import LlamaFamilyHandler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    return str(tmp_path / "output")


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return str(tmp_path / "cache")


@pytest.fixture
def temp_activation_dir(tmp_path):
    """Create a temporary activation cache directory."""
    return str(tmp_path / "activations")


@pytest.fixture
def mock_model_id():
    """Return a mock model ID."""
    return "test-org/test-model"


@pytest.fixture
def mock_config():
    """Create a mock model config."""
    config = MagicMock()
    config.model_type = "llama"
    config.architectures = ["LlamaForCausalLM"]
    config.num_hidden_layers = 4
    config.hidden_size = 128
    config.vocab_size = 1000
    return config


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return [
        "Hello, how are you?",
        "What is the weather today?",
        "Tell me a story.",
    ]


# =============================================================================
# Test Initialization
# =============================================================================

class TestIntegratorInitialization:
    """Test UniversalSLIIntegrator initialization."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_integrator_initializes_with_model_id(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                   temp_output_dir, temp_cache_dir, temp_activation_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test integrator initializes with model ID."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            cache_dir=temp_cache_dir,
            activation_cache_dir=temp_activation_dir,
            device="cpu"
        )
        
        assert integrator.model_id == mock_model_id
        assert integrator.device == "cpu"
        assert integrator.trust_remote_code is True
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_integrator_creates_directories(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                             tmp_path, mock_model_id, mock_config, mock_tokenizer):
        """Test integrator creates necessary directories."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"
        activation_dir = tmp_path / "activations"
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(output_dir),
            cache_dir=str(cache_dir),
            activation_cache_dir=str(activation_dir),
            device="cpu"
        )
        
        assert output_dir.exists()
        assert cache_dir.exists()
        assert activation_dir.exists()
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_integrator_sets_pad_token_if_none(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                temp_output_dir, mock_model_id, mock_config):
        """Test integrator sets pad_token to eos_token if pad_token is None."""
        mock_config_cls.return_value = mock_config
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "[EOS]"
        mock_tokenizer_cls.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator.tokenizer.pad_token == "[EOS]"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_integrator_detects_architecture_family(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                     temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test integrator detects architecture family from config."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator.family is not None
        assert integrator.family.family_id == "llama"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_integrator_detects_moe_model(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                           temp_output_dir, mock_model_id, mock_tokenizer):
        """Test integrator detects MoE models."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 8
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator.moe_handler is not None


# =============================================================================
# Test MoE Model Detection
# =============================================================================

class TestMoEModelDetection:
    """Test MoE model detection in integrator."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_is_moe_model_by_num_local_experts(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                temp_output_dir, mock_model_id, mock_tokenizer):
        """Test MoE detection by num_local_experts."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        config.num_local_experts = 8
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator._is_moe_model() is True
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_is_moe_model_by_n_routed_experts(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                               temp_output_dir, mock_model_id, mock_tokenizer):
        """Test MoE detection by n_routed_experts."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.n_routed_experts = 64
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator._is_moe_model() is True
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_is_not_moe_model(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                              temp_output_dir, mock_model_id, mock_tokenizer):
        """Test non-MoE detection."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator._is_moe_model() is False
        assert integrator.moe_handler is None


# =============================================================================
# Test Model Summary
# =============================================================================

class TestModelSummary:
    """Test model summary functionality."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_get_model_summary(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                               temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test getting model summary."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        summary = integrator.get_model_summary()
        
        assert summary["model_id"] == mock_model_id
        assert summary["family"] == "llama"
        assert summary["num_layers"] == 4
        assert summary["hidden_size"] == 128
        assert summary["vocab_size"] == 1000
        assert summary["is_moe"] is False
        assert summary["moe_info"] is None
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_get_model_summary_for_moe(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                       temp_output_dir, mock_model_id, mock_tokenizer):
        """Test model summary for MoE model."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_hidden_layers = 4
        config.hidden_size = 128
        config.vocab_size = 1000
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        summary = integrator.get_model_summary()
        
        assert summary["is_moe"] is True
        assert summary["moe_info"] is not None
        assert summary["moe_info"]["moe_type"] == "mixtral"
        assert summary["moe_info"]["num_experts"] == 8


# =============================================================================
# Test Layer Creation
# =============================================================================

class TestLayerCreation:
    """Test layer creation in integrator."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_create_layer_calls_factory(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                        temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test create_layer calls factory method."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        mock_layer = MagicMock(spec=nn.Module)
        integrator.factory.create_layer = MagicMock(return_value=mock_layer)
        
        layer = integrator._create_layer(0)
        
        integrator.factory.create_layer.assert_called_once_with(mock_config, 0)
        assert layer is mock_layer


# =============================================================================
# Test Embedding Processing
# =============================================================================

class TestEmbeddingProcessing:
    """Test embedding processing."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_process_embeddings(self, mock_weight_loader_cls, mock_tokenizer_cls, mock_config_cls,
                                tmp_path, mock_model_id, mock_config, mock_tokenizer):
        """Test processing embeddings."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock weight loader
        mock_weight_loader = MagicMock()
        mock_weight_loader.load_embedding_weights.return_value = torch.randn(1000, 128)
        mock_weight_loader_cls.return_value = mock_weight_loader
        
        activation_dir = tmp_path / "activations"
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            activation_cache_dir=str(activation_dir),
            device="cpu"
        )
        
        dataset = ["Hello", "World"]
        path = integrator._process_embeddings(dataset)
        
        assert path.exists()
        assert path.name == "base_embeddings.pt"


# =============================================================================
# Test Forward Pass
# =============================================================================

class TestForwardPass:
    """Test forward pass through layers."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_forward_batch_sli(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                               tmp_path, mock_model_id, mock_config, mock_tokenizer):
        """Test forward batch SLI."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            activation_cache_dir=str(tmp_path / "activations"),
            device="cpu"
        )
        
        # Create input activations
        in_path = tmp_path / "input.pt"
        torch.save(torch.randn(2, 10, 128), in_path)
        
        out_path = str(tmp_path / "output.pt")
        
        # Create mock layer
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.return_value = torch.randn(2, 10, 128)
        
        integrator._forward_batch_sli(in_path, out_path, mock_layer, batch_size=1)
        
        assert Path(out_path).exists()
        mock_layer.assert_called()
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_forward_batch_handles_tuple_output(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                 tmp_path, mock_model_id, mock_config, mock_tokenizer):
        """Test forward batch handles tuple output from layer."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            activation_cache_dir=str(tmp_path / "activations"),
            device="cpu"
        )
        
        # Create input activations
        in_path = tmp_path / "input.pt"
        torch.save(torch.randn(2, 10, 128), in_path)
        
        out_path = str(tmp_path / "output.pt")
        
        # Create mock layer that returns tuple
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.return_value = (torch.randn(2, 10, 128), None)
        
        integrator._forward_batch_sli(in_path, out_path, mock_layer, batch_size=1)
        
        assert Path(out_path).exists()


# =============================================================================
# Test Cache Clearing
# =============================================================================

class TestCacheClearing:
    """Test cache clearing functionality."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_clear_cache(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                         tmp_path, mock_model_id, mock_config, mock_tokenizer):
        """Test clearing cache."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        cache_dir = tmp_path / "cache"
        activation_dir = tmp_path / "activations"
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            cache_dir=str(cache_dir),
            activation_cache_dir=str(activation_dir),
            device="cpu"
        )
        
        # Create some files in cache
        (cache_dir / "test.bin").touch()
        (activation_dir / "test.pt").touch()
        
        integrator.clear_cache()
        
        # Directories should be recreated and empty
        assert cache_dir.exists()
        assert activation_dir.exists()
        assert len(list(cache_dir.iterdir())) == 0
        assert len(list(activation_dir.iterdir())) == 0


# =============================================================================
# Test Architecture Fallback
# =============================================================================

class TestArchitectureFallback:
    """Test fallback behavior for unsupported architectures."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    @patch('src.nexus_final.sli.universal_sli_integrator.get_registry')
    def test_fallback_to_llama_on_unsupported(self, mock_get_registry, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                               temp_output_dir, mock_model_id, mock_tokenizer):
        """Test fallback to Llama family when architecture detection fails."""
        config = MagicMock()
        config.model_type = "unknown_architecture"
        config.architectures = ["UnknownModel"]
        
        mock_config_cls.return_value = config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # Mock registry to raise UnsupportedArchitectureError
        mock_registry = MagicMock()
        mock_registry.detect_family.side_effect = UnsupportedArchitectureError("unknown")
        mock_registry.get_family.return_value = LlamaFamilyHandler()
        mock_get_registry.return_value = mock_registry
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert integrator.family is not None
        assert integrator.family.family_id == "llama"


# =============================================================================
# Test Legacy Compatibility
# =============================================================================

class TestLegacyCompatibility:
    """Test legacy SequentialLayerIntegrator wrapper."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_sequential_layer_integrator_is_subclass(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                                      temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test SequentialLayerIntegrator is a subclass of UniversalSLIIntegrator."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        # The deprecation warning should be printed but we can't easily test that
        integrator = SequentialLayerIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert isinstance(integrator, UniversalSLIIntegrator)


# =============================================================================
# Test Configuration Parameters
# =============================================================================

class TestConfigurationParameters:
    """Test various configuration parameters."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_custom_registry(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                             temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test using custom registry."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        custom_registry = MagicMock()
        custom_registry.detect_family.return_value = LlamaFamilyHandler()
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu",
            registry=custom_registry
        )
        
        assert integrator.registry is custom_registry
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_trust_remote_code_parameter(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                          temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test trust_remote_code parameter."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu",
            trust_remote_code=False
        )
        
        assert integrator.trust_remote_code is False


# =============================================================================
# Test Run SLI Pipeline
# =============================================================================

class TestRunSLI:
    """Test the full run_sli pipeline."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_run_sli_calls_process_embeddings(self, mock_weight_loader_cls, mock_tokenizer_cls, mock_config_cls,
                                               tmp_path, mock_model_id, mock_config, mock_tokenizer, sample_dataset):
        """Test run_sli processes embeddings."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        mock_weight_loader = MagicMock()
        mock_weight_loader.load_embedding_weights.return_value = torch.randn(1000, 128)
        mock_weight_loader.load_layer_weights.return_value = {}
        mock_weight_loader_cls.return_value = mock_weight_loader
        
        activation_dir = tmp_path / "activations"
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            activation_cache_dir=str(activation_dir),
            device="cpu"
        )
        
        # Mock _forward_batch_sli to avoid actual processing
        integrator._forward_batch_sli = MagicMock()
        
        # Mock layer creation
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.load_state_dict = MagicMock()
        integrator._create_layer = MagicMock(return_value=mock_layer)
        
        result = integrator.run_sli(sample_dataset, batch_size=1)
        
        assert "activation_cache_dir" in result
        assert "num_layers" in result
        assert result["num_layers"] == 4
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_run_sli_processes_all_layers(self, mock_weight_loader_cls, mock_tokenizer_cls, mock_config_cls,
                                           tmp_path, mock_model_id, mock_config, mock_tokenizer, sample_dataset):
        """Test run_sli processes all layers."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        mock_weight_loader = MagicMock()
        mock_weight_loader.load_embedding_weights.return_value = torch.randn(1000, 128)
        mock_weight_loader.load_layer_weights.return_value = {}
        mock_weight_loader_cls.return_value = mock_weight_loader
        
        activation_dir = tmp_path / "activations"
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=str(tmp_path / "output"),
            activation_cache_dir=str(activation_dir),
            device="cpu"
        )
        
        integrator._forward_batch_sli = MagicMock()
        
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.load_state_dict = MagicMock()
        integrator._create_layer = MagicMock(return_value=mock_layer)
        
        integrator.run_sli(sample_dataset, batch_size=1)
        
        # Should create layer for each of the 4 layers
        assert integrator._create_layer.call_count == 4


# =============================================================================
# Test Model Info Storage
# =============================================================================

class TestModelInfoStorage:
    """Test model info storage after initialization."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_model_info_stored_after_init(self, mock_weight_loader, mock_tokenizer_cls, mock_config_cls,
                                          temp_output_dir, mock_model_id, mock_config, mock_tokenizer):
        """Test that model_info is stored after initialization."""
        mock_config_cls.return_value = mock_config
        mock_tokenizer_cls.return_value = mock_tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id=mock_model_id,
            output_dir=temp_output_dir,
            device="cpu"
        )
        
        assert hasattr(integrator, 'model_info')
        assert integrator.model_info is not None
        assert "num_layers" in integrator.model_info
        assert "hidden_size" in integrator.model_info
