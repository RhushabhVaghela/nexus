
import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from nexus.models.sli.universal_sli_integrator import UniversalSLIIntegrator
from nexus.models.sli.architecture_registry import ArchitectureRegistry, ArchitectureFamily
from nexus.models.sli.exceptions import UnsupportedArchitectureError

# Mock families
class MockLlamaFamily(ArchitectureFamily):
    family_id = "llama"
    family_name = "Llama"
    architectures = ["LlamaForCausalLM"]
    def get_layer_prefix(self, idx, type="decoder"): return f"model.layers.{idx}."
    def create_layer(self, config, idx, type="decoder"): return nn.Linear(10, 10)

class MockGPTFamily(ArchitectureFamily):
    family_id = "gpt"
    family_name = "GPT"
    architectures = ["GPT2LMHeadModel"]
    def get_layer_prefix(self, idx, type="decoder"): return f"transformer.h.{idx}."
    def create_layer(self, config, idx, type="decoder"): return nn.Linear(10, 10)

@pytest.fixture
def mock_registry():
    registry = ArchitectureRegistry()
    registry._families = {} # Clear for test
    registry.register("llama", MockLlamaFamily())
    registry.register("gpt", MockGPTFamily())
    return registry

@patch("src.nexus.models.sli.universal_sli_integrator.AutoConfig")
@patch("src.nexus.models.sli.universal_sli_integrator.AutoTokenizer")
@patch("src.nexus.models.sli.universal_sli_integrator.UniversalWeightLoader")
def test_initialization_llama(mock_loader, mock_tokenizer, mock_config, mock_registry):
    # Setup mocks
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "llama"
    config.architectures = ["LlamaForCausalLM"]
    config.num_hidden_layers = 2
    config.hidden_size = 10
    config.vocab_size = 100
    mock_config.from_pretrained.return_value = config
    
    # Test init
    integrator = UniversalSLIIntegrator(
        model_id="meta-llama/Llama-2-7b-hf",
        registry=mock_registry,
        device="cpu"
    )
    
    assert integrator.family.family_id == "llama"
    assert integrator.model_info["num_layers"] == 2

@patch("src.nexus.models.sli.universal_sli_integrator.AutoConfig")
@patch("src.nexus.models.sli.universal_sli_integrator.AutoTokenizer")
@patch("src.nexus.models.sli.universal_sli_integrator.UniversalWeightLoader")
def test_initialization_gpt(mock_loader, mock_tokenizer, mock_config, mock_registry):
    # Setup mocks
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "gpt2"
    config.architectures = ["GPT2LMHeadModel"]
    config.n_layer = 4 # GPT2 uses n_layer
    config.n_embd = 10
    config.n_vocab = 100
    mock_config.from_pretrained.return_value = config
    
    # Test init
    integrator = UniversalSLIIntegrator(
        model_id="gpt2",
        registry=mock_registry,
        device="cpu"
    )
    
    assert integrator.family.family_id == "gpt"
    assert integrator.model_info["num_layers"] == 4

@patch("src.nexus.models.sli.universal_sli_integrator.AutoConfig")
@patch("src.nexus.models.sli.universal_sli_integrator.AutoTokenizer")
@patch("src.nexus.models.sli.universal_sli_integrator.UniversalWeightLoader")
def test_fallback_behavior(mock_loader, mock_tokenizer, mock_config, mock_registry):
    # Setup mocks for unknown model
    config = MagicMock(spec=PretrainedConfig)
    config.model_type = "unknown_type"
    config.architectures = ["UnknownModel"]
    # Add minimal attrs so it doesn't crash on info extraction
    config.num_hidden_layers = 1
    config.hidden_size = 10
    config.vocab_size = 100
    mock_config.from_pretrained.return_value = config
    
    # Test init
    integrator = UniversalSLIIntegrator(
        model_id="unknown/model",
        registry=mock_registry,
        device="cpu"
    )
    
    # Should fallback to Llama
    assert integrator.family.family_id == "llama"
