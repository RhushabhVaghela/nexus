"""
Integration test suite for Universal SLI across different architectures.

This module tests the end-to-end SLI pipeline for various architecture families
to ensure the Universal SLI system works correctly with real-world configurations.

Total test cases: ~25
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.nexus_final.sli import UniversalSLIIntegrator, ArchitectureRegistry
from src.nexus_final.sli.architecture_registry import (
    LlamaFamilyHandler,
    QwenFamilyHandler,
    GPTFamilyHandler,
    ChatGLMFamilyHandler,
    T5FamilyHandler,
    BLOOMFamilyHandler,
    OPTFamilyHandler,
    MambaFamilyHandler,
    MoEFamilyHandler,
    PhiFamilyHandler,
    GemmaFamilyHandler,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def integration_dirs(tmp_path):
    """Create temporary directories for integration tests."""
    return {
        "output": tmp_path / "output",
        "cache": tmp_path / "cache",
        "activations": tmp_path / "activations",
    }


@pytest.fixture
def sample_text_dataset():
    """Sample text dataset for integration tests."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "Large language models can generate human-like text.",
        "Artificial intelligence has many practical applications.",
        "Natural language processing enables computers to understand text.",
    ]


# =============================================================================
# Test Architecture Registry Integration
# =============================================================================

class TestArchitectureRegistryIntegration:
    """Test the architecture registry with real configurations."""
    
    def test_registry_singleton_integration(self):
        """Test that registry singleton works correctly."""
        from src.nexus_final.sli.architecture_registry import get_registry
        
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2
        
        # Should have all families registered
        families = reg1.list_families()
        assert "llama" in families
        assert "gpt" in families
        assert "qwen" in families
        assert "chatglm" in families
        assert "t5" in families
        assert "bloom" in families
        assert "opt" in families
        assert "mamba" in families
        assert "moe" in families
        assert "phi" in families
        assert "gemma" in families
    
    def test_detect_llama_from_hf_config(self):
        """Test detecting Llama architecture from HF config structure."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        
        registry = ArchitectureRegistry()
        family = registry.detect_family(config)
        
        assert family.family_id == "llama"
        assert family.get_layer_prefix(0) == "model.layers.0."
    
    def test_detect_mistral_from_hf_config(self):
        """Test detecting Mistral (Llama-family) from HF config."""
        config = MagicMock()
        config.model_type = "mistral"
        config.architectures = ["MistralForCausalLM"]
        
        registry = ArchitectureRegistry()
        family = registry.detect_family(config)
        
        # Mistral is part of Llama family
        assert family.family_id == "llama"
    
    def test_detect_gpt2_from_hf_config(self):
        """Test detecting GPT-2 architecture from HF config."""
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        config.n_layer = 12
        config.n_embd = 768
        
        registry = ArchitectureRegistry()
        family = registry.detect_family(config)
        
        assert family.family_id == "gpt"
        assert family.get_layer_prefix(0) == "transformer.h.0."
    
    def test_detect_mixtral_moe_from_hf_config(self):
        """Test detecting Mixtral MoE from HF config."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        
        registry = ArchitectureRegistry()
        family = registry.detect_family(config)
        
        assert family.family_id == "moe"
        assert family.get_num_experts(config) == 8
        assert family.get_top_k(config) == 2


# =============================================================================
# Test Llama-Based Models Integration
# =============================================================================

class TestLlamaBasedIntegration:
    """Integration tests for Llama-based models."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_llama_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Llama model initialization."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        config.vocab_size = 32000
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="meta-llama/Llama-2-7b-hf",
            output_dir=str(integration_dirs["output"]),
            cache_dir=str(integration_dirs["cache"]),
            activation_cache_dir=str(integration_dirs["activations"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "llama"
        assert integrator.model_info["num_layers"] == 32
        assert integrator.model_info["hidden_size"] == 4096
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_mistral_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Mistral model initialization."""
        config = MagicMock()
        config.model_type = "mistral"
        config.architectures = ["MistralForCausalLM"]
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        config.vocab_size = 32000
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="mistralai/Mistral-7B-v0.1",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        # Mistral uses Llama family
        assert integrator.family.family_id == "llama"


# =============================================================================
# Test GPT-Based Models Integration
# =============================================================================

class TestGPTBasedIntegration:
    """Integration tests for GPT-based models."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_gpt2_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test GPT-2 model initialization."""
        config = MagicMock()
        config.model_type = "gpt2"
        config.architectures = ["GPT2LMHeadModel"]
        config.n_layer = 12
        config.n_embd = 768
        config.vocab_size = 50257
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="gpt2",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "gpt"
        assert integrator.model_info["num_layers"] == 12
        assert integrator.model_info["hidden_size"] == 768
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_falcon_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Falcon model initialization (GPT family)."""
        config = MagicMock()
        config.model_type = "falcon"
        config.architectures = ["FalconForCausalLM"]
        config.num_hidden_layers = 32
        config.hidden_size = 4544
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="tiiuae/falcon-7b",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "gpt"


# =============================================================================
# Test ChatGLM-Based Models Integration
# =============================================================================

class TestChatGLMBasedIntegration:
    """Integration tests for ChatGLM-based models."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_chatglm_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test ChatGLM model initialization."""
        config = MagicMock()
        config.model_type = "chatglm"
        config.architectures = ["ChatGLMForConditionalGeneration"]
        config.num_hidden_layers = 28
        config.hidden_size = 4096
        config.vocab_size = 65024
        config.name_or_path = "/tmp/chatglm"
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="THUDM/chatglm3-6b",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "chatglm"
        assert integrator.family.trust_remote_code is True


# =============================================================================
# Test MoE Models Integration
# =============================================================================

class TestMoEModelsIntegration:
    """Integration tests for MoE models."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_mixtral_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Mixtral MoE model initialization."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "moe"
        assert integrator.moe_handler is not None
        assert integrator.moe_config.num_experts == 8
        assert integrator.moe_config.top_k == 2
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_qwen2_moe_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Qwen2-MoE model initialization."""
        config = MagicMock()
        config.model_type = "qwen2_moe"
        config.architectures = ["Qwen2MoeForCausalLM"]
        config.num_hidden_layers = 24
        config.hidden_size = 2048
        config.num_experts = 60
        config.num_experts_per_tok = 4
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="Qwen/Qwen2-57B-A14B",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "moe"
        assert integrator.moe_handler is not None
        assert integrator.moe_config.num_experts == 60
        assert integrator.moe_config.top_k == 4
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_deepseek_moe_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test DeepSeek-MoE model initialization."""
        config = MagicMock()
        config.model_type = "deepseek"
        config.architectures = ["DeepseekMoeForCausalLM"]
        config.num_hidden_layers = 28
        config.hidden_size = 2048
        config.n_routed_experts = 64
        config.n_shared_experts = 2
        config.num_experts_per_tok = 6
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="deepseek-ai/deepseek-moe-16b-base",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "moe"
        assert integrator.moe_handler is not None
        assert integrator.moe_config.has_shared_experts is True
        assert integrator.moe_config.num_shared_experts == 2


# =============================================================================
# Test Other Architecture Families
# =============================================================================

class TestOtherArchitectureIntegration:
    """Integration tests for other architecture families."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_t5_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test T5 model initialization."""
        config = MagicMock()
        config.model_type = "t5"
        config.architectures = ["T5ForConditionalGeneration"]
        config.num_hidden_layers = 12
        config.d_model = 768
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="google/flan-t5-base",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "t5"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_bloom_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test BLOOM model initialization."""
        config = MagicMock()
        config.model_type = "bloom"
        config.architectures = ["BloomForCausalLM"]
        config.num_hidden_layers = 24
        config.hidden_size = 1024
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="bigscience/bloom-560m",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "bloom"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_opt_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test OPT model initialization."""
        config = MagicMock()
        config.model_type = "opt"
        config.architectures = ["OPTForCausalLM"]
        config.num_hidden_layers = 12
        config.hidden_size = 768
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="facebook/opt-125m",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "opt"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_phi_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Phi model initialization."""
        config = MagicMock()
        config.model_type = "phi"
        config.architectures = ["PhiForCausalLM"]
        config.num_hidden_layers = 24
        config.hidden_size = 2048
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="microsoft/phi-2",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "phi"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_gemma_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Gemma model initialization."""
        config = MagicMock()
        config.model_type = "gemma"
        config.architectures = ["GemmaForCausalLM"]
        config.num_hidden_layers = 18
        config.hidden_size = 2048
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="google/gemma-2b",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "gemma"
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_mamba_model_initialization(self, mock_weight_loader, mock_tokenizer, mock_config, integration_dirs):
        """Test Mamba model initialization."""
        config = MagicMock()
        config.model_type = "mamba"
        config.architectures = ["MambaForCausalLM"]
        config.num_hidden_layers = 24
        config.hidden_size = 768
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        integrator = UniversalSLIIntegrator(
            model_id="state-spaces/mamba-370m",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "mamba"


# =============================================================================
# Test End-to-End Pipeline
# =============================================================================

class TestEndToEndPipeline:
    """Test end-to-end SLI pipeline."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_end_to_end_llama_pipeline(self, mock_weight_loader_cls, mock_tokenizer, mock_config, 
                                        integration_dirs, sample_text_dataset):
        """Test end-to-end pipeline for Llama-style model."""
        config = MagicMock()
        config.model_type = "llama"
        config.architectures = ["LlamaForCausalLM"]
        config.num_hidden_layers = 4  # Small for testing
        config.hidden_size = 128
        config.vocab_size = 1000
        
        mock_config.return_value = config
        
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = tokenizer
        
        # Mock weight loader
        mock_weight_loader = MagicMock()
        mock_weight_loader.load_embedding_weights.return_value = torch.randn(1000, 128)
        mock_weight_loader.load_layer_weights.return_value = {}
        mock_weight_loader_cls.return_value = mock_weight_loader
        
        integrator = UniversalSLIIntegrator(
            model_id="test/llama-model",
            output_dir=str(integration_dirs["output"]),
            cache_dir=str(integration_dirs["cache"]),
            activation_cache_dir=str(integration_dirs["activations"]),
            device="cpu"
        )
        
        # Mock layer creation and forward pass
        integrator._forward_batch_sli = MagicMock()
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.load_state_dict = MagicMock()
        integrator._create_layer = MagicMock(return_value=mock_layer)
        
        result = integrator.run_sli(sample_text_dataset, batch_size=2)
        
        assert result is not None
        assert "activation_cache_dir" in result
        assert "num_layers" in result
        assert result["num_layers"] == 4
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    def test_end_to_end_moe_pipeline(self, mock_weight_loader_cls, mock_tokenizer, mock_config,
                                      integration_dirs, sample_text_dataset):
        """Test end-to-end pipeline for MoE model."""
        config = MagicMock()
        config.model_type = "mixtral"
        config.architectures = ["MixtralForCausalLM"]
        config.num_hidden_layers = 4
        config.hidden_size = 128
        config.vocab_size = 1000
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        
        mock_config.return_value = config
        
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = tokenizer
        
        mock_weight_loader = MagicMock()
        mock_weight_loader.load_embedding_weights.return_value = torch.randn(1000, 128)
        mock_weight_loader.load_layer_weights.return_value = {}
        mock_weight_loader_cls.return_value = mock_weight_loader
        
        integrator = UniversalSLIIntegrator(
            model_id="test/mixtral-model",
            output_dir=str(integration_dirs["output"]),
            cache_dir=str(integration_dirs["cache"]),
            activation_cache_dir=str(integration_dirs["activations"]),
            device="cpu"
        )
        
        # Verify MoE handler is initialized
        assert integrator.moe_handler is not None
        
        # Mock layer creation and forward pass
        integrator._forward_batch_sli = MagicMock()
        mock_layer = MagicMock(spec=nn.Module)
        mock_layer.load_state_dict = MagicMock()
        integrator._create_layer = MagicMock(return_value=mock_layer)
        
        result = integrator.run_sli(sample_text_dataset, batch_size=1)
        
        assert result is not None
        assert integrator.get_model_summary()["is_moe"] is True


# =============================================================================
# Test Cross-Architecture Consistency
# =============================================================================

class TestCrossArchitectureConsistency:
    """Test consistency across architecture families."""
    
    def test_all_families_have_weight_prefix_method(self):
        """Test that all family handlers have get_layer_prefix method."""
        registry = ArchitectureRegistry()
        
        for family_id in registry.list_families():
            family = registry.get_family(family_id)
            assert hasattr(family, 'get_layer_prefix')
            assert callable(family.get_layer_prefix)
            
            # Should return a string
            prefix = family.get_layer_prefix(0)
            assert isinstance(prefix, str)
            assert len(prefix) > 0
    
    def test_all_families_have_create_layer_method(self):
        """Test that all family handlers have create_layer method."""
        registry = ArchitectureRegistry()
        
        for family_id in registry.list_families():
            family = registry.get_family(family_id)
            assert hasattr(family, 'create_layer')
            assert callable(family.create_layer)
    
    def test_all_families_have_get_embedding_name(self):
        """Test that all family handlers have get_embedding_name method."""
        registry = ArchitectureRegistry()
        
        for family_id in registry.list_families():
            family = registry.get_family(family_id)
            assert hasattr(family, 'get_embedding_name')
            assert callable(family.get_embedding_name)
            
            embedding_name = family.get_embedding_name()
            assert isinstance(embedding_name, str)
    
    def test_family_info_consistency(self):
        """Test that all family info has consistent structure."""
        registry = ArchitectureRegistry()
        info = registry.get_family_info()
        
        required_keys = ["name", "model_types", "architectures", "trust_remote_code"]
        
        for family_id, family_data in info.items():
            for key in required_keys:
                assert key in family_data, f"{family_id} missing {key}"


# =============================================================================
# Test Error Recovery
# =============================================================================

class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('src.nexus_final.sli.universal_sli_integrator.UniversalWeightLoader')
    @patch('src.nexus_final.sli.universal_sli_integrator.get_registry')
    def test_unsupported_architecture_fallback(self, mock_get_registry, mock_weight_loader, 
                                                mock_tokenizer, mock_config, integration_dirs):
        """Test fallback behavior for unsupported architectures."""
        config = MagicMock()
        config.model_type = "unknown_model"
        config.architectures = ["UnknownArchitecture"]
        
        mock_config.return_value = config
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        mock_tokenizer.return_value = tokenizer
        
        # Mock registry to raise UnsupportedArchitectureError
        mock_registry = MagicMock()
        from src.nexus_final.sli.exceptions import UnsupportedArchitectureError
        mock_registry.detect_family.side_effect = UnsupportedArchitectureError("unknown")
        mock_registry.get_family.return_value = LlamaFamilyHandler()
        mock_get_registry.return_value = mock_registry
        
        # Should not raise, but fallback to Llama
        integrator = UniversalSLIIntegrator(
            model_id="test/unknown-model",
            output_dir=str(integration_dirs["output"]),
            device="cpu"
        )
        
        assert integrator.family.family_id == "llama"
