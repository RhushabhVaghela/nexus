"""
Test suite for the Architecture Registry module.

This module tests the ArchitectureRegistry class and all architecture family handlers.
It covers:
- Registry initialization and singleton pattern
- Auto-detection of all 12 architecture families
- Registration of custom architectures
- Error handling for unknown architectures
- Family-specific functionality

Total test cases: ~40
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nexus_final.sli.architecture_registry import (
    ArchitectureRegistry,
    ArchitectureFamily,
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
    get_registry,
)
from src.nexus_final.sli.exceptions import UnsupportedArchitectureError, LayerCreationError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fresh_registry():
    """Create a fresh registry instance for testing (bypasses singleton)."""
    registry = ArchitectureRegistry.__new__(ArchitectureRegistry)
    registry._families = {}
    registry._register_default_families()
    return registry


@pytest.fixture
def mock_llama_config():
    """Create a mock Llama-style config."""
    config = MagicMock()
    config.model_type = "llama"
    config.architectures = ["LlamaForCausalLM"]
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.vocab_size = 32000
    return config


@pytest.fixture
def mock_qwen_config():
    """Create a mock Qwen-style config."""
    config = MagicMock()
    config.model_type = "qwen2"
    config.architectures = ["Qwen2ForCausalLM"]
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.vocab_size = 151936
    return config


@pytest.fixture
def mock_gpt2_config():
    """Create a mock GPT-2 style config."""
    config = MagicMock()
    config.model_type = "gpt2"
    config.architectures = ["GPT2LMHeadModel"]
    config.n_layer = 12
    config.n_embd = 768
    config.vocab_size = 50257
    return config


@pytest.fixture
def mock_chatglm_config():
    """Create a mock ChatGLM-style config."""
    config = MagicMock()
    config.model_type = "chatglm"
    config.architectures = ["ChatGLMForConditionalGeneration"]
    config.num_hidden_layers = 28
    config.hidden_size = 4096
    config.vocab_size = 65024
    config.name_or_path = "/tmp/chatglm"
    return config


@pytest.fixture
def mock_t5_config():
    """Create a mock T5-style config."""
    config = MagicMock()
    config.model_type = "t5"
    config.architectures = ["T5ForConditionalGeneration"]
    config.num_hidden_layers = 12
    config.d_model = 768
    config.vocab_size = 32128
    return config


@pytest.fixture
def mock_bloom_config():
    """Create a mock BLOOM-style config."""
    config = MagicMock()
    config.model_type = "bloom"
    config.architectures = ["BloomForCausalLM"]
    config.num_hidden_layers = 24
    config.hidden_size = 1024
    config.vocab_size = 250880
    return config


@pytest.fixture
def mock_opt_config():
    """Create a mock OPT-style config."""
    config = MagicMock()
    config.model_type = "opt"
    config.architectures = ["OPTForCausalLM"]
    config.num_hidden_layers = 12
    config.hidden_size = 768
    config.vocab_size = 50272
    return config


@pytest.fixture
def mock_mamba_config():
    """Create a mock Mamba-style config."""
    config = MagicMock()
    config.model_type = "mamba"
    config.architectures = ["MambaForCausalLM"]
    config.num_hidden_layers = 24
    config.hidden_size = 768
    config.vocab_size = 50280
    return config


@pytest.fixture
def mock_mixtral_config():
    """Create a mock Mixtral MoE-style config."""
    config = MagicMock()
    config.model_type = "mixtral"
    config.architectures = ["MixtralForCausalLM"]
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.vocab_size = 32000
    config.num_local_experts = 8
    config.num_experts_per_tok = 2
    return config


@pytest.fixture
def mock_phi_config():
    """Create a mock Phi-style config."""
    config = MagicMock()
    config.model_type = "phi"
    config.architectures = ["PhiForCausalLM"]
    config.num_hidden_layers = 24
    config.hidden_size = 2048
    config.vocab_size = 51200
    return config


@pytest.fixture
def mock_gemma_config():
    """Create a mock Gemma-style config."""
    config = MagicMock()
    config.model_type = "gemma"
    config.architectures = ["GemmaForCausalLM"]
    config.num_hidden_layers = 18
    config.hidden_size = 2048
    config.vocab_size = 256128
    return config


@pytest.fixture
def mock_deepseek_moe_config():
    """Create a mock DeepSeek MoE-style config."""
    config = MagicMock()
    config.model_type = "deepseek"
    config.architectures = ["DeepseekMoeForCausalLM"]
    config.num_hidden_layers = 28
    config.hidden_size = 2048
    config.vocab_size = 102400
    config.n_routed_experts = 64
    config.n_shared_experts = 2
    config.num_experts_per_tok = 6
    return config


@pytest.fixture
def mock_unknown_config():
    """Create a mock unknown architecture config."""
    config = MagicMock()
    config.model_type = "unknown_architecture"
    config.architectures = ["UnknownModel"]
    return config


# =============================================================================
# Test ArchitectureFamily Base Class
# =============================================================================

class TestArchitectureFamilyBase:
    """Test the ArchitectureFamily abstract base class."""
    
    def test_family_has_required_attributes(self):
        """Test that family classes have required class attributes."""
        families = [
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
        ]
        
        for family_cls in families:
            assert hasattr(family_cls, 'family_id'), f"{family_cls.__name__} missing family_id"
            assert hasattr(family_cls, 'family_name'), f"{family_cls.__name__} missing family_name"
            assert hasattr(family_cls, 'model_types'), f"{family_cls.__name__} missing model_types"
            assert hasattr(family_cls, 'architectures'), f"{family_cls.__name__} missing architectures"
    
    def test_family_matches_model_type(self):
        """Test family matching by model_type."""
        llama = LlamaFamilyHandler()
        assert llama.matches("llama", []) is True
        assert llama.matches("mistral", []) is True
        assert llama.matches("gpt2", []) is False
    
    def test_family_matches_architecture(self):
        """Test family matching by architecture name."""
        llama = LlamaFamilyHandler()
        assert llama.matches("", ["LlamaForCausalLM"]) is True
        assert llama.matches("", ["MistralForCausalLM"]) is True
        assert llama.matches("", ["GPT2LMHeadModel"]) is False
    
    def test_family_matches_partial_architecture(self):
        """Test family matching with partial architecture name match."""
        llama = LlamaFamilyHandler()
        assert llama.matches("", ["LlamaForSequenceClassification"]) is True
        assert llama.matches("", ["CustomLlamaModel"]) is True
    
    def test_family_matches_case_insensitive(self):
        """Test that family matching is case-insensitive."""
        llama = LlamaFamilyHandler()
        assert llama.matches("LLAMA", []) is True
        assert llama.matches("Llama", []) is True
        assert llama.matches("", ["LLAMAFORCAUSALLM"]) is True
    
    def test_family_get_embedding_name_default(self):
        """Test default embedding name."""
        llama = LlamaFamilyHandler()
        assert llama.get_embedding_name() == "model.embed_tokens"
    
    def test_family_get_lm_head_name_default(self):
        """Test default LM head name."""
        llama = LlamaFamilyHandler()
        assert llama.get_lm_head_name() == "lm_head"


# =============================================================================
# Test Architecture Registry Singleton
# =============================================================================

class TestArchitectureRegistrySingleton:
    """Test the ArchitectureRegistry singleton pattern."""
    
    def test_registry_is_singleton(self):
        """Test that registry follows singleton pattern."""
        reg1 = ArchitectureRegistry()
        reg2 = ArchitectureRegistry()
        assert reg1 is reg2
    
    def test_get_registry_returns_same_instance(self):
        """Test get_registry function returns singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
    
    def test_registry_initializes_once(self, fresh_registry):
        """Test registry initializes families only once."""
        families_before = len(fresh_registry.list_families())
        fresh_registry._register_default_families()
        families_after = len(fresh_registry.list_families())
        # Should not duplicate registrations
        assert families_before == families_after


# =============================================================================
# Test Architecture Auto-Detection - All 12 Families
# =============================================================================

class TestArchitectureAutoDetection:
    """Test auto-detection for all 12 architecture families."""
    
    def test_detect_llama_family(self, fresh_registry, mock_llama_config):
        """Test detection of Llama family architectures."""
        family = fresh_registry.detect_family(mock_llama_config)
        assert family.family_id == "llama"
    
    def test_detect_qwen_family(self, fresh_registry, mock_qwen_config):
        """Test detection of Qwen family architectures."""
        family = fresh_registry.detect_family(mock_qwen_config)
        assert family.family_id == "qwen"
    
    def test_detect_gpt_family(self, fresh_registry, mock_gpt2_config):
        """Test detection of GPT family architectures."""
        family = fresh_registry.detect_family(mock_gpt2_config)
        assert family.family_id == "gpt"
    
    def test_detect_chatglm_family(self, fresh_registry, mock_chatglm_config):
        """Test detection of ChatGLM family architectures."""
        family = fresh_registry.detect_family(mock_chatglm_config)
        assert family.family_id == "chatglm"
    
    def test_detect_t5_family(self, fresh_registry, mock_t5_config):
        """Test detection of T5 family architectures."""
        family = fresh_registry.detect_family(mock_t5_config)
        assert family.family_id == "t5"
    
    def test_detect_bloom_family(self, fresh_registry, mock_bloom_config):
        """Test detection of BLOOM family architectures."""
        family = fresh_registry.detect_family(mock_bloom_config)
        assert family.family_id == "bloom"
    
    def test_detect_opt_family(self, fresh_registry, mock_opt_config):
        """Test detection of OPT family architectures."""
        family = fresh_registry.detect_family(mock_opt_config)
        assert family.family_id == "opt"
    
    def test_detect_mamba_family(self, fresh_registry, mock_mamba_config):
        """Test detection of Mamba family architectures."""
        family = fresh_registry.detect_family(mock_mamba_config)
        assert family.family_id == "mamba"
    
    def test_detect_moe_family_mixtral(self, fresh_registry, mock_mixtral_config):
        """Test detection of MoE family (Mixtral) architectures."""
        family = fresh_registry.detect_family(mock_mixtral_config)
        assert family.family_id == "moe"
    
    def test_detect_phi_family(self, fresh_registry, mock_phi_config):
        """Test detection of Phi family architectures."""
        family = fresh_registry.detect_family(mock_phi_config)
        assert family.family_id == "phi"
    
    def test_detect_gemma_family(self, fresh_registry, mock_gemma_config):
        """Test detection of Gemma family architectures."""
        family = fresh_registry.detect_family(mock_gemma_config)
        assert family.family_id == "gemma"
    
    def test_detect_deepseek_moe_family(self, fresh_registry, mock_deepseek_moe_config):
        """Test detection of DeepSeek MoE (detected as MoE family)."""
        family = fresh_registry.detect_family(mock_deepseek_moe_config)
        assert family.family_id == "moe"
    
    def test_detect_unknown_family_raises_error(self, fresh_registry, mock_unknown_config):
        """Test that unknown architectures raise UnsupportedArchitectureError."""
        with pytest.raises(UnsupportedArchitectureError) as exc_info:
            fresh_registry.detect_family(mock_unknown_config)
        
        assert "unknown_architecture" in str(exc_info.value)
        assert "UnknownModel" in str(exc_info.value)


# =============================================================================
# Test Layer Prefix Generation
# =============================================================================

class TestLayerPrefixGeneration:
    """Test layer prefix generation for each family."""
    
    def test_llama_layer_prefix(self):
        """Test Llama layer prefix format."""
        handler = LlamaFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.layers.0."
        assert handler.get_layer_prefix(5) == "model.layers.5."
        assert handler.get_layer_prefix(31) == "model.layers.31."
    
    def test_qwen_layer_prefix(self):
        """Test Qwen layer prefix format."""
        handler = QwenFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.layers.0."
        assert handler.get_layer_prefix(10) == "model.layers.10."
    
    def test_gpt_layer_prefix(self):
        """Test GPT layer prefix format."""
        handler = GPTFamilyHandler()
        assert handler.get_layer_prefix(0) == "transformer.h.0."
        assert handler.get_layer_prefix(5) == "transformer.h.5."
    
    def test_chatglm_layer_prefix(self):
        """Test ChatGLM layer prefix format."""
        handler = ChatGLMFamilyHandler()
        assert handler.get_layer_prefix(0) == "transformer.encoder.layers.0."
        assert handler.get_layer_prefix(10) == "transformer.encoder.layers.10."
    
    def test_t5_layer_prefix(self):
        """Test T5 layer prefix format (encoder/decoder)."""
        handler = T5FamilyHandler()
        assert handler.get_layer_prefix(0, "encoder") == "encoder.block.0."
        assert handler.get_layer_prefix(0, "decoder") == "decoder.block.0."
        assert handler.get_layer_prefix(5, "encoder") == "encoder.block.5."
    
    def test_bloom_layer_prefix(self):
        """Test BLOOM layer prefix format."""
        handler = BLOOMFamilyHandler()
        assert handler.get_layer_prefix(0) == "transformer.h.0."
        assert handler.get_layer_prefix(10) == "transformer.h.10."
    
    def test_opt_layer_prefix(self):
        """Test OPT layer prefix format."""
        handler = OPTFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.decoder.layers.0."
        assert handler.get_layer_prefix(10) == "model.decoder.layers.10."
    
    def test_mamba_layer_prefix(self):
        """Test Mamba layer prefix format."""
        handler = MambaFamilyHandler()
        assert handler.get_layer_prefix(0) == "backbone.layers.0."
        assert handler.get_layer_prefix(10) == "backbone.layers.10."
    
    def test_moe_layer_prefix(self):
        """Test MoE layer prefix format."""
        handler = MoEFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.layers.0."
        assert handler.get_layer_prefix(10) == "model.layers.10."
    
    def test_phi_layer_prefix(self):
        """Test Phi layer prefix format."""
        handler = PhiFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.layers.0."
        assert handler.get_layer_prefix(10) == "model.layers.10."
    
    def test_gemma_layer_prefix(self):
        """Test Gemma layer prefix format."""
        handler = GemmaFamilyHandler()
        assert handler.get_layer_prefix(0) == "model.layers.0."
        assert handler.get_layer_prefix(10) == "model.layers.10."


# =============================================================================
# Test Config Attribute Extraction
# =============================================================================

class TestConfigAttributeExtraction:
    """Test extraction of config attributes."""
    
    def test_get_num_layers_variants(self, fresh_registry):
        """Test getting num layers with different attribute names."""
        handler = LlamaFamilyHandler()
        
        config1 = MagicMock()
        config1.num_hidden_layers = 32
        assert handler.get_num_layers(config1) == 32
        
        config2 = MagicMock()
        config2.n_layer = 12
        assert handler.get_num_layers(config2) == 12
        
        config3 = MagicMock()
        config3.num_layers = 24
        assert handler.get_num_layers(config3) == 24
    
    def test_get_num_layers_raises_on_missing(self, fresh_registry):
        """Test that missing layer count raises ValueError."""
        handler = LlamaFamilyHandler()
        config = MagicMock()
        config.spec = []
        
        with pytest.raises(ValueError) as exc_info:
            handler.get_num_layers(config)
        assert "Cannot determine number of layers" in str(exc_info.value)
    
    def test_get_hidden_size_variants(self, fresh_registry):
        """Test getting hidden size with different attribute names."""
        handler = LlamaFamilyHandler()
        
        config1 = MagicMock()
        config1.hidden_size = 4096
        assert handler.get_hidden_size(config1) == 4096
        
        config2 = MagicMock()
        config2.d_model = 768
        assert handler.get_hidden_size(config2) == 768
        
        config3 = MagicMock()
        config3.n_embd = 512
        assert handler.get_hidden_size(config3) == 512
    
    def test_get_vocab_size_variants(self, fresh_registry):
        """Test getting vocab size with different attribute names."""
        handler = LlamaFamilyHandler()
        
        config1 = MagicMock()
        config1.vocab_size = 32000
        assert handler.get_vocab_size(config1) == 32000
        
        config2 = MagicMock()
        config2.n_vocab = 50257
        assert handler.get_vocab_size(config2) == 50257


# =============================================================================
# Test Custom Architecture Registration
# =============================================================================

class TestCustomArchitectureRegistration:
    """Test registration of custom architecture families."""
    
    def test_register_custom_family(self, fresh_registry):
        """Test registering a custom architecture family."""
        custom_family = MagicMock(spec=ArchitectureFamily)
        custom_family.family_id = "custom"
        
        fresh_registry.register("custom", custom_family)
        
        retrieved = fresh_registry.get_family("custom")
        assert retrieved is custom_family
    
    def test_list_families_includes_custom(self, fresh_registry):
        """Test that custom families appear in list_families."""
        custom_family = MagicMock(spec=ArchitectureFamily)
        custom_family.family_id = "custom"
        
        original_count = len(fresh_registry.list_families())
        fresh_registry.register("custom", custom_family)
        
        families = fresh_registry.list_families()
        assert "custom" in families
        assert len(families) == original_count + 1
    
    def test_get_family_returns_none_for_unknown(self, fresh_registry):
        """Test that get_family returns None for unknown family."""
        result = fresh_registry.get_family("nonexistent")
        assert result is None


# =============================================================================
# Test Family Info Retrieval
# =============================================================================

class TestFamilyInfoRetrieval:
    """Test retrieval of family information."""
    
    def test_get_family_info_returns_dict(self, fresh_registry):
        """Test that get_family_info returns proper structure."""
        info = fresh_registry.get_family_info()
        
        assert isinstance(info, dict)
        assert "llama" in info
        assert "qwen" in info
        assert "gpt" in info
    
    def test_family_info_contains_required_fields(self, fresh_registry):
        """Test that family info contains all required fields."""
        info = fresh_registry.get_family_info()
        
        for family_id, family_data in info.items():
            assert "name" in family_data
            assert "model_types" in family_data
            assert "architectures" in family_data
            assert "trust_remote_code" in family_data
    
    def test_llama_family_info_content(self, fresh_registry):
        """Test Llama family info content."""
        info = fresh_registry.get_family_info()
        llama_info = info["llama"]
        
        assert "Llama-Based" in llama_info["name"]
        assert "llama" in llama_info["model_types"]
        assert "mistral" in llama_info["model_types"]
        assert "LlamaForCausalLM" in llama_info["architectures"]


# =============================================================================
# Test MoE Model Detection
# =============================================================================

class TestMoEModelDetection:
    """Test MoE model detection logic."""
    
    def test_is_moe_model_by_num_local_experts(self, fresh_registry):
        """Test MoE detection via num_local_experts."""
        config = MagicMock()
        config.num_local_experts = 8
        config.n_routed_experts = None
        config.moe_intermediate_size = None
        
        assert fresh_registry._is_moe_model(config) is True
    
    def test_is_moe_model_by_n_routed_experts(self, fresh_registry):
        """Test MoE detection via n_routed_experts."""
        config = MagicMock()
        config.num_local_experts = None
        config.n_routed_experts = 64
        config.moe_intermediate_size = None
        
        assert fresh_registry._is_moe_model(config) is True
    
    def test_is_moe_model_by_moe_intermediate_size(self, fresh_registry):
        """Test MoE detection via moe_intermediate_size."""
        config = MagicMock()
        config.num_local_experts = None
        config.n_routed_experts = None
        config.moe_intermediate_size = 2048
        
        assert fresh_registry._is_moe_model(config) is True
    
    def test_is_not_moe_model(self, fresh_registry):
        """Test non-MoE detection."""
        config = MagicMock()
        config.num_local_experts = None
        config.n_routed_experts = None
        config.moe_intermediate_size = None
        
        assert fresh_registry._is_moe_model(config) is False


# =============================================================================
# Test Family-Specific Features
# =============================================================================

class TestFamilySpecificFeatures:
    """Test family-specific features and behaviors."""
    
    def test_chatglm_trust_remote_code(self):
        """Test ChatGLM requires trust_remote_code."""
        handler = ChatGLMFamilyHandler()
        assert handler.trust_remote_code is True
    
    def test_llama_no_trust_remote_code(self):
        """Test Llama doesn't require trust_remote_code."""
        handler = LlamaFamilyHandler()
        assert handler.trust_remote_code is False
    
    def test_moe_get_num_experts(self, mock_mixtral_config):
        """Test MoE get_num_experts method."""
        handler = MoEFamilyHandler()
        assert handler.get_num_experts(mock_mixtral_config) == 8
    
    def test_moe_get_top_k(self, mock_mixtral_config):
        """Test MoE get_top_k method."""
        handler = MoEFamilyHandler()
        assert handler.get_top_k(mock_mixtral_config) == 2
    
    def test_moe_get_expert_prefix(self):
        """Test MoE expert prefix generation."""
        handler = MoEFamilyHandler()
        prefix = handler.get_expert_prefix(5, 3)
        assert prefix == "model.layers.5.block_sparse_moe.experts.3."
    
    def test_moe_detect_subtype_mixtral(self, mock_mixtral_config):
        """Test MoE subtype detection for Mixtral."""
        handler = MoEFamilyHandler()
        assert handler._detect_subtype(mock_mixtral_config) == "mixtral"
    
    def test_moe_detect_subtype_deepseek(self, mock_deepseek_moe_config):
        """Test MoE subtype detection for DeepSeek."""
        handler = MoEFamilyHandler()
        assert handler._detect_subtype(mock_deepseek_moe_config) == "deepseek"
    
    def test_gpt_detect_subtype_gpt2(self, mock_gpt2_config):
        """Test GPT subtype detection for GPT-2."""
        handler = GPTFamilyHandler()
        assert handler._detect_subtype(mock_gpt2_config) == "gpt2"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_unsupported_architecture_error_message(self):
        """Test UnsupportedArchitectureError message format."""
        error = UnsupportedArchitectureError("unknown", ["UnknownModel"])
        message = str(error)
        
        assert "unknown" in message
        assert "UnknownModel" in message
        assert "not yet supported" in message
    
    def test_layer_creation_error(self):
        """Test LayerCreationError."""
        cause = ImportError("No module named 'transformers'")
        error = LayerCreationError(5, "llama", cause)
        message = str(error)
        
        assert "layer 5" in message
        assert "llama" in message
        assert "No module named" in message


# =============================================================================
# Test Variant Configurations
# =============================================================================

class TestVariantConfigurations:
    """Test detection of architecture variants."""
    
    def test_detect_llama3_variant(self, fresh_registry):
        """Test detection of Llama3 variant."""
        config = MagicMock()
        config.model_type = "llama3"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "llama"
    
    def test_detect_codellama_variant(self, fresh_registry):
        """Test detection of CodeLlama variant."""
        config = MagicMock()
        config.model_type = "codellama"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "llama"
    
    def test_detect_vicuna_variant(self, fresh_registry):
        """Test detection of Vicuna variant."""
        config = MagicMock()
        config.model_type = "vicuna"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "llama"
    
    def test_detect_qwen2_5_variant(self, fresh_registry):
        """Test detection of Qwen2.5 variant."""
        config = MagicMock()
        config.model_type = "qwen2_5"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "qwen"
    
    def test_detect_glm4_variant(self, fresh_registry):
        """Test detection of GLM4 variant."""
        config = MagicMock()
        config.model_type = "glm4"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "chatglm"
    
    def test_detect_mamba2_variant(self, fresh_registry):
        """Test detection of Mamba2 variant."""
        config = MagicMock()
        config.model_type = "mamba2"
        config.architectures = []
        
        family = fresh_registry.detect_family(config)
        assert family.family_id == "mamba"
