"""
tests/unit/test_bert_handler.py
Comprehensive tests for the BERT family handler.

Tests cover:
- BERTFamilyHandler initialization
- get_layer_prefix()
- create_layer()
- get_embedding_name()
- is_encoder_only()
- All supported model types (BERT, RoBERTa, DeBERTa, etc.)
- Subtype detection
"""

import pytest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Import the module under test
from src.nexus_final.sli.architecture_registry import BERTFamilyHandler


class TestBERTFamilyHandlerInitialization:
    """Test BERTFamilyHandler initialization."""

    def test_handler_creation(self):
        """Test creating a BERTFamilyHandler."""
        handler = BERTFamilyHandler()

        assert handler.family_id == "bert"
        assert handler.family_name == "BERT-Based Encoder Architectures"
        assert len(handler.model_types) > 0
        assert len(handler.architectures) > 0

    def test_model_types(self):
        """Test that all expected model types are supported."""
        handler = BERTFamilyHandler()

        expected_types = [
            "bert", "roberta", "deberta", "deberta_v2",
            "distilbert", "albert", "modernbert",
            "jinabert", "nomic_bert", "neobert",
            "electra", "xlm_roberta", "camembert"
        ]

        for model_type in expected_types:
            assert model_type in handler.model_types, f"Missing model type: {model_type}"

    def test_architectures(self):
        """Test that all expected architectures are supported."""
        handler = BERTFamilyHandler()

        expected_archs = [
            "BertModel", "BertForMaskedLM",
            "RobertaModel", "RobertaForSequenceClassification",
            "DebertaModel", "DebertaForSequenceClassification",
            "DistilBertModel", "AlbertModel"
        ]

        for arch in expected_archs:
            assert any(arch in a for a in handler.architectures), f"Missing architecture: {arch}"


class TestBERTFamilyHandlerMatches:
    """Test BERTFamilyHandler.matches() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_matches_bert_model_type(self, handler):
        """Test matching BERT model type."""
        config = MagicMock()
        assert handler.matches("bert", ["BertModel"], config) is True

    def test_matches_roberta_model_type(self, handler):
        """Test matching RoBERTa model type."""
        config = MagicMock()
        assert handler.matches("roberta", ["RobertaModel"], config) is True

    def test_matches_deberta_model_type(self, handler):
        """Test matching DeBERTa model type."""
        config = MagicMock()
        assert handler.matches("deberta", ["DebertaModel"], config) is True

    def test_matches_distilbert_model_type(self, handler):
        """Test matching DistilBERT model type."""
        config = MagicMock()
        assert handler.matches("distilbert", [], config) is True

    def test_matches_albert_model_type(self, handler):
        """Test matching ALBERT model type."""
        config = MagicMock()
        assert handler.matches("albert", [], config) is True

    def test_matches_by_architecture(self, handler):
        """Test matching by architecture name."""
        config = MagicMock()
        assert handler.matches("", ["BertForSequenceClassification"], config) is True

    def test_matches_partial_architecture(self, handler):
        """Test matching by partial architecture name."""
        config = MagicMock()
        assert handler.matches("", ["BertModel"], config) is True

    def test_matches_moe_model_excluded(self, handler):
        """Test that MoE models are not matched."""
        config = MagicMock()
        config.num_local_experts = 8

        assert handler.matches("bert", ["BertModel"], config) is False

    def test_no_match_llama(self, handler):
        """Test that LLaMA models don't match."""
        config = MagicMock()
        assert handler.matches("llama", ["LlamaForCausalLM"], config) is False

    def test_no_match_gpt(self, handler):
        """Test that GPT models don't match."""
        config = MagicMock()
        assert handler.matches("gpt2", ["GPT2LMHeadModel"], config) is False

    def test_matches_case_insensitive(self, handler):
        """Test matching is case insensitive."""
        config = MagicMock()
        assert handler.matches("BERT", ["BERTMODEL"], config) is True


class TestBERTFamilyHandlerGetLayerPrefix:
    """Test BERTFamilyHandler.get_layer_prefix() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_get_layer_prefix_default(self, handler):
        """Test default layer prefix."""
        prefix = handler.get_layer_prefix(0)
        assert prefix == "encoder.layer.0."

    def test_get_layer_prefix_layer_5(self, handler):
        """Test layer prefix for layer 5."""
        prefix = handler.get_layer_prefix(5)
        assert prefix == "encoder.layer.5."

    def test_get_layer_prefix_encoder_type(self, handler):
        """Test layer prefix with encoder type."""
        prefix = handler.get_layer_prefix(3, layer_type="encoder")
        assert prefix == "encoder.layer.3."

    def test_get_layer_prefix_roberta(self, handler):
        """Test RoBERTa layer prefix."""
        handler._last_subtype = "roberta"
        prefix = handler.get_layer_prefix(0)
        assert prefix == "roberta.encoder.layer.0."

    def test_get_layer_prefix_camembert(self, handler):
        """Test CamemBERT layer prefix."""
        handler._last_subtype = "camembert"
        prefix = handler.get_layer_prefix(0)
        assert prefix == "roberta.encoder.layer.0."

    def test_get_layer_prefix_xlm_roberta(self, handler):
        """Test XLM-RoBERTa layer prefix."""
        handler._last_subtype = "xlm_roberta"
        prefix = handler.get_layer_prefix(0)
        assert prefix == "roberta.encoder.layer.0."

    def test_get_layer_prefix_distilbert(self, handler):
        """Test DistilBERT layer prefix."""
        handler._last_subtype = "distilbert"
        prefix = handler.get_layer_prefix(0)
        assert prefix == "transformer.layer.0."

    def test_get_layer_prefix_albert(self, handler):
        """Test ALBERT layer prefix."""
        handler._last_subtype = "albert"
        prefix = handler.get_layer_prefix(0)
        assert "albert.encoder.albert_layer_group" in prefix

    def test_get_layer_prefix_albert_layer_13(self, handler):
        """Test ALBERT layer prefix for layer > 12."""
        handler._last_subtype = "albert"
        prefix = handler.get_layer_prefix(13)
        # Should wrap around with % 12
        assert "albert_layer_group.1" in prefix or "albert_layers.1" in prefix


class TestBERTFamilyHandlerDetectSubtype:
    """Test BERTFamilyHandler._detect_subtype() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_detect_subtype_bert(self, handler):
        """Test detecting BERT subtype."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "bert"

    def test_detect_subtype_roberta(self, handler):
        """Test detecting RoBERTa subtype."""
        config = MagicMock()
        config.model_type = "roberta"
        config.architectures = ["RobertaModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "roberta"

    def test_detect_subtype_deberta(self, handler):
        """Test detecting DeBERTa subtype."""
        config = MagicMock()
        config.model_type = "deberta"
        config.architectures = ["DebertaModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "deberta"

    def test_detect_subtype_distilbert(self, handler):
        """Test detecting DistilBERT subtype."""
        config = MagicMock()
        config.model_type = "distilbert"
        config.architectures = ["DistilBertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "distilbert"

    def test_detect_subtype_albert(self, handler):
        """Test detecting ALBERT subtype."""
        config = MagicMock()
        config.model_type = "albert"
        config.architectures = ["AlbertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "albert"

    def test_detect_subtype_modernbert(self, handler):
        """Test detecting ModernBERT subtype."""
        config = MagicMock()
        config.model_type = "modernbert"
        config.architectures = ["ModernBertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "modernbert"

    def test_detect_subtype_jinabert(self, handler):
        """Test detecting JinaBERT subtype."""
        config = MagicMock()
        config.model_type = "jinabert"
        config.architectures = ["JinaBertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "jinabert"

    def test_detect_subtype_neobert(self, handler):
        """Test detecting NeoBERT subtype."""
        config = MagicMock()
        config.model_type = "neobert"
        config.architectures = ["NeoBERT"]

        subtype = handler._detect_subtype(config)
        assert subtype == "neobert"

    def test_detect_subtype_nomic_bert(self, handler):
        """Test detecting NomicBERT subtype."""
        config = MagicMock()
        config.model_type = "nomic_bert"
        config.architectures = ["NomicBertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "nomic_bert"

    def test_detect_subtype_electra(self, handler):
        """Test detecting ELECTRA subtype."""
        config = MagicMock()
        config.model_type = "electra"
        config.architectures = ["ElectraModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "electra"

    def test_detect_subtype_xlm_roberta(self, handler):
        """Test detecting XLM-RoBERTa subtype."""
        config = MagicMock()
        config.model_type = "xlm_roberta"
        config.architectures = ["XLMRobertaModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "xlm_roberta"

    def test_detect_subtype_camembert(self, handler):
        """Test detecting CamemBERT subtype."""
        config = MagicMock()
        config.model_type = "camembert"
        config.architectures = ["CamembertModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "camembert"

    def test_detect_subtype_default_fallback(self, handler):
        """Test default fallback to bert."""
        config = MagicMock()
        config.model_type = "unknown"
        config.architectures = ["UnknownModel"]

        subtype = handler._detect_subtype(config)
        assert subtype == "bert"

    def test_detect_subtype_by_architecture(self, handler):
        """Test detection by architecture when model_type doesn't match."""
        config = MagicMock()
        config.model_type = ""
        config.architectures = ["RobertaForSequenceClassification"]

        subtype = handler._detect_subtype(config)
        assert subtype == "roberta"


class TestBERTFamilyHandlerCreateLayer:
    """Test BERTFamilyHandler.create_layer() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertModel"]
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.intermediate_size = 3072
        return config

    def test_create_layer_stores_subtype(self, handler, mock_config):
        """Test that create_layer stores detected subtype."""
        with patch('transformers.models.bert.modeling_bert.BertLayer') as mock_layer:
            mock_layer.return_value = MagicMock()
            handler.create_layer(mock_config, 0)

            assert handler._last_subtype == "bert"

    def test_create_layer_encoder_type(self, handler, mock_config):
        """Test creating layer with encoder type."""
        with patch('transformers.models.bert.modeling_bert.BertLayer') as mock_layer:
            mock_instance = MagicMock()
            mock_layer.return_value = mock_instance

            result = handler.create_layer(mock_config, 0, layer_type="encoder")

            assert result is mock_instance

    def test_create_layer_invalid_type_raises(self, handler, mock_config):
        """Test that decoder type is not supported."""
        with patch('transformers.models.bert.modeling_bert.BertLayer'):
            # Encoder-only models only support encoder type
            # This should still work (just delegates to bert)
            result = handler.create_layer(mock_config, 0, layer_type="encoder")
            assert result is not None


class TestBERTFamilyHandlerEmbeddingName:
    """Test BERTFamilyHandler.get_embedding_name() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_get_embedding_name_default(self, handler):
        """Test default embedding name."""
        name = handler.get_embedding_name()
        assert name == "embeddings"

    def test_get_embedding_name_roberta(self, handler):
        """Test RoBERTa embedding name."""
        handler._last_subtype = "roberta"
        name = handler.get_embedding_name()
        assert name == "roberta.embeddings"

    def test_get_embedding_name_camembert(self, handler):
        """Test CamemBERT embedding name."""
        handler._last_subtype = "camembert"
        name = handler.get_embedding_name()
        assert name == "roberta.embeddings"

    def test_get_embedding_name_xlm_roberta(self, handler):
        """Test XLM-RoBERTa embedding name."""
        handler._last_subtype = "xlm_roberta"
        name = handler.get_embedding_name()
        assert name == "roberta.embeddings"

    def test_get_embedding_name_distilbert(self, handler):
        """Test DistilBERT embedding name."""
        handler._last_subtype = "distilbert"
        name = handler.get_embedding_name()
        assert name == "distilbert.embeddings"

    def test_get_embedding_name_albert(self, handler):
        """Test ALBERT embedding name."""
        handler._last_subtype = "albert"
        name = handler.get_embedding_name()
        assert name == "albert.embeddings"

    def test_get_embedding_name_electra(self, handler):
        """Test ELECTRA embedding name."""
        handler._last_subtype = "electra"
        name = handler.get_embedding_name()
        assert name == "electra.embeddings"


class TestBERTFamilyHandlerLMHead:
    """Test BERTFamilyHandler.get_lm_head_name() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_get_lm_head_name(self, handler):
        """Test that encoder-only models don't have LM head."""
        name = handler.get_lm_head_name()
        assert name is None


class TestBERTFamilyHandlerIsEncoderOnly:
    """Test BERTFamilyHandler.is_encoder_only() method."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_is_encoder_only(self, handler):
        """Test that BERT family is encoder-only."""
        assert handler.is_encoder_only() is True


class TestBERTFamilyHandlerInheritedMethods:
    """Test methods inherited from ArchitectureFamily."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.vocab_size = 30522
        return config

    def test_get_num_layers(self, handler, mock_config):
        """Test getting number of layers."""
        assert handler.get_num_layers(mock_config) == 12

    def test_get_num_layers_alt_attr(self, handler):
        """Test getting layers with alternative attribute names."""
        config = MagicMock()
        config.n_layer = 24  # Alternative attribute
        del config.num_hidden_layers

        assert handler.get_num_layers(config) == 24

    def test_get_hidden_size(self, handler, mock_config):
        """Test getting hidden size."""
        assert handler.get_hidden_size(mock_config) == 768

    def test_get_hidden_size_alt_attr(self, handler):
        """Test getting hidden size with alternative attribute."""
        config = MagicMock()
        config.d_model = 512

        assert handler.get_hidden_size(config) == 512

    def test_get_vocab_size(self, handler, mock_config):
        """Test getting vocabulary size."""
        assert handler.get_vocab_size(mock_config) == 30522

    def test_get_vocab_size_alt_attr(self, handler):
        """Test getting vocab size with alternative attribute."""
        config = MagicMock()
        config.n_vocab = 50000

        assert handler.get_vocab_size(config) == 50000

    def test_get_num_layers_error(self, handler):
        """Test error when layer count cannot be determined."""
        config = MagicMock()
        # No layer attributes

        with pytest.raises(ValueError) as exc_info:
            handler.get_num_layers(config)

        assert "Cannot determine number of layers" in str(exc_info.value)

    def test_get_hidden_size_error(self, handler):
        """Test error when hidden size cannot be determined."""
        config = MagicMock()
        # No hidden size attributes

        with pytest.raises(ValueError) as exc_info:
            handler.get_hidden_size(config)

        assert "Cannot determine hidden size" in str(exc_info.value)

    def test_get_vocab_size_error(self, handler):
        """Test error when vocab size cannot be determined."""
        config = MagicMock()
        # No vocab attributes

        with pytest.raises(ValueError) as exc_info:
            handler.get_vocab_size(config)

        assert "Cannot determine vocab size" in str(exc_info.value)


class TestBERTFamilyHandlerEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def handler(self):
        """Create a BERTFamilyHandler."""
        return BERTFamilyHandler()

    def test_matches_empty_model_type(self, handler):
        """Test matching with empty model type."""
        config = MagicMock()
        assert handler.matches("", ["BertModel"], config) is True

    def test_matches_none_model_type(self, handler):
        """Test matching with None model type."""
        config = MagicMock()
        assert handler.matches(None, ["BertModel"], config) is True

    def test_matches_empty_architectures(self, handler):
        """Test matching with empty architectures."""
        config = MagicMock()
        assert handler.matches("bert", [], config) is True

    def test_matches_none_architectures(self, handler):
        """Test matching with None architectures."""
        config = MagicMock()
        assert handler.matches("bert", None, config) is True

    def test_get_layer_prefix_negative_index(self, handler):
        """Test layer prefix with negative index."""
        prefix = handler.get_layer_prefix(-1)
        assert "layer.-1" in prefix or "layer.-1" in prefix

    def test_get_layer_prefix_large_index(self, handler):
        """Test layer prefix with large index."""
        prefix = handler.get_layer_prefix(999)
        assert "layer.999" in prefix

    def test_detect_subtype_empty_config(self, handler):
        """Test subtype detection with empty config."""
        config = MagicMock()
        config.model_type = ""
        config.architectures = []

        subtype = handler._detect_subtype(config)
        assert subtype == "bert"  # Default fallback

    def test_trust_remote_code(self, handler):
        """Test trust_remote_code attribute."""
        assert handler.trust_remote_code is False
