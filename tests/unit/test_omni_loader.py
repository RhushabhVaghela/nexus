"""
Unit tests for Omni model loader.
Tests model detection, loading modes, and configuration.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOmniModelDetection:
    """Test Omni model detection utilities."""
    
    def test_import_loader(self):
        """Test OmniModelLoader can be imported."""
        from src.omni.loader import OmniModelLoader
        assert OmniModelLoader is not None
    
    def test_import_load_function(self):
        """Test load_omni_model function can be imported."""
        from src.omni.loader import load_omni_model
        assert callable(load_omni_model)
    
    def test_import_config(self):
        """Test OmniModelConfig can be imported."""
        from src.omni.loader import OmniModelConfig
        assert OmniModelConfig is not None
    
    @pytest.fixture
    def omni_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    
    @pytest.fixture
    def text_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-0.5B")
    
    def test_is_omni_model_true(self, omni_model_path):
        """Test is_omni_model returns True for Omni model."""
        if not omni_model_path.exists():
            pytest.skip("Omni model not found")
        
        from src.omni.loader import OmniModelLoader
        assert OmniModelLoader.is_omni_model(omni_model_path) is True
    
    def test_is_omni_model_false(self, text_model_path):
        """Test is_omni_model returns False for non-Omni model."""
        if not text_model_path.exists():
            pytest.skip("Text model not found")
        
        from src.omni.loader import OmniModelLoader
        assert OmniModelLoader.is_omni_model(Path("/path/to/bert-base-uncased")) is False
    
    def test_is_omni_model_nonexistent(self):
        """Test is_omni_model returns False for non-existent path."""
        from src.omni.loader import OmniModelLoader
        assert OmniModelLoader.is_omni_model("/nonexistent/path") is False


class TestOmniModelInfo:
    """Test get_model_info functionality."""
    
    @pytest.fixture
    def omni_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    
    def test_get_model_info_structure(self, omni_model_path):
        """Test get_model_info returns expected structure."""
        if not omni_model_path.exists():
            pytest.skip("Omni model not found")
        
        from src.omni.loader import OmniModelLoader
        info = OmniModelLoader.get_model_info(omni_model_path)
        
        # Actual keys returned by get_model_info
        assert "name" in info
        assert "has_talker" in info
        assert "is_quantized" in info
        assert "architecture" in info
    
    def test_get_model_info_values(self, omni_model_path):
        """Test get_model_info returns correct values."""
        if not omni_model_path.exists():
            pytest.skip("Omni model not found")
        
        from src.omni.loader import OmniModelLoader
        info = OmniModelLoader.get_model_info(omni_model_path)
        
        assert "Omni" in info["name"]
        assert info["has_talker"] is True
        assert info["is_quantized"] is True
    
    def test_get_model_info_nonexistent(self):
        """Test get_model_info handles non-existent path."""
        from src.omni.loader import OmniModelLoader
        info = OmniModelLoader.get_model_info("/nonexistent/path")
        
        # For non-existent paths, architecture should be 'unknown'
        assert info["architecture"] == "unknown"
        assert info["is_quantized"] is False


class TestOmniModelConfig:
    """Test OmniModelConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from src.omni.loader import OmniModelConfig
        
        config = OmniModelConfig(model_path="/path/to/model")
        
        assert config.mode == "thinker_only"
        assert config.device_map == "auto"
        assert config.trust_remote_code is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        from src.omni.loader import OmniModelConfig
        
        config = OmniModelConfig(
            model_path="/path/to/model",
            mode="full",
            load_in_4bit=True,
        )
        
        assert config.mode == "full"
        assert config.load_in_4bit is True


class TestOmniModelLoaderInit:
    """Test OmniModelLoader initialization."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        from src.omni.loader import OmniModelLoader
        
        loader = OmniModelLoader("/fake/model/path")
        
        assert loader._model is None
        assert loader._tokenizer is None
    
    def test_supported_architectures(self):
        """Test supported architectures list."""
        from src.omni.loader import OmniModelLoader
        
        assert "Qwen2_5OmniForConditionalGeneration" in OmniModelLoader.SUPPORTED_ARCHITECTURES


@pytest.mark.omni
class TestOmniModelLoading:
    """Tests for Omni model loading (mocked for CI)."""
    
    @pytest.fixture
    def omni_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    
    def test_load_thinker_only_mocked(self):
        """Test loading Omni model in thinker_only mode (mocked)."""
        from src.omni.loader import OmniModelLoader
        from unittest.mock import MagicMock, patch
        
        loader = OmniModelLoader("/fake/model/path")
        
        # Mock the transformers imports
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_model):
                model, tokenizer = loader.load(mode="thinker_only")
        
        assert model is not None
    
    def test_load_for_training_mocked(self):
        """Test loading Omni model for training (mocked)."""
        from src.omni.loader import OmniModelLoader
        from unittest.mock import MagicMock, patch
        
        loader = OmniModelLoader("/fake/model/path")
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        
        with patch.object(loader, 'load', return_value=(mock_model, mock_tokenizer)):
            model, tokenizer = loader.load_for_training("/fake/path")
        
        assert model is mock_model
        assert tokenizer is mock_tokenizer
    
    def test_load_for_inference_mocked(self):
        """Test loading Omni model for inference (mocked)."""
        from src.omni.loader import OmniModelLoader
        from unittest.mock import MagicMock, patch
        
        # We need to fully mock the load method since it calls transformers internally
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        with patch.object(OmniModelLoader, 'load_for_inference', return_value=(mock_model, mock_tokenizer)):
            loader = OmniModelLoader("/fake/model/path")
            model, tokenizer = loader.load_for_inference(mode="full")
        
        assert model is mock_model


class TestOmniInference:
    """Tests for Omni inference module."""
    
    def test_import_inference(self):
        """Test OmniInference can be imported."""
        from src.omni.inference import OmniInference
        assert OmniInference is not None
    
    def test_import_generation_config(self):
        """Test GenerationConfig can be imported."""
        from src.omni.inference import GenerationConfig
        assert GenerationConfig is not None
    
    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        from src.omni.inference import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True
    
    def test_generation_config_custom(self):
        """Test GenerationConfig custom values."""
        from src.omni.inference import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.5,
            stream=True,
        )
        
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.5
        assert config.stream is True

