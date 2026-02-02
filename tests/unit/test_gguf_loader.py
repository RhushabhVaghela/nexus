"""
Unit tests for GGUF loader
10+ tests covering GGUfLoader, GGUFConverter, and GGUFConfig
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from nexus.models.gguf import GGUfLoader, GGUFConfig, GGUFConverter


class TestGGUFConfig:
    """Tests for GGUFConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GGUFConfig(model_path="/path/to/model.gguf")
        assert config.model_path == "/path/to/model.gguf"
        assert config.n_ctx == 8192
        assert config.n_batch == 512
        assert config.n_gpu_layers == -1
        assert config.n_threads == -1
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GGUFConfig(
            model_path="/path/model.gguf",
            n_ctx=4096,
            n_gpu_layers=35,
            temperature=0.5,
            chat_format="chatml"
        )
        assert config.n_ctx == 4096
        assert config.n_gpu_layers == 35
        assert config.temperature == 0.5
        assert config.chat_format == "chatml"


class TestGGUfLoader:
    """Tests for GGUfLoader class."""
    
    @pytest.fixture
    def mock_llama(self):
        """Create a mock llama module."""
        with patch.dict('sys.modules', {'llama_cpp': MagicMock()}):
            mock_llama = MagicMock()
            mock_llama.Llama = MagicMock
            yield mock_llama
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        config = GGUFConfig(model_path="test.gguf")
        loader = GGUfLoader(config)
        assert loader.config == config
        assert loader.model is None
    
    def test_ensure_llama_cpp_installed(self):
        """Test check for llama-cpp-python."""
        config = GGUFConfig(model_path="test.gguf")
        loader = GGUfLoader(config)
        
        with patch.dict('sys.modules', {'llama_cpp': MagicMock()}):
            loader._ensure_llama_cpp()
            assert loader._llama_module is not None
    
    def test_ensure_llama_cpp_missing(self):
        """Test error when llama-cpp-python is missing."""
        config = GGUFConfig(model_path="test.gguf")
        loader = GGUfLoader(config)
        
        with patch.dict('sys.modules', {}, clear=True):
            with pytest.raises(ImportError):
                loader._ensure_llama_cpp()
    
    def test_get_context_size(self):
        """Test getting context size."""
        config = GGUFConfig(model_path="test.gguf", n_ctx=4096)
        loader = GGUfLoader(config)
        assert loader.get_context_size() == 4096
    
    def test_tokenize(self):
        """Test tokenization."""
        config = GGUFConfig(model_path="test.gguf")
        loader = GGUfLoader(config)
        
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3, 4]
        loader.model = mock_model
        
        tokens = loader.tokenize("Hello world")
        assert tokens == [1, 2, 3, 4]
    
    def test_detokenize(self):
        """Test detokenization."""
        config = GGUFConfig(model_path="test.gguf")
        loader = GGUfLoader(config)
        
        mock_model = MagicMock()
        mock_model.detokenize.return_value = b"Hello world"
        loader.model = mock_model
        
        text = loader.detokenize([1, 2, 3])
        assert text == "Hello world"
    
    def test_list_gguf_files(self, tmp_path):
        """Test listing GGUF files."""
        # Create test files
        (tmp_path / "model1.gguf").touch()
        (tmp_path / "model2.gguf").touch()
        (tmp_path / "not_gguf.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "model3.gguf").touch()
        
        files = GGUfLoader.list_gguf_files(str(tmp_path))
        assert len(files) == 3
        assert any("model1.gguf" in f for f in files)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        config = GGUFConfig(model_path="test.gguf")
        
        with patch.object(GGUfLoader, 'load') as mock_load, \
             patch.object(GGUfLoader, 'unload') as mock_unload:
            
            with GGUfLoader(config) as loader:
                mock_load.assert_called_once()
            
            mock_unload.assert_called_once()


class TestGGUFConverter:
    """Tests for GGUFConverter class."""
    
    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = GGUFConverter()
        assert converter._gguf_module is None
    
    def test_estimate_gguf_size(self):
        """Test GGUF size estimation."""
        converter = GGUFConverter()
        
        # Test Q4_K_M estimation
        size_bytes, human = converter.estimate_gguf_size(
            "test/model",
            quantization="Q4_K_M"
        )
        assert size_bytes > 0
        assert "GB" in human
    
    def test_create_quantization_config_q4(self):
        """Test Q4 quantization recommendation."""
        converter = GGUFConverter()
        quant = converter.create_quantization_config(
            target_size_gb=4.0,
            model_params_b=7.0
        )
        assert quant in ["Q4_K_M", "Q4_0", "Q3_K_M"]
    
    def test_create_quantization_config_q8(self):
        """Test Q8 quantization recommendation."""
        converter = GGUFConverter()
        quant = converter.create_quantization_config(
            target_size_gb=8.0,
            model_params_b=7.0
        )
        assert quant in ["Q6_K", "Q8_0"]
    
    def test_validate_gguf_not_found(self):
        """Test validation of non-existent file."""
        converter = GGUFConverter()
        report = converter.validate_gguf("/nonexistent/model.gguf")
        assert report["valid"] is False
        assert "not found" in report["error"].lower()
    
    def test_map_tensor_name(self):
        """Test tensor name mapping."""
        converter = GGUFConverter()
        
        # Test common mappings
        assert "embed" in converter._map_tensor_name("token_embd")
        assert "layers" in converter._map_tensor_name("blk.0")
        assert "layernorm" in converter._map_tensor_name("attn_norm")


class TestGGUFBatchLoader:
    """Tests for GGUFBatchLoader class."""
    
    def test_batch_loader_init(self):
        """Test batch loader initialization."""
        from nexus.models.gguf.gguf_loader import GGUFBatchLoader
        loader = GGUFBatchLoader()
        assert len(loader.models) == 0
    
    def test_load_and_get_model(self):
        """Test loading and retrieving models."""
        from nexus.models.gguf.gguf_loader import GGUFBatchLoader
        
        loader = GGUFBatchLoader()
        mock_model = Mock()
        
        with patch.object(GGUfLoader, 'load', return_value=mock_model):
            config = GGUFConfig(model_path="test.gguf")
            loader.load_model("test_model", config)
            
            retrieved = loader.get_model("test_model")
            assert retrieved is not None
    
    def test_unload_model(self):
        """Test unloading a specific model."""
        from nexus.models.gguf.gguf_loader import GGUFBatchLoader
        
        loader = GGUFBatchLoader()
        mock_model = Mock()
        
        with patch.object(GGUfLoader, 'load', return_value=mock_model):
            config = GGUFConfig(model_path="test.gguf")
            loader.load_model("test_model", config)
            loader.unload_model("test_model")
            
            assert loader.get_model("test_model") is None
    
    def test_unload_all(self):
        """Test unloading all models."""
        from nexus.models.gguf.gguf_loader import GGUFBatchLoader
        
        loader = GGUFBatchLoader()
        mock_model = Mock()
        
        with patch.object(GGUfLoader, 'load', return_value=mock_model):
            config = GGUFConfig(model_path="test.gguf")
            loader.load_model("model1", config)
            loader.load_model("model2", config)
            
            loader.unload_all()
            assert len(loader.models) == 0


class TestGGUfIntegration:
    """Integration-style tests for GGUF functionality."""
    
    def test_pytorch_to_gguf_interface(self):
        """Test PyTorch to GGUF conversion interface."""
        converter = GGUFConverter()
        
        with patch('pathlib.Path.mkdir'):
            output_path = converter.pytorch_to_gguf(
                "test/model",
                "/output/model.gguf",
                quantization="Q4_K_M"
            )
            assert "model.gguf" in output_path
    
    def test_quantization_recommendations(self):
        """Test different quantization recommendations."""
        converter = GGUFConverter()
        
        test_cases = [
            (4.0, 7.0, ["Q4", "Q3"]),  # 4GB target, 7B model
            (8.0, 7.0, ["Q6", "Q8"]),  # 8GB target, 7B model
            (14.0, 7.0, ["F16"]),      # 14GB target, 7B model
        ]
        
        for target_size, params, expected_prefixes in test_cases:
            quant = converter.create_quantization_config(target_size, params)
            assert any(prefix in quant for prefix in expected_prefixes), \
                f"Expected {expected_prefixes} in {quant}"
