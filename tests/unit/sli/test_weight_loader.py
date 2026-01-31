"""
Test suite for the Universal Weight Loader module.

This module tests the UniversalWeightLoader class which handles loading weights
from multiple formats (safetensors, bin, pt) with shard management and caching.

Total test cases: ~30
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, mock_open

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.nexus_final.sli.weight_loader import UniversalWeightLoader
from src.nexus_final.sli.exceptions import FormatDetectionError, WeightMapError, WeightLoadingError
from src.nexus_final.sli.architecture_registry import LlamaFamilyHandler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return str(tmp_path / "cache")


@pytest.fixture
def mock_model_id():
    """Return a mock model ID."""
    return "test-org/test-model"


@pytest.fixture
def weight_loader(temp_cache_dir, mock_model_id):
    """Create a UniversalWeightLoader with mocked format detection."""
    with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
        with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
            loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
            return loader


@pytest.fixture
def mock_llama_family():
    """Create a mock Llama family handler."""
    return LlamaFamilyHandler()


@pytest.fixture
def mock_weight_map():
    """Create a mock weight map for testing."""
    return {
        "model.embed_tokens.weight": "model.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.v_proj.weight": "model.safetensors",
        "model.layers.0.self_attn.o_proj.weight": "model.safetensors",
        "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.down_proj.weight": "model.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model.safetensors",
    }


# =============================================================================
# Test Initialization
# =============================================================================

class TestWeightLoaderInitialization:
    """Test UniversalWeightLoader initialization."""
    
    def test_loader_initializes_with_cache_dir(self, temp_cache_dir, mock_model_id):
        """Test loader initializes with cache directory."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                assert loader.cache_dir == Path(temp_cache_dir)
                assert loader.model_id == mock_model_id
    
    def test_loader_creates_cache_dir(self, tmp_path, mock_model_id):
        """Test loader creates cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(str(cache_dir), mock_model_id)
                assert cache_dir.exists()
    
    def test_loader_initializes_empty_shard_cache(self, temp_cache_dir, mock_model_id):
        """Test loader initializes with empty shard cache."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                assert isinstance(loader.loaded_shards, dict)
                assert len(loader.loaded_shards) == 0
    
    def test_loader_stores_format(self, temp_cache_dir, mock_model_id):
        """Test loader stores detected format."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                assert loader.format == '.safetensors'


# =============================================================================
# Test Format Detection
# =============================================================================

class TestFormatDetection:
    """Test weight format auto-detection."""
    
    @patch('requests.head')
    def test_detect_safetensors_format(self, mock_head, temp_cache_dir, mock_model_id):
        """Test detection of safetensors format."""
        mock_head.return_value = MagicMock(status_code=200)
        
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = Path(temp_cache_dir)
        loader.download_fn = loader._default_download
        
        format_ext = loader._detect_format()
        assert format_ext == '.safetensors'
    
    @patch('requests.head')
    def test_detect_bin_format_fallback(self, mock_head, temp_cache_dir, mock_model_id):
        """Test fallback to bin format if safetensors not found."""
        # First calls return 404, last returns 200
        def side_effect(*args, **kwargs):
            url = args[0]
            if 'safetensors' in url:
                return MagicMock(status_code=404)
            elif 'pytorch_model.bin' in url:
                return MagicMock(status_code=200)
            return MagicMock(status_code=404)
        
        mock_head.side_effect = side_effect
        
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = Path(temp_cache_dir)
        loader.download_fn = loader._default_download
        
        format_ext = loader._detect_format()
        assert format_ext == '.bin'
    
    @patch('requests.head')
    def test_detect_format_raises_error(self, mock_head, temp_cache_dir, mock_model_id):
        """Test that unsupported format raises FormatDetectionError."""
        mock_head.return_value = MagicMock(status_code=404)
        
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = Path(temp_cache_dir)
        loader.download_fn = loader._default_download
        
        with pytest.raises(FormatDetectionError) as exc_info:
            loader._detect_format()
        
        assert mock_model_id in str(exc_info.value)


# =============================================================================
# Test Weight Map Loading
# =============================================================================

class TestWeightMapLoading:
    """Test weight map loading functionality."""
    
    def test_load_weight_map_from_index_file(self, temp_cache_dir, mock_model_id):
        """Test loading weight map from index file."""
        # Create a mock index file
        cache_path = Path(temp_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors",
            }
        }
        
        index_file = cache_path / "model.safetensors.index.json"
        with open(index_file, 'w') as f:
            json.dump(index_data, f)
        
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = cache_path
        loader.format = '.safetensors'
        loader.download_fn = loader._default_download
        
        weight_map = loader._load_weight_map()
        
        assert weight_map["layer1.weight"] == "model-00001-of-00002.safetensors"
        assert weight_map["layer2.weight"] == "model-00002-of-00002.safetensors"
    
    def test_load_weight_map_single_file_model(self, temp_cache_dir, mock_model_id):
        """Test weight map for single-file model (no index)."""
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = Path(temp_cache_dir)
        loader.format = '.safetensors'
        loader.download_fn = loader._default_download
        
        weight_map = loader._load_weight_map()
        
        # Should return default single-file mapping
        assert "__all__" in weight_map
        assert weight_map["__all__"] == "model.safetensors"
    
    def test_load_weight_map_invalid_json_raises_error(self, temp_cache_dir, mock_model_id):
        """Test that invalid JSON raises WeightMapError."""
        cache_path = Path(temp_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create invalid JSON file
        index_file = cache_path / "model.safetensors.index.json"
        with open(index_file, 'w') as f:
            f.write("invalid json {{{")
        
        loader = UniversalWeightLoader.__new__(UniversalWeightLoader)
        loader.model_id = mock_model_id
        loader.cache_dir = cache_path
        loader.format = '.safetensors'
        loader.download_fn = loader._default_download
        
        with pytest.raises(WeightMapError):
            loader._load_weight_map()


# =============================================================================
# Test Shard Loading
# =============================================================================

class TestShardLoading:
    """Test shard loading and caching."""
    
    @patch('safetensors.torch.load_file')
    def test_load_safetensors_shard(self, mock_load_file, temp_cache_dir, mock_model_id):
        """Test loading a safetensors shard."""
        mock_weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(20, 20),
        }
        mock_load_file.return_value = mock_weights
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Create the shard file
                shard_path = Path(temp_cache_dir) / "model.safetensors"
                shard_path.touch()
                
                weights = loader._load_shard("model.safetensors")
                
                assert "layer1.weight" in weights
                assert "layer2.weight" in weights
    
    @patch('torch.load')
    def test_load_pytorch_bin_shard(self, mock_torch_load, temp_cache_dir, mock_model_id):
        """Test loading a PyTorch bin shard."""
        mock_weights = {
            "layer1.weight": torch.randn(10, 10),
        }
        mock_torch_load.return_value = mock_weights
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.bin'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Create the shard file
                shard_path = Path(temp_cache_dir) / "pytorch_model.bin"
                shard_path.touch()
                
                weights = loader._load_shard("pytorch_model.bin")
                
                assert "layer1.weight" in weights
    
    def test_shard_caching(self, temp_cache_dir, mock_model_id):
        """Test that loaded shards are cached."""
        mock_weights = {"layer1.weight": torch.randn(10, 10)}
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Manually add to cache
                loader.loaded_shards["model.safetensors"] = mock_weights
                
                # Should return cached version without loading
                weights = loader._load_shard("model.safetensors")
                assert weights is mock_weights


# =============================================================================
# Test Layer Weight Loading
# =============================================================================

class TestLayerWeightLoading:
    """Test loading weights for specific layers."""
    
    def test_load_layer_weights(self, temp_cache_dir, mock_model_id, mock_weight_map):
        """Test loading weights for a specific layer."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=mock_weight_map):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Mock the shard loading
                mock_shard_weights = {
                    "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
                    "model.layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
                }
                loader.loaded_shards["model.safetensors"] = mock_shard_weights
                
                family = LlamaFamilyHandler()
                weights = loader.load_layer_weights(0, family)
                
                assert len(weights) > 0
                assert any("self_attn" in k for k in weights.keys())
    
    def test_load_layer_weights_empty_for_unknown_layer(self, temp_cache_dir, mock_model_id, mock_weight_map):
        """Test loading weights for unknown layer returns empty dict."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=mock_weight_map):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                family = LlamaFamilyHandler()
                weights = loader.load_layer_weights(999, family)
                
                assert weights == {}


# =============================================================================
# Test Embedding Weight Loading
# =============================================================================

class TestEmbeddingWeightLoading:
    """Test loading embedding weights."""
    
    def test_load_embedding_weights(self, temp_cache_dir, mock_model_id, mock_weight_map):
        """Test loading embedding weights."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=mock_weight_map):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Mock the shard loading
                mock_embedding = torch.randn(32000, 4096)
                mock_shard_weights = {
                    "model.embed_tokens.weight": mock_embedding,
                }
                loader.loaded_shards["model.safetensors"] = mock_shard_weights
                
                family = LlamaFamilyHandler()
                embedding = loader.load_embedding_weights(family)
                
                assert embedding is mock_embedding
    
    def test_load_embedding_weights_with_variations(self, temp_cache_dir, mock_model_id):
        """Test loading embedding with alternative weight names."""
        weight_map_with_variations = {
            "transformer.wte.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
        }
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=weight_map_with_variations):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                mock_embedding = torch.randn(50257, 768)
                mock_shard_weights = {
                    "transformer.wte.weight": mock_embedding,
                }
                loader.loaded_shards["model.safetensors"] = mock_shard_weights
                
                family = LlamaFamilyHandler()
                embedding = loader.load_embedding_weights(family)
                
                assert embedding is mock_embedding
    
    def test_load_embedding_weights_not_found_raises_error(self, temp_cache_dir, mock_model_id):
        """Test that missing embedding raises WeightLoadingError."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                family = LlamaFamilyHandler()
                
                with pytest.raises(WeightLoadingError) as exc_info:
                    loader.load_embedding_weights(family)
                
                assert "Embedding weight not found" in str(exc_info.value)


# =============================================================================
# Test Cache Management
# =============================================================================

class TestCacheManagement:
    """Test cache clearing and management."""
    
    def test_clear_all_shards(self, temp_cache_dir, mock_model_id):
        """Test clearing all loaded shards."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Add mock shards to cache
                loader.loaded_shards["shard1.safetensors"] = {"layer1": torch.randn(10, 10)}
                loader.loaded_shards["shard2.safetensors"] = {"layer2": torch.randn(20, 20)}
                
                loader.clear_shards()
                
                assert len(loader.loaded_shards) == 0
    
    def test_clear_specific_shards(self, temp_cache_dir, mock_model_id):
        """Test clearing specific shards."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Add mock shards to cache
                loader.loaded_shards["shard1.safetensors"] = {"layer1": torch.randn(10, 10)}
                loader.loaded_shards["shard2.safetensors"] = {"layer2": torch.randn(20, 20)}
                
                loader.clear_shards(shard_names=["shard1.safetensors"])
                
                assert "shard1.safetensors" not in loader.loaded_shards
                assert "shard2.safetensors" in loader.loaded_shards


# =============================================================================
# Test Weight Info
# =============================================================================

class TestWeightInfo:
    """Test weight information retrieval."""
    
    def test_get_weight_info(self, temp_cache_dir, mock_model_id, mock_weight_map):
        """Test getting weight information."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=mock_weight_map):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                info = loader.get_weight_info()
                
                assert info["format"] == ".safetensors"
                assert info["num_weights"] == len(mock_weight_map)
                assert "loaded_shards" in info
    
    def test_get_weight_info_no_shards_loaded(self, temp_cache_dir, mock_model_id, mock_weight_map):
        """Test weight info when no shards are loaded."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value=mock_weight_map):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                info = loader.get_weight_info()
                
                assert len(info["loaded_shards"]) == 0


# =============================================================================
# Test Download Function
# =============================================================================

class TestDownloadFunction:
    """Test the default download functionality."""
    
    @patch('requests.get')
    def test_download_shard(self, mock_get, temp_cache_dir, mock_model_id):
        """Test downloading a shard file."""
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'data chunk'] * 10
        mock_get.return_value = mock_response
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                path = loader._default_download("model.safetensors")
                
                assert Path(path).exists()
                assert "model.safetensors" in path
    
    def test_download_returns_cached_file(self, temp_cache_dir, mock_model_id):
        """Test that cached files are returned without re-downloading."""
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                # Create cached file
                shard_path = Path(temp_cache_dir) / "model.safetensors"
                shard_path.touch()
                
                with patch('requests.get') as mock_get:
                    path = loader._default_download("model.safetensors")
                    mock_get.assert_not_called()
                    assert str(shard_path) == path


# =============================================================================
# Test URL Checking
# =============================================================================

class TestURLChecking:
    """Test URL existence checking."""
    
    @patch('requests.head')
    def test_check_url_exists_true(self, mock_head, temp_cache_dir, mock_model_id):
        """Test URL exists check returns True."""
        mock_head.return_value = MagicMock(status_code=200)
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                exists = loader._check_url_exists("https://example.com/model.bin")
                assert exists is True
    
    @patch('requests.head')
    def test_check_url_exists_false(self, mock_head, temp_cache_dir, mock_model_id):
        """Test URL exists check returns False."""
        mock_head.return_value = MagicMock(status_code=404)
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                exists = loader._check_url_exists("https://example.com/missing.bin")
                assert exists is False
    
    @patch('requests.head')
    def test_check_url_exists_exception_returns_false(self, mock_head, temp_cache_dir, mock_model_id):
        """Test URL check returns False on exception."""
        mock_head.side_effect = Exception("Connection error")
        
        with patch.object(UniversalWeightLoader, '_detect_format', return_value='.safetensors'):
            with patch.object(UniversalWeightLoader, '_load_weight_map', return_value={}):
                loader = UniversalWeightLoader(temp_cache_dir, mock_model_id)
                
                exists = loader._check_url_exists("https://example.com/model.bin")
                assert exists is False
