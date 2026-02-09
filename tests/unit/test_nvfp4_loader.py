"""
Comprehensive unit tests for NVFP4 Streaming Loader.

Tests cover:
- NVFP4Tensor class (QuantizedTensor)
- Block-wise quantization/dequantization
- Mixed precision loading
- Hardware acceleration paths (with fallback)
- Error handling for unsupported hardware
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Import the module under test
from nexus.models.sli.nvfp4_loader import (
    NVFP4StreamingLoader,
    NVFP4Quantizer,
    NVFP4Config,
    NVFP4Mode,
    QuantizedTensor,
    NVFP4QuantizationError,
    get_nvfp4_config,
    quantize_to_nvfp4,
    dequantize_from_nvfp4,
    NVFP4_AVAILABLE,
)
from nexus.models.sli.exceptions import SLIError, WeightLoadingError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(128, 128)


@pytest.fixture
def sample_2d_tensor():
    """Create a 2D weight-like tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(256, 512)


@pytest.fixture
def sample_linear_layer():
    """Create a sample linear layer for testing."""
    torch.manual_seed(42)
    return nn.Linear(512, 256)


@pytest.fixture
def nvfp4_config_software():
    """Create a software mode NVFP4 config."""
    # Use float32 for CPU compatibility
    return NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16, compute_dtype=torch.float32)


@pytest.fixture
def nvfp4_config_mixed():
    """Create a mixed mode NVFP4 config."""
    return NVFP4Config(mode=NVFP4Mode.MIXED, block_size=16)


@pytest.fixture
def nvfp4_config_hardware():
    """Create a hardware mode NVFP4 config."""
    return NVFP4Config(mode=NVFP4Mode.HARDWARE, block_size=16)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "nvfp4_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def nvfp4_loader(temp_cache_dir):
    """Create an NVFP4 streaming loader with temp cache."""
    config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
    return NVFP4StreamingLoader(config=config, cache_dir=temp_cache_dir, device="cpu")


# ============================================================================
# Test QuantizedTensor Data Class
# ============================================================================

class TestQuantizedTensor:
    """Test suite for QuantizedTensor data class."""

    def test_quantized_tensor_creation(self, sample_tensor):
        """Test basic QuantizedTensor creation."""
        quantized_data = sample_tensor.to(torch.float8_e4m3fn)
        scale = torch.tensor([1.0])
        
        qt = QuantizedTensor(
            data=quantized_data,
            scale=scale,
            orig_shape=sample_tensor.shape,
            block_size=16,
            dtype=torch.float8_e4m3fn
        )
        
        assert qt.data is quantized_data
        assert torch.equal(qt.scale, scale)
        assert qt.orig_shape == sample_tensor.shape
        assert qt.block_size == 16
        assert qt.dtype == torch.float8_e4m3fn

    def test_quantized_tensor_with_different_shapes(self):
        """Test QuantizedTensor with various tensor shapes."""
        shapes = [
            (16, 16),
            (32, 64),
            (128, 256, 64),
            (1,),
            (1024,),
        ]
        
        for shape in shapes:
            data = torch.randn(shape).to(torch.float8_e4m3fn)
            scale = torch.tensor([1.0])
            
            qt = QuantizedTensor(
                data=data,
                scale=scale,
                orig_shape=shape,
                block_size=16,
                dtype=torch.float8_e4m3fn
            )
            
            assert qt.orig_shape == shape

    def test_quantized_tensor_empty_shape(self):
        """Test QuantizedTensor with empty shape (scalar)."""
        data = torch.tensor(1.0).to(torch.float8_e4m3fn)
        scale = torch.tensor([1.0])
        
        qt = QuantizedTensor(
            data=data,
            scale=scale,
            orig_shape=(),
            block_size=16,
            dtype=torch.float8_e4m3fn
        )
        
        assert qt.orig_shape == ()


# ============================================================================
# Test NVFP4Config
# ============================================================================

class TestNVFP4Config:
    """Test suite for NVFP4Config dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NVFP4Config()
        
        assert config.mode == NVFP4Mode.MIXED
        assert config.block_size == 16
        assert config.compute_dtype == torch.bfloat16
        assert config.attention_dtype == torch.bfloat16
        assert config.enable_scaling is True
        assert config.stochastic_rounding is True
        assert config.amax_history_len == 1024
        assert config.mixed_precision_threshold == 4096

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = NVFP4Config(
            mode=NVFP4Mode.SOFTWARE,
            block_size=32,
            compute_dtype=torch.float16,
            attention_dtype=torch.float32,
            enable_scaling=False,
            stochastic_rounding=False,
            amax_history_len=512,
            mixed_precision_threshold=2048
        )
        
        assert config.mode == NVFP4Mode.SOFTWARE
        assert config.block_size == 32
        assert config.compute_dtype == torch.float16
        assert config.attention_dtype == torch.float32
        assert config.enable_scaling is False
        assert config.stochastic_rounding is False
        assert config.amax_history_len == 512
        assert config.mixed_precision_threshold == 2048

    def test_config_invalid_block_size(self):
        """Test that invalid block size raises ValueError."""
        with pytest.raises(ValueError, match="block_size must be multiple of 16"):
            NVFP4Config(block_size=15)
        
        with pytest.raises(ValueError, match="block_size must be multiple of 16"):
            NVFP4Config(block_size=17)
        
        with pytest.raises(ValueError, match="block_size must be multiple of 16"):
            NVFP4Config(block_size=20)

    def test_config_hardware_fallback(self):
        """Test that hardware mode falls back to software when TE unavailable."""
        with patch('src.nexus.models.sli.nvfp4_loader.NVFP4_AVAILABLE', False):
            with patch('src.nexus.models.sli.nvfp4_loader.logger') as mock_logger:
                config = NVFP4Config(mode=NVFP4Mode.HARDWARE)
                
                # Should fall back to software mode
                assert config.mode == NVFP4Mode.SOFTWARE
                # Warning should be logged
                mock_logger.warning.assert_called_once()

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=32)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['mode'] == 'software'
        assert config_dict['block_size'] == 32
        assert config_dict['compute_dtype'] == 'torch.bfloat16'
        assert config_dict['enable_scaling'] is True
        assert config_dict['stochastic_rounding'] is True

    def test_config_from_dict(self):
        """Test configuration deserialization from dict."""
        data = {
            'mode': 'mixed',
            'block_size': 32,
            'compute_dtype': 'torch.float16',
            'attention_dtype': 'torch.float32',
            'ffn_dtype': 'torch.float8_e4m3fn',
            'enable_scaling': False,
            'stochastic_rounding': False,
            'amax_history_len': 512,
            'mixed_precision_threshold': 2048
        }
        
        config = NVFP4Config.from_dict(data)
        
        assert config.mode == NVFP4Mode.MIXED
        assert config.block_size == 32
        assert config.compute_dtype == torch.float16
        assert config.attention_dtype == torch.float32
        assert config.enable_scaling is False
        assert config.amax_history_len == 512

    def test_config_from_dict_defaults(self):
        """Test configuration from dict with missing values uses defaults."""
        data = {'mode': 'software'}
        
        config = NVFP4Config.from_dict(data)
        
        assert config.mode == NVFP4Mode.SOFTWARE
        assert config.block_size == 16  # Default
        assert config.enable_scaling is True  # Default


# ============================================================================
# Test NVFP4Quantizer
# ============================================================================

class TestNVFP4Quantizer:
    """Test suite for NVFP4Quantizer class."""

    def test_quantizer_initialization(self, nvfp4_config_software):
        """Test quantizer initialization."""
        quantizer = NVFP4Quantizer(nvfp4_config_software)
        
        assert quantizer.config == nvfp4_config_software
        assert isinstance(quantizer._amax_history, dict)
        assert isinstance(quantizer._lock, type(threading.RLock()))

    def test_quantizer_default_config(self):
        """Test quantizer with default config."""
        quantizer = NVFP4Quantizer()
        
        assert quantizer.config is not None
        assert quantizer.config.mode == NVFP4Mode.MIXED

    def test_quantize_tensor_2d(self, sample_2d_tensor):
        """Test quantizing a 2D tensor."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(sample_2d_tensor, name="test_weight")
        
        assert isinstance(quantized, QuantizedTensor)
        assert quantized.orig_shape == sample_2d_tensor.shape
        assert quantized.block_size == 16
        assert quantized.dtype == torch.float8_e4m3fn

    def test_quantize_tensor_1d(self):
        """Test quantizing a 1D tensor."""
        tensor = torch.randn(256)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="test_bias")
        
        assert isinstance(quantized, QuantizedTensor)
        assert quantized.orig_shape == (256,)

    def test_quantize_tensor_multidim(self):
        """Test quantizing a multi-dimensional tensor."""
        tensor = torch.randn(4, 8, 16, 32)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="test_multidim")
        
        assert isinstance(quantized, QuantizedTensor)
        assert quantized.orig_shape == (4, 8, 16, 32)

    def test_quantize_tensor_stochastic_rounding(self):
        """Test stochastic rounding during quantization."""
        tensor = torch.randn(64, 64, requires_grad=True)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, stochastic_rounding=True)
        quantizer = NVFP4Quantizer(config)
        
        # Quantize twice and verify different results due to stochastic rounding
        quantized1 = quantizer.quantize_tensor(tensor, name="test_stochastic")
        quantized2 = quantizer.quantize_tensor(tensor, name="test_stochastic")
        
        # Data should be different due to stochastic rounding
        assert not torch.equal(quantized1.data, quantized2.data)

    def test_quantize_tensor_no_stochastic_rounding(self):
        """Test deterministic quantization without stochastic rounding."""
        tensor = torch.randn(64, 64)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, stochastic_rounding=False)
        quantizer = NVFP4Quantizer(config)
        
        # Quantize twice and verify same results
        quantized1 = quantizer.quantize_tensor(tensor, name="test_deterministic")
        quantized2 = quantizer.quantize_tensor(tensor, name="test_deterministic")
        
        # Data should be same without stochastic rounding
        assert torch.equal(quantized1.data, quantized2.data)

    def test_quantize_tensor_override_stochastic(self):
        """Test overriding stochastic rounding per-call."""
        tensor = torch.randn(64, 64, requires_grad=True)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, stochastic_rounding=True)
        quantizer = NVFP4Quantizer(config)
        
        # Override with False
        quantized = quantizer.quantize_tensor(tensor, name="test_override", use_stochastic_rounding=False)
        
        assert isinstance(quantized, QuantizedTensor)

    def test_quantize_tensor_with_padding(self):
        """Test quantizing tensor that requires padding."""
        # Shape not divisible by block_size
        tensor = torch.randn(100, 100)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="test_padded")
        
        assert quantized.orig_shape == (100, 100)
        # Internal shape should be padded
        assert quantized.data.shape[0] >= 100 * 100 / 16

    def test_dequantize_tensor(self, sample_2d_tensor):
        """Test dequantizing a quantized tensor."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(sample_2d_tensor, name="test_dequant")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == sample_2d_tensor.shape
        assert dequantized.dtype == config.compute_dtype

    def test_dequantize_preserves_shape(self):
        """Test that dequantization preserves original shape."""
        shapes = [(128, 256), (64, 64), (256, 128), (512, 512)]
        
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        for shape in shapes:
            tensor = torch.randn(shape)
            quantized = quantizer.quantize_tensor(tensor, name=f"test_{shape}")
            dequantized = quantizer.dequantize_tensor(quantized)
            
            assert dequantized.shape == shape, f"Shape mismatch for {shape}"

    def test_amax_history_tracking(self, sample_2d_tensor):
        """Test that amax history is properly tracked."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, amax_history_len=10)
        quantizer = NVFP4Quantizer(config)
        
        # Quantize multiple times
        for i in range(15):
            tensor = torch.randn_like(sample_2d_tensor)
            quantizer.quantize_tensor(tensor, name="tracked_weight")
        
        # History should be limited to amax_history_len
        assert len(quantizer._amax_history["tracked_weight"]) <= 10

    def test_quantize_dequantize_roundtrip(self, sample_2d_tensor):
        """Test quantization-dequantization roundtrip."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(sample_2d_tensor, name="roundtrip")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        # Should be close but not exact due to quantization error
        error = torch.abs(sample_2d_tensor - dequantized).mean()
        assert error < 0.1  # Reasonable error bound

    def test_quantize_hardware_unavailable(self):
        """Test hardware quantization falls back when TE unavailable."""
        with patch('src.nexus.models.sli.nvfp4_loader.NVFP4_AVAILABLE', False):
            config = NVFP4Config(mode=NVFP4Mode.HARDWARE)
            quantizer = NVFP4Quantizer(config)
            
            tensor = torch.randn(64, 64)
            
            # Should use software fallback
            quantized = quantizer.quantize_tensor(tensor, name="test")
            assert isinstance(quantized, QuantizedTensor)

    def test_dequantize_hardware_unavailable(self):
        """Test hardware dequantization falls back when TE unavailable."""
        with patch('src.nexus.models.sli.nvfp4_loader.NVFP4_AVAILABLE', False):
            config = NVFP4Config(mode=NVFP4Mode.HARDWARE)
            quantizer = NVFP4Quantizer(config)
            
            # Create a mock quantized tensor
            qt = QuantizedTensor(
                data=torch.randn(64, 64).to(torch.float8_e4m3fn),
                scale=torch.tensor([1.0]),
                orig_shape=(64, 64),
                block_size=16,
                dtype=torch.float8_e4m3fn
            )
            
            # Should use software fallback
            dequantized = quantizer.dequantize_tensor(qt)
            assert dequantized.shape == (64, 64)

    def test_thread_safety_amax_history(self):
        """Test thread safety of amax history updates."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        errors = []
        
        def quantize_worker():
            try:
                for _ in range(50):
                    tensor = torch.randn(64, 64)
                    quantizer.quantize_tensor(tensor, name="concurrent_weight")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=quantize_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# Test NVFP4StreamingLoader
# ============================================================================

class TestNVFP4StreamingLoader:
    """Test suite for NVFP4StreamingLoader class."""

    def test_loader_initialization(self, temp_cache_dir):
        """Test loader initialization."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=temp_cache_dir, device="cpu")
        
        assert loader.config == config
        assert loader.device == "cpu"
        assert loader.quantizer is not None
        assert loader.cache_dir == Path(temp_cache_dir)

    def test_loader_initialization_no_cache(self):
        """Test loader initialization without cache."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=None, device="cpu")
        
        assert loader.cache_dir is None

    def test_loader_initialization_creates_cache_dir(self, tmp_path):
        """Test that loader creates cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()
        
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=str(cache_dir), device="cpu")
        
        assert cache_dir.exists()

    def test_load_layer_from_weights(self, nvfp4_loader):
        """Test loading a layer from weights."""
        weights = {
            "weight": torch.randn(256, 512),
            "bias": torch.randn(256),
        }
        
        layer = nvfp4_loader.load_layer(
            model_id="test_model",
            layer_idx=0,
            layer_weights=weights
        )
        
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, "weight")
        assert hasattr(layer, "bias")

    def test_load_layer_no_weights_no_cache(self):
        """Test loading layer with no weights and no cache raises error."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=None, device="cpu")
        
        with pytest.raises(WeightLoadingError):
            loader.load_layer(model_id="test_model", layer_idx=0)

    def test_quantize_layer_ffn(self, nvfp4_loader, sample_linear_layer):
        """Test quantizing an FFN layer."""
        quantized = nvfp4_loader.quantize_layer(sample_linear_layer, is_attention=False)
        
        assert isinstance(quantized, nn.Module)
        # Should have quantized weight data
        assert hasattr(quantized, "weight_quantized") or hasattr(quantized, "weight")

    def test_quantize_layer_attention(self, nvfp4_loader, sample_linear_layer):
        """Test quantizing an attention layer."""
        quantized = nvfp4_loader.quantize_layer(sample_linear_layer, is_attention=True)
        
        assert isinstance(quantized, nn.Module)
        # In mixed mode, attention should use attention_dtype

    def test_dequantize_layer(self, nvfp4_loader, sample_linear_layer):
        """Test dequantizing a quantized layer."""
        quantized = nvfp4_loader.quantize_layer(sample_linear_layer, is_attention=False)
        dequantized = nvfp4_loader.dequantize_layer(quantized)
        
        assert isinstance(dequantized, nn.Module)
        assert hasattr(dequantized, "weight")
        assert dequantized.weight.dtype == torch.bfloat16

    def test_dequantize_preserves_parameters(self, nvfp4_loader):
        """Test that dequantization preserves all parameters."""
        layer = nn.Linear(100, 50)
        layer.weight.data = torch.randn(50, 100)
        layer.bias.data = torch.randn(50)
        
        quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        dequantized = nvfp4_loader.dequantize_layer(quantized)
        
        assert dequantized.weight.shape == (50, 100)
        assert dequantized.bias.shape == (50,)

    def test_cache_layer(self, nvfp4_loader, sample_linear_layer):
        """Test caching a layer."""
        result = nvfp4_loader.cache_layer("test_model", 0, sample_linear_layer)
        
        assert result is True
        # Check that cache file was created
        cache_files = list(nvfp4_loader.cache_dir.glob("*.pt"))
        assert len(cache_files) >= 1

    def test_load_from_cache(self, nvfp4_loader, sample_linear_layer):
        """Test loading a layer from cache."""
        # First cache the layer
        nvfp4_loader.cache_layer("test_model", 0, sample_linear_layer)
        
        # Then load it
        layer = nvfp4_loader.load_layer("test_model", 0, source="cache")
        
        assert isinstance(layer, nn.Module)

    def test_cache_layer_no_cache_dir(self):
        """Test caching when no cache directory is configured."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=None, device="cpu")
        
        layer = nn.Linear(100, 100)
        result = loader.cache_layer("test_model", 0, layer)
        
        assert result is False

    def test_clear_cache(self, nvfp4_loader, sample_linear_layer):
        """Test clearing the cache."""
        # Add some layers to cache
        for i in range(3):
            nvfp4_loader.cache_layer("test_model", i, sample_linear_layer)
        
        # Verify cache has files
        cache_files = list(nvfp4_loader.cache_dir.glob("*.pt"))
        assert len(cache_files) >= 1
        
        # Clear cache
        nvfp4_loader.clear_cache()
        
        # Verify cache is empty
        cache_files = list(nvfp4_loader.cache_dir.glob("*.pt"))
        assert len(cache_files) == 0

    def test_clear_cache_no_cache_dir(self):
        """Test clearing cache when no cache directory."""
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        loader = NVFP4StreamingLoader(config=config, cache_dir=None, device="cpu")
        
        # Should not raise an error
        loader.clear_cache()

    def test_get_stats(self, nvfp4_loader, sample_linear_layer):
        """Test getting loader statistics."""
        # Load and quantize some layers
        weights = {"weight": torch.randn(100, 100)}
        layer = nvfp4_loader.load_layer("test_model", 0, layer_weights=weights)
        nvfp4_loader.quantize_layer(layer, is_attention=False)
        
        stats = nvfp4_loader.get_stats()
        
        assert isinstance(stats, dict)
        assert "layers_loaded" in stats
        assert "layers_quantized" in stats
        assert stats["layers_loaded"] >= 1
        assert stats["layers_quantized"] >= 1

    def test_stats_thread_safety(self, nvfp4_loader):
        """Test statistics tracking is thread-safe."""
        errors = []
        
        def worker():
            try:
                for _ in range(20):
                    weights = {"weight": torch.randn(50, 50)}
                    layer = nvfp4_loader.load_layer("test_model", 0, layer_weights=weights)
                    nvfp4_loader.quantize_layer(layer, is_attention=False)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"
        
        stats = nvfp4_loader.get_stats()
        assert stats["layers_loaded"] >= 20 * 3


# ============================================================================
# Test Mixed Precision Loading
# ============================================================================

class TestMixedPrecision:
    """Test suite for mixed precision loading functionality."""

    def test_mixed_precision_attention_vs_ffn(self):
        """Test that attention and FFN layers use different dtypes in mixed mode."""
        config = NVFP4Config(mode=NVFP4Mode.MIXED)
        loader = NVFP4StreamingLoader(config=config, device="cpu")
        
        attention_layer = nn.Linear(512, 512)
        ffn_layer = nn.Linear(512, 2048)
        
        quantized_attention = loader.quantize_layer(attention_layer, is_attention=True)
        quantized_ffn = loader.quantize_layer(ffn_layer, is_attention=False)
        
        # Both should be successfully quantized
        assert isinstance(quantized_attention, nn.Module)
        assert isinstance(quantized_ffn, nn.Module)

    def test_mixed_precision_threshold(self):
        """Test mixed precision threshold configuration."""
        config = NVFP4Config(
            mode=NVFP4Mode.MIXED,
            mixed_precision_threshold=1024
        )
        loader = NVFP4StreamingLoader(config=config, device="cpu")
        
        # Small layer (below threshold)
        small_layer = nn.Linear(512, 512)
        # Large layer (above threshold)
        large_layer = nn.Linear(8192, 8192)
        
        quantized_small = loader.quantize_layer(small_layer, is_attention=False)
        quantized_large = loader.quantize_layer(large_layer, is_attention=False)
        
        assert isinstance(quantized_small, nn.Module)
        assert isinstance(quantized_large, nn.Module)


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling."""

    def test_nvfp4_quantization_error(self):
        """Test NVFP4QuantizationError creation."""
        error = NVFP4QuantizationError("layer_0")
        assert "layer_0" in str(error)
        assert error.layer_name == "layer_0"

    def test_nvfp4_quantization_error_with_message(self):
        """Test NVFP4QuantizationError with custom message."""
        error = NVFP4QuantizationError("layer_0", "Custom error message")
        assert "Custom error message" in str(error)

    def test_nvfp4_quantization_error_inheritance(self):
        """Test NVFP4QuantizationError inherits from SLIError."""
        error = NVFP4QuantizationError("layer_0")
        assert isinstance(error, SLIError)

    def test_weight_loading_error(self):
        """Test WeightLoadingError handling."""
        cause = ValueError("Original error")
        error = WeightLoadingError("weight_name", "shard_name", cause)
        
        assert "weight_name" in str(error)
        assert "shard_name" in str(error)
        assert "Original error" in str(error)


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_get_nvfp4_config(self):
        """Test get_nvfp4_config convenience function."""
        config = get_nvfp4_config(mode="software", block_size=32)
        
        assert config.mode == NVFP4Mode.SOFTWARE
        assert config.block_size == 32

    def test_get_nvfp4_config_default(self):
        """Test get_nvfp4_config with defaults."""
        config = get_nvfp4_config()
        
        assert config.mode == NVFP4Mode.MIXED
        assert config.block_size == 16

    def test_get_nvfp4_config_dtype_mapping(self):
        """Test get_nvfp4_config with different dtypes."""
        config_bfloat16 = get_nvfp4_config(compute_dtype="bfloat16")
        config_float16 = get_nvfp4_config(compute_dtype="float16")
        config_float32 = get_nvfp4_config(compute_dtype="float32")
        
        assert config_bfloat16.compute_dtype == torch.bfloat16
        assert config_float16.compute_dtype == torch.float16
        assert config_float32.compute_dtype == torch.float32

    def test_quantize_to_nvfp4(self, sample_2d_tensor):
        """Test quantize_to_nvfp4 convenience function."""
        quantized = quantize_to_nvfp4(sample_2d_tensor, block_size=16)
        
        assert isinstance(quantized, QuantizedTensor)
        assert quantized.orig_shape == sample_2d_tensor.shape

    def test_dequantize_from_nvfp4(self, sample_2d_tensor):
        """Test dequantize_from_nvfp4 convenience function."""
        quantized = quantize_to_nvfp4(sample_2d_tensor, block_size=16)
        dequantized = dequantize_from_nvfp4(quantized)
        
        assert dequantized.shape == sample_2d_tensor.shape

    def test_roundtrip_convenience_functions(self, sample_2d_tensor):
        """Test roundtrip using convenience functions."""
        quantized = quantize_to_nvfp4(sample_2d_tensor)
        dequantized = dequantize_from_nvfp4(quantized)
        
        # Should be close to original
        error = torch.abs(sample_2d_tensor - dequantized).mean()
        assert error < 0.1


# ============================================================================
# Test Hardware Acceleration
# ============================================================================

class TestHardwareAcceleration:
    """Test suite for hardware acceleration paths."""

    def test_hardware_mode_fallback_warning(self):
        """Test that hardware mode produces fallback warning when TE unavailable."""
        with patch('src.nexus.models.sli.nvfp4_loader.NVFP4_AVAILABLE', False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = NVFP4Config(mode=NVFP4Mode.HARDWARE)
                
                assert config.mode == NVFP4Mode.SOFTWARE
                # Check that warning was issued
                warning_messages = [str(warning.message) for warning in w]
                assert any("Transformer Engine not available" in msg or 
                          "Hardware mode requested" in msg for msg in warning_messages)

    def test_quantize_with_hardware_mode_unavailable(self):
        """Test quantization when hardware mode is unavailable."""
        with patch('src.nexus.models.sli.nvfp4_loader.NVFP4_AVAILABLE', False):
            config = NVFP4Config(mode=NVFP4Mode.HARDWARE)
            quantizer = NVFP4Quantizer(config)
            tensor = torch.randn(64, 64)
            
            # Should work with software fallback
            quantized = quantizer.quantize_tensor(tensor, name="test")
            assert isinstance(quantized, QuantizedTensor)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_quantize_zero_tensor(self):
        """Test quantizing a zero tensor."""
        tensor = torch.zeros(64, 64)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="zero_tensor")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        # Should handle zero values gracefully
        assert dequantized.shape == tensor.shape

    def test_quantize_very_small_tensor(self):
        """Test quantizing a tensor with very small values."""
        tensor = torch.randn(64, 64) * 1e-10
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="small_tensor")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_quantize_very_large_tensor(self):
        """Test quantizing a tensor with very large values."""
        tensor = torch.randn(64, 64) * 1e10
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="large_tensor")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_quantize_nan_values(self):
        """Test quantizing tensor with NaN values."""
        tensor = torch.randn(64, 64)
        tensor[0, 0] = float('nan')
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        # Should handle NaN gracefully
        quantized = quantizer.quantize_tensor(tensor, name="nan_tensor")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_quantize_inf_values(self):
        """Test quantizing tensor with Inf values."""
        tensor = torch.randn(64, 64)
        tensor[0, 0] = float('inf')
        tensor[0, 1] = float('-inf')
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        # Should handle Inf gracefully
        quantized = quantizer.quantize_tensor(tensor, name="inf_tensor")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_empty_layer_quantization(self, nvfp4_loader):
        """Test quantizing an empty layer."""
        layer = nn.Module()
        quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        
        assert isinstance(quantized, nn.Module)

    def test_single_element_tensor(self):
        """Test quantizing a single-element tensor."""
        tensor = torch.tensor([1.0])
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="single")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_exact_block_size_tensor(self):
        """Test quantizing tensor with exact block size dimensions."""
        tensor = torch.randn(16, 16)  # Exact 16x16 block
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE, block_size=16)
        quantizer = NVFP4Quantizer(config)
        
        quantized = quantizer.quantize_tensor(tensor, name="exact_block")
        dequantized = quantizer.dequantize_tensor(quantized)
        
        assert dequantized.shape == tensor.shape

    def test_layer_with_buffers_only(self, nvfp4_loader):
        """Test quantizing layer with only buffers, no parameters."""
        layer = nn.Module()
        layer.register_buffer("running_mean", torch.randn(100))
        layer.register_buffer("running_var", torch.randn(100))
        
        quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        
        assert isinstance(quantized, nn.Module)

    def test_very_large_layer(self, nvfp4_loader):
        """Test quantizing a very large layer."""
        layer = nn.Linear(4096, 11008)  # Typical FFN size
        
        quantized = nvfp4_loader.quantize_layer(layer, is_attention=False)
        dequantized = nvfp4_loader.dequantize_layer(quantized)
        
        assert dequantized.weight.shape == (11008, 4096)


# ============================================================================
# Test Performance Characteristics
# ============================================================================

class TestPerformanceCharacteristics:
    """Test suite for performance-related characteristics."""

    def test_quantization_speed(self):
        """Test that quantization completes in reasonable time."""
        tensor = torch.randn(1024, 1024)
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        
        start = time.time()
        quantized = quantizer.quantize_tensor(tensor, name="speed_test")
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0
        assert isinstance(quantized, QuantizedTensor)

    def test_memory_efficiency(self):
        """Test that quantized tensors use less memory."""
        tensor = torch.randn(1024, 1024, dtype=torch.float32)
        original_bytes = tensor.numel() * tensor.element_size()
        
        config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
        quantizer = NVFP4Quantizer(config)
        quantized = quantizer.quantize_tensor(tensor, name="memory_test")
        
        # FP8 uses 1 byte per element vs 4 bytes for FP32
        quantized_bytes = quantized.data.numel() * quantized.data.element_size()
        
        # Should be significantly smaller
        assert quantized_bytes < original_bytes * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
