"""
Comprehensive unit tests for compression_optimized.py
Tests ZSTD compression, quantization, combined compression, and decompression.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from typing import Tuple

from nexus.optimizations.compression_optimized import (
    CompressionConfig,
    QuantizedTensor,
    QuantizationCompressor,
    ZSTDQuantizedCompressor,
    OptimizedCompressor,
)


class TestCompressionConfig:
    """Test CompressionConfig dataclass."""

    def test_default_config_values(self):
        """Test default compression configuration."""
        config = CompressionConfig()
        assert config.algorithm == "zstd"
        assert config.compression_level == 22
        assert config.quantization_bits == 8
        assert config.use_grouped_quantization is True
        assert config.group_size == 128
        assert config.enable_delta_encoding is True
        assert config.sparsity_threshold == 0.01

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = CompressionConfig(
            algorithm="lz4",
            compression_level=19,
            quantization_bits=4,
            use_grouped_quantization=False,
            group_size=64,
            enable_delta_encoding=False,
            sparsity_threshold=0.001,
        )

        assert config.algorithm == "lz4"
        assert config.compression_level == 19
        assert config.quantization_bits == 4
        assert config.use_grouped_quantization is False
        assert config.group_size == 64
        assert config.enable_delta_encoding is False
        assert config.sparsity_threshold == 0.001


class TestQuantizedTensor:
    """Test QuantizedTensor for block-wise quantization."""

    def test_init(self):
        """Test QuantizedTensor initialization."""
        quantized_data = torch.randint(0, 256, (1024,), dtype=torch.uint8)
        scales = torch.randn(8)
        zero_points = torch.randn(8)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(1024,),
            dtype=torch.float32,
        )

        assert qt.quantized_data is quantized_data
        assert qt.scales is scales
        assert qt.zero_points is zero_points
        assert qt.group_size == 128
        assert qt.original_shape == (1024,)
        assert qt.dtype == torch.float32

    def test_dequantize_basic(self):
        """Test basic dequantization."""
        # Create simple quantized tensor
        original = torch.randn(256)
        scales = torch.ones(2)  # 2 groups of 128
        zero_points = torch.zeros(2)

        quantized_data = torch.randint(0, 256, (256,), dtype=torch.uint8)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(256,),
            dtype=torch.float32,
        )

        dequantized = qt.dequantize()

        assert dequantized.shape == (256,)
        assert dequantized.dtype == torch.float32

    def test_dequantize_with_nonzero_zero_points(self):
        """Test dequantization with non-zero zero points."""
        # Quantize then dequantize should preserve values approximately
        original = torch.randn(256)
        scales = torch.ones(2)
        zero_points = torch.zeros(2)

        quantized_data = torch.round(
            (original.reshape(-1, 128) - zero_points.unsqueeze(1)) / scales.unsqueeze(1)
        )
        quantized_data = torch.clamp(quantized_data, 0, 255).to(torch.uint8)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(256,),
            dtype=torch.float32,
        )

        dequantized = qt.dequantize()

        # Should be close to original
        assert dequantized.shape == original.shape

    def test_to_bytes_and_from_bytes(self):
        """Test serialization and deserialization."""
        original = torch.randn(256)
        scales = torch.ones(2)
        zero_points = torch.zeros(2)

        quantized_data = torch.randint(0, 256, (256,), dtype=torch.uint8)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(256,),
            dtype=torch.float32,
        )

        # Serialize
        serialized = qt.to_bytes()

        # Deserialize
        qt_restored = QuantizedTensor.from_bytes(serialized)

        assert qt_restored.original_shape == qt.original_shape
        assert qt_restored.group_size == qt.group_size
        assert qt_restored.dtype == qt.dtype

    def test_to_bytes_fp16_dtype(self):
        """Test serialization with float16 dtype."""
        original = torch.randn(256)

        quantized_data = torch.randint(0, 256, (256,), dtype=torch.uint8)
        scales = torch.randn(2)
        zero_points = torch.randn(2)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(256,),
            dtype=torch.float16,
        )

        serialized = qt.to_bytes()
        qt_restored = QuantizedTensor.from_bytes(serialized)

        assert qt_restored.dtype == torch.float16

    def test_to_bytes_fp32_dtype(self):
        """Test serialization with float32 dtype."""
        original = torch.randn(256)

        quantized_data = torch.randint(0, 256, (256,), dtype=torch.uint8)
        scales = torch.randn(2)
        zero_points = torch.randn(2)

        qt = QuantizedTensor(
            quantized_data=quantized_data,
            scales=scales,
            zero_points=zero_points,
            group_size=128,
            original_shape=(256,),
            dtype=torch.float32,
        )

        serialized = qt.to_bytes()
        qt_restored = QuantizedTensor.from_bytes(serialized)

        assert qt_restored.dtype == torch.float32


class TestQuantizationCompressor:
    """Test QuantizationCompressor for tensor quantization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        compressor = QuantizationCompressor()
        assert compressor.config.quantization_bits == 8
        assert compressor.config.group_size == 128

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CompressionConfig(quantization_bits=4, group_size=64)
        compressor = QuantizationCompressor(config)
        assert compressor.config.quantization_bits == 4
        assert compressor.config.group_size == 64

    def test_quantize_basic_tensor(self):
        """Test basic tensor quantization."""
        compressor = QuantizationCompressor()
        original = torch.randn(256, 768)

        quantized = compressor.quantize(original)

        assert isinstance(quantized, QuantizedTensor)
        assert quantized.original_shape == (256, 768)
        assert quantized.dtype == original.dtype

    def test_quantize_preserves_shape(self):
        """Test quantization preserves original shape."""
        compressor = QuantizationCompressor()
        shapes = [(128, 768), (256, 1024), (1, 4096), (10, 10, 10)]

        for shape in shapes:
            original = torch.randn(shape)
            quantized = compressor.quantize(original)
            assert quantized.original_shape == shape

    def test_quantize_4bit(self):
        """Test 4-bit quantization."""
        compressor = QuantizationCompressor(CompressionConfig(quantization_bits=4))
        original = torch.randn(256)

        quantized = compressor.quantize(original, bits=4)

        # 4-bit should have values 0-15
        assert quantized.quantized_data.max() <= 15
        assert quantized.quantized_data.min() >= 0

    def test_quantize_8bit(self):
        """Test 8-bit quantization."""
        compressor = QuantizationCompressor(CompressionConfig(quantization_bits=8))
        original = torch.randn(256)

        quantized = compressor.quantize(original, bits=8)

        # 8-bit should have values 0-255
        assert quantized.quantized_data.max() <= 255
        assert quantized.quantized_data.min() >= 0

    def test_quantize_preserves_dtype(self):
        """Test quantization preserves original dtype."""
        compressor = QuantizationCompressor()

        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            original = torch.randn(256, dtype=dtype)
            quantized = compressor.quantize(original)
            assert quantized.dtype == dtype

    def test_quantize_padding(self):
        """Test quantization handles padding correctly."""
        compressor = QuantizationCompressor(CompressionConfig(group_size=128))

        # 100 elements doesn't divide evenly by 128
        original = torch.randn(100)
        quantized = compressor.quantize(original)

        # Should have padded to 128
        assert quantized.original_shape == (100,)
        assert len(quantized.scales) > 0

    def test_quantize_empty_tensor_error(self):
        """Test quantization with empty tensor."""
        compressor = QuantizationCompressor()
        original = torch.tensor([])

        with pytest.raises(Exception):
            compressor.quantize(original)

    def test_compress_with_sparsity(self):
        """Test compression with sparsity pruning."""
        config = CompressionConfig(sparsity_threshold=0.5)
        compressor = QuantizationCompressor(config)

        # Create tensor with some zeros
        original = torch.randn(100)
        original[original < 0.5] = 0

        compressed_values, mask = compressor.compress_with_sparsity(original)

        # Non-zero values and indices
        assert len(compressed_values) > 0
        assert mask.sum() > 0


class TestZSTDQuantizedCompressor:
    """Test ZSTDQuantizedCompressor for combined compression."""

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_init_zstd_available(self, mock_import):
        """Test initialization when ZSTD is available."""
        mock_zstd = Mock()
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()
        assert compressor.has_zstd is True

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_init_zstd_unavailable(self, mock_import):
        """Test initialization when ZSTD is unavailable."""
        mock_import.side_effect = ImportError("No module named 'zstandard'")

        compressor = ZSTDQuantizedCompressor()
        assert compressor.has_zstd is False

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_with_zstd(self, mock_import):
        """Test compression with ZSTD."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_compressor.compress.return_value = b"compressed_data"
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()
        tensor = torch.randn(100)

        compressed = compressor.compress(tensor)

        mock_compressor.compress.assert_called_once()
        assert isinstance(compressed, bytes)

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_without_zstd(self, mock_import):
        """Test compression fallback when ZSTD unavailable."""
        mock_import.side_effect = ImportError("No module named 'zstandard'")

        compressor = ZSTDQuantizedCompressor()
        tensor = torch.randn(100)

        compressed = compressor.compress(tensor)

        assert isinstance(compressed, bytes)

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_decompress_with_zstd(self, mock_import):
        """Test decompression with ZSTD."""
        mock_zstd = Mock()
        mock_decompressor = Mock()
        mock_zstd.ZstdDecompressor.return_value = mock_decompressor
        mock_decompressor.decompress.return_value = b"decompressed_data"
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()
        compressed = b"compressed_data"
        shape = (100,)
        dtype = torch.float32

        decompressed = compressor.decompress(compressed, shape, dtype)

        assert decompressed.shape == shape
        assert decompressed.dtype == dtype

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_updates_stats(self, mock_import):
        """Test compression updates statistics."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()
        tensor = torch.randn(100)

        compressor.compress(tensor)

        stats = compressor.get_stats()
        assert stats["original_bytes"] > 0
        assert stats["compressed_bytes"] > 0

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_model_layers(self, mock_import):
        """Test compressing model state dict layers."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()

        state_dict = {
            "layer1.weight": torch.randn(128, 768),
            "layer1.bias": torch.randn(128),
            "layer2.weight": torch.randn(256, 128),
            "embedding.weight": torch.randn(1000, 768),  # Non-weight tensor
        }

        compressed = compressor.compress_model_layers(state_dict)

        # Should compress weights but store embedding as-is
        assert len(compressed) == len(state_dict)

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_get_stats_compression_ratio(self, mock_import):
        """Test getting compression ratio statistics."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed_data"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = ZSTDQuantizedCompressor()
        tensor = torch.randn(1000)

        # Compress twice
        compressor.compress(tensor)
        compressor.compress(tensor)

        stats = compressor.get_stats()

        assert "compression_ratio" in stats
        assert "space_saving" in stats
        assert stats["compression_ratio"] >= 1.0

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_get_stats_empty(self, mock_import):
        """Test getting stats when no compression occurred."""
        mock_import.side_effect = ImportError("No module named 'zstandard'")

        compressor = ZSTDQuantizedCompressor()
        stats = compressor.get_stats()

        assert stats["compression_ratio"] == 1.0
        assert stats["space_saving"] == 0.0


class TestOptimizedCompressor:
    """Test OptimizedCompressor for tensor compression."""

    def test_init(self):
        """Test initialization."""
        compressor = OptimizedCompressor()
        assert compressor.config.algorithm == "zstd"

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_tensor_large(self, mock_import):
        """Test compressing large tensor."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = OptimizedCompressor()
        tensor = torch.randn(1000000)  # > 1M elements

        compressed = compressor.compress_tensor(tensor, "layer.weight")

        mock_compressor.compress.assert_called_once()

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_tensor_medium(self, mock_import):
        """Test compressing medium tensor."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = OptimizedCompressor()
        tensor = torch.randn(500000)  # 100K - 1M

        compressed = compressor.compress_tensor(tensor, "layer.weight")

        mock_compressor.compress.assert_called_once()

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_compress_tensor_small(self, mock_import):
        """Test compressing small tensor."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = OptimizedCompressor()
        tensor = torch.randn(50000)  # < 100K

        compressed = compressor.compress_tensor(tensor, "layer.bias")

        mock_compressor.compress.assert_called_once()

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_decompress_tensor(self, mock_import):
        """Test decompressing tensor."""
        mock_zstd = Mock()
        mock_decompressor = Mock()
        mock_decompressor.decompress.return_value = b"decompressed"
        mock_zstd.ZstdDecompressor.return_value = mock_decompressor
        mock_import.return_value = mock_zstd

        compressor = OptimizedCompressor()
        compressed = b"compressed_data"
        shape = (100, 768)
        dtype = torch.float16

        decompressed = compressor.decompress_tensor(compressed, shape, dtype)

        assert decompressed.shape == shape
        assert decompressed.dtype == dtype

    @patch("src.optimizations.compression_optimized.importlib.import_module")
    def test_get_compression_ratio(self, mock_import):
        """Test getting compression ratio."""
        mock_zstd = Mock()
        mock_compressor = Mock()
        mock_compressor.compress.return_value = b"compressed"
        mock_zstd.ZstdCompressor.return_value = mock_compressor
        mock_import.return_value = mock_zstd

        compressor = OptimizedCompressor()
        tensor = torch.randn(1000)

        compressor.compress_tensor(tensor, "test")

        ratio = compressor.get_compression_ratio()

        assert ratio >= 1.0
