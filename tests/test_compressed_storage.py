"""
Tests for Compressed Storage Module

Comprehensive test suite covering:
- Layer compression and decompression
- Storage management
- LZ4 compression (if available)
- Statistics tracking
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.sli.compressed_storage import (
    CompressedLayerStorage,
    LayerCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    QuantizationType,
    CompressedEntry,
    compress_layer_to_storage,
    load_compressed_layer,
)


# Check if compression libraries are available
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


class TestCompressionConfig:
    """Test cases for CompressionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CompressionConfig()
        
        assert config.algorithm == CompressionAlgorithm.LZ4
        assert config.compression_level == 3
        assert config.enable_quantization == False

    def test_lz4_unavailable_fallback(self):
        """Test fallback when LZ4 unavailable."""
        with patch('nexus.models.sli.compressed_storage.LZ4_AVAILABLE', False):
            config = CompressionConfig(algorithm=CompressionAlgorithm.LZ4)
            assert config.algorithm == CompressionAlgorithm.NONE

    def test_custom_config(self):
        """Test custom configuration."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.NONE,
            compression_level=9,
            enable_quantization=True,
            quantization_type=QuantizationType.FP16
        )
        
        assert config.algorithm == CompressionAlgorithm.NONE
        assert config.compression_level == 9
        assert config.enable_quantization == True
        assert config.quantization_type == QuantizationType.FP16


class TestLayerCompressor:
    """Test cases for LayerCompressor."""

    @pytest.fixture
    def compressor(self):
        """Create a compressor."""
        return LayerCompressor(CompressionConfig(algorithm=CompressionAlgorithm.NONE))

    @pytest.fixture
    def test_layer(self):
        """Create a test layer."""
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

    def test_compress_decompress_no_compression(self, compressor, test_layer):
        """Test compression/decompression with no compression."""
        compressed_data, entry = compressor.compress_layer(test_layer, "test_layer")
        
        assert isinstance(compressed_data, bytes)
        assert isinstance(entry, CompressedEntry)
        assert entry.layer_id == "test_layer"
        assert entry.algorithm == CompressionAlgorithm.NONE

    def test_compression_ratio_calculation(self, compressor, test_layer):
        """Test compression ratio is calculated."""
        compressed_data, entry = compressor.compress_layer(test_layer, "test_layer")
        
        assert entry.compression_ratio > 0
        assert entry.original_size > 0
        assert entry.compressed_size > 0

    def test_checksum_computation(self, compressor, test_layer):
        """Test checksum computation."""
        compressed_data, entry = compressor.compress_layer(test_layer, "test_layer")
        
        assert len(entry.checksum_original) == 32  # MD5 hex
        assert len(entry.checksum_compressed) == 32

    @pytest.mark.skipif(not LZ4_AVAILABLE, reason="LZ4 not available")
    def test_lz4_compression(self, test_layer):
        """Test LZ4 compression."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.LZ4,
            compression_level=3
        )
        compressor = LayerCompressor(config)
        
        compressed_data, entry = compressor.compress_layer(test_layer, "test_layer")
        
        assert entry.algorithm == CompressionAlgorithm.LZ4
        assert entry.compression_ratio >= 1.0

    def test_stats_tracking(self, compressor, test_layer):
        """Test statistics tracking."""
        compressor.compress_layer(test_layer, "test_layer")
        
        stats = compressor.get_stats()
        assert 'total_compressed' in stats
        assert stats['total_compressed'] == 1


class TestCompressedLayerStorage:
    """Test cases for CompressedLayerStorage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create compressed storage."""
        return CompressedLayerStorage(
            storage_dir=temp_dir,
            config=CompressionConfig(algorithm=CompressionAlgorithm.NONE)
        )

    @pytest.fixture
    def test_layer(self):
        """Create a test layer."""
        return nn.Linear(512, 512)

    def test_store_layer(self, storage, test_layer):
        """Test storing a layer."""
        entry = storage.store_layer("test_layer", test_layer)
        
        assert isinstance(entry, CompressedEntry)
        assert entry.layer_id == "test_layer"
        assert entry.file_path is not None
        assert Path(entry.file_path).exists()

    def test_load_layer(self, storage, test_layer):
        """Test loading a compressed layer."""
        # Store first
        storage.store_layer("test_layer", test_layer)
        
        # Load
        loaded_layer = storage.load_layer("test_layer")
        
        assert loaded_layer is not None
        assert isinstance(loaded_layer, nn.Module)

    def test_load_nonexistent_layer(self, storage):
        """Test loading a non-existent layer."""
        loaded = storage.load_layer("nonexistent")
        assert loaded is None

    def test_delete_layer(self, storage, test_layer):
        """Test deleting a layer."""
        storage.store_layer("test_layer", test_layer)
        
        success = storage.delete_layer("test_layer")
        assert success == True
        
        # Verify deletion
        assert storage.load_layer("test_layer") is None

    def test_list_layers(self, storage, test_layer):
        """Test listing stored layers."""
        storage.store_layer("layer1", test_layer)
        storage.store_layer("layer2", test_layer)
        
        layers = storage.list_layers()
        assert len(layers) == 2
        assert "layer1" in layers
        assert "layer2" in layers

    def test_compression_stats(self, storage, test_layer):
        """Test getting compression stats."""
        storage.store_layer("test_layer", test_layer)
        
        stats = storage.get_compression_stats()
        assert 'total_compressed' in stats
        assert 'total_layers' in stats
        assert stats['total_layers'] == 1

    def test_get_entry_info(self, storage, test_layer):
        """Test getting entry info."""
        storage.store_layer("test_layer", test_layer)
        
        entry = storage.get_entry_info("test_layer")
        assert entry is not None
        assert entry.layer_id == "test_layer"

    def test_clear_all(self, storage, test_layer):
        """Test clearing all layers."""
        storage.store_layer("layer1", test_layer)
        storage.store_layer("layer2", test_layer)
        
        storage.clear_all()
        
        assert len(storage.list_layers()) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_compress_layer_to_storage(self, temp_dir):
        """Test compress and store convenience function."""
        layer = nn.Linear(256, 256)
        
        entry = compress_layer_to_storage(
            layer,
            "test_layer",
            storage_dir=temp_dir,
            algorithm=CompressionAlgorithm.NONE
        )
        
        assert isinstance(entry, CompressedEntry)

    def test_load_compressed_layer(self, temp_dir):
        """Test load convenience function."""
        layer = nn.Linear(256, 256)
        compress_layer_to_storage(
            layer,
            "test_layer",
            storage_dir=temp_dir,
            algorithm=CompressionAlgorithm.NONE
        )
        
        loaded = load_compressed_layer("test_layer", storage_dir=temp_dir)
        assert loaded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])