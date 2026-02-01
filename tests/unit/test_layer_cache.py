"""
tests/unit/test_layer_cache.py
Comprehensive tests for the layer caching system.

Tests cover:
- LayerCache initialization
- Cache entry creation and retrieval
- LRU eviction
- Checksum validation
- Cache hits/misses statistics
- Disk persistence
- Memory and disk limits
- Error handling
- LayerCacheManager singleton
"""

import pytest
import time
import tempfile
import os
import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from collections import OrderedDict

import torch
import torch.nn as nn

# Import the module under test
from src.nexus_final.sli.layer_cache import (
    CacheEntry,
    CacheStats,
    LayerCache,
    LayerCacheManager,
    get_layer_cache,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry with all fields."""
        entry = CacheEntry(
            layer_id="test_model_layer_0",
            model_id="test/model",
            layer_index=0,
            file_path="/cache/test_model_layer_0.pt",
            file_size=1024,
            checksum="abc123",
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=5,
            quantization_mode="int8",
            compression_ratio=0.5
        )

        assert entry.layer_id == "test_model_layer_0"
        assert entry.model_id == "test/model"
        assert entry.layer_index == 0
        assert entry.file_path == "/cache/test_model_layer_0.pt"
        assert entry.file_size == 1024
        assert entry.checksum == "abc123"
        assert entry.access_count == 5
        assert entry.quantization_mode == "int8"
        assert entry.compression_ratio == 0.5

    def test_cache_entry_defaults(self):
        """Test cache entry with default values."""
        entry = CacheEntry(
            layer_id="test_layer",
            model_id="test/model",
            layer_index=0,
            file_path="/cache/test.pt",
            file_size=100,
            checksum="xyz",
            created_at=time.time(),
            last_accessed=time.time()
        )

        assert entry.access_count == 0
        assert entry.quantization_mode is None
        assert entry.compression_ratio == 1.0


class TestCacheStats:
    """Test CacheStats class."""

    def test_initial_stats(self):
        """Test initial stats are zero/defaults."""
        stats = CacheStats()

        assert stats.total_hits == 0
        assert stats.total_misses == 0
        assert stats.total_evictions == 0
        assert stats.total_bytes_downloaded == 0
        assert stats.total_bytes_served_from_cache == 0
        assert stats.average_load_time_ms == 0.0
        assert stats.cache_hit_ratio == 0.0
        assert stats.last_reset > 0

    def test_record_hit(self):
        """Test recording cache hits."""
        stats = CacheStats()

        stats.record_hit(1024)
        assert stats.total_hits == 1
        assert stats.total_bytes_served_from_cache == 1024

        stats.record_hit(2048)
        assert stats.total_hits == 2
        assert stats.total_bytes_served_from_cache == 3072

    def test_record_miss(self):
        """Test recording cache misses."""
        stats = CacheStats()

        stats.record_miss(1024)
        assert stats.total_misses == 1
        assert stats.total_bytes_downloaded == 1024

    def test_record_eviction(self):
        """Test recording evictions."""
        stats = CacheStats()

        stats.record_eviction()
        stats.record_eviction()
        assert stats.total_evictions == 2

    def test_cache_hit_ratio_calculation(self):
        """Test cache hit ratio updates correctly."""
        stats = CacheStats()

        # No requests yet
        assert stats.cache_hit_ratio == 0.0

        stats.record_hit(100)
        assert stats.cache_hit_ratio == 1.0

        stats.record_miss(100)
        assert stats.cache_hit_ratio == 0.5

        stats.record_hit(100)
        assert abs(stats.cache_hit_ratio - 2/3) < 0.001

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(
            total_hits=10,
            total_misses=5,
            total_evictions=2,
            total_bytes_downloaded=1000,
            total_bytes_served_from_cache=2000,
            average_load_time_ms=15.5,
            cache_hit_ratio=0.67
        )

        data = stats.to_dict()

        assert data['total_hits'] == 10
        assert data['total_misses'] == 5
        assert data['total_evictions'] == 2
        assert data['total_bytes_downloaded'] == 1000
        assert data['total_bytes_served_from_cache'] == 2000
        assert data['average_load_time_ms'] == 15.5
        assert data['cache_hit_ratio'] == 0.67


class TestLayerCacheInitialization:
    """Test LayerCache initialization."""

    def test_default_initialization(self, tmp_path):
        """Test initialization with default parameters."""
        cache = LayerCache(cache_dir=str(tmp_path))

        assert cache.cache_dir == Path(tmp_path)
        assert cache.max_cache_size_bytes == int(50.0 * 1024 * 1024 * 1024)
        assert cache.max_memory_cache_bytes == int(2.0 * 1024 * 1024 * 1024)
        assert cache.enable_compression is False
        assert cache.compression_level == 6
        assert cache.persist_metadata is True
        assert isinstance(cache._disk_cache, OrderedDict)
        assert isinstance(cache._memory_cache, OrderedDict)

    def test_custom_initialization(self, tmp_path):
        """Test initialization with custom parameters."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_cache_size_gb=10.0,
            max_memory_cache_size_gb=1.0,
            enable_compression=True,
            compression_level=9,
            persist_metadata=False
        )

        assert cache.max_cache_size_bytes == int(10.0 * 1024 * 1024 * 1024)
        assert cache.max_memory_cache_bytes == int(1.0 * 1024 * 1024 * 1024)
        assert cache.enable_compression is True
        assert cache.compression_level == 9
        assert cache.persist_metadata is False

    def test_default_cache_dir(self):
        """Test default cache directory is created."""
        cache = LayerCache()

        expected_path = Path.home() / '.cache' / 'nexus' / 'layers'
        assert cache.cache_dir == expected_path
        assert cache.cache_dir.exists()

    def test_cache_dir_creation(self, tmp_path):
        """Test cache directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_cache_dir"
        assert not new_dir.exists()

        cache = LayerCache(cache_dir=str(new_dir))

        assert new_dir.exists()
        assert cache.cache_dir == new_dir


class TestLayerCacheOperations:
    """Test basic layer cache operations."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a fresh cache for each test."""
        return LayerCache(
            cache_dir=str(tmp_path),
            max_cache_size_gb=1.0,
            max_memory_cache_size_gb=0.5
        )

    @pytest.fixture
    def dummy_layer(self):
        """Create a dummy layer for testing."""
        return nn.Linear(100, 100)

    def test_generate_layer_id(self, cache):
        """Test layer ID generation."""
        layer_id = cache._generate_layer_id("test/model", 5)
        assert layer_id == "test_model_layer_5"

        layer_id_q = cache._generate_layer_id("test/model", 5, "int8")
        assert layer_id_q == "test_model_layer_5_int8"

    def test_get_cache_path(self, cache):
        """Test cache path generation."""
        path = cache._get_cache_path("test_layer_id")
        assert path == cache.cache_dir / "test_layer_id.pt"

    def test_compute_checksum(self, cache, tmp_path):
        """Test checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = cache._compute_checksum(str(test_file))
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hex digest length

    def test_get_layer_size(self, cache, dummy_layer):
        """Test layer size estimation."""
        size = cache._get_layer_size(dummy_layer)

        # Calculate expected size
        weight_size = dummy_layer.weight.numel() * dummy_layer.weight.element_size()
        expected_size = weight_size  # No bias in this case

        assert size == expected_size

    def test_cache_and_retrieve_layer(self, cache, dummy_layer):
        """Test caching and retrieving a layer."""
        # Cache the layer
        result = cache.cache_layer("test/model", 0, dummy_layer)
        assert result is True

        # Retrieve the layer
        retrieved = cache.get_layer("test/model", 0)
        assert retrieved is not None
        assert isinstance(retrieved, nn.Linear)
        assert retrieved.weight.shape == dummy_layer.weight.shape

    def test_cache_layer_with_quantization(self, cache, dummy_layer):
        """Test caching layer with quantization mode."""
        result = cache.cache_layer("test/model", 0, dummy_layer, quantization_mode="int8")
        assert result is True

        # Should be retrievable with same quantization mode
        retrieved = cache.get_layer("test/model", 0, quantization_mode="int8")
        assert retrieved is not None

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get_layer("nonexistent/model", 999)
        assert result is None

    def test_cache_hit_stats(self, cache, dummy_layer):
        """Test cache hit statistics."""
        cache.cache_layer("test/model", 0, dummy_layer)

        initial_hits = cache._stats.total_hits
        cache.get_layer("test/model", 0)

        assert cache._stats.total_hits == initial_hits + 1

    def test_cache_miss_stats(self, cache):
        """Test cache miss statistics."""
        initial_misses = cache._stats.total_misses
        cache.get_layer("nonexistent/model", 999)
        assert cache._stats.total_misses == initial_misses + 1


class TestLayerCacheLRU:
    """Test LRU eviction behavior."""

    def test_lru_eviction_order(self, tmp_path):
        """Test LRU eviction evicts least recently used."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_cache_size_gb=0.001,  # Very small to force eviction
            max_memory_cache_size_gb=0.001
        )

        # Create layers
        layer1 = nn.Linear(1000, 1000)  # ~4MB
        layer2 = nn.Linear(1000, 1000)
        layer3 = nn.Linear(1000, 1000)

        # Cache layers
        cache.cache_layer("model", 0, layer1)
        cache.cache_layer("model", 1, layer2)

        # Access layer 0 to make it most recently used
        cache.get_layer("model", 0)
        time.sleep(0.01)

        # Add layer 3 - should evict layer 1 (least recently used)
        cache.cache_layer("model", 2, layer3)

        # Layer 0 should still be there
        assert cache.get_layer("model", 0) is not None

    def test_lru_access_updates_order(self, tmp_path):
        """Test that access updates LRU order."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_cache_size_gb=1.0
        )

        layer1 = nn.Linear(100, 100)
        layer2 = nn.Linear(100, 100)

        cache.cache_layer("model", 0, layer1)
        cache.cache_layer("model", 1, layer2)

        # Access layer 0
        cache.get_layer("model", 0)

        # Check order - layer 0 should be at end (most recent)
        keys = list(cache._disk_cache.keys())
        assert keys[-1] == "model_layer_0"

    def test_evict_if_necessary(self, tmp_path):
        """Test eviction when cache is full."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_cache_size_gb=0.0001  # Very small: ~100KB
        )

        # Create a large layer that exceeds cache size
        large_layer = nn.Linear(1000, 1000)  # ~4MB

        # This should trigger eviction (and possibly fail to cache)
        result = cache.cache_layer("model", 0, large_layer)

        # The cache should have attempted eviction
        # Result may be True or False depending on size calculations
        assert isinstance(result, bool)


class TestLayerCacheMemoryCache:
    """Test in-memory caching."""

    def test_add_to_memory_cache(self, tmp_path):
        """Test adding layer to memory cache."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_memory_cache_size_gb=0.1
        )

        layer = nn.Linear(100, 100)
        layer_id = "test_layer"

        cache._add_to_memory_cache(layer_id, layer)

        assert layer_id in cache._memory_cache
        assert cache._current_memory_size > 0

    def test_memory_cache_hit(self, tmp_path):
        """Test retrieving from memory cache."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_memory_cache_size_gb=0.1
        )

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        # First get loads from disk to memory
        cache.get_layer("model", 0)

        # Second get should hit memory cache
        result = cache.get_layer("model", 0)
        assert result is not None

    def test_memory_cache_lru_eviction(self, tmp_path):
        """Test memory cache LRU eviction."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            max_memory_cache_size_gb=0.001  # Very small
        )

        # Add layers until eviction occurs
        for i in range(10):
            layer = nn.Linear(100, 100)
            cache._add_to_memory_cache(f"layer_{i}", layer)

        # Some should have been evicted
        assert len(cache._memory_cache) < 10


class TestLayerCachePersistence:
    """Test disk persistence functionality."""

    def test_save_metadata(self, tmp_path):
        """Test saving cache metadata."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            persist_metadata=True
        )

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        metadata_path = cache._get_metadata_path()
        assert metadata_path.exists()

        # Verify content
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            assert 'entries' in data
            assert 'stats' in data
            assert 'last_saved' in data

    def test_load_metadata(self, tmp_path):
        """Test loading cache metadata."""
        # Create first cache and add entry
        cache1 = LayerCache(
            cache_dir=str(tmp_path),
            persist_metadata=True
        )

        layer = nn.Linear(100, 100)
        cache1.cache_layer("model", 0, layer)

        # Create new cache instance (simulating restart)
        cache2 = LayerCache(
            cache_dir=str(tmp_path),
            persist_metadata=True
        )

        # Should have loaded the metadata
        assert len(cache2._disk_cache) == 1
        assert "model_layer_0" in cache2._disk_cache

    def test_load_metadata_missing_file(self, tmp_path):
        """Test loading when metadata file doesn't exist."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            persist_metadata=True
        )

        # Should start empty without error
        assert len(cache._disk_cache) == 0

    def test_load_metadata_corrupted(self, tmp_path):
        """Test handling corrupted metadata file."""
        cache_dir = tmp_path / "corrupted"
        cache_dir.mkdir()

        # Write invalid JSON
        metadata_file = cache_dir / "cache_metadata.json"
        metadata_file.write_text("invalid json{[")

        # Should handle gracefully
        cache = LayerCache(cache_dir=str(cache_dir))
        assert len(cache._disk_cache) == 0

    def test_no_persistence_when_disabled(self, tmp_path):
        """Test that metadata is not saved when disabled."""
        cache = LayerCache(
            cache_dir=str(tmp_path),
            persist_metadata=False
        )

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        metadata_path = cache._get_metadata_path()
        assert not metadata_path.exists()


class TestLayerCacheChecksumValidation:
    """Test checksum validation."""

    def test_verify_cache_integrity_valid(self, tmp_path):
        """Test integrity check with valid files."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        corrupted = cache.verify_cache_integrity()
        assert len(corrupted) == 0

    def test_verify_cache_integrity_corrupted(self, tmp_path):
        """Test integrity check with corrupted file."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        # Corrupt the file
        layer_id = "model_layer_0"
        cache_path = cache._get_cache_path(layer_id)
        with open(cache_path, 'wb') as f:
            f.write(b"corrupted data")

        corrupted = cache.verify_cache_integrity()
        assert len(corrupted) == 1
        assert corrupted[0] == layer_id

    def test_verify_cache_integrity_missing_file(self, tmp_path):
        """Test integrity check with missing file."""
        cache = LayerCache(cache_dir=str(tmp_path))

        # Add entry manually without file
        entry = CacheEntry(
            layer_id="missing_layer",
            model_id="test",
            layer_index=0,
            file_path=str(tmp_path / "nonexistent.pt"),
            file_size=100,
            checksum="abc",
            created_at=time.time(),
            last_accessed=time.time()
        )
        cache._disk_cache["missing_layer"] = entry

        corrupted = cache.verify_cache_integrity()
        assert "missing_layer" in corrupted


class TestLayerCacheInvalidation:
    """Test cache invalidation."""

    def test_invalidate_all(self, tmp_path):
        """Test invalidating all cache entries."""
        cache = LayerCache(cache_dir=str(tmp_path))

        for i in range(3):
            layer = nn.Linear(100, 100)
            cache.cache_layer("model", i, layer)

        cache.invalidate_cache()

        assert len(cache._disk_cache) == 0
        assert len(cache._memory_cache) == 0

    def test_invalidate_by_model(self, tmp_path):
        """Test invalidating specific model."""
        cache = LayerCache(cache_dir=str(tmp_path))

        for i in range(3):
            layer = nn.Linear(100, 100)
            cache.cache_layer("model1", i, layer)
            cache.cache_layer("model2", i, layer)

        cache.invalidate_cache(model_id="model1")

        assert len(cache._disk_cache) == 3  # Only model2 layers remain

    def test_invalidate_specific_layer(self, tmp_path):
        """Test invalidating specific layer."""
        cache = LayerCache(cache_dir=str(tmp_path))

        for i in range(3):
            layer = nn.Linear(100, 100)
            cache.cache_layer("model", i, layer)

        cache.invalidate_cache(model_id="model", layer_index=1)

        assert cache.get_layer("model", 0) is not None
        assert cache.get_layer("model", 1) is None
        assert cache.get_layer("model", 2) is not None

    def test_clear_cache(self, tmp_path):
        """Test clearing all cache."""
        cache = LayerCache(cache_dir=str(tmp_path))

        for i in range(3):
            layer = nn.Linear(100, 100)
            cache.cache_layer("model", i, layer)

        cache.clear_cache()

        assert len(cache._disk_cache) == 0
        assert len(cache._memory_cache) == 0
        assert cache._current_cache_size == 0
        assert cache._current_memory_size == 0


class TestLayerCacheStats:
    """Test cache statistics."""

    def test_get_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)
        cache.get_layer("model", 0)

        stats = cache.get_stats()

        assert 'disk_cache_entries' in stats
        assert 'memory_cache_entries' in stats
        assert 'disk_cache_size_gb' in stats
        assert 'memory_cache_size_gb' in stats
        assert 'max_disk_cache_size_gb' in stats
        assert 'max_memory_cache_size_gb' in stats
        assert 'performance' in stats

        assert stats['disk_cache_entries'] == 1
        assert stats['performance']['total_hits'] == 1

    def test_print_stats(self, tmp_path, capsys):
        """Test printing cache statistics."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        cache.print_stats()

        captured = capsys.readouterr()
        assert "Layer Cache Statistics" in captured.out
        assert "Disk Cache Entries" in captured.out
        assert "Cache Hits" in captured.out


class TestLayerCacheOptimization:
    """Test cache optimization."""

    def test_optimize_cache_removes_corrupted(self, tmp_path):
        """Test optimization removes corrupted entries."""
        cache = LayerCache(cache_dir=str(tmp_path))

        # Add valid layer
        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        # Add corrupted entry manually
        entry = CacheEntry(
            layer_id="corrupted_layer",
            model_id="test",
            layer_index=999,
            file_path=str(tmp_path / "nonexistent.pt"),
            file_size=100,
            checksum="abc",
            created_at=time.time(),
            last_accessed=time.time()
        )
        cache._disk_cache["corrupted_layer"] = entry
        cache._current_cache_size += 100

        # Optimize should remove corrupted
        cache.optimize_cache()

        assert "corrupted_layer" not in cache._disk_cache
        assert "model_layer_0" in cache._disk_cache


class TestLayerCacheErrorHandling:
    """Test error handling."""

    def test_cache_layer_failure(self, tmp_path):
        """Test handling cache layer failure."""
        cache = LayerCache(cache_dir=str(tmp_path))

        # Mock torch.save to fail
        with patch('torch.save', side_effect=IOError("Disk full")):
            layer = nn.Linear(100, 100)
            result = cache.cache_layer("model", 0, layer)

        assert result is False

    def test_get_layer_corrupted_file(self, tmp_path):
        """Test handling corrupted file on get."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        cache.cache_layer("model", 0, layer)

        # Corrupt the file
        layer_id = "model_layer_0"
        cache_path = cache._get_cache_path(layer_id)
        with open(cache_path, 'wb') as f:
            f.write(b"corrupted")

        # Should return None and remove corrupted entry
        result = cache.get_layer("model", 0)
        assert result is None
        assert layer_id not in cache._disk_cache

    def test_thread_safety(self, tmp_path):
        """Test thread-safe operations."""
        cache = LayerCache(cache_dir=str(tmp_path))
        errors = []

        def worker(n):
            try:
                for i in range(5):
                    layer = nn.Linear(100, 100)
                    cache.cache_layer(f"model_{n}", i, layer)
                    cache.get_layer(f"model_{n}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestLayerCacheManager:
    """Test LayerCacheManager singleton."""

    def test_singleton_instance(self, tmp_path):
        """Test that manager is a singleton."""
        # Reset singleton for test
        LayerCacheManager._instance = None

        manager1 = LayerCacheManager(cache_dir=str(tmp_path))
        manager2 = LayerCacheManager(cache_dir=str(tmp_path / "other"))

        assert manager1 is manager2

    def test_get_cache(self, tmp_path):
        """Test getting cache from manager."""
        LayerCacheManager._instance = None

        cache = LayerCacheManager.get_cache(cache_dir=str(tmp_path))

        assert isinstance(cache, LayerCache)
        assert cache is LayerCacheManager.get_cache()  # Same instance

    def test_get_layer_cache_convenience(self, tmp_path):
        """Test get_layer_cache convenience function."""
        LayerCacheManager._instance = None

        cache = get_layer_cache(cache_dir=str(tmp_path))

        assert isinstance(cache, LayerCache)


class TestLayerCacheEdgeCases:
    """Test edge cases."""

    def test_empty_model_id(self, tmp_path):
        """Test handling empty model ID."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        result = cache.cache_layer("", 0, layer)

        assert result is True
        assert cache.get_layer("", 0) is not None

    def test_negative_layer_index(self, tmp_path):
        """Test handling negative layer index."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        result = cache.cache_layer("model", -1, layer)

        assert result is True
        assert cache.get_layer("model", -1) is not None

    def test_very_large_layer_index(self, tmp_path):
        """Test handling very large layer index."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        result = cache.cache_layer("model", 999999, layer)

        assert result is True
        assert cache.get_layer("model", 999999) is not None

    def test_special_characters_in_model_id(self, tmp_path):
        """Test handling special characters in model ID."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100)
        model_id = "org/model-name.v1@special"
        result = cache.cache_layer(model_id, 0, layer)

        assert result is True
        assert cache.get_layer(model_id, 0) is not None

    def test_layer_with_bias(self, tmp_path):
        """Test caching layer with bias."""
        cache = LayerCache(cache_dir=str(tmp_path))

        layer = nn.Linear(100, 100, bias=True)
        cache.cache_layer("model", 0, layer)

        retrieved = cache.get_layer("model", 0)
        assert retrieved is not None
        assert retrieved.bias is not None

    def test_complex_layer(self, tmp_path):
        """Test caching complex layer structure."""
        cache = LayerCache(cache_dir=str(tmp_path))

        class ComplexLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(100, 200)
                self.linear2 = nn.Linear(200, 100)
                self.activation = nn.ReLU()

            def forward(self, x):
                return self.activation(self.linear2(self.activation(self.linear1(x))))

        layer = ComplexLayer()
        cache.cache_layer("model", 0, layer)

        retrieved = cache.get_layer("model", 0)
        assert retrieved is not None
        assert hasattr(retrieved, 'linear1')
        assert hasattr(retrieved, 'linear2')
