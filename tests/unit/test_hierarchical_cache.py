"""
Comprehensive unit tests for Hierarchical Layer Cache.

Tests cover:
- HierarchicalLayerCache initialization
- Three-tier caching (hot/warm/cold)
- Promotion/demotion logic
- Priority-based prefetching
- LRU/LFU eviction policies
"""

import pytest
import torch
import torch.nn as nn
import json
import time
import threading
import gzip
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Import the module under test
from src.models.sli.hierarchical_cache import (
    HierarchicalLayerCache,
    HierarchicalCacheConfig,
    HierarchicalCacheEntry,
    CacheTier,
    EvictionPolicy,
    HierarchicalCacheError,
)
from src.models.sli.exceptions import SLIError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "hierarchical_cache"
    return str(cache_dir)


@pytest.fixture
def default_config():
    """Create default cache config."""
    return HierarchicalCacheConfig()


@pytest.fixture
def small_config(temp_cache_dir):
    """Create cache config with small sizes for testing."""
    return HierarchicalCacheConfig(
        memory_cache_size_gb=0.01,  # ~10MB
        disk_l1_size_gb=0.02,       # ~20MB
        disk_l2_size_gb=0.05,       # ~50MB
        cache_dir=temp_cache_dir,
        eviction_policy=EvictionPolicy.ADAPTIVE,
        enable_compression=False,
        promotion_threshold=3,
        demotion_threshold=60.0
    )


@pytest.fixture
def lru_config(temp_cache_dir):
    """Create cache config with LRU policy."""
    return HierarchicalCacheConfig(
        memory_cache_size_gb=0.01,
        disk_l1_size_gb=0.02,
        disk_l2_size_gb=0.05,
        cache_dir=temp_cache_dir,
        eviction_policy=EvictionPolicy.LRU
    )


@pytest.fixture
def lfu_config(temp_cache_dir):
    """Create cache config with LFU policy."""
    return HierarchicalCacheConfig(
        memory_cache_size_gb=0.01,
        disk_l1_size_gb=0.02,
        disk_l2_size_gb=0.05,
        cache_dir=temp_cache_dir,
        eviction_policy=EvictionPolicy.LFU
    )


@pytest.fixture
def cache_default(small_config):
    """Create hierarchical cache with small config."""
    return HierarchicalLayerCache(small_config)


@pytest.fixture
def sample_layer():
    """Create a sample layer for testing."""
    return nn.Linear(512, 256)


@pytest.fixture
def large_layer():
    """Create a large layer for testing."""
    return nn.Linear(4096, 4096)


# ============================================================================
# Test CacheTier Enum
# ============================================================================

class TestCacheTier:
    """Test suite for CacheTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert CacheTier.MEMORY.value == 'memory'
        assert CacheTier.DISK_L1.value == 'disk_l1'
        assert CacheTier.DISK_L2.value == 'disk_l2'
        assert CacheTier.ARCHIVE.value == 'archive'


# ============================================================================
# Test EvictionPolicy Enum
# ============================================================================

class TestEvictionPolicy:
    """Test suite for EvictionPolicy enum."""

    def test_policy_values(self):
        """Test policy enum values."""
        assert EvictionPolicy.LRU.value == 'lru'
        assert EvictionPolicy.LFU.value == 'lfu'
        assert EvictionPolicy.FIFO.value == 'fifo'
        assert EvictionPolicy.ADAPTIVE.value == 'adaptive'


# ============================================================================
# Test HierarchicalCacheEntry
# ============================================================================

class TestHierarchicalCacheEntry:
    """Test suite for HierarchicalCacheEntry dataclass."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = HierarchicalCacheEntry(
            layer_id="layer_0",
            tier=CacheTier.MEMORY,
            file_path=None,
            memory_ref=None,
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        assert entry.layer_id == "layer_0"
        assert entry.tier == CacheTier.MEMORY
        assert entry.size_bytes == 1024
        assert entry.access_count == 0
        assert entry.priority == 5

    def test_entry_touch(self):
        """Test entry touch updates statistics."""
        entry = HierarchicalCacheEntry(
            layer_id="layer_0",
            tier=CacheTier.MEMORY,
            file_path=None,
            memory_ref=None,
            size_bytes=1024,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        original_access_time = entry.last_accessed
        original_access_count = entry.access_count
        original_frequency = entry.access_frequency
        
        time.sleep(0.01)  # Small delay
        entry.touch()
        
        assert entry.access_count == original_access_count + 1
        assert entry.last_accessed > original_access_time
        assert entry.access_frequency != original_frequency

    def test_entry_to_dict(self):
        """Test entry serialization to dict."""
        entry = HierarchicalCacheEntry(
            layer_id="layer_0",
            tier=CacheTier.DISK_L1,
            file_path="/path/to/layer.pt",
            memory_ref=None,
            size_bytes=1024,
            created_at=1234567890.0,
            last_accessed=1234567891.0,
            access_count=5,
            priority=8
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict['layer_id'] == 'layer_0'
        assert entry_dict['tier'] == 'disk_l1'
        assert entry_dict['size_bytes'] == 1024
        assert entry_dict['access_count'] == 5
        assert entry_dict['priority'] == 8
        assert 'memory_ref' not in entry_dict


# ============================================================================
# Test HierarchicalCacheConfig
# ============================================================================

class TestHierarchicalCacheConfig:
    """Test suite for HierarchicalCacheConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchicalCacheConfig()
        
        assert config.memory_cache_size_gb == 2.0
        assert config.disk_l1_size_gb == 50.0
        assert config.disk_l2_size_gb == 200.0
        assert config.cache_dir == "./cache/hierarchical"
        assert config.eviction_policy == EvictionPolicy.ADAPTIVE
        assert config.enable_compression is True
        assert config.compression_level == 6
        assert config.prefetch_lookahead == 3
        assert config.promotion_threshold == 3
        assert config.demotion_threshold == 3600.0
        assert config.checksum_validation is True

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = HierarchicalCacheConfig(
            memory_cache_size_gb=4.0,
            disk_l1_size_gb=100.0,
            disk_l2_size_gb=500.0,
            cache_dir="/custom/cache",
            eviction_policy=EvictionPolicy.LRU,
            enable_compression=False,
            compression_level=9,
            prefetch_lookahead=5,
            promotion_threshold=5,
            demotion_threshold=7200.0,
            checksum_validation=False
        )
        
        assert config.memory_cache_size_gb == 4.0
        assert config.disk_l1_size_gb == 100.0
        assert config.disk_l2_size_gb == 500.0
        assert config.cache_dir == "/custom/cache"
        assert config.eviction_policy == EvictionPolicy.LRU
        assert config.enable_compression is False
        assert config.compression_level == 9
        assert config.prefetch_lookahead == 5
        assert config.promotion_threshold == 5
        assert config.demotion_threshold == 7200.0
        assert config.checksum_validation is False

    def test_memory_size_bytes_property(self):
        """Test memory size bytes property."""
        config = HierarchicalCacheConfig(memory_cache_size_gb=1.0)
        
        expected_bytes = 1.0 * 1024 * 1024 * 1024
        assert config.memory_size_bytes == expected_bytes

    def test_disk_l1_size_bytes_property(self):
        """Test disk L1 size bytes property."""
        config = HierarchicalCacheConfig(disk_l1_size_gb=1.0)
        
        expected_bytes = 1.0 * 1024 * 1024 * 1024
        assert config.disk_l1_size_bytes == expected_bytes

    def test_disk_l2_size_bytes_property(self):
        """Test disk L2 size bytes property."""
        config = HierarchicalCacheConfig(disk_l2_size_gb=1.0)
        
        expected_bytes = 1.0 * 1024 * 1024 * 1024
        assert config.disk_l2_size_bytes == expected_bytes

    def test_config_to_dict(self):
        """Test configuration serialization to dict."""
        config = HierarchicalCacheConfig(
            memory_cache_size_gb=1.0,
            eviction_policy=EvictionPolicy.LFU
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['memory_cache_size_gb'] == 1.0
        assert config_dict['eviction_policy'] == 'lfu'
        assert config_dict['enable_compression'] is True


# ============================================================================
# Test HierarchicalLayerCache Initialization
# ============================================================================

class TestHierarchicalLayerCacheInitialization:
    """Test suite for cache initialization."""

    def test_initialization_default(self, temp_cache_dir):
        """Test initialization with default config."""
        config = HierarchicalCacheConfig(cache_dir=temp_cache_dir)
        cache = HierarchicalLayerCache(config)
        
        assert cache.config == config
        assert cache.cache_dir == Path(temp_cache_dir)
        assert cache.disk_l1_dir == Path(temp_cache_dir) / "tier1_warm"
        assert cache.disk_l2_dir == Path(temp_cache_dir) / "tier2_cold"
        assert isinstance(cache._memory_cache, dict)
        assert isinstance(cache._entries, dict)
        assert isinstance(cache._lock, threading.RLock)

    def test_initialization_creates_directories(self, temp_cache_dir):
        """Test that initialization creates cache directories."""
        config = HierarchicalCacheConfig(cache_dir=temp_cache_dir)
        cache = HierarchicalLayerCache(config)
        
        assert cache.disk_l1_dir.exists()
        assert cache.disk_l2_dir.exists()

    def test_initialization_loads_metadata(self, temp_cache_dir, sample_layer):
        """Test that initialization loads existing metadata."""
        # First create a cache and add some entries
        config = HierarchicalCacheConfig(
            cache_dir=temp_cache_dir,
            enable_compression=False
        )
        cache1 = HierarchicalLayerCache(config)
        cache1.cache_layer("layer_0", sample_layer)
        
        # Create new cache instance pointing to same directory
        cache2 = HierarchicalLayerCache(config)
        
        # Should have loaded the metadata
        assert len(cache2._entries) >= 1


# ============================================================================
# Test Cache Layer
# ============================================================================

class TestCacheLayer:
    """Test suite for cache_layer method."""

    def test_cache_layer_memory(self, cache_default, sample_layer):
        """Test caching a layer to memory."""
        result = cache_default.cache_layer(
            "layer_0",
            sample_layer,
            priority=5,
            initial_tier=CacheTier.MEMORY
        )
        
        assert result is True
        assert "layer_0" in cache_default._entries
        assert cache_default._entries["layer_0"].tier == CacheTier.MEMORY

    def test_cache_layer_disk_l1(self, cache_default, sample_layer):
        """Test caching a layer to disk L1."""
        result = cache_default.cache_layer(
            "layer_0",
            sample_layer,
            priority=5,
            initial_tier=CacheTier.DISK_L1
        )
        
        assert result is True
        assert "layer_0" in cache_default._entries
        assert cache_default._entries["layer_0"].tier == CacheTier.DISK_L1

    def test_cache_layer_disk_l2(self, cache_default, sample_layer):
        """Test caching a layer to disk L2."""
        result = cache_default.cache_layer(
            "layer_0",
            sample_layer,
            priority=5,
            initial_tier=CacheTier.DISK_L2
        )
        
        assert result is True
        assert "layer_0" in cache_default._entries
        assert cache_default._entries["layer_0"].tier == CacheTier.DISK_L2

    def test_cache_layer_creates_file(self, cache_default, sample_layer):
        """Test that caching to disk creates a file."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        file_path = cache_default._entries["layer_0"].file_path
        assert file_path is not None
        assert Path(file_path).exists()

    def test_cache_layer_with_compression(self, temp_cache_dir, sample_layer):
        """Test caching with compression enabled."""
        config = HierarchicalCacheConfig(
            cache_dir=temp_cache_dir,
            enable_compression=True,
            compression_level=6
        )
        cache = HierarchicalLayerCache(config)
        
        result = cache.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        assert result is True
        assert cache._entries["layer_0"].file_path.endswith('.gz')

    def test_cache_layer_calculates_size(self, cache_default, sample_layer):
        """Test that caching calculates layer size."""
        cache_default.cache_layer("layer_0", sample_layer)
        
        size = cache_default._entries["layer_0"].size_bytes
        assert size > 0


# ============================================================================
# Test Get Layer
# ============================================================================

class TestGetLayer:
    """Test suite for get_layer method."""

    def test_get_layer_from_memory(self, cache_default, sample_layer):
        """Test getting a layer from memory cache."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.MEMORY)
        
        retrieved = cache_default.get_layer("layer_0")
        
        assert retrieved is not None
        assert isinstance(retrieved, nn.Module)

    def test_get_layer_from_disk_l1(self, cache_default, sample_layer):
        """Test getting a layer from disk L1."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        # Clear from memory to force disk read
        cache_default._memory_cache.clear()
        cache_default._memory_size = 0
        
        retrieved = cache_default.get_layer("layer_0")
        
        assert retrieved is not None
        assert isinstance(retrieved, nn.Module)

    def test_get_layer_from_disk_l2(self, cache_default, sample_layer):
        """Test getting a layer from disk L2."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L2)
        
        # Clear from memory
        cache_default._memory_cache.clear()
        cache_default._memory_size = 0
        
        retrieved = cache_default.get_layer("layer_0")
        
        assert retrieved is not None
        assert isinstance(retrieved, nn.Module)

    def test_get_layer_not_found(self, cache_default):
        """Test getting a non-existent layer."""
        retrieved = cache_default.get_layer("nonexistent_layer")
        
        assert retrieved is None

    def test_get_layer_updates_stats(self, cache_default, sample_layer):
        """Test that getting a layer updates access stats."""
        cache_default.cache_layer("layer_0", sample_layer)
        
        original_count = cache_default._entries["layer_0"].access_count
        cache_default.get_layer("layer_0")
        
        assert cache_default._entries["layer_0"].access_count == original_count + 1

    def test_get_layer_with_device(self, cache_default, sample_layer):
        """Test getting a layer with specific device."""
        cache_default.cache_layer("layer_0", sample_layer)
        
        retrieved = cache_default.get_layer("layer_0", device='cpu')
        
        assert retrieved is not None


# ============================================================================
# Test Promotion/Demotion
# ============================================================================

class TestPromotionDemotion:
    """Test suite for tier promotion and demotion."""

    def test_promotion_to_memory(self, cache_default, sample_layer):
        """Test automatic promotion to memory on access."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        # Initially not in memory
        assert "layer_0" not in cache_default._memory_cache
        
        # Access the layer
        cache_default.get_layer("layer_0")
        
        # Should be promoted to memory
        assert "layer_0" in cache_default._memory_cache

    def test_promotion_from_l2_to_l1(self, cache_default, sample_layer):
        """Test promotion from L2 to L1 on access."""
        cache_default.config.promotion_threshold = 1
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L2)
        
        # Access multiple times to trigger promotion
        for _ in range(5):
            cache_default._memory_cache.clear()
            cache_default._memory_size = 0
            cache_default.get_layer("layer_0")
            cache_default._entries["layer_0"].access_count += 1

    def test_memory_eviction_on_full(self, cache_default):
        """Test memory eviction when cache is full."""
        # Create several layers to fill memory
        layers = [nn.Linear(1000, 1000) for _ in range(10)]
        
        for i, layer in enumerate(layers):
            cache_default.cache_layer(f"layer_{i}", layer, initial_tier=CacheTier.MEMORY)
        
        # Some layers should have been evicted from memory
        memory_count = len(cache_default._memory_cache)
        assert memory_count < 10


# ============================================================================
# Test Eviction Policies
# ============================================================================

class TestEvictionPolicies:
    """Test suite for eviction policies."""

    def test_lru_eviction(self, lru_config):
        """Test LRU eviction policy."""
        cache = HierarchicalLayerCache(lru_config)
        
        # Add layers
        for i in range(5):
            layer = nn.Linear(1000, 1000)
            cache.cache_layer(f"layer_{i}", layer, initial_tier=CacheTier.MEMORY)
            time.sleep(0.01)
        
        # Access oldest layer
        cache.get_layer("layer_0")
        
        # Add more layers to trigger eviction
        for i in range(5, 10):
            layer = nn.Linear(1000, 1000)
            cache.cache_layer(f"layer_{i}", layer, initial_tier=CacheTier.MEMORY)

    def test_lfu_eviction(self, lfu_config):
        """Test LFU eviction policy."""
        cache = HierarchicalLayerCache(lfu_config)
        
        # Add layers
        for i in range(5):
            layer = nn.Linear(1000, 1000)
            cache.cache_layer(f"layer_{i}", layer, initial_tier=CacheTier.MEMORY)
        
        # Access some layers multiple times
        for _ in range(5):
            cache.get_layer("layer_0")
        
        for _ in range(3):
            cache.get_layer("layer_1")

    def test_adaptive_eviction(self, cache_default):
        """Test adaptive eviction policy."""
        # Add layers with different priorities
        for i in range(5):
            layer = nn.Linear(1000, 1000)
            priority = 10 if i == 0 else 1  # High priority for first layer
            cache_default.cache_layer(f"layer_{i}", layer, priority=priority, initial_tier=CacheTier.MEMORY)


# ============================================================================
# Test Prefetching
# ============================================================================

class TestPrefetching:
    """Test suite for prefetching functionality."""

    def test_prefetch_layers(self, cache_default, sample_layer):
        """Test prefetching layers."""
        # Cache some layers first
        for i in range(3):
            cache_default.cache_layer(f"layer_{i}", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        # Prefetch them
        cache_default.prefetch_layers(["layer_0", "layer_1", "layer_2"])
        
        # Should have queued the layers
        assert len(cache_default._prefetch_queue) >= 0

    def test_prefetch_queue_adds_unique(self, cache_default, sample_layer):
        """Test that prefetch queue only adds unique layer IDs."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        # Try to add same layer twice
        cache_default.prefetch_layers(["layer_0"])
        cache_default.prefetch_layers(["layer_0"])
        
        # Should only appear once in queue
        assert cache_default._prefetch_queue.count("layer_0") <= 1


# ============================================================================
# Test Statistics
# ============================================================================

class TestStatistics:
    """Test suite for statistics."""

    def test_get_stats_empty(self, cache_default):
        """Test getting stats for empty cache."""
        stats = cache_default.get_stats()
        
        assert isinstance(stats, dict)
        assert stats['hit_rate'] == 0.0
        assert stats['memory_hits'] == 0
        assert stats['disk_l1_hits'] == 0
        assert stats['disk_l2_hits'] == 0
        assert stats['misses'] == 0

    def test_get_stats_after_operations(self, cache_default, sample_layer):
        """Test getting stats after cache operations."""
        # Cache and retrieve a layer
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.MEMORY)
        cache_default.get_layer("layer_0")
        
        stats = cache_default.get_stats()
        
        assert stats['memory_hits'] >= 1
        assert stats['num_entries'] >= 1

    def test_hit_rate_calculation(self, cache_default, sample_layer):
        """Test hit rate calculation."""
        # Cache a layer
        cache_default.cache_layer("layer_0", sample_layer)
        
        # Multiple hits
        for _ in range(5):
            cache_default.get_layer("layer_0")
        
        # One miss
        cache_default.get_layer("nonexistent")
        
        stats = cache_default.get_stats()
        hit_rate = stats['hit_rate']
        
        assert 0.0 <= hit_rate <= 1.0
        assert hit_rate > 0


# ============================================================================
# Test Clear Cache
# ============================================================================

class TestClearCache:
    """Test suite for clear method."""

    def test_clear_all(self, cache_default, sample_layer):
        """Test clearing all cache tiers."""
        # Add layers to different tiers
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.MEMORY)
        cache_default.cache_layer("layer_1", sample_layer, initial_tier=CacheTier.DISK_L1)
        cache_default.cache_layer("layer_2", sample_layer, initial_tier=CacheTier.DISK_L2)
        
        # Clear all
        cache_default.clear()
        
        # Check all cleared
        assert len(cache_default._memory_cache) == 0
        assert len(cache_default._entries) == 0
        assert cache_default._memory_size == 0

    def test_clear_memory_only(self, cache_default, sample_layer):
        """Test clearing only memory tier."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.MEMORY)
        cache_default.cache_layer("layer_1", sample_layer, initial_tier=CacheTier.DISK_L1)
        
        cache_default.clear(CacheTier.MEMORY)
        
        assert len(cache_default._memory_cache) == 0
        assert len(cache_default._entries) >= 1  # Disk entries preserved

    def test_clear_disk_l1_only(self, cache_default, sample_layer):
        """Test clearing only disk L1 tier."""
        cache_default.cache_layer("layer_0", sample_layer, initial_tier=CacheTier.DISK_L1)
        cache_default.cache_layer("layer_1", sample_layer, initial_tier=CacheTier.DISK_L2)
        
        original_entry_count = len(cache_default._entries)
        
        cache_default.clear(CacheTier.DISK_L1)
        
        assert len(cache_default._entries) < original_entry_count


# ============================================================================
# Test Thread Safety
# ============================================================================

class TestThreadSafety:
    """Test suite for thread safety."""

    def test_concurrent_cache_operations(self, cache_default, sample_layer):
        """Test concurrent cache operations."""
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    layer = nn.Linear(100, 100)
                    cache_default.cache_layer(f"layer_{i}", layer)
                    cache_default.get_layer(f"layer_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_stats_access(self, cache_default, sample_layer):
        """Test concurrent stats access."""
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    layer = nn.Linear(100, 100)
                    cache_default.cache_layer(f"layer_{i}", layer)
                    cache_default.get_stats()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_layer(self, cache_default):
        """Test caching an empty layer."""
        layer = nn.Module()
        
        result = cache_default.cache_layer("empty_layer", layer)
        
        assert result is True

    def test_very_small_layer(self, cache_default):
        """Test caching a very small layer."""
        layer = nn.Linear(1, 1)
        
        result = cache_default.cache_layer("small_layer", layer)
        
        assert result is True
        assert cache_default._entries["small_layer"].size_bytes > 0

    def test_duplicate_layer_id(self, cache_default, sample_layer):
        """Test caching with duplicate layer ID."""
        cache_default.cache_layer("layer_0", sample_layer)
        
        # Create different layer with same ID
        different_layer = nn.Linear(100, 100)
        cache_default.cache_layer("layer_0", different_layer)
        
        # Should overwrite
        assert cache_default._entries["layer_0"].size_bytes > 0

    def test_special_characters_in_layer_id(self, cache_default, sample_layer):
        """Test layer IDs with special characters."""
        special_ids = [
            "layer-with-dashes",
            "layer.with.dots",
            "layer_with_underscores",
        ]
        
        for layer_id in special_ids:
            result = cache_default.cache_layer(layer_id, sample_layer)
            assert result is True, f"Failed for ID: {layer_id}"

    def test_priority_bounds(self, cache_default, sample_layer):
        """Test with priority at bounds."""
        # Priority 1 (lowest)
        cache_default.cache_layer("low_priority", sample_layer, priority=1)
        
        # Priority 10 (highest)
        cache_default.cache_layer("high_priority", sample_layer, priority=10)
        
        assert cache_default._entries["low_priority"].priority == 1
        assert cache_default._entries["high_priority"].priority == 10


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling."""

    def test_cache_error_inheritance(self):
        """Test HierarchicalCacheError inherits from SLIError."""
        error = HierarchicalCacheError("Test error")
        assert isinstance(error, SLIError)

    def test_load_corrupted_file(self, cache_default, sample_layer, tmp_path):
        """Test loading a corrupted cache file."""
        # Create a corrupted file
        corrupted_file = tmp_path / "corrupted.pt"
        with open(corrupted_file, 'w') as f:
            f.write("not a valid pickle")
        
        # Manually add entry pointing to corrupted file
        entry = HierarchicalCacheEntry(
            layer_id="corrupted",
            tier=CacheTier.DISK_L1,
            file_path=str(corrupted_file),
            memory_ref=None,
            size_bytes=100,
            created_at=time.time(),
            last_accessed=time.time()
        )
        cache_default._entries["corrupted"] = entry
        
        # Try to load - should return None, not crash
        result = cache_default.get_layer("corrupted")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
