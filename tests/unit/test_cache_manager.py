"""
tests/unit/test_cache_manager.py
Comprehensive tests for cache management functionality.

Tests cover:
- LRU Cache implementation
- LFU Cache implementation
- TTL Cache implementation
- Cache eviction policies
- Cache statistics
- Model and Dataset specialized caches
- CacheManager with disk persistence
"""

import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.cache_manager import (
    EvictionPolicy,
    CacheEntry,
    CacheStats,
    LRUCache,
    LFUCache,
    TTLCache,
    CacheManager,
    ModelCache,
    DatasetCache,
    get_cache_manager,
    get_model_cache,
    get_dataset_cache,
    cache_get,
    cache_put,
    cache_delete,
    cache_clear,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(key="test", value="data")
        
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.size == 0
        assert entry.access_count == 0
        assert entry.ttl is None
    
    def test_entry_is_expired_no_ttl(self):
        """Test entry without TTL never expires."""
        entry = CacheEntry(key="test", value="data")
        assert entry.is_expired is False
    
    def test_entry_is_expired_with_ttl(self):
        """Test entry expiration with TTL."""
        entry = CacheEntry(key="test", value="data", ttl=0.1)
        assert entry.is_expired is False
        
        time.sleep(0.15)
        assert entry.is_expired is True
    
    def test_entry_touch(self):
        """Test touching updates access metadata."""
        entry = CacheEntry(key="test", value="data")
        initial_access = entry.accessed_at
        initial_count = entry.access_count
        
        time.sleep(0.01)
        entry.touch()
        
        assert entry.accessed_at > initial_access
        assert entry.access_count == initial_count + 1
    
    def test_entry_age(self):
        """Test entry age calculation."""
        entry = CacheEntry(key="test", value="data")
        time.sleep(0.05)
        
        assert entry.age > 0
    
    def test_entry_last_accessed(self):
        """Test last accessed calculation."""
        entry = CacheEntry(key="test", value="data")
        time.sleep(0.05)
        
        assert entry.last_accessed > 0


class TestCacheStats:
    """Test CacheStats class."""
    
    def test_initial_stats(self):
        """Test initial stats are zero."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.insertions == 0
    
    def test_total_requests(self):
        """Test total requests calculation."""
        stats = CacheStats(hits=10, misses=5)
        
        assert stats.total_requests == 15
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        
        assert stats.hit_rate == 75.0
    
    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests."""
        stats = CacheStats()
        
        assert stats.hit_rate == 0.0
    
    def test_miss_rate(self):
        """Test miss rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        
        assert stats.miss_rate == 25.0
    
    def test_reset(self):
        """Test stats reset."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        stats.reset()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0


class TestLRUCache:
    """Test LRU Cache implementation."""
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_get_missing_key(self):
        """Test get returns None for missing key."""
        cache = LRUCache(max_size=10)
        
        assert cache.get("missing") is None
        assert cache.stats.misses == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction order."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_order_update_on_access(self):
        """Test LRU order updates on access."""
        cache = LRUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Move key1 to most recently used
        cache.put("key3", "value3")  # Should evict key2
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
    
    def test_delete(self):
        """Test delete operation."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False
    
    def test_clear(self):
        """Test clear operation."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0
    
    def test_update_existing_key(self):
        """Test updating existing key."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        
        assert cache.get("key1") == "value2"
    
    def test_ttl_expiration(self):
        """Test TTL expiration in LRU cache."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1", ttl=0.1)
        assert cache.get("key1") == "value1"
        
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_keys(self):
        """Test getting all keys."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        keys = cache.keys()
        
        assert "key1" in keys
        assert "key2" in keys
    
    def test_len(self):
        """Test cache length."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert len(cache) == 2


class TestLFUCache:
    """Test LFU Cache implementation."""
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LFUCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_lfu_eviction(self):
        """Test LFU eviction order."""
        cache = LFUCache(max_size=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 multiple times to increase frequency
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")
        
        cache.put("key3", "value3")  # Should evict key2 (lower frequency)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
    
    def test_frequency_tracking(self):
        """Test frequency tracking on access."""
        cache = LFUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        
        assert cache._cache["key1"].access_count == 2
    
    def test_delete(self):
        """Test delete operation."""
        cache = LFUCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
    
    def test_clear(self):
        """Test clear operation."""
        cache = LFUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.clear()
        
        assert cache.get("key1") is None
        assert len(cache._cache) == 0


class TestTTLCache:
    """Test TTL Cache implementation."""
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = TTLCache(default_ttl=3600, max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLCache(default_ttl=0.1, max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_custom_ttl_per_entry(self):
        """Test custom TTL per entry."""
        cache = TTLCache(default_ttl=3600, max_size=10)
        
        cache.put("key1", "value1", ttl=0.1)
        cache.put("key2", "value2", ttl=3600)
        
        time.sleep(0.15)
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_oldest_entry_eviction(self):
        """Test eviction of oldest entry."""
        cache = TTLCache(default_ttl=3600, max_size=2)
        
        cache.put("key1", "value1")
        time.sleep(0.01)
        cache.put("key2", "value2")
        time.sleep(0.01)
        cache.put("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = TTLCache(default_ttl=0.1, max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        time.sleep(0.15)
        # Cleanup happens on put, so add another entry
        cache.put("key3", "value3")
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheManager:
    """Test CacheManager class."""
    
    def test_put_and_get(self):
        """Test basic put and get."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1")
        assert manager.get("key1") == "value1"
    
    def test_delete(self):
        """Test delete operation."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1")
        assert manager.delete("key1") is True
        assert manager.get("key1") is None
    
    def test_clear_all_policies(self):
        """Test clearing all caches."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1", policy=EvictionPolicy.LRU)
        manager.put("key2", "value2", policy=EvictionPolicy.LFU)
        manager.clear()
        
        assert manager.get("key1") is None
        assert manager.get("key2") is None
    
    def test_clear_specific_policy(self):
        """Test clearing specific policy cache."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1", policy=EvictionPolicy.LRU)
        manager.put("key2", "value2", policy=EvictionPolicy.LFU)
        manager.clear(policy=EvictionPolicy.LRU)
        
        assert manager.get("key1") is None
        assert manager.get("key2") == "value2"
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1")
        manager.get("key1")
        manager.get("missing")
        
        stats = manager.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
    
    def test_disk_persistence(self):
        """Test disk persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir, enable_disk_cache=True)
            
            manager.put("key1", "value1", persist=True)
            
            # Create new manager to test loading from disk
            manager2 = CacheManager(cache_dir=tmpdir, enable_disk_cache=True)
            assert manager2.get("key1") == "value1"
    
    def test_different_policies(self):
        """Test different eviction policies."""
        manager = CacheManager(enable_disk_cache=False)
        
        manager.put("key1", "value1", policy=EvictionPolicy.LRU)
        manager.put("key2", "value2", policy=EvictionPolicy.LFU)
        manager.put("key3", "value3", policy=EvictionPolicy.TTL, ttl=3600)
        
        assert manager.get("key1", policy=EvictionPolicy.LRU) == "value1"
        assert manager.get("key2", policy=EvictionPolicy.LFU) == "value2"
        assert manager.get("key3", policy=EvictionPolicy.TTL) == "value3"


class TestModelCache:
    """Test ModelCache specialized cache."""
    
    def test_cache_model(self):
        """Test caching a model."""
        model_cache = ModelCache(max_models=3)
        
        mock_model = {"name": "test_model"}
        model_cache.cache_model("model1", mock_model)
        
        assert model_cache.get_model("model1") == mock_model
    
    def test_evict_model(self):
        """Test evicting a model."""
        model_cache = ModelCache(max_models=3)
        
        mock_model = {"name": "test_model"}
        model_cache.cache_model("model1", mock_model)
        model_cache.evict_model("model1")
        
        assert model_cache.get_model("model1") is None
    
    def test_clear_models(self):
        """Test clearing all models."""
        model_cache = ModelCache(max_models=3)
        
        model_cache.cache_model("model1", {"name": "model1"})
        model_cache.cache_model("model2", {"name": "model2"})
        model_cache.clear_models()
        
        assert model_cache.get_model("model1") is None
        assert model_cache.get_model("model2") is None
    
    def test_model_key_prefix(self):
        """Test that model keys are properly prefixed."""
        model_cache = ModelCache(max_models=3)
        
        mock_model = {"name": "test"}
        model_cache.cache_model("gpt2", mock_model)
        
        # Should not be accessible without prefix
        assert model_cache.cache.get("gpt2") is None
        # Should be accessible through proper method
        assert model_cache.get_model("gpt2") == mock_model


class TestDatasetCache:
    """Test DatasetCache specialized cache."""
    
    def test_cache_dataset(self):
        """Test caching a dataset."""
        dataset_cache = DatasetCache(max_datasets=5)
        
        mock_dataset = {"name": "test_dataset"}
        dataset_cache.cache_dataset("dataset1", mock_dataset)
        
        assert dataset_cache.get_dataset("dataset1") == mock_dataset
    
    def test_cache_dataset_with_split(self):
        """Test caching dataset with split."""
        dataset_cache = DatasetCache(max_datasets=5)
        
        train_data = {"split": "train"}
        test_data = {"split": "test"}
        
        dataset_cache.cache_dataset("dataset1", train_data, split="train")
        dataset_cache.cache_dataset("dataset1", test_data, split="test")
        
        assert dataset_cache.get_dataset("dataset1", "train") == train_data
        assert dataset_cache.get_dataset("dataset1", "test") == test_data
    
    def test_evict_dataset(self):
        """Test evicting a dataset."""
        dataset_cache = DatasetCache(max_datasets=5)
        
        dataset_cache.cache_dataset("dataset1", {"data": "test"})
        dataset_cache.evict_dataset("dataset1")
        
        assert dataset_cache.get_dataset("dataset1") is None
    
    def test_clear_datasets(self):
        """Test clearing all datasets."""
        dataset_cache = DatasetCache(max_datasets=5)
        
        dataset_cache.cache_dataset("dataset1", {"data": "1"})
        dataset_cache.cache_dataset("dataset2", {"data": "2"})
        dataset_cache.clear_datasets()
        
        assert dataset_cache.get_dataset("dataset1") is None
        assert dataset_cache.get_dataset("dataset2") is None


class TestGlobalCacheFunctions:
    """Test global cache functions."""
    
    def test_get_cache_manager_singleton(self):
        """Test get_cache_manager returns singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2
    
    def test_get_model_cache(self):
        """Test get_model_cache returns ModelCache."""
        cache = get_model_cache()
        
        assert isinstance(cache, ModelCache)
    
    def test_get_dataset_cache(self):
        """Test get_dataset_cache returns DatasetCache."""
        cache = get_dataset_cache()
        
        assert isinstance(cache, DatasetCache)
    
    def test_cache_get_put_delete_clear(self):
        """Test global cache convenience functions."""
        # Clear first
        cache_clear()
        
        # Test put and get
        cache_put("key1", "value1")
        assert cache_get("key1") == "value1"
        
        # Test delete
        cache_delete("key1")
        assert cache_get("key1") is None


class TestCacheEvictionPolicies:
    """Test eviction policy behavior."""
    
    def test_lru_policy(self):
        """Test LRU policy enum."""
        assert EvictionPolicy.LRU.value == "lru"
    
    def test_lfu_policy(self):
        """Test LFU policy enum."""
        assert EvictionPolicy.LFU.value == "lfu"
    
    def test_ttl_policy(self):
        """Test TTL policy enum."""
        assert EvictionPolicy.TTL.value == "ttl"
    
    def test_size_policy(self):
        """Test SIZE policy enum."""
        assert EvictionPolicy.SIZE.value == "size"
    
    def test_priority_policy(self):
        """Test PRIORITY policy enum."""
        assert EvictionPolicy.PRIORITY.value == "priority"


class TestCacheEdgeCases:
    """Test cache edge cases."""
    
    def test_put_large_value(self):
        """Test handling of large values."""
        cache = LRUCache(max_size=10, max_memory_mb=1)
        
        # Try to put value larger than max memory
        large_value = "x" * (2 * 1024 * 1024)  # 2MB string
        result = cache.put("key1", large_value)
        
        # Should fail or handle gracefully
        assert result is False or cache.get("key1") is None
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading
        
        cache = LRUCache(max_size=100)
        errors = []
        
        def worker(n):
            try:
                for i in range(10):
                    cache.put(f"key_{n}_{i}", f"value_{i}")
                    cache.get(f"key_{n}_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_empty_string_key(self):
        """Test empty string key handling."""
        cache = LRUCache(max_size=10)
        
        cache.put("", "value")
        assert cache.get("") == "value"
    
    def test_none_value(self):
        """Test None value handling."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", None)
        assert cache.get("key1") is None  # Can't distinguish from missing
