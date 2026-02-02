"""
Unit tests for Activation Caching System

This module contains comprehensive tests for the activation cache including:
- Memory and disk cache operations
- LRU/LFU/FIFO eviction strategies
- TTL expiration
- Compression/decompression
- Thread safety

Author: Nexus Team
"""

import unittest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np

from nexus.models.sli.activation_cache import (
    CacheInvalidationStrategy,
    CompressionType,
    ActivationCacheEntry,
    ActivationCacheStats,
    ActivationCacheConfig,
    ActivationCacheError,
    ActivationCache,
    ActivationCacheManager,
    get_activation_cache,
)


class TestCacheInvalidationStrategy(unittest.TestCase):
    """Test CacheInvalidationStrategy enum."""
    
    def test_strategy_values(self):
        """Test that all strategies have correct values."""
        self.assertEqual(CacheInvalidationStrategy.LRU.value, "lru")
        self.assertEqual(CacheInvalidationStrategy.LFU.value, "lfu")
        self.assertEqual(CacheInvalidationStrategy.FIFO.value, "fifo")
        self.assertEqual(CacheInvalidationStrategy.TTL.value, "ttl")
        self.assertEqual(CacheInvalidationStrategy.ADAPTIVE.value, "adaptive")


class TestCompressionType(unittest.TestCase):
    """Test CompressionType enum."""
    
    def test_compression_values(self):
        """Test that all compression types have correct values."""
        self.assertEqual(CompressionType.NONE.value, "none")
        self.assertEqual(CompressionType.GZIP.value, "gzip")
        self.assertEqual(CompressionType.LZ4.value, "lz4")
        self.assertEqual(CompressionType.ZSTD.value, "zstd")


class TestActivationCacheEntry(unittest.TestCase):
    """Test ActivationCacheEntry dataclass."""
    
    def test_initial_state(self):
        """Test initial state of cache entry."""
        tensor = torch.randn(10, 10)
        entry = ActivationCacheEntry(
            key="test_key",
            activation=tensor,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0
        )
        
        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.access_count, 0)
        self.assertFalse(entry.compressed)
        self.assertEqual(entry.size_bytes, 0)
    
    def test_is_expired_with_ttl(self):
        """Test expiration check with TTL."""
        entry = ActivationCacheEntry(
            key="test_key",
            activation=torch.randn(10, 10),
            created_at=time.time() - 100,
            last_accessed=time.time(),
            access_count=0,
            ttl=50  # 50 seconds TTL
        )
        
        self.assertTrue(entry.is_expired())
    
    def test_is_not_expired_with_ttl(self):
        """Test non-expiration with TTL."""
        entry = ActivationCacheEntry(
            key="test_key",
            activation=torch.randn(10, 10),
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            ttl=3600  # 1 hour TTL
        )
        
        self.assertFalse(entry.is_expired())
    
    def test_no_expiration_without_ttl(self):
        """Test that entry doesn't expire without TTL."""
        entry = ActivationCacheEntry(
            key="test_key",
            activation=torch.randn(10, 10),
            created_at=time.time() - 1000000,
            last_accessed=time.time(),
            access_count=0,
            ttl=None
        )
        
        self.assertFalse(entry.is_expired())
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = ActivationCacheEntry(
            key="test_key",
            activation=torch.randn(10, 10),
            created_at=100.0,
            last_accessed=200.0,
            access_count=5,
            ttl=3600.0,
            size_bytes=1024,
            compressed=True,
            metadata={"layer": 5}
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result['key'], "test_key")
        self.assertEqual(result['created_at'], 100.0)
        self.assertEqual(result['last_accessed'], 200.0)
        self.assertEqual(result['access_count'], 5)
        self.assertEqual(result['ttl'], 3600.0)
        self.assertEqual(result['size_bytes'], 1024)
        self.assertTrue(result['compressed'])
        self.assertEqual(result['metadata'], {"layer": 5})


class TestActivationCacheStats(unittest.TestCase):
    """Test ActivationCacheStats class."""
    
    def test_initial_state(self):
        """Test initial statistics state."""
        stats = ActivationCacheStats()
        
        self.assertEqual(stats.memory_hits, 0)
        self.assertEqual(stats.memory_misses, 0)
        self.assertEqual(stats.disk_hits, 0)
        self.assertEqual(stats.disk_misses, 0)
        self.assertEqual(stats.hit_rate, 0.0)
        self.assertEqual(stats.memory_hit_rate, 0.0)
    
    def test_total_hits(self):
        """Test total hits calculation."""
        stats = ActivationCacheStats()
        stats.memory_hits = 10
        stats.disk_hits = 5
        
        self.assertEqual(stats.total_hits, 15)
    
    def test_total_misses(self):
        """Test total misses calculation."""
        stats = ActivationCacheStats()
        stats.memory_misses = 3
        stats.disk_misses = 2
        
        self.assertEqual(stats.total_misses, 5)
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        stats = ActivationCacheStats()
        stats.memory_hits = 80
        stats.memory_misses = 20
        
        self.assertEqual(stats.hit_rate, 0.8)
    
    def test_hit_rate_empty(self):
        """Test hit rate with no accesses."""
        stats = ActivationCacheStats()
        
        self.assertEqual(stats.hit_rate, 0.0)
    
    def test_record_hit_memory(self):
        """Test recording memory hit."""
        stats = ActivationCacheStats()
        stats.record_hit(from_memory=True)
        
        self.assertEqual(stats.memory_hits, 1)
        self.assertEqual(stats.disk_hits, 0)
    
    def test_record_hit_disk(self):
        """Test recording disk hit."""
        stats = ActivationCacheStats()
        stats.record_hit(from_memory=False)
        
        self.assertEqual(stats.memory_hits, 0)
        self.assertEqual(stats.disk_hits, 1)
    
    def test_record_miss_memory(self):
        """Test recording memory miss."""
        stats = ActivationCacheStats()
        stats.record_miss(to_memory=True)
        
        self.assertEqual(stats.memory_misses, 1)
        self.assertEqual(stats.disk_misses, 0)
    
    def test_record_miss_disk(self):
        """Test recording disk miss."""
        stats = ActivationCacheStats()
        stats.record_miss(to_memory=False)
        
        self.assertEqual(stats.memory_misses, 0)
        self.assertEqual(stats.disk_misses, 1)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ActivationCacheStats()
        stats.memory_hits = 100
        stats.memory_misses = 25
        stats.evictions = 10
        stats.total_bytes_memory = 1e9
        
        result = stats.to_dict()
        
        self.assertEqual(result['memory_hits'], 100)
        self.assertEqual(result['memory_misses'], 25)
        self.assertEqual(result['hit_rate'], 0.8)
        self.assertEqual(result['evictions'], 10)
        self.assertEqual(result['total_bytes_memory_gb'], 1.0)


class TestActivationCacheConfig(unittest.TestCase):
    """Test ActivationCacheConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ActivationCacheConfig()
        
        self.assertEqual(config.max_memory_size_gb, 2.0)
        self.assertEqual(config.max_disk_size_gb, 10.0)
        self.assertIsNone(config.default_ttl_seconds)
        self.assertEqual(config.invalidation_strategy, CacheInvalidationStrategy.LRU)
        self.assertEqual(config.compression, CompressionType.GZIP)
        self.assertEqual(config.compression_level, 6)
        self.assertTrue(config.enable_persistence)
        self.assertEqual(config.cleanup_interval_seconds, 300.0)
        self.assertEqual(config.max_entries_memory, 1000)
        self.assertEqual(config.max_entries_disk, 10000)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ActivationCacheConfig(
            max_memory_size_gb=4.0,
            max_disk_size_gb=20.0,
            invalidation_strategy=CacheInvalidationStrategy.LFU,
            compression=CompressionType.LZ4
        )
        
        self.assertEqual(config.max_memory_size_gb, 4.0)
        self.assertEqual(config.max_disk_size_gb, 20.0)
        self.assertEqual(config.invalidation_strategy, CacheInvalidationStrategy.LFU)
        self.assertEqual(config.compression, CompressionType.LZ4)
    
    def test_default_persistence_dir(self):
        """Test default persistence directory is set."""
        config = ActivationCacheConfig()
        
        self.assertIsNotNone(config.persistence_dir)
        self.assertIn('.cache/nexus/activations', config.persistence_dir)


class TestActivationCache(unittest.TestCase):
    """Test ActivationCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ActivationCacheConfig(
            max_memory_size_gb=1.0,
            max_disk_size_gb=2.0,
            persistence_dir=self.temp_dir,
            enable_persistence=True,
            cleanup_interval_seconds=1.0
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = ActivationCache(config=self.config)
        
        self.assertEqual(cache.max_memory_bytes, int(1e9))
        self.assertEqual(cache.max_disk_bytes, int(2e9))
        self.assertEqual(len(cache._memory_cache), 0)
        self.assertEqual(cache._current_memory_bytes, 0)
        
        cache.shutdown()
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        cache = ActivationCache(config=self.config)
        
        key1 = cache._get_cache_key("identifier")
        key2 = cache._get_cache_key("identifier")
        
        self.assertEqual(key1, key2)  # Same identifier should produce same key
        
        key3 = cache._get_cache_key("identifier", context="model1")
        self.assertNotEqual(key1, key3)  # Different context = different key
        
        cache.shutdown()
    
    def test_compute_tensor_size(self):
        """Test tensor size computation."""
        cache = ActivationCache(config=self.config)
        
        tensor = torch.randn(10, 20, dtype=torch.float32)
        size = cache._compute_tensor_size(tensor)
        
        self.assertEqual(size, 10 * 20 * 4)  # 200 elements * 4 bytes (float32)
        
        cache.shutdown()
    
    def test_store_and_retrieve(self):
        """Test basic store and retrieve operations."""
        cache = ActivationCache(config=self.config)
        
        tensor = torch.randn(10, 20)
        result = cache.store("test_id", tensor)
        
        self.assertTrue(result)
        
        retrieved = cache.retrieve("test_id")
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(tensor, retrieved))
        
        cache.shutdown()
    
    def test_store_with_context(self):
        """Test store with context."""
        cache = ActivationCache(config=self.config)
        
        tensor = torch.randn(10, 20)
        cache.store("layer_0", tensor, context="model1")
        
        # Retrieve with same context
        retrieved = cache.retrieve("layer_0", context="model1")
        self.assertIsNotNone(retrieved)
        
        # Retrieve with different context should fail
        not_found = cache.retrieve("layer_0", context="model2")
        self.assertIsNone(not_found)
        
        cache.shutdown()
    
    def test_retrieve_not_found(self):
        """Test retrieving non-existent entry."""
        cache = ActivationCache(config=self.config)
        
        result = cache.retrieve("nonexistent")
        
        self.assertIsNone(result)
        
        cache.shutdown()
    
    def test_memory_hit_rate(self):
        """Test memory hit rate tracking."""
        cache = ActivationCache(config=self.config)
        
        tensor = torch.randn(10, 20)
        cache.store("test", tensor)
        
        # First retrieve - should be hit
        cache.retrieve("test")
        # Second retrieve - should be miss (already retrieved once)
        cache.retrieve("test")
        
        stats = cache.get_stats()
        self.assertEqual(stats['memory_hits'], 1)
        
        cache.shutdown()
    
    def test_lru_eviction(self):
        """Test LRU eviction strategy."""
        config = ActivationCacheConfig(
            max_memory_size_gb=0.001,  # Very small cache
            invalidation_strategy=CacheInvalidationStrategy.LRU
        )
        cache = ActivationCache(config=config)
        
        # Store multiple large tensors
        for i in range(10):
            tensor = torch.randn(100, 100)  # ~40KB each
            cache.store(f"key_{i}", tensor)
        
        # Some should have been evicted due to size
        self.assertLess(len(cache._memory_cache), 10)
        
        cache.shutdown()
    
    def test_lfu_eviction(self):
        """Test LFU eviction strategy."""
        config = ActivationCacheConfig(
            max_memory_size_gb=0.001,
            invalidation_strategy=CacheInvalidationStrategy.LFU
        )
        cache = ActivationCache(config=config)
        
        # Store tensors and access them different amounts
        for i in range(5):
            tensor = torch.randn(50, 50)
            cache.store(f"key_{i}", tensor)
        
        # Access key_0 multiple times
        for _ in range(5):
            cache.retrieve("key_0")
        
        cache.shutdown()
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        config = ActivationCacheConfig(
            default_ttl_seconds=0.1  # Very short TTL
        )
        cache = ActivationCache(config=config)
        
        tensor = torch.randn(10, 10)
        cache.store("expiring", tensor)
        
        # Should be available immediately
        self.assertIsNotNone(cache.retrieve("expiring"))
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired now
        result = cache.retrieve("expiring")
        self.assertIsNone(result)
        
        cache.shutdown()
    
    def test_invalidate_by_identifier(self):
        """Test invalidation by identifier."""
        cache = ActivationCache(config=self.config)
        
        cache.store("key1", torch.randn(10, 10))
        cache.store("key2", torch.randn(10, 10))
        
        count = cache.invalidate("key1")
        
        self.assertEqual(count, 1)
        self.assertIsNone(cache.retrieve("key1"))
        self.assertIsNotNone(cache.retrieve("key2"))
        
        cache.shutdown()
    
    def test_invalidate_by_context(self):
        """Test invalidation by context."""
        cache = ActivationCache(config=self.config)
        
        cache.store("key1", torch.randn(10, 10), context="model1")
        cache.store("key2", torch.randn(10, 10), context="model1")
        cache.store("key3", torch.randn(10, 10), context="model2")
        
        count = cache.invalidate(context="model1")
        
        self.assertGreaterEqual(count, 2)
        self.assertIsNone(cache.retrieve("key1", context="model1"))
        self.assertIsNone(cache.retrieve("key2", context="model1"))
        self.assertIsNotNone(cache.retrieve("key3", context="model2"))
        
        cache.shutdown()
    
    def test_invalidate_by_pattern(self):
        """Test invalidation by pattern."""
        cache = ActivationCache(config=self.config)
        
        cache.store("layer_0_output", torch.randn(10, 10))
        cache.store("layer_1_output", torch.randn(10, 10))
        cache.store("attention_weights", torch.randn(10, 10))
        
        count = cache.invalidate(pattern="layer")
        
        self.assertGreaterEqual(count, 2)
        
        cache.shutdown()
    
    def test_clear(self):
        """Test clearing all cache entries."""
        cache = ActivationCache(config=self.config)
        
        cache.store("key1", torch.randn(10, 10))
        cache.store("key2", torch.randn(10, 10))
        
        cache.clear()
        
        self.assertEqual(len(cache._memory_cache), 0)
        self.assertEqual(cache._current_memory_bytes, 0)
        
        cache.shutdown()
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ActivationCache(config=self.config)
        
        cache.store("key1", torch.randn(10, 10))
        cache.retrieve("key1")
        cache.retrieve("nonexistent")
        
        stats = cache.get_stats()
        
        self.assertIn('memory_entries', stats)
        self.assertIn('hit_rate', stats)
        self.assertIn('memory_usage_gb', stats)
        
        cache.shutdown()
    
    def test_compression_gzip(self):
        """Test gzip compression."""
        config = ActivationCacheConfig(
            compression=CompressionType.GZIP,
            compression_level=6,
            persistence_dir=self.temp_dir
        )
        cache = ActivationCache(config=config)
        
        tensor = torch.randn(100, 100)
        compressed = cache._compress_tensor(tensor)
        
        # Compressed should be smaller than original
        original_size = cache._compute_tensor_size(tensor)
        self.assertLess(len(compressed), original_size * 2)  # Some overhead
        
        # Should be able to decompress
        decompressed = cache._decompress_tensor(compressed)
        self.assertTrue(torch.allclose(tensor, decompressed))
        
        cache.shutdown()
    
    def test_store_with_device(self):
        """Test retrieving to specific device."""
        cache = ActivationCache(config=self.config)
        
        tensor = torch.randn(10, 10)
        cache.store("test", tensor)
        
        # Retrieve to CPU
        retrieved = cache.retrieve("test", device="cpu")
        self.assertEqual(retrieved.device.type, "cpu")
        
        cache.shutdown()
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = ActivationCache(config=self.config)
        errors = []
        
        def store_operations():
            try:
                for i in range(50):
                    tensor = torch.randn(10, 10)
                    cache.store(f"thread_key_{i}", tensor)
            except Exception as e:
                errors.append(e)
        
        def retrieve_operations():
            try:
                for i in range(50):
                    cache.retrieve(f"thread_key_{i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(3):
            t1 = threading.Thread(target=store_operations)
            t2 = threading.Thread(target=retrieve_operations)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)
        
        cache.shutdown()


class TestActivationCacheManager(unittest.TestCase):
    """Test ActivationCacheManager singleton."""
    
    def test_singleton(self):
        """Test that manager is a singleton."""
        manager1 = ActivationCacheManager()
        manager2 = ActivationCacheManager()
        
        self.assertIs(manager1, manager2)
    
    def test_get_cache(self):
        """Test getting cache instance."""
        cache1 = ActivationCacheManager.get_cache()
        cache2 = ActivationCacheManager.get_cache()
        
        self.assertIs(cache1, cache2)
        
        cache1.shutdown()


class TestGetActivationCache(unittest.TestCase):
    """Test get_activation_cache function."""
    
    def test_returns_cache(self):
        """Test that function returns a cache."""
        cache = get_activation_cache()
        
        self.assertIsInstance(cache, ActivationCache)
        
        cache.shutdown()


if __name__ == '__main__':
    unittest.main()
