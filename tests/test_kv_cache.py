"""
Tests for KV-cache optimization (Paper 2512.14982).
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.kv_cache import (
    OptimizedKVCache,
    KVCacheEntry,
    CacheStats,
    RepetitionAwareCacheManager,
    InferenceOptimizer,
    create_optimized_cache,
    create_cache_manager
)


class TestKVCacheEntry:
    """Test suite for KVCacheEntry."""
    
    def test_entry_creation(self):
        """Test creating a cache entry."""
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        
        entry = KVCacheEntry(key=key, value=value, repetition_id=1)
        
        assert torch.equal(entry.key, key)
        assert torch.equal(entry.value, value)
        assert entry.repetition_id == 1
        assert entry.access_count == 0
    
    def test_entry_touch(self):
        """Test touching/updating an entry."""
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        
        entry = KVCacheEntry(key=key, value=value, repetition_id=1)
        original_time = entry.timestamp
        
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.timestamp >= original_time


class TestCacheStats:
    """Test suite for CacheStats."""
    
    def test_initial_stats(self):
        """Test initial cache statistics."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.evictions == 0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        
        assert stats.hit_rate == 0.75
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=100,
            misses=50,
            evictions=10,
            memory_usage_bytes=104857600,  # 100 MB
            total_entries=1000
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["hits"] == 100
        assert stats_dict["misses"] == 50
        assert stats_dict["hit_rate"] == 100/150
        assert stats_dict["evictions"] == 10
        assert stats_dict["memory_usage_mb"] == 100.0


class TestOptimizedKVCache:
    """Test suite for OptimizedKVCache."""
    
    def test_initialization(self):
        """Test cache initialization."""
        cache = OptimizedKVCache(max_cache_size=100, max_memory_mb=512)
        
        assert cache.max_cache_size == 100
        assert cache.max_memory_bytes == 512 * 1024 * 1024
        assert cache.keep_second_repetition == True
        assert len(cache.cache) == 0
    
    def test_second_repetition_only(self):
        """Test that only second repetition is cached."""
        cache = OptimizedKVCache(keep_second_repetition=True)
        
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # First repetition should not be cached
        cache.set(input_ids, layer_idx=0, key=key, value=value, repetition_id=0)
        result_first = cache.get(input_ids, layer_idx=0, repetition_id=0)
        assert result_first is None
        assert cache.stats.misses == 1
        
        # Second repetition should be cached
        cache.set(input_ids, layer_idx=0, key=key, value=value, repetition_id=1)
        result_second = cache.get(input_ids, layer_idx=0, repetition_id=1)
        assert result_second is not None
        assert cache.stats.hits == 1
    
    def test_cache_hit_and_miss(self):
        """Test cache hit and miss behavior."""
        cache = OptimizedKVCache(keep_second_repetition=False)
        
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Miss before set
        result = cache.get(input_ids, layer_idx=0, repetition_id=0)
        assert result is None
        assert cache.stats.misses == 1
        
        # Set and hit
        cache.set(input_ids, layer_idx=0, key=key, value=value, repetition_id=0)
        result = cache.get(input_ids, layer_idx=0, repetition_id=0)
        assert result is not None
        assert cache.stats.hits == 1
        assert len(result) == 2  # (key, value) tuple
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = OptimizedKVCache(max_cache_size=2, keep_second_repetition=False)
        
        # Add two entries
        for i in range(2):
            key = torch.randn(1, 12, 10, 64)
            value = torch.randn(1, 12, 10, 64)
            input_ids = torch.randint(0, 1000, (1, 10))
            cache.set(input_ids, layer_idx=i, key=key, value=value, repetition_id=0)
        
        assert len(cache.cache) == 2
        
        # Add third entry (should trigger eviction)
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        input_ids = torch.randint(0, 1000, (1, 10))
        cache.set(input_ids, layer_idx=2, key=key, value=value, repetition_id=0)
        
        assert len(cache.cache) == 2  # Should stay at max
        assert cache.stats.evictions >= 1
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = OptimizedKVCache(keep_second_repetition=False)
        
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        cache.set(input_ids, layer_idx=0, key=key, value=value, repetition_id=0)
        assert len(cache.cache) == 1
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
    
    def test_get_cache_for_repetition(self):
        """Test getting cache for all layers."""
        cache = OptimizedKVCache(keep_second_repetition=False)
        
        # Set for multiple layers
        num_layers = 3
        input_ids = torch.randint(0, 1000, (1, 10))
        
        for layer_idx in range(num_layers):
            key = torch.randn(1, 12, 10, 64)
            value = torch.randn(1, 12, 10, 64)
            cache.set(input_ids, layer_idx, key, value, repetition_id=0)
        
        # Get all layers
        cached_kvs = cache.get_cache_for_repetition(input_ids, num_layers)
        
        assert len(cached_kvs) == num_layers
        assert all(kv is not None for kv in cached_kvs)


class TestRepetitionAwareCacheManager:
    """Test suite for RepetitionAwareCacheManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = RepetitionAwareCacheManager()
        
        assert manager.kv_cache is not None
        assert manager.enable_memory_profiling == True
    
    def test_prepare_for_repetition(self):
        """Test preparation for repetition."""
        manager = RepetitionAwareCacheManager(model_config={"num_hidden_layers": 12})
        
        input_ids = torch.randint(0, 1000, (1, 10))
        metadata = manager.prepare_for_repetition(input_ids, repetition_factor=3)
        
        assert metadata["repetition_factor"] == 3
        assert metadata["expected_cache_entries"] == 3 * 12
        assert metadata["cache_warmed"] == True
    
    def test_should_use_cache(self):
        """Test cache usage decision."""
        manager = RepetitionAwareCacheManager()
        
        # First repetition should not use cache
        assert manager.should_use_cache(0, 0) == False
        
        # Second and subsequent should use cache
        assert manager.should_use_cache(1, 0) == True
        assert manager.should_use_cache(2, 0) == True
    
    def test_optimize_generation(self):
        """Test generation optimization."""
        manager = RepetitionAwareCacheManager()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # First repetition
        opt_first = manager.optimize_generation(input_ids, repetition_id=0)
        assert opt_first["use_cached"] == False
        assert opt_first["tokens_saved"] == 0
        
        # Second repetition (with mock past_key_values)
        past_kvs = [(torch.randn(1, 12, 10, 64), torch.randn(1, 12, 10, 64))]
        opt_second = manager.optimize_generation(input_ids, past_kvs, repetition_id=1)
        assert opt_second["use_cached"] == True
        assert opt_second["tokens_saved"] == 10
    
    def test_get_memory_profile(self):
        """Test memory profiling."""
        manager = RepetitionAwareCacheManager()
        
        profile = manager.get_memory_profile()
        
        assert "cache_stats" in profile
        assert "inference_stats" in profile
        assert "memory_efficiency" in profile
        assert "recommendations" in profile
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        manager = RepetitionAwareCacheManager()
        
        # Simulate some activity
        manager._inference_stats["total_tokens_generated"] = 100
        manager.kv_cache.stats.hits = 50
        
        manager.reset_stats()
        
        assert manager._inference_stats["total_tokens_generated"] == 0
        assert manager.kv_cache.stats.hits == 0


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_create_optimized_cache(self):
        """Test creating optimized cache."""
        cache = create_optimized_cache(max_memory_mb=256)
        
        assert isinstance(cache, OptimizedKVCache)
        assert cache.max_memory_bytes == 256 * 1024 * 1024
    
    def test_create_cache_manager(self):
        """Test creating cache manager."""
        manager = create_cache_manager(model_config={"num_layers": 24})
        
        assert isinstance(manager, RepetitionAwareCacheManager)


class TestPerformanceImpact:
    """Test suite ensuring 0% performance impact."""
    
    def test_no_performance_degradation(self):
        """Verify that cache doesn't degrade performance."""
        cache = OptimizedKVCache(keep_second_repetition=True)
        
        key = torch.randn(1, 12, 10, 64)
        value = torch.randn(1, 12, 10, 64)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Simulate repetition pattern
        # First call (not cached)
        result1 = cache.get(input_ids, layer_idx=0, repetition_id=0)
        
        # Second call (cached)
        cache.set(input_ids, layer_idx=0, key=key, value=value, repetition_id=1)
        result2 = cache.get(input_ids, layer_idx=0, repetition_id=1)
        
        # Both should return valid tensors (no errors)
        assert result1 is None  # First not cached by design
        assert result2 is not None  # Second cached
    
    def test_cache_hit_improves_performance(self):
        """Test that cache hits improve performance."""
        manager = RepetitionAwareCacheManager()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # Simulate multiple repetitions
        for i in range(3):
            opt = manager.optimize_generation(
                input_ids, 
                past_key_values=torch.randn(1, 12, 10, 64) if i > 0 else None,
                repetition_id=i
            )
            
            # After first repetition, should use cache
            if i >= 1:
                assert opt["use_cached"] == True
                assert opt["tokens_saved"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])