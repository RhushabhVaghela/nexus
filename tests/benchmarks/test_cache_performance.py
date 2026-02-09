"""
Cache Performance Benchmarks

Benchmarks for Activation Cache performance:
- Memory hit rate benchmarks
- Disk cache performance
- Compression ratio benchmarks
- LRU/LFU eviction benchmarks
- Multi-tier caching performance

Usage:
    python -m pytest tests/benchmarks/test_cache_performance.py -v

Author: Nexus Team
"""

import pytest
import time
import tempfile
import statistics
from typing import Dict, List, Tuple
import torch

from src.models.sli.activation_cache import (
    ActivationCache,
    ActivationCacheConfig,
    CacheInvalidationStrategy,
    CompressionType,
)


def benchmark_hit_rate(
    cache_size_gb: float = 0.5,
    num_operations: int = 1000,
    hit_ratio: float = 0.8
) -> Dict[str, float]:
    """
    Benchmark cache hit rate performance.
    
    Args:
        cache_size_gb: Cache size in GB
        num_operations: Number of operations to perform
        hit_ratio: Target hit ratio
    
    Returns:
        Dictionary with hit rate statistics
    """
    config = ActivationCacheConfig(
        max_memory_size_gb=cache_size_gb,
        enable_persistence=False
    )
    cache = ActivationCache(config=config)
    
    # Pre-populate cache to achieve target hit ratio
    num_unique = int(num_operations * (1 - hit_ratio))
    for i in range(num_unique):
        tensor = torch.randn(100, 100)
        cache.store(f"key_{i}", tensor)
    
    # Perform mixed operations
    hits = 0
    misses = 0
    
    start = time.time()
    for i in range(num_operations):
        if i % 100 < int(hit_ratio * 100):
            # Hit (access existing key)
            key_idx = i % num_unique
            result = cache.retrieve(f"key_{key_idx}")
            if result is not None:
                hits += 1
            else:
                misses += 1
        else:
            # Miss (access new key or store)
            key_idx = num_unique + (i % 50)
            if cache.retrieve(f"key_{key_idx}") is not None:
                hits += 1
            else:
                tensor = torch.randn(100, 100)
                cache.store(f"new_key_{i}", tensor)
                misses += 1
    
    elapsed = time.time() - start
    
    stats = cache.get_stats()
    cache.shutdown()
    
    return {
        'actual_hit_rate': stats['hit_rate'],
        'operations_per_second': num_operations / elapsed,
        'avg_operation_time_ms': (elapsed / num_operations) * 1000,
        'total_hits': stats['total_hits'],
        'total_misses': stats['total_misses'],
    }


def benchmark_compression_ratio(
    compression_type: CompressionType = CompressionType.GZIP,
    sizes: List[Tuple[int, int]] = [(100, 100), (500, 500), (1000, 1000)]
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark compression ratios for different tensor sizes.
    
    Returns:
        Dictionary with compression statistics per size
    """
    config = ActivationCacheConfig(compression=compression_type)
    cache = ActivationCache(config=config)
    
    results = {}
    
    for rows, cols in sizes:
        tensor = torch.randn(rows, cols)
        original_size = cache._compute_tensor_size(tensor)
        
        start = time.time()
        compressed = cache._compress_tensor(tensor)
        compression_time = (time.time() - start) * 1000
        
        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Verify decompression
        start = time.time()
        decompressed = cache._decompress_tensor(compressed)
        decompression_time = (time.time() - start) * 1000
        
        # Verify integrity
        valid = torch.allclose(tensor, decompressed, atol=1e-5)
        
        results[f"{rows}x{cols}"] = {
            'original_bytes': original_size,
            'compressed_bytes': compressed_size,
            'compression_ratio': ratio,
            'space_saved_percent': (1 - 1/ratio) * 100,
            'compression_time_ms': compression_time,
            'decompression_time_ms': decompression_time,
            'valid': valid,
        }
    
    cache.shutdown()
    return results


def benchmark_eviction_strategies(
    num_entries: int = 100,
    access_pattern: str = "random"
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different eviction strategies.
    
    Returns:
        Dictionary with results per strategy
    """
    strategies = [
        CacheInvalidationStrategy.LRU,
        CacheInvalidationStrategy.LFU,
        CacheInvalidationStrategy.FIFO,
    ]
    
    results = {}
    
    for strategy in strategies:
        config = ActivationCacheConfig(
            max_memory_size_gb=0.01,  # Very small cache to force evictions
            invalidation_strategy=strategy,
            max_entries_memory=20,  # Force eviction after 20 entries
            enable_persistence=False
        )
        cache = ActivationCache(config=config)
        
        # Populate cache
        for i in range(num_entries):
            tensor = torch.randn(50, 50)
            cache.store(f"key_{i}", tensor)
        
        # Access based on pattern
        if access_pattern == "random":
            import random
            for _ in range(50):
                key_idx = random.randint(0, num_entries - 1)
                cache.retrieve(f"key_{key_idx}")
        elif access_pattern == "sequential":
            for i in range(50):
                cache.retrieve(f"key_{i % num_entries}")
        
        stats = cache.get_stats()
        cache.shutdown()
        
        results[strategy.value] = {
            'final_entries': stats['memory_entries'],
            'evictions': stats['evictions'],
            'hit_rate': stats['hit_rate'],
        }
    
    return results


def benchmark_disk_cache_performance(
    num_entries: int = 50,
    entry_size: Tuple[int, int] = (200, 200)
) -> Dict[str, float]:
    """
    Benchmark disk cache read/write performance.
    
    Returns:
        Dictionary with timing results
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ActivationCacheConfig(
            max_memory_size_gb=0.01,  # Small memory cache
            max_disk_size_gb=1.0,
            enable_persistence=True,
            persistence_dir=temp_dir,
            compression=CompressionType.GZIP
        )
        cache = ActivationCache(config=config)
        
        rows, cols = entry_size
        
        # Write to disk
        write_times = []
        for i in range(num_entries):
            tensor = torch.randn(rows, cols)
            start = time.time()
            cache.store(f"disk_key_{i}", tensor, persist=True)
            write_times.append((time.time() - start) * 1000)
        
        # Clear memory cache to force disk reads
        cache._memory_cache.clear()
        cache._current_memory_bytes = 0
        
        # Read from disk
        read_times = []
        for i in range(num_entries):
            start = time.time()
            result = cache.retrieve(f"disk_key_{i}")
            read_times.append((time.time() - start) * 1000)
            assert result is not None, f"Failed to retrieve disk_key_{i}"
        
        cache.shutdown()
        
        return {
            'avg_write_time_ms': statistics.mean(write_times),
            'avg_read_time_ms': statistics.mean(read_times),
            'max_write_time_ms': max(write_times),
            'max_read_time_ms': max(read_times),
            'total_entries': num_entries,
        }


def benchmark_memory_overhead(
    num_entries: int = 1000
) -> Dict[str, float]:
    """
    Benchmark memory overhead of cache infrastructure.
    
    Returns:
        Dictionary with memory statistics
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline_memory = process.memory_info().rss
    
    config = ActivationCacheConfig(
        max_memory_size_gb=10.0,
        enable_persistence=False
    )
    cache = ActivationCache(config=config)
    
    # Store entries
    tensor_size = 100 * 100 * 4  # float32
    for i in range(num_entries):
        tensor = torch.randn(100, 100)
        cache.store(f"key_{i}", tensor)
    
    # Measure memory
    current_memory = process.memory_info().rss
    cache_memory = current_memory - baseline_memory
    
    stats = cache.get_stats()
    cache.shutdown()
    
    data_memory = num_entries * tensor_size
    overhead = cache_memory - data_memory
    overhead_percent = (overhead / data_memory) * 100 if data_memory > 0 else 0
    
    return {
        'baseline_memory_mb': baseline_memory / (1024 * 1024),
        'cache_memory_mb': cache_memory / (1024 * 1024),
        'data_memory_mb': data_memory / (1024 * 1024),
        'overhead_mb': overhead / (1024 * 1024),
        'overhead_percent': overhead_percent,
    }


# Pytest benchmark tests
@pytest.mark.benchmark
class TestCacheBenchmarks:
    """Cache performance benchmark tests."""
    
    def test_hit_rate_80_percent(self):
        """Benchmark 80% hit rate scenario."""
        results = benchmark_hit_rate(
            cache_size_gb=0.5,
            num_operations=500,
            hit_ratio=0.8
        )
        
        print(f"\n80% Hit Rate Benchmark:")
        print(f"  Actual hit rate: {results['actual_hit_rate']:.2%}")
        print(f"  Operations/sec: {results['operations_per_second']:.0f}")
        print(f"  Avg operation time: {results['avg_operation_time_ms']:.3f} ms")
        
        assert results['actual_hit_rate'] >= 0.7, "Hit rate should be near target"
    
    def test_compression_ratios(self):
        """Benchmark compression ratios."""
        results = benchmark_compression_ratio(
            compression_type=CompressionType.GZIP,
            sizes=[(100, 100), (500, 500)]
        )
        
        print(f"\nCompression Ratios:")
        for size, stats in results.items():
            print(f"  {size}:")
            print(f"    Ratio: {stats['compression_ratio']:.2f}x")
            print(f"    Space saved: {stats['space_saved_percent']:.1f}%")
            print(f"    Valid: {stats['valid']}")
            
            assert stats['valid'], f"Compression/decompression should preserve data for {size}"
            assert stats['compression_ratio'] > 1.0, f"Should achieve compression for {size}"
    
    def test_eviction_strategies(self):
        """Benchmark different eviction strategies."""
        results = benchmark_eviction_strategies(
            num_entries=50,
            access_pattern="random"
        )
        
        print(f"\nEviction Strategies:")
        for strategy, stats in results.items():
            print(f"  {strategy}:")
            print(f"    Entries: {stats['final_entries']}")
            print(f"    Evictions: {stats['evictions']}")
            print(f"    Hit rate: {stats['hit_rate']:.2%}")
    
    def test_disk_cache_performance(self):
        """Benchmark disk cache read/write."""
        results = benchmark_disk_cache_performance(
            num_entries=20,
            entry_size=(100, 100)
        )
        
        print(f"\nDisk Cache Performance:")
        print(f"  Avg write time: {results['avg_write_time_ms']:.2f} ms")
        print(f"  Avg read time: {results['avg_read_time_ms']:.2f} ms")
        print(f"  Max write time: {results['max_write_time_ms']:.2f} ms")
        print(f"  Max read time: {results['max_read_time_ms']:.2f} ms")
        
        # Disk operations should be slower but still reasonable
        assert results['avg_read_time_ms'] < 100, "Disk reads should complete within 100ms"


if __name__ == "__main__":
    print("=" * 60)
    print("Cache Performance Benchmarks")
    print("=" * 60)
    
    # Hit rate
    results = benchmark_hit_rate(num_operations=500, hit_ratio=0.8)
    print(f"\nHit Rate: {results['actual_hit_rate']:.2%}")
    print(f"Throughput: {results['operations_per_second']:.0f} ops/sec")
    
    # Compression
    results = benchmark_compression_ratio(sizes=[(500, 500)])
    print(f"\nCompression Ratio: {list(results.values())[0]['compression_ratio']:.2f}x")
    
    print("\n" + "=" * 60)
