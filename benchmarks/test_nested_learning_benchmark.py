#!/usr/bin/env python3
"""
Nested Learning Benchmark Suite

Benchmarks for nested learning benefits including:
- Cache hit rate comparison (nested vs standard LRU)
- I/O reduction measurement
- Update frequency strategies
- Performance with different tier sizes

Usage:
    pytest benchmarks/test_nested_learning_benchmark.py -v
    pytest benchmarks/test_nested_learning_benchmark.py --benchmark-save=nested_results
    pytest benchmarks/test_nested_learning_benchmark.py --benchmark-json=nested_results.json
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Set
from collections import OrderedDict

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nexus.models.sli.nested_scheduler import (
    NestedUpdateScheduler,
    NestedUpdateConfig,
    UpdateGroup,
    get_nested_scheduler,
)
from nexus.models.sli.hierarchical_cache import (
    HierarchicalLayerCache,
    HierarchicalCacheConfig,
    CacheTier,
    EvictionPolicy,
)


@dataclass
class CacheComparisonResult:
    """Result of cache comparison."""
    cache_type: str
    hit_rate: float
    memory_hits: int
    disk_l1_hits: int
    disk_l2_hits: int
    misses: int
    total_requests: int


@dataclass
class TierPerformanceResult:
    """Result of tier performance test."""
    tier_name: str
    hit_rate: float
    avg_load_time_ms: float
    total_bytes_read: int
    total_bytes_written: int


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_layers():
    """Create sample transformer layers."""
    layers = {}
    for i in range(20):
        if i % 2 == 0:  # Attention layers
            layers[f"layer_{i}_attn"] = nn.Linear(4096, 4096)
        else:  # FFN layers
            layers[f"layer_{i}_ffn"] = nn.Linear(4096, 11008)
    return layers


@pytest.fixture
def hierarchical_cache(tmp_path):
    """Create hierarchical cache."""
    config = HierarchicalCacheConfig(
        memory_cache_size_gb=0.1,  # Small for testing
        disk_l1_size_gb=0.5,
        disk_l2_size_gb=1.0,
        cache_dir=str(tmp_path / "hierarchical_cache"),
        eviction_policy=EvictionPolicy.ADAPTIVE,
        enable_compression=True,
    )
    return HierarchicalLayerCache(config)


@pytest.fixture
def nested_scheduler():
    """Create nested update scheduler."""
    config = NestedUpdateConfig(
        fast_interval=1,
        medium_interval=10,
        slow_interval=100,
        fast_layers={0, 1, 2, 3},
        medium_layers={4, 5, 6, 7, 8, 9},
        slow_layers={10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        warmup_steps=10,
    )
    return NestedUpdateScheduler(config, num_layers=20)


# ============================================================================
# Cache Hit Rate Comparison Benchmarks
# ============================================================================

class TestCacheHitRateComparison:
    """Benchmark cache hit rate: nested vs standard LRU."""
    
    def test_standard_lru_hit_rate(self, sample_layers, tmp_path, benchmark):
        """Measure standard LRU cache hit rate."""
        def simulate_lru_access():
            # Simple LRU cache simulation
            cache = OrderedDict()
            hits = 0
            misses = 0
            max_size = 5
            
            # Simulate sequential access pattern
            for epoch in range(10):
                for layer_name in sample_layers.keys():
                    if layer_name in cache:
                        # Move to end (most recent)
                        cache.move_to_end(layer_name)
                        hits += 1
                    else:
                        misses += 1
                        if len(cache) >= max_size:
                            # Evict oldest
                            cache.popitem(last=False)
                        cache[layer_name] = sample_layers[layer_name]
            
            return hits / (hits + misses) if (hits + misses) > 0 else 0
        
        result = benchmark(simulate_lru_access)
        assert 0 <= result <= 1
    
    def test_nested_hit_rate(self, sample_layers, nested_scheduler, benchmark):
        """Measure nested scheduler access hit rate."""
        def simulate_nested_access():
            hits = 0
            misses = 0
            
            # Simulate training steps
            for step in range(100):
                update_layers = nested_scheduler.get_update_layers(step)
                
                # Check which layers are "hit" (need updating)
                for layer_idx in range(20):
                    if layer_idx in update_layers:
                        hits += 1
                    else:
                        misses += 1
                
                nested_scheduler.step()
            
            return hits / (hits + misses) if (hits + misses) > 0 else 0
        
        result = benchmark(simulate_nested_access)
        assert 0 <= result <= 1
    
    def test_hierarchical_cache_hit_rate(self, sample_layers, hierarchical_cache, benchmark):
        """Measure hierarchical cache hit rate."""
        def simulate_hierarchical_access():
            # First, cache all layers
            for layer_name, layer in sample_layers.items():
                hierarchical_cache.cache_layer(layer_name, layer, priority=5)
            
            hits = 0
            misses = 0
            
            # Simulate access pattern with some locality
            for epoch in range(3):
                # Sequential access
                for layer_name in list(sample_layers.keys())[:10]:
                    layer = hierarchical_cache.get_layer(layer_name)
                    if layer is not None:
                        hits += 1
                    else:
                        misses += 1
                
                # Re-access first 5 (temporal locality)
                for layer_name in list(sample_layers.keys())[:5]:
                    layer = hierarchical_cache.get_layer(layer_name)
                    if layer is not None:
                        hits += 1
                    else:
                        misses += 1
            
            return hits / (hits + misses) if (hits + misses) > 0 else 0
        
        result = benchmark(simulate_hierarchical_access)
        assert 0 <= result <= 1
    
    def test_hit_rate_comparison_detailed(self, sample_layers, hierarchical_cache, nested_scheduler):
        """Detailed hit rate comparison with statistics."""
        # Standard LRU simulation
        lru_cache = OrderedDict()
        lru_hits = 0
        lru_misses = 0
        lru_max_size = 5
        
        # Nested scheduler simulation
        nested_hits = 0
        nested_misses = 0
        
        # Hierarchical cache - populate first
        for layer_name, layer in sample_layers.items():
            hierarchical_cache.cache_layer(layer_name, layer, priority=5)
        
        hier_hits = 0
        hier_misses = 0
        
        # Simulate 100 steps
        num_steps = 100
        
        for step in range(num_steps):
            # LRU access pattern
            for layer_name in list(sample_layers.keys())[:10]:
                if layer_name in lru_cache:
                    lru_cache.move_to_end(layer_name)
                    lru_hits += 1
                else:
                    lru_misses += 1
                    if len(lru_cache) >= lru_max_size:
                        lru_cache.popitem(last=False)
                    lru_cache[layer_name] = sample_layers[layer_name]
            
            # Nested scheduler access
            update_layers = nested_scheduler.get_update_layers(step)
            for layer_idx in range(20):
                if layer_idx in update_layers:
                    nested_hits += 1
                else:
                    nested_misses += 1
            nested_scheduler.step()
            
            # Hierarchical cache access
            for layer_name in list(sample_layers.keys())[:10]:
                layer = hierarchical_cache.get_layer(layer_name)
                if layer is not None:
                    hier_hits += 1
                else:
                    hier_misses += 1
        
        lru_hit_rate = lru_hits / (lru_hits + lru_misses) if (lru_hits + lru_misses) > 0 else 0
        nested_hit_rate = nested_hits / (nested_hits + nested_misses) if (nested_hits + nested_misses) > 0 else 0
        hier_hit_rate = hier_hits / (hier_hits + hier_misses) if (hier_hits + hier_misses) > 0 else 0
        
        results = {
            "standard_lru": {
                "hits": lru_hits,
                "misses": lru_misses,
                "hit_rate": lru_hit_rate,
            },
            "nested_scheduler": {
                "hits": nested_hits,
                "misses": nested_misses,
                "hit_rate": nested_hit_rate,
            },
            "hierarchical_cache": {
                "hits": hier_hits,
                "misses": hier_misses,
                "hit_rate": hier_hit_rate,
            },
            "comparison": {
                "nested_vs_lru": ((nested_hit_rate / lru_hit_rate) - 1) * 100 if lru_hit_rate > 0 else 0,
                "hier_vs_lru": ((hier_hit_rate / lru_hit_rate) - 1) * 100 if lru_hit_rate > 0 else 0,
            }
        }
        
        print("\n" + "="*80)
        print("CACHE HIT RATE COMPARISON")
        print("="*80)
        print(f"{'Cache Type':<25} {'Hits':<10} {'Misses':<10} {'Hit Rate':<15}")
        print("-"*60)
        print(f"{'Standard LRU':<25} {lru_hits:<10} {lru_misses:<10} {lru_hit_rate:<14.2%}")
        print(f"{'Nested Scheduler':<25} {nested_hits:<10} {nested_misses:<10} {nested_hit_rate:<14.2%}")
        print(f"{'Hierarchical Cache':<25} {hier_hits:<10} {hier_misses:<10} {hier_hit_rate:<14.2%}")
        print("="*80)
        print(f"Nested vs LRU improvement: {results['comparison']['nested_vs_lru']:+.1f}%")
        print(f"Hierarchical vs LRU improvement: {results['comparison']['hier_vs_lru']:+.1f}%")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nested_hit_rate_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Hierarchical cache should have better hit rate than simple LRU
        assert hier_hit_rate >= lru_hit_rate * 0.8, "Hierarchical cache hit rate should be comparable or better than LRU"


# ============================================================================
# I/O Reduction Measurement Benchmarks
# ============================================================================

class TestIOReduction:
    """Benchmark I/O reduction from nested learning."""
    
    def test_full_update_io_cost(self, benchmark):
        """Measure I/O cost without nested learning (baseline)."""
        def simulate_full_updates():
            num_layers = 20
            steps = 100
            io_operations = 0
            
            for step in range(steps):
                # Update all layers every step
                for layer_idx in range(num_layers):
                    io_operations += 1  # Simulate I/O per layer update
            
            return io_operations
        
        result = benchmark(simulate_full_updates)
        assert result == 2000  # 20 layers * 100 steps
    
    def test_nested_update_io_cost(self, nested_scheduler, benchmark):
        """Measure I/O cost with nested learning."""
        def simulate_nested_updates():
            io_operations = 0
            steps = 100
            
            for step in range(steps):
                update_layers = nested_scheduler.get_update_layers(step)
                io_operations += len(update_layers)
                nested_scheduler.step()
            
            return io_operations
        
        result = benchmark(simulate_nested_updates)
        assert result > 0
        assert result < 2000  # Should be less than full updates
    
    def test_io_reduction_comparison(self, nested_scheduler):
        """Detailed I/O reduction analysis."""
        num_layers = 20
        steps = 1000
        
        # Full updates (baseline)
        full_io = num_layers * steps
        
        # Nested updates
        nested_io = 0
        group_io = {UpdateGroup.FAST: 0, UpdateGroup.MEDIUM: 0, UpdateGroup.SLOW: 0}
        
        for step in range(steps):
            for layer_idx in range(num_layers):
                if nested_scheduler.should_update(layer_idx, step):
                    nested_io += 1
                    group = nested_scheduler.get_group(layer_idx)
                    group_io[group] += 1
            nested_scheduler.step()
        
        # Calculate savings
        io_savings = full_io - nested_io
        savings_pct = (io_savings / full_io) * 100
        
        results = {
            "full_updates_io": full_io,
            "nested_updates_io": nested_io,
            "io_savings": io_savings,
            "savings_percentage": savings_pct,
            "per_group_io": {
                group.value: count for group, count in group_io.items()
            },
            "compute_savings": nested_scheduler.get_compute_savings(),
        }
        
        print("\n" + "="*80)
        print("I/O REDUCTION MEASUREMENT")
        print("="*80)
        print(f"Full updates I/O:        {full_io:,} operations")
        print(f"Nested updates I/O:      {nested_io:,} operations")
        print(f"I/O savings:             {io_savings:,} operations ({savings_pct:.1f}%)")
        print("-"*60)
        print("Per-group I/O operations:")
        for group, count in group_io.items():
            print(f"  {group.value:<10}: {count:,}")
        print("-"*60)
        print(f"Computed savings:        {results['compute_savings']:.1%}")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nested_io_reduction.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assert significant I/O savings
        assert savings_pct > 30, f"Should achieve >30% I/O savings, got {savings_pct:.1f}%"


# ============================================================================
# Update Frequency Strategies Benchmarks
# ============================================================================

class TestUpdateFrequencyStrategies:
    """Benchmark different update frequency strategies."""
    
    def test_fast_interval_1(self, benchmark):
        """Test update frequency with interval 1."""
        config = NestedUpdateConfig(
            fast_interval=1,
            medium_interval=10,
            slow_interval=100,
            fast_layers={0, 1},
            medium_layers={2, 3},
            slow_layers={4, 5},
        )
        scheduler = NestedUpdateScheduler(config, num_layers=6)
        
        def count_updates():
            updates = 0
            for step in range(100):
                updates += len(scheduler.get_update_layers(step))
                scheduler.step()
            return updates
        
        result = benchmark(count_updates)
        assert result > 0
    
    def test_fast_interval_5(self, benchmark):
        """Test update frequency with interval 5."""
        config = NestedUpdateConfig(
            fast_interval=1,
            medium_interval=5,
            slow_interval=50,
            fast_layers={0, 1},
            medium_layers={2, 3},
            slow_layers={4, 5},
        )
        scheduler = NestedUpdateScheduler(config, num_layers=6)
        
        def count_updates():
            updates = 0
            for step in range(100):
                updates += len(scheduler.get_update_layers(step))
                scheduler.step()
            return updates
        
        result = benchmark(count_updates)
        assert result > 0
    
    def test_strategy_comparison(self):
        """Compare different update strategies."""
        strategies = {
            "conservative": {
                "fast_interval": 1,
                "medium_interval": 20,
                "slow_interval": 200,
            },
            "balanced": {
                "fast_interval": 1,
                "medium_interval": 10,
                "slow_interval": 100,
            },
            "aggressive": {
                "fast_interval": 1,
                "medium_interval": 5,
                "slow_interval": 50,
            },
        }
        
        num_layers = 24
        steps = 500
        results = {}
        
        for name, intervals in strategies.items():
            config = NestedUpdateConfig(
                fast_interval=intervals["fast_interval"],
                medium_interval=intervals["medium_interval"],
                slow_interval=intervals["slow_interval"],
                fast_layers=set(range(int(num_layers * 0.2))),
                medium_layers=set(range(int(num_layers * 0.2), int(num_layers * 0.8))),
                slow_layers=set(range(int(num_layers * 0.8), num_layers)),
            )
            scheduler = NestedUpdateScheduler(config, num_layers=num_layers)
            
            total_updates = 0
            for step in range(steps):
                total_updates += len(scheduler.get_update_layers(step))
                scheduler.step()
            
            full_updates = num_layers * steps
            savings_pct = ((full_updates - total_updates) / full_updates) * 100
            
            results[name] = {
                "intervals": intervals,
                "total_updates": total_updates,
                "full_updates": full_updates,
                "savings_percentage": savings_pct,
                "compute_savings": scheduler.get_compute_savings(),
            }
        
        print("\n" + "="*80)
        print("UPDATE FREQUENCY STRATEGY COMPARISON")
        print("="*80)
        print(f"{'Strategy':<15} {'Fast':<6} {'Medium':<8} {'Slow':<8} {'Updates':<12} {'Savings':<10}")
        print("-"*70)
        
        for name, data in results.items():
            intervals = data["intervals"]
            print(f"{name:<15} {intervals['fast_interval']:<6} {intervals['medium_interval']:<8} "
                  f"{intervals['slow_interval']:<8} {data['total_updates']:<12,} {data['savings_percentage']:<9.1f}%")
        
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nested_strategy_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Conservative should have highest savings
        assert results["conservative"]["savings_percentage"] > results["aggressive"]["savings_percentage"]


# ============================================================================
# Tier Size Performance Benchmarks
# ============================================================================

class TestTierSizePerformance:
    """Benchmark performance with different tier sizes."""
    
    def test_small_memory_tier(self, sample_layers, tmp_path, benchmark):
        """Test hierarchical cache with small memory tier."""
        config = HierarchicalCacheConfig(
            memory_cache_size_gb=0.05,  # 50MB
            disk_l1_size_gb=0.2,
            disk_l2_size_gb=0.5,
            cache_dir=str(tmp_path / "small_cache"),
        )
        cache = HierarchicalLayerCache(config)
        
        def populate_and_access():
            # Cache all layers
            for name, layer in sample_layers.items():
                cache.cache_layer(name, layer, priority=5)
            
            # Access all layers
            for name in sample_layers.keys():
                cache.get_layer(name)
            
            return cache.get_stats()
        
        result = benchmark(populate_and_access)
        assert "hit_rate" in result
    
    def test_large_memory_tier(self, sample_layers, tmp_path, benchmark):
        """Test hierarchical cache with large memory tier."""
        config = HierarchicalCacheConfig(
            memory_cache_size_gb=0.5,  # 500MB
            disk_l1_size_gb=0.5,
            disk_l2_size_gb=1.0,
            cache_dir=str(tmp_path / "large_cache"),
        )
        cache = HierarchicalLayerCache(config)
        
        def populate_and_access():
            for name, layer in sample_layers.items():
                cache.cache_layer(name, layer, priority=5)
            
            for name in sample_layers.keys():
                cache.get_layer(name)
            
            return cache.get_stats()
        
        result = benchmark(populate_and_access)
        assert "hit_rate" in result
    
    def test_tier_size_comparison(self, sample_layers, tmp_path):
        """Compare performance across different tier sizes."""
        tier_configs = {
            "small": {
                "memory_cache_size_gb": 0.05,
                "disk_l1_size_gb": 0.2,
                "disk_l2_size_gb": 0.5,
            },
            "medium": {
                "memory_cache_size_gb": 0.1,
                "disk_l1_size_gb": 0.5,
                "disk_l2_size_gb": 1.0,
            },
            "large": {
                "memory_cache_size_gb": 0.5,
                "disk_l1_size_gb": 1.0,
                "disk_l2_size_gb": 2.0,
            },
        }
        
        results = {}
        
        for name, tier_sizes in tier_configs.items():
            config = HierarchicalCacheConfig(
                memory_cache_size_gb=tier_sizes["memory_cache_size_gb"],
                disk_l1_size_gb=tier_sizes["disk_l1_size_gb"],
                disk_l2_size_gb=tier_sizes["disk_l2_size_gb"],
                cache_dir=str(tmp_path / f"tier_test_{name}"),
                eviction_policy=EvictionPolicy.ADAPTIVE,
            )
            cache = HierarchicalLayerCache(config)
            
            # Populate cache
            start = time.perf_counter()
            for layer_name, layer in sample_layers.items():
                cache.cache_layer(layer_name, layer, priority=5)
            populate_time = (time.perf_counter() - start) * 1000
            
            # Access pattern: sequential + repeat
            start = time.perf_counter()
            for _ in range(3):
                for layer_name in list(sample_layers.keys())[:10]:
                    cache.get_layer(layer_name)
            access_time = (time.perf_counter() - start) * 1000
            
            stats = cache.get_stats()
            
            results[name] = {
                "tier_sizes": tier_sizes,
                "populate_time_ms": populate_time,
                "access_time_ms": access_time,
                "hit_rate": stats["hit_rate"],
                "memory_hits": stats["memory_hits"],
                "disk_l1_hits": stats["disk_l1_hits"],
                "disk_l2_hits": stats["disk_l2_hits"],
                "misses": stats["misses"],
            }
        
        print("\n" + "="*80)
        print("TIER SIZE PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Config':<10} {'Mem (GB)':<10} {'L1 (GB)':<10} {'L2 (GB)':<10} {'Hit Rate':<12} {'Populate (ms)':<15} {'Access (ms)':<15}")
        print("-"*80)
        
        for name, data in results.items():
            sizes = data["tier_sizes"]
            print(f"{name:<10} {sizes['memory_cache_size_gb']:<10.2f} {sizes['disk_l1_size_gb']:<10.2f} "
                  f"{sizes['disk_l2_size_gb']:<10.2f} {data['hit_rate']:<11.2%} "
                  f"{data['populate_time_ms']:<14.2f} {data['access_time_ms']:<14.2f}")
        
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nested_tier_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Larger memory tier should have better hit rate
        assert results["large"]["hit_rate"] >= results["small"]["hit_rate"]


# ============================================================================
# Combined Nested Learning Benchmark
# ============================================================================

class TestCombinedNestedLearning:
    """Test combined nested scheduler + hierarchical cache performance."""
    
    def test_combined_performance(self, sample_layers, tmp_path):
        """Test combined nested learning and caching performance."""
        # Setup components
        cache_config = HierarchicalCacheConfig(
            memory_cache_size_gb=0.2,
            disk_l1_size_gb=0.5,
            disk_l2_size_gb=1.0,
            cache_dir=str(tmp_path / "combined_cache"),
        )
        cache = HierarchicalLayerCache(cache_config)
        
        scheduler_config = NestedUpdateConfig(
            fast_interval=1,
            medium_interval=10,
            slow_interval=100,
            fast_layers=set(range(4)),
            medium_layers=set(range(4, 12)),
            slow_layers=set(range(12, 20)),
        )
        scheduler = NestedUpdateScheduler(scheduler_config, num_layers=20)
        
        # Pre-cache all layers
        for layer_name, layer in sample_layers.items():
            cache.cache_layer(layer_name, layer, priority=5)
        
        # Simulate training
        steps = 200
        total_io_operations = 0
        cache_hits = 0
        cache_misses = 0
        
        for step in range(steps):
            update_layers = scheduler.get_update_layers(step)
            
            for layer_idx in update_layers:
                layer_name = list(sample_layers.keys())[layer_idx]
                layer = cache.get_layer(layer_name)
                
                if layer is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    # Simulate cache miss I/O
                    total_io_operations += 10
                
                total_io_operations += 1  # Base I/O per update
            
            scheduler.step()
        
        # Calculate metrics
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        compute_savings = scheduler.get_compute_savings()
        
        results = {
            "training_steps": steps,
            "total_io_operations": total_io_operations,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "compute_savings": compute_savings,
            "combined_efficiency": cache_hit_rate * compute_savings,
            "cache_stats": cache.get_stats(),
            "scheduler_stats": scheduler.get_stats(),
        }
        
        print("\n" + "="*80)
        print("COMBINED NESTED LEARNING PERFORMANCE")
        print("="*80)
        print(f"Training steps:          {results['training_steps']}")
        print(f"Total I/O operations:    {results['total_io_operations']:,}")
        print(f"Cache hits:              {results['cache_hits']:,}")
        print(f"Cache misses:            {results['cache_misses']:,}")
        print(f"Cache hit rate:          {results['cache_hit_rate']:.2%}")
        print(f"Compute savings:         {results['compute_savings']:.1%}")
        print(f"Combined efficiency:     {results['combined_efficiency']:.2%}")
        print("="*80)
        
        # Save results
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "nested_combined_performance.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Assert good combined performance
        assert cache_hit_rate > 0.5, "Cache hit rate should be >50%"
        assert compute_savings > 0.2, "Compute savings should be >20%"


# ============================================================================
# JSON Report Generation
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Generate comprehensive JSON report after all tests."""
    try:
        output_path = Path("benchmarks/results")
        output_path.mkdir(exist_ok=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "benchmark_type": "Nested Learning Performance Benchmarks",
        }
        
        result_files = [
            "nested_hit_rate_comparison.json",
            "nested_io_reduction.json",
            "nested_strategy_comparison.json",
            "nested_tier_comparison.json",
            "nested_combined_performance.json",
        ]
        
        for filename in result_files:
            filepath = output_path / filename
            if filepath.exists():
                with open(filepath) as f:
                    report[filename.replace(".json", "")] = json.load(f)
        
        with open(output_path / "nested_learning_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Comprehensive nested learning benchmark report saved to: {output_path / 'nested_learning_benchmark_report.json'}")
        
    except Exception as e:
        print(f"Warning: Could not generate final report: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
