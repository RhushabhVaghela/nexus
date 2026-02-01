# Layer Caching System

## Overview

The Layer Caching System is a core component of Nexus SLI (Selective Layer Inference) that optimizes layer loading performance through intelligent caching. It implements a two-tier caching strategy (memory + SSD) with LRU eviction to minimize redundant downloads and disk I/O.

## Features

- **Two-Tier Caching**: Hot layers in GPU/CPU memory, warm layers on fast SSD
- **LRU Eviction**: Automatic removal of least recently used layers
- **Checksum Validation**: Ensures cache integrity with MD5 checksums
- **Persistent Metadata**: Cache state survives process restarts
- **Thread-Safe**: Safe for concurrent access across multiple workers
- **Statistics**: Detailed cache hit/miss metrics for optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER CACHE SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │  MEMORY CACHE   │    │   DISK CACHE    │                 │
│  │  (Hot Layers)   │    │  (Warm Layers)  │                 │
│  │                 │    │                 │                 │
│  │ • Layer 5       │    │ • Layer 1.pt    │                 │
│  │ • Layer 6       │    │ • Layer 2.pt    │                 │
│  │ • Layer 7       │    │ • Layer 3.pt    │                 │
│  │                 │    │ • Layer 4.pt    │                 │
│  └────────┬────────┘    └────────┬────────┘                 │
│           │                      │                           │
│           └──────────┬───────────┘                           │
│                      │                                       │
│           ┌──────────▼───────────┐                          │
│           │   CACHE MANAGER     │                          │
│           │                     │                          │
│           │ • LRU Tracking      │                          │
│           │ • Size Management   │                          │
│           │ • Metadata I/O      │                          │
│           └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from nexus_final.sli.layer_cache import LayerCache

# Initialize cache
cache = LayerCache(
    cache_dir="~/.cache/nexus/layers",
    max_cache_size_gb=50.0,
    max_memory_cache_size_gb=2.0
)

# Try to get layer from cache
layer = cache.get_layer("qwen-7b", layer_index=5)

if layer is None:
    # Cache miss - download and cache
    layer = download_from_hub("Qwen/Qwen2.5-7B", layer_idx=5)
    cache.cache_layer("qwen-7b", 5, layer)

# Use layer for inference
output = layer(hidden_states)
```

### Global Singleton

```python
from nexus_final.sli.layer_cache import get_layer_cache

# Get or create global cache instance
cache = get_layer_cache(
    cache_dir="~/.cache/nexus/layers",
    max_cache_size_gb=100.0
)

# Use anywhere in your code
layer = cache.get_layer("model_id", layer_idx)
```

## Configuration

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str` | `~/.cache/nexus/layers` | Directory for cached layers |
| `max_cache_size_gb` | `float` | `50.0` | Maximum disk cache size in GB |
| `max_memory_cache_size_gb` | `float` | `2.0` | Maximum memory cache size in GB |
| `enable_compression` | `bool` | `False` | Enable layer compression |
| `compression_level` | `int` | `6` | Compression level (1-9) |
| `persist_metadata` | `bool` | `True` | Save cache metadata to disk |

### Configuration Examples

#### Development Setup

```python
# Small cache for development
cache = LayerCache(
    cache_dir="./dev_cache",
    max_cache_size_gb=10.0,
    max_memory_cache_size_gb=1.0
)
```

#### Production Setup

```python
# Large cache for production inference
cache = LayerCache(
    cache_dir="/nvme/nexus_cache",  # Fast NVMe drive
    max_cache_size_gb=200.0,
    max_memory_cache_size_gb=8.0,
    enable_compression=True
)
```

#### Multi-User Setup

```python
# Shared cache for multiple users
import os
cache_dir = os.path.expanduser("~/.cache/nexus/layers")

cache = LayerCache(
    cache_dir=cache_dir,
    max_cache_size_gb=500.0,  # Large shared cache
    max_memory_cache_size_gb=4.0
)
```

## API Reference

### Core Methods

#### `get_layer(model_id, layer_index, quantization_mode=None, device='cpu')`

Retrieve a layer from cache.

```python
layer = cache.get_layer(
    model_id="qwen-7b",
    layer_index=5,
    quantization_mode="nf4",  # Optional
    device="cuda:0"           # Target device
)
```

**Returns:** The layer module, or `None` if not in cache.

#### `cache_layer(model_id, layer_index, layer, quantization_mode=None)`

Cache a layer to disk and memory.

```python
success = cache.cache_layer(
    model_id="qwen-7b",
    layer_index=5,
    layer=layer_module,
    quantization_mode="nf4"
)
```

**Returns:** `True` if caching succeeded.

#### `download_and_cache(model_id, layer_index, repo_id, filename, **kwargs)`

Download from HuggingFace Hub and cache.

```python
layer = cache.download_and_cache(
    model_id="qwen-7b",
    layer_index=5,
    repo_id="Qwen/Qwen2.5-7B",
    filename="model-00001-of-00004.safetensors"
)
```

### Cache Management

#### `invalidate_cache(model_id=None, layer_index=None)`

Remove layers from cache.

```python
# Invalidate specific layer
cache.invalidate_cache("qwen-7b", layer_index=5)

# Invalidate all layers of a model
cache.invalidate_cache("qwen-7b")

# Clear entire cache
cache.invalidate_cache()
```

#### `clear_cache()`

Clear all cached layers.

```python
cache.clear_cache()
```

#### `verify_cache_integrity()`

Check for corrupted cache entries.

```python
corrupted = cache.verify_cache_integrity()
if corrupted:
    print(f"Found {len(corrupted)} corrupted entries")
    cache.optimize_cache()  # Remove corrupted entries
```

#### `optimize_cache()`

Remove corrupted entries and defragment.

```python
cache.optimize_cache()
```

### Statistics

#### `get_stats()`

Get comprehensive cache statistics.

```python
stats = cache.get_stats()
print(json.dumps(stats, indent=2))
```

**Example Output:**

```json
{
  "disk_cache_entries": 128,
  "memory_cache_entries": 16,
  "disk_cache_size_gb": 45.3,
  "memory_cache_size_gb": 1.8,
  "max_disk_cache_size_gb": 50.0,
  "max_memory_cache_size_gb": 2.0,
  "performance": {
    "total_hits": 1500,
    "total_misses": 128,
    "total_evictions": 50,
    "cache_hit_ratio": 0.921
  }
}
```

#### `print_stats()`

Pretty-print cache statistics.

```python
cache.print_stats()
```

**Output:**

```
============================================================
Layer Cache Statistics
============================================================
Disk Cache Entries: 128
Memory Cache Entries: 16
Disk Cache Size: 45.3 GB / 50.0 GB
Memory Cache Size: 1.8 GB / 2.0 GB

Performance:
  Cache Hits: 1500
  Cache Misses: 128
  Hit Ratio: 92.1%
  Evictions: 50
  Data Downloaded: 12.5 GB
  Data Served from Cache: 450.0 GB
============================================================
```

## LRU Eviction

### How It Works

1. **Access Tracking**: Every `get_layer()` updates the access time
2. **Ordering**: Most recently accessed layers are at the end of the queue
3. **Eviction**: When cache is full, least recently used layers are removed first
4. **Two-Tier**: Memory cache evicts to disk, disk cache deletes files

### Eviction Flow

```
User requests Layer 1
    │
    ▼
Check Memory Cache (MISS)
    │
    ▼
Check Disk Cache (HIT)
    │
    ▼
Move to Memory Cache
    │
    ▼
If Memory Cache Full:
    Evict LRU layer from memory
    (moves to disk if not present)
```

### Monitoring Evictions

```python
stats = cache.get_stats()
evictions = stats['performance']['total_evictions']
hits = stats['performance']['total_hits']

if evictions > hits * 0.1:  # More than 10% evictions
    print("Consider increasing cache size")
```

## Integration Examples

### With SLI Inference

```python
from nexus_final.sli.layer_cache import LayerCache
from nexus_final.sli.io_optimizer import IOOptimizer

class SLIInference:
    def __init__(self):
        self.cache = LayerCache(max_cache_size_gb=100.0)
        self.optimizer = IOOptimizer(self.cache)
    
    def forward(self, input_ids, active_layers):
        hidden_states = self.embed(input_ids)
        
        for layer_idx in active_layers:
            # Get layer (from cache or load)
            layer = self.optimizer.get_layer_with_prefetch(
                self.model_id,
                layer_idx,
                len(self.layers)
            )
            
            hidden_states = layer(hidden_states)
        
        return self.lm_head(hidden_states)
```

### With Quantization

```python
from nexus_final.sli.layer_cache import LayerCache
from nexus_final.sli.quantization import get_nf4_config, quantize_layer

cache = LayerCache()
config = get_nf4_config()

# Cache quantized layers (4x smaller)
for i, layer in enumerate(model.layers):
    quantized = quantize_layer(layer, config)
    cache.cache_layer("model", i, quantized, quantization_mode="nf4")

# Load quantized layer
layer = cache.get_layer("model", 5, quantization_mode="nf4")
```

### Distributed Caching

```python
from nexus_final.sli.layer_cache import LayerCache
import torch.distributed as dist

# Each rank maintains its own cache
local_cache = LayerCache(
    cache_dir=f"~/.cache/nexus/rank_{dist.get_rank()}"
)

# Rank 0 downloads, others benefit from NFS/shared storage
if dist.get_rank() == 0:
    layer = download_from_hub(model_id, layer_idx)
    local_cache.cache_layer(model_id, layer_idx, layer)

dist.barrier()  # Wait for download

# All ranks can now load from cache
layer = local_cache.get_layer(model_id, layer_idx)
```

## Best Practices

### 1. Cache Sizing

```python
# Rule of thumb: 2-3x model size for full coverage
model_size_gb = 28  # 7B parameter model in FP16
cache = LayerCache(
    max_cache_size_gb=model_size_gb * 2.5  # 70GB
)
```

### 2. Memory Cache

```python
# Memory cache should fit in CPU RAM
# Typical: 5-10% of disk cache size
cache = LayerCache(
    max_cache_size_gb=100.0,
    max_memory_cache_size_gb=5.0  # 5% of disk cache
)
```

### 3. Directory Selection

```python
# Priority: NVMe SSD > SATA SSD > Network Storage > HDD
cache_paths = {
    'nvme': '/nvme/nexus_cache',      # Best
    'ssd': '/ssd/nexus_cache',        # Good
    'nfs': '/shared/nexus_cache',     # Acceptable
    'hdd': '/data/nexus_cache',       # Slow
}

cache = LayerCache(cache_dir=cache_paths['nvme'])
```

### 4. Cache Warmup

```python
# Pre-populate cache before serving traffic
for layer_idx in range(num_layers):
    layer = load_layer(layer_idx)
    cache.cache_layer("model", layer_idx, layer)

print("Cache warmup complete")
```

## Performance Tuning

### Monitoring

```python
import time

# Monitor cache performance
class CacheMonitor:
    def __init__(self, cache):
        self.cache = cache
        self.start_stats = cache.get_stats()
    
    def report(self):
        end_stats = self.cache.get_stats()
        
        hits = end_stats['performance']['total_hits'] - \
               self.start_stats['performance']['total_hits']
        misses = end_stats['performance']['total_misses'] - \
                 self.start_stats['performance']['total_misses']
        
        hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        print(f"Cache Performance:")
        print(f"  Hits: {hits}")
        print(f"  Misses: {misses}")
        print(f"  Hit Ratio: {hit_ratio:.1%}")
```

### Optimization Checklist

- [ ] Cache directory on fast SSD (NVMe preferred)
- [ ] Sufficient cache size (2x model size)
- [ ] Memory cache appropriate for available RAM
- [ ] Monitor hit ratio (target >90%)
- [ ] Verify checksum integrity regularly
- [ ] Clean up corrupted entries

## Troubleshooting

### Cache Not Persisting

**Problem:** Cache is empty after restart

**Solution:**

```python
cache = LayerCache(
    persist_metadata=True,  # Ensure metadata saving is enabled
    cache_dir="/persistent/path"  # Not temp directory
)
```

### Low Hit Ratio

**Problem:** Cache hit ratio below 50%

**Diagnosis:**

```python
stats = cache.get_stats()
print(f"Disk entries: {stats['disk_cache_entries']}")
print(f"Evictions: {stats['performance']['total_evictions']}")
print(f"Hit ratio: {stats['performance']['cache_hit_ratio']:.1%}")
```

**Solutions:**

1. Increase `max_cache_size_gb`
2. Check for cache thrashing (high evictions)
3. Verify cache directory has sufficient space

### Slow Cache Access

**Problem:** Layers loading slowly even from cache

**Diagnosis:**

```bash
# Check disk speed
dd if=/dev/zero of=~/test_cache/test.img bs=1G count=1 oflag=direct

# Check cache directory filesystem
df -T ~/.cache/nexus/layers
```

**Solutions:**

1. Move cache to faster drive (NVMe)
2. Enable compression for faster reads
3. Increase memory cache size

### Memory Errors

**Problem:** Out of memory when using cache

**Solution:**

```python
cache = LayerCache(
    max_memory_cache_size_gb=0.5,  # Reduce memory cache
    max_cache_size_gb=100.0        # Keep large disk cache
)
```

## API Summary

| Method | Purpose |
|--------|---------|
| `get_layer()` | Retrieve layer from cache |
| `cache_layer()` | Store layer in cache |
| `download_and_cache()` | Download and cache in one step |
| `invalidate_cache()` | Remove layers from cache |
| `clear_cache()` | Clear entire cache |
| `get_stats()` | Get cache statistics |
| `verify_cache_integrity()` | Check for corruption |
| `optimize_cache()` | Clean up and defragment |

## See Also

- [I/O Optimization Guide](IO_OPTIMIZATION.md)
- [Quantization Module](../src/nexus_final/sli/quantization.py)
- [I/O Optimizer](../src/nexus_final/sli/io_optimizer.py)
