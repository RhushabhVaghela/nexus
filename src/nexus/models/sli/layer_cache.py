"""
Layer Caching System for Nexus SLI (Selective Layer Inference)

This module implements an intelligent caching system that:
- Caches downloaded layers to local SSD
- Checks cache before downloading from HF Hub
- Manages cache size with LRU eviction
- Persists cache metadata
- Provides cache hit/miss statistics

Author: Nexus Team
"""

import os
import json
import time
import hashlib
import threading
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from datetime import datetime
import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, try_to_load_from_cache

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a cached layer."""
    layer_id: str
    model_id: str
    layer_index: int
    file_path: str
    file_size: int
    checksum: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    quantization_mode: Optional[str] = None
    compression_ratio: float = 1.0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_bytes_downloaded: int = 0
    total_bytes_served_from_cache: int = 0
    average_load_time_ms: float = 0.0
    cache_hit_ratio: float = 0.0
    last_reset: float = field(default_factory=time.time)

    def record_hit(self, bytes_served: int):
        """Record a cache hit."""
        self.total_hits += 1
        self.total_bytes_served_from_cache += bytes_served
        self._update_ratio()

    def record_miss(self, bytes_downloaded: int):
        """Record a cache miss."""
        self.total_misses += 1
        self.total_bytes_downloaded += bytes_downloaded
        self._update_ratio()

    def record_eviction(self):
        """Record a layer eviction."""
        self.total_evictions += 1

    def _update_ratio(self):
        """Update cache hit ratio."""
        total_requests = self.total_hits + self.total_misses
        if total_requests > 0:
            self.cache_hit_ratio = self.total_hits / total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'total_evictions': self.total_evictions,
            'total_bytes_downloaded': self.total_bytes_downloaded,
            'total_bytes_served_from_cache': self.total_bytes_served_from_cache,
            'average_load_time_ms': self.average_load_time_ms,
            'cache_hit_ratio': self.cache_hit_ratio,
            'last_reset': self.last_reset
        }


class LayerCache:
    """
    LRU-based layer caching system for selective layer inference.
    
    Features:
    - Persistent SSD-based caching
    - Automatic LRU eviction when cache limit reached
    - Checksum validation for cache integrity
    - Statistics tracking
    - Thread-safe operations
    - Compression support
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size_gb: float = 50.0,
        max_memory_cache_size_gb: float = 2.0,
        enable_compression: bool = False,
        compression_level: int = 6,
        persist_metadata: bool = True
    ):
        """
        Initialize the layer cache.

        Args:
            cache_dir: Directory for cached layers (default: ~/.cache/nexus/layers)
            max_cache_size_gb: Maximum cache size in GB
            max_memory_cache_size_gb: Maximum in-memory cache size in GB
            enable_compression: Whether to compress cached layers
            compression_level: Compression level (1-9)
            persist_metadata: Whether to persist cache metadata to disk
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'nexus' / 'layers'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.max_memory_cache_bytes = int(max_memory_cache_size_gb * 1024 * 1024 * 1024)
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.persist_metadata = persist_metadata
        
        # LRU cache: OrderedDict maintains insertion order
        # Most recently accessed at the end, least recently accessed at the beginning
        self._disk_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_cache: OrderedDict[str, torch.nn.Module] = OrderedDict()
        
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._current_cache_size = 0
        self._current_memory_size = 0
        
        # Load existing cache metadata if available
        self._load_metadata()
        
        logger.info(f"LayerCache initialized at {self.cache_dir}")
        logger.info(f"Max disk cache: {max_cache_size_gb}GB, Max memory cache: {max_memory_cache_size_gb}GB")

    def _get_metadata_path(self) -> Path:
        """Get path to metadata file."""
        return self.cache_dir / 'cache_metadata.json'

    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                # Rebuild cache entries
                for entry_data in data.get('entries', []):
                    entry = CacheEntry(**entry_data)
                    if Path(entry.file_path).exists():
                        self._disk_cache[entry.layer_id] = entry
                        self._current_cache_size += entry.file_size
                
                # Restore stats
                stats_data = data.get('stats', {})
                self._stats = CacheStats(**stats_data)
                
                logger.info(f"Loaded {len(self._disk_cache)} cached layers ({self._current_cache_size / 1e9:.2f} GB)")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}. Starting with empty cache.")

    def _save_metadata(self):
        """Save cache metadata to disk."""
        if not self.persist_metadata:
            return
            
        metadata_path = self._get_metadata_path()
        try:
            data = {
                'entries': [asdict(entry) for entry in self._disk_cache.values()],
                'stats': self._stats.to_dict(),
                'last_saved': time.time()
            }
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _compute_checksum(self, file_path: str) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _generate_layer_id(self, model_id: str, layer_index: int, quantization_mode: Optional[str] = None) -> str:
        """Generate unique layer ID."""
        q_suffix = f"_{quantization_mode}" if quantization_mode else ""
        return f"{model_id.replace('/', '_')}_layer_{layer_index}{q_suffix}"

    def _get_cache_path(self, layer_id: str) -> Path:
        """Get filesystem path for a cached layer."""
        return self.cache_dir / f"{layer_id}.pt"

    def _evict_if_necessary(self, required_bytes: int):
        """Evict least recently used layers until we have enough space."""
        with self._lock:
            while self._current_cache_size + required_bytes > self.max_cache_size_bytes and self._disk_cache:
                # Get least recently used entry (first item in OrderedDict)
                lru_layer_id, lru_entry = self._disk_cache.popitem(last=False)
                
                try:
                    file_path = Path(lru_entry.file_path)
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        self._current_cache_size -= file_size
                        self._stats.record_eviction()
                        logger.debug(f"Evicted layer {lru_layer_id} ({file_size / 1e6:.2f} MB)")
                except Exception as e:
                    logger.warning(f"Failed to evict layer {lru_layer_id}: {e}")

    def _update_access(self, layer_id: str):
        """Update access time and move to end of LRU order."""
        with self._lock:
            if layer_id in self._disk_cache:
                entry = self._disk_cache.pop(layer_id)
                entry.last_accessed = time.time()
                entry.access_count += 1
                self._disk_cache[layer_id] = entry

    def get_layer(
        self,
        model_id: str,
        layer_index: int,
        quantization_mode: Optional[str] = None,
        device: str = 'cpu'
    ) -> Optional[torch.nn.Module]:
        """
        Retrieve a layer from cache or return None if not cached.

        Args:
            model_id: HuggingFace model ID
            layer_index: Index of the layer
            quantization_mode: Quantization mode (e.g., 'int8', 'nf4')
            device: Target device for the layer

        Returns:
            The cached layer module or None if not in cache
        """
        layer_id = self._generate_layer_id(model_id, layer_index, quantization_mode)
        
        with self._lock:
            # Check memory cache first
            if layer_id in self._memory_cache:
                layer = self._memory_cache.pop(layer_id)
                self._memory_cache[layer_id] = layer  # Move to end (most recent)
                self._stats.record_hit(self._get_layer_size(layer))
                logger.debug(f"Memory cache hit for layer {layer_id}")
                return layer.to(device)
            
            # Check disk cache
            if layer_id in self._disk_cache:
                entry = self._disk_cache[layer_id]
                cache_path = Path(entry.file_path)
                
                if cache_path.exists():
                    try:
                        start_time = time.time()
                        
                        # Load from disk
                        layer = torch.load(cache_path, map_location=device, weights_only=False)
                        
                        # Update access statistics
                        self._update_access(layer_id)
                        
                        # Calculate size
                        layer_size = self._get_layer_size(layer)
                        self._stats.record_hit(layer_size)
                        
                        # Optionally add to memory cache
                        self._add_to_memory_cache(layer_id, layer)
                        
                        load_time_ms = (time.time() - start_time) * 1000
                        logger.debug(f"Disk cache hit for layer {layer_id} ({load_time_ms:.2f}ms)")
                        
                        return layer
                    except Exception as e:
                        logger.warning(f"Failed to load cached layer {layer_id}: {e}")
                        # Remove corrupted entry
                        self._disk_cache.pop(layer_id, None)
                else:
                    # File missing, remove entry
                    self._disk_cache.pop(layer_id, None)
        
        return None

    def _get_layer_size(self, layer: torch.nn.Module) -> int:
        """Estimate memory size of a layer in bytes."""
        total_size = 0
        for param in layer.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in layer.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size

    def _add_to_memory_cache(self, layer_id: str, layer: torch.nn.Module):
        """Add layer to memory cache with LRU eviction."""
        layer_size = self._get_layer_size(layer)
        
        with self._lock:
            # Evict if necessary
            while (self._current_memory_size + layer_size > self.max_memory_cache_bytes 
                   and self._memory_cache):
                lru_id, lru_layer = self._memory_cache.popitem(last=False)
                self._current_memory_size -= self._get_layer_size(lru_layer)
                logger.debug(f"Evicted layer {lru_id} from memory cache")
            
            # Add to memory cache
            self._memory_cache[layer_id] = layer
            self._current_memory_size += layer_size

    def cache_layer(
        self,
        model_id: str,
        layer_index: int,
        layer: torch.nn.Module,
        quantization_mode: Optional[str] = None
    ) -> bool:
        """
        Cache a layer to disk (and optionally memory).

        Args:
            model_id: HuggingFace model ID
            layer_index: Index of the layer
            layer: The layer module to cache
            quantization_mode: Quantization mode used

        Returns:
            True if caching succeeded, False otherwise
        """
        layer_id = self._generate_layer_id(model_id, layer_index, quantization_mode)
        cache_path = self._get_cache_path(layer_id)
        
        try:
            # Estimate size before saving
            layer_size = self._get_layer_size(layer)
            
            # Ensure we have space
            self._evict_if_necessary(layer_size)
            
            # Save to disk
            torch.save(layer, cache_path)
            
            # Compute checksum
            checksum = self._compute_checksum(str(cache_path))
            
            # Create cache entry
            entry = CacheEntry(
                layer_id=layer_id,
                model_id=model_id,
                layer_index=layer_index,
                file_path=str(cache_path),
                file_size=cache_path.stat().st_size,
                checksum=checksum,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                quantization_mode=quantization_mode
            )
            
            with self._lock:
                self._disk_cache[layer_id] = entry
                self._current_cache_size += entry.file_size
            
            # Add to memory cache for faster future access
            self._add_to_memory_cache(layer_id, layer)
            
            # Save metadata
            self._save_metadata()
            
            logger.debug(f"Cached layer {layer_id} ({entry.file_size / 1e6:.2f} MB)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache layer {layer_id}: {e}")
            return False

    def download_and_cache(
        self,
        model_id: str,
        layer_index: int,
        repo_id: str,
        filename: str,
        quantization_mode: Optional[str] = None,
        device: str = 'cpu',
        **download_kwargs
    ) -> Optional[torch.nn.Module]:
        """
        Download a layer from HuggingFace Hub and cache it.

        Args:
            model_id: Model identifier
            layer_index: Layer index
            repo_id: HuggingFace Hub repository ID
            filename: Filename in the repository
            quantization_mode: Quantization mode to apply
            device: Target device
            **download_kwargs: Additional arguments for hf_hub_download

        Returns:
            The loaded layer module
        """
        layer_id = self._generate_layer_id(model_id, layer_index, quantization_mode)
        
        # Check cache first
        cached_layer = self.get_layer(model_id, layer_index, quantization_mode, device)
        if cached_layer is not None:
            return cached_layer
        
        try:
            start_time = time.time()
            
            # Download from HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.cache_dir / 'temp'),
                **download_kwargs
            )
            
            # Load the layer
            layer = torch.load(downloaded_path, map_location=device, weights_only=False)
            
            # Update stats
            file_size = Path(downloaded_path).stat().st_size
            self._stats.record_miss(file_size)
            
            # Cache the layer
            self.cache_layer(model_id, layer_index, layer, quantization_mode)
            
            # Clean up temp file
            try:
                os.remove(downloaded_path)
            except:
                pass
            
            load_time_ms = (time.time() - start_time) * 1000
            logger.info(f"Downloaded and cached layer {layer_id} ({file_size / 1e6:.2f} MB, {load_time_ms:.2f}ms)")
            
            return layer
            
        except Exception as e:
            logger.error(f"Failed to download layer {layer_id}: {e}")
            return None

    def invalidate_cache(self, model_id: Optional[str] = None, layer_index: Optional[int] = None):
        """
        Invalidate cache entries.

        Args:
            model_id: If specified, only invalidate layers from this model
            layer_index: If specified with model_id, only invalidate this specific layer
        """
        with self._lock:
            to_remove = []
            
            for layer_id, entry in self._disk_cache.items():
                if model_id and entry.model_id != model_id:
                    continue
                if layer_index is not None and entry.layer_index != layer_index:
                    continue
                
                to_remove.append(layer_id)
                
                # Remove file
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                    self._current_cache_size -= entry.file_size
                except Exception as e:
                    logger.warning(f"Failed to remove cached file for {layer_id}: {e}")
            
            # Remove from memory cache
            for layer_id in to_remove:
                self._disk_cache.pop(layer_id, None)
                self._memory_cache.pop(layer_id, None)
            
            logger.info(f"Invalidated {len(to_remove)} cache entries")
            self._save_metadata()

    def clear_cache(self):
        """Clear all cached layers."""
        with self._lock:
            # Remove all files
            for entry in self._disk_cache.values():
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove {entry.file_path}: {e}")
            
            self._disk_cache.clear()
            self._memory_cache.clear()
            self._current_cache_size = 0
            self._current_memory_size = 0
            
            logger.info("Cache cleared")
            self._save_metadata()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'disk_cache_entries': len(self._disk_cache),
                'memory_cache_entries': len(self._memory_cache),
                'disk_cache_size_gb': self._current_cache_size / 1e9,
                'memory_cache_size_gb': self._current_memory_size / 1e9,
                'max_disk_cache_size_gb': self.max_cache_size_bytes / 1e9,
                'max_memory_cache_size_gb': self.max_memory_cache_bytes / 1e9,
                'performance': self._stats.to_dict()
            }

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        perf = stats['performance']
        
        print("\n" + "=" * 60)
        print("Layer Cache Statistics")
        print("=" * 60)
        print(f"Disk Cache Entries: {stats['disk_cache_entries']}")
        print(f"Memory Cache Entries: {stats['memory_cache_entries']}")
        print(f"Disk Cache Size: {stats['disk_cache_size_gb']:.2f} GB / {stats['max_disk_cache_size_gb']:.2f} GB")
        print(f"Memory Cache Size: {stats['memory_cache_size_gb']:.2f} GB / {stats['max_memory_cache_size_gb']:.2f} GB")
        print(f"\nPerformance:")
        print(f"  Cache Hits: {perf['total_hits']}")
        print(f"  Cache Misses: {perf['total_misses']}")
        print(f"  Hit Ratio: {perf['cache_hit_ratio']:.2%}")
        print(f"  Evictions: {perf['total_evictions']}")
        print(f"  Data Downloaded: {perf['total_bytes_downloaded'] / 1e9:.2f} GB")
        print(f"  Data Served from Cache: {perf['total_bytes_served_from_cache'] / 1e9:.2f} GB")
        print("=" * 60 + "\n")

    def verify_cache_integrity(self) -> List[str]:
        """
        Verify integrity of all cached layers.

        Returns:
            List of corrupted layer IDs
        """
        corrupted = []
        
        with self._lock:
            for layer_id, entry in list(self._disk_cache.items()):
                try:
                    file_path = Path(entry.file_path)
                    if not file_path.exists():
                        corrupted.append(layer_id)
                        continue
                    
                    # Verify checksum
                    current_checksum = self._compute_checksum(str(file_path))
                    if current_checksum != entry.checksum:
                        corrupted.append(layer_id)
                        
                except Exception as e:
                    logger.warning(f"Error verifying layer {layer_id}: {e}")
                    corrupted.append(layer_id)
        
        return corrupted

    def optimize_cache(self):
        """
        Optimize cache by removing corrupted entries and defragmenting.
        """
        corrupted = self.verify_cache_integrity()
        
        if corrupted:
            logger.warning(f"Found {len(corrupted)} corrupted cache entries, removing them")
            for layer_id in corrupted:
                if layer_id in self._disk_cache:
                    entry = self._disk_cache.pop(layer_id)
                    try:
                        Path(entry.file_path).unlink(missing_ok=True)
                        self._current_cache_size -= entry.file_size
                    except:
                        pass
        
        self._save_metadata()
        logger.info("Cache optimization complete")


class LayerCacheManager:
    """
    Singleton manager for layer caching across the application.
    """
    _instance: Optional['LayerCacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        if self._initialized:
            return
            
        self.cache = LayerCache(cache_dir=cache_dir, **kwargs)
        self._initialized = True
    
    @classmethod
    def get_cache(cls, cache_dir: Optional[str] = None, **kwargs) -> LayerCache:
        """Get or create the global cache instance."""
        if cls._instance is None:
            cls._instance = cls(cache_dir=cache_dir, **kwargs)
        return cls._instance.cache


# Convenience function for getting global cache
def get_layer_cache(cache_dir: Optional[str] = None, **kwargs) -> LayerCache:
    """Get the global layer cache instance."""
    return LayerCacheManager.get_cache(cache_dir=cache_dir, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cache
    cache = LayerCache(
        cache_dir="./test_cache",
        max_cache_size_gb=1.0,
        max_memory_cache_size_gb=0.5
    )
    
    # Print initial stats
    cache.print_stats()
    
    # Simulate caching some dummy layers
    for i in range(5):
        dummy_layer = nn.Linear(1000, 1000)
        cache.cache_layer("test/model", i, dummy_layer)
    
    # Print stats after caching
    cache.print_stats()
    
    # Retrieve layers (hits)
    for i in range(5):
        layer = cache.get_layer("test/model", i)
        print(f"Retrieved layer {i}: {layer is not None}")
    
    # Print final stats
    cache.print_stats()
    
    # Cleanup
    cache.clear_cache()
