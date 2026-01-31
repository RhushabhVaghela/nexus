"""
src/utils/cache_manager.py

Cache management with multiple eviction policies:
- LRU (Least Recently Used) eviction
- LFU (Least Frequently Used) eviction
- TTL (Time To Live) expiration
- Size-based eviction (max cache size)
- Priority-based eviction
- Cache hit/miss statistics

Integration with model loader and dataset loader.
"""

import os
import time
import pickle
import hashlib
import threading
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import logging
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    SIZE = "size"         # Size-based
    PRIORITY = "priority" # Priority-based


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    size: int = 0
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    @property
    def age(self) -> float:
        """Get the age of the entry in seconds."""
        return time.time() - self.created_at
    
    @property
    def last_accessed(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.accessed_at
    
    def touch(self):
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    insertions: int = 0
    total_size: int = 0
    
    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self.total_requests
        if total == 0:
            return 0.0
        return (self.hits / total) * 100
    
    @property
    def miss_rate(self) -> float:
        """Cache miss rate as a percentage."""
        total = self.total_requests
        if total == 0:
            return 0.0
        return (self.misses / total) * 100
    
    def reset(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0


class BaseCache(ABC):
    """Base class for cache implementations."""
    
    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: float = 1024,
                 policy: EvictionPolicy = EvictionPolicy.LRU):
        """Initialize the cache."""
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, **kwargs) -> bool:
        """Put a value into the cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all entries from the cache."""
        pass
    
    @abstractmethod
    def evict(self) -> Optional[str]:
        """Evict an entry from the cache."""
        pass


class LRUCache(BaseCache):
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 1024):
        super().__init__(max_size, max_memory_mb, EvictionPolicy.LRU)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self.stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            priority: int = 0) -> bool:
        """Put a value into the cache."""
        with self._lock:
            # Calculate size (approximate)
            size = self._estimate_size(value)
            
            # Check if value is too large
            if size > self.max_memory_bytes:
                logger.warning(f"Value too large to cache: {size} bytes")
                return False
            
            # Evict entries if necessary
            while (len(self._cache) >= self.max_size or 
                   self.stats.total_size + size > self.max_memory_bytes):
                if not self.evict():
                    break
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                priority=priority,
                ttl=ttl
            )
            
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self.stats.total_size -= old_entry.size
            
            self._cache[key] = entry
            self.stats.total_size += size
            self.stats.insertions += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key not in self._cache:
                return False
            self._remove_entry(key)
            return True
    
    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self.stats.total_size = 0
    
    def evict(self) -> Optional[str]:
        """Evict the least recently used entry."""
        if not self._cache:
            return None
        
        # Get first item (least recently used)
        key, entry = self._cache.popitem(last=False)
        self.stats.total_size -= entry.size
        self.stats.evictions += 1
        
        return key
    
    def _remove_entry(self, key: str):
        """Remove an entry from the cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats.total_size -= entry.size
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return sys.getsizeof(value) if hasattr(sys, 'getsizeof') else 1024
    
    def keys(self) -> List[str]:
        """Get all keys in the cache."""
        with self._lock:
            return list(self._cache.keys())
    
    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            return len(self._cache)


class LFUCache(BaseCache):
    """Least Frequently Used (LFU) cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 1024):
        super().__init__(max_size, max_memory_mb, EvictionPolicy.LFU)
        self._cache: Dict[str, CacheEntry] = {}
        self._frequency: Dict[int, Set[str]] = defaultdict(set)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update frequency
            old_freq = entry.access_count
            self._frequency[old_freq].discard(key)
            if not self._frequency[old_freq]:
                del self._frequency[old_freq]
            
            entry.touch()
            self._frequency[entry.access_count].add(key)
            
            self.stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None,
            priority: int = 0) -> bool:
        """Put a value into the cache."""
        with self._lock:
            size = self._estimate_size(value)
            
            if size > self.max_memory_bytes:
                logger.warning(f"Value too large to cache: {size} bytes")
                return False
            
            # Evict entries if necessary
            while (len(self._cache) >= self.max_size or 
                   self.stats.total_size + size > self.max_memory_bytes):
                if not self.evict():
                    break
            
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                priority=priority,
                ttl=ttl
            )
            
            if key in self._cache:
                old_entry = self._cache[key]
                self.stats.total_size -= old_entry.size
                self._frequency[old_entry.access_count].discard(key)
            
            self._cache[key] = entry
            self._frequency[0].add(key)
            self.stats.total_size += size
            self.stats.insertions += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key not in self._cache:
                return False
            self._remove_entry(key)
            return True
    
    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._frequency.clear()
            self.stats.total_size = 0
    
    def evict(self) -> Optional[str]:
        """Evict the least frequently used entry."""
        if not self._cache:
            return None
        
        # Find the minimum frequency
        min_freq = min(self._frequency.keys())
        
        # Get one of the least frequently used items
        key = next(iter(self._frequency[min_freq]))
        
        self._remove_entry(key)
        self.stats.evictions += 1
        
        return key
    
    def _remove_entry(self, key: str):
        """Remove an entry from the cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats.total_size -= entry.size
            self._frequency[entry.access_count].discard(key)
            if not self._frequency[entry.access_count]:
                del self._frequency[entry.access_count]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return sys.getsizeof(value) if hasattr(sys, 'getsizeof') else 1024


class TTLCache(BaseCache):
    """Time To Live (TTL) cache implementation."""
    
    def __init__(self, default_ttl: float = 3600, max_size: int = 1000,
                 max_memory_mb: float = 1024):
        super().__init__(max_size, max_memory_mb, EvictionPolicy.TTL)
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            entry.touch()
            self.stats.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None,
            priority: int = 0) -> bool:
        """Put a value into the cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            size = self._estimate_size(value)
            
            if size > self.max_memory_bytes:
                logger.warning(f"Value too large to cache: {size} bytes")
                return False
            
            # Clean up expired entries
            self._cleanup_expired()
            
            # Evict entries if necessary
            while (len(self._cache) >= self.max_size or 
                   self.stats.total_size + size > self.max_memory_bytes):
                if not self.evict():
                    break
            
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                priority=priority,
                ttl=ttl
            )
            
            if key in self._cache:
                old_entry = self._cache[key]
                self.stats.total_size -= old_entry.size
            
            self._cache[key] = entry
            self.stats.total_size += size
            self.stats.insertions += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key not in self._cache:
                return False
            self._remove_entry(key)
            return True
    
    def clear(self):
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self.stats.total_size = 0
    
    def evict(self) -> Optional[str]:
        """Evict the oldest entry."""
        if not self._cache:
            return None
        
        # Find the oldest entry
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k].created_at)
        
        self._remove_entry(oldest_key)
        self.stats.evictions += 1
        
        return oldest_key
    
    def _remove_entry(self, key: str):
        """Remove an entry from the cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats.total_size -= entry.size
    
    def _cleanup_expired(self):
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired
        ]
        for key in expired_keys:
            self._remove_entry(key)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return sys.getsizeof(value) if hasattr(sys, 'getsizeof') else 1024


class CacheManager:
    """
    Unified cache manager supporting multiple eviction policies.
    
    Features:
    - Multiple cache backends (LRU, LFU, TTL, etc.)
    - Persistent disk cache
    - Cache statistics
    - Namespace support
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 default_policy: EvictionPolicy = EvictionPolicy.LRU,
                 max_size: int = 1000,
                 max_memory_mb: float = 1024,
                 enable_disk_cache: bool = True):
        """Initialize the cache manager."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".nexus" / "cache"
        self.default_policy = default_policy
        self.enable_disk_cache = enable_disk_cache
        
        # Create caches for different policies
        self._caches: Dict[EvictionPolicy, BaseCache] = {
            EvictionPolicy.LRU: LRUCache(max_size, max_memory_mb),
            EvictionPolicy.LFU: LFUCache(max_size, max_memory_mb),
            EvictionPolicy.TTL: TTLCache(3600, max_size, max_memory_mb),
        }
        
        self._default_cache = self._caches[default_policy]
        
        # Create cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache(self, policy: Optional[EvictionPolicy] = None) -> BaseCache:
        """Get a cache by policy."""
        policy = policy or self.default_policy
        return self._caches.get(policy, self._default_cache)
    
    def get(self, key: str, policy: Optional[EvictionPolicy] = None) -> Optional[Any]:
        """Get a value from the cache."""
        cache = self.get_cache(policy)
        
        # Try memory cache first
        value = cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache if enabled
        if self.enable_disk_cache:
            value = self._load_from_disk(key)
            if value is not None:
                # Put back in memory cache
                cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, value: Any, 
            policy: Optional[EvictionPolicy] = None,
            persist: bool = False,
            **kwargs) -> bool:
        """Put a value into the cache."""
        cache = self.get_cache(policy)
        
        # Store in memory cache
        success = cache.put(key, value, **kwargs)
        
        # Persist to disk if requested
        if success and persist and self.enable_disk_cache:
            self._save_to_disk(key, value)
        
        return success
    
    def delete(self, key: str, policy: Optional[EvictionPolicy] = None) -> bool:
        """Delete a value from the cache."""
        cache = self.get_cache(policy)
        
        # Delete from memory cache
        memory_deleted = cache.delete(key)
        
        # Delete from disk cache
        disk_deleted = False
        if self.enable_disk_cache:
            disk_deleted = self._delete_from_disk(key)
        
        return memory_deleted or disk_deleted
    
    def clear(self, policy: Optional[EvictionPolicy] = None):
        """Clear the cache."""
        if policy:
            cache = self.get_cache(policy)
            cache.clear()
        else:
            for cache in self._caches.values():
                cache.clear()
    
    def get_stats(self, policy: Optional[EvictionPolicy] = None) -> CacheStats:
        """Get cache statistics."""
        if policy:
            return self.get_cache(policy).stats
        
        # Aggregate stats
        total_stats = CacheStats()
        for cache in self._caches.values():
            total_stats.hits += cache.stats.hits
            total_stats.misses += cache.stats.misses
            total_stats.evictions += cache.stats.evictions
            total_stats.insertions += cache.stats.insertions
            total_stats.total_size += cache.stats.total_size
        
        return total_stats
    
    def _get_disk_path(self, key: str) -> Path:
        """Get the disk path for a key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load a value from disk cache."""
        path = self._get_disk_path(key)
        
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            return None
    
    def _save_to_disk(self, key: str, value: Any) -> bool:
        """Save a value to disk cache."""
        path = self._get_disk_path(key)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
            return False
    
    def _delete_from_disk(self, key: str) -> bool:
        """Delete a value from disk cache."""
        path = self._get_disk_path(key)
        
        if not path.exists():
            return False
        
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to delete cache from disk: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# MODEL AND DATASET CACHE
# ═══════════════════════════════════════════════════════════════

class ModelCache:
    """Specialized cache for model loading."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None,
                 max_models: int = 3):
        """Initialize the model cache."""
        self.cache = cache_manager or CacheManager(
            default_policy=EvictionPolicy.LRU,
            max_size=max_models,
            max_memory_mb=20480  # 20GB default for models
        )
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a cached model."""
        key = f"model:{model_name}"
        return self.cache.get(key)
    
    def cache_model(self, model_name: str, model: Any, 
                   persist: bool = False) -> bool:
        """Cache a model."""
        key = f"model:{model_name}"
        return self.cache.put(key, model, persist=persist)
    
    def evict_model(self, model_name: str) -> bool:
        """Evict a model from cache."""
        key = f"model:{model_name}"
        return self.cache.delete(key)
    
    def clear_models(self):
        """Clear all cached models."""
        self.cache.clear()


class DatasetCache:
    """Specialized cache for dataset loading."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None,
                 max_datasets: int = 10,
                 max_memory_mb: float = 2048):
        """Initialize the dataset cache."""
        self.cache = cache_manager or CacheManager(
            default_policy=EvictionPolicy.LRU,
            max_size=max_datasets,
            max_memory_mb=max_memory_mb
        )
    
    def get_dataset(self, dataset_name: str, split: str = "train") -> Optional[Any]:
        """Get a cached dataset."""
        key = f"dataset:{dataset_name}:{split}"
        return self.cache.get(key)
    
    def cache_dataset(self, dataset_name: str, dataset: Any,
                     split: str = "train", ttl: float = 3600) -> bool:
        """Cache a dataset."""
        key = f"dataset:{dataset_name}:{split}"
        return self.cache.put(key, dataset, ttl=ttl)
    
    def evict_dataset(self, dataset_name: str, split: str = "train") -> bool:
        """Evict a dataset from cache."""
        key = f"dataset:{dataset_name}:{split}"
        return self.cache.delete(key)
    
    def clear_datasets(self):
        """Clear all cached datasets."""
        self.cache.clear()


# ═══════════════════════════════════════════════════════════════
# GLOBAL CACHE INSTANCE
# ═══════════════════════════════════════════════════════════════

_global_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def get_model_cache() -> ModelCache:
    """Get the global model cache."""
    return ModelCache(get_cache_manager())


def get_dataset_cache() -> DatasetCache:
    """Get the global dataset cache."""
    return DatasetCache(get_cache_manager())


# Convenience functions

def cache_get(key: str, **kwargs) -> Optional[Any]:
    """Get a value from the global cache."""
    return get_cache_manager().get(key, **kwargs)


def cache_put(key: str, value: Any, **kwargs) -> bool:
    """Put a value into the global cache."""
    return get_cache_manager().put(key, value, **kwargs)


def cache_delete(key: str, **kwargs) -> bool:
    """Delete a value from the global cache."""
    return get_cache_manager().delete(key, **kwargs)


def cache_clear():
    """Clear the global cache."""
    get_cache_manager().clear()


__all__ = [
    'CacheManager',
    'LRUCache',
    'LFUCache',
    'TTLCache',
    'ModelCache',
    'DatasetCache',
    'CacheEntry',
    'CacheStats',
    'EvictionPolicy',
    'get_cache_manager',
    'get_model_cache',
    'get_dataset_cache',
    'cache_get',
    'cache_put',
    'cache_delete',
    'cache_clear',
]
