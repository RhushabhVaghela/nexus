"""
Advanced Activation Caching System for Nexus SLI

Features:
- LRU cache with TTL (Time-To-Live)
- Persistent disk cache with compression
- Cache invalidation strategies
- Hit/miss metrics
- Multi-tier caching (memory + disk)
- Automatic cache cleanup

Author: Nexus Team
"""

import os
import time
import json
import gzip
import pickle
import hashlib
import threading
import logging
from typing import Dict, Optional, Any, List, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from pathlib import Path
from enum import Enum
import heapq
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np

# Memory guard integration (WSL-aware memory safety)
try:
    from nexus.utils.memory_guard import guard, MemoryPressure

    _GUARD_AVAILABLE = guard is not None and MemoryPressure is not None
except ImportError:
    guard = None
    MemoryPressure = None
    _GUARD_AVAILABLE = False

from .exceptions import SLIError

logger = logging.getLogger(__name__)


class CacheInvalidationStrategy(Enum):
    """Cache invalidation strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CompressionType(Enum):
    """Compression algorithms for disk cache."""

    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class ActivationCacheEntry:
    """Entry in the activation cache."""

    key: str
    activation: torch.Tensor
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None  # Time-to-live in seconds
    size_bytes: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary (without tensor)."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "ttl": self.ttl,
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "metadata": self.metadata,
        }


@dataclass
class ActivationCacheStats:
    """Statistics for activation cache."""

    memory_hits: int = 0
    memory_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    evictions: int = 0
    expirations: int = 0
    compressions: int = 0
    decompressions: int = 0
    total_bytes_memory: int = 0
    total_bytes_disk: int = 0
    total_bytes_saved: int = 0
    avg_compression_ratio: float = 1.0

    @property
    def total_hits(self) -> int:
        return self.memory_hits + self.disk_hits

    @property
    def total_misses(self) -> int:
        return self.memory_misses + self.disk_misses

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    @property
    def memory_hit_rate(self) -> float:
        total = self.memory_hits + self.memory_misses
        return self.memory_hits / total if total > 0 else 0.0

    def record_hit(self, from_memory: bool = True):
        """Record a cache hit."""
        if from_memory:
            self.memory_hits += 1
        else:
            self.disk_hits += 1

    def record_miss(self, to_memory: bool = True):
        """Record a cache miss."""
        if to_memory:
            self.memory_misses += 1
        else:
            self.disk_misses += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "disk_hits": self.disk_hits,
            "disk_misses": self.disk_misses,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.hit_rate,
            "memory_hit_rate": self.memory_hit_rate,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "compressions": self.compressions,
            "decompressions": self.decompressions,
            "total_bytes_memory_gb": self.total_bytes_memory / 1e9,
            "total_bytes_disk_gb": self.total_bytes_disk / 1e9,
            "total_bytes_saved_gb": self.total_bytes_saved / 1e9,
            "avg_compression_ratio": self.avg_compression_ratio,
        }


@dataclass
class ActivationCacheConfig:
    """Configuration for activation cache."""

    max_memory_size_gb: float = 2.0
    max_disk_size_gb: float = 10.0
    default_ttl_seconds: Optional[float] = None
    invalidation_strategy: CacheInvalidationStrategy = CacheInvalidationStrategy.LRU
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    enable_persistence: bool = True
    persistence_dir: Optional[str] = None
    cleanup_interval_seconds: float = 300.0
    max_entries_memory: int = 1000
    max_entries_disk: int = 10000

    def __post_init__(self):
        if self.persistence_dir is None:
            self.persistence_dir = str(Path.home() / ".cache" / "nexus" / "activations")


class ActivationCacheError(SLIError):
    """Raised when activation caching fails."""

    pass


class ActivationCache:
    """
    Advanced activation caching system with LRU + TTL and persistent disk cache.

    Features:
    - Two-tier caching (memory + disk)
    - LRU eviction with optional TTL
    - Persistent disk cache with compression
    - Configurable invalidation strategies
    - Comprehensive hit/miss metrics

    Example:
        >>> cache = ActivationCache(
        ...     max_memory_size_gb=4.0,
        ...     max_disk_size_gb=20.0,
        ...     default_ttl_seconds=3600
        ... )
        >>>
        >>> # Store activation
        >>> cache.store("layer_0_output", activation, ttl=1800)
        >>>
        >>> # Retrieve activation
        >>> cached = cache.retrieve("layer_0_output")
        >>> if cached is not None:
        ...     print(f"Cache hit! Shape: {cached.shape}")
    """

    def __init__(self, config: Optional[ActivationCacheConfig] = None):
        """
        Initialize activation cache.

        Args:
            config: Cache configuration
        """
        self.config = config or ActivationCacheConfig()

        # Convert sizes to bytes
        self.max_memory_bytes = int(self.config.max_memory_size_gb * 1e9)
        self.max_disk_bytes = int(self.config.max_disk_size_gb * 1e9)

        # Memory cache (OrderedDict for LRU)
        self._memory_cache: OrderedDict[str, ActivationCacheEntry] = OrderedDict()
        self._current_memory_bytes = 0

        # Disk cache tracking
        self._disk_cache: Dict[str, ActivationCacheEntry] = {}
        self._current_disk_bytes = 0
        self._disk_dir = Path(self.config.persistence_dir)
        if self.config.enable_persistence:
            self._disk_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._stats = ActivationCacheStats()

        # Thread safety
        self._lock = threading.RLock()

        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Load existing disk cache
        if self.config.enable_persistence:
            self._load_disk_cache_metadata()
            self._start_cleanup_thread()

        logger.info(
            f"ActivationCache initialized (memory: {self.config.max_memory_size_gb}GB, "
            f"disk: {self.config.max_disk_size_gb}GB)"
        )

    def _get_cache_key(self, identifier: str, context: Optional[str] = None) -> str:
        """Generate a cache key."""
        if context:
            return hashlib.md5(f"{context}:{identifier}".encode()).hexdigest()
        return hashlib.md5(identifier.encode()).hexdigest()

    def _compute_tensor_size(self, tensor: torch.Tensor) -> int:
        """Compute memory size of tensor in bytes."""
        return tensor.numel() * tensor.element_size()

    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        """Compress tensor data."""
        # Convert to numpy for serialization
        if tensor.is_cuda:
            tensor = tensor.cpu()

        np_array = tensor.numpy()
        dtype = str(tensor.dtype).split(".")[-1]
        shape = tensor.shape

        # Serialize with pickle
        data = pickle.dumps({"data": np_array, "dtype": dtype, "shape": shape})

        # Compress
        if self.config.compression == CompressionType.GZIP:
            compressed = gzip.compress(
                data, compresslevel=self.config.compression_level
            )
        elif self.config.compression == CompressionType.LZ4:
            try:
                import lz4.frame

                compressed = lz4.frame.compress(data)
            except ImportError:
                logger.warning("lz4 not available, using gzip")
                compressed = gzip.compress(
                    data, compresslevel=self.config.compression_level
                )
        elif self.config.compression == CompressionType.ZSTD:
            try:
                import zstandard

                compressed = zstandard.compress(data)
            except ImportError:
                logger.warning("zstandard not available, using gzip")
                compressed = gzip.compress(
                    data, compresslevel=self.config.compression_level
                )
        else:
            compressed = data

        self._stats.compressions += 1
        return compressed

    def _decompress_tensor(self, data: bytes) -> torch.Tensor:
        """Decompress tensor data."""
        # Decompress
        if self.config.compression == CompressionType.GZIP:
            decompressed = gzip.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            try:
                import lz4.frame

                decompressed = lz4.frame.decompress(data)
            except ImportError:
                decompressed = gzip.decompress(data)
        elif self.config.compression == CompressionType.ZSTD:
            try:
                import zstandard

                decompressed = zstandard.decompress(data)
            except ImportError:
                decompressed = gzip.decompress(data)
        else:
            decompressed = data

        # Deserialize
        parsed = pickle.loads(decompressed)
        np_array = parsed["data"]
        dtype = getattr(torch, parsed["dtype"])
        shape = parsed["shape"]

        tensor = torch.from_numpy(np_array).view(shape).to(dtype)

        self._stats.decompressions += 1
        return tensor

    def _evict_memory_entry(self) -> Optional[str]:
        """Evict an entry from memory cache based on strategy."""
        if not self._memory_cache:
            return None

        if self.config.invalidation_strategy == CacheInvalidationStrategy.LRU:
            # Evict least recently used
            key, entry = self._memory_cache.popitem(last=False)

        elif self.config.invalidation_strategy == CacheInvalidationStrategy.LFU:
            # Evict least frequently used
            min_count = min(e.access_count for e in self._memory_cache.values())
            for key, entry in self._memory_cache.items():
                if entry.access_count == min_count:
                    del self._memory_cache[key]
                    break

        elif self.config.invalidation_strategy == CacheInvalidationStrategy.FIFO:
            # Evict oldest
            key, entry = self._memory_cache.popitem(last=False)

        else:  # Default to LRU
            key, entry = self._memory_cache.popitem(last=False)

        self._current_memory_bytes -= entry.size_bytes
        self._stats.evictions += 1

        return key

    def _evict_disk_entry(self) -> Optional[str]:
        """Evict an entry from disk cache."""
        if not self._disk_cache:
            return None

        # Simple LRU for disk
        oldest_key = min(
            self._disk_cache.keys(), key=lambda k: self._disk_cache[k].last_accessed
        )
        entry = self._disk_cache.pop(oldest_key)

        # Remove file
        file_path = self._disk_dir / f"{oldest_key}.pt"
        if file_path.exists():
            file_path.unlink()

        self._current_disk_bytes -= entry.size_bytes
        self._stats.evictions += 1

        return oldest_key

    def _make_room_in_memory(self, required_bytes: int):
        """Make room in memory cache for new entry."""
        while (
            self._current_memory_bytes + required_bytes > self.max_memory_bytes
            or len(self._memory_cache) >= self.config.max_entries_memory
        ):
            if not self._memory_cache:
                break
            self._evict_memory_entry()

    def _make_room_on_disk(self, required_bytes: int):
        """Make room on disk for new entry."""
        while (
            self._current_disk_bytes + required_bytes > self.max_disk_bytes
            or len(self._disk_cache) >= self.config.max_entries_disk
        ):
            if not self._disk_cache:
                break
            self._evict_disk_entry()

    def _save_to_disk(self, entry: ActivationCacheEntry):
        """Save an entry to disk cache."""
        if not self.config.enable_persistence:
            return

        file_path = self._disk_dir / f"{entry.key}.pt"

        try:
            # Compress and save
            compressed_data = self._compress_tensor(entry.activation)

            # Make room
            self._make_room_on_disk(len(compressed_data))

            with open(file_path, "wb") as f:
                f.write(compressed_data)

            entry.compressed = True
            entry.size_bytes = len(compressed_data)
            self._current_disk_bytes += len(compressed_data)
            self._stats.total_bytes_disk += len(compressed_data)

            # Track compression ratio
            original_size = self._compute_tensor_size(entry.activation)
            ratio = (
                original_size / len(compressed_data)
                if len(compressed_data) > 0
                else 1.0
            )
            self._stats.total_bytes_saved += original_size - len(compressed_data)

            # Update running average
            n = self._stats.compressions
            self._stats.avg_compression_ratio = (
                self._stats.avg_compression_ratio * (n - 1) + ratio
            ) / n

            self._disk_cache[entry.key] = entry

        except Exception as e:
            logger.warning(f"Failed to save entry to disk: {e}")

    def _load_from_disk(self, key: str) -> Optional[torch.Tensor]:
        """Load an entry from disk cache."""
        file_path = self._disk_dir / f"{key}.pt"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "rb") as f:
                compressed_data = f.read()

            tensor = self._decompress_tensor(compressed_data)

            # Update access time
            if key in self._disk_cache:
                self._disk_cache[key].last_accessed = time.time()

            return tensor

        except Exception as e:
            logger.warning(f"Failed to load entry from disk: {e}")
            return None

    def _load_disk_cache_metadata(self):
        """Load metadata for existing disk cache."""
        metadata_file = self._disk_dir / "metadata.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                key = entry_data["key"]
                file_path = self._disk_dir / f"{key}.pt"

                if file_path.exists():
                    entry = ActivationCacheEntry(
                        key=key,
                        activation=None,  # Not loaded yet
                        created_at=entry_data["created_at"],
                        last_accessed=entry_data["last_accessed"],
                        access_count=entry_data["access_count"],
                        ttl=entry_data.get("ttl"),
                        size_bytes=file_path.stat().st_size,
                        compressed=True,
                        metadata=entry_data.get("metadata", {}),
                    )
                    self._disk_cache[key] = entry
                    self._current_disk_bytes += entry.size_bytes

            logger.info(f"Loaded {len(self._disk_cache)} entries from disk cache")

        except Exception as e:
            logger.warning(f"Failed to load disk cache metadata: {e}")

    def _save_disk_cache_metadata(self):
        """Save metadata for disk cache."""
        metadata_file = self._disk_dir / "metadata.json"

        try:
            data = {
                "entries": [entry.to_dict() for entry in self._disk_cache.values()],
                "saved_at": time.time(),
            }

            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save disk cache metadata: {e}")

    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        current_time = time.time()

        with self._lock:
            # Check memory cache
            expired_keys = [
                key for key, entry in self._memory_cache.items() if entry.is_expired()
            ]

            for key in expired_keys:
                entry = self._memory_cache.pop(key)
                self._current_memory_bytes -= entry.size_bytes
                self._stats.expirations += 1
                logger.debug(f"Expired memory entry: {key}")

            # Check disk cache
            expired_disk_keys = [
                key for key, entry in self._disk_cache.items() if entry.is_expired()
            ]

            for key in expired_disk_keys:
                self._evict_disk_entry()
                self._stats.expirations += 1
                logger.debug(f"Expired disk entry: {key}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""

        def cleanup_loop():
            while not self._shutdown:
                time.sleep(self.config.cleanup_interval_seconds)
                if not self._shutdown:
                    self._cleanup_expired_entries()
                    self._save_disk_cache_metadata()

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop, daemon=True, name="activation-cache-cleanup"
        )
        self._cleanup_thread.start()

    def store(
        self,
        identifier: str,
        activation: torch.Tensor,
        context: Optional[str] = None,
        ttl: Optional[float] = None,
        persist: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store an activation in the cache.

        Args:
            identifier: Unique identifier for this activation
            activation: Tensor to cache
            context: Optional context for namespacing
            ttl: Time-to-live in seconds (None for no expiration)
            persist: Whether to persist to disk
            metadata: Optional metadata dictionary

        Returns:
            True if stored successfully
        """
        # Memory guard: refuse store under high pressure, evict under critical
        if _GUARD_AVAILABLE:
            try:
                pressure = guard.get_pressure()
                pressure_order = list(MemoryPressure)
                pressure_idx = pressure_order.index(pressure)
                critical_idx = pressure_order.index(MemoryPressure.CRITICAL)
                high_idx = pressure_order.index(MemoryPressure.HIGH)

                if pressure_idx >= critical_idx:
                    # Proactively evict oldest entries before refusing
                    with self._lock:
                        for _ in range(min(3, len(self._memory_cache))):
                            self._evict_memory_entry()
                    logger.warning(
                        f"[Cache] CRITICAL pressure ({pressure.value}) — "
                        f"evicted entries, refusing store for '{identifier}'"
                    )
                    return False
                elif pressure_idx >= high_idx:
                    logger.info(
                        f"[Cache] HIGH pressure ({pressure.value}) — "
                        f"skipping cache store for '{identifier}'"
                    )
                    return False
            except Exception:
                pass

        key = self._get_cache_key(identifier, context)

        tensor_size_gb = activation.numel() * activation.element_size() / 1e9
        ctx = (
            guard.safe_allocate(
                estimated_ram_gb=tensor_size_gb,
                estimated_vram_gb=0,
                operation="activation_cache_store",
            )
            if (_GUARD_AVAILABLE and guard is not None)
            else nullcontext()
        )
        with ctx:
            with self._lock:
                # Compute size
                size_bytes = self._compute_tensor_size(activation)

                # Make room
                self._make_room_in_memory(size_bytes)

                # Create entry
                entry = ActivationCacheEntry(
                    key=key,
                    activation=activation.detach().cpu(),
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=0,
                    ttl=ttl or self.config.default_ttl_seconds,
                    size_bytes=size_bytes,
                    compressed=False,
                    metadata=metadata or {},
                )

                # Store in memory
                self._memory_cache[key] = entry
                self._current_memory_bytes += size_bytes
                self._stats.total_bytes_memory += size_bytes

                # Persist to disk if enabled
                if persist and self.config.enable_persistence:
                    self._save_to_disk(entry)

                return True

    def retrieve(
        self,
        identifier: str,
        context: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve an activation from the cache.

        Args:
            identifier: Unique identifier
            context: Optional context for namespacing
            device: Device to move tensor to

        Returns:
            Cached tensor or None if not found
        """
        key = self._get_cache_key(identifier, context)

        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache.pop(key)

                # Check TTL
                if entry.is_expired():
                    self._current_memory_bytes -= entry.size_bytes
                    self._stats.expirations += 1
                    return None

                # Update stats
                entry.last_accessed = time.time()
                entry.access_count += 1

                # Move to end (most recently used)
                self._memory_cache[key] = entry

                self._stats.record_hit(from_memory=True)

                activation = entry.activation
                if device:
                    activation = activation.to(device)
                return activation

            self._stats.record_miss(to_memory=True)

            # Check disk cache
            if key in self._disk_cache:
                entry = self._disk_cache[key]

                # Check TTL
                if entry.is_expired():
                    self._evict_disk_entry()
                    return None

                # Load from disk
                activation = self._load_from_disk(key)

                if activation is not None:
                    # Update stats
                    entry.last_accessed = time.time()
                    entry.access_count += 1

                    self._stats.record_hit(from_memory=False)

                    # Optionally bring back to memory
                    if (
                        self._current_memory_bytes + entry.size_bytes
                        < self.max_memory_bytes
                    ):
                        promote_size_gb = entry.size_bytes / 1e9
                        promote_ctx = (
                            guard.safe_allocate(
                                estimated_ram_gb=promote_size_gb,
                                estimated_vram_gb=0,
                                operation="activation_cache_promote",
                            )
                            if (_GUARD_AVAILABLE and guard is not None)
                            else nullcontext()
                        )
                        with promote_ctx:
                            memory_entry = ActivationCacheEntry(
                                key=key,
                                activation=activation,
                                created_at=entry.created_at,
                                last_accessed=time.time(),
                                access_count=entry.access_count,
                                ttl=entry.ttl,
                                size_bytes=self._compute_tensor_size(activation),
                                compressed=False,
                                metadata=entry.metadata,
                            )
                            self._memory_cache[key] = memory_entry
                            self._current_memory_bytes += memory_entry.size_bytes

                    if device:
                        activation = activation.to(device)
                    return activation

        self._stats.record_miss(to_memory=False)
        return None

    def invalidate(
        self,
        identifier: Optional[str] = None,
        context: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            identifier: Specific identifier to invalidate
            context: Context to invalidate all entries for
            pattern: Pattern to match keys against

        Returns:
            Number of entries invalidated
        """
        count = 0

        with self._lock:
            if identifier:
                key = self._get_cache_key(identifier, context)

                if key in self._memory_cache:
                    entry = self._memory_cache.pop(key)
                    self._current_memory_bytes -= entry.size_bytes
                    count += 1

                if key in self._disk_cache:
                    self._evict_disk_entry()
                    count += 1

            elif context:
                # Invalidate by context
                prefix = hashlib.md5(f"{context}:".encode()).hexdigest()[:8]

                keys_to_remove = [
                    k for k in self._memory_cache.keys() if k.startswith(prefix)
                ]
                for key in keys_to_remove:
                    entry = self._memory_cache.pop(key)
                    self._current_memory_bytes -= entry.size_bytes
                    count += 1

                disk_keys = [k for k in self._disk_cache.keys() if k.startswith(prefix)]
                for key in disk_keys:
                    self._evict_disk_entry()
                    count += 1

            elif pattern:
                # Pattern matching (simple substring match)
                keys_to_remove = [
                    k for k in list(self._memory_cache.keys()) if pattern in k
                ]
                for key in keys_to_remove:
                    entry = self._memory_cache.pop(key)
                    self._current_memory_bytes -= entry.size_bytes
                    count += 1

                disk_keys = [k for k in list(self._disk_cache.keys()) if pattern in k]
                for key in disk_keys:
                    self._evict_disk_entry()
                    count += 1

        return count

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._current_memory_bytes = 0

            for key in list(self._disk_cache.keys()):
                self._evict_disk_entry()

            self._disk_cache.clear()
            self._current_disk_bytes = 0

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.to_dict()
            stats["memory_entries"] = len(self._memory_cache)
            stats["disk_entries"] = len(self._disk_cache)
            stats["memory_usage_gb"] = self._current_memory_bytes / 1e9
            stats["disk_usage_gb"] = self._current_disk_bytes / 1e9
            return stats

    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("Activation Cache Statistics")
        print("=" * 60)
        print(f"Memory Entries: {stats['memory_entries']}")
        print(f"Disk Entries: {stats['disk_entries']}")
        print(f"Memory Usage: {stats['memory_usage_gb']:.2f} GB")
        print(f"Disk Usage: {stats['disk_usage_gb']:.2f} GB")
        print(f"\nHit Rate: {stats['hit_rate']:.2%}")
        print(f"Memory Hit Rate: {stats['memory_hit_rate']:.2%}")
        print(f"Total Hits: {stats['total_hits']}")
        print(f"Total Misses: {stats['total_misses']}")
        print(f"\nEvictions: {stats['evictions']}")
        print(f"Expirations: {stats['expirations']}")
        print(f"Compressions: {stats['compressions']}")
        print(f"Avg Compression Ratio: {stats['avg_compression_ratio']:.2f}x")
        print("=" * 60 + "\n")

    def shutdown(self):
        """Shutdown the cache and cleanup."""
        self._shutdown = True

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        self._save_disk_cache_metadata()
        logger.info("ActivationCache shut down")


class ActivationCacheManager:
    """Singleton manager for activation caching."""

    _instance: Optional["ActivationCacheManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ActivationCacheConfig] = None):
        if self._initialized:
            return
        self.cache = ActivationCache(config=config)
        self._initialized = True

    @classmethod
    def get_cache(
        cls, config: Optional[ActivationCacheConfig] = None
    ) -> "ActivationCache":
        """Get or create the global cache instance."""
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance.cache


def get_activation_cache(
    config: Optional[ActivationCacheConfig] = None,
) -> ActivationCache:
    """Get the global activation cache instance."""
    return ActivationCacheManager.get_cache(config=config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Advanced Activation Cache")
    print("=" * 60)

    # Create cache
    cache = ActivationCache(
        config=ActivationCacheConfig(
            max_memory_size_gb=1.0, max_disk_size_gb=2.0, default_ttl_seconds=60.0
        )
    )

    # Store some activations
    print("\nStoring activations...")
    for i in range(5):
        activation = torch.randn(1, 1024, 4096)
        cache.store(f"layer_{i}_output", activation, ttl=30.0)
        print(f"  Stored layer_{i}_output: {activation.shape}")

    # Retrieve some
    print("\nRetrieving activations...")
    for i in range(7):
        cached = cache.retrieve(f"layer_{i}_output")
        if cached is not None:
            print(f"  Retrieved layer_{i}_output: {cached.shape} ✓")
        else:
            print(f"  layer_{i}_output: Not found ✗")

    # Print stats
    cache.print_stats()

    # Shutdown
    cache.shutdown()

    print("\n" + "=" * 60)
