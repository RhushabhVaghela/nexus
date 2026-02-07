"""
Hierarchical Layer Cache for Nexus SLI

Implements a three-tier caching system:
- Tier 1 (Hot): In-memory cache for frequently accessed layers
- Tier 2 (Warm): Disk L1 (SSD) for recently used layers
- Tier 3 (Cold): Disk L2 (HDD/Network) for archival storage

Features:
- Automatic promotion/demotion based on access patterns
- Priority-based prefetching
- Cache eviction policies (LRU, LFU)
- Compression for disk storage

Author: Nexus Team
"""

import logging
import os
import json
import time
import shutil
import threading
import gzip
from typing import Dict, Optional, Any, List, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import OrderedDict
from enum import Enum
import pickle

import torch
import torch.nn as nn

from .exceptions import SLIError

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tiers in hierarchical system."""

    MEMORY = "memory"  # Hot: In-memory
    DISK_L1 = "disk_l1"  # Warm: Fast SSD
    DISK_L2 = "disk_l2"  # Cold: Slow storage/HDD
    ARCHIVE = "archive"  # Archived: Network storage


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class HierarchicalCacheEntry:
    """Entry metadata for cached layer."""

    layer_id: str
    tier: CacheTier
    file_path: Optional[str]
    memory_ref: Optional[nn.Module]
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    access_frequency: float = 0.0  # For LFU
    priority: int = 5  # 1-10, higher = more important
    compression_ratio: float = 1.0
    checksum: Optional[str] = None

    def touch(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
        # Decay frequency for LFU
        self.access_frequency = 0.9 * self.access_frequency + 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding memory reference)."""
        return {
            "layer_id": self.layer_id,
            "tier": self.tier.value,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "access_frequency": self.access_frequency,
            "priority": self.priority,
            "compression_ratio": self.compression_ratio,
            "checksum": self.checksum,
        }


@dataclass
class HierarchicalCacheConfig:
    """Configuration for hierarchical cache.

    Attributes:
        memory_cache_size_gb: Max size for memory tier
        disk_l1_size_gb: Max size for disk L1 tier
        disk_l2_size_gb: Max size for disk L2 tier
        cache_dir: Base directory for disk caches
        eviction_policy: Policy for evicting entries
        enable_compression: Enable gzip compression for disk
        compression_level: Compression level (1-9)
        prefetch_lookahead: Number of layers to prefetch
        promotion_threshold: Access count to promote tier
        demotion_threshold: Time in seconds before demotion
        checksum_validation: Validate checksums on load
    """

    memory_cache_size_gb: float = 2.0
    disk_l1_size_gb: float = 50.0
    disk_l2_size_gb: float = 200.0
    cache_dir: str = "./cache/hierarchical"
    eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE
    enable_compression: bool = True
    compression_level: int = 6
    prefetch_lookahead: int = 3
    promotion_threshold: int = 3  # Accesses to promote
    demotion_threshold: float = 3600.0  # Seconds before demotion
    checksum_validation: bool = True

    @property
    def memory_size_bytes(self) -> int:
        return int(self.memory_cache_size_gb * 1024 * 1024 * 1024)

    @property
    def disk_l1_size_bytes(self) -> int:
        return int(self.disk_l1_size_gb * 1024 * 1024 * 1024)

    @property
    def disk_l2_size_bytes(self) -> int:
        return int(self.disk_l2_size_gb * 1024 * 1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_cache_size_gb": self.memory_cache_size_gb,
            "disk_l1_size_gb": self.disk_l1_size_gb,
            "disk_l2_size_gb": self.disk_l2_size_gb,
            "cache_dir": self.cache_dir,
            "eviction_policy": self.eviction_policy.value,
            "enable_compression": self.enable_compression,
            "compression_level": self.compression_level,
            "prefetch_lookahead": self.prefetch_lookahead,
            "promotion_threshold": self.promotion_threshold,
            "demotion_threshold": self.demotion_threshold,
            "checksum_validation": self.checksum_validation,
        }


class HierarchicalCacheError(SLIError):
    """Raised when hierarchical cache operation fails."""

    pass


class HierarchicalLayerCache:
    """Three-tier hierarchical layer cache.

    Manages layers across three storage tiers:
    1. Memory (Hot): Fastest access, limited capacity
    2. Disk L1 (Warm): SSD storage for recently used
    3. Disk L2 (Cold): Larger, slower storage

    Automatic promotion/demotion based on:
    - Access frequency (promote hot items)
    - Time since last access (demote cold items)
    - Priority levels (preserve important layers)

    Example:
        >>> cache = HierarchicalLayerCache(HierarchicalCacheConfig())
        >>>
        >>> # Store layer
        >>> cache.cache_layer("model_layer_0", layer)
        >>>
        >>> # Retrieve with automatic tier management
        >>> layer = cache.get_layer("model_layer_0")
        >>>
        >>> # Prefetch upcoming layers
        >>> cache.prefetch_layers(["model_layer_1", "model_layer_2"])
    """

    def __init__(self, config: Optional[HierarchicalCacheConfig] = None):
        """Initialize hierarchical layer cache.

        Args:
            config: Cache configuration
        """
        self.config = config or HierarchicalCacheConfig()

        # Setup directories
        self.cache_dir = Path(self.config.cache_dir)
        self.disk_l1_dir = self.cache_dir / "tier1_warm"
        self.disk_l2_dir = self.cache_dir / "tier2_cold"

        self.disk_l1_dir.mkdir(parents=True, exist_ok=True)
        self.disk_l2_dir.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._memory_cache: OrderedDict[str, nn.Module] = OrderedDict()

        # Entry metadata
        self._entries: Dict[str, HierarchicalCacheEntry] = {}

        # Size tracking
        self._memory_size = 0
        self._disk_l1_size = 0
        self._disk_l2_size = 0

        # Statistics
        self._stats = {
            "memory_hits": 0,
            "disk_l1_hits": 0,
            "disk_l2_hits": 0,
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
            "evictions": 0,
            "prefetch_hits": 0,
            "bytes_read": 0,
            "bytes_written": 0,
        }

        # Prefetch queue
        self._prefetch_queue: List[str] = []
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()

        # Main lock
        self._lock = threading.RLock()

        # Load existing cache
        self._load_metadata()

        logger.info(f"HierarchicalLayerCache initialized at {self.cache_dir}")
        logger.info(f"  Memory: {self.config.memory_cache_size_gb}GB")
        logger.info(f"  Disk L1: {self.config.disk_l1_size_gb}GB")
        logger.info(f"  Disk L2: {self.config.disk_l2_size_gb}GB")

    def _get_metadata_path(self) -> Path:
        """Get path to metadata file."""
        return self.cache_dir / "cache_metadata.json"

    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    data = json.load(f)

                for entry_data in data.get("entries", []):
                    entry = HierarchicalCacheEntry(
                        layer_id=entry_data["layer_id"],
                        tier=CacheTier(entry_data["tier"]),
                        file_path=entry_data.get("file_path"),
                        memory_ref=None,
                        size_bytes=entry_data["size_bytes"],
                        created_at=entry_data["created_at"],
                        last_accessed=entry_data["last_accessed"],
                        access_count=entry_data.get("access_count", 0),
                        access_frequency=entry_data.get("access_frequency", 0.0),
                        priority=entry_data.get("priority", 5),
                        compression_ratio=entry_data.get("compression_ratio", 1.0),
                        checksum=entry_data.get("checksum"),
                    )

                    # Verify file exists for disk tiers
                    if entry.tier in (CacheTier.DISK_L1, CacheTier.DISK_L2):
                        if entry.file_path and Path(entry.file_path).exists():
                            self._entries[entry.layer_id] = entry
                            if entry.tier == CacheTier.DISK_L1:
                                self._disk_l1_size += entry.size_bytes
                            else:
                                self._disk_l2_size += entry.size_bytes

                logger.info(f"Loaded {len(self._entries)} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_path = self._get_metadata_path()
        try:
            data = {
                "entries": [entry.to_dict() for entry in self._entries.values()],
                "saved_at": time.time(),
            }
            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def get_layer(
        self, layer_id: str, device: str = "cpu", blocking: bool = True
    ) -> Optional[nn.Module]:
        """Retrieve a layer from cache.

        Args:
            layer_id: Layer identifier
            device: Target device
            blocking: Wait for load if True

        Returns:
            Layer module or None if not cached
        """
        with self._lock:
            entry = self._entries.get(layer_id)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Try memory first
            if layer_id in self._memory_cache:
                layer = self._memory_cache[layer_id]
                self._memory_cache.move_to_end(layer_id)
                entry.touch()
                self._stats["memory_hits"] += 1
                return layer.to(device)

            # Load from disk
            if entry.tier == CacheTier.DISK_L1:
                layer = self._load_from_disk(entry, device)
                if layer is not None:
                    self._stats["disk_l1_hits"] += 1
                    self._stats["bytes_read"] += entry.size_bytes
                    self._promote_to_memory(layer_id, layer, entry)
                    return layer

            elif entry.tier == CacheTier.DISK_L2:
                layer = self._load_from_disk(entry, device)
                if layer is not None:
                    self._stats["disk_l2_hits"] += 1
                    self._stats["bytes_read"] += entry.size_bytes
                    self._promote_to_memory(layer_id, layer, entry)
                    # Also promote tier
                    self._promote_tier(layer_id, entry)
                    return layer

            self._stats["misses"] += 1
            return None

    def _load_from_disk(
        self, entry: HierarchicalCacheEntry, device: str
    ) -> Optional[nn.Module]:
        """Load layer from disk."""
        if entry.file_path is None or not Path(entry.file_path).exists():
            return None

        try:
            if entry.file_path.endswith(".gz"):
                # Decompress
                with gzip.open(entry.file_path, "rb") as f:
                    layer = pickle.load(f)
            else:
                layer = torch.load(
                    entry.file_path, map_location=device, weights_only=False
                )

            entry.touch()
            return layer
        except Exception as e:
            logger.error(f"Failed to load layer {entry.layer_id}: {e}")
            return None

    def _promote_to_memory(
        self, layer_id: str, layer: nn.Module, entry: HierarchicalCacheEntry
    ):
        """Promote layer to memory cache."""
        layer_size = self._get_layer_size(layer)

        # Evict if necessary
        while (
            self._memory_size + layer_size > self.config.memory_size_bytes
            and self._memory_cache
        ):
            self._evict_from_memory()

        # Add to memory
        self._memory_cache[layer_id] = layer
        self._memory_size += layer_size
        entry.memory_ref = layer

    def _evict_from_memory(self):
        """Evict least valuable layer from memory."""
        if not self._memory_cache:
            return

        # Find best candidate for eviction
        evict_id = None
        min_score = float("inf")

        for layer_id, layer in self._memory_cache.items():
            entry = self._entries[layer_id]

            # Score based on policy
            if self.config.eviction_policy == EvictionPolicy.LRU:
                score = entry.last_accessed
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                score = -entry.access_frequency
            else:  # ADAPTIVE
                # Combine recency, frequency, and priority
                time_factor = time.time() - entry.last_accessed
                score = (time_factor / (entry.access_frequency + 0.1)) / entry.priority

            if score < min_score:
                min_score = score
                evict_id = layer_id

        if evict_id:
            layer = self._memory_cache.pop(evict_id)
            self._memory_size -= self._get_layer_size(layer)

            entry = self._entries[evict_id]
            entry.memory_ref = None

            self._stats["evictions"] += 1

    def _promote_tier(self, layer_id: str, entry: HierarchicalCacheEntry):
        """Promote layer to higher tier."""
        if entry.tier == CacheTier.DISK_L2:
            # Promote to L1
            new_path = self.disk_l1_dir / f"{layer_id}.pt"
            if entry.file_path and Path(entry.file_path).exists():
                shutil.move(entry.file_path, new_path)
                entry.file_path = str(new_path)
                entry.tier = CacheTier.DISK_L1
                self._disk_l2_size -= entry.size_bytes
                self._disk_l1_size += entry.size_bytes
                self._stats["promotions"] += 1
                logger.debug(f"Promoted {layer_id} to L1")

    def _demote_tier(self, layer_id: str, entry: HierarchicalCacheEntry):
        """Demote layer to lower tier."""
        if entry.tier == CacheTier.DISK_L1:
            # Demote to L2
            new_path = self.disk_l2_dir / f"{layer_id}.pt"
            if entry.file_path and Path(entry.file_path).exists():
                shutil.move(entry.file_path, new_path)
                entry.file_path = str(new_path)
                entry.tier = CacheTier.DISK_L2
                self._disk_l1_size -= entry.size_bytes
                self._disk_l2_size += entry.size_bytes
                self._stats["demotions"] += 1
                logger.debug(f"Demoted {layer_id} to L2")

    def cache_layer(
        self,
        layer_id: str,
        layer: nn.Module,
        priority: int = 5,
        initial_tier: CacheTier = CacheTier.DISK_L1,
    ) -> bool:
        """Cache a layer.

        Args:
            layer_id: Layer identifier
            layer: Layer module to cache
            priority: Priority level (1-10)
            initial_tier: Initial cache tier

        Returns:
            True if caching succeeded
        """
        try:
            layer_size = self._get_layer_size(layer)

            with self._lock:
                # Create entry
                entry = HierarchicalCacheEntry(
                    layer_id=layer_id,
                    tier=initial_tier,
                    file_path=None,
                    memory_ref=None,
                    size_bytes=layer_size,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    priority=priority,
                )

                # Save to disk
                if initial_tier in (CacheTier.DISK_L1, CacheTier.DISK_L2):
                    saved = self._save_to_disk(layer_id, layer, entry)
                    if not saved:
                        return False

                # Store in memory if space available
                if layer_size <= self.config.memory_size_bytes - self._memory_size:
                    self._promote_to_memory(layer_id, layer, entry)

                self._entries[layer_id] = entry
                self._stats["bytes_written"] += layer_size

            self._save_metadata()
            return True

        except Exception as e:
            logger.error(f"Failed to cache layer {layer_id}: {e}")
            return False

    def _save_to_disk(
        self, layer_id: str, layer: nn.Module, entry: HierarchicalCacheEntry
    ) -> bool:
        """Save layer to disk."""
        # Determine path
        if entry.tier == CacheTier.DISK_L1:
            dir_path = self.disk_l1_dir
            max_size = self.config.disk_l1_size_bytes
            current_size = self._disk_l1_size
        else:
            dir_path = self.disk_l2_dir
            max_size = self.config.disk_l2_size_bytes
            current_size = self._disk_l2_size

        # Evict if necessary
        layer_size = entry.size_bytes
        while current_size + layer_size > max_size and self._entries:
            self._evict_from_disk(entry.tier)
            if entry.tier == CacheTier.DISK_L1:
                current_size = self._disk_l1_size
            else:
                current_size = self._disk_l2_size

        # Save
        file_path = dir_path / f"{layer_id}.pt"
        if self.config.enable_compression:
            file_path = dir_path / f"{layer_id}.pt.gz"

        try:
            if self.config.enable_compression:
                with gzip.open(
                    file_path, "wb", compresslevel=self.config.compression_level
                ) as f:
                    pickle.dump(layer, f)
            else:
                torch.save(layer, file_path)

            entry.file_path = str(file_path)

            if entry.tier == CacheTier.DISK_L1:
                self._disk_l1_size += layer_size
            else:
                self._disk_l2_size += layer_size

            return True
        except Exception as e:
            logger.error(f"Failed to save layer to disk: {e}")
            return False

    def _evict_from_disk(self, tier: CacheTier):
        """Evict least valuable layer from disk tier."""
        # Find candidates in this tier
        candidates = [
            (layer_id, entry)
            for layer_id, entry in self._entries.items()
            if entry.tier == tier
        ]

        if not candidates:
            return

        # Sort by value (lower = better to evict)
        if self.config.eviction_policy == EvictionPolicy.LRU:
            candidates.sort(key=lambda x: x[1].last_accessed)
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            candidates.sort(key=lambda x: x[1].access_frequency)
        else:
            candidates.sort(
                key=lambda x: (x[1].priority, time.time() - x[1].last_accessed)
            )

        # Evict first candidate
        evict_id, entry = candidates[0]

        if entry.file_path and Path(entry.file_path).exists():
            Path(entry.file_path).unlink()

        del self._entries[evict_id]

        if tier == CacheTier.DISK_L1:
            self._disk_l1_size -= entry.size_bytes
        else:
            self._disk_l2_size -= entry.size_bytes

    def _get_layer_size(self, layer: nn.Module) -> int:
        """Calculate layer size in bytes."""
        total = 0
        for param in layer.parameters():
            total += param.numel() * param.element_size()
        for buffer in layer.buffers():
            total += buffer.numel() * buffer.element_size()
        return total

    def prefetch_layers(self, layer_ids: List[str], priority: int = 5):
        """Prefetch layers into memory.

        Args:
            layer_ids: List of layer IDs to prefetch
            priority: Priority for prefetching
        """
        with self._prefetch_lock:
            for layer_id in layer_ids:
                if layer_id not in self._prefetch_queue:
                    self._prefetch_queue.append(layer_id)

        # Start prefetch thread if not running
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker, daemon=True
            )
            self._prefetch_thread.start()

    def _prefetch_worker(self):
        """Background prefetch worker."""
        while True:
            with self._prefetch_lock:
                if not self._prefetch_queue:
                    break
                layer_id = self._prefetch_queue.pop(0)

            # Prefetch into memory
            layer = self.get_layer(layer_id)
            if layer is not None:
                self._stats["prefetch_hits"] += 1

            # Small delay to not overwhelm I/O
            time.sleep(0.01)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = (
                self._stats["memory_hits"]
                + self._stats["disk_l1_hits"]
                + self._stats["disk_l2_hits"]
            )
            total_requests = total_hits + self._stats["misses"]

            hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

            return {
                "hit_rate": hit_rate,
                "memory_hits": self._stats["memory_hits"],
                "disk_l1_hits": self._stats["disk_l1_hits"],
                "disk_l2_hits": self._stats["disk_l2_hits"],
                "misses": self._stats["misses"],
                "promotions": self._stats["promotions"],
                "demotions": self._stats["demotions"],
                "evictions": self._stats["evictions"],
                "memory_size_gb": self._memory_size / 1e9,
                "disk_l1_size_gb": self._disk_l1_size / 1e9,
                "disk_l2_size_gb": self._disk_l2_size / 1e9,
                "num_entries": len(self._entries),
            }

    def clear(self, tier: Optional[CacheTier] = None):
        """Clear cache.

        Args:
            tier: Specific tier to clear (None for all)
        """
        with self._lock:
            if tier is None or tier == CacheTier.MEMORY:
                self._memory_cache.clear()
                self._memory_size = 0

            to_remove = []
            for layer_id, entry in self._entries.items():
                if tier is None or entry.tier == tier:
                    if entry.file_path and Path(entry.file_path).exists():
                        Path(entry.file_path).unlink()
                    to_remove.append(layer_id)

                    if entry.tier == CacheTier.DISK_L1:
                        self._disk_l1_size -= entry.size_bytes
                    elif entry.tier == CacheTier.DISK_L2:
                        self._disk_l2_size -= entry.size_bytes

            for layer_id in to_remove:
                del self._entries[layer_id]

        self._save_metadata()
        logger.info(f"Cleared cache (tier: {tier})")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Hierarchical Layer Cache")
    print("=" * 60)

    # Create cache
    config = HierarchicalCacheConfig(
        memory_cache_size_gb=0.5,
        disk_l1_size_gb=1.0,
        disk_l2_size_gb=2.0,
        enable_compression=True,
    )

    cache = HierarchicalLayerCache(config)

    # Create test layers
    print("\nCaching layers...")
    for i in range(5):
        layer = nn.Linear(1000, 1000)
        success = cache.cache_layer(f"test_layer_{i}", layer, priority=i + 1)
        print(f"  Layer {i}: {'cached' if success else 'failed'}")

    # Retrieve layers
    print("\nRetrieving layers...")
    for i in range(5):
        layer = cache.get_layer(f"test_layer_{i}")
        print(f"  Layer {i}: {'hit' if layer is not None else 'miss'}")

    # Show stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Memory hits: {stats['memory_hits']}")
    print(f"  Disk L1 hits: {stats['disk_l1_hits']}")

    print("\n" + "=" * 60)
