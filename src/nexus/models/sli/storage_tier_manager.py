"""
Storage Tier Manager for Nexus SLI (Selective Layer Inference)

Implements intelligent hot/cold tiering for layer storage:
- Hot tier: RAM cache for frequently accessed layers
- Warm tier: Fast NVMe SSD for active window
- Cold tier: Slower storage for archived layers
- Automatic promotion/demotion based on access patterns

Author: Nexus Team
"""

import os
import time
import json
import threading
import logging
from typing import Dict, Optional, Any, List, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum, auto
from collections import OrderedDict
import heapq

import torch
import torch.nn as nn
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageTier(Enum):
    """Storage tier levels."""
    HOT = "hot"      # RAM - Fastest, most expensive
    WARM = "warm"    # NVMe SSD - Fast, moderate cost
    COLD = "cold"    # SATA SSD/HDD - Slow, cheap
    ARCHIVE = "archive"  # Network/object storage - Slowest, cheapest


class AccessPattern(Enum):
    """Access pattern types."""
    SEQUENTIAL = auto()  # Layer N+1 follows layer N
    STRIDED = auto()     # Fixed stride pattern
    RANDOM = auto()      # No predictable pattern
    BURST = auto()       # Bursty access
    TEMPORAL = auto()    # Repeated access to same layers


@dataclass
class TieredEntry:
    """Entry tracking a layer across storage tiers."""
    layer_id: str
    model_id: str
    layer_index: int
    current_tier: StorageTier
    size_bytes: int = 0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    first_accessed: float = field(default_factory=time.time)
    load_time_ms: float = 0.0
    
    # Tier locations
    hot_path: Optional[str] = None
    warm_path: Optional[str] = None
    cold_path: Optional[str] = None
    
    # Scoring
    priority_score: float = 0.0
    access_frequency: float = 0.0  # accesses per second
    
    def update_access(self):
        """Record an access."""
        self.access_count += 1
        self.last_accessed = time.time()
        
        # Update frequency
        time_window = max(1.0, self.last_accessed - self.first_accessed)
        self.access_frequency = self.access_count / time_window
        
        # Update priority score (LRU + Frequency)
        recency_score = 1.0 / (1.0 + time.time() - self.last_accessed)
        self.priority_score = self.access_frequency * 0.7 + recency_score * 0.3


@dataclass
class TierConfig:
    """Configuration for a storage tier."""
    max_size_gb: float
    eviction_policy: str = "lru"  # lru, lfu, fifo
    enable_compression: bool = False
    compression_level: int = 3


@dataclass
class StorageTierConfig:
    """Configuration for the tier manager."""
    # Tier paths
    hot_tier_path: Optional[str] = None  # RAM (in-memory)
    warm_tier_path: Optional[str] = None  # Fast NVMe SSD
    cold_tier_path: Optional[str] = None  # Slower storage
    archive_tier_path: Optional[str] = None  # Archive storage
    
    # Tier limits
    hot_max_memory_gb: float = 4.0
    warm_max_size_gb: float = 50.0
    cold_max_size_gb: float = 200.0
    
    # Promotion/demotion thresholds
    hot_promotion_threshold: int = 3  # Accesses to promote to hot
    warm_promotion_threshold: int = 1  # Accesses to promote to warm
    hot_demotion_idle_seconds: float = 60.0
    warm_demotion_idle_seconds: float = 300.0
    
    # Auto-tiering
    enable_auto_tiering: bool = True
    auto_tiering_interval_seconds: float = 60.0
    
    # Pattern detection
    enable_pattern_detection: bool = True
    pattern_history_size: int = 100
    
    def __post_init__(self):
        """Set default paths if not provided."""
        if self.warm_tier_path is None:
            self.warm_tier_path = str(Path.home() / '.cache' / 'nexus' / 'tier_warm')
        if self.cold_tier_path is None:
            self.cold_tier_path = str(Path.home() / '.cache' / 'nexus' / 'tier_cold')


@dataclass
class TierStats:
    """Statistics for a storage tier."""
    entries: int = 0
    total_size_bytes: int = 0
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    avg_load_time_ms: float = 0.0
    
    def record_access(self, hit: bool, load_time_ms: float = 0.0):
        """Record a tier access."""
        self.accesses += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        
        if load_time_ms > 0:
            self.avg_load_time_ms = (
                (self.avg_load_time_ms * (self.hits - 1) + load_time_ms)
                / self.hits
            )
    
    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio."""
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entries': self.entries,
            'size_gb': self.total_size_bytes / 1e9,
            'accesses': self.accesses,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hit_ratio,
            'promotions': self.promotions,
            'demotions': self.demotions,
            'evictions': self.evictions,
            'avg_load_time_ms': self.avg_load_time_ms,
        }


class StorageTierManager:
    """
    Manages storage tiers for layer caching.
    
    Automatically promotes/demotes layers between tiers based on:
    - Access frequency
    - Recency of access
    - Memory pressure
    - Predicted future access
    
    Example:
        >>> manager = StorageTierManager()
        >>> 
        >>> # Store layer in appropriate tier
        >>> manager.store_layer(layer, "model1_layer_0")
        >>> 
        >>> # Access triggers promotion if needed
        >>> layer = manager.get_layer("model1_layer_0")
        >>> 
        >>> # Check tier statistics
        >>> stats = manager.get_all_stats()
    """

    def __init__(
        self,
        config: Optional[StorageTierConfig] = None,
        layer_loader: Optional[Callable[[str, int], nn.Module]] = None,
    ):
        """
        Initialize the storage tier manager.

        Args:
            config: Tier configuration
            layer_loader: Callback to load layers from source
        """
        self.config = config or StorageTierConfig()
        self.layer_loader = layer_loader
        
        # Tier storage
        self._hot_cache: OrderedDict[str, nn.Module] = OrderedDict()  # In-memory
        self._entries: Dict[str, TieredEntry] = {}
        
        # Tier sizes
        self._hot_size_bytes = 0
        self._warm_size_bytes = 0
        self._cold_size_bytes = 0
        
        # Tier statistics
        self._tier_stats: Dict[StorageTier, TierStats] = {
            tier: TierStats() for tier in StorageTier
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create tier directories
        self._ensure_tier_dirs()
        
        # Load existing metadata
        self._load_metadata()
        
        # Start auto-tiering thread
        self._auto_tiering_thread: Optional[threading.Thread] = None
        self._shutdown = False
        if self.config.enable_auto_tiering:
            self._start_auto_tiering()
        
        # Pattern tracking
        self._access_history: List[Tuple[str, float]] = []
        self._detected_pattern: AccessPattern = AccessPattern.SEQUENTIAL
        
        logger.info(
            f"StorageTierManager initialized (hot: {self.config.hot_max_memory_gb}GB, "
            f"warm: {self.config.warm_max_size_gb}GB, cold: {self.config.cold_max_size_gb}GB)"
        )

    def _ensure_tier_dirs(self):
        """Ensure tier directories exist."""
        for path in [self.config.warm_tier_path, self.config.cold_tier_path]:
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self) -> Path:
        """Get metadata file path."""
        return Path(self.config.warm_tier_path) / 'tier_metadata.json'

    def _load_metadata(self):
        """Load tier metadata."""
        metadata_path = self._get_metadata_path()
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                for entry_data in data.get('entries', []):
                    entry = TieredEntry(
                        layer_id=entry_data['layer_id'],
                        model_id=entry_data['model_id'],
                        layer_index=entry_data['layer_index'],
                        current_tier=StorageTier(entry_data['current_tier']),
                        size_bytes=entry_data['size_bytes'],
                        access_count=entry_data.get('access_count', 0),
                        last_accessed=entry_data.get('last_accessed', time.time()),
                        first_accessed=entry_data.get('first_accessed', time.time()),
                        hot_path=entry_data.get('hot_path'),
                        warm_path=entry_data.get('warm_path'),
                        cold_path=entry_data.get('cold_path'),
                    )
                    
                    # Verify file exists in current tier
                    if self._verify_entry_files(entry):
                        self._entries[entry.layer_id] = entry
                        self._update_tier_size(entry.current_tier, entry.size_bytes)
                
                logger.info(f"Loaded {len(self._entries)} tiered entries")
            except Exception as e:
                logger.warning(f"Failed to load tier metadata: {e}")

    def _save_metadata(self):
        """Save tier metadata."""
        try:
            metadata_path = self._get_metadata_path()
            data = {
                'entries': [
                    {
                        'layer_id': entry.layer_id,
                        'model_id': entry.model_id,
                        'layer_index': entry.layer_index,
                        'current_tier': entry.current_tier.value,
                        'size_bytes': entry.size_bytes,
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed,
                        'first_accessed': entry.first_accessed,
                        'hot_path': entry.hot_path,
                        'warm_path': entry.warm_path,
                        'cold_path': entry.cold_path,
                    }
                    for entry in self._entries.values()
                ],
                'stats': {
                    tier.value: stats.to_dict()
                    for tier, stats in self._tier_stats.items()
                },
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tier metadata: {e}")

    def _verify_entry_files(self, entry: TieredEntry) -> bool:
        """Verify that entry files exist."""
        if entry.current_tier == StorageTier.WARM and entry.warm_path:
            return Path(entry.warm_path).exists()
        elif entry.current_tier == StorageTier.COLD and entry.cold_path:
            return Path(entry.cold_path).exists()
        return True  # HOT tier doesn't need file verification

    def _update_tier_size(self, tier: StorageTier, size_bytes: int, add: bool = True):
        """Update tier size tracking."""
        delta = size_bytes if add else -size_bytes
        
        if tier == StorageTier.HOT:
            self._hot_size_bytes += delta
        elif tier == StorageTier.WARM:
            self._warm_size_bytes += delta
        elif tier == StorageTier.COLD:
            self._cold_size_bytes += delta

    def _get_tier_path(self, layer_id: str, tier: StorageTier) -> Path:
        """Get storage path for a tier."""
        if tier == StorageTier.WARM:
            return Path(self.config.warm_tier_path) / f"{layer_id}.pt"
        elif tier == StorageTier.COLD:
            return Path(self.config.cold_tier_path) / f"{layer_id}.pt"
        else:
            raise ValueError(f"No file storage for tier {tier}")

    def _get_layer_size(self, layer: nn.Module) -> int:
        """Calculate layer size in bytes."""
        total = 0
        for param in layer.parameters():
            total += param.numel() * param.element_size()
        for buffer in layer.buffers():
            total += buffer.numel() * buffer.element_size()
        return total

    def _evict_from_hot(self, target_bytes: Optional[int] = None):
        """Evict layers from hot tier to make room."""
        with self._lock:
            target = target_bytes or int(self._hot_size_bytes * 0.2)
            evicted = 0
            
            while self._hot_size_bytes > target and self._hot_cache:
                # Get oldest entry
                layer_id, layer = self._hot_cache.popitem(last=False)
                
                if layer_id in self._entries:
                    entry = self._entries[layer_id]
                    
                    # Move to warm tier if accessed recently
                    if entry.access_count >= self.config.warm_promotion_threshold:
                        self._move_to_warm(layer_id, layer, entry)
                    else:
                        # Just update size tracking
                        self._hot_size_bytes -= entry.size_bytes
                        entry.hot_path = None
                
                evicted += 1
            
            if evicted > 0:
                logger.debug(f"Evicted {evicted} layers from hot tier")

    def _move_to_warm(self, layer_id: str, layer: nn.Module, entry: TieredEntry):
        """Move a layer to the warm tier."""
        # Save to warm storage
        warm_path = self._get_tier_path(layer_id, StorageTier.WARM)
        
        try:
            torch.save(layer, warm_path)
            
            # Update tracking
            self._hot_size_bytes -= entry.size_bytes
            entry.hot_path = None
            entry.warm_path = str(warm_path)
            
            old_tier = entry.current_tier
            entry.current_tier = StorageTier.WARM
            
            self._update_tier_size(StorageTier.WARM, entry.size_bytes)
            self._tier_stats[StorageTier.WARM].promotions += 1
            
            logger.debug(f"Moved {layer_id} from hot to warm tier")
            
        except Exception as e:
            logger.error(f"Failed to move {layer_id} to warm tier: {e}")

    def _move_to_hot(self, layer_id: str, entry: TieredEntry, layer: nn.Module):
        """Move a layer to the hot tier."""
        with self._lock:
            # Check if we need to make room
            layer_size = entry.size_bytes
            max_hot = int(self.config.hot_max_memory_gb * 1e9)
            
            if self._hot_size_bytes + layer_size > max_hot:
                self._evict_from_hot(layer_size)
            
            # Add to hot cache
            self._hot_cache[layer_id] = layer
            entry.hot_path = "memory"
            
            old_tier = entry.current_tier
            if old_tier != StorageTier.HOT:
                self._update_tier_size(old_tier, entry.size_bytes, add=False)
                self._update_tier_size(StorageTier.HOT, entry.size_bytes)
                entry.current_tier = StorageTier.HOT
                self._tier_stats[StorageTier.HOT].promotions += 1
                
                logger.debug(f"Promoted {layer_id} to hot tier")

    def _load_from_tier(self, layer_id: str, entry: TieredEntry) -> Optional[nn.Module]:
        """Load a layer from its current tier."""
        start_time = time.time()
        
        try:
            if entry.current_tier == StorageTier.HOT and layer_id in self._hot_cache:
                layer = self._hot_cache[layer_id]
                load_time_ms = (time.time() - start_time) * 1000
                self._tier_stats[StorageTier.HOT].record_access(True, load_time_ms)
                return layer
            
            elif entry.current_tier == StorageTier.WARM and entry.warm_path:
                layer = torch.load(entry.warm_path, weights_only=False)
                load_time_ms = (time.time() - start_time) * 1000
                self._tier_stats[StorageTier.WARM].record_access(True, load_time_ms)
                return layer
            
            elif entry.current_tier == StorageTier.COLD and entry.cold_path:
                layer = torch.load(entry.cold_path, weights_only=False)
                load_time_ms = (time.time() - start_time) * 1000
                self._tier_stats[StorageTier.COLD].record_access(True, load_time_ms)
                return layer
            
        except Exception as e:
            logger.error(f"Failed to load {layer_id} from {entry.current_tier}: {e}")
            self._tier_stats[entry.current_tier].record_access(False)
        
        return None

    def store_layer(
        self,
        layer: nn.Module,
        layer_id: str,
        model_id: str = "",
        layer_index: int = 0,
        preferred_tier: StorageTier = StorageTier.WARM
    ) -> TieredEntry:
        """
        Store a layer in the tier system.

        Args:
            layer: Layer to store
            layer_id: Unique identifier
            model_id: Model identifier
            layer_index: Layer index
            preferred_tier: Preferred initial tier

        Returns:
            Tiered entry metadata
        """
        with self._lock:
            size_bytes = self._get_layer_size(layer)
            
            entry = TieredEntry(
                layer_id=layer_id,
                model_id=model_id,
                layer_index=layer_index,
                current_tier=preferred_tier,
                size_bytes=size_bytes,
            )
            
            # Store in appropriate tier
            if preferred_tier == StorageTier.HOT:
                self._move_to_hot(layer_id, entry, layer)
            elif preferred_tier == StorageTier.WARM:
                warm_path = self._get_tier_path(layer_id, StorageTier.WARM)
                torch.save(layer, warm_path)
                entry.warm_path = str(warm_path)
                self._update_tier_size(StorageTier.WARM, size_bytes)
            elif preferred_tier == StorageTier.COLD:
                cold_path = self._get_tier_path(layer_id, StorageTier.COLD)
                torch.save(layer, cold_path)
                entry.cold_path = str(cold_path)
                self._update_tier_size(StorageTier.COLD, size_bytes)
            
            self._entries[layer_id] = entry
            self._save_metadata()
            
            return entry

    def get_layer(self, layer_id: str, auto_promote: bool = True) -> Optional[nn.Module]:
        """
        Get a layer from storage tiers.

        Args:
            layer_id: Layer identifier
            auto_promote: Whether to auto-promote based on access

        Returns:
            Layer module or None
        """
        with self._lock:
            if layer_id not in self._entries:
                return None
            
            entry = self._entries[layer_id]
            
            # Try to load from current tier
            layer = self._load_from_tier(layer_id, entry)
            
            if layer is not None:
                entry.update_access()
                
                # Auto-promotion logic
                if auto_promote:
                    if entry.access_count >= self.config.hot_promotion_threshold:
                        if entry.current_tier != StorageTier.HOT:
                            self._move_to_hot(layer_id, entry, layer)
                    elif entry.access_count >= self.config.warm_promotion_threshold:
                        if entry.current_tier == StorageTier.COLD:
                            # Promote to warm
                            warm_path = self._get_tier_path(layer_id, StorageTier.WARM)
                            torch.save(layer, warm_path)
                            entry.warm_path = str(warm_path)
                            entry.current_tier = StorageTier.WARM
                            self._tier_stats[StorageTier.WARM].promotions += 1
                
                return layer
            
            return None

    def _start_auto_tiering(self):
        """Start background auto-tiering thread."""
        def auto_tier_worker():
            while not self._shutdown:
                try:
                    time.sleep(self.config.auto_tiering_interval_seconds)
                    self._perform_auto_tiering()
                except Exception as e:
                    logger.error(f"Auto-tiering error: {e}")
        
        self._auto_tiering_thread = threading.Thread(
            target=auto_tier_worker,
            daemon=True,
            name="AutoTiering"
        )
        self._auto_tiering_thread.start()
        logger.info("Auto-tiering thread started")

    def _perform_auto_tiering(self):
        """Perform automatic tiering adjustments."""
        with self._lock:
            current_time = time.time()
            
            for entry in list(self._entries.values()):
                idle_time = current_time - entry.last_accessed
                
                # Demote hot to warm
                if entry.current_tier == StorageTier.HOT:
                    if idle_time > self.config.hot_demotion_idle_seconds:
                        if entry.layer_id in self._hot_cache:
                            layer = self._hot_cache.pop(entry.layer_id)
                            self._move_to_warm(entry.layer_id, layer, entry)
                
                # Demote warm to cold
                elif entry.current_tier == StorageTier.WARM:
                    if idle_time > self.config.warm_demotion_idle_seconds:
                        try:
                            cold_path = self._get_tier_path(entry.layer_id, StorageTier.COLD)
                            if entry.warm_path and Path(entry.warm_path).exists():
                                layer = torch.load(entry.warm_path)
                                torch.save(layer, cold_path)
                                Path(entry.warm_path).unlink()
                                
                                entry.cold_path = str(cold_path)
                                entry.warm_path = None
                                entry.current_tier = StorageTier.COLD
                                
                                self._update_tier_size(StorageTier.WARM, entry.size_bytes, add=False)
                                self._update_tier_size(StorageTier.COLD, entry.size_bytes)
                                self._tier_stats[StorageTier.COLD].promotions += 1
                        except Exception as e:
                            logger.warning(f"Failed to demote {entry.layer_id}: {e}")

    def get_stats(self, tier: Optional[StorageTier] = None) -> Dict[str, Any]:
        """Get tier statistics."""
        with self._lock:
            if tier:
                return self._tier_stats[tier].to_dict()
            
            return {
                'hot': self._tier_stats[StorageTier.HOT].to_dict(),
                'warm': self._tier_stats[StorageTier.WARM].to_dict(),
                'cold': self._tier_stats[StorageTier.COLD].to_dict(),
                'hot_size_gb': self._hot_size_bytes / 1e9,
                'warm_size_gb': self._warm_size_bytes / 1e9,
                'cold_size_gb': self._cold_size_bytes / 1e9,
                'total_entries': len(self._entries),
            }

    def get_entry_tier(self, layer_id: str) -> Optional[StorageTier]:
        """Get the current tier of a layer."""
        with self._lock:
            if layer_id in self._entries:
                return self._entries[layer_id].current_tier
            return None

    def clear_tier(self, tier: StorageTier):
        """Clear all layers from a tier."""
        with self._lock:
            to_remove = [
                layer_id for layer_id, entry in self._entries.items()
                if entry.current_tier == tier
            ]
            
            for layer_id in to_remove:
                self.delete_layer(layer_id)

    def delete_layer(self, layer_id: str) -> bool:
        """Delete a layer from all tiers."""
        with self._lock:
            if layer_id not in self._entries:
                return False
            
            entry = self._entries.pop(layer_id)
            
            # Remove from hot cache
            if layer_id in self._hot_cache:
                del self._hot_cache[layer_id]
                self._hot_size_bytes -= entry.size_bytes
            
            # Remove files
            try:
                if entry.warm_path:
                    Path(entry.warm_path).unlink(missing_ok=True)
                    self._warm_size_bytes -= entry.size_bytes
                if entry.cold_path:
                    Path(entry.cold_path).unlink(missing_ok=True)
                    self._cold_size_bytes -= entry.size_bytes
            except Exception as e:
                logger.warning(f"Error deleting layer files: {e}")
            
            return True

    def shutdown(self):
        """Shutdown the tier manager."""
        self._shutdown = True
        self._save_metadata()
        
        if self._auto_tiering_thread:
            self._auto_tiering_thread.join(timeout=5.0)
        
        logger.info("StorageTierManager shut down")


# Convenience function
def create_tier_manager(
    hot_memory_gb: float = 4.0,
    warm_storage_path: Optional[str] = None,
    **kwargs
) -> StorageTierManager:
    """
    Create a storage tier manager.
    
    Args:
        hot_memory_gb: Max RAM for hot tier
        warm_storage_path: Path for warm tier storage
        **kwargs: Additional config options
    
    Returns:
        Configured StorageTierManager
    """
    config = StorageTierConfig(
        hot_max_memory_gb=hot_memory_gb,
        warm_tier_path=warm_storage_path,
        **kwargs
    )
    return StorageTierManager(config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Storage Tier Manager")
    print("=" * 60)
    
    # Create tier manager
    manager = StorageTierManager(
        config=StorageTierConfig(
            hot_max_memory_gb=1.0,
            warm_max_size_gb=5.0,
            enable_auto_tiering=False,  # Disable for testing
        )
    )
    
    # Create test layers
    print("\nStoring test layers...")
    for i in range(5):
        layer = nn.Linear(1024, 1024)
        entry = manager.store_layer(
            layer,
            f"test_layer_{i}",
            model_id="test_model",
            layer_index=i,
            preferred_tier=StorageTier.WARM
        )
        print(f"  Layer {i}: stored in {entry.current_tier.value} tier")
    
    # Simulate accesses with promotion
    print("\nSimulating layer accesses...")
    for i in range(5):
        for access in range(4):  # Multiple accesses to trigger promotion
            layer = manager.get_layer(f"test_layer_{i}")
        
        current_tier = manager.get_entry_tier(f"test_layer_{i}")
        print(f"  Layer {i}: now in {current_tier.value if current_tier else 'unknown'} tier")
    
    # Show stats
    print("\nTier Statistics:")
    stats = manager.get_stats()
    for tier_name, tier_stats in stats.items():
        if isinstance(tier_stats, dict):
            print(f"  {tier_name}:")
            for key, value in tier_stats.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {tier_name}: {tier_stats}")
    
    # Cleanup
    manager.clear_tier(StorageTier.WARM)
    manager.shutdown()
    
    print("\n" + "=" * 60)