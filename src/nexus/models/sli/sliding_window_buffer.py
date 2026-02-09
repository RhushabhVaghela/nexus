"""
Sliding Window Buffer for Nexus SLI (Selective Layer Inference)

Implements a sliding window that keeps N layers in memory for smooth
transitions and optimal I/O performance during sequential layer inference.

Features:
- Configurable window size (default 3-5 layers)
- Overlap between windows for smooth transitions
- Automatic adjustment based on memory availability
- LRU eviction when window slides
- Memory budget tracking and enforcement

Author: Nexus Team
"""

import os
import time
import threading
import logging
from typing import Dict, Optional, Any, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import centralized memory guard for WSL-aware thresholds
try:
    from nexus.utils.memory_guard import guard as _memory_guard, MemoryPressure

    _GUARD_AVAILABLE = True
except ImportError:
    _memory_guard = None
    MemoryPressure = None  # type: ignore[assignment, misc]
    _GUARD_AVAILABLE = False


class WindowState(Enum):
    """State of a layer in the sliding window."""

    LOADING = "loading"
    READY = "ready"
    ACTIVE = "active"
    EVICTING = "evicting"
    EVICTED = "evicted"


@dataclass
class WindowEntry:
    """Entry for a layer in the sliding window."""

    layer_id: str
    layer_index: int
    model_id: str
    layer: Optional[nn.Module] = None
    state: WindowState = WindowState.LOADING
    size_bytes: int = 0
    loaded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    priority: int = 5  # 0-10, higher = more important


@dataclass
class WindowStats:
    """Statistics for the sliding window."""

    window_slides: int = 0
    layers_loaded: int = 0
    layers_evicted: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_accesses: int = 0
    avg_load_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    window_resizes: int = 0

    def record_load(self, load_time_ms: float):
        """Record a layer load."""
        self.layers_loaded += 1
        self.avg_load_time_ms = (
            self.avg_load_time_ms * (self.layers_loaded - 1) + load_time_ms
        ) / self.layers_loaded

    def record_access(self, hit: bool):
        """Record a window access."""
        self.total_accesses += 1
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        if self.total_accesses == 0:
            return 0.0
        return self.cache_hits / self.total_accesses

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "window_slides": self.window_slides,
            "layers_loaded": self.layers_loaded,
            "layers_evicted": self.layers_evicted,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": self.hit_ratio,
            "avg_load_time_ms": self.avg_load_time_ms,
            "peak_memory_gb": self.peak_memory_bytes / 1e9,
            "current_memory_gb": self.current_memory_bytes / 1e9,
            "window_resizes": self.window_resizes,
        }


def _get_guard_threshold(key: str, fallback: float) -> float:
    """Pull a threshold from memory_guard if available, else use fallback."""
    if _GUARD_AVAILABLE and _memory_guard is not None:
        thresholds = _memory_guard.get_thresholds()
        return thresholds.get(key, fallback)
    return fallback


@dataclass
class SlidingWindowConfig:
    """Configuration for the sliding window buffer.

    Thresholds default to WSL-aware values from memory_guard when available,
    falling back to conservative static values otherwise.
    """

    min_window_size: int = 3
    max_window_size: int = 7
    default_window_size: int = 5
    memory_threshold_percent: float = field(
        default_factory=lambda: _get_guard_threshold("ram_high_percent", 85.0)
    )
    aggressive_eviction_threshold: float = field(
        default_factory=lambda: _get_guard_threshold("ram_critical_percent", 95.0)
    )
    vram_threshold_percent: float = field(
        default_factory=lambda: _get_guard_threshold("vram_high_percent", 85.0)
    )
    aggressive_vram_threshold: float = field(
        default_factory=lambda: _get_guard_threshold("vram_critical_percent", 94.0)
    )
    overlap_layers: int = 1
    preload_ahead: int = 2
    enable_dynamic_resize: bool = True
    enable_priority_boost: bool = True
    max_memory_gb: Optional[float] = None


class SlidingWindowBuffer:
    """
    Sliding window buffer that maintains N layers in memory.

    This enables smooth layer transitions by:
    1. Keeping a window of layers in memory
    2. Preloading upcoming layers before they're needed
    3. Evicting oldest layers when window slides
    4. Dynamically adjusting window size based on memory pressure

    Example:
        >>> window = SlidingWindowBuffer(window_size=5)
        >>> window.initialize_window("model1", start_layer=0)
        >>>
        >>> # Process layers sequentially
        >>> for i in range(num_layers):
        ...     layer = window.get_layer("model1", i)
        ...     output = layer(input_tensor)
        ...     window.slide_window()  # Advance window
    """

    def __init__(
        self,
        window_size: int = 5,
        config: Optional[SlidingWindowConfig] = None,
        layer_loader: Optional[Callable[[str, int], nn.Module]] = None,
    ):
        """
        Initialize the sliding window buffer.

        Args:
            window_size: Number of layers to keep in window
            config: Window configuration
            layer_loader: Optional callback to load layers
        """
        self.config = config or SlidingWindowConfig()
        self.window_size = min(
            max(window_size, self.config.min_window_size), self.config.max_window_size
        )
        self.layer_loader = layer_loader

        # Window state
        self._window: OrderedDict[str, WindowEntry] = OrderedDict()
        self._current_model_id: Optional[str] = None
        self._current_layer_index: int = 0
        self._total_layers: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = WindowStats()

        # Memory tracking
        self._current_memory_bytes = 0
        self._peak_memory_bytes = 0

        # Callbacks
        self._on_load_callbacks: List[Callable[[str, nn.Module], None]] = []
        self._on_evict_callbacks: List[Callable[[str], None]] = []

        logger.info(
            f"SlidingWindowBuffer initialized (window_size={self.window_size}, "
            f"overlap={self.config.overlap_layers})"
        )

    def register_load_callback(self, callback: Callable[[str, nn.Module], None]):
        """Register callback for when a layer is loaded."""
        self._on_load_callbacks.append(callback)

    def register_evict_callback(self, callback: Callable[[str], None]):
        """Register callback for when a layer is evicted."""
        self._on_evict_callbacks.append(callback)

    def _get_layer_id(self, model_id: str, layer_index: int) -> str:
        """Generate unique layer ID."""
        return f"{model_id}_layer_{layer_index}"

    def _get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage (worst of RAM/VRAM percent, total bytes).

        Uses torch.cuda.mem_get_info() for accurate VRAM readings that
        include non-PyTorch allocations (unlike memory_allocated() which
        only tracks PyTorch tensors).
        """
        if not PSUTIL_AVAILABLE:
            # Fallback: only check VRAM if psutil unavailable
            vram_percent = 0.0
            gpu_memory_used = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    used = total - free
                    gpu_memory_used += used
                    device_percent = (used / total * 100) if total > 0 else 0
                    vram_percent = max(vram_percent, device_percent)
            return vram_percent, gpu_memory_used

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # RAM usage
        system_memory = psutil.virtual_memory()
        ram_percent = system_memory.percent

        # VRAM usage — use mem_get_info for accurate readings
        vram_percent = 0.0
        gpu_memory_used = 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                gpu_memory_used += used
                device_percent = (used / total * 100) if total > 0 else 0
                vram_percent = max(vram_percent, device_percent)

        # Use worst-of (RAM vs VRAM) for conservative safety
        worst_percent = max(ram_percent, vram_percent)
        total_memory = memory_info.rss + gpu_memory_used

        return worst_percent, total_memory

    def _check_memory_pressure(self) -> Tuple[bool, bool]:
        """
        Check memory pressure levels.

        When memory_guard is available, uses its graduated pressure system
        (SAFE → ELEVATED → HIGH → CRITICAL → DEADLY) for decisions.
        Falls back to simple percentage comparison otherwise.

        Returns:
            Tuple of (should_evict, aggressive_eviction)
        """
        # Prefer memory_guard's pressure classification when available
        if _GUARD_AVAILABLE and _memory_guard is not None:
            pressure = _memory_guard.get_pressure()
            if pressure == MemoryPressure.HIGH:
                return True, False
            elif pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
                return True, True
            else:
                # SAFE or ELEVATED — no eviction needed
                return False, False

        # Fallback: simple percentage comparison
        percent_used, _ = self._get_memory_usage()
        should_evict = percent_used > self.config.memory_threshold_percent
        aggressive = percent_used > self.config.aggressive_eviction_threshold

        return should_evict, aggressive

    def _evict_oldest(self, count: int = 1):
        """Evict the oldest layers from the window."""
        with self._lock:
            evicted = 0
            while evicted < count and self._window:
                # Get oldest entry (first in OrderedDict)
                oldest_id, oldest_entry = self._window.popitem(last=False)

                if oldest_entry.layer is not None:
                    # Update memory tracking
                    self._current_memory_bytes -= oldest_entry.size_bytes
                    self._stats.current_memory_bytes = self._current_memory_bytes

                    # Notify callbacks
                    for callback in self._on_evict_callbacks:
                        try:
                            callback(oldest_id)
                        except Exception as e:
                            logger.warning(f"Evict callback error: {e}")

                    # Clear reference
                    oldest_entry.layer = None
                    oldest_entry.state = WindowState.EVICTED

                    evicted += 1
                    self._stats.layers_evicted += 1

                    logger.debug(f"Evicted layer {oldest_id}")

    def _load_layer_into_window(
        self, model_id: str, layer_index: int, priority: int = 5
    ) -> Optional[WindowEntry]:
        """Load a layer into the window."""
        layer_id = self._get_layer_id(model_id, layer_index)

        with self._lock:
            # Check if already in window
            if layer_id in self._window:
                entry = self._window[layer_id]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry

            # Check memory pressure before loading
            should_evict, aggressive = self._check_memory_pressure()
            if should_evict:
                # Evict more aggressively if needed
                evict_count = 2 if aggressive else 1
                self._evict_oldest(evict_count)

            start_time = time.time()

            try:
                ctx = (
                    _memory_guard.safe_allocate(
                        estimated_ram_gb=0,
                        estimated_vram_gb=1.0,
                        operation="sliding_window_load_layer",
                    )
                    if (_GUARD_AVAILABLE and _memory_guard is not None)
                    else nullcontext()
                )
                with ctx:
                    # Load the layer
                    if self.layer_loader:
                        layer = self.layer_loader(model_id, layer_index)
                    else:
                        layer = self._default_layer_loader(model_id, layer_index)

                    if layer is None:
                        return None

                    # Calculate layer size
                    size_bytes = self._get_layer_size(layer)

                    # Check if we need to evict to make room
                    max_memory = self._get_max_memory_bytes()
                    if self._current_memory_bytes + size_bytes > max_memory:
                        self._evict_oldest(1)

                    # Create window entry
                    entry = WindowEntry(
                        layer_id=layer_id,
                        layer_index=layer_index,
                        model_id=model_id,
                        layer=layer,
                        state=WindowState.READY,
                        size_bytes=size_bytes,
                        loaded_at=time.time(),
                        priority=priority,
                    )

                    # Add to window
                    self._window[layer_id] = entry
                    self._current_memory_bytes += size_bytes

                    # Update peak memory
                    if self._current_memory_bytes > self._peak_memory_bytes:
                        self._peak_memory_bytes = self._current_memory_bytes
                        self._stats.peak_memory_bytes = self._peak_memory_bytes

                    # Update stats
                    load_time_ms = (time.time() - start_time) * 1000
                    self._stats.record_load(load_time_ms)
                    self._stats.current_memory_bytes = self._current_memory_bytes

                    # Notify callbacks
                    for callback in self._on_load_callbacks:
                        try:
                            callback(layer_id, layer)
                        except Exception as e:
                            logger.warning(f"Load callback error: {e}")

                    logger.debug(
                        f"Loaded layer {layer_id} ({size_bytes / 1e6:.2f} MB, "
                        f"{load_time_ms:.2f}ms)"
                    )

                    return entry

            except Exception as e:
                logger.error(f"Failed to load layer {layer_id}: {e}")
                return None

    def _default_layer_loader(
        self, model_id: str, layer_index: int
    ) -> Optional[nn.Module]:
        """Default layer loader - should be overridden."""
        logger.warning("No layer_loader provided, cannot load layers")
        return None

    def _get_layer_size(self, layer: nn.Module) -> int:
        """Calculate memory size of a layer."""
        total_size = 0
        for param in layer.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in layer.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size

    def _get_max_memory_bytes(self) -> int:
        """Get maximum allowed memory for the window."""
        if self.config.max_memory_gb:
            return int(self.config.max_memory_gb * 1e9)

        # Default: use 50% of available system memory
        if PSUTIL_AVAILABLE:
            system_memory = psutil.virtual_memory()
            return int(system_memory.available * 0.5)
        else:
            # Fallback: assume 8GB available
            return int(8 * 1e9 * 0.5)

    def initialize_window(
        self, model_id: str, start_layer: int = 0, total_layers: Optional[int] = None
    ):
        """
        Initialize the window for a new model.

        Args:
            model_id: Model identifier
            start_layer: Starting layer index
            total_layers: Total number of layers in model
        """
        with self._lock:
            # Clear existing window
            self.clear_window()

            self._current_model_id = model_id
            self._current_layer_index = start_layer
            self._total_layers = total_layers or float("inf")

            # Preload initial window
            end_layer = min(start_layer + self.window_size, self._total_layers)
            for i in range(start_layer, end_layer):
                self._load_layer_into_window(model_id, i, priority=10)

            logger.info(
                f"Window initialized for {model_id}: layers {start_layer}-{end_layer - 1}"
            )

    def get_layer(
        self, model_id: str, layer_index: int, auto_advance: bool = False
    ) -> Optional[nn.Module]:
        """
        Get a layer from the window.

        Args:
            model_id: Model identifier
            layer_index: Layer index to retrieve
            auto_advance: Whether to automatically slide window after access

        Returns:
            The layer module, or None if not available
        """
        layer_id = self._get_layer_id(model_id, layer_index)

        with self._lock:
            # Check if in window
            if layer_id in self._window:
                entry = self._window[layer_id]
                entry.last_accessed = time.time()
                entry.access_count += 1
                entry.state = WindowState.ACTIVE

                # Move to end (most recent)
                self._window.move_to_end(layer_id)

                self._stats.record_access(hit=True)

                if auto_advance and layer_index == self._current_layer_index:
                    self.slide_window()

                return entry.layer

            self._stats.record_access(hit=False)

        # Not in window - load it
        logger.debug(f"Layer {layer_id} not in window, loading on-demand")
        entry = self._load_layer_into_window(model_id, layer_index, priority=10)

        if entry:
            entry.state = WindowState.ACTIVE

            if auto_advance:
                self.slide_window()

            return entry.layer

        return None

    def slide_window(self, steps: int = 1):
        """
        Slide the window forward.

        Args:
            steps: Number of layers to advance
        """
        with self._lock:
            if not self._current_model_id:
                return

            self._current_layer_index += steps

            # Determine which layers to keep (overlap)
            keep_indices = set()
            for i in range(self.config.overlap_layers):
                keep_idx = self._current_layer_index + self.window_size - 1 - i
                keep_indices.add(keep_idx)

            # Evict layers that are too old
            to_evict = []
            for layer_id, entry in self._window.items():
                if (
                    entry.layer_index
                    < self._current_layer_index - self.config.overlap_layers
                ):
                    to_evict.append(layer_id)

            # Evict oldest layers first
            for layer_id in to_evict:
                self._evict_layer(layer_id)

            # Preload new layers at the front of the window
            preload_start = (
                self._current_layer_index
                + self.window_size
                - self.config.overlap_layers
            )
            preload_end = min(
                preload_start + steps + self.config.preload_ahead, self._total_layers
            )

            for i in range(preload_start, preload_end):
                layer_id = self._get_layer_id(self._current_model_id, i)
                if layer_id not in self._window:
                    # Load with lower priority (background)
                    self._load_layer_into_window(self._current_model_id, i, priority=3)

            self._stats.window_slides += 1

            logger.debug(
                f"Window slid to layer {self._current_layer_index}, "
                f"loaded {preload_end - preload_start} new layers"
            )

    def _evict_layer(self, layer_id: str):
        """Evict a specific layer from the window."""
        with self._lock:
            if layer_id not in self._window:
                return

            entry = self._window.pop(layer_id)

            if entry.layer is not None:
                self._current_memory_bytes -= entry.size_bytes
                self._stats.current_memory_bytes = self._current_memory_bytes

                # Notify callbacks
                for callback in self._on_evict_callbacks:
                    try:
                        callback(layer_id)
                    except Exception as e:
                        logger.warning(f"Evict callback error: {e}")

                entry.layer = None
                entry.state = WindowState.EVICTED

                self._stats.layers_evicted += 1

    def adjust_window_size(self, new_size: int):
        """
        Dynamically adjust the window size.

        Args:
            new_size: New window size
        """
        with self._lock:
            old_size = self.window_size
            self.window_size = min(
                max(new_size, self.config.min_window_size), self.config.max_window_size
            )

            if self.window_size != old_size:
                self._stats.window_resizes += 1

                # Evict excess layers if shrinking
                if self.window_size < old_size:
                    excess = len(self._window) - self.window_size
                    if excess > 0:
                        self._evict_oldest(excess)

                logger.info(f"Window size adjusted: {old_size} -> {self.window_size}")

    def optimize_for_memory(self):
        """Optimize window size based on current memory pressure.

        Uses MemoryGuard graduated pressure levels when available:
            CRITICAL/DEADLY → aggressive shrink (-2)
            HIGH → moderate shrink (-1)
            SAFE/ELEVATED with <50% usage → grow (+1)
        """
        if _GUARD_AVAILABLE and _memory_guard is not None:
            pressure = _memory_guard.get_pressure()
            if pressure in (MemoryPressure.CRITICAL, MemoryPressure.DEADLY):
                # Aggressively reduce window
                new_size = max(self.config.min_window_size, self.window_size - 2)
                self.adjust_window_size(new_size)
            elif pressure == MemoryPressure.HIGH:
                # Moderately reduce window
                new_size = max(self.config.min_window_size, self.window_size - 1)
                self.adjust_window_size(new_size)
            elif (
                pressure == MemoryPressure.SAFE
                and self.window_size < self.config.max_window_size
            ):
                # Safe to grow
                new_size = min(self.config.max_window_size, self.window_size + 1)
                self.adjust_window_size(new_size)
            return

        # Fallback: percentage-based optimization
        percent_used, _ = self._get_memory_usage()

        if percent_used > self.config.aggressive_eviction_threshold:
            # Aggressively reduce window
            new_size = max(self.config.min_window_size, self.window_size - 2)
            self.adjust_window_size(new_size)
        elif percent_used > self.config.memory_threshold_percent:
            # Moderately reduce window
            new_size = max(self.config.min_window_size, self.window_size - 1)
            self.adjust_window_size(new_size)
        elif percent_used < 50 and self.window_size < self.config.max_window_size:
            # Can increase window
            new_size = min(self.config.max_window_size, self.window_size + 1)
            self.adjust_window_size(new_size)

    def get_window_layers(self) -> List[int]:
        """Get list of layer indices currently in window."""
        with self._lock:
            return sorted([entry.layer_index for entry in self._window.values()])

    def is_layer_in_window(self, model_id: str, layer_index: int) -> bool:
        """Check if a layer is currently in the window."""
        layer_id = self._get_layer_id(model_id, layer_index)
        with self._lock:
            return layer_id in self._window

    def get_layer_state(self, model_id: str, layer_index: int) -> Optional[WindowState]:
        """Get the state of a layer in the window."""
        layer_id = self._get_layer_id(model_id, layer_index)
        with self._lock:
            if layer_id in self._window:
                return self._window[layer_id].state
            return None

    def clear_window(self):
        """Clear all layers from the window."""
        with self._lock:
            for layer_id in list(self._window.keys()):
                self._evict_layer(layer_id)

            self._window.clear()
            self._current_memory_bytes = 0
            self._current_model_id = None
            self._current_layer_index = 0

            logger.info("Window cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get window statistics."""
        with self._lock:
            stats = self._stats.to_dict()
            stats["window_size"] = self.window_size
            stats["current_layers"] = len(self._window)
            stats["window_range"] = self.get_window_layers()
            return stats

    def print_stats(self):
        """Print window statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("Sliding Window Buffer Statistics")
        print("=" * 60)
        print(f"Window Size: {stats['window_size']}")
        print(f"Current Layers: {stats['current_layers']}")
        print(f"Window Range: {stats['window_range']}")
        print(f"Window Slides: {stats['window_slides']}")
        print(f"Layers Loaded: {stats['layers_loaded']}")
        print(f"Layers Evicted: {stats['layers_evicted']}")
        print(f"Cache Hit Ratio: {stats['hit_ratio']:.2%}")
        print(f"Avg Load Time: {stats['avg_load_time_ms']:.2f}ms")
        print(f"Peak Memory: {stats['peak_memory_gb']:.2f} GB")
        print(f"Current Memory: {stats['current_memory_gb']:.2f} GB")
        print(f"Window Resizes: {stats['window_resizes']}")
        print("=" * 60 + "\n")


class AdaptiveSlidingWindow(SlidingWindowBuffer):
    """
    Adaptive sliding window that automatically adjusts based on workload.

    This extends the base sliding window with:
    - Pattern recognition for prefetch optimization
    - Dynamic window sizing based on hit rates
    - Workload-specific optimizations
    """

    def __init__(
        self,
        window_size: int = 5,
        config: Optional[SlidingWindowConfig] = None,
        layer_loader: Optional[Callable[[str, int], nn.Module]] = None,
    ):
        super().__init__(window_size, config, layer_loader)

        # Pattern tracking
        self._access_pattern: List[int] = []
        self._max_pattern_history = 100
        self._pattern_window = 10

        # Adaptive parameters
        self._hit_rate_history: List[float] = []
        self._adaptation_interval = 50
        self._access_count = 0

    def get_layer(
        self, model_id: str, layer_index: int, auto_advance: bool = False
    ) -> Optional[nn.Module]:
        """Get layer with pattern tracking."""
        # Track access pattern
        self._access_pattern.append(layer_index)
        if len(self._access_pattern) > self._max_pattern_history:
            self._access_pattern.pop(0)

        self._access_count += 1

        # Periodic adaptation
        if self._access_count % self._adaptation_interval == 0:
            self._adapt_window_size()

        return super().get_layer(model_id, layer_index, auto_advance)

    def _adapt_window_size(self):
        """Adapt window size based on performance metrics."""
        stats = self.get_stats()
        hit_ratio = stats["hit_ratio"]

        self._hit_rate_history.append(hit_ratio)
        if len(self._hit_rate_history) > 10:
            self._hit_rate_history.pop(0)

        # Calculate trend
        if len(self._hit_rate_history) >= 3:
            recent_avg = sum(self._hit_rate_history[-3:]) / 3
            older_avg = sum(self._hit_rate_history[:-3]) / max(
                1, len(self._hit_rate_history) - 3
            )

            if recent_avg < 0.7 and older_avg > 0.8:
                # Hit rate dropping, might need larger window
                self.adjust_window_size(self.window_size + 1)
            elif recent_avg > 0.95 and self.window_size > self.config.min_window_size:
                # Very high hit rate, can try smaller window
                self.adjust_window_size(self.window_size - 1)

    def predict_next_layers(self, count: int = 3) -> List[int]:
        """Predict next likely layers based on access pattern."""
        if len(self._access_pattern) < 2:
            # Default: sequential prediction
            return list(
                range(self._current_layer_index, self._current_layer_index + count)
            )

        # Analyze recent pattern
        recent = self._access_pattern[-self._pattern_window :]

        # Check for sequential pattern
        is_sequential = all(
            recent[i + 1] - recent[i] == 1 for i in range(len(recent) - 1)
        )

        if is_sequential:
            # Continue sequential pattern
            last = recent[-1]
            return list(range(last + 1, last + 1 + count))

        # Check for strided pattern
        if len(recent) >= 3:
            stride = recent[-1] - recent[-2]
            if all(recent[i + 1] - recent[i] == stride for i in range(len(recent) - 1)):
                # Predict strided pattern
                last = recent[-1]
                return [last + stride * (i + 1) for i in range(count)]

        # Default to sequential
        last = recent[-1] if recent else self._current_layer_index
        return list(range(last + 1, last + 1 + count))

    def preload_predicted_layers(self):
        """Preload layers predicted by pattern analysis."""
        if not self._current_model_id:
            return

        predicted = self.predict_next_layers(self.config.preload_ahead)

        for layer_index in predicted:
            layer_id = self._get_layer_id(self._current_model_id, layer_index)
            if layer_id not in self._window:
                self._load_layer_into_window(
                    self._current_model_id,
                    layer_index,
                    priority=2,  # Low priority for predicted layers
                )


# Convenience function for creating window buffer
def create_sliding_window(
    window_size: int = 5, adaptive: bool = True, **kwargs
) -> SlidingWindowBuffer:
    """
    Create a sliding window buffer.

    Args:
        window_size: Number of layers to keep in window
        adaptive: Whether to use adaptive window sizing
        **kwargs: Additional configuration options

    Returns:
        Configured sliding window buffer
    """
    config = SlidingWindowConfig(**kwargs)

    if adaptive:
        return AdaptiveSlidingWindow(window_size, config)
    else:
        return SlidingWindowBuffer(window_size, config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Sliding Window Buffer")
    print("=" * 60)

    # Create mock layer loader
    def mock_loader(model_id: str, layer_index: int) -> nn.Module:
        # Simulate loading time
        time.sleep(0.01)
        return nn.Linear(1024, 1024)

    # Create sliding window
    window = AdaptiveSlidingWindow(window_size=5, layer_loader=mock_loader)

    # Initialize for model with 20 layers
    window.initialize_window("test_model", start_layer=0, total_layers=20)

    # Simulate sequential processing
    print("\nSimulating sequential layer processing...")
    for i in range(10):
        layer = window.get_layer("test_model", i, auto_advance=True)
        print(f"  Layer {i}: loaded={layer is not None}")

        # Every few layers, show window state
        if i % 3 == 0:
            layers_in_window = window.get_window_layers()
            print(f"    Window layers: {layers_in_window}")

    # Print final stats
    window.print_stats()

    # Cleanup
    window.clear_window()

    print("\n" + "=" * 60)
