"""
Smart Layer Prefetching Engine for Nexus SLI

Implements predictive layer loading based on access patterns with:
- Multi-layer lookahead (3-5 layers)
- Background thread pool for parallel loading
- Integration with sliding window buffer
- Pattern recognition and prediction
- Adaptive prefetch depth

Author: Nexus Team
"""

import os
import time
import threading
import logging
from typing import Dict, Optional, Any, List, Set, Tuple, Callable, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np

from .sliding_window_buffer import SlidingWindowBuffer, WindowState
from .exceptions import SLIError

logger = logging.getLogger(__name__)

# Import centralized memory guard for WSL-aware thresholds
try:
    from nexus.utils.memory_guard import guard as _memory_guard, MemoryPressure

    _GUARD_AVAILABLE = True
except ImportError:
    _memory_guard = None
    MemoryPressure = None  # type: ignore[assignment, misc]
    _GUARD_AVAILABLE = False


def _get_guard_threshold(key: str, fallback: float) -> float:
    """Pull a threshold from memory_guard if available, else use fallback."""
    if _GUARD_AVAILABLE and _memory_guard is not None:
        thresholds = _memory_guard.get_thresholds()
        return thresholds.get(key, fallback)
    return fallback


class PrefetchPattern(Enum):
    """Detected access patterns."""

    SEQUENTIAL = "sequential"
    STRIDED = "strided"
    RANDOM = "random"
    BURST = "burst"
    TEMPORAL = "temporal"


class PrefetchPriority(Enum):
    """Prefetch priority levels."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    SPECULATIVE = 4


@dataclass
class PrefetchStats:
    """Statistics for prefetch operations."""

    total_prefetches: int = 0
    successful_prefetches: int = 0
    failed_prefetches: int = 0
    cache_hits: int = 0
    pattern_predictions: int = 0
    pattern_hits: int = 0
    avg_prefetch_time_ms: float = 0.0
    total_bytes_prefetched: int = 0
    current_lookahead: int = 3
    pattern_accuracy: float = 0.0

    def record_prefetch(self, success: bool, time_ms: float, bytes_loaded: int = 0):
        """Record a prefetch operation."""
        self.total_prefetches += 1
        if success:
            self.successful_prefetches += 1
            self.total_bytes_prefetched += bytes_loaded
        else:
            self.failed_prefetches += 1

        # Update running average
        self.avg_prefetch_time_ms = (
            self.avg_prefetch_time_ms * (self.total_prefetches - 1) + time_ms
        ) / self.total_prefetches

    def record_pattern_hit(self, hit: bool):
        """Record pattern prediction accuracy."""
        self.pattern_predictions += 1
        if hit:
            self.pattern_hits += 1

        if self.pattern_predictions > 0:
            self.pattern_accuracy = self.pattern_hits / self.pattern_predictions

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_prefetches": self.total_prefetches,
            "successful_prefetches": self.successful_prefetches,
            "failed_prefetches": self.failed_prefetches,
            "success_rate": self.successful_prefetches / max(1, self.total_prefetches),
            "cache_hits": self.cache_hits,
            "pattern_predictions": self.pattern_predictions,
            "pattern_hits": self.pattern_hits,
            "pattern_accuracy": self.pattern_accuracy,
            "avg_prefetch_time_ms": self.avg_prefetch_time_ms,
            "total_bytes_prefetched_gb": self.total_bytes_prefetched / 1e9,
            "current_lookahead": self.current_lookahead,
        }


@dataclass
class LayerAccess:
    """Record of a layer access."""

    layer_index: int
    timestamp: float
    model_id: str
    access_type: str = "forward"  # forward, backward, etc.


@dataclass
class PrefetchConfig:
    """Configuration for prefetch engine.

    memory_threshold_percent defaults to WSL-aware value from memory_guard
    when available, falling back to 85% otherwise.
    """

    min_lookahead: int = 3
    max_lookahead: int = 5
    default_lookahead: int = 3
    thread_pool_size: int = 8
    max_concurrent_prefetches: int = 6
    pattern_window_size: int = 20
    enable_adaptive_lookahead: bool = True
    enable_pattern_recognition: bool = True
    prefetch_timeout: float = 30.0
    memory_threshold_percent: float = field(
        default_factory=lambda: _get_guard_threshold("vram_high_percent", 85.0)
    )
    pattern_confidence_threshold: float = 0.7
    burst_detection_threshold: int = 3


class PatternPredictor:
    """
    Predicts future layer accesses based on historical patterns.

    Detects:
    - Sequential access (layer 0, 1, 2, 3...)
    - Strided access (layer 0, 2, 4, 6...)
    - Burst access (rapid repeated access to same layer)
    - Temporal patterns (time-based access patterns)
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.access_history: Deque[LayerAccess] = deque(maxlen=window_size)
        self.current_pattern = PrefetchPattern.SEQUENTIAL
        self.pattern_confidence = 1.0
        self.stride = 1
        self._burst_counter: Dict[int, int] = defaultdict(int)

    def record_access(self, access: LayerAccess):
        """Record a layer access."""
        self.access_history.append(access)
        self._burst_counter[access.layer_index] += 1

        # Update pattern detection
        self._detect_pattern()

    def _detect_pattern(self):
        """Detect the current access pattern."""
        if len(self.access_history) < 3:
            return

        indices = [a.layer_index for a in self.access_history]

        # Check for sequential pattern
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]

        if len(set(diffs)) == 1 and diffs[0] == 1:
            self.current_pattern = PrefetchPattern.SEQUENTIAL
            self.pattern_confidence = min(1.0, len(indices) / 10)
            self.stride = 1
            return

        # Check for strided pattern
        if len(set(diffs)) == 1 and diffs[0] != 0:
            self.current_pattern = PrefetchPattern.STRIDED
            self.pattern_confidence = min(1.0, len(indices) / 10)
            self.stride = diffs[0]
            return

        # Check for burst pattern
        max_burst = max(self._burst_counter.values()) if self._burst_counter else 0
        if max_burst >= 3:
            self.current_pattern = PrefetchPattern.BURST
            self.pattern_confidence = min(1.0, max_burst / 10)
            return

        # Random pattern
        self.current_pattern = PrefetchPattern.RANDOM
        self.pattern_confidence = 0.0
        self.stride = 1

    def predict_next_layers(self, count: int = 5) -> List[int]:
        """Predict next N layer indices."""
        if not self.access_history:
            return list(range(count))

        last_idx = self.access_history[-1].layer_index

        if self.current_pattern == PrefetchPattern.SEQUENTIAL:
            return [last_idx + i + 1 for i in range(count)]

        elif self.current_pattern == PrefetchPattern.STRIDED:
            return [last_idx + self.stride * (i + 1) for i in range(count)]

        elif self.current_pattern == PrefetchPattern.BURST:
            # Predict same layer for burst
            most_accessed = max(self._burst_counter.items(), key=lambda x: x[1])[0]
            return [most_accessed] * count

        else:  # RANDOM
            # Fallback to sequential
            return [last_idx + i + 1 for i in range(count)]

    def get_pattern_info(self) -> Dict[str, Any]:
        """Get current pattern information."""
        return {
            "pattern": self.current_pattern.value,
            "confidence": self.pattern_confidence,
            "stride": self.stride,
            "history_length": len(self.access_history),
        }


class PrefetchEngine:
    """
    Smart Layer Prefetching Engine.

    Features:
    - Predictive layer loading based on access patterns
    - Multi-layer lookahead (3-5 layers)
    - Background thread pool for parallel loading
    - Integration with sliding window buffer
    - Adaptive lookahead depth based on hit rates

    Example:
        >>> engine = PrefetchEngine(sliding_window=my_window)
        >>> engine.start()
        >>>
        >>> # Record accesses and trigger prefetches
        >>> for i in range(num_layers):
        >>>     engine.record_access("model1", i)
        >>>     layer = sliding_window.get_layer("model1", i)
        >>>
        >>> # Get prefetched layers
        >>> prefetched = engine.get_prefetched_layer("model1_layer_5")
    """

    def __init__(
        self,
        sliding_window: Optional[SlidingWindowBuffer] = None,
        layer_loader: Optional[Callable[[str, int], nn.Module]] = None,
        config: Optional[PrefetchConfig] = None,
    ):
        """
        Initialize prefetch engine.

        Args:
            sliding_window: Optional sliding window buffer to integrate with
            layer_loader: Callback function to load layers (model_id, layer_idx) -> layer
            config: Prefetch configuration
        """
        self.config = config or PrefetchConfig()
        self.sliding_window = sliding_window
        self.layer_loader = layer_loader

        # Pattern predictor
        self.pattern_predictor = PatternPredictor(
            window_size=self.config.pattern_window_size
        )

        # Thread pool for parallel prefetching
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.thread_pool_size, thread_name_prefix="prefetch-"
        )

        # Prefetch state
        self._prefetch_buffer: Dict[str, nn.Module] = {}
        self._in_progress: Dict[str, Future] = {}
        self._prefetched_ids: Set[str] = set()
        self._buffer_lock = threading.RLock()

        # Statistics
        self._stats = PrefetchStats()
        self._current_lookahead = self.config.default_lookahead

        # Control
        self._shutdown = False
        self._active = False

        # Model tracking
        self._current_model_id: Optional[str] = None
        self._total_layers: int = 0

        logger.info(
            f"PrefetchEngine initialized (lookahead={self._current_lookahead}, "
            f"threads={self.config.thread_pool_size})"
        )

    def start(self):
        """Start the prefetch engine."""
        self._active = True
        self._shutdown = False
        logger.info("PrefetchEngine started")

    def stop(self):
        """Stop the prefetch engine."""
        self._active = False
        self._shutdown = True
        self.executor.shutdown(wait=True)
        logger.info("PrefetchEngine stopped")

    def _get_layer_id(self, model_id: str, layer_index: int) -> str:
        """Generate unique layer ID."""
        return f"{model_id}_layer_{layer_index}"

    def record_access(
        self, model_id: str, layer_index: int, access_type: str = "forward"
    ):
        """
        Record a layer access and trigger predictive prefetching.

        Args:
            model_id: Model identifier
            layer_index: Layer index accessed
            access_type: Type of access (forward, backward, etc.)
        """
        if not self._active:
            return

        access = LayerAccess(
            layer_index=layer_index,
            timestamp=time.time(),
            model_id=model_id,
            access_type=access_type,
        )

        # Update pattern predictor
        old_pattern = self.pattern_predictor.current_pattern
        self.pattern_predictor.record_access(access)

        # Check if pattern changed
        if old_pattern != self.pattern_predictor.current_pattern:
            logger.debug(
                f"Pattern changed: {old_pattern.value} -> "
                f"{self.pattern_predictor.current_pattern.value} "
                f"(confidence: {self.pattern_predictor.pattern_confidence:.2f})"
            )

        # Update current model tracking
        self._current_model_id = model_id

        # Trigger predictive prefetch
        self._trigger_prefetch(model_id, layer_index)

        # Adaptive lookahead adjustment
        if self.config.enable_adaptive_lookahead:
            self._adapt_lookahead()

    def _trigger_prefetch(self, model_id: str, current_layer: int):
        """Trigger prefetch based on pattern prediction."""
        if not self.config.enable_pattern_recognition:
            # Simple sequential prefetch
            layers_to_prefetch = [
                current_layer + i + 1 for i in range(self._current_lookahead)
            ]
        else:
            # Pattern-based prefetch
            layers_to_prefetch = self.pattern_predictor.predict_next_layers(
                self._current_lookahead
            )

        # Filter valid layers
        layers_to_prefetch = [
            idx
            for idx in layers_to_prefetch
            if idx >= 0 and (self._total_layers == 0 or idx < self._total_layers)
        ]

        # Prefetch layers in parallel
        for layer_idx in layers_to_prefetch:
            self._prefetch_layer_async(model_id, layer_idx)

    def _prefetch_layer_async(self, model_id: str, layer_index: int):
        """Prefetch a layer asynchronously.

        Skips prefetch if memory pressure is HIGH or above to avoid
        triggering OOM during speculative layer loads.
        """
        # Check memory pressure before prefetching — don't add load under pressure
        if _GUARD_AVAILABLE and _memory_guard is not None:
            pressure = _memory_guard.get_pressure()
            if pressure in (
                MemoryPressure.HIGH,
                MemoryPressure.CRITICAL,
                MemoryPressure.DEADLY,
            ):
                logger.debug(
                    f"Skipping prefetch for layer {layer_index} — "
                    f"memory pressure is {pressure.name}"
                )
                return

        layer_id = self._get_layer_id(model_id, layer_index)

        with self._buffer_lock:
            # Skip if already prefetched or in progress
            if layer_id in self._prefetched_ids or layer_id in self._in_progress:
                return

            # Skip if in sliding window
            if self.sliding_window and self.sliding_window.is_layer_in_window(
                model_id, layer_index
            ):
                return

            self._prefetched_ids.add(layer_id)

        # Submit prefetch task
        future = self.executor.submit(self._load_layer, model_id, layer_index, layer_id)

        with self._buffer_lock:
            self._in_progress[layer_id] = future

    def _load_layer(
        self, model_id: str, layer_index: int, layer_id: str
    ) -> Optional[nn.Module]:
        """Load a layer into the prefetch buffer."""
        start_time = time.time()

        try:
            ctx = (
                _memory_guard.safe_allocate(
                    estimated_ram_gb=0,
                    estimated_vram_gb=1.0,
                    operation="prefetch_load_layer",
                )
                if (_GUARD_AVAILABLE and _memory_guard is not None)
                else nullcontext()
            )
            with ctx:
                # Use sliding window loader if available
                if self.sliding_window and hasattr(
                    self.sliding_window, "_load_layer_into_window"
                ):
                    entry = self.sliding_window._load_layer_into_window(
                        model_id,
                        layer_index,
                        priority=3,  # Low priority for prefetch
                    )
                    layer = entry.layer if entry else None
                elif self.layer_loader:
                    layer = self.layer_loader(model_id, layer_index)
                else:
                    logger.warning(
                        f"No layer loader available for prefetch of {layer_id}"
                    )
                    return None

                load_time_ms = (time.time() - start_time) * 1000

                if layer is not None:
                    # Calculate size
                    size_bytes = sum(
                        p.numel() * p.element_size() for p in layer.parameters()
                    )

                    with self._buffer_lock:
                        self._prefetch_buffer[layer_id] = layer
                        self._in_progress.pop(layer_id, None)

                    self._stats.record_prefetch(True, load_time_ms, size_bytes)

                    logger.debug(
                        f"Prefetched layer {layer_id} ({size_bytes / 1e6:.2f}MB, "
                        f"{load_time_ms:.2f}ms)"
                    )

                    return layer
                else:
                    self._stats.record_prefetch(False, load_time_ms)
                    with self._buffer_lock:
                        self._in_progress.pop(layer_id, None)
                    return None

        except Exception as e:
            load_time_ms = (time.time() - start_time) * 1000
            self._stats.record_prefetch(False, load_time_ms)

            with self._buffer_lock:
                self._in_progress.pop(layer_id, None)

            logger.warning(f"Failed to prefetch layer {layer_id}: {e}")
            return None

    def get_prefetched_layer(self, layer_id: str) -> Optional[nn.Module]:
        """
        Get a layer from the prefetch buffer.

        Args:
            layer_id: Layer identifier

        Returns:
            Layer if available in prefetch buffer, None otherwise
        """
        with self._buffer_lock:
            layer = self._prefetch_buffer.pop(layer_id, None)
            if layer is not None:
                self._stats.cache_hits += 1
                self._prefetched_ids.discard(layer_id)
            return layer

    def wait_for_prefetch(
        self, layer_ids: List[str], timeout: Optional[float] = None
    ) -> Dict[str, nn.Module]:
        """
        Wait for multiple prefetches to complete.

        Args:
            layer_ids: List of layer IDs to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Dictionary of completed prefetches
        """
        results = {}
        start_time = time.time()
        remaining = set(layer_ids)

        while remaining and (timeout is None or time.time() - start_time < timeout):
            for layer_id in list(remaining):
                with self._buffer_lock:
                    if layer_id in self._prefetch_buffer:
                        results[layer_id] = self._prefetch_buffer.pop(layer_id)
                        remaining.remove(layer_id)
                        self._prefetched_ids.discard(layer_id)
                    elif layer_id in self._in_progress:
                        future = self._in_progress.get(layer_id)
                        if future and future.done():
                            try:
                                layer = future.result()
                                if layer:
                                    results[layer_id] = layer
                            except Exception as e:
                                logger.error(f"Prefetch failed for {layer_id}: {e}")
                            remaining.remove(layer_id)
                            self._in_progress.pop(layer_id, None)

                if not remaining:
                    break

            if remaining:
                time.sleep(0.001)

        return results

    def _adapt_lookahead(self):
        """Adaptively adjust lookahead depth based on hit rates."""
        if not self.config.enable_adaptive_lookahead:
            return

        # Check if we have enough data
        if self._stats.total_prefetches < 20:
            return

        success_rate = self._stats.successful_prefetches / self._stats.total_prefetches
        pattern_confidence = self.pattern_predictor.pattern_confidence

        # Adjust lookahead based on success rate and pattern confidence
        if (
            success_rate > 0.8
            and pattern_confidence > self.config.pattern_confidence_threshold
        ):
            # High success rate - can increase lookahead
            if self._current_lookahead < self.config.max_lookahead:
                self._current_lookahead += 1
                self._stats.current_lookahead = self._current_lookahead
                logger.debug(f"Increased lookahead to {self._current_lookahead}")

        elif success_rate < 0.5 or pattern_confidence < 0.3:
            # Low success rate - decrease lookahead
            if self._current_lookahead > self.config.min_lookahead:
                self._current_lookahead -= 1
                self._stats.current_lookahead = self._current_lookahead
                logger.debug(f"Decreased lookahead to {self._current_lookahead}")

    def prefetch_layers_parallel(
        self,
        model_id: str,
        layer_indices: List[int],
        priority: PrefetchPriority = PrefetchPriority.NORMAL,
    ) -> List[Future]:
        """
        Prefetch multiple layers in parallel.

        Args:
            model_id: Model identifier
            layer_indices: List of layer indices to prefetch
            priority: Prefetch priority

        Returns:
            List of futures for the prefetch operations
        """
        futures = []

        for layer_idx in layer_indices:
            layer_id = self._get_layer_id(model_id, layer_idx)

            with self._buffer_lock:
                if layer_id in self._prefetched_ids or layer_id in self._in_progress:
                    continue
                self._prefetched_ids.add(layer_id)

            future = self.executor.submit(
                self._load_layer, model_id, layer_idx, layer_id
            )

            with self._buffer_lock:
                self._in_progress[layer_id] = future

            futures.append(future)

        return futures

    def set_model_info(self, model_id: str, total_layers: int):
        """
        Set current model information for smarter prefetching.

        Args:
            model_id: Model identifier
            total_layers: Total number of layers in model
        """
        self._current_model_id = model_id
        self._total_layers = total_layers

        # Clear old prefetches
        self.clear_buffer()

        logger.debug(f"Set model info: {model_id} with {total_layers} layers")

    def clear_buffer(self):
        """Clear the prefetch buffer."""
        with self._buffer_lock:
            self._prefetch_buffer.clear()
            self._prefetched_ids.clear()

            # Cancel in-progress prefetches
            for future in self._in_progress.values():
                if not future.done():
                    future.cancel()

            self._in_progress.clear()

        logger.debug("Prefetch buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get prefetch statistics."""
        with self._buffer_lock:
            stats = self._stats.to_dict()
            stats["pattern_info"] = self.pattern_predictor.get_pattern_info()
            stats["buffer_size"] = len(self._prefetch_buffer)
            stats["in_progress"] = len(self._in_progress)
            stats["prefetched_ids"] = len(self._prefetched_ids)
            return stats

    def get_buffer_state(self) -> Dict[str, Any]:
        """Get current buffer state."""
        with self._buffer_lock:
            return {
                "buffered_layers": list(self._prefetch_buffer.keys()),
                "in_progress": list(self._in_progress.keys()),
                "prefetched_ids": list(self._prefetched_ids),
            }


def create_prefetch_engine(
    sliding_window: Optional[SlidingWindowBuffer] = None,
    layer_loader: Optional[Callable[[str, int], nn.Module]] = None,
    lookahead: int = 3,
    thread_pool_size: int = 8,
    **kwargs,
) -> PrefetchEngine:
    """
    Create a prefetch engine with common configurations.

    Args:
        sliding_window: Optional sliding window buffer
        layer_loader: Layer loader callback
        lookahead: Number of layers to prefetch ahead
        thread_pool_size: Number of threads in pool
        **kwargs: Additional config options

    Returns:
        Configured PrefetchEngine
    """
    config = PrefetchConfig(
        default_lookahead=lookahead, thread_pool_size=thread_pool_size, **kwargs
    )

    engine = PrefetchEngine(
        sliding_window=sliding_window, layer_loader=layer_loader, config=config
    )

    return engine


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Smart Layer Prefetching Engine")
    print("=" * 60)

    # Mock layer loader
    def mock_loader(model_id: str, layer_idx: int) -> nn.Module:
        time.sleep(0.01)  # Simulate loading time
        return nn.Linear(1024, 1024)

    # Create prefetch engine
    engine = create_prefetch_engine(
        layer_loader=mock_loader, lookahead=3, thread_pool_size=4
    )

    engine.start()

    # Simulate sequential access pattern
    print("\nSimulating sequential access pattern...")
    for i in range(10):
        engine.record_access("test_model", i)
        time.sleep(0.005)  # Small delay between accesses

    # Check stats
    stats = engine.get_stats()
    print(f"\nPattern detected: {stats['pattern_info']['pattern']}")
    print(f"Pattern confidence: {stats['pattern_info']['confidence']:.2f}")
    print(f"Total prefetches: {stats['total_prefetches']}")
    print(f"Successful prefetches: {stats['successful_prefetches']}")

    # Stop engine
    engine.stop()

    print("\n" + "=" * 60)
