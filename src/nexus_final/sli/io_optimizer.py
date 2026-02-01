"""
I/O Optimization Module for Nexus SLI (Selective Layer Inference)

This module implements I/O optimization strategies:
- Async layer pre-fetching
- Overlapping compute with I/O
- Parallel layer downloads where possible
- SSD wear leveling awareness
- Pipeline parallelism for I/O operations

Author: Nexus Team
"""

import os
import asyncio
import threading
import queue
from typing import Dict, Optional, Any, List, Callable, Coroutine, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import time
import logging
from collections import deque
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IOPriority(Enum):
    """I/O operation priority levels."""
    CRITICAL = 0  # Immediate execution required
    HIGH = 1      # Should be loaded soon
    NORMAL = 2    # Standard priority
    LOW = 3       # Can be deferred
    PREPREFETCH = 4  # Speculative pre-fetch


@dataclass
class IORequest:
    """Represents an I/O operation request."""
    layer_id: str
    model_id: str
    layer_index: int
    priority: IOPriority = IOPriority.NORMAL
    callback: Optional[Callable[[nn.Module], None]] = None
    timestamp: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class IOStats:
    """I/O performance statistics."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    avg_latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    queue_depth: int = 0
    concurrent_ops: int = 0
    cache_hits: int = 0
    prefetch_hits: int = 0
    
    def record_completion(self, latency_ms: float, bytes_transferred: int):
        """Record successful I/O completion."""
        self.completed_requests += 1
        # Update running average
        self.avg_latency_ms = (
            (self.avg_latency_ms * (self.completed_requests - 1) + latency_ms)
            / self.completed_requests
        )
    
    def record_failure(self):
        """Record I/O failure."""
        self.failed_requests += 1
    
    def record_prefetch_hit(self):
        """Record pre-fetched layer hit."""
        self.prefetch_hits += 1


class AsyncLayerPrefetcher:
    """
    Asynchronous layer pre-fetcher with intelligent scheduling.
    
    Features:
    - Priority-based request queue
    - Lookahead pre-fetching
    - Parallel downloads
    - Compute-I/O overlap
    """

    def __init__(
        self,
        layer_cache: Any,  # LayerCache instance
        max_concurrent_downloads: int = 4,
        prefetch_lookahead: int = 2,
        enable_parallel_downloads: bool = True,
        io_thread_count: int = 4,
    ):
        """
        Initialize the async prefetcher.

        Args:
            layer_cache: LayerCache instance for caching
            max_concurrent_downloads: Maximum parallel downloads
            prefetch_lookahead: Number of future layers to prefetch
            enable_parallel_downloads: Whether to download in parallel
            io_thread_count: Number of I/O threads
        """
        self.layer_cache = layer_cache
        self.max_concurrent_downloads = max_concurrent_downloads
        self.prefetch_lookahead = prefetch_lookahead
        self.enable_parallel_downloads = enable_parallel_downloads
        
        # Request queue with priority
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._in_progress: Dict[str, Future] = {}
        self._prefetched: Set[str] = set()
        self._lock = threading.RLock()
        
        # Thread pool for I/O operations
        self._executor = ThreadPoolExecutor(max_workers=io_thread_count)
        self._shutdown = False
        
        # Statistics
        self._stats = IOStats()
        
        # Start worker threads
        self._workers: List[threading.Thread] = []
        for i in range(max_concurrent_downloads):
            worker = threading.Thread(target=self._io_worker, daemon=True, name=f"IO-Worker-{i}")
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"AsyncLayerPrefetcher initialized with {max_concurrent_downloads} workers")

    def _io_worker(self):
        """Background I/O worker thread."""
        while not self._shutdown:
            try:
                # Get request from queue (priority-based)
                priority, request = self._request_queue.get(timeout=1.0)
                
                if self._shutdown:
                    break
                
                # Process the request
                self._process_request(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"I/O worker error: {e}")

    def _process_request(self, request: IORequest):
        """Process a single I/O request."""
        layer_id = request.layer_id
        
        with self._lock:
            if layer_id in self._in_progress:
                # Already being processed
                return
            
            self._in_progress[layer_id] = None  # Will be replaced with Future
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_layer = self.layer_cache.get_layer(
                request.model_id,
                request.layer_index,
                device='cpu'
            )
            
            if cached_layer is not None:
                # Cache hit
                self._stats.cache_hits += 1
                if request.callback:
                    request.callback(cached_layer)
                return
            
            # Submit download task to thread pool
            future = self._executor.submit(
                self._download_and_cache,
                request
            )
            
            with self._lock:
                self._in_progress[layer_id] = future
            
            # Wait for completion
            layer = future.result(timeout=request.timeout)
            
            if layer and request.callback:
                request.callback(layer)
            
            # Record statistics
            latency_ms = (time.time() - start_time) * 1000
            self._stats.record_completion(latency_ms, 0)
            
        except Exception as e:
            logger.error(f"Failed to process request for {layer_id}: {e}")
            self._stats.record_failure()
            
            # Retry if applicable
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                request.priority = IOPriority(request.priority.value - 1)  # Increase priority
                self.submit_request(request)
        
        finally:
            with self._lock:
                self._in_progress.pop(layer_id, None)

    def _download_and_cache(self, request: IORequest) -> Optional[nn.Module]:
        """Download and cache a layer."""
        # This would integrate with the actual download logic
        # For now, return None to indicate cache miss
        logger.debug(f"Downloading layer {request.layer_id}")
        return None

    def submit_request(self, request: IORequest) -> Future:
        """
        Submit an I/O request.

        Args:
            request: The I/O request

        Returns:
            Future representing the pending operation
        """
        if self._shutdown:
            raise RuntimeError("Prefetcher is shut down")
        
        self._stats.total_requests += 1
        
        # Priority queue uses tuple (priority, timestamp, request)
        # Lower priority value = higher priority
        self._request_queue.put((request.priority.value, request))
        
        # Create a future for tracking
        future = Future()
        
        return future

    def prefetch_layer(
        self,
        model_id: str,
        layer_index: int,
        priority: IOPriority = IOPriority.PREPREFETCH
    ):
        """
        Pre-fetch a layer for future use.

        Args:
            model_id: Model identifier
            layer_index: Layer index to prefetch
            priority: Priority of the prefetch
        """
        layer_id = f"{model_id}_layer_{layer_index}"
        
        # Skip if already in progress or prefetched
        with self._lock:
            if layer_id in self._in_progress or layer_id in self._prefetched:
                return
            self._prefetched.add(layer_id)
        
        request = IORequest(
            layer_id=layer_id,
            model_id=model_id,
            layer_index=layer_index,
            priority=priority,
        )
        
        self.submit_request(request)

    def prefetch_layers_ahead(
        self,
        model_id: str,
        current_layer: int,
        total_layers: int,
        lookahead: Optional[int] = None
    ):
        """
        Pre-fetch layers ahead of current execution.

        Args:
            model_id: Model identifier
            current_layer: Current layer index
            total_layers: Total number of layers
            lookahead: Number of layers to prefetch (default: self.prefetch_lookahead)
        """
        if lookahead is None:
            lookahead = self.prefetch_lookahead
        
        for i in range(1, lookahead + 1):
            next_layer = current_layer + i
            if next_layer < total_layers:
                self.prefetch_layer(
                    model_id,
                    next_layer,
                    priority=IOPriority.PREPREFETCH
                )

    def get_layer_async(
        self,
        model_id: str,
        layer_index: int,
        callback: Optional[Callable[[nn.Module], None]] = None,
        priority: IOPriority = IOPriority.NORMAL
    ) -> Future:
        """
        Asynchronously get a layer.

        Args:
            model_id: Model identifier
            layer_index: Layer index
            callback: Optional callback when layer is loaded
            priority: Request priority

        Returns:
            Future representing the pending layer load
        """
        layer_id = f"{model_id}_layer_{layer_index}"
        
        # Check if already prefetched
        with self._lock:
            if layer_id in self._prefetched:
                self._stats.record_prefetch_hit()
        
        request = IORequest(
            layer_id=layer_id,
            model_id=model_id,
            layer_index=layer_index,
            priority=priority,
            callback=callback,
        )
        
        return self.submit_request(request)

    def wait_for_layer(self, layer_id: str, timeout: Optional[float] = None) -> Optional[nn.Module]:
        """Wait for a specific layer to be loaded."""
        with self._lock:
            if layer_id in self._in_progress:
                future = self._in_progress[layer_id]
                if future is not None:
                    try:
                        return future.result(timeout=timeout)
                    except Exception as e:
                        logger.error(f"Error waiting for layer {layer_id}: {e}")
        return None

    def shutdown(self):
        """Shutdown the prefetcher and cleanup resources."""
        self._shutdown = True
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("AsyncLayerPrefetcher shut down")

    def get_stats(self) -> Dict[str, Any]:
        """Get I/O statistics."""
        return {
            'total_requests': self._stats.total_requests,
            'completed_requests': self._stats.completed_requests,
            'failed_requests': self._stats.failed_requests,
            'avg_latency_ms': self._stats.avg_latency_ms,
            'prefetch_hits': self._stats.prefetch_hits,
            'queue_size': self._request_queue.qsize(),
            'in_progress': len(self._in_progress),
        }


class ComputeIOOverlap:
    """
    Manages overlapping computation with I/O operations.
    
    This enables pipeline parallelism where:
    - GPU computes on current layer
    - CPU/Disk I/O loads next layer(s)
    """

    def __init__(
        self,
        prefetcher: AsyncLayerPrefetcher,
        pipeline_depth: int = 2,
    ):
        """
        Initialize compute-I/O overlap manager.

        Args:
            prefetcher: AsyncLayerPrefetcher instance
            pipeline_depth: Number of layers to pipeline
        """
        self.prefetcher = prefetcher
        self.pipeline_depth = pipeline_depth
        
        # Pipeline state
        self._current_layer: int = 0
        self._ready_layers: Dict[int, nn.Module] = {}
        self._pending_layers: Dict[int, Future] = {}
        self._lock = threading.RLock()
        
    def start_pipeline(self, model_id: str, start_layer: int = 0):
        """Initialize the pipeline for a model."""
        with self._lock:
            self._current_layer = start_layer
            self._ready_layers.clear()
            self._pending_layers.clear()
        
        # Pre-fetch initial layers
        self._prefetch_pipeline(model_id, start_layer)

    def _prefetch_pipeline(self, model_id: str, current_layer: int):
        """Pre-fetch layers for the pipeline."""
        for i in range(self.pipeline_depth):
            layer_idx = current_layer + i
            with self._lock:
                if layer_idx not in self._ready_layers and layer_idx not in self._pending_layers:
                    future = self.prefetcher.get_layer_async(
                        model_id,
                        layer_idx,
                        priority=IOPriority.HIGH if i == 0 else IOPriority.NORMAL
                    )
                    self._pending_layers[layer_idx] = future

    def get_next_layer(self, model_id: str, timeout: Optional[float] = None) -> Optional[nn.Module]:
        """
        Get the next layer for computation, blocking until ready.

        Args:
            model_id: Model identifier
            timeout: Maximum wait time

        Returns:
            The layer module, or None if timeout
        """
        with self._lock:
            layer_idx = self._current_layer
            
            # Check if already ready
            if layer_idx in self._ready_layers:
                layer = self._ready_layers.pop(layer_idx)
                self._current_layer += 1
                
                # Trigger prefetch for next layers
                self._prefetch_pipeline(model_id, self._current_layer)
                return layer
            
            # Wait for pending layer
            if layer_idx in self._pending_layers:
                future = self._pending_layers.pop(layer_idx)
                self._current_layer += 1
        
        # Wait outside the lock
        try:
            layer = future.result(timeout=timeout)
            
            # Trigger prefetch
            self._prefetch_pipeline(model_id, self._current_layer)
            
            return layer
        except Exception as e:
            logger.error(f"Failed to get layer {layer_idx}: {e}")
            return None

    def submit_compute(self, layer: nn.Module, inputs: torch.Tensor) -> Future:
        """
        Submit computation task to be overlapped with I/O.

        Args:
            layer: Layer to compute on
            inputs: Input tensor

        Returns:
            Future for the computation result
        """
        return self.prefetcher._executor.submit(layer, inputs)


class SSDWearLeveling:
    """
    SSD wear leveling awareness for cache management.
    
    Distributes write operations across the cache to minimize
    write amplification and extend SSD lifespan.
    """

    def __init__(
        self,
        cache_dir: Path,
        num_zones: int = 4,
        max_writes_per_zone: int = 1000,
    ):
        """
        Initialize SSD wear leveling manager.

        Args:
            cache_dir: Base cache directory
            num_zones: Number of write zones
            max_writes_per_zone: Maximum writes before rotating
        """
        self.cache_dir = Path(cache_dir)
        self.num_zones = num_zones
        self.max_writes_per_zone = max_writes_per_zone
        
        # Zone management
        self._zone_counters = [0] * num_zones
        self._current_zone = 0
        self._write_history: deque = deque(maxlen=1000)
        
        # Create zone directories
        self._zones = [self.cache_dir / f"zone_{i}" for i in range(num_zones)]
        for zone in self._zones:
            zone.mkdir(parents=True, exist_ok=True)

    def get_write_zone(self) -> Path:
        """Get the next write zone for balanced wear."""
        # Find zone with minimum writes
        min_writes = min(self._zone_counters)
        zone_idx = self._zone_counters.index(min_writes)
        
        self._zone_counters[zone_idx] += 1
        self._write_history.append((time.time(), zone_idx))
        
        return self._zones[zone_idx]

    def get_zone_for_layer(self, layer_id: str) -> Path:
        """Get zone for a specific layer (consistent hashing)."""
        # Use hash to consistently map layer to zone
        zone_idx = hash(layer_id) % self.num_zones
        return self._zones[zone_idx]

    def record_read(self, layer_id: str):
        """Record a read operation for statistics."""
        # Track read patterns for optimization
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get wear leveling statistics."""
        return {
            'zone_write_counts': self._zone_counters.copy(),
            'total_writes': sum(self._zone_counters),
            'write_balance': max(self._zone_counters) - min(self._zone_counters) if self._zone_counters else 0,
        }


class ParallelDownloader:
    """
    Handles parallel layer downloads with connection pooling.
    """

    def __init__(
        self,
        max_connections: int = 8,
        connection_timeout: float = 30.0,
        chunk_size: int = 8192,
    ):
        """
        Initialize parallel downloader.

        Args:
            max_connections: Maximum concurrent HTTP connections
            connection_timeout: Connection timeout in seconds
            chunk_size: Download chunk size
        """
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.chunk_size = chunk_size
        
        # Connection pool (would integrate with aiohttp/httpx)
        self._semaphore = asyncio.Semaphore(max_connections)
        self._download_stats = {
            'total_downloads': 0,
            'active_downloads': 0,
            'failed_downloads': 0,
            'bytes_downloaded': 0,
        }

    async def download_layer_async(
        self,
        url: str,
        output_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download a layer asynchronously.

        Args:
            url: Download URL
            output_path: Output file path
            progress_callback: Optional progress callback(current, total)

        Returns:
            True if successful
        """
        async with self._semaphore:
            self._download_stats['active_downloads'] += 1
            
            try:
                # This is a placeholder - would use aiohttp in production
                # async with aiohttp.ClientSession() as session:
                #     async with session.get(url) as response:
                #         ... download logic
                
                # Simulate download
                await asyncio.sleep(0.1)
                
                self._download_stats['total_downloads'] += 1
                return True
                
            except Exception as e:
                logger.error(f"Download failed: {e}")
                self._download_stats['failed_downloads'] += 1
                return False
            finally:
                self._download_stats['active_downloads'] -= 1

    def download_layers_parallel(
        self,
        urls: List[Tuple[str, Path]],
        max_concurrent: Optional[int] = None
    ) -> List[bool]:
        """
        Download multiple layers in parallel.

        Args:
            urls: List of (url, output_path) tuples
            max_concurrent: Maximum concurrent downloads

        Returns:
            List of success booleans
        """
        if max_concurrent:
            old_max = self.max_connections
            self._semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_all():
            tasks = [
                self.download_layer_async(url, path)
                for url, path in urls
            ]
            return await asyncio.gather(*tasks)
        
        try:
            results = asyncio.run(download_all())
        finally:
            if max_concurrent:
                self.max_connections = old_max
                self._semaphore = asyncio.Semaphore(old_max)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        return self._download_stats.copy()


class IOOptimizer:
    """
    Main I/O optimizer integrating all optimization strategies.
    
    This is the main interface for I/O optimization in Nexus SLI.
    """

    def __init__(
        self,
        layer_cache: Any,
        enable_prefetch: bool = True,
        enable_parallel_download: bool = True,
        enable_wear_leveling: bool = True,
        max_concurrent_downloads: int = 4,
        prefetch_lookahead: int = 2,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the I/O optimizer.

        Args:
            layer_cache: LayerCache instance
            enable_prefetch: Enable async prefetching
            enable_parallel_download: Enable parallel downloads
            enable_wear_leveling: Enable SSD wear leveling
            max_concurrent_downloads: Max parallel downloads
            prefetch_lookahead: Number of layers to prefetch ahead
            cache_dir: Cache directory for wear leveling
        """
        self.layer_cache = layer_cache
        
        # Initialize components
        self.prefetcher = None
        self.compute_overlap = None
        self.wear_leveling = None
        self.downloader = None
        
        if enable_prefetch:
            self.prefetcher = AsyncLayerPrefetcher(
                layer_cache=layer_cache,
                max_concurrent_downloads=max_concurrent_downloads,
                prefetch_lookahead=prefetch_lookahead,
            )
            
            self.compute_overlap = ComputeIOOverlap(
                prefetcher=self.prefetcher,
                pipeline_depth=prefetch_lookahead,
            )
        
        if enable_wear_leveling and cache_dir:
            self.wear_leveling = SSDWearLeveling(cache_dir=cache_dir)
        
        if enable_parallel_download:
            self.downloader = ParallelDownloader(max_connections=max_concurrent_downloads)
        
        self._enabled = True

    def enable(self):
        """Enable I/O optimizations."""
        self._enabled = True

    def disable(self):
        """Disable I/O optimizations."""
        self._enabled = False

    def prefetch_layers(
        self,
        model_id: str,
        layer_indices: List[int],
        priority: IOPriority = IOPriority.NORMAL
    ):
        """Pre-fetch multiple layers."""
        if not self._enabled or not self.prefetcher:
            return
        
        for idx in layer_indices:
            self.prefetcher.prefetch_layer(model_id, idx, priority)

    def get_layer_with_prefetch(
        self,
        model_id: str,
        layer_index: int,
        total_layers: int,
        device: str = 'cpu'
    ) -> Optional[nn.Module]:
        """
        Get a layer with automatic prefetching of next layers.

        Args:
            model_id: Model identifier
            layer_index: Current layer index
            total_layers: Total number of layers in model
            device: Target device

        Returns:
            The layer module
        """
        # First check cache directly
        layer = self.layer_cache.get_layer(model_id, layer_index, device=device)
        
        if layer is not None:
            # Cache hit - prefetch next layers
            if self._enabled and self.prefetcher:
                self.prefetcher.prefetch_layers_ahead(
                    model_id,
                    layer_index,
                    total_layers
                )
            return layer
        
        # Cache miss - synchronous load (would be async in production)
        # For now, return None to indicate not implemented
        return None

    def start_compute_pipeline(self, model_id: str, start_layer: int = 0):
        """Start compute-I/O overlapping pipeline."""
        if self._enabled and self.compute_overlap:
            self.compute_overlap.start_pipeline(model_id, start_layer)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive I/O statistics."""
        stats = {
            'enabled': self._enabled,
        }
        
        if self.prefetcher:
            stats['prefetcher'] = self.prefetcher.get_stats()
        
        if self.wear_leveling:
            stats['wear_leveling'] = self.wear_leveling.get_stats()
        
        if self.downloader:
            stats['downloader'] = self.downloader.get_stats()
        
        return stats

    def shutdown(self):
        """Shutdown the optimizer and cleanup resources."""
        if self.prefetcher:
            self.prefetcher.shutdown()
        
        logger.info("IOOptimizer shut down")


# Singleton instance
_io_optimizer: Optional[IOOptimizer] = None


def get_io_optimizer(
    layer_cache: Any = None,
    **kwargs
) -> IOOptimizer:
    """Get or create the global I/O optimizer instance."""
    global _io_optimizer
    
    if _io_optimizer is None:
        if layer_cache is None:
            raise ValueError("layer_cache required for first initialization")
        _io_optimizer = IOOptimizer(layer_cache=layer_cache, **kwargs)
    
    return _io_optimizer


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing IO Optimizer Module")
    print("=" * 60)
    
    # Mock layer cache for testing
    class MockLayerCache:
        def get_layer(self, model_id, layer_index, device='cpu'):
            return None
    
    # Create optimizer
    mock_cache = MockLayerCache()
    optimizer = IOOptimizer(
        layer_cache=mock_cache,
        enable_prefetch=True,
        enable_wear_leveling=True,
        cache_dir=Path("./test_cache"),
    )
    
    print("\n1. Testing prefetch:")
    optimizer.prefetch_layers("test_model", [0, 1, 2, 3])
    print("Prefetch submitted")
    
    print("\n2. I/O Stats:")
    stats = optimizer.get_stats()
    print(f"  Enabled: {stats['enabled']}")
    if 'prefetcher' in stats:
        print(f"  Total requests: {stats['prefetcher']['total_requests']}")
    
    print("\n3. Wear Leveling Stats:")
    if 'wear_leveling' in stats:
        wl_stats = stats['wear_leveling']
        print(f"  Zone writes: {wl_stats['zone_write_counts']}")
    
    # Cleanup
    optimizer.shutdown()
    
    print("\n" + "=" * 60)
