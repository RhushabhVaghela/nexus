"""
tests/unit/test_io_optimizer.py
Comprehensive tests for the I/O optimizer module.

Tests cover:
- AsyncLayerPrefetcher
- ComputeIOOverlap
- SSDWearLeveling
- ParallelDownloader
- Priority-based I/O
- Async operations
"""

import pytest
import time
import asyncio
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from concurrent.futures import Future

import torch
import torch.nn as nn

# Import the module under test
from src.nexus_final.sli.io_optimizer import (
    IOPriority,
    IORequest,
    IOStats,
    AsyncLayerPrefetcher,
    ComputeIOOverlap,
    SSDWearLeveling,
    ParallelDownloader,
    IOOptimizer,
    get_io_optimizer,
)


class TestIOPriority:
    """Test IOPriority enum."""

    def test_priority_values(self):
        """Test priority enum values are ordered correctly."""
        assert IOPriority.CRITICAL.value == 0
        assert IOPriority.HIGH.value == 1
        assert IOPriority.NORMAL.value == 2
        assert IOPriority.LOW.value == 3
        assert IOPriority.PREPREFETCH.value == 4

    def test_priority_ordering(self):
        """Test that lower values are higher priority."""
        assert IOPriority.CRITICAL.value < IOPriority.HIGH.value
        assert IOPriority.HIGH.value < IOPriority.NORMAL.value
        assert IOPriority.NORMAL.value < IOPriority.LOW.value
        assert IOPriority.LOW.value < IOPriority.PREPREFETCH.value


class TestIORequest:
    """Test IORequest dataclass."""

    def test_request_creation(self):
        """Test creating an IO request."""
        request = IORequest(
            layer_id="model_layer_0",
            model_id="test/model",
            layer_index=0,
            priority=IOPriority.HIGH,
            timeout=30.0,
            max_retries=5
        )

        assert request.layer_id == "model_layer_0"
        assert request.model_id == "test/model"
        assert request.layer_index == 0
        assert request.priority == IOPriority.HIGH
        assert request.timeout == 30.0
        assert request.max_retries == 5
        assert request.retry_count == 0
        assert request.timestamp > 0

    def test_request_defaults(self):
        """Test IO request with default values."""
        request = IORequest(
            layer_id="test",
            model_id="model",
            layer_index=0
        )

        assert request.priority == IOPriority.NORMAL
        assert request.callback is None
        assert request.timeout is None
        assert request.max_retries == 3
        assert request.retry_count == 0


class TestIOStats:
    """Test IOStats class."""

    def test_initial_stats(self):
        """Test initial stats are zero."""
        stats = IOStats()

        assert stats.total_requests == 0
        assert stats.completed_requests == 0
        assert stats.failed_requests == 0
        assert stats.cancelled_requests == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.throughput_mbps == 0.0
        assert stats.queue_depth == 0
        assert stats.concurrent_ops == 0
        assert stats.cache_hits == 0
        assert stats.prefetch_hits == 0

    def test_record_completion(self):
        """Test recording completion updates averages."""
        stats = IOStats()

        stats.record_completion(100.0, 1024)
        assert stats.completed_requests == 1
        assert stats.avg_latency_ms == 100.0

        stats.record_completion(200.0, 2048)
        assert stats.completed_requests == 2
        assert stats.avg_latency_ms == 150.0  # Running average

    def test_record_failure(self):
        """Test recording failures."""
        stats = IOStats()

        stats.record_failure()
        stats.record_failure()

        assert stats.failed_requests == 2

    def test_record_prefetch_hit(self):
        """Test recording prefetch hits."""
        stats = IOStats()

        stats.record_prefetch_hit()
        stats.record_prefetch_hit()

        assert stats.prefetch_hits == 2


class TestAsyncLayerPrefetcher:
    """Test AsyncLayerPrefetcher class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    @pytest.fixture
    def prefetcher(self, mock_cache):
        """Create a prefetcher with mock cache."""
        return AsyncLayerPrefetcher(
            layer_cache=mock_cache,
            max_concurrent_downloads=2,
            prefetch_lookahead=2,
            enable_parallel_downloads=True,
            io_thread_count=2
        )

    def test_initialization(self, mock_cache, prefetcher):
        """Test prefetcher initialization."""
        assert prefetcher.layer_cache == mock_cache
        assert prefetcher.max_concurrent_downloads == 2
        assert prefetcher.prefetch_lookahead == 2
        assert prefetcher.enable_parallel_downloads is True
        assert prefetcher._shutdown is False

    def test_initialization_defaults(self, mock_cache):
        """Test prefetcher with default parameters."""
        prefetcher = AsyncLayerPrefetcher(layer_cache=mock_cache)

        assert prefetcher.max_concurrent_downloads == 4
        assert prefetcher.prefetch_lookahead == 2
        assert prefetcher.enable_parallel_downloads is True

    def test_submit_request(self, prefetcher):
        """Test submitting an I/O request."""
        request = IORequest(
            layer_id="test_layer",
            model_id="test/model",
            layer_index=0,
            priority=IOPriority.HIGH
        )

        future = prefetcher.submit_request(request)

        assert isinstance(future, Future)
        assert prefetcher._stats.total_requests == 1

    def test_submit_request_shutdown(self, prefetcher):
        """Test submitting request when shut down raises error."""
        prefetcher._shutdown = True

        request = IORequest(
            layer_id="test_layer",
            model_id="test/model",
            layer_index=0
        )

        with pytest.raises(RuntimeError) as exc_info:
            prefetcher.submit_request(request)

        assert "shut down" in str(exc_info.value)

    def test_prefetch_layer(self, prefetcher):
        """Test prefetching a layer."""
        prefetcher.prefetch_layer("test/model", 5)

        # Should add to prefetched set
        assert "test_model_layer_5" in prefetcher._prefetched
        assert prefetcher._stats.total_requests == 1

    def test_prefetch_layer_duplicate(self, prefetcher):
        """Test prefetching same layer twice is ignored."""
        prefetcher.prefetch_layer("test/model", 5)
        initial_requests = prefetcher._stats.total_requests

        prefetcher.prefetch_layer("test/model", 5)

        assert prefetcher._stats.total_requests == initial_requests

    def test_prefetch_layers_ahead(self, prefetcher):
        """Test prefetching layers ahead."""
        prefetcher.prefetch_layers_ahead("test/model", 0, 10, lookahead=3)

        # Should have prefetched layers 1, 2, 3
        assert "test_model_layer_1" in prefetcher._prefetched
        assert "test_model_layer_2" in prefetcher._prefetched
        assert "test_model_layer_3" in prefetcher._prefetched
        assert "test_model_layer_4" not in prefetcher._prefetched

    def test_prefetch_layers_ahead_default_lookahead(self, prefetcher):
        """Test prefetching with default lookahead."""
        prefetcher.prefetch_layers_ahead("test/model", 0, 10)

        # Should use prefetch_lookahead (2)
        assert "test_model_layer_1" in prefetcher._prefetched
        assert "test_model_layer_2" in prefetcher._prefetched

    def test_prefetch_layers_ahead_boundary(self, prefetcher):
        """Test prefetching respects total layers boundary."""
        prefetcher.prefetch_layers_ahead("test/model", 8, 10, lookahead=5)

        # Should only prefetch up to layer 9
        assert "test_model_layer_9" in prefetcher._prefetched
        assert "test_model_layer_10" not in prefetcher._prefetched

    def test_get_layer_async(self, prefetcher):
        """Test async layer retrieval."""
        future = prefetcher.get_layer_async(
            "test/model",
            layer_index=5,
            priority=IOPriority.HIGH
        )

        assert isinstance(future, Future)
        assert prefetcher._stats.total_requests >= 1

    def test_get_layer_async_records_prefetch_hit(self, prefetcher):
        """Test that retrieving a prefetched layer records hit."""
        # First prefetch
        prefetcher.prefetch_layer("test/model", 5)

        # Then request (should record prefetch hit)
        future = prefetcher.get_layer_async("test/model", layer_index=5)

        assert prefetcher._stats.prefetch_hits >= 1

    def test_wait_for_layer_not_in_progress(self, prefetcher):
        """Test waiting for layer not in progress returns None."""
        result = prefetcher.wait_for_layer("nonexistent_layer", timeout=0.1)
        assert result is None

    def test_shutdown(self, prefetcher):
        """Test shutdown sets flag and stops workers."""
        prefetcher.shutdown()

        assert prefetcher._shutdown is True

    def test_get_stats(self, prefetcher):
        """Test getting prefetcher statistics."""
        # Submit some requests
        for i in range(3):
            request = IORequest(
                layer_id=f"layer_{i}",
                model_id="test",
                layer_index=i
            )
            prefetcher.submit_request(request)

        stats = prefetcher.get_stats()

        assert 'total_requests' in stats
        assert 'completed_requests' in stats
        assert 'failed_requests' in stats
        assert 'avg_latency_ms' in stats
        assert 'prefetch_hits' in stats
        assert stats['total_requests'] >= 3


class TestComputeIOOverlap:
    """Test ComputeIOOverlap class."""

    @pytest.fixture
    def mock_prefetcher(self):
        """Create a mock prefetcher."""
        prefetcher = MagicMock()
        future = Future()
        future.set_result(nn.Linear(100, 100))
        prefetcher.get_layer_async.return_value = future
        return prefetcher

    @pytest.fixture
    def overlap(self, mock_prefetcher):
        """Create compute overlap manager."""
        return ComputeIOOverlap(
            prefetcher=mock_prefetcher,
            pipeline_depth=2
        )

    def test_initialization(self, mock_prefetcher, overlap):
        """Test initialization."""
        assert overlap.prefetcher == mock_prefetcher
        assert overlap.pipeline_depth == 2
        assert overlap._current_layer == 0

    def test_start_pipeline(self, mock_prefetcher, overlap):
        """Test starting pipeline."""
        overlap.start_pipeline("test/model", start_layer=5)

        assert overlap._current_layer == 5
        assert len(overlap._ready_layers) == 0

    def test_prefetch_pipeline(self, mock_prefetcher, overlap):
        """Test pipeline prefetching."""
        overlap.start_pipeline("test/model", start_layer=0)

        # Should have prefetched layers 0 and 1
        assert mock_prefetcher.get_layer_async.call_count >= 2

    def test_get_next_layer_from_ready(self, overlap):
        """Test getting layer from ready cache."""
        # Manually add layer to ready
        layer = nn.Linear(100, 100)
        overlap._ready_layers[0] = layer

        result = overlap.get_next_layer("test/model")

        assert result is layer
        assert overlap._current_layer == 1

    def test_get_next_layer_timeout(self, mock_prefetcher, overlap):
        """Test getting layer with timeout."""
        overlap.start_pipeline("test/model", start_layer=0)

        # Create a future that won't complete
        future = Future()
        mock_prefetcher.get_layer_async.return_value = future
        overlap._pending_layers[0] = future

        result = overlap.get_next_layer("test/model", timeout=0.01)

        assert result is None

    def test_submit_compute(self, mock_prefetcher, overlap):
        """Test submitting compute task."""
        layer = nn.Linear(100, 100)
        inputs = torch.randn(1, 100)

        future = overlap.submit_compute(layer, inputs)

        assert isinstance(future, Future)


class TestSSDWearLeveling:
    """Test SSDWearLeveling class."""

    @pytest.fixture
    def wear_leveler(self, tmp_path):
        """Create a wear leveler."""
        return SSDWearLeveling(
            cache_dir=tmp_path,
            num_zones=4,
            max_writes_per_zone=1000
        )

    def test_initialization(self, tmp_path, wear_leveler):
        """Test initialization creates zones."""
        assert wear_leveler.cache_dir == Path(tmp_path)
        assert wear_leveler.num_zones == 4
        assert wear_leveler.max_writes_per_zone == 1000
        assert len(wear_leveler._zones) == 4

        # All zones should exist
        for zone in wear_leveler._zones:
            assert zone.exists()

    def test_initialization_creates_zones(self, tmp_path):
        """Test that zones are created on initialization."""
        wear_leveler = SSDWearLeveling(cache_dir=tmp_path)

        for i in range(4):
            zone_path = tmp_path / f"zone_{i}"
            assert zone_path.exists()

    def test_get_write_zone(self, wear_leveler):
        """Test getting write zone balances writes."""
        zone1 = wear_leveler.get_write_zone()
        zone2 = wear_leveler.get_write_zone()

        assert isinstance(zone1, Path)
        assert isinstance(zone2, Path)

        # Should have incremented counters
        assert wear_leveler._zone_counters[0] >= 1

    def test_get_write_zone_balanced(self, wear_leveler):
        """Test that writes are distributed across zones."""
        # Write many times
        for _ in range(20):
            wear_leveler.get_write_zone()

        # Check that writes are balanced
        max_writes = max(wear_leveler._zone_counters)
        min_writes = min(wear_leveler._zone_counters)

        # Difference should be minimal (at most 1)
        assert max_writes - min_writes <= 1

    def test_get_zone_for_layer_consistency(self, wear_leveler):
        """Test that same layer ID always maps to same zone."""
        zone1 = wear_leveler.get_zone_for_layer("layer_123")
        zone2 = wear_leveler.get_zone_for_layer("layer_123")

        assert zone1 == zone2

    def test_get_zone_for_layer_distribution(self, wear_leveler):
        """Test that different layers are distributed."""
        zones = set()
        for i in range(100):
            zone = wear_leveler.get_zone_for_layer(f"layer_{i}")
            zones.add(zone)

        # Should use multiple zones
        assert len(zones) > 1

    def test_record_read(self, wear_leveler):
        """Test recording a read operation."""
        # Should not raise
        wear_leveler.record_read("layer_123")

    def test_get_stats(self, wear_leveler):
        """Test getting wear leveling statistics."""
        # Add some writes
        for _ in range(10):
            wear_leveler.get_write_zone()

        stats = wear_leveler.get_stats()

        assert 'zone_write_counts' in stats
        assert 'total_writes' in stats
        assert 'write_balance' in stats
        assert stats['total_writes'] == 10
        assert len(stats['zone_write_counts']) == 4


class TestParallelDownloader:
    """Test ParallelDownloader class."""

    @pytest.fixture
    def downloader(self):
        """Create a parallel downloader."""
        return ParallelDownloader(
            max_connections=4,
            connection_timeout=30.0,
            chunk_size=8192
        )

    def test_initialization(self, downloader):
        """Test initialization."""
        assert downloader.max_connections == 4
        assert downloader.connection_timeout == 30.0
        assert downloader.chunk_size == 8192

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        downloader = ParallelDownloader()

        assert downloader.max_connections == 8
        assert downloader.connection_timeout == 30.0
        assert downloader.chunk_size == 8192

    def test_get_stats_initial(self, downloader):
        """Test initial stats."""
        stats = downloader.get_stats()

        assert stats['total_downloads'] == 0
        assert stats['active_downloads'] == 0
        assert stats['failed_downloads'] == 0
        assert stats['bytes_downloaded'] == 0

    @pytest.mark.asyncio
    async def test_download_layer_async(self, downloader, tmp_path):
        """Test async layer download."""
        output_path = tmp_path / "test_layer.bin"

        result = await downloader.download_layer_async(
            url="http://example.com/layer.bin",
            output_path=output_path
        )

        # Result is boolean - in real implementation would download
        assert isinstance(result, bool)

    def test_download_layers_parallel(self, downloader, tmp_path):
        """Test downloading multiple layers in parallel."""
        urls = [
            ("http://example.com/layer1.bin", tmp_path / "layer1.bin"),
            ("http://example.com/layer2.bin", tmp_path / "layer2.bin"),
        ]

        results = downloader.download_layers_parallel(urls)

        assert len(results) == 2
        assert all(isinstance(r, bool) for r in results)

    def test_download_layers_parallel_custom_concurrent(self, downloader, tmp_path):
        """Test parallel download with custom concurrency."""
        urls = [
            ("http://example.com/layer1.bin", tmp_path / "layer1.bin"),
        ]

        results = downloader.download_layers_parallel(urls, max_concurrent=2)

        assert len(results) == 1


class TestIOOptimizer:
    """Test IOOptimizer class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    @pytest.fixture
    def optimizer(self, mock_cache, tmp_path):
        """Create an I/O optimizer."""
        return IOOptimizer(
            layer_cache=mock_cache,
            enable_prefetch=True,
            enable_parallel_download=True,
            enable_wear_leveling=True,
            max_concurrent_downloads=4,
            prefetch_lookahead=2,
            cache_dir=tmp_path
        )

    def test_initialization_all_enabled(self, mock_cache, tmp_path, optimizer):
        """Test initialization with all features enabled."""
        assert optimizer.layer_cache == mock_cache
        assert optimizer.prefetcher is not None
        assert optimizer.compute_overlap is not None
        assert optimizer.wear_leveling is not None
        assert optimizer.downloader is not None
        assert optimizer._enabled is True

    def test_initialization_prefetch_disabled(self, mock_cache, tmp_path):
        """Test initialization with prefetch disabled."""
        optimizer = IOOptimizer(
            layer_cache=mock_cache,
            enable_prefetch=False,
            cache_dir=tmp_path
        )

        assert optimizer.prefetcher is None
        assert optimizer.compute_overlap is None

    def test_initialization_wear_leveling_disabled(self, mock_cache):
        """Test initialization with wear leveling disabled."""
        optimizer = IOOptimizer(
            layer_cache=mock_cache,
            enable_wear_leveling=False
        )

        assert optimizer.wear_leveling is None

    def test_enable_disable(self, mock_cache, tmp_path, optimizer):
        """Test enable and disable methods."""
        optimizer.disable()
        assert optimizer._enabled is False

        optimizer.enable()
        assert optimizer._enabled is True

    def test_prefetch_layers(self, mock_cache, tmp_path, optimizer):
        """Test prefetching layers."""
        optimizer.prefetch_layers("test/model", [0, 1, 2, 3])

        # Should have submitted requests
        assert optimizer.prefetcher._stats.total_requests >= 4

    def test_prefetch_layers_disabled(self, mock_cache, tmp_path, optimizer):
        """Test prefetching when disabled does nothing."""
        optimizer.disable()
        initial_requests = optimizer.prefetcher._stats.total_requests

        optimizer.prefetch_layers("test/model", [0, 1, 2])

        assert optimizer.prefetcher._stats.total_requests == initial_requests

    def test_get_layer_with_prefetch_cache_hit(self, mock_cache, optimizer):
        """Test getting layer with cache hit."""
        layer = nn.Linear(100, 100)
        mock_cache.get_layer.return_value = layer

        result = optimizer.get_layer_with_prefetch(
            "test/model",
            layer_index=5,
            total_layers=10
        )

        assert result is layer

    def test_get_layer_with_prefetch_cache_miss(self, mock_cache, optimizer):
        """Test getting layer with cache miss."""
        mock_cache.get_layer.return_value = None

        result = optimizer.get_layer_with_prefetch(
            "test/model",
            layer_index=5,
            total_layers=10
        )

        assert result is None

    def test_start_compute_pipeline(self, mock_cache, tmp_path, optimizer):
        """Test starting compute pipeline."""
        optimizer.start_compute_pipeline("test/model", start_layer=5)

        assert optimizer.compute_overlap._current_layer == 5

    def test_start_compute_pipeline_disabled(self, mock_cache, tmp_path, optimizer):
        """Test starting compute pipeline when disabled."""
        optimizer.disable()

        # Should not raise
        optimizer.start_compute_pipeline("test/model", start_layer=5)

    def test_get_stats(self, mock_cache, tmp_path, optimizer):
        """Test getting optimizer statistics."""
        stats = optimizer.get_stats()

        assert 'enabled' in stats
        assert stats['enabled'] is True
        assert 'prefetcher' in stats
        assert 'wear_leveling' in stats
        assert 'downloader' in stats

    def test_shutdown(self, mock_cache, tmp_path, optimizer):
        """Test shutdown."""
        optimizer.shutdown()

        assert optimizer.prefetcher._shutdown is True


class TestGetIOOptimizer:
    """Test get_io_optimizer function."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    def test_get_io_optimizer_creates_instance(self, mock_cache):
        """Test getting optimizer creates instance."""
        # Reset global instance
        import src.nexus_final.sli.io_optimizer as io_module
        io_module._io_optimizer = None

        optimizer = get_io_optimizer(layer_cache=mock_cache)

        assert isinstance(optimizer, IOOptimizer)

    def test_get_io_optimizer_returns_existing(self, mock_cache):
        """Test getting optimizer returns existing instance."""
        import src.nexus_final.sli.io_optimizer as io_module
        io_module._io_optimizer = None

        optimizer1 = get_io_optimizer(layer_cache=mock_cache)
        optimizer2 = get_io_optimizer()

        assert optimizer1 is optimizer2

    def test_get_io_optimizer_requires_cache_first_time(self):
        """Test first call requires layer_cache."""
        import src.nexus_final.sli.io_optimizer as io_module
        io_module._io_optimizer = None

        with pytest.raises(ValueError) as exc_info:
            get_io_optimizer()

        assert "layer_cache required" in str(exc_info.value)


class TestIOOptimizerEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    @pytest.fixture
    def mock_prefetcher(self):
        """Create a mock prefetcher."""
        prefetcher = MagicMock()
        future = Future()
        future.set_result(nn.Linear(100, 100))
        prefetcher.get_layer_async.return_value = future
        return prefetcher

    def test_io_request_negative_timeout(self):
        """Test IO request with negative timeout."""
        request = IORequest(
            layer_id="test",
            model_id="model",
            layer_index=0,
            timeout=-1.0
        )

        assert request.timeout == -1.0

    def test_io_request_zero_retries(self):
        """Test IO request with zero retries."""
        request = IORequest(
            layer_id="test",
            model_id="model",
            layer_index=0,
            max_retries=0
        )

        assert request.max_retries == 0

    def test_wear_leveling_single_zone(self, tmp_path):
        """Test wear leveling with single zone."""
        wear_leveler = SSDWearLeveling(
            cache_dir=tmp_path,
            num_zones=1
        )

        zone = wear_leveler.get_write_zone()
        assert zone == wear_leveler._zones[0]

    def test_compute_overlap_zero_pipeline_depth(self, mock_prefetcher):
        """Test compute overlap with zero pipeline depth."""
        overlap = ComputeIOOverlap(
            prefetcher=mock_prefetcher,
            pipeline_depth=0
        )

        assert overlap.pipeline_depth == 0

    def test_prefetcher_zero_workers(self, mock_cache):
        """Test prefetcher with zero workers."""
        prefetcher = AsyncLayerPrefetcher(
            layer_cache=mock_cache,
            max_concurrent_downloads=0
        )

        assert len(prefetcher._workers) == 0

    def test_parallel_downloader_zero_connections(self):
        """Test parallel downloader with zero connections."""
        downloader = ParallelDownloader(max_connections=0)

        assert downloader.max_connections == 0
