"""
Tests for Enhanced I/O Optimizer Module

Comprehensive test suite covering:
- Enhanced prefetch buffer
- Pattern recognition
- Parallel loading
- Lock-free queue
- Priority queue
"""

import pytest
import time
import threading
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexus.models.sli.io_optimizer import (
    IOOptimizer,
    EnhancedPrefetchBuffer,
    AccessPattern,
    LockFreeQueue,
    IOPriority,
    IORequest,
)


class TestAccessPattern:
    """Test cases for AccessPattern."""

    def test_initial_state(self):
        """Test initial pattern state."""
        pattern = AccessPattern()

        assert pattern.pattern_type == "sequential"
        assert pattern.stride == 1
        assert pattern.confidence == 0.0

    def test_sequential_detection(self):
        """Test sequential pattern detection."""
        pattern = AccessPattern()

        # Simulate sequential access
        for i in range(5):
            pattern.update(i)

        assert pattern.pattern_type == "sequential"
        assert pattern.confidence > 0

    def test_strided_detection(self):
        """Test strided pattern detection."""
        pattern = AccessPattern()

        # Simulate strided access (every 2nd layer)
        for i in range(0, 10, 2):
            pattern.update(i)

        assert pattern.pattern_type == "strided"
        assert pattern.stride == 2

    def test_predict_sequential(self):
        """Test prediction with sequential pattern."""
        pattern = AccessPattern()

        for i in range(5):
            pattern.update(i)

        predictions = pattern.predict_next(3)
        assert predictions == [5, 6, 7]

    def test_predict_strided(self):
        """Test prediction with strided pattern."""
        pattern = AccessPattern()

        for i in range(0, 6, 2):
            pattern.update(i)

        predictions = pattern.predict_next(3)
        assert predictions == [6, 8, 10]


class TestLockFreeQueue:
    """Test cases for LockFreeQueue."""

    def test_initialization(self):
        """Test queue initialization."""
        queue = LockFreeQueue(capacity=100)

        assert queue.capacity == 100
        assert queue.qsize() == 0
        assert queue.empty()

    def test_put_and_get(self):
        """Test put and get operations."""
        queue = LockFreeQueue(capacity=10)

        request = IORequest(layer_id="test", model_id="model", layer_index=0)

        success = queue.put(request)
        assert success
        assert queue.qsize() == 1

        retrieved = queue.get()
        assert retrieved == request
        assert queue.qsize() == 0

    def test_queue_full(self):
        """Test queue full behavior."""
        queue = LockFreeQueue(capacity=2)

        request1 = IORequest(layer_id="test1", model_id="model", layer_index=0)
        request2 = IORequest(layer_id="test2", model_id="model", layer_index=1)
        request3 = IORequest(layer_id="test3", model_id="model", layer_index=2)

        assert queue.put(request1)
        assert queue.put(request2)
        assert not queue.put(request3)  # Should fail - queue full

    def test_queue_empty_get(self):
        """Test get from empty queue."""
        queue = LockFreeQueue(capacity=10)

        result = queue.get()
        assert result is None


class TestEnhancedPrefetchBuffer:
    """Test cases for EnhancedPrefetchBuffer."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    @pytest.fixture
    def buffer(self, mock_cache):
        """Create enhanced prefetch buffer."""
        buf = EnhancedPrefetchBuffer(
            layer_cache=mock_cache,
            max_concurrent_downloads=4,
            prefetch_lookahead=5,
            enable_pattern_recognition=True,
            enable_priority_queue=True,
            io_thread_count=4,
        )
        yield buf
        buf.shutdown()

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.prefetch_lookahead == 5
        assert buffer.max_concurrent_downloads == 4
        assert buffer.enable_pattern_recognition
        assert buffer.enable_priority_queue

    def test_submit_request(self, buffer):
        """Test submitting a request."""
        request = IORequest(
            layer_id="test_layer",
            model_id="test_model",
            layer_index=0,
            priority=IOPriority.NORMAL,
        )

        future = buffer.submit_request(request)
        assert future is not None

    def test_record_access(self, buffer):
        """Test recording layer access."""
        buffer.record_access("model", 0)

        assert len(buffer._access_pattern.recent_indices) == 1
        assert buffer._layer_access_counts["model_layer_0"] == 1

    def test_prefetch_layers_parallel(self, buffer):
        """Test parallel prefetching."""
        futures = buffer.prefetch_layers_parallel(
            "model", [0, 1, 2, 3, 4], priority=IOPriority.NORMAL
        )

        assert len(futures) == 5

    def test_pattern_recognition(self, buffer):
        """Test pattern recognition."""
        # Build sequential pattern
        for i in range(5):
            buffer.record_access("model", i)

        assert buffer._access_pattern.pattern_type == "sequential"
        assert buffer._access_pattern.confidence > 0

    def test_get_stats(self, buffer):
        """Test getting statistics."""
        buffer.record_access("model", 0)

        stats = buffer.get_stats()
        assert "pattern_type" in stats
        assert "prefetch_buffer_size" in stats

    def test_shutdown(self, buffer):
        """Test buffer shutdown."""
        buffer.shutdown()
        assert buffer._shutdown


class TestIOOptimizerEnhanced:
    """Test cases for IOOptimizer with enhanced features."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock layer cache."""
        cache = MagicMock()
        cache.get_layer.return_value = None
        return cache

    @pytest.fixture
    def optimizer(self, mock_cache):
        """Create I/O optimizer with enhanced prefetch."""
        opt = IOOptimizer(
            layer_cache=mock_cache,
            enable_prefetch=True,
            use_enhanced_prefetch=True,
            prefetch_lookahead=5,
            max_concurrent_downloads=8,
            io_thread_count=8,
        )
        yield opt
        opt.shutdown()

    def test_enhanced_prefetch_enabled(self, optimizer):
        """Test that enhanced prefetch is enabled."""
        assert optimizer.enhanced_prefetcher is not None
        assert optimizer._use_enhanced

    def test_prefetch_layers_parallel(self, optimizer):
        """Test parallel prefetch method."""
        optimizer.prefetch_layers_parallel("model", [0, 1, 2, 3, 4])

        # Should not raise any errors

    def test_get_layer_with_prefetch(self, optimizer, mock_cache):
        """Test getting layer with prefetch."""
        # Setup cache to return a layer
        mock_layer = nn.Linear(256, 256)
        mock_cache.get_layer.return_value = mock_layer

        layer = optimizer.get_layer_with_prefetch("model", 0, 10)

        assert layer is not None

    def test_get_stats(self, optimizer):
        """Test getting optimizer stats."""
        stats = optimizer.get_stats()

        assert "enabled" in stats
        assert "use_enhanced_prefetch" in stats
        assert stats["use_enhanced_prefetch"] == True

    def test_start_compute_pipeline(self, optimizer):
        """Test starting compute pipeline."""
        optimizer.start_compute_pipeline("model", start_layer=0)

        # Should not raise any errors


class TestThreadSafety:
    """Test thread safety of enhanced components."""

    def test_concurrent_access_pattern_updates(self):
        """Test concurrent access pattern updates."""
        pattern = AccessPattern()

        errors = []

        def update_pattern(idx):
            try:
                for _ in range(10):
                    pattern.update(idx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_pattern, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_queue_operations(self):
        """Test concurrent queue operations."""
        queue = LockFreeQueue(capacity=1000)

        errors = []
        put_count = [0]
        get_count = [0]

        def producer():
            try:
                for i in range(100):
                    request = IORequest(
                        layer_id=f"layer_{i}", model_id="model", layer_index=i
                    )
                    if queue.put(request):
                        put_count[0] += 1
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(100):
                    request = queue.get()
                    if request:
                        get_count[0] += 1
            except Exception as e:
                errors.append(e)

        producers = [threading.Thread(target=producer) for _ in range(3)]
        consumers = [threading.Thread(target=consumer) for _ in range(3)]

        for t in producers + consumers:
            t.start()
        for t in producers + consumers:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
