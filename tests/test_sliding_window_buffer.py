"""
Tests for Sliding Window Buffer Module

Comprehensive test suite covering:
- Window initialization and management
- Layer loading and eviction
- Pattern recognition
- Memory optimization
- Thread safety
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.sli.sliding_window_buffer import (
    SlidingWindowBuffer,
    AdaptiveSlidingWindow,
    SlidingWindowConfig,
    WindowState,
    WindowEntry,
    create_sliding_window,
)


class TestSlidingWindowBuffer:
    """Test cases for SlidingWindowBuffer."""

    @pytest.fixture
    def mock_layer_loader(self):
        """Create a mock layer loader."""
        def loader(model_id: str, layer_index: int) -> nn.Module:
            return nn.Linear(1024, 1024)
        return loader

    @pytest.fixture
    def window(self, mock_layer_loader):
        """Create a sliding window buffer."""
        return SlidingWindowBuffer(
            window_size=5,
            config=SlidingWindowConfig(max_memory_gb=1.0),
            layer_loader=mock_layer_loader
        )

    def test_initialization(self, window):
        """Test buffer initialization."""
        assert window.window_size == 5
        assert len(window._window) == 0
        assert window._current_model_id is None

    def test_initialize_window(self, window):
        """Test window initialization."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        assert window._current_model_id == "test_model"
        assert window._current_layer_index == 0
        assert window._total_layers == 20
        # Should preload initial window
        assert len(window._window) <= 5

    def test_get_layer_from_window(self, window):
        """Test getting a layer from the window."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        layer = window.get_layer("test_model", 0)
        assert layer is not None
        assert isinstance(layer, nn.Module)

    def test_window_slide(self, window):
        """Test sliding the window forward."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        # Access first layer
        window.get_layer("test_model", 0)
        
        # Slide window
        window.slide_window()
        assert window._current_layer_index == 1

    def test_eviction(self, window):
        """Test layer eviction when window slides."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        # Access layers
        for i in range(5):
            window.get_layer("test_model", i)
        
        # Slide past overlap
        for _ in range(3):
            window.slide_window()
        
        # Check that oldest layers were evicted
        layers_in_window = window.get_window_layers()
        assert 0 not in layers_in_window  # Should be evicted

    def test_memory_tracking(self, window):
        """Test memory usage tracking."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        initial_memory = window._current_memory_bytes
        
        # Load a layer
        window.get_layer("test_model", 0)
        
        assert window._current_memory_bytes > initial_memory

    def test_get_stats(self, window):
        """Test statistics collection."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        window.get_layer("test_model", 0)
        
        stats = window.get_stats()
        assert 'window_size' in stats
        assert 'current_layers' in stats
        assert 'hit_ratio' in stats

    def test_clear_window(self, window):
        """Test clearing the window."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        window.get_layer("test_model", 0)
        
        window.clear_window()
        
        assert len(window._window) == 0
        assert window._current_memory_bytes == 0

    def test_is_layer_in_window(self, window):
        """Test checking if layer is in window."""
        window.initialize_window("test_model", start_layer=0, total_layers=20)
        
        assert window.is_layer_in_window("test_model", 0)
        assert not window.is_layer_in_window("other_model", 0)


class TestAdaptiveSlidingWindow:
    """Test cases for AdaptiveSlidingWindow."""

    @pytest.fixture
    def adaptive_window(self):
        """Create an adaptive sliding window."""
        def loader(model_id: str, layer_index: int) -> nn.Module:
            return nn.Linear(512, 512)
        
        return AdaptiveSlidingWindow(
            window_size=5,
            config=SlidingWindowConfig(),
            layer_loader=loader
        )

    def test_pattern_tracking(self, adaptive_window):
        """Test access pattern tracking."""
        adaptive_window.initialize_window("model", start_layer=0, total_layers=20)
        
        # Simulate sequential access
        for i in range(5):
            adaptive_window.get_layer("model", i)
        
        assert adaptive_window._access_pattern.pattern_type == "sequential"
        assert adaptive_window._access_pattern.confidence > 0

    def test_predict_next_layers(self, adaptive_window):
        """Test layer prediction."""
        adaptive_window.initialize_window("model", start_layer=0, total_layers=20)
        
        # Build sequential pattern
        for i in range(5):
            adaptive_window.get_layer("model", i)
        
        predictions = adaptive_window.predict_next(3)
        assert predictions == [5, 6, 7]

    def test_adapt_window_size(self, adaptive_window):
        """Test dynamic window size adaptation."""
        adaptive_window.initialize_window("model", start_layer=0, total_layers=20)
        
        initial_size = adaptive_window.window_size
        
        # Simulate many accesses with high hit rate
        for _ in range(100):
            adaptive_window.get_layer("model", 0)
        
        # Check if adaptation occurred
        stats = adaptive_window.get_stats()
        assert 'window_resizes' in stats


class TestWindowEntry:
    """Test cases for WindowEntry."""

    def test_entry_creation(self):
        """Test creating a window entry."""
        entry = WindowEntry(
            layer_id="test_layer",
            layer_index=0,
            model_id="test_model"
        )
        
        assert entry.layer_id == "test_layer"
        assert entry.layer_index == 0
        assert entry.state == WindowState.LOADING

    def test_entry_state_transitions(self):
        """Test state transitions."""
        entry = WindowEntry(
            layer_id="test_layer",
            layer_index=0,
            model_id="test_model"
        )
        
        entry.state = WindowState.READY
        assert entry.state == WindowState.READY
        
        entry.state = WindowState.ACTIVE
        assert entry.state == WindowState.ACTIVE


class TestSlidingWindowConfig:
    """Test cases for SlidingWindowConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SlidingWindowConfig()
        
        assert config.min_window_size == 3
        assert config.max_window_size == 7
        assert config.default_window_size == 5
        assert config.overlap_layers == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = SlidingWindowConfig(
            min_window_size=2,
            max_window_size=10,
            overlap_layers=2
        )
        
        assert config.min_window_size == 2
        assert config.max_window_size == 10
        assert config.overlap_layers == 2


class TestThreadSafety:
    """Test thread safety of sliding window."""

    def test_concurrent_access(self):
        """Test concurrent access to window."""
        def loader(model_id: str, layer_index: int) -> nn.Module:
            return nn.Linear(256, 256)
        
        window = SlidingWindowBuffer(
            window_size=5,
            layer_loader=loader
        )
        
        window.initialize_window("model", start_layer=0, total_layers=20)
        
        results = []
        errors = []
        
        def access_layer(idx):
            try:
                layer = window.get_layer("model", idx)
                results.append((idx, layer is not None))
            except Exception as e:
                errors.append((idx, str(e)))
        
        threads = [
            threading.Thread(target=access_layer, args=(i,))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10


class TestCreateSlidingWindow:
    """Test factory function."""

    def test_create_adaptive(self):
        """Test creating adaptive window."""
        window = create_sliding_window(window_size=5, adaptive=True)
        assert isinstance(window, AdaptiveSlidingWindow)

    def test_create_non_adaptive(self):
        """Test creating non-adaptive window."""
        window = create_sliding_window(window_size=5, adaptive=False)
        assert isinstance(window, SlidingWindowBuffer)
        assert not isinstance(window, AdaptiveSlidingWindow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])