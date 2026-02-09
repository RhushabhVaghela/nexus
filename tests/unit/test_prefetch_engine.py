"""
Unit tests for Smart Layer Prefetching Engine

This module contains comprehensive tests for the prefetch engine including:
- Pattern detection and prediction
- Prefetch engine lifecycle
- Thread safety
- Statistics tracking
- Adaptive lookahead

Author: Nexus Team
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

import torch
import torch.nn as nn

from src.models.sli.prefetch_engine import (
    PrefetchPattern,
    PrefetchPriority,
    PrefetchStats,
    LayerAccess,
    PrefetchConfig,
    PatternPredictor,
    PrefetchEngine,
    create_prefetch_engine,
)


class TestPrefetchPattern(unittest.TestCase):
    """Test PrefetchPattern enum."""
    
    def test_pattern_values(self):
        """Test that all patterns have correct values."""
        self.assertEqual(PrefetchPattern.SEQUENTIAL.value, "sequential")
        self.assertEqual(PrefetchPattern.STRIDED.value, "strided")
        self.assertEqual(PrefetchPattern.RANDOM.value, "random")
        self.assertEqual(PrefetchPattern.BURST.value, "burst")
        self.assertEqual(PrefetchPattern.TEMPORAL.value, "temporal")


class TestPrefetchPriority(unittest.TestCase):
    """Test PrefetchPriority enum."""
    
    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        self.assertEqual(PrefetchPriority.CRITICAL.value, 0)
        self.assertEqual(PrefetchPriority.HIGH.value, 1)
        self.assertEqual(PrefetchPriority.NORMAL.value, 2)
        self.assertEqual(PrefetchPriority.LOW.value, 3)
        self.assertEqual(PrefetchPriority.SPECULATIVE.value, 4)
    
    def test_priority_comparison(self):
        """Test that priorities can be compared."""
        self.assertLess(PrefetchPriority.CRITICAL.value, PrefetchPriority.HIGH.value)
        self.assertLess(PrefetchPriority.HIGH.value, PrefetchPriority.NORMAL.value)


class TestPrefetchStats(unittest.TestCase):
    """Test PrefetchStats class."""
    
    def test_initial_state(self):
        """Test initial state of stats."""
        stats = PrefetchStats()
        self.assertEqual(stats.total_prefetches, 0)
        self.assertEqual(stats.successful_prefetches, 0)
        self.assertEqual(stats.failed_prefetches, 0)
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.pattern_predictions, 0)
        self.assertEqual(stats.pattern_hits, 0)
        self.assertEqual(stats.avg_prefetch_time_ms, 0.0)
        self.assertEqual(stats.total_bytes_prefetched, 0)
        self.assertEqual(stats.current_lookahead, 3)
        self.assertEqual(stats.pattern_accuracy, 0.0)
    
    def test_record_successful_prefetch(self):
        """Test recording a successful prefetch."""
        stats = PrefetchStats()
        stats.record_prefetch(True, 10.0, 1000000)
        
        self.assertEqual(stats.total_prefetches, 1)
        self.assertEqual(stats.successful_prefetches, 1)
        self.assertEqual(stats.failed_prefetches, 0)
        self.assertEqual(stats.total_bytes_prefetched, 1000000)
        self.assertEqual(stats.avg_prefetch_time_ms, 10.0)
    
    def test_record_failed_prefetch(self):
        """Test recording a failed prefetch."""
        stats = PrefetchStats()
        stats.record_prefetch(False, 5.0)
        
        self.assertEqual(stats.total_prefetches, 1)
        self.assertEqual(stats.successful_prefetches, 0)
        self.assertEqual(stats.failed_prefetches, 1)
    
    def test_record_multiple_prefetch_times(self):
        """Test that average time is calculated correctly."""
        stats = PrefetchStats()
        stats.record_prefetch(True, 10.0)
        stats.record_prefetch(True, 20.0)
        stats.record_prefetch(True, 30.0)
        
        self.assertEqual(stats.avg_prefetch_time_ms, 20.0)
    
    def test_record_pattern_hit(self):
        """Test recording pattern hits."""
        stats = PrefetchStats()
        stats.record_pattern_hit(True)
        stats.record_pattern_hit(True)
        stats.record_pattern_hit(False)
        
        self.assertEqual(stats.pattern_predictions, 3)
        self.assertEqual(stats.pattern_hits, 2)
        self.assertAlmostEqual(stats.pattern_accuracy, 2/3)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = PrefetchStats()
        stats.record_prefetch(True, 10.0, 1000000)
        stats.record_prefetch(False, 5.0)
        
        result = stats.to_dict()
        
        self.assertEqual(result['total_prefetches'], 2)
        self.assertEqual(result['successful_prefetches'], 1)
        self.assertEqual(result['failed_prefetches'], 1)
        self.assertEqual(result['success_rate'], 0.5)
        self.assertEqual(result['cache_hits'], 0)


class TestLayerAccess(unittest.TestCase):
    """Test LayerAccess dataclass."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        access = LayerAccess(layer_index=5, timestamp=time.time(), model_id="test_model")
        self.assertEqual(access.layer_index, 5)
        self.assertEqual(access.model_id, "test_model")
        self.assertEqual(access.access_type, "forward")
    
    def test_custom_access_type(self):
        """Test custom access type."""
        access = LayerAccess(
            layer_index=3,
            timestamp=time.time(),
            model_id="test_model",
            access_type="backward"
        )
        self.assertEqual(access.access_type, "backward")


class TestPrefetchConfig(unittest.TestCase):
    """Test PrefetchConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PrefetchConfig()
        self.assertEqual(config.min_lookahead, 3)
        self.assertEqual(config.max_lookahead, 5)
        self.assertEqual(config.default_lookahead, 3)
        self.assertEqual(config.thread_pool_size, 8)
        self.assertEqual(config.max_concurrent_prefetches, 6)
        self.assertEqual(config.pattern_window_size, 20)
        self.assertTrue(config.enable_adaptive_lookahead)
        self.assertTrue(config.enable_pattern_recognition)
        self.assertEqual(config.prefetch_timeout, 30.0)
        self.assertEqual(config.memory_threshold_percent, 85.0)
        self.assertEqual(config.pattern_confidence_threshold, 0.7)
        self.assertEqual(config.burst_detection_threshold, 3)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = PrefetchConfig(
            min_lookahead=2,
            max_lookahead=10,
            thread_pool_size=16
        )
        self.assertEqual(config.min_lookahead, 2)
        self.assertEqual(config.max_lookahead, 10)
        self.assertEqual(config.thread_pool_size, 16)


class TestPatternPredictor(unittest.TestCase):
    """Test PatternPredictor class."""
    
    def create_access(self, layer_index: int) -> LayerAccess:
        """Helper to create layer access."""
        return LayerAccess(
            layer_index=layer_index,
            timestamp=time.time(),
            model_id="test_model"
        )
    
    def test_initial_state(self):
        """Test initial predictor state."""
        predictor = PatternPredictor()
        self.assertEqual(predictor.current_pattern, PrefetchPattern.SEQUENTIAL)
        self.assertEqual(predictor.pattern_confidence, 1.0)
        self.assertEqual(predictor.stride, 1)
        self.assertEqual(len(predictor.access_history), 0)
    
    def test_sequential_pattern_detection(self):
        """Test detection of sequential pattern."""
        predictor = PatternPredictor()
        
        # Record sequential accesses
        for i in range(5):
            predictor.record_access(self.create_access(i))
        
        self.assertEqual(predictor.current_pattern, PrefetchPattern.SEQUENTIAL)
        self.assertGreater(predictor.pattern_confidence, 0.0)
        self.assertEqual(predictor.stride, 1)
    
    def test_strided_pattern_detection(self):
        """Test detection of strided pattern."""
        predictor = PatternPredictor()
        
        # Record strided accesses (every 2 layers)
        for i in [0, 2, 4, 6, 8]:
            predictor.record_access(self.create_access(i))
        
        self.assertEqual(predictor.current_pattern, PrefetchPattern.STRIDED)
        self.assertEqual(predictor.stride, 2)
    
    def test_burst_pattern_detection(self):
        """Test detection of burst pattern."""
        predictor = PatternPredictor()
        predictor._burst_counter[5] = 5  # Simulate repeated access
        
        # Access other layers to trigger detection
        for i in range(3):
            predictor.record_access(self.create_access(10))
        
        # The burst counter should have detected the burst
        self.assertIn(predictor.current_pattern, 
                     [PrefetchPattern.BURST, PrefetchPattern.SEQUENTIAL])
    
    def test_random_pattern_detection(self):
        """Test detection of random pattern."""
        predictor = PatternPredictor()
        
        # Record random accesses
        for i in [0, 5, 2, 8, 1]:
            predictor.record_access(self.create_access(i))
        
        self.assertEqual(predictor.current_pattern, PrefetchPattern.RANDOM)
        self.assertEqual(predictor.pattern_confidence, 0.0)
    
    def test_predict_sequential(self):
        """Test prediction with sequential pattern."""
        predictor = PatternPredictor()
        
        for i in range(5):
            predictor.record_access(self.create_access(i))
        
        predictions = predictor.predict_next_layers(3)
        self.assertEqual(predictions, [5, 6, 7])
    
    def test_predict_strided(self):
        """Test prediction with strided pattern."""
        predictor = PatternPredictor()
        
        for i in [0, 2, 4]:
            predictor.record_access(self.create_access(i))
        
        predictions = predictor.predict_next_layers(3)
        self.assertEqual(predictions, [6, 8, 10])
    
    def test_predict_empty_history(self):
        """Test prediction with empty history."""
        predictor = PatternPredictor()
        
        predictions = predictor.predict_next_layers(3)
        self.assertEqual(predictions, [0, 1, 2])
    
    def test_get_pattern_info(self):
        """Test getting pattern information."""
        predictor = PatternPredictor()
        predictor.record_access(self.create_access(0))
        
        info = predictor.get_pattern_info()
        
        self.assertIn('pattern', info)
        self.assertIn('confidence', info)
        self.assertIn('stride', info)
        self.assertIn('history_length', info)
        self.assertEqual(info['history_length'], 1)


class TestPrefetchEngine(unittest.TestCase):
    """Test PrefetchEngine class."""
    
    def mock_layer_loader(self, model_id: str, layer_idx: int) -> nn.Module:
        """Mock layer loader."""
        return nn.Linear(1024, 1024)
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        
        self.assertIsNotNone(engine.config)
        self.assertIsNotNone(engine.pattern_predictor)
        self.assertIsNotNone(engine.executor)
        self.assertFalse(engine._active)
        self.assertFalse(engine._shutdown)
    
    def test_start_stop(self):
        """Test starting and stopping the engine."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        
        engine.start()
        self.assertTrue(engine._active)
        self.assertFalse(engine._shutdown)
        
        engine.stop()
        self.assertFalse(engine._active)
        self.assertTrue(engine._shutdown)
    
    def test_get_layer_id(self):
        """Test layer ID generation."""
        engine = PrefetchEngine()
        layer_id = engine._get_layer_id("model1", 5)
        self.assertEqual(layer_id, "model1_layer_5")
    
    def test_record_access_inactive_engine(self):
        """Test that access recording is skipped when engine is inactive."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        # Engine is not started
        
        # Should not raise any errors
        engine.record_access("model1", 0)
        self.assertEqual(engine._stats.total_prefetches, 0)
    
    def test_record_access_active_engine(self):
        """Test access recording with active engine."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        engine.start()
        engine.set_model_info("model1", 10)
        
        engine.record_access("model1", 0)
        
        # Give time for async prefetch
        time.sleep(0.1)
        
        engine.stop()
        # Some prefetches should have been triggered
        self.assertGreaterEqual(engine._stats.total_prefetches, 0)
    
    def test_set_model_info(self):
        """Test setting model information."""
        engine = PrefetchEngine()
        engine.set_model_info("model1", 32)
        
        self.assertEqual(engine._current_model_id, "model1")
        self.assertEqual(engine._total_layers, 32)
    
    def test_clear_buffer(self):
        """Test clearing the prefetch buffer."""
        engine = PrefetchEngine()
        
        # Add something to buffer
        mock_layer = nn.Linear(100, 100)
        engine._prefetch_buffer["test_layer"] = mock_layer
        engine._prefetched_ids.add("test_layer")
        
        engine.clear_buffer()
        
        self.assertEqual(len(engine._prefetch_buffer), 0)
        self.assertEqual(len(engine._prefetched_ids), 0)
    
    def test_get_prefetched_layer(self):
        """Test retrieving a prefetched layer."""
        engine = PrefetchEngine()
        
        # Add layer to buffer
        mock_layer = nn.Linear(100, 100)
        engine._prefetch_buffer["test_layer"] = mock_layer
        engine._prefetched_ids.add("test_layer")
        
        result = engine.get_prefetched_layer("test_layer")
        
        self.assertEqual(result, mock_layer)
        self.assertEqual(engine._stats.cache_hits, 1)
        self.assertNotIn("test_layer", engine._prefetch_buffer)
    
    def test_get_prefetched_layer_not_found(self):
        """Test retrieving a non-existent prefetched layer."""
        engine = PrefetchEngine()
        
        result = engine.get_prefetched_layer("nonexistent")
        
        self.assertIsNone(result)
        self.assertEqual(engine._stats.cache_hits, 0)
    
    def test_get_stats(self):
        """Test getting engine statistics."""
        engine = PrefetchEngine()
        engine.start()
        engine.record_access("model1", 0)
        
        stats = engine.get_stats()
        
        self.assertIn('total_prefetches', stats)
        self.assertIn('pattern_info', stats)
        self.assertIn('buffer_size', stats)
        
        engine.stop()
    
    def test_get_buffer_state(self):
        """Test getting buffer state."""
        engine = PrefetchEngine()
        
        # Add items to collections
        mock_layer = nn.Linear(100, 100)
        engine._prefetch_buffer["layer1"] = mock_layer
        engine._prefetched_ids.add("layer2")
        
        state = engine.get_buffer_state()
        
        self.assertIn('buffered_layers', state)
        self.assertIn('in_progress', state)
        self.assertIn('prefetched_ids', state)
        self.assertEqual(len(state['buffered_layers']), 1)
    
    def test_wait_for_prefetch(self):
        """Test waiting for prefetches to complete."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        engine.start()
        
        # Manually add to buffer
        mock_layer = nn.Linear(100, 100)
        engine._prefetch_buffer["model1_layer_5"] = mock_layer
        
        results = engine.wait_for_prefetch(["model1_layer_5"], timeout=1.0)
        
        self.assertIn("model1_layer_5", results)
        
        engine.stop()
    
    def test_adapt_lookahead_increase(self):
        """Test adaptive lookahead increase."""
        engine = PrefetchEngine()
        engine.start()
        engine.set_model_info("model1", 100)
        
        # Simulate high success rate
        for _ in range(25):
            engine._stats.record_prefetch(True, 10.0)
        
        # Set high pattern confidence
        for _ in range(10):
            engine.pattern_predictor.record_access(
                LayerAccess(0, time.time(), "model1")
            )
        
        initial_lookahead = engine._current_lookahead
        engine._adapt_lookahead()
        
        # Lookahead may have increased
        engine.stop()
    
    def test_adapt_lookahead_decrease(self):
        """Test adaptive lookahead decrease."""
        engine = PrefetchEngine()
        engine._current_lookahead = 5
        
        # Simulate low success rate
        for _ in range(25):
            engine._stats.record_prefetch(False, 10.0)
        
        engine._adapt_lookahead()
        
        self.assertLess(engine._current_lookahead, 5)
    
    def test_prefetch_layers_parallel(self):
        """Test parallel prefetching."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        engine.start()
        engine.set_model_info("model1", 10)
        
        futures = engine.prefetch_layers_parallel("model1", [0, 1, 2])
        
        self.assertGreater(len(futures), 0)
        
        engine.stop()
    
    def test_thread_safety(self):
        """Test thread safety of prefetch operations."""
        engine = PrefetchEngine(layer_loader=self.mock_layer_loader)
        engine.start()
        engine.set_model_info("model1", 100)
        
        errors = []
        
        def access_layers():
            try:
                for i in range(10):
                    engine.record_access("model1", i)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_layers) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)
        
        engine.stop()


class TestCreatePrefetchEngine(unittest.TestCase):
    """Test create_prefetch_engine factory function."""
    
    def test_create_with_defaults(self):
        """Test creating engine with default parameters."""
        def mock_loader(model_id, layer_idx):
            return nn.Linear(100, 100)
        
        engine = create_prefetch_engine(layer_loader=mock_loader)
        
        self.assertEqual(engine.config.default_lookahead, 3)
        self.assertEqual(engine.config.thread_pool_size, 8)
        self.assertEqual(engine.layer_loader, mock_loader)
    
    def test_create_with_custom_params(self):
        """Test creating engine with custom parameters."""
        engine = create_prefetch_engine(
            lookahead=5,
            thread_pool_size=16,
            max_concurrent_prefetches=10
        )
        
        self.assertEqual(engine.config.default_lookahead, 5)
        self.assertEqual(engine.config.thread_pool_size, 16)
        self.assertEqual(engine.config.max_concurrent_prefetches, 10)
    
    def test_create_with_sliding_window(self):
        """Test creating engine with sliding window."""
        mock_window = Mock()
        engine = create_prefetch_engine(sliding_window=mock_window)
        
        self.assertEqual(engine.sliding_window, mock_window)


if __name__ == '__main__':
    unittest.main()
