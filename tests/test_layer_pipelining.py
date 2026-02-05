"""
Comprehensive unit tests for layer_pipelining.py
Tests layer pipelining, batch scheduling, memory optimization, and throughput metrics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

from src.optimizations.layer_pipelining import (
    PipelineConfig,
    StaleActivationPredictor,
    SpeculativeLayerExecutor,
    LayerPipeliningOptimizer,
)


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_config_values(self):
        """Test default pipelining configuration."""
        config = PipelineConfig()
        assert config.num_stages == 4
        assert config.micro_batch_size == 1
        assert config.use_speculative_execution is True
        assert config.speculation_window == 2
        assert config.confidence_threshold == 0.85
        assert config.stale_activation_tolerance == 0.1

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            num_stages=8,
            micro_batch_size=2,
            use_speculative_execution=False,
            speculation_window=4,
            confidence_threshold=0.9,
            stale_activation_tolerance=0.05,
        )

        assert config.num_stages == 8
        assert config.micro_batch_size == 2
        assert config.use_speculative_execution is False
        assert config.speculation_window == 4
        assert config.confidence_threshold == 0.9
        assert config.stale_activation_tolerance == 0.05


class TestStaleActivationPredictor:
    """Test StaleActivationPredictor for activation prediction."""

    def test_init(self):
        """Test initialization."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        assert predictor.hidden_size == 768
        assert predictor.num_layers == 80

    def test_predict_activation_first_call(self):
        """Test activation prediction on first call."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        layer_idx = 5
        current_activation = torch.randn(2, 128, 768)

        predicted, confidence = predictor.predict_activation(
            layer_idx, current_activation
        )

        # First call has no stale data, should use current
        assert predicted.shape == current_activation.shape
        assert confidence == 0.5

    def test_predict_activation_with_stale_data(self):
        """Test activation prediction with stale data."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        layer_idx = 5
        current_activation = torch.randn(2, 128, 768)
        stale_activation = torch.randn(2, 128, 768)

        # First call stores current activation
        predicted1, confidence1 = predictor.predict_activation(
            layer_idx, current_activation
        )

        # Second call should use stale activation
        predicted2, confidence2 = predictor.predict_activation(
            layer_idx, current_activation
        )

        # Should be different because using stale data + delta
        assert not torch.allclose(predicted1, predicted2)

    def test_predict_activation_updates_buffer(self):
        """Test that buffer is updated after prediction."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        layer_idx = 5
        activation1 = torch.randn(2, 128, 768)
        activation2 = torch.randn(2, 128, 768)

        predictor.predict_activation(layer_idx, activation1)

        # Buffer should contain activation1
        assert layer_idx in predictor.stale_buffer
        assert predictor.stale_buffer[layer_idx].equal(activation1)

        predictor.predict_activation(layer_idx, activation2)

        # Buffer should now contain activation2
        assert predictor.stale_buffer[layer_idx].equal(activation2)

    def test_predict_activation_different_layers(self):
        """Test prediction for different layers."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        activation = torch.randn(2, 128, 768)

        # Predict for different layers
        pred1, conf1 = predictor.predict_activation(0, activation)
        pred2, conf2 = predictor.predict_activation(10, activation)
        pred3, conf3 = predictor.predict_activation(79, activation)

        # Should have different buffers
        assert 0 in predictor.stale_buffer
        assert 10 in predictor.stale_buffer
        assert 79 in predictor.stale_buffer

    def test_clear_buffer(self):
        """Test clearing the stale buffer."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        # Add some data
        for i in range(10):
            predictor.stale_buffer[i] = torch.randn(2, 128, 768)

        predictor.clear_buffer()

        assert len(predictor.stale_buffer) == 0

    def test_predict_activation_thread_safety(self):
        """Test thread safety of prediction."""
        predictor = StaleActivationPredictor(hidden_size=768, num_layers=80)

        import threading

        results = []

        def predict():
            activation = torch.randn(2, 128, 768)
            predicted, confidence = predictor.predict_activation(5, activation)
            results.append((predicted.shape, confidence))

        # Run in multiple threads
        threads = [threading.Thread(target=predict) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete successfully
        assert len(results) == 5


class TestSpeculativeLayerExecutor:
    """Test SpeculativeLayerExecutor for speculative execution."""

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_init(self, mock_thread_class, mock_predictor_class):
        """Test initialization."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        mock_layers = [Mock() for _ in range(10)]
        config = PipelineConfig()

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        assert len(executor.layers) == 10
        assert executor.predictor is mock_predictor
        assert executor.config is config

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_execute_with_speculation_no_speculation(
        self, mock_thread_class, mock_predictor_class
    ):
        """Test execution without speculation."""
        mock_predictor = Mock()
        mock_predictor.predict_activation.return_value = (torch.randn(2, 128, 768), 0.7)
        mock_predictor_class.return_value = mock_predictor

        mock_layers = []
        for i in range(5):
            layer = Mock()
            layer.return_value = torch.randn(2, 128, 768)
            mock_layers.append(layer)

        config = PipelineConfig(use_speculative_execution=False)

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = executor.execute_with_speculation(hidden_states, 0, 5)

        # All layers should execute
        assert metrics["layers_executed"] == 5
        assert metrics["speculative_executions"] == 0

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_execute_with_speculation_high_confidence(
        self, mock_thread_class, mock_predictor_class
    ):
        """Test execution with high confidence speculation."""
        mock_predictor = Mock()
        mock_predictor.predict_activation.return_value = (torch.randn(2, 128, 768), 0.9)
        mock_predictor_class.return_value = mock_predictor

        mock_layers = []
        for i in range(5):
            layer = Mock()
            layer.return_value = torch.randn(2, 128, 768)
            mock_layers.append(layer)

        config = PipelineConfig(
            use_speculative_execution=True,
            confidence_threshold=0.85,
            stale_activation_tolerance=0.1,
        )

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = executor.execute_with_speculation(hidden_states, 0, 5)

        # Should have speculative executions
        assert metrics["speculative_executions"] > 0

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_execute_with_speculation_low_confidence(
        self, mock_thread_class, mock_predictor_class
    ):
        """Test execution with low confidence speculation."""
        mock_predictor = Mock()
        mock_predictor.predict_activation.return_value = (torch.randn(2, 128, 768), 0.5)
        mock_predictor_class.return_value = mock_predictor

        mock_layers = []
        for i in range(5):
            layer = Mock()
            layer.return_value = torch.randn(2, 128, 768)
            mock_layers.append(layer)

        config = PipelineConfig(
            use_speculative_execution=True,
            confidence_threshold=0.85,
            stale_activation_tolerance=0.1,
        )

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = executor.execute_with_speculation(hidden_states, 0, 5)

        # Low confidence should not speculate
        assert metrics["speculative_executions"] == 0

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_get_stats(self, mock_thread_class, mock_predictor_class):
        """Test getting execution statistics."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        mock_layers = [Mock() for _ in range(5)]
        config = PipelineConfig()

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        # Simulate some executions
        executor.execution_stats["speculative_hits"] = 10
        executor.execution_stats["speculative_misses"] = 2
        executor.execution_stats["verification_time_ms"] = 100
        executor.execution_stats["total_tokens"] = 100

        stats = executor.get_stats()

        assert stats["speculative_hits"] == 10
        assert stats["speculative_misses"] == 2
        assert stats["speculative_hit_rate"] == (10 / 12)
        assert "estimated_speedup" in stats

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.threading.Thread")
    def test_get_stats_empty(self, mock_thread_class, mock_predictor_class):
        """Test getting stats when no executions."""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        mock_layers = [Mock() for _ in range(5)]
        config = PipelineConfig()

        executor = SpeculativeLayerExecutor(
            layers=mock_layers, predictor=mock_predictor, config=config
        )

        stats = executor.get_stats()

        assert stats["speculative_hit_rate"] == 0.0


class TestLayerPipeliningOptimizer:
    """Test LayerPipeliningOptimizer for complete pipelining."""

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_init(self, mock_executor_class, mock_predictor_class):
        """Test initialization."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()

        mock_predictor_class.return_value = mock_predictor
        mock_executor_class.return_value = mock_executor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=80, hidden_size=768
        )

        assert optimizer.num_layers == 80
        assert optimizer.hidden_size == 768
        assert optimizer.predictor is mock_predictor
        assert optimizer.executor is mock_executor

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_forward_no_speculation(self, mock_executor_class, mock_predictor_class):
        """Test forward pass without speculation."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor.execute_with_speculation.return_value = (
            torch.randn(2, 128, 768),
            {"layers_executed": 80, "speculative_executions": 0},
        )
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=80, hidden_size=768
        )
        optimizer.layers = [Mock() for _ in range(80)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = optimizer.forward(hidden_states, use_speculation=False)

        assert output.shape == hidden_states.shape

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_forward_with_speculation(self, mock_executor_class, mock_predictor_class):
        """Test forward pass with speculation."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor.execute_with_speculation.return_value = (
            torch.randn(2, 128, 768),
            {"layers_executed": 60, "speculative_executions": 20},
        )
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=80, hidden_size=768
        )
        optimizer.layers = [Mock() for _ in range(80)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = optimizer.forward(hidden_states, use_speculation=True)

        assert metrics["speculative_executions"] > 0

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_forward_no_layers(self, mock_executor_class, mock_predictor_class):
        """Test forward pass with no layers."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=0, hidden_size=768
        )
        optimizer.layers = []

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = optimizer.forward(hidden_states, use_speculation=False)

        # Should return input unchanged
        assert torch.allclose(output, hidden_states)

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_reset(self, mock_executor_class, mock_predictor_class):
        """Test resetting optimizer state."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=80, hidden_size=768
        )

        optimizer.reset()

        mock_predictor.clear_buffer.assert_called_once()

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_get_performance_report(self, mock_executor_class, mock_predictor_class):
        """Test getting performance report."""
        mock_model = Mock()
        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor.get_stats.return_value = {
            "speculative_hits": 10,
            "speculative_misses": 2,
            "estimated_speedup": 1.2,
        }
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=80, hidden_size=768
        )

        report = optimizer.get_performance_report()

        assert report["optimizer"] == "LayerPipeliningOptimizer"
        assert report["num_layers"] == 80
        assert report["speculation_enabled"] is True
        assert "performance" in report

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_extract_layers_from_model(self, mock_executor_class, mock_predictor_class):
        """Test layer extraction from model."""
        mock_model = Mock()
        mock_model.model.layers = [Mock() for _ in range(10)]

        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=5, hidden_size=768
        )

        assert len(optimizer.layers) == 5

    @patch("src.optimizations.layer_pipelining.StaleActivationPredictor")
    @patch("src.optimizations.layer_pipelining.SpeculativeLayerExecutor")
    def test_extract_layers_transformer_h(
        self, mock_executor_class, mock_predictor_class
    ):
        """Test layer extraction from transformer.h."""
        mock_model = Mock()
        mock_model.transformer.h = [Mock() for _ in range(12)]

        mock_predictor = Mock()
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_predictor_class.return_value = mock_predictor

        optimizer = LayerPipeliningOptimizer(
            model=mock_model, num_layers=8, hidden_size=768
        )

        assert len(optimizer.layers) == 8
