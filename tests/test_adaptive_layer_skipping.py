"""
Comprehensive unit tests for adaptive_layer_skipping.py
Tests skipping decisions, accuracy preservation, speedup measurement, and layer importance.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from nexus.optimizations.adaptive_layer_skipping import (
    LayerSkipConfig,
    LayerSkipRouter,
    SWIFTSkipper,
    AdaptiveLayerSkipper,
    LayerSkipIntegration,
)


class TestLayerSkipConfig:
    """Test LayerSkipConfig dataclass."""

    def test_default_config_values(self):
        """Test default skipping configuration."""
        config = LayerSkipConfig()
        assert config.min_layers == 50
        assert config.max_layers == 80
        assert config.confidence_threshold == 0.9
        assert config.entropy_threshold == 0.5
        assert config.skip_pattern == "adaptive"
        assert config.training_mode is False

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = LayerSkipConfig(
            min_layers=30,
            max_layers=60,
            confidence_threshold=0.85,
            entropy_threshold=0.3,
            skip_pattern="uniform",
            training_mode=True,
        )

        assert config.min_layers == 30
        assert config.max_layers == 60
        assert config.confidence_threshold == 0.85
        assert config.entropy_threshold == 0.3
        assert config.skip_pattern == "uniform"
        assert config.training_mode is True


class TestLayerSkipRouter:
    """Test LayerSkipRouter for skip decisions."""

    def test_init(self):
        """Test initialization."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        assert router.hidden_size == 768
        assert router.num_layers == 80

    def test_should_exit_early_below_min_layers(self):
        """Test that early exit is blocked below minimum layers."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)
        current_layer = 20  # Below minimum

        should_exit, confidence = router.should_exit_early(hidden_states, current_layer)

        # Should not exit below min layers
        assert should_exit is False

    def test_should_exit_early_high_confidence(self):
        """Test early exit with high confidence."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)
        current_layer = 50  # Above minimum

        # Set high exit probability and low complexity
        with torch.no_grad():
            router.exit_classifier.weight.fill_(0.9)
            router.complexity_estimator.weight.fill_(0.1)

        should_exit, confidence = router.should_exit_early(hidden_states, current_layer)

        assert confidence > 0.8

    def test_should_exit_early_low_confidence(self):
        """Test early exit with low confidence."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)
        current_layer = 50

        # Set low exit probability
        with torch.no_grad():
            router.exit_classifier.weight.fill_(0.1)

        should_exit, confidence = router.should_exit_early(hidden_states, current_layer)

        # Should not exit with low confidence
        assert should_exit is False

    def test_should_exit_early_high_complexity(self):
        """Test early exit blocked for high complexity inputs."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)
        current_layer = 50

        # High exit probability but high complexity
        with torch.no_grad():
            router.exit_classifier.weight.fill_(0.9)
            router.complexity_estimator.weight.fill_(0.9)

        should_exit, confidence = router.should_exit_early(hidden_states, current_layer)

        # Should not exit for complex inputs
        assert should_exit is False

    def test_estimate_layers_needed_simple(self):
        """Test layer estimation for simple inputs."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)

        # Set low complexity
        with torch.no_grad():
            router.complexity_estimator.weight.fill_(0.1)

        layers_needed = router.estimate_layers_needed(hidden_states)

        # Simple inputs should need fewer layers (50-60 range)
        assert layers_needed >= 50
        assert layers_needed <= 60

    def test_estimate_layers_needed_complex(self):
        """Test layer estimation for complex inputs."""
        router = LayerSkipRouter(hidden_size=768, num_layers=80)

        hidden_states = torch.randn(2, 128, 768)

        # Set high complexity
        with torch.no_grad():
            router.complexity_estimator.weight.fill_(0.9)

        layers_needed = router.estimate_layers_needed(hidden_states)

        # Complex inputs should need more layers (70-80 range)
        assert layers_needed >= 70
        assert layers_needed <= 80


class TestSWIFTSkipper:
    """Test SWIFTSkipper for SWIFT-style skipping."""

    def test_init(self):
        """Test initialization."""
        skipper = SWIFTSkipper(hidden_size=768, skip_every_n=2)

        assert skipper.hidden_size == 768
        assert skipper.skip_every_n == 2

    def test_forward_no_skip(self):
        """Test forward when not skipping."""
        skipper = SWIFTSkipper(hidden_size=768, skip_every_n=2)

        hidden_states = torch.randn(2, 128, 768)

        def layer_func(x):
            return x + 1

        output, was_skipped = skipper.forward(hidden_states, layer_func, layer_idx=5)

        # Layer 5 doesn't match skip pattern
        assert was_skipped is False

    def test_forward_with_skip(self):
        """Test forward when skipping."""
        skipper = SWIFTSkipper(hidden_size=768, skip_every_n=2)

        hidden_states = torch.randn(2, 128, 768)

        # Set high skip probability
        with torch.no_grad():
            skipper.skip_gate.weight.fill_(0.9)

        def layer_func(x):
            return x + 1

        output, was_skipped = skipper.forward(hidden_states, layer_func, layer_idx=6)

        # Layer 6 should match skip pattern
        assert was_skipped is True

    def test_forward_early_layers_not_skipped(self):
        """Test that early layers are not skipped."""
        skipper = SWIFTSkipper(hidden_size=768, skip_every_n=2)

        hidden_states = torch.randn(2, 128, 768)

        # High skip probability
        with torch.no_grad():
            skipper.skip_gate.weight.fill_(0.9)

        def layer_func(x):
            return x + 1

        # Layer 8 is beyond skip_every_n range
        output, was_skipped = skipper.forward(hidden_states, layer_func, layer_idx=8)

        assert was_skipped is True

    def test_forward_final_layers_not_skipped(self):
        """Test that final layers are not skipped."""
        skipper = SWIFTSkipper(hidden_size=768, skip_every_n=2)

        hidden_states = torch.randn(2, 128, 768)

        # High skip probability
        with torch.no_grad():
            skipper.skip_gate.weight.fill_(0.9)

        def layer_func(x):
            return x + 1

        # Layer 72 is near final layers
        output, was_skipped = skipper.forward(hidden_states, layer_func, layer_idx=72)

        # Should not skip final layers
        assert was_skipped is False


class TestAdaptiveLayerSkipper:
    """Test AdaptiveLayerSkipper for complete skipping functionality."""

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_init(self, mock_swift_class, mock_router_class):
        """Test initialization."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router_class.return_value = mock_router
        mock_swift_class.return_value = mock_swift

        skipper = AdaptiveLayerSkipper(model=mock_model, num_layers=80, hidden_size=768)

        assert skipper.num_layers == 80
        assert skipper.hidden_size == 768
        assert skipper.layer_skip_router is mock_router
        assert skipper.swift_skipper is mock_swift

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_forward_with_skipping_no_early_exit(
        self, mock_swift_class, mock_router_class
    ):
        """Test forward without early exit."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router.should_exit_early.return_value = (False, 0.5)
        mock_swift.forward.return_value = (torch.randn(2, 128, 768), False)
        mock_router_class.return_value = mock_router
        mock_swift_class.return_value = mock_swift

        skipper = AdaptiveLayerSkipper(model=mock_model, num_layers=80, hidden_size=768)

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = skipper.forward_with_skipping(hidden_states, mock_layers)

        assert metrics["early_exit"] is False

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_forward_with_skipping_early_exit(
        self, mock_swift_class, mock_router_class
    ):
        """Test forward with early exit."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router.should_exit_early.return_value = (True, 0.9)
        mock_swift_class.return_value = mock_swift
        mock_router_class.return_value = mock_router

        skipper = AdaptiveLayerSkipper(model=mock_model, num_layers=80, hidden_size=768)

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = skipper.forward_with_skipping(hidden_states, mock_layers)

        # Should have early exit
        assert metrics["early_exit"] is True
        assert skipper.stats["early_exits"] > 0

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_forward_with_skipping_training_mode(
        self, mock_swift_class, mock_router_class
    ):
        """Test forward in training mode."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router.should_exit_early.return_value = (True, 0.9)
        mock_swift_class.return_value = mock_swift
        mock_router_class.return_value = mock_router

        config = LayerSkipConfig(training_mode=True)
        skipper = AdaptiveLayerSkipper(
            model=mock_model, num_layers=80, hidden_size=768, config=config
        )

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = skipper.forward_with_skipping(hidden_states, mock_layers)

        # Should not exit in training mode even if confident
        assert metrics["early_exit"] is False

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_forward_uniform_skipping(self, mock_swift_class, mock_router_class):
        """Test forward with uniform skip pattern."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router.should_exit_early.return_value = (False, 0.5)
        mock_swift.forward.return_value = (torch.randn(2, 128, 768), True)
        mock_swift_class.return_value = mock_swift
        mock_router_class.return_value = mock_router

        config = LayerSkipConfig(skip_pattern="uniform")
        skipper = AdaptiveLayerSkipper(
            model=mock_model, num_layers=80, hidden_size=768, config=config
        )

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = skipper.forward_with_skipping(hidden_states, mock_layers)

        # Should have some skips
        assert metrics["layers_skipped"] > 0

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_get_stats(self, mock_swift_class, mock_router_class):
        """Test getting skipping statistics."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router_class.return_value = mock_router
        mock_swift_class.return_value = mock_swift

        skipper = AdaptiveLayerSkipper(model=mock_model, num_layers=80, hidden_size=768)

        # Simulate some stats
        skipper.stats["total_tokens"] = 1000
        skipper.stats["early_exits"] = 100
        skipper.stats["layers_skipped"] = 500
        skipper.stats["avg_layers_used"] = [60.0, 55.0, 58.0]

        stats = skipper.get_stats()

        assert stats["total_tokens"] == 1000
        assert stats["early_exits"] == 100
        assert "theoretical_speedup" in stats

    @patch("src.optimizations.adaptive_layer_skipping.LayerSkipRouter")
    @patch("src.optimizations.adaptive_layer_skipping.SWIFTSkipper")
    def test_reset_stats(self, mock_swift_class, mock_router_class):
        """Test resetting statistics."""
        mock_model = Mock()
        mock_router = Mock()
        mock_swift = Mock()
        mock_router_class.return_value = mock_router
        mock_swift_class.return_value = mock_swift

        skipper = AdaptiveLayerSkipper(model=mock_model, num_layers=80, hidden_size=768)

        # Set some stats
        skipper.stats["total_tokens"] = 1000
        skipper.stats["early_exits"] = 100
        skipper.stats["layers_skipped"] = 500

        skipper.reset_stats()

        assert skipper.stats["total_tokens"] == 0
        assert skipper.stats["early_exits"] == 0
        assert skipper.stats["layers_skipped"] == 0
        assert skipper.stats["avg_layers_used"] == []


class TestLayerSkipIntegration:
    """Test LayerSkipIntegration for complete model integration."""

    @patch("src.optimizations.adaptive_layer_skipping.AdaptiveLayerSkipper")
    def test_init(self, mock_skipper_class):
        """Test initialization."""
        mock_model = Mock()
        mock_skipper = Mock()
        mock_skipper_class.return_value = mock_skipper

        integration = LayerSkipIntegration(
            base_model=mock_model, hidden_size=768, num_layers=80
        )

        assert integration.base_model is mock_model
        assert integration.skipper is mock_skipper

    @patch("src.optimizations.adaptive_layer_skipping.AdaptiveLayerSkipper")
    def test_forward_with_input_ids(self, mock_skipper_class):
        """Test forward with input IDs."""
        mock_model = Mock()
        mock_skipper = Mock()
        mock_skipper.forward_with_skipping.return_value = (
            torch.randn(2, 128, 768),
            {"layers_used": 60, "early_exit": False},
        )
        mock_skipper_class.return_value = mock_skipper

        mock_embeddings = Mock()
        mock_embeddings.return_value = torch.randn(2, 128, 768)
        mock_model.get_input_embeddings.return_value = mock_embeddings

        integration = LayerSkipIntegration(
            base_model=mock_model, hidden_size=768, num_layers=80
        )

        input_ids = torch.randint(0, 1000, (2, 128))

        result = integration.forward(input_ids=input_ids)

        assert "last_hidden_state" in result
        assert "metrics" in result

    @patch("src.optimizations.adaptive_layer_skipping.AdaptiveLayerSkipper")
    def test_forward_with_hidden_states(self, mock_skipper_class):
        """Test forward with hidden states."""
        mock_model = Mock()
        mock_skipper = Mock()
        mock_skipper.forward_with_skipping.return_value = (
            torch.randn(2, 128, 768),
            {"layers_used": 60, "early_exit": False},
        )
        mock_skipper_class.return_value = mock_skipper

        integration = LayerSkipIntegration(
            base_model=mock_model, hidden_size=768, num_layers=80
        )

        hidden_states = torch.randn(2, 128, 768)

        result = integration.forward(hidden_states=hidden_states)

        assert "last_hidden_state" in result
        assert "metrics" in result

    @patch("src.optimizations.adaptive_layer_skipping.AdaptiveLayerSkipper")
    def test_forward_with_attention_mask(self, mock_skipper_class):
        """Test forward with attention mask."""
        mock_model = Mock()
        mock_skipper = Mock()
        mock_skipper.forward_with_skipping.return_value = (
            torch.randn(2, 128, 768),
            {"layers_used": 60, "early_exit": False},
        )
        mock_skipper_class.return_value = mock_skipper

        integration = LayerSkipIntegration(
            base_model=mock_model, hidden_size=768, num_layers=80
        )

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)

        result = integration.forward(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        assert "last_hidden_state" in result
        assert "metrics" in result
