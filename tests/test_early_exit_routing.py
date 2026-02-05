"""
Comprehensive unit tests for early_exit_routing.py
Tests router decision logic, early exit thresholds, confidence scoring, and fallback behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from src.optimizations.early_exit_routing import (
    DynamicRoutingConfig,
    TokenRouter,
    DynamicLayerRouter,
    EarlyExitRouter,
    AdaptiveExitLayer,
)


class TestDynamicRoutingConfig:
    """Test DynamicRoutingConfig dataclass."""

    def test_default_config_values(self):
        """Test default routing configuration."""
        config = DynamicRoutingConfig()
        assert config.min_layers == 30
        assert config.max_layers == 80
        assert config.confidence_threshold == 0.85
        assert config.entropy_threshold == 0.5
        assert config.use_token_routing is True
        assert config.use_layer_routing is True
        assert config.training_mode is False

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = DynamicRoutingConfig(
            min_layers=20,
            max_layers=60,
            confidence_threshold=0.9,
            entropy_threshold=0.3,
            use_token_routing=False,
            use_layer_routing=False,
            training_mode=True,
        )

        assert config.min_layers == 20
        assert config.max_layers == 60
        assert config.confidence_threshold == 0.9
        assert config.entropy_threshold == 0.3
        assert config.use_token_routing is False
        assert config.use_layer_routing is False
        assert config.training_mode is True


class TestTokenRouter:
    """Test TokenRouter for per-token routing decisions."""

    def test_init(self):
        """Test initialization."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        assert router.hidden_size == 768
        assert router.num_layers == 80

    def test_estimate_exit_layer_low_complexity(self):
        """Test exit layer estimation for low complexity tokens."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Set low complexity
        with torch.no_grad():
            for param in router.complexity_estimator.parameters():
                param.fill_(0.1)

        exit_layers = router.estimate_exit_layer(hidden_states)

        # Low complexity should exit early (around 40% of layers)
        expected_exit = int(80 * 0.4)  # 32 layers
        assert (exit_layers == expected_exit).all()

    def test_estimate_exit_layer_high_complexity(self):
        """Test exit layer estimation for high complexity tokens."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Set high complexity
        with torch.no_grad():
            for param in router.complexity_estimator.parameters():
                param.fill_(0.9)

        exit_layers = router.estimate_exit_layer(hidden_states)

        # High complexity should exit late (around 70% of layers)
        expected_exit = int(80 * 0.7)  # 56 layers
        assert (exit_layers == expected_exit).all()

    def test_estimate_exit_layer_very_high_complexity(self):
        """Test exit layer estimation for very high complexity tokens."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Set very high complexity
        with torch.no_grad():
            for param in router.complexity_estimator.parameters():
                param.fill_(1.0)

        exit_layers = router.estimate_exit_layer(hidden_states)

        # Very high complexity should exit at max layers
        expected_exit = 80
        assert (exit_layers == expected_exit).all()

    def test_should_exit_at_layer_confident(self):
        """Test exit decision when confident."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)
        layer_idx = 50

        # Set high confidence
        with torch.no_grad():
            for predictor in router.exit_predictor:
                for param in predictor.parameters():
                    param.fill_(0.9)

        exit_mask, confidence = router.should_exit_at_layer(hidden_states, layer_idx)

        # Should exit with high confidence
        assert confidence.mean() > 0.8

    def test_should_exit_at_layer_not_confident(self):
        """Test exit decision when not confident."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)
        layer_idx = 50

        # Set low confidence
        with torch.no_grad():
            for predictor in router.exit_predictor:
                for param in predictor.parameters():
                    param.fill_(0.1)

        exit_mask, confidence = router.should_exit_at_layer(hidden_states, layer_idx)

        # Should not exit with low confidence
        assert confidence.mean() < 0.5

    def test_should_exit_at_layer_final_layer(self):
        """Test that all tokens must exit at final layer."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)
        layer_idx = 90  # Beyond num_layers

        exit_mask, confidence = router.should_exit_at_layer(hidden_states, layer_idx)

        # All should exit at final layer
        assert exit_mask.all()
        assert confidence.mean() == 1.0

    def test_should_exit_different_tokens(self):
        """Test that different tokens can have different exit decisions."""
        router = TokenRouter(hidden_size=768, num_layers=80)

        batch_size = 4
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)
        layer_idx = 50

        # Mix of confidence levels
        with torch.no_grad():
            for i, predictor in enumerate(router.exit_predictor):
                for param in predictor.parameters():
                    param.fill_(0.3 + i * 0.1)

        exit_mask, confidence = router.should_exit_at_layer(hidden_states, layer_idx)

        # Some tokens should exit, some shouldn't
        assert exit_mask.any() and (~exit_mask).any()


class TestDynamicLayerRouter:
    """Test DynamicLayerRouter for layer selection."""

    def test_init(self):
        """Test initialization."""
        router = DynamicLayerRouter(hidden_size=768, num_layers=80)

        assert router.hidden_size == 768
        assert router.num_layers == 80
        assert router.layer_gates.shape == (80,)

    def test_compute_layer_mask_all_executed(self):
        """Test layer mask when all layers should execute."""
        router = DynamicLayerRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Set high importance for all layers
        with torch.no_grad():
            router.layer_importance.weight.fill_(0.9)
            router.layer_gates.fill_(1.0)

        layer_mask = router.compute_layer_mask(hidden_states, min_layers=30)

        # Minimum layers should be executed
        assert layer_mask[:30].sum() == 30

    def test_compute_layer_mask_some_skipped(self):
        """Test layer mask when some layers can be skipped."""
        router = DynamicLayerRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        # Set low importance for later layers
        with torch.no_grad():
            router.layer_importance.weight.fill_(0.1)
            router.layer_gates.fill_(0.1)

        layer_mask = router.compute_layer_mask(hidden_states, min_layers=30)

        # Should skip some layers
        assert layer_mask.sum() < 80

    def test_compute_layer_mask_min_layers_always_executed(self):
        """Test that minimum layers are always executed."""
        router = DynamicLayerRouter(hidden_size=768, num_layers=80)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, 768)

        layer_mask = router.compute_layer_mask(hidden_states, min_layers=40)

        # First 40 layers should be 1
        assert (layer_mask[:40] == 1.0).all()


class TestEarlyExitRouter:
    """Test EarlyExitRouter for complete early exit functionality."""

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_init(self, mock_layer_router_class, mock_token_router_class):
        """Test initialization."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        assert router.num_layers == 80
        assert router.hidden_size == 768
        assert router.token_router is mock_token_router
        assert router.layer_router is mock_layer_router

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_forward_with_routing_all_layers(
        self, mock_layer_router_class, mock_token_router_class
    ):
        """Test forward pass when all layers execute."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()
        mock_layer_router.compute_layer_mask.return_value = torch.ones(80)
        mock_token_router.should_exit_at_layer.return_value = (
            torch.zeros(2, 128, dtype=torch.bool),
            torch.zeros(2, 128) + 0.1,
        )
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        # Create mock layers
        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = router.forward_with_routing(hidden_states, mock_layers)

        assert metrics["layers_executed"] == 80

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_forward_with_routing_early_exit(
        self, mock_layer_router_class, mock_token_router_class
    ):
        """Test forward pass with early exit."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()

        # Layers 0-49 execute, layer 50 causes early exit
        layer_mask = torch.ones(80)
        mock_layer_router.compute_layer_mask.return_value = layer_mask

        # At layer 50, tokens exit
        should_exit = torch.zeros(2, 128, dtype=torch.bool)
        should_exit[:, :, 50] = True  # Some tokens exit
        mock_token_router.should_exit_at_layer.return_value = (
            should_exit,
            torch.zeros(2, 128) + 0.9,
        )
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)

        output, metrics = router.forward_with_routing(hidden_states, mock_layers)

        # Should have some early exits
        assert router.stats["early_exits"] > 0

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_forward_with_routing_with_attention_mask(
        self, mock_layer_router_class, mock_token_router_class
    ):
        """Test forward pass with attention mask."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()
        mock_layer_router.compute_layer_mask.return_value = torch.ones(80)
        mock_token_router.should_exit_at_layer.return_value = (
            torch.zeros(2, 128, dtype=torch.bool),
            torch.zeros(2, 128) + 0.1,
        )
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        mock_layers = [Mock() for _ in range(80)]
        for i, layer in enumerate(mock_layers):
            layer.return_value = [torch.randn(2, 128, 768)]

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)

        output, metrics = router.forward_with_routing(
            hidden_states, mock_layers, attention_mask=attention_mask
        )

        # Should process with mask
        assert output.shape == hidden_states.shape

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_get_stats(self, mock_layer_router_class, mock_token_router_class):
        """Test getting routing statistics."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        # Simulate some stats
        router.stats["total_tokens"] = 1000
        router.stats["early_exits"] = 200
        router.stats["avg_exit_layer"] = [60.0, 55.0, 58.0]

        stats = router.get_stats()

        assert stats["total_tokens"] == 1000
        assert stats["early_exits"] == 200
        assert "theoretical_speedup" in stats

    @patch("src.optimizations.early_exit_routing.TokenRouter")
    @patch("src.optimizations.early_exit_routing.DynamicLayerRouter")
    def test_reset_stats(self, mock_layer_router_class, mock_token_router_class):
        """Test resetting statistics."""
        mock_model = Mock()
        mock_token_router = Mock()
        mock_layer_router = Mock()
        mock_token_router_class.return_value = mock_token_router
        mock_layer_router_class.return_value = mock_layer_router

        router = EarlyExitRouter(model=mock_model, num_layers=80, hidden_size=768)

        # Set some stats
        router.stats["total_tokens"] = 1000
        router.stats["early_exits"] = 200

        router.reset_stats()

        assert router.stats["total_tokens"] == 0
        assert router.stats["early_exits"] == 0
        assert router.stats["avg_exit_layer"] == []


class TestAdaptiveExitLayer:
    """Test AdaptiveExitLayer for wrapping layers with exit capability."""

    def test_init(self):
        """Test initialization."""
        base_layer = Mock()
        exit_predictor = Mock()

        wrapper = AdaptiveExitLayer(
            base_layer=base_layer, exit_predictor=exit_predictor, layer_idx=5
        )

        assert wrapper.base_layer is base_layer
        assert wrapper.exit_predictor is exit_predictor
        assert wrapper.layer_idx == 5

    def test_forward_no_exit(self):
        """Test forward pass when not exiting."""
        base_layer = Mock()
        base_layer.return_value = [torch.randn(2, 128, 768)]

        exit_predictor = Mock()
        exit_predictor.return_value = torch.zeros(2, 128, 1) + 0.1  # Low confidence

        wrapper = AdaptiveExitLayer(
            base_layer=base_layer, exit_predictor=exit_predictor, layer_idx=5
        )

        hidden_states = torch.randn(2, 128, 768)

        output, should_exit = wrapper.forward(hidden_states)

        base_layer.assert_called_once()
        assert should_exit.sum() == 0  # No exits

    def test_forward_with_exit(self):
        """Test forward pass when exiting."""
        base_layer = Mock()
        base_layer.return_value = [torch.randn(2, 128, 768)]

        exit_predictor = Mock()
        exit_predictor.return_value = torch.zeros(2, 128, 1) + 0.9  # High confidence

        wrapper = AdaptiveExitLayer(
            base_layer=base_layer, exit_predictor=exit_predictor, layer_idx=50
        )

        hidden_states = torch.randn(2, 128, 768)

        output, should_exit = wrapper.forward(hidden_states)

        base_layer.assert_called_once()
        assert should_exit.all()  # All exit

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        base_layer = Mock()
        base_layer.return_value = [torch.randn(2, 128, 768)]

        exit_predictor = Mock()
        exit_predictor.return_value = torch.zeros(2, 128, 1) + 0.5

        wrapper = AdaptiveExitLayer(
            base_layer=base_layer, exit_predictor=exit_predictor, layer_idx=5
        )

        hidden_states = torch.randn(2, 128, 768)
        attention_mask = torch.ones(2, 128)

        output, should_exit = wrapper.forward(
            hidden_states, attention_mask=attention_mask
        )

        base_layer.assert_called_once()
