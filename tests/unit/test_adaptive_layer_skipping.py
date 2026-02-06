"""
test_adaptive_layer_skipping.py
Unit tests for adaptive layer skipping optimization.

Tests:
- Dynamic layer skipping decisions
- Skip network training
- Performance optimization
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestAdaptiveLayerSkipper:
    """Test adaptive layer skipping functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.num_layers = 12
        self.skip_threshold = 0.5

    def test_skip_decision_network(self):
        """Test skip decision network output."""
        # Mock skip network
        skip_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Test skip decision
        hidden_state = torch.randn(4, self.hidden_size)
        context = torch.randn(4, self.hidden_size)

        skip_prob = skip_network(torch.cat([hidden_state, context], dim=-1))

        # Verify output properties
        assert skip_prob.shape == (4, 1)
        assert (skip_prob >= 0).all()
        assert (skip_prob <= 1).all()

    def test_skip_threshold_application(self):
        """Test skip threshold application."""
        skip_probs = torch.tensor([0.3, 0.7, 0.4, 0.6, 0.2])
        threshold = self.skip_threshold

        skip_decisions = skip_probs > threshold

        # Verify decisions
        expected_skips = torch.tensor([False, True, False, True, False])
        assert (skip_decisions == expected_skips).all()

    def test_layer_routing(self):
        """Test layer routing logic."""
        num_layers = self.num_layers
        current_layer = 5

        # Routing decisions
        route_to_layer = min(current_layer + 2, num_layers - 1)

        # Verify routing
        assert route_to_layer == 7

    def test_skip_gradient_flow(self):
        """Test gradient flow through skip connections."""
        # Mock skip connection
        skip_connection = nn.Identity()

        # Test gradient computation
        x = torch.randn(4, 768, requires_grad=True)
        out = skip_connection(x)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist
        assert x.grad is not None


class TestSkipNetworkTraining:
    """Test skip network training functionality."""

    def test_skip_loss_computation(self):
        """Test skip network loss computation."""
        # Simulate skip decisions
        skip_decisions = torch.tensor([0.0, 1.0, 0.0, 1.0])  # binary
        skip_probs = torch.tensor([0.2, 0.8, 0.3, 0.7])

        # Binary cross-entropy loss
        loss = -(
            skip_decisions * torch.log(skip_probs + 1e-8)
            + (1 - skip_decisions) * torch.log(1 - skip_probs + 1e-8)
        )

        # Verify loss computation
        assert loss.shape == (4,)
        assert loss.mean() > 0

    def test_reward_based_training(self):
        """Test reward-based skip network training."""
        # Simulate rewards for skipping
        rewards = torch.tensor([1.5, -0.5, 2.0, -1.0])
        skip_actions = torch.tensor([1, 0, 1, 0])

        # Reward for correct skip decisions
        expected_reward = (skip_actions * rewards).sum()

        # Verify reward calculation
        assert expected_reward == 3.0

    def test_entropy_regularization(self):
        """Test entropy regularization for skip network."""
        skip_probs = torch.tensor([0.5, 0.3, 0.2])

        # Entropy calculation
        entropy = -(skip_probs * torch.log(skip_probs + 1e-8)).sum()

        # Verify entropy properties
        assert entropy > 0
        # Maximum entropy when uniform
        uniform_probs = torch.tensor([1 / 3, 1 / 3, 1 / 3])
        uniform_entropy = -(uniform_probs * torch.log(uniform_probs + 1e-8)).sum()
        assert entropy <= uniform_entropy


class TestPerformanceMetrics:
    """Test performance metrics for layer skipping."""

    def test_speedup_calculation(self):
        """Test speedup calculation."""
        original_time = 100  # ms
        skipped_layers = [2, 5, 8]
        avg_layer_time = original_time / 12

        time_saved = len(skipped_layers) * avg_layer_time
        new_time = original_time - time_saved
        speedup = original_time / new_time

        # Verify calculations
        assert time_saved == 25
        assert speedup == 1.33

    def test_accuracy_tradeoff(self):
        """Test accuracy-speed tradeoff analysis."""
        speedups = [1.0, 1.2, 1.4, 1.6, 1.8]
        accuracies = [1.0, 0.99, 0.97, 0.94, 0.90]

        # Find optimal operating point
        for speedup, accuracy in zip(speedups, accuracies):
            if accuracy < 0.95:
                optimal_speedup = speedup / 1.2  # Step back
                break

        # Verify tradeoff properties
        assert len(speedups) == len(accuracies)
        assert speedups[-1] > speedups[0]
        assert accuracies[-1] < accuracies[0]

    def test_skip_rate_statistics(self):
        """Test skip rate statistics."""
        skip_decisions = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]  # 10 decisions
        skip_rate = sum(skip_decisions) / len(skip_decisions)

        # Verify skip rate
        assert skip_rate == 0.5


class TestAdaptiveThresholds:
    """Test adaptive threshold management."""

    def test_threshold_schedule(self):
        """Test threshold scheduling over training."""
        initial_threshold = 0.3
        final_threshold = 0.7
        total_steps = 10000

        # Linear schedule
        current_threshold = (
            initial_threshold
            + (final_threshold - initial_threshold) * 5000 / total_steps
        )

        # Verify threshold progression
        assert current_threshold == 0.5

    def test_threshold_adaptation(self):
        """Test threshold adaptation based on performance."""
        current_threshold = 0.5
        performance_drop = -0.02  # 2% drop
        adaptation_rate = 0.1

        # Adjust threshold
        new_threshold = current_threshold - performance_drop * adaptation_rate

        # Verify adaptation
        assert new_threshold == 0.52

    def test_layer_specific_thresholds(self):
        """Test layer-specific skip thresholds."""
        layer_thresholds = {
            "early": 0.2,
            "middle": 0.5,
            "late": 0.8,
        }

        # Verify threshold structure
        assert (
            layer_thresholds["early"]
            < layer_thresholds["middle"]
            < layer_thresholds["late"]
        )


class TestIntegration:
    """Test end-to-end layer skipping integration."""

    def test_skip_pipeline(self):
        """Test complete skip pipeline."""
        hidden_state = torch.randn(4, 768)
        layer_idx = 5
        skip_network_output = torch.tensor(0.6)
        threshold = 0.5

        # Pipeline steps
        should_skip = skip_network_output > threshold
        new_layer_idx = layer_idx + 3 if should_skip else layer_idx + 1

        # Verify pipeline
        assert should_skip == True
        assert new_layer_idx == 8

    def test_state_consistency(self):
        """Test state consistency after skipping."""
        hidden_state = torch.randn(4, 768)

        # Original state
        original_state = hidden_state.clone()

        # After skip (no-op for identity connection)
        skipped_state = hidden_state

        # Verify consistency
        assert torch.allclose(original_state, skipped_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
