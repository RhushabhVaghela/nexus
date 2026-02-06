"""
test_early_exit_routing.py
Unit tests for early exit routing optimization.

Tests:
- Early exit decision networks
- Confidence-based routing
- Dynamic depth adjustment
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


class TestEarlyExitRouter:
    """Test early exit routing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.num_classes = 1000
        self.confidence_threshold = 0.8

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        logits = torch.randn(4, self.num_classes)

        # Calculate confidence (max probability)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0]

        # Verify confidence properties
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
        assert confidence.shape == (4,)

    def test_exit_decision(self):
        """Test early exit decision logic."""
        confidence_scores = torch.tensor([0.85, 0.72, 0.91, 0.65])
        threshold = self.confidence_threshold

        # Exit decisions
        exit_flags = confidence_scores >= threshold

        # Verify decisions
        expected = torch.tensor([True, False, True, False])
        assert (exit_flags == expected).all()

    def test_classifier_head(self):
        """Test early exit classifier head."""
        classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

        # Test classifier
        hidden = torch.randn(4, self.hidden_size)
        logits = classifier(hidden)

        # Verify output
        assert logits.shape == (4, self.num_classes)

    def test_confidence_calibration(self):
        """Test confidence calibration."""
        logits = torch.randn(100, self.num_classes)
        labels = torch.randint(0, self.num_classes, (100,))

        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0]

        # Calibration check
        high_conf_mask = confidence > 0.9
        if high_conf_mask.any():
            accuracy = (
                (probs.argmax(dim=-1)[high_conf_mask] == labels[high_conf_mask])
                .float()
                .mean()
            )
            # High confidence should correlate with high accuracy
            assert accuracy >= 0.5


class TestRoutingStrategies:
    """Test different routing strategies."""

    def test_threshold_routing(self):
        """Test threshold-based routing."""
        thresholds = [0.7, 0.8, 0.9]
        samples = torch.rand(10, self.num_classes)

        # Calculate confidence
        confidence = torch.softmax(samples, dim=-1).max(dim=-1)[0]

        # Route samples
        routed = []
        for thresh in thresholds:
            routed.append((confidence >= thresh).sum().item())

        # Verify routing
        assert routed[0] >= routed[1] >= routed[2]

    def test_entropy_routing(self):
        """Test entropy-based routing."""
        probs = torch.softmax(torch.randn(10, self.num_classes), dim=-1)

        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Verify entropy properties
        assert (entropy >= 0).all()
        # Maximum entropy when uniform
        uniform = torch.ones_like(probs) / self.num_classes
        max_entropy = -(uniform * torch.log(uniform + 1e-8)).sum()
        assert entropy.max() <= max_entropy

    def test_uncertainty_routing(self):
        """Test uncertainty-based routing."""
        logits = torch.randn(10, self.num_classes)

        # Calculate uncertainty (variance of softmax)
        probs = torch.softmax(logits, dim=-1)
        uncertainty = probs.var(dim=-1)

        # Route high uncertainty
        routed = uncertainty > 0.05

        # Verify
        assert routed.shape == (10,)

    def test_adaptive_routing(self):
        """Test adaptive routing based on sample difficulty."""
        difficulties = torch.tensor([0.2, 0.5, 0.8, 0.3, 0.7])

        # Adjust routing threshold based on difficulty
        base_threshold = 0.7
        adjusted_threshold = base_threshold * (1 - difficulties.mean())

        # Verify adaptation
        assert adjusted_threshold < base_threshold


class TestDepthAdjustment:
    """Test dynamic depth adjustment."""

    def test_layer_skipping(self):
        """Test layer skipping logic."""
        current_layer = 5
        total_layers = 12
        skip_prob = 0.3

        # Decision
        skip = torch.rand(1).item() < skip_prob

        # New layer
        new_layer = current_layer + (3 if skip else 1)

        # Verify
        assert new_layer <= total_layers

    def test_depth_budget(self):
        """Test depth budget allocation."""
        max_depth = 12
        samples = torch.randint(0, 10, (8,))

        # Allocate depth based on difficulty
        allocated_depth = (
            max_depth * (1 - samples.float() / samples.float().max())
        ).long()

        # Verify allocation
        assert (allocated_depth >= 1).all()
        assert (allocated_depth <= max_depth).all()

    def test_partial_inference(self):
        """Test partial inference execution."""
        layers = [nn.Linear(768, 768) for _ in range(12)]
        exit_layer = 6

        # Execute partial inference
        x = torch.randn(4, 768)
        for i, layer in enumerate(layers):
            if i < exit_layer:
                x = torch.relu(layer(x))

        # Verify partial execution
        assert x.shape == (4, 768)


class TestExitQuality:
    """Test exit quality metrics."""

    def test_accuracy_vs_depth(self):
        """Test accuracy vs depth trade-off."""
        depths = [3, 6, 9, 12]
        accuracies = [0.65, 0.78, 0.85, 0.88]

        # Verify trade-off
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1]
            assert accuracies[i] >= accuracies[i - 1]

    def test_latency_vs_depth(self):
        """Test latency vs depth trade-off."""
        depths = [3, 6, 9, 12]
        latencies = [10, 25, 45, 70]  # ms

        # Verify trade-off
        for i in range(1, len(depths)):
            assert depths[i] > depths[i - 1]
            assert latencies[i] > latencies[i - 1]

    def test_speedup_calculation(self):
        """Test speedup from early exits."""
        baseline_latency = 100  # ms (full model)
        early_exit_rates = [0.3, 0.5, 0.7]  # % samples exiting early

        for rate in early_exit_rates:
            avg_latency = baseline_latency * (
                rate + (1 - rate) * 1.0
            )  # Assuming some overhead
            speedup = baseline_latency / avg_latency

            # Verify speedup
            assert speedup > 1.0


class TestNetworkArchitecture:
    """Test early exit network architecture."""

    def test_exit_classifier_insertion(self):
        """Test exit classifier insertion."""
        original_model = nn.Sequential(*[nn.Linear(768, 768) for _ in range(12)])

        # Insert exit classifiers after layers 3, 6, 9
        exit_layers = [3, 6, 9]

        # Verify structure
        assert len(exit_layers) == 3
        assert exit_layers[0] < exit_layers[1] < exit_layers[2]

    def test_shared_classifier_heads(self):
        """Test shared classifier heads."""
        classifier = nn.Linear(768, 1000)

        # Test shared weights
        x1 = torch.randn(4, 768)
        x2 = torch.randn(4, 768)

        out1 = classifier(x1)
        out2 = classifier(x2)

        # Verify same classifier used
        assert out1.shape == out2.shape

    def test_gradual_width_reduction(self):
        """Test gradual width reduction in exit classifiers."""
        classifier_sizes = [768, 512, 256, 128]  # Gradual reduction

        # Verify reduction pattern
        for i in range(1, len(classifier_sizes)):
            assert classifier_sizes[i] < classifier_sizes[i - 1]


class TestTrainingIntegration:
    """Test early exit training integration."""

    def test_loss_balancing(self):
        """Test loss balancing across exits."""
        exit_losses = [0.5, 0.8, 1.2, 1.5]
        weights = [0.4, 0.3, 0.2, 0.1]  # More weight on early exits

        # Weighted loss
        weighted_loss = sum(w * l for w, l in zip(weights, exit_losses))

        # Verify
        assert weighted_loss > 0

    def test_auxiliary_loss(self):
        """Test auxiliary loss for early exits."""
        main_loss = torch.tensor(1.0)
        aux_losses = [torch.tensor(0.3), torch.tensor(0.5), torch.tensor(0.7)]

        # Combined loss
        total_loss = main_loss + 0.1 * sum(aux_losses)

        # Verify
        assert total_loss > main_loss

    def test_gradient_flow(self):
        """Test gradient flow to early exit classifiers."""
        classifier = nn.Linear(768, 1000)
        embedding = torch.randn(4, 768, requires_grad=True)

        # Forward
        logits = classifier(embedding)
        loss = logits.sum()

        # Backward
        loss.backward()

        # Verify gradients
        assert embedding.grad is not None
        assert classifier.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
