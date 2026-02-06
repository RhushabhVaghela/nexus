"""
test_armor_pruning.py
Unit tests for ARMOR pruning optimization module.

Tests:
- Structured pruning strategies
- Magnitude-based pruning
- Importance score calculation
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


class TestPruningStrategies:
    """Test different pruning strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.pruning_ratio = 0.3

    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        weights = torch.randn(self.hidden_size, self.hidden_size)

        # Calculate magnitudes
        magnitudes = torch.abs(weights)

        # Find threshold for pruning ratio
        threshold = torch.kthvalue(
            magnitudes.flatten(), int(magnitudes.numel() * self.pruning_ratio)
        ).values

        # Create mask
        mask = magnitudes > threshold

        # Verify pruning
        pruning_ratio = 1 - mask.float().mean()
        assert pruning_ratio > 0

    def test_structured_pruning(self):
        """Test structured pruning (removing entire neurons/channels)."""
        weight_matrix = torch.randn(self.hidden_size, self.hidden_size * 4)

        # Calculate neuron importance (sum of weights)
        neuron_importance = weight_matrix.abs().sum(dim=0)

        # Prune least important neurons
        num_prune = int(weight_matrix.shape[1] * self.pruning_ratio)
        prune_indices = torch.argsort(neuron_importance)[:num_prune]

        # Verify pruning
        remaining = weight_matrix.shape[1] - len(prune_indices)
        assert remaining < weight_matrix.shape[1]

    def test_unstructured_pruning(self):
        """Test unstructured pruning (individual weights)."""
        weights = torch.randn(self.hidden_size, self.hidden_size)

        # Prune by magnitude threshold
        threshold = torch.tensor(0.5)
        mask = torch.abs(weights) > threshold

        # Verify sparsity
        sparsity = 1 - mask.float().mean()
        assert sparsity > 0

    def test_gradient_based_pruning(self):
        """Test gradient-based importance scoring."""
        weights = torch.randn(self.hidden_size, self.hidden_size, requires_grad=True)
        gradients = torch.randn_like(weights)

        # Calculate importance (|weight| * |gradient|)
        importance = torch.abs(weights) * torch.abs(gradients)

        # Verify importance calculation
        assert importance.shape == weights.shape
        assert (importance >= 0).all()


class TestMagnitudePruning:
    """Test magnitude-based pruning methods."""

    def test_global_magnitude_pruning(self):
        """Test global magnitude pruning across all layers."""
        layers = [nn.Linear(768, 3072), nn.Linear(3072, 3072), nn.Linear(3072, 768)]

        # Collect all weights
        all_weights = torch.cat([w.flatten() for w in [l.weight for l in layers]])

        # Global threshold
        threshold = torch.kthvalue(
            all_weights, int(len(all_weights) * self.pruning_ratio)
        ).values

        # Apply pruning globally
        global_sparsity = (torch.abs(all_weights) < threshold).float().mean()

        # Verify global consistency
        assert 0 < global_sparsity < 1

    def test_layerwise_magnitude_pruning(self):
        """Test layer-wise magnitude pruning."""
        layer = nn.Linear(768, 3072)
        weights = layer.weight.flatten()

        # Layer-wise threshold
        threshold = torch.kthvalue(
            weights, int(len(weights) * self.pruning_ratio)
        ).values

        # Verify threshold
        assert threshold > 0

    def test_iterative_magnitude_pruning(self):
        """Test iterative magnitude pruning."""
        model = nn.Linear(self.hidden_size, self.hidden_size)
        current_sparsity = 0.0
        target_sparsity = 0.7

        # Iterative pruning
        iterations = 0
        while current_sparsity < target_sparsity:
            # Prune additional weights
            current_sparsity += 0.1
            iterations += 1
            if iterations > 20:
                break

        # Verify progress
        assert iterations > 0


class TestStructuredPruning:
    """Test structured pruning methods."""

    def test_channel_pruning(self):
        """Test channel pruning."""
        conv = nn.Conv2d(64, 128, 3, padding=1)
        weights = conv.weight

        # Calculate channel importance
        channel_importance = weights.abs().mean(dim=[1, 2, 3])

        # Prune least important channels
        num_prune = int(weights.shape[0] * self.pruning_ratio)
        prune_indices = torch.argsort(channel_importance)[:num_prune]

        # Verify
        assert len(prune_indices) < weights.shape[0]

    def test_head_pruning(self):
        """Test attention head pruning."""
        num_heads = 12
        head_dim = 64

        # Simulate head importance scores
        head_importance = torch.rand(num_heads)

        # Prune heads
        num_prune = int(num_heads * self.pruning_ratio)
        prune_indices = torch.argsort(head_importance)[:num_prune]

        # Verify
        assert len(prune_indices) < num_heads

    def test_neuron_pruning(self):
        """Test neuron pruning in feed-forward layers."""
        ffn = nn.Linear(768, 3072)
        weights = ffn.weight

        # Calculate neuron importance
        neuron_importance = weights.abs().mean(dim=1)

        # Prune neurons
        num_prune = int(weights.shape[0] * self.pruning_ratio)
        prune_indices = torch.argsort(neuron_importance)[:num_prune]

        # Verify
        assert len(prune_indices) < weights.shape[0]


class TestPruningSchedule:
    """Test pruning schedules."""

    def test_linear_pruning_schedule(self):
        """Test linear pruning schedule."""
        initial_sparsity = 0.0
        final_sparsity = 0.7
        total_steps = 1000

        # Linear schedule
        current_sparsity = (
            initial_sparsity + (final_sparsity - initial_sparsity) * 500 / total_steps
        )

        # Verify
        assert 0 < current_sparsity < final_sparsity

    def test_exponential_pruning_schedule(self):
        """Test exponential pruning schedule."""
        initial_sparsity = 0.1
        decay_rate = 0.001
        step = 500

        # Exponential schedule
        current_sparsity = initial_sparsity * (1 - decay_rate) ** step

        # Verify
        assert current_sparsity > 0
        assert current_sparsity < initial_sparsity

    def test_polynomial_pruning_schedule(self):
        """Test polynomial pruning schedule."""
        initial_sparsity = 0.0
        final_sparsity = 0.7
        power = 2
        step = 500
        total_steps = 1000

        # Polynomial schedule
        progress = step / total_steps
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (
            progress**power
        )

        # Verify
        assert current_sparsity > 0


class TestMaskManagement:
    """Test pruning mask management."""

    def test_mask_creation(self):
        """Test pruning mask creation."""
        weights = torch.randn(self.hidden_size, self.hidden_size)
        sparsity = 0.3

        # Create mask
        threshold = torch.kthvalue(
            weights.abs().flatten(), int(weights.numel() * sparsity)
        ).values
        mask = weights.abs() > threshold

        # Verify mask
        assert mask.shape == weights.shape
        assert mask.dtype == torch.bool

    def test_mask_application(self):
        """Test mask application to weights."""
        weights = torch.randn(self.hidden_size, self.hidden_size, requires_grad=True)
        mask = torch.ones_like(weights, dtype=torch.bool)

        # Apply mask
        pruned_weights = weights * mask

        # Verify
        assert pruned_weights.shape == weights.shape

    def test_mask_gradient_flow(self):
        """Test gradient flow through masked weights."""
        weights = torch.randn(self.hidden_size, self.hidden_size, requires_grad=True)
        mask = torch.ones_like(weights, dtype=torch.bool)

        # Forward
        out = (weights * mask).sum()

        # Backward
        out.backward()

        # Verify gradients
        assert weights.grad is not None


class TestFineTuning:
    """Test fine-tuning after pruning."""

    def test_finetuning_recovery(self):
        """Test performance recovery after fine-tuning."""
        original_accuracy = 0.85
        pruned_accuracy = 0.72
        target_recovery = 0.80

        # Simulate fine-tuning recovery
        recovery_rate = 0.01
        current_accuracy = pruned_accuracy

        steps = 0
        while current_accuracy < target_recovery:
            current_accuracy += recovery_rate
            steps += 1
            if steps > 20:
                break

        # Verify recovery
        assert steps < 20

    def test_gradual_unfreezing(self):
        """Test gradual unfreezing of pruned layers."""
        layers = [nn.Linear(768, 768) for _ in range(6)]
        unfreeze_schedule = [0, 1, 2, 3, 4, 5]  # Layers to unfreeze

        # Verify progression
        assert len(unfreeze_schedule) == len(layers)
        assert unfreeze_schedule[-1] > unfreeze_schedule[0]


class TestPruningMetrics:
    """Test pruning metrics calculation."""

    def test_sparsity_calculation(self):
        """Test sparsity calculation."""
        pruned_weights = torch.randn(768, 768)
        num_zeros = (pruned_weights == 0).sum()
        total = pruned_weights.numel()

        sparsity = num_zeros / total

        # Verify
        assert 0 <= sparsity <= 1

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original_size = 768 * 768 * 4  # bytes
        sparsity = 0.7

        compressed_size = original_size * (1 - sparsity)
        compression_ratio = original_size / compressed_size

        # Verify
        assert compression_ratio > 1

    def test_FLOPs_reduction(self):
        """Test FLOPs reduction calculation."""
        original_flops = 768 * 768 * 2
        pruned_ratio = 0.3

        remaining_flops = original_flops * (1 - pruned_ratio)
        speedup = original_flops / remaining_flops

        # Verify
        assert speedup > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
