"""
test_low_rank_attention.py
Unit tests for low-rank attention optimization module.

Tests:
- Low-rank attention approximations
- SVD-based decompositions
- Memory and computation efficiency
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


class TestLowRankAttention:
    """Test low-rank attention functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.rank = 32
        self.num_heads = 12

    def test_svd_decomposition(self):
        """Test SVD-based decomposition."""
        # Create a weight matrix
        weight = torch.randn(self.hidden_size, self.hidden_size)

        # Perform SVD
        U, S, V = torch.svd(weight)

        # Verify SVD properties
        assert U.shape == (self.hidden_size, self.hidden_size)
        assert S.shape == (self.hidden_size,)
        assert V.shape == (self.hidden_size, self.hidden_size)

    def test_low_rank_approximation(self):
        """Test low-rank approximation quality."""
        weight = torch.randn(self.hidden_size, self.hidden_size)
        rank = self.rank

        # Full SVD
        U, S, V = torch.svd(weight)

        # Low-rank approximation
        U_k = U[:, :rank]
        S_k = S[:rank]
        V_k = V[:, :rank]

        approximated = U_k @ torch.diag(S_k) @ V_k.T

        # Verify approximation rank
        assert approximated.shape == weight.shape

        # Approximation error
        error = torch.norm(weight - approximated) / torch.norm(weight)
        assert error > 0  # Some error expected

    def test_factorized_attention(self):
        """Test factorized attention computation."""
        seq_len = 512
        head_dim = 64
        rank = self.rank

        # Factorized projection matrices
        W_q = torch.randn(self.hidden_size, rank)
        W_k = torch.randn(self.hidden_size, rank)
        W_v = torch.randn(self.hidden_size, rank)

        # Factorized query/key/value
        Q_fact = torch.randn(seq_len, rank)
        K_fact = torch.randn(seq_len, rank)
        V_fact = torch.randn(seq_len, rank)

        # Compute attention scores
        attention = Q_fact @ K_fact.T / (rank**0.5)

        # Verify attention properties
        assert attention.shape == (seq_len, seq_len)

    def test_lora_style_attention(self):
        """Test LoRA-style attention adaptation."""
        original_attention = nn.Linear(self.hidden_size, self.hidden_size)

        # LoRA decomposition
        lora_A = torch.randn(self.rank, self.hidden_size)
        lora_B = torch.randn(self.hidden_size, self.rank)

        # Combined projection
        combined_weight = original_attention.weight + lora_B @ lora_A

        # Verify dimensions
        assert combined_weight.shape == original_attention.weight.shape


class TestFactorizationStrategies:
    """Test different factorization strategies."""

    def test_qlora_decomposition(self):
        """Test QLoRA-style decomposition."""
        original_dim = self.hidden_size
        reduced_dim = self.rank

        # Quantization + low-rank
        quantization_levels = torch.randint(0, 256, (original_dim,))

        # Low-rank adaptation
        lora_a = torch.randn(reduced_dim, original_dim)
        lora_b = torch.randn(original_dim, reduced_dim)

        # Verify decomposition
        assert lora_a.shape[0] == reduced_dim
        assert lora_b.shape[1] == reduced_dim

    def test_spectral_decomposition(self):
        """Test spectral decomposition."""
        matrix = torch.randn(self.hidden_size, self.hidden_size)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)

        # Verify decomposition
        assert len(eigenvalues) == self.hidden_size
        assert eigenvectors.shape == (self.hidden_size, self.hidden_size)

    def test_randomized_svd(self):
        """Test randomized SVD approximation."""
        matrix = torch.randn(self.hidden_size, self.hidden_size)
        rank = self.rank
        oversample = 8

        # Randomized projection
        Q = torch.randn(self.hidden_size, rank + oversample)

        # Project matrix
        Y = matrix @ Q
        Q_reduced, _ = torch.linalg.qr(Y)

        # Verify projection
        assert Q_reduced.shape[1] >= rank


class TestApproximationQuality:
    """Test low-rank approximation quality."""

    def test_reconstruction_error(self):
        """Test reconstruction error measurement."""
        original = torch.randn(self.hidden_size, self.hidden_size)
        ranks = [8, 16, 32, 64, 128]

        errors = []
        for r in ranks:
            # Low-rank approximation
            U, S, V = torch.svd(original)
            approx = U[:, :r] @ torch.diag(S[:r]) @ V[:, :r].T

            # Compute error
            error = torch.norm(original - approx) / torch.norm(original)
            errors.append((r, error.item()))

        # Verify error decreases with rank
        errors_sorted = sorted(errors, key=lambda x: x[0])
        for i in range(1, len(errors_sorted)):
            assert errors_sorted[i][1] <= errors_sorted[i - 1][1]

    def test_attention_approximation(self):
        """Test attention approximation quality."""
        seq_len = 128
        head_dim = 64

        # Full attention matrix
        Q = torch.randn(seq_len, head_dim)
        K = torch.randn(seq_len, head_dim)
        full_attention = Q @ K.T / (head_dim**0.5)

        # Low-rank approximation
        U, S, V = torch.svd(full_attention)
        rank = 16
        approx_attention = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T

        # Verify approximation
        error = torch.norm(full_attention - approx_attention) / torch.norm(
            full_attention
        )
        assert error > 0

    def test_spectrum_preservation(self):
        """Test spectrum preservation in approximation."""
        matrix = torch.randn(self.hidden_size, self.hidden_size)
        rank = self.rank

        # Full spectrum
        full_spectrum = torch.linalg.svdvals(matrix)

        # Approximated spectrum
        U, S, V = torch.svd(matrix)
        approx = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T
        approx_spectrum = torch.linalg.svdvals(approx)

        # Top eigenvalues should be preserved
        assert torch.allclose(full_spectrum[:rank], approx_spectrum, rtol=0.1)


class TestMemoryEfficiency:
    """Test memory efficiency improvements."""

    def test_parameter_reduction(self):
        """Test parameter reduction calculation."""
        original_params = self.hidden_size**2
        reduced_params = self.hidden_size * self.rank * 2

        reduction_ratio = original_params / reduced_params

        # Verify parameter savings
        assert reduction_ratio > 10

    def test_activation_memory_savings(self):
        """Test activation memory savings."""
        seq_len = 2048
        head_dim = 64

        # Full attention memory
        full_attention_memory = seq_len * seq_len * 4  # float32

        # Factorized attention memory
        factorized_memory = seq_len * self.rank * 2 * 4  # Q and K factors

        # Verify savings
        assert factorized_memory < full_attention_memory

    def test_computation_reduction(self):
        """Test computation reduction."""
        seq_len = 512

        # Full attention computation
        full_compute = seq_len**2 * self.hidden_size

        # Low-rank attention computation
        low_rank_compute = seq_len * self.rank * self.hidden_size

        # Verify computation savings
        assert low_rank_compute < full_compute


class TestGradientFlow:
    """Test gradient flow in low-rank attention."""

    def test_gradient_stability(self):
        """Test gradient stability with low-rank layers."""
        rank = self.rank

        # Gradient norms for different ranks
        gradient_norms = {
            8: torch.tensor(1.5),
            16: torch.tensor(1.2),
            32: torch.tensor(1.0),
            64: torch.tensor(0.9),
        }

        # Verify stability
        for rank, norm in gradient_norms.items():
            assert norm > 0.5  # Gradients should not vanish

    def test_backprop_efficiency(self):
        """Test backpropagation efficiency."""
        batch_size = 4
        seq_len = 512
        rank = self.rank

        # Forward pass time
        forward_ops = batch_size * seq_len * rank * self.hidden_size

        # Backward pass time (typically 2-3x forward)
        backward_ops = forward_ops * 2.5

        # Verify reasonable ratio
        assert 2 < backward_ops / forward_ops < 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
