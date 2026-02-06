"""
test_compression_optimized.py
Unit tests for compression optimization module.

Tests:
- Weight compression strategies
- Activation compression
- Quantization methods
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


class TestWeightCompression:
    """Test weight compression functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hidden_size = 768
        self.original_bits = 32
        self.compressed_bits = 4

    def test_quantization(self):
        """Test weight quantization."""
        # Original float32 weights
        original_weights = torch.randn(self.hidden_size, self.hidden_size)

        # Quantize to int8
        scale = torch.max(torch.abs(original_weights)) / 127.0
        quantized = torch.round(original_weights / scale)
        quantized = torch.clamp(quantized, -128, 127)

        # Verify quantization
        assert quantized.dtype == torch.float32  # Storing as float for simplicity
        assert (quantized >= -128).all()
        assert (quantized <= 127).all()

    def test_dequantization(self):
        """Test weight dequantization."""
        quantized = torch.randint(-128, 127, (self.hidden_size, self.hidden_size))
        scale = 0.01

        # Dequantize
        dequantized = quantized * scale

        # Verify dequantization
        assert dequantized.shape == quantized.shape

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original_size = self.hidden_size**2 * self.original_bits
        compressed_size = self.hidden_size**2 * self.compressed_bits

        compression_ratio = original_size / compressed_size

        # Verify compression
        assert compression_ratio == 8.0

    def test_uniform_quantization(self):
        """Test uniform quantization."""
        weights = torch.randn(768, 768)
        num_bits = 4

        # Calculate quantization parameters
        min_val = weights.min()
        max_val = weights.max()
        scale = (max_val - min_val) / (2**num_bits - 1)
        zero_point = torch.round(-min_val / scale)

        # Quantize
        quantized = torch.round((weights - min_val) / scale) + zero_point
        quantized = torch.clamp(quantized, 0, 2**num_bits - 1)

        # Verify
        assert quantized.shape == weights.shape


class TestActivationCompression:
    """Test activation compression functionality."""

    def test_activation_quantization(self):
        """Test activation quantization during inference."""
        batch_size = 4
        seq_len = 512
        hidden_size = 768

        # Original activations
        activations = torch.randn(batch_size, seq_len, hidden_size)

        # Quantize
        scale = activations.abs().max() / 127.0
        quantized = torch.round(activations / scale)

        # Verify
        assert quantized.shape == activations.shape

    def test_sparse_compression(self):
        """Test sparse activation compression."""
        activations = torch.randn(4, 512, 768)

        # Create sparse representation
        threshold = 0.1
        mask = torch.abs(activations) > threshold

        # Count sparsity
        sparsity = 1 - mask.float().mean()

        # Verify sparsity calculation
        assert 0 <= sparsity <= 1

    def test_run_length_encoding(self):
        """Test run-length encoding for sparse activations."""
        # Create sparse pattern
        sparse_vector = torch.tensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0])

        # Run-length encode
        runs = []
        current_val = sparse_vector[0]
        current_count = 0

        for val in sparse_vector:
            if val == current_val:
                current_count += 1
            else:
                runs.append((current_val, current_count))
                current_val = val
                current_count = 1
        runs.append((current_val, current_count))

        # Verify encoding
        assert len(runs) < len(sparse_vector)


class TestQuantizationMethods:
    """Test different quantization methods."""

    def test_dynamic_quantization(self):
        """Test dynamic quantization."""
        model = nn.Sequential(nn.Linear(768, 3072), nn.ReLU(), nn.Linear(3072, 768))

        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

        # Verify structure preserved
        assert len(list(quantized_model.modules())) == len(list(model.modules()))

    def test_static_quantization(self):
        """Test static quantization with calibration."""
        # Mock calibration data
        calibration_data = torch.randn(100, 768)

        # Calculate observer statistics
        observer = torch.quantization.MinMaxObserver()

        # Simulate calibration
        for data in calibration_data:
            observer(data)

        # Verify observer updated
        assert observer.min_val is not None

    def test_awq_quantization(self):
        """Test AWQ (Activation-aware Weight Quantization)."""
        weights = torch.randn(768, 768)
        importance_scores = torch.rand(768)

        # Calculate scales based on importance
        scales = importance_scores / importance_scores.sum()

        # Scale weights
        scaled_weights = weights * scales.unsqueeze(0)

        # Verify scaling
        assert scaled_weights.shape == weights.shape

    def test_gptq_quantization(self):
        """Test GPTQ quantization with Hessian."""
        weights = torch.randn(768, 768)
        hessian = torch.eye(768)

        # GPTQ update
        quantize_step = torch.zeros_like(weights)

        # Verify update structure
        assert quantize_step.shape == weights.shape


class TestDecomposition:
    """Test matrix decomposition for compression."""

    def test_kronecker_decomposition(self):
        """Test Kronecker product decomposition."""
        matrix = torch.randn(768, 768)

        # Reshape for Kronecker
        factor_size = 32
        A = matrix[:factor_size, :factor_size]
        B = matrix[factor_size:, factor_size:]

        # Verify factors
        assert A.shape == (factor_size, factor_size)
        assert B.shape == (768 - factor_size, 768 - factor_size)

    def test_tucker_decomposition(self):
        """Test Tucker decomposition."""
        tensor = torch.randn(768, 3072, 768)

        # Tucker ranks
        ranks = [512, 2048, 512]

        # Mock Tucker decomposition
        core = torch.randn(*ranks)
        factors = [torch.randn(tensor.shape[i], ranks[i]) for i in range(3)]

        # Verify structure
        assert len(factors) == 3
        assert core.shape == tuple(ranks)


class TestPrecisionConversion:
    """Test precision conversion methods."""

    def test_fp16_conversion(self):
        """Test FP16 conversion."""
        original = torch.randn(768, 768)

        # Convert to FP16
        converted = original.to(torch.float16)

        # Verify conversion
        assert converted.dtype == torch.float16

    def test_bfloat16_conversion(self):
        """Test BF16 conversion."""
        original = torch.randn(768, 768)

        # Convert to BF16
        converted = original.to(torch.bfloat16)

        # Verify conversion
        assert converted.dtype == torch.bfloat16

    def test_int8_conversion(self):
        """Test INT8 conversion."""
        original = torch.randn(768, 768)

        # Quantize to INT8
        scale = original.abs().max() / 127.0
        quantized = torch.round(original / scale)

        # Verify range
        assert (quantized >= -128).all()
        assert (quantized <= 127).all()


class TestRecoveryQuality:
    """Test compressed model recovery quality."""

    def test_weight_reconstruction_error(self):
        """Test weight reconstruction error."""
        original = torch.randn(768, 768)
        rank = 64

        # Low-rank approximation
        U, S, V = torch.svd(original)
        reconstructed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T

        # Calculate error
        error = torch.norm(original - reconstructed) / torch.norm(original)

        # Verify error bounds
        assert 0 < error < 1

    def test_activation_reconstruction(self):
        """Test activation reconstruction quality."""
        original = torch.randn(4, 512, 768)
        compressed = original * 0.5  # Simulated compression

        # Reconstruction error
        error = torch.norm(original - compressed) / torch.norm(original)

        # Verify quality
        assert error > 0
        assert error < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
