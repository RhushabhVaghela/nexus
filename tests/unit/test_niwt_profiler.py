import unittest
import shutil
import tempfile
import os

try:
    import torch
    import numpy as np
    from nexus.models.profiler import StreamingPCAProfiler

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "torch/numpy/sklearn not available")
class _Base(unittest.TestCase):
    """Marker base to skip all tests if deps missing."""

    pass


class TestNIWTProfiler(_Base):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_intrinsic_dimension_calculation(self):
        profiler = StreamingPCAProfiler("mock", [], self.test_dir)

        # Case 1: Variance decays quickly (Low Rank)
        # [0.8, 0.1, 0.05, 0.01, ...] -> 0.9 @ idx 1 (2 components)
        var_ratio = np.array([0.8, 0.15, 0.04, 0.01])
        dim = profiler._calculate_intrinsic_dimension(var_ratio, threshold=0.95)
        self.assertEqual(dim, 2)  # 0.8+0.15 = 0.95

        # Case 2: Variance spread out (High Rank)
        var_ratio_flat = np.array([0.1] * 10)  # Sum=1.0
        dim_flat = profiler._calculate_intrinsic_dimension(
            var_ratio_flat, threshold=0.95
        )
        self.assertEqual(dim_flat, 10)

    def test_perturbation_analysis_logic(self):
        """Test perturbation analysis with mocked model/tokenizer/dataset."""
        from unittest.mock import MagicMock, patch

        profiler = StreamingPCAProfiler("mock", ["layer.0", "layer.1"], self.test_dir)

        # Create mock model
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.named_modules.return_value = [
            ("layer.0", MagicMock()),
            ("layer.1", MagicMock()),
        ]

        # Mock forward pass returns an object with logits
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 100)
        mock_model.__call__ = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_output

        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        # Make tokenizer output movable to device
        tok_output = MagicMock()
        tok_output.to = MagicMock(return_value=tok_output)
        tok_output.__iter__ = MagicMock(return_value=iter(["input_ids"]))
        tok_output.__getitem__ = MagicMock(return_value=torch.randint(0, 1000, (1, 10)))
        tok_output.keys = MagicMock(return_value=["input_ids", "attention_mask"])
        mock_tokenizer.return_value = tok_output

        # Create mock dataset
        mock_dataset = ["Hello world", "Test sentence", "Another sample"]

        # Run perturbation analysis
        profiler._perturbation_analysis(
            mock_model, mock_tokenizer, mock_dataset, n_samples=2
        )

        # Verify results: critical scores should be populated for both layers
        self.assertIn("layer.0", profiler.critical_scores)
        self.assertIn("layer.1", profiler.critical_scores)

        # Scores should be non-negative floats
        for layer_name, score in profiler.critical_scores.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
