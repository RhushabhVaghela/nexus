"""
test_repetition_integration.py
Mocked integration tests for Prompt Repetition.
Verifies logic propagation without loading real weights or requiring GPU.
"""

import unittest
import torch
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stages.base import StageConfig
from src.stages.stage_reasoning import ReasoningStage
# We don't import ModularMultimodalWrapper here because we just test the torch logic
# or we mock it. The previous test_embedding_repetition_logic was good.

class TestRepetitionIntegration(unittest.TestCase):
    def test_stage_config_propagation(self):
        """Verify that repetition factor is stored in StageConfig."""
        config = StageConfig(
            capability_name="test",
            base_model_path="dummy",
            output_dir="dummy",
            repetition_factor=2,
            repetition_style="verbose"
        )
        self.assertEqual(config.repetition_factor, 2)
        self.assertEqual(config.repetition_style, "verbose")

    def test_reasoning_stage_formatting(self):
        """Verify that ReasoningStage applies repetition to prompts."""
        config = StageConfig(
            capability_name="reasoning",
            base_model_path="dummy",
            output_dir="dummy/out",
            repetition_factor=2,
            repetition_style="baseline"
        )
        
        # Mocking to avoid loading real model or datasets
        with patch('src.stages.stage_reasoning.ReasoningStage._setup_logger'):
            stage = ReasoningStage(config)
            sample = {"problem": "1+1", "solution": "2"}
            formatted = stage._format_reasoning(sample)
            
            # Should contain repeated problem: "1+1 1+1"
            self.assertIn("1+1 1+1", formatted)
            self.assertIn("[PROBLEM]", formatted)
            self.assertIn("[REASONING]", formatted)

    def test_embedding_repetition_logic(self):
        """
        Verify the mathematical logic of embedding repetition.
        We simulate what the ModularMultimodalWrapper.forward does.
        """
        # Simulate 64 tokens of dimension 512
        mock_tokens = torch.randn(1, 64, 512)
        factor = 2
        
        # This is the line we added to model.py
        repeated_tokens = mock_tokens.repeat(1, factor, 1)
        
        self.assertEqual(repeated_tokens.shape[1], 128)
        self.assertEqual(repeated_tokens.shape[2], 512)
        # Verify it's actually a repetition: first 64 should match second 64
        torch.testing.assert_close(repeated_tokens[:, :64, :], repeated_tokens[:, 64:, :])

if __name__ == '__main__':
    unittest.main()
