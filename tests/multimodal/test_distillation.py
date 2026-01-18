
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Mock torch for distillation test too
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["transformers"] = MagicMock()

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from multimodal.distillation import DistillationEngine

class TestDistillation(unittest.TestCase):
    def test_distill_image_mock(self):
        """Test mock distillation returns expected structure"""
        engine = DistillationEngine(teacher_model="test-mock")
        # Mock file operations since we don't have real files
        with patch("builtins.open", MagicMock()):
            result = engine.distill_image("test.png")
        
        self.assertIn("teacher_response", result)
        self.assertEqual(result["model"], "test-mock")

if __name__ == '__main__':
    unittest.main()
