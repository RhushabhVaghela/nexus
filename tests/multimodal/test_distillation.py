
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Safe patching
from unittest.mock import patch

from multimodal.distillation import DistillationEngine

class TestDistillation(unittest.TestCase):
    def test_distill_image_mock(self):
        """Test mock distillation returns expected structure"""
        engine = DistillationEngine(teacher_model="test-mock-teacher", student_model="test-mock-student")
        # Mock file operations since we don't have real files
        with patch("builtins.open", MagicMock()):
            result = engine.distill_image("test.png")
        
        self.assertIn("teacher_response", result)
        self.assertEqual(result["model"], "test-mock-teacher")

if __name__ == '__main__':
    unittest.main()
