import unittest
import torch
import os
import shutil
import tempfile
from src.nexus_final.architect import NeuralArchitect

class TestNexusStudent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.test_dir, "nexus_student_test.py")
        self.architect = NeuralArchitect()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_student_synthesis(self):
        """Verify the architect generates valid Python code for the student."""
        config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        }
        
        self.architect.synthesize_student_model(
            self.output_path,
            "gpt2", # Small model for fast test
            config
        )
        
        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify content
        with open(self.output_path, 'r') as f:
            content = f.read()
            self.assertIn('class NexusStudent(nn.Module):', content)
            self.assertIn('"r": 16', content)

    def test_dynamic_rank_logic(self):
        """Verify the rank capping logic."""
        # Case 1: Low Intrinsic Dimension
        profile_low = {"teacher_A": {"intrinsic_dimension": 20}}
        conf_low = self.architect.determine_adapter_config("teacher_A", profile_low, max_rank_limit=128)
        self.assertEqual(conf_low['r'], 20)
        
        # Case 2: High Intrinsic Dimension (Capped)
        profile_high = {"teacher_B": {"intrinsic_dimension": 5000}}
        conf_high = self.architect.determine_adapter_config("teacher_B", profile_high, max_rank_limit=128)
        self.assertEqual(conf_high['r'], 128)
        
        # Case 3: Floor enforcement
        profile_tiny = {"teacher_C": {"intrinsic_dimension": 1}}
        conf_tiny = self.architect.determine_adapter_config("teacher_C", profile_tiny, max_rank_limit=128)
        self.assertEqual(conf_tiny['r'], 4) # Should floor to 4

if __name__ == '__main__':
    unittest.main()
