import unittest
import torch
import os
import sys
import shutil
import tempfile
from unittest.mock import MagicMock

# --- Integration Test for the Mini Pipeline ---
# Flow:
# 1. Config Load
# 2. Registry Load (Mock)
# 3. Profiler (Mock Output)
# 4. Architect (Synthesis)
# 5. Training Loop (Mock Step)

class TestPipelineMini(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_src = os.path.join(self.test_dir, "nexus_student_int.py")
        
        # Mock Profile Data
        self.profile_path = os.path.join(self.test_dir, "profile.json")
        with open(self.profile_path, 'w') as f:
            f.write('{"teacher_A": {"intrinsic_dimension": 32}}')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_flow(self):
        # 1. Architect Synthesis
        from src.nexus_final.architect import NeuralArchitect
        
        architect = NeuralArchitect()
        
        # Mock Config from Profiler Output
        config = architect.determine_adapter_config(
            "teacher_A", 
            architect.load_profiling_data(self.profile_path),
            max_rank_limit=64
        )
        
        # Synthesize
        architect.synthesize_student_model(
            self.output_src,
            "gpt2",
            config
        )
        
        # 2. Import Synthesized Model
        sys.path.append(self.test_dir)
        import nexus_student_int as ns
        
        model = ns.build_student()
        self.assertIsNotNone(model)
        
        # 3. Mock Training Step (Distill)
        # We verify that the model accepts input and produces output
        # mimicking the behaviour inside `distill.py`
        
        input_ids = torch.randint(0, 100, (1, 10))
        output = model(input_ids)
        
        # Check output structure
        self.assertTrue(hasattr(output, 'logits'))
        self.assertEqual(output.logits.shape, (1, 10, 50257)) # GPT2 vocab size

if __name__ == '__main__':
    unittest.main()
