import unittest
import unittest.mock as mock
from pathlib import Path
import sys
import tempfile
import shutil
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.omni.sequential_pipeline import run_sequential_training

class TestSequentialPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.base_model = self.tmp_dir / "base_model"
        self.base_model.mkdir()
        self.output_dir = self.tmp_dir / "output"
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @mock.patch("subprocess.Popen")
    @mock.patch("src.capability_registry.CapabilityRegistry")
    def test_pipeline_flow(self, mock_registry_cls, mock_popen):
        # Setup mock registry
        mock_registry = mock_registry_cls.return_value
        mock_cap = mock.Mock()
        mock_cap.training_script = "dummy_script.py"
        mock_registry.get.return_value = mock_cap
        
        # Setup mock popen
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = iter(["Step 1...", "Step 2...", "Done"])
        mock_popen.return_value = mock_proc
        
        # Run pipeline
        stages = ["cot", "tools"]
        
        # We need to ensure that the "final_model_*" discovery works in the test
        # We'll mock the internal Path.glob or just let it fall back to stage_output
        
        run_sequential_training(
            str(self.base_model),
            stages,
            str(self.output_dir),
            sample_size=10
        )
        
        # Verify popen was called twice
        self.assertEqual(mock_popen.call_count, 2)
        
        # Verify first call used base_model
        first_call_args = mock_popen.call_args_list[0][0][0]
        self.assertIn(str(self.base_model), first_call_args)
        
        # Verify second call used the output of first stage as base
        second_call_args = mock_popen.call_args_list[1][0][0]
        # Check if the stage_1_cot directory path is passed as the base model
        found = any("stage_1_cot" in arg for arg in second_call_args)
        self.assertTrue(found, f"Expected stage_1_cot in {second_call_args}")

if __name__ == "__main__":
    unittest.main()
