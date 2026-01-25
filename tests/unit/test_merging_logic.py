
import torch
import unittest
from pathlib import Path
import sys
import shutil
import tempfile
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestMergingLogic(unittest.TestCase):
    
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.cp1 = self.tmp_dir / "cp1"
        self.cp2 = self.tmp_dir / "cp2"
        self.cp1.mkdir()
        self.cp2.mkdir()
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_linear_merge(self):
        """Test simple averaging."""
        shape = (4, 4)
        key = "w"
        
        # We need to mock sys.modules["transformers"] effectively to avoid imports
        transformers_mock = MagicMock()
        
        # Setup base model mock
        base_model_mock = MagicMock()
        base_model_mock.state_dict.return_value = {key: torch.full(shape, 1.0)}
        transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = base_model_mock
        
        # Setup safetensors to return dicts so we hit the fast path
        mock_modules = {
            "safetensors": MagicMock(),
            "safetensors.torch": MagicMock(),
            "transformers": transformers_mock,
            "accelerate": MagicMock(), # Prevent accelerate imports
            "deepspeed": None # Simulate not installed
        }
        
        # Mock loading: base (not used via load_file for base, but for checkpoints)
        # Checkpoints: cp1, cp2
        # Logic: 
        # 1. Base loaded via AutoModel (we mocked state_dict above)
        # 2. Checkpoints loaded via load_file if file exists, else AutoModel
        
        # Let's ensure file exists check passes so we use load_file (easier to mock)
        with patch("pathlib.Path.exists", return_value=True):
             mock_modules["safetensors.torch"].load_file.side_effect = [
                {key: torch.full(shape, 2.0)}, # cp1
                {key: torch.full(shape, 0.0)}  # cp2
             ]
             
             with patch.dict(sys.modules, mock_modules):
                if "src.omni.unify_checkpoints" in sys.modules:
                    del sys.modules["src.omni.unify_checkpoints"]
                    
                from src.omni.unify_checkpoints import merge_checkpoints
                
                merge_checkpoints(
                    "base",
                    [str(self.cp1), str(self.cp2)],
                    "output",
                    method="linear"
                )
                
                # Check save_pretrained was called on base_model
                args, _ = base_model_mock.save_pretrained.call_args
                if args:
                    # verify logic: 
                    # merged = zeros + 2.0*0.5 + 0.0*0.5 = 1.0
                    # base_model.load_state_dict(merged) called.
                    # We can check load_state_dict arg
                    load_args = base_model_mock.load_state_dict.call_args[0][0]
                    self.assertTrue(torch.allclose(load_args[key].float(), torch.full(shape, 1.0)))

    def test_task_arithmetic(self):
        """Test adding deltas."""
        shape = (4, 4)
        key = "w"
        
        transformers_mock = MagicMock()
        base_model_mock = MagicMock()
        base_model_mock.state_dict.return_value = {key: torch.full(shape, 1.0)}
        transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = base_model_mock
        
        mock_modules = {
            "safetensors": MagicMock(),
            "safetensors.torch": MagicMock(),
            "transformers": transformers_mock,
            "accelerate": MagicMock(),
            "deepspeed": None
        }
        
        with patch("pathlib.Path.exists", return_value=True):
            mock_modules["safetensors.torch"].load_file.side_effect = [
                {key: torch.full(shape, 2.0)}, # cp1
                {key: torch.full(shape, 0.0)}  # cp2
            ]
            
            with patch.dict(sys.modules, mock_modules):
                if "src.omni.unify_checkpoints" in sys.modules:
                    del sys.modules["src.omni.unify_checkpoints"]
                    
                from src.omni.unify_checkpoints import merge_checkpoints
                
                merge_checkpoints(
                    "base",
                    [str(self.cp1), str(self.cp2)],
                    "output",
                    method="task_arithmetic"
                )
                
                # base(1.0) + (cp1(2.0)-base(1.0))*0.5 + (cp2(0.0)-base(1.0))*0.5
                # 1.0 + 1.0*0.5 + (-1.0)*0.5 = 1.0
                
                load_args = base_model_mock.load_state_dict.call_args[0][0]
                self.assertTrue(torch.allclose(load_args[key].float(), torch.full(shape, 1.0)))

if __name__ == "__main__":
    unittest.main()
