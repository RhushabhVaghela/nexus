
import unittest
import sys
import importlib.util
from unittest.mock import MagicMock
from pathlib import Path

# Mock DL Libs BEFORE import
sys.modules["unsloth"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["wandb"] = MagicMock()

# Mock torch completely
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch

# Fix for vram_gb calculation
# Code: vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
mock_props = MagicMock()
mock_props.total_memory = 24 * (1024**3) # 24 GB
mock_torch.cuda.get_device_properties.return_value = mock_props
mock_torch.cuda.is_available.return_value = True

def load_module(name, relative_path):
    root = Path(__file__).parent.parent / "src"
    path = root / relative_path
    if not path.exists():
        raise ImportError(f"Cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load 10_sft
sft_mod = load_module("sft_10", "10_sft_training.py")

class TestTrainingLogic(unittest.TestCase):

    def test_format_sample(self):
        """Test format_sample logic in 10_sft_training.py"""
        sample = {
            "domain": "fullstack",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ]
        }
        res = sft_mod.format_sample(sample)
        msgs = res["text"]
        
        # Check System Prompt Injection
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Fullstack", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")

    def test_format_sample_existing_system(self):
        """Test that existing system prompt comes first."""
        sample = {
            "domain": "fullstack",
            "messages": [
                {"role": "system", "content": "Existing"},
                {"role": "user", "content": "hi"}
            ]
        }
        res = sft_mod.format_sample(sample)
        msgs = res["text"]
        
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "Existing")
        self.assertEqual(len(msgs), 2)

if __name__ == '__main__':
    unittest.main()
