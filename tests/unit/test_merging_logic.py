
import torch
import unittest
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.omni.unify_checkpoints import merge_checkpoints

class TestMergingLogic(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.base_path = self.tmp_dir / "base"
        self.cp1_path = self.tmp_dir / "cp1"
        self.cp2_path = self.tmp_dir / "cp2"
        self.output_path = self.tmp_dir / "output"
        
        for p in [self.base_path, self.cp1_path, self.cp2_path]:
            p.mkdir()
            
        # Create dummy state dicts
        # We'll use a real GPT2 key
        self.shape = (4, 4) # GPT2 wte is usually larger, but for dummy it's fine if we mock config
        self.key = "transformer.wte.weight"
        self.base_weights = torch.ones(self.shape) * 1.0
        self.cp1_weights = torch.ones(self.shape) * 2.0  # Delta = +1.0
        self.cp2_weights = torch.ones(self.shape) * 0.0  # Delta = -1.0
        
        from transformers import GPT2Config, AutoModelForCausalLM
        config_dict = {
            "architectures": ["GPT2LMHeadModel"], 
            "model_type": "gpt2",
            "vocab_size": 4,
            "n_embd": 4,
            "n_layer": 0,
            "n_head": 1,
            "n_positions": 1024,
            "tie_word_embeddings": False
        }
        config = GPT2Config(**config_dict)
        model = AutoModelForCausalLM.from_config(config)
        
        # Keys that we want to vary
        v_key = "transformer.wte.weight"
        self.key = v_key
        
        # Save them as if they were real models
        from safetensors.torch import save_file
        
        def save_dummy(path, val):
            sd = {}
            for k, v in model.state_dict().items():
                if k == v_key:
                    sd[k] = torch.ones(v.shape) * val
                else:
                    sd[k] = torch.zeros(v.shape)
            save_file(sd, f"{path}/model.safetensors")
            with open(f"{path}/config.json", "w") as f:
                import json
                json.dump(config_dict, f)

        self.save_dummy = save_dummy
        save_dummy(self.base_path, 1.0)
        save_dummy(self.cp1_path, 2.0)
        save_dummy(self.cp2_path, 0.0)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_linear_merge(self):
        """Test simple averaging."""
        merge_checkpoints(
            str(self.base_path),
            [str(self.cp1_path), str(self.cp2_path)],
            str(self.output_path),
            method="linear"
        )
        
        from safetensors.torch import load_file
        merged = load_file(str(self.output_path / "model.safetensors"))
        actual = merged[self.key].to(torch.float32)
        expected = torch.ones(self.shape) * 1.0
        self.assertTrue(torch.allclose(actual, expected, atol=1e-1))

    def test_task_arithmetic(self):
        """Test adding deltas to base."""
        merge_checkpoints(
            str(self.base_path),
            [str(self.cp1_path), str(self.cp2_path)],
            str(self.output_path),
            method="task_arithmetic"
        )
        
        from safetensors.torch import load_file
        merged = load_file(str(self.output_path / "model.safetensors"))
        actual = merged[self.key].to(torch.float32)
        expected = torch.ones(self.shape) * 1.0
        self.assertTrue(torch.allclose(actual, expected, atol=1e-1))

    def test_ties_merging(self):
        """Test TIES merging logic."""
        # Custom setup for TIES
        self.save_dummy(self.base_path, 0.0)
        self.save_dummy(self.cp1_path, 10.0)
        self.save_dummy(self.cp2_path, -1.0)
        
        merge_checkpoints(
            str(self.base_path),
            [str(self.cp1_path), str(self.cp2_path)],
            str(self.output_path),
            method="ties",
            density=1.0
        )
        
        from safetensors.torch import load_file
        merged = load_file(str(self.output_path / "model.safetensors"))
        actual = merged[self.key].to(torch.float32)
        # CP1 delta (10) * 0.5 = 5.0. CP2 delta (-1) discarded by sign selection.
        expected = torch.ones(self.shape) * 5.0
        self.assertTrue(torch.allclose(actual, expected, atol=1e-1))

if __name__ == "__main__":
    unittest.main()
