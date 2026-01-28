import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.nexus_final.utils.memory import estimate_model_vram_gb, should_use_sli

class MockConfig:
    def __init__(self, hidden_size=None, num_hidden_layers=None, d_model=None, num_layers=None):
        if hidden_size: self.hidden_size = hidden_size
        if num_hidden_layers: self.num_hidden_layers = num_hidden_layers
        if d_model: self.d_model = d_model
        if num_layers: self.num_layers = num_layers

class TestMemoryTrigger(unittest.TestCase):
    def test_estimate_vram_7b(self):
        # Qwen/Llama 7B (approx)
        # config: hidden=4096, layers=32
        config = MockConfig(hidden_size=4096, num_hidden_layers=32)
        
        # 4096^2 * 32 * 12 params ~= 6.4B params
        # 4-bit = 0.5 bytes per param -> 3.2GB
        # +20% overhead -> ~3.84GB
        
        gb = estimate_model_vram_gb(config, bits=4)
        # Rough check logic: 6.7B parameters * 0.5 bytes = 3.35 GB + overhead
        print(f"Est 7B (4-bit): {gb:.2f} GB")
        self.assertTrue(3.0 < gb < 5.0)

    def test_estimate_vram_70b(self):
        # Llama-3 70B (approx) - actually hidden is usually much larger than 4096, 
        # typically 8192 for 70B
        config = MockConfig(hidden_size=8192, num_hidden_layers=80) 
        
        # 8192^2 * 80 * 12 ~= 64B params
        # 4-bit -> 32GB
        # +20% -> 38.4GB
        
        gb = estimate_model_vram_gb(config, bits=4)
        print(f"Est 70B (4-bit): {gb:.2f} GB")
        self.assertTrue(35.0 < gb < 45.0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    def test_should_use_sli_trigger(self, mock_mem, mock_avail):
        # Scenario: 16GB VRAM card
        mock_mem.return_value = (16 * 1024**3, 16 * 1024**3) # Free, Total
        
        # Case 1: Small model (7B) -> ~4GB VRAM needed.
        # Should NOT use SLI (4GB < 16GB * 0.8)
        config_small = MockConfig(hidden_size=4096, num_hidden_layers=32)
        self.assertFalse(should_use_sli(config_small))
        
        # Case 2: Massive model (70B) -> ~40GB VRAM needed.
        # Should USE SLI (40GB > 16GB * 0.8)
        config_large = MockConfig(hidden_size=8192, num_hidden_layers=80)
        self.assertTrue(should_use_sli(config_large))

    @patch("torch.cuda.is_available", return_value=False)
    def test_should_use_sli_cpu(self, mock_avail):
        # If no CUDA, defaults to False (CPU path handles it or fails differently)
        # Or maybe SLI is useful for RAM too? 
        # Current logic says "if not cuda: return False"
        config = MockConfig(hidden_size=8192, num_hidden_layers=80)
        self.assertFalse(should_use_sli(config))

if __name__ == "__main__":
    unittest.main()
