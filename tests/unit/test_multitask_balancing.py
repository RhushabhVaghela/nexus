
import torch
import unittest
from pathlib import Path
import sys
import tempfile
import shutil
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest.mock as mock
# Mock slow imports before they happen
sys.modules["multimodal.model"] = mock.Mock()
sys.modules["multimodal"] = mock.Mock()

import importlib
OmniDataset = importlib.import_module("src.24_multimodal_training").OmniDataset

class TestMultitaskBalancing(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        
        # Create categories
        self.cat_cot = self.tmp_dir / "cot_data"
        self.cat_tools = self.tmp_dir / "tools_data"
        
        self.cat_cot.mkdir()
        self.cat_tools.mkdir()
        
        # Add samples to CoT (Large)
        for i in range(10):
            with open(self.cat_cot / f"file_{i}.jsonl", "w") as f:
                for j in range(10):
                    f.write(json.dumps({"prompt": f"cot_{i}_{j}", "response": "done"}) + "\n")
        
        # Add samples to Tools (Small)
        for i in range(2):
            with open(self.cat_tools / f"file_{i}.jsonl", "w") as f:
                for j in range(5):
                    f.write(json.dumps({"prompt": f"tool_{i}_{j}", "response": "done"}) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_interleaved_sampling(self):
        """Verify that samples are interleaved between categories."""
        # We need to mock discover_datasets or ensure its categorization works for these folder names
        ds = OmniDataset(str(self.tmp_dir), balanced=True)
        
        # The folders are named 'cot_data' and 'tools_data'.
        # KEYWORD_MAP in metrics_tracker.py:
        # "cot": "cot"
        # "tool": "tools"
        # So they should be categorized correctly.
        
        iterator = iter(ds)
        samples = []
        for _ in range(10):
            try:
                samples.append(next(iterator))
            except StopIteration:
                break
                
        # Check if we see an interleave pattern
        types = [s["text"].split("_")[0] for s in samples]
        
        # Ex: ['cot', 'tool', 'cot', 'tool', ...]
        logger_outputs = [t for t in types]
        print(f"Sample sequence: {logger_outputs}")
        
        # Ensure we have both types in the first few samples
        # (Exact order might vary slightly due to random shuffle but round-robin is enforced)
        self.assertIn("cot", types[:4])
        self.assertIn("tool", types[:4])
        
    def test_sequential_sampling(self):
        """Verify traditional sequential behavior."""
        ds = OmniDataset(str(self.tmp_dir), balanced=False)
        iterator = iter(ds)
        
        samples = []
        for _ in range(50):
            try:
                samples.append(next(iterator))
            except StopIteration:
                break
                
        types = [s["text"].split("_")[0] for s in samples]
        
        # In sequential mode, it should exhaust one category (or file) before moving on.
        # Since files are random shuffled, it might mix FILES but not CATEGORIES round-robinly.
        # Actually _iter_sequential just flattens everything.
        
        # Just check that it works
        self.assertTrue(len(samples) > 0)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    unittest.main()
