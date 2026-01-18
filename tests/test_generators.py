
import unittest
import sys
import importlib.util
from pathlib import Path

# Load Modules using importlib since names have numbers
def load_module(name, relative_path):
    root = Path(__file__).parent.parent / "src"
    path = root / relative_path
    if not path.exists():
        raise ImportError(f"Cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Helper for mocking sys.path inside modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Load Generators
mod_05 = load_module("mod_05", "05_generate_repetitive_dataset.py")
mod_06 = load_module("mod_06", "06_generate_preference_dataset.py")

class TestGenerators(unittest.TestCase):

    def test_05_repetitive_engine(self):
        """Test PromptRepetitionEngine from 05_generate_repetitive_dataset.py"""
        engine = mod_05.PromptRepetitionEngine()
        # Mocking the generator call
        engine.category_counters = {k: 0 for k in mod_05.GENERATOR_WEIGHTS.keys()}
        sample = engine.generate_trajectory()
        
        if sample:
            self.assertIn("messages", sample)
            self.assertIn("category", sample)
            self.assertEqual(sample["domain"], "factual_knowledge")

    def test_06_preference_engine(self):
        """Test PreferencePairEngine from 06_generate_preference_dataset.py"""
        engine = mod_06.PreferencePairEngine()
        sample = engine.generate_preference_pair()
        
        if sample:
            self.assertIn("prompt", sample)
            self.assertIn("chosen", sample)
            self.assertIn("rejected", sample)
            self.assertIn("training_mode", sample)

if __name__ == '__main__':
    unittest.main()
