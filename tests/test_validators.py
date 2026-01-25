
import unittest
import sys
import importlib.util
from pathlib import Path

def load_module(name, relative_path):
    root = Path(__file__).parent.parent / "src"
    path = root / relative_path
    if not path.exists():
        raise ImportError(f"Cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_07 = load_module("mod_07", "07_validate_all_datasets.py")

class TestValidators(unittest.TestCase):

    def setUp(self):
        # Initialize CONFIG for the module
        mod_07.CONFIG = {
            "min_messages": 2,
            "max_messages": 50,
            "min_content_length": 50,
            "max_content_length": 100000,
        }
        self.validator = mod_07.DatasetValidator()

    def test_schema_valid(self):
        sample = {
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"}
            ]
        }
        self.assertTrue(self.validator.validate_schema(sample))

    def test_schema_invalid(self):
        sample = {"messages": []} # Too few
        self.assertFalse(self.validator.validate_schema(sample))
        self.assertEqual(self.validator.stats["too_few_messages"], 1)

    def test_content_valid(self):
        # Default min length 50
        sample = {
            "messages": [
                {"role": "user", "content": "A" * 30},
                {"role": "assistant", "content": "B" * 30}
            ]
        }
        self.assertTrue(self.validator.validate_content(sample))

if __name__ == '__main__':
    unittest.main()
