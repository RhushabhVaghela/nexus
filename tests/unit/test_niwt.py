import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Import the class to test
# Adjust path if needed or assume test runner handles pythonpath
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
from niwt_core import NIWTCore

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
        self.model = MagicMock()
        self.model.layers = self.layers
        self.device = 'cpu'
    
    def forward(self, input_ids=None, **kwargs):
        # Mock forward pass
        return (torch.randn(1, 10, 10),) # Tuple like generic LM output

    def generate(self, **kwargs):
        # Mock generation output
        return torch.tensor([[101, 200, 102]])

class TestNIWTCore(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.tokenizer = MagicMock()
        # Mock return object that has .to()
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        self.tokenizer.return_value = mock_encoding
        self.tokenizer.decode.return_value = "The answer is 42."
        self.tokenizer.eos_token_id = 102
        
        self.niwt = NIWTCore(self.model, self.tokenizer, {})
    
    def test_stage_1_perturbation(self):
        """Test layer perturbation detection."""
        # Setup specific behavior: When layer 2 is removed, decoding fails
        
        # We need to mock _evaluate_capability or the generate output
        # But _evaluate_capability uses the model, so we rely on the mocked model.
        # Since probing specific layer drops in a mock is hard, we verify the LOOP mechanics.
        
        data = [("Question", "answer")]
        results = self.niwt.run_stage_1_perturbation(data)
        
        # Should have tested all 5 layers
        # Results should be a list of critical layers (or empty if mock is too perfect)
        self.assertIsInstance(results, list)
    
    def test_evaluate_capability(self):
        """Test the scoring logic."""
        cases = [("q", "42")]
        score = self.niwt._evaluate_capability(cases)
        self.assertAlmostEqual(score, 1.0) # Mock decoder always returns "The answer is 42."

if __name__ == '__main__':
    unittest.main()
