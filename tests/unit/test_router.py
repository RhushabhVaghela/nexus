import unittest
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/nexus_core/student'))
try:
    from router import SparseIntentRouter
except ImportError:
    # Fallback if pathing is tricky in test runner
    sys.path.append(os.path.abspath("src/nexus_core/student"))
    from router import SparseIntentRouter

class TestRouter(unittest.TestCase):
    def setUp(self):
        self.input_dim = 128
        self.num_towers = 5
        self.router = SparseIntentRouter(self.input_dim, self.num_towers)

    def test_neural_routing_shape(self):
        """Verify output shape of the router."""
        batch_size = 4
        x = torch.randn(batch_size, self.input_dim)
        scores, indices = self.router(x)
        
        self.assertEqual(scores.shape, (batch_size, 1)) # Top-1 by default
        self.assertEqual(indices.shape, (batch_size, 1))

    def test_heuristic_override(self):
        """Verify that heuristic keyword forces specific tower."""
        x = torch.randn(1, self.input_dim)
        
        # Force 'vision' (index 1 in our list)
        scores, indices = self.router(x, heuristic_override="vision")
        
        self.assertEqual(indices.item(), 1) # Must pick index 1
        
    def test_keyword_detection(self):
        """Verify text parser."""
        prompt = "Can you describe this image?"
        tag = self.router.get_routing_map(prompt)
        self.assertEqual(tag, "vision")
        
        prompt = "Write a python script to solve this"
        tag = self.router.get_routing_map(prompt)
        self.assertEqual(tag, "reasoning")

if __name__ == '__main__':
    unittest.main()
