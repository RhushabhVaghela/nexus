import torch
import unittest
import sys

sys.path.append("/mnt/d/Research Experiments/nexus/src")

from nexus_core.student.sparse_router import SparseIntentRouter, HardModalityRouter

class TestSparseRouter(unittest.TestCase):
    def test_forward_pass(self):
        dim = 128
        router = SparseIntentRouter(input_dim=dim, num_towers=4)
        
        # Test input handling (pooled vs sequence)
        inputs_pooled = torch.randn(2, dim) # Batch, Dim
        probs, mask = router(inputs_pooled)
        
        self.assertEqual(probs.shape, (2, 4))
        self.assertEqual(mask.shape, (2, 4))
        
        inputs_seq = torch.randn(2, 10, dim) # Batch, Seq, Dim
        probs_seq, mask_seq = router(inputs_seq)
        
        self.assertEqual(probs_seq.shape, (2, 4)) # Router pools sequence
        
    def test_hard_router(self):
        # Test keyword matching
        text_vision = "I see a picture of a cat"
        active_vision = HardModalityRouter.route_by_keywords(text_vision)
        self.assertIn("vision", active_vision)
        self.assertIn("reasoning", active_vision)
        
        text_audio = "Listen to this sound"
        active_audio = HardModalityRouter.route_by_keywords(text_audio)
        self.assertIn("audio", active_audio)
        
        text_gen = "Generate an image of a dog"
        active_gen = HardModalityRouter.route_by_keywords(text_gen)
        self.assertIn("generation", active_gen)

if __name__ == '__main__':
    unittest.main()
