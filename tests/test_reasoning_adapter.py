import torch
import unittest
import sys
import os

# Add src to path
sys.path.append("/mnt/d/Research Experiments/nexus/src")

from nexus_core.adapters.reasoning_adapter import ReasoningAdapter

class TestReasoningAdapter(unittest.TestCase):
    def test_profile_loading(self):
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=2048)
        profile_path = "mock_critical_layers.json"
        
        # Create mock profile
        import json
        with open(profile_path, 'w') as f:
            json.dump({"critical_layers": [{"layer": 16}, {"layer": 24}]}, f)
        
        try:
            adapter.load_profile(profile_path)
            self.assertTrue(len(adapter.critical_layers) > 0)
            self.assertIn(16, adapter.critical_layers)
        finally:
            if os.path.exists(profile_path):
                os.remove(profile_path)
        
        print(f"\nLoaded critical layers: {adapter.critical_layers}")

    def test_forward_pass(self):
        adapter = ReasoningAdapter(teacher_dim=4096, student_dim=1024)
        
        batch_size = 2
        seq_len = 10
        teacher_dim = 4096
        student_dim = 1024
        
        teacher_states = torch.randn(batch_size, seq_len, teacher_dim)
        student_query = torch.randn(batch_size, seq_len, student_dim)
        
        # Test without student query
        out, gate = adapter(teacher_states)
        self.assertEqual(out.shape, (batch_size, seq_len, student_dim))
        self.assertEqual(gate.shape, (batch_size, seq_len, 1))
        
        # Test with student query
        out_q, gate_q = adapter(teacher_states, student_query)
        self.assertEqual(out_q.shape, (batch_size, seq_len, student_dim))
        
if __name__ == '__main__':
    unittest.main()
