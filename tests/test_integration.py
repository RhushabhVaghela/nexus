import torch
import unittest
import sys

sys.path.append("/mnt/d/Research Experiments/nexus/src")

from nexus_core.student.core import NexusStudentCore
from nexus_core.adapters.reasoning_adapter import ReasoningAdapter
from nexus_core.adapters.vision_adapter import VisionAdapter
from nexus_core.adapters.audio_adapter import AudioAdapter

class TestStudentIntegration(unittest.TestCase):
    def test_end_to_end_shapes(self):
        # Config
        d_model = 4096
        vocab_size = 1000 # Small for test
        batch_size = 2
        seq_len = 16
        
        # 1. Instantiate Core
        student = NexusStudentCore(d_model=d_model, vocab_size=vocab_size)
        
        # 2. Instantiate Adapters (Ensuring dims match)
        # Teacher dims might vary.
        teacher_dim_reasoning = 4096
        teacher_dim_vision = 2048
        
        reasoning_adapter = ReasoningAdapter(hidden_size=teacher_dim_reasoning, student_dim=d_model) # MUST match d_model
        vision_adapter = VisionAdapter(teacher_dim=teacher_dim_vision, student_dim=d_model)
        
        # 3. Simulate Inputs
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        student_query = torch.randn(batch_size, seq_len, d_model) # Approx student state
        
        # Teacher Outputs (Mock)
        teacher_reasoning_out = torch.randn(batch_size, seq_len, teacher_dim_reasoning)
        teacher_vision_out = torch.randn(batch_size, seq_len, teacher_dim_vision)
        
        # 4. Run Adapters
        r_out, _ = reasoning_adapter(teacher_reasoning_out, student_query)
        v_out, _ = vision_adapter(teacher_vision_out) # Vision adapter might not use student query yet?
        
        print(f"Reasoning Adapter Out: {r_out.shape}")
        print(f"Vision Adapter Out: {v_out.shape}")
        
        self.assertEqual(r_out.shape[-1], d_model)
        self.assertEqual(v_out.shape[-1], d_model)
        
        # 5. Run Student Core with Activations
        tower_activations = {
            'reasoning': r_out,
            'vision': v_out
        }
        
        logits = student(input_ids, tower_activations)
        print(f"Student Logits: {logits.shape}")
        
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))

if __name__ == '__main__':
    unittest.main()
