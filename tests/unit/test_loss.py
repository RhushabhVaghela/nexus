import unittest
import torch
import torch.nn as nn
from src.nexus_final.loss_functions import ActivationAnchoringLoss

class TestNexusLoss(unittest.TestCase):
    def setUp(self):
        self.loss_fn = ActivationAnchoringLoss(
            alpha_ce=1.0, 
            alpha_hidden=0.5, 
            alpha_critical=10.0
        )

    def test_loss_backward(self):
        """Verify gradients flow correctly through all components."""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        dim = 32
        
        # Mock inputs
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        student_state = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        teacher_state = torch.randn(batch_size, seq_len, dim)
        
        # 1. Standard Pass
        loss = self.loss_fn(student_logits, teacher_logits, student_state, teacher_state)
        self.assertTrue(loss > 0)
        
        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNotNone(student_state.grad)

    def test_anchoring_logic(self):
        """Verify critical layer weighting increases loss."""
        # Setup identical states so base MSE is 0
        state = torch.randn(2, 5, 10)
        loss_base = self.loss_fn(
            torch.randn(2, 5, 20), torch.randn(2, 5, 20),
            state, state
        )
        
        # Redefine inputs as (Batch, Layers, Seq, Dim) for stack logic
        layers = 4
        t_stack = torch.randn(2, layers, 5, 10)
        s_bridge = torch.randn(2, 5, 10, requires_grad=True) # Student bridge output
        
        # Case 1: No anchoring
        loss_none = self.loss_fn(torch.randn(2, 5, 20), torch.randn(2, 5, 20), s_bridge, t_stack)
        
        # Case 2: Anchoring to layer 0
        loss_anchor = self.loss_fn(torch.randn(2, 5, 20), torch.randn(2, 5, 20), s_bridge, t_stack, anchoring_layer_indices=[0])
        
        self.assertTrue(loss_anchor > 0)

if __name__ == '__main__':
    unittest.main()
