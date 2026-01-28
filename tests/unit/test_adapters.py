import unittest
import torch
import torch.nn as nn

# Mock Architecture Class (representing the Adapter)
class MockReasoningAdapter(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)
        self.norm = nn.LayerNorm(student_dim)
    
    def forward(self, x):
        return self.norm(self.proj(x))

class TestAdapters(unittest.TestCase):
    def setUp(self):
        self.teacher_dim = 256
        self.student_dim = 128
        self.adapter = MockReasoningAdapter(self.teacher_dim, self.student_dim)

    def test_dimensions(self):
        """Verify output dimensions match student core expectation."""
        batch_size = 4
        x = torch.randn(batch_size, self.teacher_dim)
        output = self.adapter(x)
        self.assertEqual(output.shape, (batch_size, self.student_dim))

    def test_gradients(self):
        """Verify gradients flow back through the adapter."""
        x = torch.randn(2, self.teacher_dim)
        output = self.adapter(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.adapter.proj.weight.grad)

    def test_zero_initialization_safety(self):
        """Test if we can initialize projection to near-zero (Neutral Residue)."""
        nn.init.normal_(self.adapter.proj.weight, mean=0.0, std=1e-5)
        x = torch.randn(1, self.teacher_dim)
        output = self.adapter(x)
        # Norm layer will standardize it, but pre-norm values should be tiny
        pre_norm = self.adapter.proj(x)
        self.assertTrue(torch.all(torch.abs(pre_norm) < 0.1))

if __name__ == '__main__':
    unittest.main()
