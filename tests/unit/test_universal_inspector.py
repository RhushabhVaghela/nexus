import torch.nn as nn
import unittest
from nexus_core.utils.universal_inspector import UniversalInspector

class MockLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

class MockWeirdArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard names missing, but has a deep encoder stack
        self.encoder = nn.Module()
        self.encoder.deep_stack = nn.ModuleList([nn.Linear(10, 10) for _ in range(12)])
        self.projection = nn.Linear(10, 5)

class MockDiffusionStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # Multiple containers, should pick the largest or deepest
        self.down_blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        self.mid_block = nn.Linear(10, 10)
        self.up_blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        
        # Artificial "Main Stack" hidden deeper
        self.transformer = nn.Module()
        self.transformer.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(24)])

class TestUniversalInspector(unittest.TestCase):
    def test_standard_llm(self):
        model = MockLLM()
        layers = UniversalInspector.find_backbone_layers(model)
        self.assertEqual(len(layers), 5)
        self.assertEqual(layers, model.model.layers)

    def test_weird_deep_stack(self):
        model = MockWeirdArchitecture()
        layers = UniversalInspector.find_backbone_layers(model)
        self.assertEqual(len(layers), 12)
        self.assertEqual(layers, model.encoder.deep_stack)

    def test_diffusion_heuristic(self):
        model = MockDiffusionStyle()
        layers = UniversalInspector.find_backbone_layers(model)
        # Should pick the 24-layer transformer stack over the 3-layer down/up blocks
        self.assertEqual(len(layers), 24)
        self.assertEqual(layers, model.transformer.blocks)

if __name__ == "__main__":
    unittest.main()
