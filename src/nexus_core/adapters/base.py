import torch
import torch.nn as nn

class BaseAdapter(nn.Module):
    """
    The 'Learned Affine Alignment' Module.
    Maps Teacher Manifold (e.g. 30B dim) -> Student Manifold (4096 dim).
    
    Research Basis:
    - 'Representation Calibration' (LIT)
    - Fixes 'Hidden Representation Mismatch'
    """
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        
        # 1. Projection (Dimension Reduction)
        # Using "Neutral Residue" init (small values) to prevent shock
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        
        # 2. Affine Alignment (Scale + Shift)
        # Allows rotating/scaling the manifold to match Student's distribution
        self.scale = nn.Parameter(torch.ones(student_dim))
        self.shift = nn.Parameter(torch.zeros(student_dim))
        
        # 3. Activation (Optional, keeps it linear if purely geometric alignment)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = x * self.scale + self.shift
        return self.act(x)
