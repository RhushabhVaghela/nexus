import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseAdapter

class VisionAdapter(nn.Module):
    """
    Adapter for the Vision Specialist Tower (Step3-VL, SigLIP).
    Maps visual tokens to Student Space.
    """
    def __init__(self, teacher_dim: int = 2048, student_dim: int = 4096):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        # AGID Projection Layer (BaseAdapter handles initialization and affine)
        self.alignment = BaseAdapter(teacher_dim, student_dim)
        
        # Gating mechanism
        self.gate_proj = nn.Linear(student_dim, 1)
        
        self.critical_layers: List[int] = []

    def load_profile(self, profile_path: str):
        import json
        import os
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                data = json.load(f)
            self.critical_layers = [item['layer'] for item in data.get('critical_layers', [])]
            print(f"[VisionAdapter] Critical Layers: {self.critical_layers}")

    def forward(self, teacher_activations: torch.Tensor):
        """
        Args:
            teacher_activations: (Batch, Seq, Teacher_Dim)
        """
        # 1. Project
        aligned = self.alignment(teacher_activations)
        
        # 2. Gate
        gate_score = torch.sigmoid(self.gate_proj(aligned))
        
        # 3. Output
        return aligned * gate_score
