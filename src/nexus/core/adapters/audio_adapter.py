import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseAdapter

class AudioAdapter(nn.Module):
    """
    Adapter for the Audio Specialist Tower (Whisper, Parakeet, Qwen3-Audio).
    """
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        # AGID Projection
        self.alignment = BaseAdapter(teacher_dim, student_dim)
        
        # Gating
        self.gate_proj = nn.Linear(student_dim, 1)
        
        self.critical_layers: List[int] = []

    def load_profile(self, profile_path: str):
        import json
        import os
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                data = json.load(f)
            self.critical_layers = [item['layer'] for item in data.get('critical_layers', [])]
            print(f"[AudioAdapter] Critical Layers: {self.critical_layers}")

    def forward(self, teacher_activations: torch.Tensor, student_query: Optional[torch.Tensor] = None):
        # 1. Project
        aligned = self.alignment(teacher_activations)
        
        # 2. Gate
        gate_score = torch.sigmoid(self.gate_proj(aligned))
        
        # 3. Output
        return aligned * gate_score, gate_score
