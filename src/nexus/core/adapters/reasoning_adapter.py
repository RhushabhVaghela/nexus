import torch
import torch.nn as nn
from typing import List, Dict, Optional
from .base import BaseAdapter

class ReasoningAdapter(nn.Module):
    """
    Adapter for the Reasoning Specialist Tower.
    Uses Activation-Guided Consolidation to distill critical reasoning capabilities
    from the teacher model.
    """
    def __init__(self, teacher_dim: int, student_dim: int):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        # Core Alignment Layer (Projection + Affine)
        # We use the BaseAdapter logic here
        self.alignment = BaseAdapter(teacher_dim, student_dim)
        
        # Gating/Attention mechanism
        # Decides importance of features
        self.gate_proj = nn.Linear(student_dim, 1)
        
        self.critical_layers: List[int] = []
        
    def load_profile(self, profile_path: str):
        """
        Load critical layers from a NIWT profile JSON.
        """
        import json
        import os
        
        if not os.path.exists(profile_path):
            print(f"[Warn] Profile {profile_path} not found. Using default full-pass.")
            return

        with open(profile_path, 'r') as f:
            data = json.load(f)
            
        self.critical_layers = [item['layer'] for item in data.get('critical_layers', [])]
        print(f"[Adapter] Configured with critical layers: {self.critical_layers}")

    def forward(self, teacher_hidden_states: torch.Tensor, student_query: Optional[torch.Tensor] = None):
        """
        Args:
            teacher_hidden_states: Tensor of shape (Batch, Seq, TeacherDim)
        """
        # 1. Align Manifolds (Teacher -> Student Space)
        aligned_features = self.alignment(teacher_hidden_states)
        
        # 2. Self-Gating (Importance weighting)
        # "Is this feature actually useful?"
        gate_score = torch.sigmoid(self.gate_proj(aligned_features))
        
        # 3. Apply Gate
        output = aligned_features * gate_score
        
        return output, gate_score
