import torch
import torch.nn as nn
from .base_tower import BaseTower
from ..adapters.reasoning_adapter import ReasoningAdapter

class ReasoningTower(BaseTower):
    def __init__(self, config, teacher_dim, student_dim):
        super().__init__(config)
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        
        # specific adapter for reasoning projection
        # Assuming ReasoningAdapter is a projection layer
        self.projection = ReasoningAdapter(teacher_dim, student_dim)
        
    def forward(self, input_ids, attention_mask=None):
        if self.frozen_teacher is None:
            raise ValueError("Teacher model not loaded.")
            
        # 1. Pass through Frozen Teacher to get hidden states
        #    We only need the output of specific layers or the final hidden state
        with torch.no_grad():
            outputs = self.frozen_teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Assuming we take the last hidden state for now, 
            # OR we take specific activation layers identified by NIWT
            teacher_hidden = outputs.hidden_states[-1] 
        
        # 2. Project to Student Space
        projected_output = self.projection(teacher_hidden)
        
        return projected_output
