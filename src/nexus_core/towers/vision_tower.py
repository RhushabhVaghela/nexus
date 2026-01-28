import torch
import torch.nn as nn
from .base_tower import BaseTower
from ..adapters.vision_adapter import VisionAdapter

class VisionTower(BaseTower):
    def __init__(self, config, teacher_dim, student_dim):
        super().__init__(config)
        self.projection = VisionAdapter(teacher_dim, student_dim)
        
    def forward(self, pixel_values):
        if self.frozen_teacher is None:
            raise ValueError("Teacher model not loaded.")
            
        with torch.no_grad():
            outputs = self.frozen_teacher.vision_model(pixel_values, output_hidden_states=True)
            teacher_hidden = outputs.last_hidden_state
            
        projected_output = self.projection(teacher_hidden)
        return projected_output
