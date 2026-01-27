import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTower(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict()
        self.frozen_teacher = None # Placeholder for the loaded teacher model
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def load_teacher(self, teacher_model):
        """
        Loads the frozen teacher model (quantized) into the tower.
        """
        self.frozen_teacher = teacher_model
        for param in self.frozen_teacher.parameters():
            param.requires_grad = False
            
    def add_adapter(self, adapter_name, adapter_module):
        self.adapters[adapter_name] = adapter_module

    def get_adapter(self, adapter_name):
        return self.adapters[adapter_name]
