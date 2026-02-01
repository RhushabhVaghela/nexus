import torch
import torch.nn as nn

class ActivationGuidedProjector(nn.Module):
    """
    Learned projection that maps external embeddings/activations 
    into the Nexus Student's hidden dimension (4096).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
