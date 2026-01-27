import torch
import torch.nn as nn
from typing import List, Dict

class ActivationAnchoringLoss(nn.Module):
    """
    Implements Importance-Weighted Activation Anchoring (IWAA) in Protected Subspaces.
    
    Research Basis:
    - Standard MSE treats all features equally.
    - IWAA penalizes divergence in 'Critical Layers' (identified by NIWT) 
      significantly more (10x) than generic layers.
    - Also adds a Variance Penalty to preserve the 'Texture/Soul' of activations.
    """
    def __init__(self, alpha_mse=1.0, beta_var=0.5, critical_weight=10.0):
        super().__init__()
        self.alpha = alpha_mse
        self.beta = beta_var
        self.critical_weight = critical_weight
        self.mse = nn.MSELoss(reduction='none') # We need element-wise to apply weights

    def forward(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor, 
        anchoring_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            student_features: (B, S, D)
            teacher_features: (B, S, D)
            anchoring_mask: (D,) or (B, S, D) - 1.0 for critical neurons, 0.0 for others.
                           If None, assumes all are equal (Standard MSE).
        """
        # 1. Base Feature Matching (MSE)
        diff = student_features - teacher_features
        mse_loss = (diff ** 2)
        
        # 2. Importance Weighting (Protected Subspaces)
        if anchoring_mask is not None:
            # Apply 10x penalty to critical regions
            weight = 1.0 + (anchoring_mask * (self.critical_weight - 1.0))
            weighted_mse = (mse_loss * weight).mean()
        else:
            weighted_mse = mse_loss.mean()
            
        # 3. Variance Penalty (Activation Anchoring)
        # We want the student to match the *variance* (texture) of the teacher, not just the mean.
        # This prevents "Mean-Collapse" (blurriness).
        student_var = torch.var(student_features, dim=-1)
        teacher_var = torch.var(teacher_features, dim=-1)
        var_loss = nn.functional.mse_loss(student_var, teacher_var)
        
        # Total Loss
        total_loss = (self.alpha * weighted_mse) + (self.beta * var_loss)
        
        return total_loss
