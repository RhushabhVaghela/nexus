import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationAnchoringLoss(nn.Module):
    """
    Implements 'Importance-Weighted Activation Anchoring (protected subspaces)' & 'Recovery Step'.
    
    Logic:
    1. Standard CE Loss for token prediction.
    2. MSE Loss for hidden states (Representation Calibration).
    3. WEIGHTED MSE for 'Critical Layers' (The 'Soul' preservation).
    """
    def __init__(self, alpha_ce=1.0, alpha_hidden=0.5, alpha_critical=5.0):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.alpha_hidden = alpha_hidden
        self.alpha_critical = alpha_critical # High penalty for critical layer divergence

    def forward(self, 
                student_logits, teacher_logits, 
                student_states, teacher_states, 
                anchoring_layer_indices=None):
        """
        Calculates the multi-objective distillation loss.
        student_states: (Batch, Seq, Dim) - Current student representation
        teacher_states: (Batch, NumLayers, Seq, Dim) - Full teacher activation stack
        anchoring_layer_indices: List[int] - Indices of critical layers to anchor against
        """
        
        # 1. KL/CE Divergence (Output Distillation)
        # Use log_softmax for student (log-probs) and softmax for teacher (probs)
        loss_ce = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits / 2.0, dim=-1), # Temperature 2.0 for softer targets
            reduction='batchmean'
        ) * (self.alpha_ce * 4.0) # Scale by T^2
        
        # 2. Hidden State Matching (The 'Bridge')
        # We assume student_states and teacher_states (if single layer) are aligned.
        # If teacher_states is multi-layer, we use the mean of the selected layers as the target.
        if teacher_states.dim() == 4: # [Batch, Layers, Seq, Dim]
            if anchoring_layer_indices is not None:
                target_states = teacher_states[:, anchoring_layer_indices, :, :].mean(dim=1)
            else:
                target_states = teacher_states.mean(dim=1)
        else:
            target_states = teacher_states

        loss_hidden = F.mse_loss(student_states, target_states) * self.alpha_hidden
        
        # 3. ACTIVATION ANCHORING (Surgical Precision)
        loss_surgical = 0.0
        if anchoring_layer_indices is not None and len(anchoring_layer_indices) > 0 and teacher_states.dim() == 4:
            # We enforce that the student CORE representation doesn't drift from 
            # the specific "soul" layers of the teacher.
            for idx in anchoring_layer_indices:
                layer_target = teacher_states[:, idx, :, :]
                loss_surgical += F.mse_loss(student_states, layer_target)
            
            loss_surgical = (loss_surgical / len(anchoring_layer_indices)) * self.alpha_critical
        
        return loss_ce + loss_hidden + loss_surgical

class RecoveryStepLoss(nn.Module):
    """
    Specific loss for the Phase 4 'Recovery Step'.
    Focuses purely on high-frequency feature reconstruction (Voice Soul, Logic CoT).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, original_features, reconstructed_features):
        # Cosine Similarity allows for magnitude shifts (Volume/Scale agnostic)
        # but enforces Directional Alignment (Tone/Reasoning Pattern).
        return 1.0 - F.cosine_similarity(original_features, reconstructed_features, dim=-1).mean()
