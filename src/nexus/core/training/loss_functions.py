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
        
        # 1. KL/CE Divergence (Output Distillation)
        loss_ce = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * self.alpha_ce
        
        # 2. Hidden State Matching (The 'Bridge')
        # Student projects 4096 -> Teacher Dim? Or Teacher projects to Student 4096?
        # NIWT 'Assignment-Guided Dimension Reduction' says we align to Student space.
        # Assume inputs are already projected to same dim.
        loss_hidden = F.mse_loss(student_states, teacher_states) * self.alpha_hidden
        
        # 3. ACTIVATION ANCHORING (The Innovation)
        loss_surgical = 0.0
        if anchoring_layer_indices is not None and len(anchoring_layer_indices) > 0:
            # anchoring_layer_indices: List[int] or Tensor of active layer indices
            # teacher_states: Tensor (Batch, NumLayers, Seq, Dim) OR (NumLayers, Batch, Seq, Dim)
            # We assume (Batch, NumLayers, ...) for standard pipelining
            
            # For each critical layer, we extract the corresponding state and add penalty
            # if the student's *projection* for that layer drifts.
            # However, `student_states` is typically the *output* state (final).
            # If we want detailed surgery, we need student to output a stack too. 
            # Assuming here that `student_states` corresponds to the mapped latent of the specific layer
            # OR that we are comparing the 'Bridge' representation.
            
            # Simplified Logic: We penalize the distance between the Student's "Bridge State"
            # and the *subset* of Teacher states marked as critical, averaged.
            # This forces the Student Bridge to align with the "Reasoning Center" of the teacher.
            
            # Identify which dimension represents layers
            # Heuristic: teacher_states.shape[1] == num_layers logic
            # Let's assume teacher_states is [Batch, Layers, Seq, Dim]
            
            try:
                # Select critical layers: (Batch, n_crit, Seq, Dim)
                crit_states = teacher_states[:, anchoring_layer_indices, :, :]
                
                # Average them to get the "Critical Center"
                target_center = crit_states.mean(dim=1) # (Batch, Seq, Dim)
                
                # Penalize Student Deviation from this center
                # This aligns the Student Core with the aggregate "Reasoning Hub"
                loss_surgical = F.mse_loss(student_states, target_center) * self.alpha_critical
                
            except Exception as e:
                # Fallback if shapes don't match (e.g. standard distillation pass)
                print(f"[Warn] Activation Anchoring Loss skipped due to shape mismatch: {e}")
                loss_surgical = 0.0 
        
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
