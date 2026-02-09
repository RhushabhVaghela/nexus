"""
Re-export from canonical location: src.models.loss_functions

This module previously contained a separate ActivationAnchoringLoss implementation
with a different API (student_features, teacher_features, anchoring_mask).
The canonical implementation lives in src.models.loss_functions and uses the full
multi-layer distillation API (student_logits, teacher_logits, student_states,
teacher_states, anchoring_layer_indices).

All imports should use the canonical path, but this re-export ensures backward
compatibility for any code that imports from this location.
"""

from nexus.models.loss_functions import ActivationAnchoringLoss, RecoveryStepLoss

__all__ = ["ActivationAnchoringLoss", "RecoveryStepLoss"]
