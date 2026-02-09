"""
Core distillation algorithms for Nexus knowledge distillation platform.

Provides 5 advanced distillation techniques:
- CoT (Chain-of-Thought) Distillation
- KPOD (Keypoint Progressive Distillation)
- Multi-Teacher Distillation
- TAID (Temporal Adaptive Interpolated Distillation)
- QCRD (Quality-Controlled Synthetic Data with Contrastive Learning)
"""

from nexus.core.distillation.cot_distillation import CoTDistiller
from nexus.core.distillation.kpod import KPODDistiller
from nexus.core.distillation.multi_teacher import MultiTeacherDistiller
from nexus.core.distillation.taid import TAIDDistiller
from nexus.core.distillation.qcrd import QCRDDistiller

__all__ = [
    "CoTDistiller",
    "KPODDistiller",
    "MultiTeacherDistiller",
    "TAIDDistiller",
    "QCRDDistiller",
]
