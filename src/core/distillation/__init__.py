"""
Core distillation algorithms for Nexus knowledge distillation platform.

Provides 5 advanced distillation techniques:
- CoT (Chain-of-Thought) Distillation
- KPOD (Keypoint Progressive Distillation)
- Multi-Teacher Distillation
- TAID (Temporal Adaptive Interpolated Distillation)
- QCRD (Quality-Controlled Synthetic Data with Contrastive Learning)
"""

from src.core.distillation.cot_distillation import CoTDistiller
from src.core.distillation.kpod import KPODDistiller
from src.core.distillation.multi_teacher import MultiTeacherDistiller
from src.core.distillation.taid import TAIDDistiller
from src.core.distillation.qcrd import QCRDDistiller

__all__ = [
    "CoTDistiller",
    "KPODDistiller",
    "MultiTeacherDistiller",
    "TAIDDistiller",
    "QCRDDistiller",
]
