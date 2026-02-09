"""
Nexus Core Training Package.

Provides training loop, loss functions, data loading, and distillation trainer.

Canonical implementations:
  - NexusTrainer: src.models.distill (re-exported here via loop.py)
  - ActivationAnchoringLoss, RecoveryStepLoss: src.models.loss_functions
    (re-exported here via loss.py and loss_functions.py)
  - NexusDistillationTrainer: student_trainer.py (standalone, uses transformers)
  - NexusDataset, get_dataloader: data_loader.py
"""

from .loop import NexusTrainer, TrainingLoop
from .loss import ActivationAnchoringLoss, RecoveryStepLoss
from .student_trainer import NexusDistillationTrainer
from .data_loader import NexusDataset, get_dataloader

__all__ = [
    "NexusTrainer",
    "TrainingLoop",
    "ActivationAnchoringLoss",
    "RecoveryStepLoss",
    "NexusDistillationTrainer",
    "NexusDataset",
    "get_dataloader",
]
