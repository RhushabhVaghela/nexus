"""
Re-export shim: NexusTrainer is canonically defined in src.models.distill.

This module re-exports NexusTrainer from the canonical location so that
existing imports like `from nexus.core.training.loop import NexusTrainer`
or `from nexus.core.training.loop import TrainingLoop` continue to work.

DO NOT add new logic here. All trainer logic belongs in src/models/distill.py.
"""

from nexus.models.distill import NexusTrainer

# Alias for backward compatibility with benchmarks that import TrainingLoop
TrainingLoop = NexusTrainer

__all__ = ["NexusTrainer", "TrainingLoop"]
