"""
Nexus Core Student Package.

Provides the student model core and routing modules.

Canonical implementations:
  - NexusStudentCore: core.py
  - SparseIntentRouter, HardModalityRouter: router.py
    (re-exported in sparse_router.py for backward compatibility)
"""

from .core import NexusStudentCore
from .router import SparseIntentRouter, HardModalityRouter

__all__ = [
    "NexusStudentCore",
    "SparseIntentRouter",
    "HardModalityRouter",
]
