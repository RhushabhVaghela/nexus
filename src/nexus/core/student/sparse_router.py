"""
Re-export from canonical location: src.core.student.router

This module previously contained a duplicate definition of SparseIntentRouter
and HardModalityRouter. The canonical implementations live in
src.core.student.router. All imports should use the canonical path, but this
re-export ensures backward compatibility for any code that imports from this
location.
"""

from nexus.core.student.router import SparseIntentRouter, HardModalityRouter

__all__ = ["SparseIntentRouter", "HardModalityRouter"]
