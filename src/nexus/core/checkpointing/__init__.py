"""
Nexus Core Checkpointing Module

Provides activation checkpointing for SLI streaming workloads:
- ActivationCheckpointer for periodic activation storage
"""

from .activation_checkpointer import ActivationCheckpointer

__all__ = [
    "ActivationCheckpointer",
]
