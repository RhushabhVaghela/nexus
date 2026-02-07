"""
Nexus Core Profiling Module

Provides neural network profiling and analysis tools:
- NIWT (Neural Information-Weighted Tower) for layer criticality analysis
- ThermalProtection for hardware safety monitoring
"""

from .niwt import NIWTCore, ThermalProtection, EvaluationDataset

__all__ = [
    "NIWTCore",
    "ThermalProtection",
    "EvaluationDataset",
]
