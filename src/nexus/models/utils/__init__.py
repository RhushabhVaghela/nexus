"""
Nexus Model Utils — memory estimation and VRAM management.
"""

from .memory import (
    check_memory_headroom,
    get_recommended_batch_size,
    estimate_model_vram_gb,
    should_use_sli,
)

__all__ = [
    "check_memory_headroom",
    "get_recommended_batch_size",
    "estimate_model_vram_gb",
    "should_use_sli",
]
