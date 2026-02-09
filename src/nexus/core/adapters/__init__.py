"""
Nexus Core Adapters Package.

Provides modality-specific adapters for projecting teacher representations
into the student's feature space.

Modules:
  - base: BaseAdapter abstract class
  - reasoning_adapter: ReasoningAdapter (text/reasoning modality)
  - vision_adapter: VisionAdapter (vision modality)
  - audio_adapter: AudioAdapter (audio modality)
"""

from .base import BaseAdapter
from .reasoning_adapter import ReasoningAdapter
from .vision_adapter import VisionAdapter
from .audio_adapter import AudioAdapter

__all__ = [
    "BaseAdapter",
    "ReasoningAdapter",
    "VisionAdapter",
    "AudioAdapter",
]
