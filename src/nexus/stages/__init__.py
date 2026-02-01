"""
Stage scripts package for capability training.

Each stage implements training for a specific capability.
"""

from .base import BaseStage
from .stage_cot import CoTStage
from .stage_reasoning import ReasoningStage
from .stage_thinking import ThinkingStage
from .stage_tools import ToolsStage
from .stage_streaming import StreamingStage
from .stage_podcast import PodcastStage
from .stage_vision_qa import VisionQAStage
from .stage_omni import OmniTrainingStage

__all__ = [
    "BaseStage", 
    "CoTStage", 
    "ReasoningStage", 
    "ThinkingStage", 
    "ToolsStage",
    "StreamingStage",
    "PodcastStage",
    "VisionQAStage",
    "OmniTrainingStage"
]
