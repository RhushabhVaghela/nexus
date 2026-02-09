"""
Stage scripts package for capability training.

Each stage implements training for a specific capability.
All stages are lazy-loaded to avoid importing torch at package import time.
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # base.py
    "BaseStage": (".base", "BaseStage"),
    # stage_cot.py
    "CoTStage": (".stage_cot", "CoTStage"),
    # stage_reasoning.py
    "ReasoningStage": (".stage_reasoning", "ReasoningStage"),
    # stage_thinking.py
    "ThinkingStage": (".stage_thinking", "ThinkingStage"),
    # stage_tools.py
    "ToolsStage": (".stage_tools", "ToolsStage"),
    # stage_streaming.py
    "StreamingStage": (".stage_streaming", "StreamingStage"),
    # stage_podcast.py
    "PodcastStage": (".stage_podcast", "PodcastStage"),
    # stage_vision_qa.py
    "VisionQAStage": (".stage_vision_qa", "VisionQAStage"),
    # stage_omni.py
    "OmniTrainingStage": (".stage_omni", "OmniTrainingStage"),
    # stage_image_gen.py
    "ImageProjector": (".stage_image_gen", "ImageProjector"),
    "ImageGenStage": (".stage_image_gen", "ImageGenStage"),
    # stage_remotion_gen.py
    "RemotionGenStage": (".stage_remotion_gen", "RemotionGenStage"),
    # stage_tri_streaming.py
    "TriStreamingStage": (".stage_tri_streaming", "TriStreamingStage"),
    # stage_video.py
    "VideoUnderstandingStage": (".stage_video", "VideoUnderstandingStage"),
    # stage_video_gen.py
    "VideoProjector": (".stage_video_gen", "VideoProjector"),
    "VideoGenStage": (".stage_video_gen", "VideoGenStage"),
    # agent_finetune.py
    "AgentFinetuneConfig": (".agent_finetune", "AgentFinetuneConfig"),
    "AgentDataset": (".agent_finetune", "AgentDataset"),
    "AgentFinetuner": (".agent_finetune", "AgentFinetuner"),
    # reasoning_grpo.py
    "ReasoningGRPOConfig": (".reasoning_grpo", "ReasoningGRPOConfig"),
    "GRPODataset": (".reasoning_grpo", "GRPODataset"),
    "GRPOTrainer": (".reasoning_grpo", "GRPOTrainer"),
    # reasoning_sft.py
    "ReasoningSFTConfig": (".reasoning_sft", "ReasoningSFTConfig"),
    "ReasoningDataset": (".reasoning_sft", "ReasoningDataset"),
    "ReasoningSFTTrainer": (".reasoning_sft", "ReasoningSFTTrainer"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # Base
    "BaseStage",
    # Stage classes
    "CoTStage",
    "ReasoningStage",
    "ThinkingStage",
    "ToolsStage",
    "StreamingStage",
    "PodcastStage",
    "VisionQAStage",
    "OmniTrainingStage",
    # Image/Video generation stages
    "ImageProjector",
    "ImageGenStage",
    "RemotionGenStage",
    "TriStreamingStage",
    "VideoUnderstandingStage",
    "VideoProjector",
    "VideoGenStage",
    # Trainers
    "AgentFinetuneConfig",
    "AgentDataset",
    "AgentFinetuner",
    "ReasoningGRPOConfig",
    "GRPODataset",
    "GRPOTrainer",
    "ReasoningSFTConfig",
    "ReasoningDataset",
    "ReasoningSFTTrainer",
]
