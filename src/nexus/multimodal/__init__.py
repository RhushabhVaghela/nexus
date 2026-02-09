"""
Multimodal processing package for Nexus.

Provides encoders, processors, fusion pipelines, reasoning wrappers,
tool execution, modality detection, and the OmniMultimodalLM model.

Sub-packages:
    - connectors: DFMConnector, OptimalTransport, FlowMatchingBlock
    - datasets: EMM1Dataset, UnifiedMultiDatasetLoader
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# The torch __spec__ patch (previously done eagerly) is now deferred to the
# first time a torch-dependent submodule is actually loaded.
# ---------------------------------------------------------------------------


def _patch_torch_spec():
    """Patch torch.__spec__ if it's None (fixes datasets loading issues)."""
    import sys

    try:
        import torch

        if getattr(torch, "__spec__", None) is None:
            from importlib.machinery import ModuleSpec

            dummy_spec = ModuleSpec(
                name="torch",
                loader=None,
                origin=getattr(torch, "__file__", "unknown"),
            )
            torch.__spec__ = dummy_spec
            sys.modules["torch"].__spec__ = dummy_spec
    except ImportError:
        pass


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # download.py
    "download_vision_data": (".download", "download_vision_data"),
    "download_audio_data": (".download", "download_audio_data"),
    "download_video_data": (".download", "download_video_data"),
    "get_test_prompts": (".download", "get_test_prompts"),
    # distillation.py
    "MultimodalDataProcessor": (".distillation", "MultimodalDataProcessor"),
    # model.py
    "OmniMultimodalLM": (".model", "OmniMultimodalLM"),
    # encoders.py
    "EncoderOutput": (".encoders", "EncoderOutput"),
    "ModalityEncoder": (".encoders", "ModalityEncoder"),
    "VisionEncoder": (".encoders", "VisionEncoder"),
    "AudioEncoder": (".encoders", "AudioEncoder"),
    "MultimodalEncoder": (".encoders", "MultimodalEncoder"),
    "RepetitionAwareEncoder": (".encoders", "RepetitionAwareEncoder"),
    # processors.py
    "ModalityData": (".processors", "ModalityData"),
    "MultimodalRepetitionProcessor": (".processors", "MultimodalRepetitionProcessor"),
    "VisionPromptProcessor": (".processors", "VisionPromptProcessor"),
    "AudioPromptProcessor": (".processors", "AudioPromptProcessor"),
    "MultimodalFusionPipeline": (".processors", "MultimodalFusionPipeline"),
    # reasoning.py
    "ReasoningLevel": (".reasoning", "ReasoningLevel"),
    "ReasoningWrapper": (".reasoning", "ReasoningWrapper"),
    # tools.py
    "Tool": (".tools", "Tool"),
    "ToolExecutor": (".tools", "ToolExecutor"),
    "get_default_executor": (".tools", "get_default_executor"),
    # detect_modalities.py
    "detect_modalities": (".detect_modalities", "detect_modalities"),
    "format_report": (".detect_modalities", "format_report"),
    # connectors sub-package
    "DFMConnector": (".connectors", "DFMConnector"),
    "OptimalTransport": (".connectors", "OptimalTransport"),
    "FlowMatchingBlock": (".connectors", "FlowMatchingBlock"),
    # datasets sub-package
    "EMM1Dataset": (".datasets", "EMM1Dataset"),
    "emm1_collate_fn": (".datasets", "emm1_collate_fn"),
    "UnifiedMultiDatasetLoader": (".datasets", "UnifiedMultiDatasetLoader"),
    # decoders.py — multimodal content decoders
    "ContentDecoder": (".decoders", "ContentDecoder"),
    "ImageDecoder": (".decoders", "ImageDecoder"),
    "AudioDecoder": (".decoders", "AudioDecoder"),
    "MultimodalVideoDecoder": (
        ".decoders",
        "VideoDecoder",
    ),  # aliased: models.decoders also has VideoDecoder
    "TextDecoder": (".decoders", "TextDecoder"),
    "MultiModalDecoder": (".decoders", "MultiModalDecoder"),
}

# Submodules whose imports should trigger the torch __spec__ patch
_TORCH_SUBMODULES = {
    ".model",
    ".encoders",
    ".processors",
    ".reasoning",
    ".connectors",
    ".datasets",
    ".decoders",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # Patch torch.__spec__ before loading torch-dependent submodules
        if module_path in _TORCH_SUBMODULES:
            _patch_torch_spec()
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = [
    # Download utilities
    "download_vision_data",
    "download_audio_data",
    "download_video_data",
    "get_test_prompts",
    "MultimodalDataProcessor",
    # Model
    "OmniMultimodalLM",
    # Encoders
    "EncoderOutput",
    "ModalityEncoder",
    "VisionEncoder",
    "AudioEncoder",
    "MultimodalEncoder",
    "RepetitionAwareEncoder",
    # Processors
    "ModalityData",
    "MultimodalRepetitionProcessor",
    "VisionPromptProcessor",
    "AudioPromptProcessor",
    "MultimodalFusionPipeline",
    # Reasoning
    "ReasoningLevel",
    "ReasoningWrapper",
    # Tools
    "Tool",
    "ToolExecutor",
    "get_default_executor",
    # Detection
    "detect_modalities",
    "format_report",
    # Connectors sub-package
    "DFMConnector",
    "OptimalTransport",
    "FlowMatchingBlock",
    # Datasets sub-package
    "EMM1Dataset",
    "emm1_collate_fn",
    "UnifiedMultiDatasetLoader",
    # Decoders
    "ContentDecoder",
    "ImageDecoder",
    "AudioDecoder",
    "MultimodalVideoDecoder",
    "TextDecoder",
    "MultiModalDecoder",
]
