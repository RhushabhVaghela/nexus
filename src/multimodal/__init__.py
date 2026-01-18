from .download import download_vision_data, download_audio_data, download_video_data, get_test_prompts
from .distillation import MultimodalDataProcessor

# Lazy import for Model to allow running download/processing without Torch
def __getattr__(name):
    if name == "OmniMultimodalLM":
        from .model import OmniMultimodalLM
        return OmniMultimodalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
