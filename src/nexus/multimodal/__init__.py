# Patch torch.__spec__ to fix datasets loading issues
import sys
import importlib.util
from importlib.machinery import ModuleSpec

# Check if torch is already imported or import it
import torch

try:
    if getattr(torch, '__spec__', None) is None:
        # Create a dummy spec
        dummy_spec = ModuleSpec(name="torch", loader=None, origin=getattr(torch, '__file__', 'unknown'))
        torch.__spec__ = dummy_spec
        sys.modules['torch'].__spec__ = dummy_spec
except Exception:
    pass

from .download import download_vision_data, download_audio_data, download_video_data, get_test_prompts
from .distillation import MultimodalDataProcessor

# Lazy import for Model to allow running download/processing without Torch
def __getattr__(name):
    if name == "OmniMultimodalLM":
        from .model import OmniMultimodalLM
        return OmniMultimodalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
