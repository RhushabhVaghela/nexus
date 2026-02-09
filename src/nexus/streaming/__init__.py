"""Streaming subsystem — memory, vision, TTS, and joint orchestration."""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "StreamingMemory": (".memory", "StreamingMemory"),
    "VisionStreamBuffer": (".vision", "VisionStreamBuffer"),
    "TTSStreamer": (".tts", "TTSStreamer"),
    "JointStreamingOrchestrator": (".joint", "JointStreamingOrchestrator"),
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
    "StreamingMemory",
    "VisionStreamBuffer",
    "TTSStreamer",
    "JointStreamingOrchestrator",
]
