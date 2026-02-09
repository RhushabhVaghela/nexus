# src/data/__init__.py
"""Data loading, organization, and streaming utilities."""

from .universal_loader import UniversalDataLoader, LoadResult

# ---------------------------------------------------------------------------
# Lazy imports for heavier data modules
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # niwt_loader.py — NIWT format data loading
    "NIWTDataLoader": (".niwt_loader", "NIWTDataLoader"),
    # organizer.py — dataset organization
    "NexusDataOrganizer": (".organizer", "NexusDataOrganizer"),
    # streaming_trainer.py — chunked/streaming data for training
    "ChunkConfig": (".streaming_trainer", "ChunkConfig"),
    "ChunkedSampleProcessor": (".streaming_trainer", "ChunkedSampleProcessor"),
    "StreamingConfig": (".streaming_trainer", "StreamingConfig"),
    "StreamingDatasetLoader": (".streaming_trainer", "StreamingDatasetLoader"),
    "load_streaming_datasets": (".streaming_trainer", "load_streaming_datasets"),
    # universal_manager.py — high-level dataset management
    "UniversalDatasetManager": (".universal_manager", "UniversalDatasetManager"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UniversalDataLoader",
    "LoadResult",
] + list(_LAZY_IMPORTS.keys())
