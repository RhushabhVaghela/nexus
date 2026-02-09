"""
Pipeline orchestration for chaining multiple training stages.

Provides a comprehensive pipeline orchestration system that chains
stages sequentially (e.g., Reasoning → Tools → Vision), manages
inter-stage data flow and checkpoints, handles stage dependencies,
and supports rollback/continue/stop failure modes.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FailureMode": (".pipeline_orchestrator", "FailureMode"),
    "StageStatus": (".pipeline_orchestrator", "StageStatus"),
    "PipelineConfig": (".pipeline_orchestrator", "PipelineConfig"),
    "CheckpointInfo": (".pipeline_orchestrator", "CheckpointInfo"),
    "PipelineError": (".pipeline_orchestrator", "PipelineError"),
    "PipelineProgress": (".pipeline_orchestrator", "PipelineProgress"),
    "PipelineResult": (".pipeline_orchestrator", "PipelineResult"),
    "CheckpointManager": (".pipeline_orchestrator", "CheckpointManager"),
    "PipelineOrchestrator": (".pipeline_orchestrator", "PipelineOrchestrator"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FailureMode",
    "StageStatus",
    "PipelineConfig",
    "CheckpointInfo",
    "PipelineError",
    "PipelineProgress",
    "PipelineResult",
    "CheckpointManager",
    "PipelineOrchestrator",
]
