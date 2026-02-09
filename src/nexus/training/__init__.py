"""
Nexus Training — training methods, controllers, and fine-tuning pipelines.

Includes DPO, ORPO, PPO training loops, dataset preparation, and
safe training controllers with thermal management.
"""

import importlib as _importlib

# ---------------------------------------------------------------------------
# Lazy imports — no submodule is loaded until an attribute is first accessed.
# ---------------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # training_methods.py
    "TrainingMethod": (".training_methods", "TrainingMethod"),
    "TrainingMethodConfig": (".training_methods", "TrainingMethodConfig"),
    "get_training_config": (".training_methods", "get_training_config"),
    "get_all_methods": (".training_methods", "get_all_methods"),
    "parse_training_method": (".training_methods", "parse_training_method"),
    # training_controller.py
    "setup_signal_handlers": (".training_controller", "setup_signal_handlers"),
    "training_step_hook": (".training_controller", "training_step_hook"),
    "check_and_cooldown": (".training_controller", "check_and_cooldown"),
    "save_emergency_checkpoint": (".training_controller", "save_emergency_checkpoint"),
    # process_manual_datasets.py — multimodal dataset processing
    "MultimodalSample": (".process_manual_datasets", "MultimodalSample"),
    "ManualDatasetProcessor": (".process_manual_datasets", "ManualDatasetProcessor"),
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
    # Training methods
    "TrainingMethod",
    "TrainingMethodConfig",
    "get_training_config",
    "get_all_methods",
    "parse_training_method",
    # Training controller
    "setup_signal_handlers",
    "training_step_hook",
    "check_and_cooldown",
    "save_emergency_checkpoint",
    # Dataset processing
    "MultimodalSample",
    "ManualDatasetProcessor",
]
