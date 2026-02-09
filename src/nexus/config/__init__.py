"""
Nexus Configuration — memory planning, validation, schema tools, and paths.
"""

from .memory_config import get_memory_config, get_device_map_stage1, print_memory_plan
from .paths import DATA_ROOT, MODELS_DIR, OUTPUT_DIR, DATASETS_DIR
from .validator import (
    ConfigValidator,
    ValidationError,
    ValidationLevel,
    ValidationResult,
    validate_config,
    validate_config_file,
)

__all__ = [
    # Paths (most commonly used — full set in nexus.config.paths)
    "DATA_ROOT",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "DATASETS_DIR",
    # Memory config
    "get_memory_config",
    "get_device_map_stage1",
    "print_memory_plan",
    # Validation
    "ConfigValidator",
    "ValidationError",
    "ValidationLevel",
    "ValidationResult",
    "validate_config",
    "validate_config_file",
]
