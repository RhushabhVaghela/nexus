"""
src/config/validator.py

Configuration validation module with JSON Schema validation, type checking,
range validation, dependency validation, and helpful error messages.

Supports validating configs/*.yaml and config/*.yaml files.
"""

import os
import re
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, message: str, path: str = "", suggestions: List[str] = None):
        self.path = path
        self.suggestions = suggestions or []
        super().__init__(message)
    
    def __str__(self) -> str:
        msg = f"Validation Error"
        if self.path:
            msg += f" at '{self.path}'"
        msg += f": {super().__str__()}"
        if self.suggestions:
            msg += f"\n  Suggestions:\n"
            for suggestion in self.suggestions:
                msg += f"    - {suggestion}"
        return msg


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, message: str, path: str = "", suggestions: List[str] = None):
        """Add an error to the result."""
        self.errors.append(ValidationError(message, path, suggestions))
        self.is_valid = False
    
    def add_warning(self, message: str, path: str = "", suggestions: List[str] = None):
        """Add a warning to the result."""
        self.warnings.append(ValidationError(message, path, suggestions))
    
    def merge(self, other: 'ValidationResult', path_prefix: str = ""):
        """Merge another validation result into this one."""
        for error in other.errors:
            full_path = f"{path_prefix}.{error.path}" if path_prefix else error.path
            self.add_error(str(error), full_path, error.suggestions)
        for warning in other.warnings:
            full_path = f"{path_prefix}.{warning.path}" if path_prefix else warning.path
            self.add_warning(str(warning), full_path, warning.suggestions)


class TypeValidator:
    """Validates Python types with helpful error messages."""
    
    VALID_TYPES = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'none': type(None)
    }
    
    @classmethod
    def validate(cls, value: Any, expected_type: Union[str, type], path: str = "") -> ValidationResult:
        """Validate that a value matches the expected type."""
        result = ValidationResult(is_valid=True)
        
        if isinstance(expected_type, str):
            expected_type = cls.VALID_TYPES.get(expected_type.lower())
            if expected_type is None:
                result.add_error(
                    f"Unknown type '{expected_type}'",
                    path,
                    [f"Valid types are: {', '.join(cls.VALID_TYPES.keys())}"]
                )
                return result
        
        if value is None and expected_type != type(None):
            result.add_error(
                f"Expected {expected_type.__name__}, got None",
                path,
                [f"Provide a valid {expected_type.__name__} value"]
            )
            return result
        
        if not isinstance(value, expected_type):
            actual_type = type(value).__name__
            result.add_error(
                f"Expected {expected_type.__name__}, got {actual_type}",
                path,
                [
                    f"Convert value to {expected_type.__name__}",
                    f"Check if the value is defined correctly in your config"
                ]
            )
        
        return result
    
    @classmethod
    def validate_list_item_types(cls, value: List[Any], item_type: Union[str, type], 
                                  path: str = "") -> ValidationResult:
        """Validate that all items in a list are of the expected type."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, list):
            result.add_error(f"Expected list, got {type(value).__name__}", path)
            return result
        
        for i, item in enumerate(value):
            item_result = cls.validate(item, item_type, f"{path}[{i}]")
            result.merge(item_result)
        
        return result
    
    @classmethod
    def validate_dict_value_types(cls, value: Dict[str, Any], value_type: Union[str, type],
                                   path: str = "") -> ValidationResult:
        """Validate that all values in a dict are of the expected type."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, dict):
            result.add_error(f"Expected dict, got {type(value).__name__}", path)
            return result
        
        for key, val in value.items():
            item_result = cls.validate(val, value_type, f"{path}.{key}")
            result.merge(item_result)
        
        return result


class RangeValidator:
    """Validates numeric ranges with helpful error messages."""
    
    @staticmethod
    def validate(value: Union[int, float], 
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 path: str = "") -> ValidationResult:
        """Validate that a numeric value is within the specified range."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, (int, float)):
            result.add_error(
                f"Expected numeric value for range validation, got {type(value).__name__}",
                path
            )
            return result
        
        if min_value is not None and value < min_value:
            result.add_error(
                f"Value {value} is below minimum {min_value}",
                path,
                [f"Set value to at least {min_value}", f"Remove the minimum constraint if not needed"]
            )
        
        if max_value is not None and value > max_value:
            result.add_error(
                f"Value {value} exceeds maximum {max_value}",
                path,
                [f"Set value to at most {max_value}", f"Remove the maximum constraint if not needed"]
            )
        
        return result
    
    @staticmethod
    def validate_length(value: Union[str, List, Dict], 
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        path: str = "") -> ValidationResult:
        """Validate the length of a string, list, or dict."""
        result = ValidationResult(is_valid=True)
        
        try:
            length = len(value)
        except TypeError:
            result.add_error(
                f"Cannot determine length of {type(value).__name__}",
                path
            )
            return result
        
        if min_length is not None and length < min_length:
            result.add_error(
                f"Length {length} is below minimum {min_length}",
                path,
                [f"Add more items to meet the minimum length requirement"]
            )
        
        if max_length is not None and length > max_length:
            result.add_error(
                f"Length {length} exceeds maximum {max_length}",
                path,
                [f"Remove items to meet the maximum length requirement"]
            )
        
        return result


class PatternValidator:
    """Validates string patterns with regex."""
    
    COMMON_PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^https?://[^\s/$.?#].[^\s]*$',
        'ip': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'semver': r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$',
        'path': r'^[\w\-./\\]+$',
        'identifier': r'^[a-zA-Z_][a-zA-Z0-9_]*$',
        'gpu_id': r'^cuda(:\d+)?$|^cpu$',
        'memory_size': r'^\d+(\.\d+)?\s*(GB|MB|KB|B|gb|mb|kb|b)$'
    }
    
    @classmethod
    def validate(cls, value: str, pattern: Union[str, re.Pattern], 
                 path: str = "") -> ValidationResult:
        """Validate that a string matches a regex pattern."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, str):
            result.add_error(
                f"Expected string for pattern validation, got {type(value).__name__}",
                path
            )
            return result
        
        if isinstance(pattern, str):
            if pattern in cls.COMMON_PATTERNS:
                pattern = cls.COMMON_PATTERNS[pattern]
            try:
                pattern = re.compile(pattern)
            except re.error as e:
                result.add_error(f"Invalid regex pattern: {e}", path)
                return result
        
        if not pattern.match(value):
            pattern_name = pattern.pattern if hasattr(pattern, 'pattern') else str(pattern)
            result.add_error(
                f"Value '{value}' does not match pattern '{pattern_name}'",
                path,
                [f"Check the format of your value", f"Refer to documentation for valid formats"]
            )
        
        return result


class DependencyValidator:
    """Validates configuration dependencies between fields."""
    
    @staticmethod
    def validate_required_if(config: Dict[str, Any], 
                             field: str, 
                             condition_field: str, 
                             condition_value: Any,
                             path: str = "") -> ValidationResult:
        """Validate that a field is present if a condition is met."""
        result = ValidationResult(is_valid=True)
        
        condition_path = f"{path}.{condition_field}" if path else condition_field
        field_path = f"{path}.{field}" if path else field
        
        actual_value = config.get(condition_field)
        if actual_value == condition_value:
            if field not in config or config[field] is None:
                result.add_error(
                    f"Field '{field}' is required when '{condition_field}' is '{condition_value}'",
                    field_path,
                    [f"Add '{field}' to your configuration", 
                     f"Change '{condition_field}' to a different value"]
                )
        
        return result
    
    @staticmethod
    def validate_incompatible(config: Dict[str, Any],
                             field1: str, 
                             field2: str,
                             path: str = "") -> ValidationResult:
        """Validate that two fields are not both set (mutually exclusive)."""
        result = ValidationResult(is_valid=True)
        
        if field1 in config and field2 in config:
            if config[field1] is not None and config[field2] is not None:
                result.add_error(
                    f"Fields '{field1}' and '{field2}' are mutually exclusive",
                    path,
                    [f"Remove '{field1}'", f"Remove '{field2}'"]
                )
        
        return result
    
    @staticmethod
    def validate_required_together(config: Dict[str, Any],
                                   fields: List[str],
                                   path: str = "") -> ValidationResult:
        """Validate that either all fields are set or none are."""
        result = ValidationResult(is_valid=True)
        
        present = [f for f in fields if f in config and config[f] is not None]
        
        if 0 < len(present) < len(fields):
            missing = set(fields) - set(present)
            result.add_error(
                f"Fields must all be set together. Missing: {', '.join(missing)}",
                path,
                [f"Add the missing fields", f"Remove all of these fields: {', '.join(present)}"]
            )
        
        return result


class HardwareValidator:
    """Validates hardware-related configuration dependencies."""
    
    @staticmethod
    def validate_gpu_settings(config: Dict[str, Any], path: str = "") -> ValidationResult:
        """Validate GPU-related settings and dependencies."""
        result = ValidationResult(is_valid=True)
        
        try:
            import torch
        except ImportError:
            result.add_warning(
                "Cannot validate GPU settings: torch not available",
                path
            )
            return result
        
        # Check if GPU is requested but CUDA is not available
        device = config.get('device', '')
        use_gpu = config.get('use_gpu', False)
        gpu_enabled = config.get('gpu_enabled', False)
        
        gpu_requested = (
            (isinstance(device, str) and 'cuda' in device.lower()) or
            use_gpu or gpu_enabled
        )
        
        if gpu_requested and not torch.cuda.is_available():
            result.add_error(
                "GPU/CUDA is requested but not available on this system",
                f"{path}.device" if path else "device",
                [
                    "Set device to 'cpu'",
                    "Install CUDA drivers and PyTorch with CUDA support",
                    "Check that your GPU is properly configured"
                ]
            )
        
        # Validate GPU device index
        if isinstance(device, str) and device.startswith('cuda:'):
            try:
                gpu_id = int(device.split(':')[1])
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    if gpu_id >= gpu_count:
                        result.add_error(
                            f"GPU device index {gpu_id} exceeds available GPUs ({gpu_count})",
                            f"{path}.device" if path else "device",
                            [f"Use a GPU index from 0 to {gpu_count - 1}", "Use 'cuda' without index for automatic selection"]
                        )
            except (ValueError, IndexError):
                result.add_error(
                    f"Invalid GPU device format: '{device}'",
                    f"{path}.device" if path else "device",
                    ["Use format 'cuda:0', 'cuda:1', etc."]
                )
        
        return result
    
    @staticmethod
    def validate_memory_settings(config: Dict[str, Any], path: str = "") -> ValidationResult:
        """Validate memory-related settings."""
        result = ValidationResult(is_valid=True)
        
        max_memory = config.get('max_memory_gb')
        vram_limit = config.get('vram_limit_gb')
        
        if max_memory is not None and max_memory < 1:
            result.add_warning(
                f"Maximum memory ({max_memory}GB) is very low and may cause issues",
                f"{path}.max_memory_gb" if path else "max_memory_gb"
            )
        
        if vram_limit is not None and vram_limit < 1:
            result.add_warning(
                f"VRAM limit ({vram_limit}GB) is very low and may cause OOM errors",
                f"{path}.vram_limit_gb" if path else "vram_limit_gb"
            )
        
        return result


class SchemaValidator:
    """JSON Schema-like validation for configurations."""
    
    def __init__(self, schema: Dict[str, Any]):
        """Initialize with a validation schema."""
        self.schema = schema
    
    def validate(self, config: Dict[str, Any], path: str = "") -> ValidationResult:
        """Validate a configuration against the schema."""
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        required = self.schema.get('required', [])
        for field in required:
            if field not in config or config[field] is None:
                result.add_error(
                    f"Required field '{field}' is missing",
                    f"{path}.{field}" if path else field,
                    [f"Add '{field}' to your configuration"]
                )
        
        # Validate properties
        properties = self.schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in config and config[field] is not None:
                field_result = self._validate_field(
                    config[field], field_schema, 
                    f"{path}.{field}" if path else field
                )
                result.merge(field_result)
        
        # Validate additional properties
        additional_properties = self.schema.get('additionalProperties', True)
        if not additional_properties:
            allowed = set(properties.keys())
            actual = set(config.keys())
            extra = actual - allowed
            if extra:
                result.add_warning(
                    f"Unexpected fields: {', '.join(extra)}",
                    path,
                    [f"Remove these fields", f"Set additionalProperties to true if they are intentional"]
                )
        
        # Validate dependencies
        dependencies = self.schema.get('dependencies', {})
        for dep_field, dep_config in dependencies.items():
            if dep_field in config and config[dep_field] is not None:
                if isinstance(dep_config, list):
                    # Simple dependency: these fields must be present
                    for required_field in dep_config:
                        if required_field not in config or config[required_field] is None:
                            result.add_error(
                                f"Field '{required_field}' is required when '{dep_field}' is present",
                                f"{path}.{required_field}" if path else required_field
                            )
                elif isinstance(dep_config, dict):
                    # Conditional dependency
                    dep_result = self.validate(config, path)
                    result.merge(dep_result)
        
        return result
    
    def _validate_field(self, value: Any, field_schema: Dict[str, Any], 
                        path: str) -> ValidationResult:
        """Validate a single field against its schema."""
        result = ValidationResult(is_valid=True)
        
        # Type validation
        if 'type' in field_schema:
            type_result = TypeValidator.validate(value, field_schema['type'], path)
            result.merge(type_result)
        
        # Range validation for numbers
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            range_result = RangeValidator.validate(
                value,
                min_value=field_schema.get('minimum'),
                max_value=field_schema.get('maximum'),
                path=path
            )
            result.merge(range_result)
        
        # Length validation for strings/lists/dicts
        if isinstance(value, (str, list, dict)):
            length_result = RangeValidator.validate_length(
                value,
                min_length=field_schema.get('minLength') or field_schema.get('minItems'),
                max_length=field_schema.get('maxLength') or field_schema.get('maxItems'),
                path=path
            )
            result.merge(length_result)
        
        # Pattern validation for strings
        if isinstance(value, str) and 'pattern' in field_schema:
            pattern_result = PatternValidator.validate(
                value, field_schema['pattern'], path
            )
            result.merge(pattern_result)
        
        # Enum validation
        if 'enum' in field_schema:
            if value not in field_schema['enum']:
                result.add_error(
                    f"Value '{value}' is not in allowed values: {field_schema['enum']}",
                    path,
                    [f"Use one of: {', '.join(map(str, field_schema['enum']))}"]
                )
        
        # Nested object validation
        if isinstance(value, dict) and 'properties' in field_schema:
            nested_schema = SchemaValidator(field_schema)
            nested_result = nested_schema.validate(value, path)
            result.merge(nested_result)
        
        # Array item validation
        if isinstance(value, list) and 'items' in field_schema:
            for i, item in enumerate(value):
                item_result = self._validate_field(
                    item, field_schema['items'], f"{path}[{i}]"
                )
                result.merge(item_result)
        
        return result


class ConfigValidator:
    """Main configuration validator for Nexus."""
    
    # Common schemas for Nexus configurations
    COMMON_SCHEMAS = {
        'model': {
            'type': 'object',
            'required': ['name', 'type'],
            'properties': {
                'name': {'type': 'str'},
                'type': {'type': 'str', 'enum': ['text', 'vision', 'audio', 'multimodal']},
                'path': {'type': 'str'},
                'device': {'type': 'str', 'pattern': 'gpu_id'},
                'quantization': {'type': 'str', 'enum': ['none', 'int8', 'int4', 'nf4', 'fp16', 'bf16']},
                'trust_remote_code': {'type': 'bool'}
            }
        },
        'training': {
            'type': 'object',
            'required': ['batch_size', 'learning_rate'],
            'properties': {
                'batch_size': {'type': 'int', 'minimum': 1, 'maximum': 1024},
                'learning_rate': {'type': 'float', 'minimum': 1e-7, 'maximum': 1.0},
                'epochs': {'type': 'int', 'minimum': 1},
                'max_steps': {'type': 'int', 'minimum': 1},
                'warmup_steps': {'type': 'int', 'minimum': 0},
                'grad_accum_steps': {'type': 'int', 'minimum': 1},
                'seed': {'type': 'int'}
            }
        },
        'hardware': {
            'type': 'object',
            'properties': {
                'device': {'type': 'str', 'pattern': 'gpu_id'},
                'use_gpu': {'type': 'bool'},
                'gpu_enabled': {'type': 'bool'},
                'max_memory_gb': {'type': 'float', 'minimum': 0.1},
                'vram_limit_gb': {'type': 'float', 'minimum': 0.1},
                'num_workers': {'type': 'int', 'minimum': 1, 'maximum': 64}
            }
        },
        'logging': {
            'type': 'object',
            'properties': {
                'level': {'type': 'str', 'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']},
                'format': {'type': 'str', 'enum': ['text', 'json']},
                'file': {'type': 'str'},
                'max_bytes': {'type': 'int', 'minimum': 1024},
                'backup_count': {'type': 'int', 'minimum': 0, 'maximum': 100}
            }
        },
        'cache': {
            'type': 'object',
            'properties': {
                'enabled': {'type': 'bool'},
                'max_size': {'type': 'int', 'minimum': 1},
                'ttl': {'type': 'int', 'minimum': 1},
                'eviction_policy': {'type': 'str', 'enum': ['lru', 'lfu', 'ttl', 'size', 'priority']}
            }
        }
    }
    
    def __init__(self):
        """Initialize the validator."""
        self.schemas = dict(self.COMMON_SCHEMAS)
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a custom schema."""
        self.schemas[name] = schema
    
    def validate_file(self, file_path: Union[str, Path], 
                      schema_name: Optional[str] = None) -> ValidationResult:
        """Validate a configuration file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            result = ValidationResult(is_valid=False)
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Load the configuration
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    config = json.load(f)
                else:
                    result = ValidationResult(is_valid=False)
                    result.add_error(f"Unsupported file format: {file_path.suffix}")
                    return result
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_error(f"Failed to load config file: {e}")
            return result
        
        # Validate against schema if specified
        if schema_name and schema_name in self.schemas:
            schema = self.schemas[schema_name]
            validator = SchemaValidator(schema)
            return validator.validate(config)
        
        # Auto-detect schema based on content
        return self._auto_validate(config, file_path)
    
    def _auto_validate(self, config: Dict[str, Any], 
                       file_path: Path) -> ValidationResult:
        """Auto-detect and validate configuration based on content."""
        result = ValidationResult(is_valid=True)
        
        # Validate hardware settings
        if any(k in config for k in ['device', 'use_gpu', 'gpu_enabled', 'vram_limit_gb']):
            hw_result = HardwareValidator.validate_gpu_settings(config)
            hw_memory_result = HardwareValidator.validate_memory_settings(config)
            result.merge(hw_result)
            result.merge(hw_memory_result)
        
        # Validate nested sections
        if 'model' in config and isinstance(config['model'], dict):
            if 'model' in self.schemas:
                schema = SchemaValidator(self.schemas['model'])
                model_result = schema.validate(config['model'], 'model')
                result.merge(model_result)
        
        if 'training' in config and isinstance(config['training'], dict):
            if 'training' in self.schemas:
                schema = SchemaValidator(self.schemas['training'])
                training_result = schema.validate(config['training'], 'training')
                result.merge(training_result)
        
        if 'hardware' in config and isinstance(config['hardware'], dict):
            if 'hardware' in self.schemas:
                schema = SchemaValidator(self.schemas['hardware'])
                hw_result = schema.validate(config['hardware'], 'hardware')
                result.merge(hw_result)
            # Also run hardware dependency validation
            hw_dep_result = HardwareValidator.validate_gpu_settings(
                config['hardware'], 'hardware'
            )
            result.merge(hw_dep_result)
        
        if 'logging' in config and isinstance(config['logging'], dict):
            if 'logging' in self.schemas:
                schema = SchemaValidator(self.schemas['logging'])
                logging_result = schema.validate(config['logging'], 'logging')
                result.merge(logging_result)
        
        if 'cache' in config and isinstance(config['cache'], dict):
            if 'cache' in self.schemas:
                schema = SchemaValidator(self.schemas['cache'])
                cache_result = schema.validate(config['cache'], 'cache')
                result.merge(cache_result)
        
        return result
    
    def validate_directory(self, directory: Union[str, Path],
                           pattern: str = "*.yaml") -> Dict[str, ValidationResult]:
        """Validate all configuration files in a directory."""
        directory = Path(directory)
        results = {}
        
        if not directory.exists():
            return {str(directory): ValidationResult(is_valid=False, 
                     errors=[ValidationError(f"Directory not found: {directory}")])}
        
        for config_file in directory.glob(pattern):
            results[str(config_file)] = self.validate_file(config_file)
        
        return results
    
    def validate_all_nexus_configs(self, root_dir: Union[str, Path] = ".") -> Dict[str, ValidationResult]:
        """Validate all Nexus configuration files."""
        root_dir = Path(root_dir)
        all_results = {}
        
        # Validate config/*.yaml
        config_dir = root_dir / "config"
        if config_dir.exists():
            results = self.validate_directory(config_dir, "*.yaml")
            all_results.update({f"config/{k}": v for k, v in results.items()})
        
        # Validate configs/*.yaml
        configs_dir = root_dir / "configs"
        if configs_dir.exists():
            results = self.validate_directory(configs_dir, "*.yaml")
            all_results.update({f"configs/{k}": v for k, v in results.items()})
        
        return all_results


def validate_config(config: Dict[str, Any], 
                    schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Convenience function to validate a configuration."""
    if schema:
        validator = SchemaValidator(schema)
        return validator.validate(config)
    
    validator = ConfigValidator()
    return validator._auto_validate(config, Path("config"))


def validate_config_file(file_path: Union[str, Path],
                         schema_name: Optional[str] = None) -> ValidationResult:
    """Convenience function to validate a configuration file."""
    validator = ConfigValidator()
    return validator.validate_file(file_path, schema_name)


def main():
    """CLI entry point for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Nexus configuration files"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to config file or directory to validate"
    )
    parser.add_argument(
        "--schema",
        type=str,
        choices=list(ConfigValidator.COMMON_SCHEMAS.keys()),
        help="Schema to validate against"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all Nexus config files (config/* and configs/*)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.all:
        results = validator.validate_all_nexus_configs(args.path)
        
        total_errors = 0
        total_warnings = 0
        
        for path, result in results.items():
            if result.errors or result.warnings:
                print(f"\n{'=' * 60}")
                print(f"Config: {path}")
                print('=' * 60)
                
                for error in result.errors:
                    print(f"  [ERROR] {error}")
                    total_errors += 1
                
                for warning in result.warnings:
                    print(f"  [WARNING] {warning}")
                    total_warnings += 1
        
        print(f"\n{'=' * 60}")
        print(f"Validation Complete: {len(results)} files checked")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")
        
        if total_errors > 0 or (args.strict and total_warnings > 0):
            exit(1)
        exit(0)
    
    else:
        path = Path(args.path)
        
        if path.is_file():
            result = validator.validate_file(path, args.schema)
        elif path.is_dir():
            results = validator.validate_directory(path)
            result = ValidationResult(is_valid=True)
            for r in results.values():
                result.merge(r)
        else:
            print(f"Error: Path not found: {path}")
            exit(1)
        
        print(f"\n{'=' * 60}")
        print(f"Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
        print('=' * 60)
        
        for error in result.errors:
            print(f"  [ERROR] {error}")
        
        for warning in result.warnings:
            print(f"  [WARNING] {warning}")
        
        if not result.errors and not result.warnings:
            print("  ✓ Configuration is valid")
        
        exit(0 if result.is_valid and not (args.strict and result.warnings) else 1)


if __name__ == "__main__":
    main()
