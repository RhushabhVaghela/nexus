"""
Custom exceptions for the Universal SLI module.
"""


class SLIError(Exception):
    """Base exception for all SLI-related errors."""
    pass


class UnsupportedArchitectureError(SLIError):
    """Raised when an unsupported architecture is encountered."""
    
    def __init__(self, model_type: str, architectures: list = None):
        self.model_type = model_type
        self.architectures = architectures or []
        msg = f"Unsupported architecture: model_type='{model_type}'"
        if architectures:
            msg += f", architectures={architectures}"
        msg += ". This architecture family is not yet supported by Universal SLI."
        super().__init__(msg)


class WeightLoadingError(SLIError):
    """Raised when weight loading fails."""
    
    def __init__(self, weight_name: str, shard_name: str = None, cause: Exception = None):
        self.weight_name = weight_name
        self.shard_name = shard_name
        self.cause = cause
        msg = f"Failed to load weight: {weight_name}"
        if shard_name:
            msg += f" from shard: {shard_name}"
        if cause:
            msg += f". Cause: {str(cause)}"
        super().__init__(msg)


class LayerCreationError(SLIError):
    """Raised when layer creation fails."""
    
    def __init__(self, layer_idx: int, family_id: str, cause: Exception = None):
        self.layer_idx = layer_idx
        self.family_id = family_id
        self.cause = cause
        msg = f"Failed to create layer {layer_idx} for family '{family_id}'"
        if cause:
            msg += f". Cause: {str(cause)}"
        super().__init__(msg)


class MoEConfigurationError(SLIError):
    """Raised when MoE configuration is invalid or unsupported."""
    
    def __init__(self, moe_type: str, message: str = None):
        self.moe_type = moe_type
        msg = message or f"Invalid or unsupported MoE configuration: {moe_type}"
        super().__init__(msg)


class FormatDetectionError(SLIError):
    """Raised when weight format cannot be detected."""
    
    def __init__(self, model_id: str, attempted_formats: list = None):
        self.model_id = model_id
        self.attempted_formats = attempted_formats or []
        msg = f"Could not detect weight format for model: {model_id}"
        if attempted_formats:
            msg += f". Attempted formats: {attempted_formats}"
        super().__init__(msg)


class WeightMapError(SLIError):
    """Raised when weight map is missing or invalid."""
    
    def __init__(self, model_id: str, index_file: str = None):
        self.model_id = model_id
        self.index_file = index_file
        msg = f"Weight map error for model: {model_id}"
        if index_file:
            msg += f". Index file: {index_file}"
        super().__init__(msg)
