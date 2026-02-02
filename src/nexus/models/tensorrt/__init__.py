"""
TensorRT-LLM Backend for Nexus

High-performance inference backend using NVIDIA TensorRT-LLM.
Supports FP8, INT8, and INT4 quantization for optimal throughput.

Components:
- trt_engine.py: TensorRT engine wrapper
- model_converter.py: PyTorch to TensorRT conversion
- inference_backend.py: Unified interface

Author: Nexus Team
"""

from .trt_engine import (
    TRTEngine,
    TRTEngineConfig,
    TRTBuildConfig,
    TRTQuantizationMode,
    TRTEngineError,
)

from .model_converter import (
    ModelConverter,
    ConversionConfig,
    LayerConverter,
    convert_model,
)

from .inference_backend import (
    TensorRTBackend,
    TensorRTConfig,
    BackendError,
)

__all__ = [
    # Engine
    'TRTEngine',
    'TRTEngineConfig',
    'TRTBuildConfig',
    'TRTQuantizationMode',
    'TRTEngineError',
    
    # Converter
    'ModelConverter',
    'ConversionConfig',
    'LayerConverter',
    'convert_model',
    
    # Backend
    'TensorRTBackend',
    'TensorRTConfig',
    'BackendError',
]


def create_tensorrt_backend(
    model_path: str,
    quantization: str = "fp16",
    max_batch_size: int = 1,
    max_seq_length: int = 2048,
    **kwargs
) -> TensorRTBackend:
    """
    Create a TensorRT backend with preset configuration.
    
    Args:
        model_path: Path to model
        quantization: Quantization mode (fp32, fp16, bf16, fp8, int8, int4)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
        **kwargs: Additional config options
        
    Returns:
        Configured TensorRTBackend
    """
    from .inference_backend import TensorRTConfig
    
    config = TensorRTConfig(
        model_path=model_path,
        quantization_mode=quantization,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        **kwargs
    )
    
    return TensorRTBackend(config)
