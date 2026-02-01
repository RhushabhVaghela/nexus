"""
Quantization Module for Nexus SLI (Selective Layer Inference)

This module implements quantization support for efficient layer storage and loading:
- 8-bit (INT8) quantization using bitsandbytes
- 4-bit (NF4) quantization
- Dynamic quantization during layer loading
- De-quantization before processing
- Configurable per-layer quantization

Author: Nexus Team
"""

import logging
from typing import Dict, Optional, Any, Callable, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

logger = logging.getLogger(__name__)

# Try to import bitsandbytes for advanced quantization
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, Linear4bit, Params4bit
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    warnings.warn("bitsandbytes not available. 8-bit and 4-bit quantization will be limited to PyTorch native methods.")


class QuantizationMode(Enum):
    """Supported quantization modes."""
    NONE = "none"
    INT8 = "int8"  # 8-bit integer
    INT8_DYNAMIC = "int8_dynamic"  # PyTorch dynamic quantization
    NF4 = "nf4"  # 4-bit Normal Float
    FP4 = "fp4"  # 4-bit Float
    INT4 = "int4"  # 4-bit integer


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    mode: QuantizationMode = QuantizationMode.NONE
    compute_dtype: torch.dtype = torch.float16
    compress_statistics: bool = True  # For 4-bit quantization
    quant_storage_dtype: torch.dtype = torch.uint8
    double_quant: bool = True  # Nested quantization for 4-bit
    quant_type: str = "nf4"  # "nf4" or "fp4"
    llm_int8_threshold: float = 6.0  # Outlier threshold for 8-bit
    llm_int8_skip_modules: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.mode.value,
            'compute_dtype': str(self.compute_dtype),
            'compress_statistics': self.compress_statistics,
            'quant_storage_dtype': str(self.quant_storage_dtype),
            'double_quant': self.double_quant,
            'quant_type': self.quant_type,
            'llm_int8_threshold': self.llm_int8_threshold,
            'llm_int8_skip_modules': self.llm_int8_skip_modules or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationConfig':
        """Create config from dictionary."""
        config = cls()
        config.mode = QuantizationMode(data.get('mode', 'none'))
        config.compute_dtype = getattr(torch, data.get('compute_dtype', 'float16').split('.')[-1])
        config.compress_statistics = data.get('compress_statistics', True)
        config.double_quant = data.get('double_quant', True)
        config.quant_type = data.get('quant_type', 'nf4')
        config.llm_int8_threshold = data.get('llm_int8_threshold', 6.0)
        config.llm_int8_skip_modules = data.get('llm_int8_skip_modules')
        return config


class LayerQuantizer:
    """
    Handles quantization and de-quantization of model layers.
    
    Supports:
    - 8-bit quantization (bitsandbytes)
    - 4-bit quantization (bitsandbytes NF4/FP4)
    - PyTorch native dynamic quantization
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize the quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if self.config.mode in [QuantizationMode.INT8, QuantizationMode.NF4, QuantizationMode.FP4]:
            if not BITSANDBYTES_AVAILABLE:
                raise ImportError(
                    f"bitsandbytes is required for {self.config.mode.value} quantization. "
                    "Install with: pip install bitsandbytes"
                )

    def quantize_layer(self, layer: nn.Module, layer_name: str = "") -> nn.Module:
        """
        Quantize a layer according to the configured mode.

        Args:
            layer: The layer to quantize
            layer_name: Name of the layer for skip configuration

        Returns:
            Quantized layer
        """
        if self.config.mode == QuantizationMode.NONE:
            return layer
        
        # Check if layer should be skipped
        if self._should_skip_layer(layer_name):
            logger.debug(f"Skipping quantization for {layer_name}")
            return layer
        
        try:
            if self.config.mode == QuantizationMode.INT8:
                return self._quantize_int8(layer)
            elif self.config.mode == QuantizationMode.INT8_DYNAMIC:
                return self._quantize_int8_dynamic(layer)
            elif self.config.mode in [QuantizationMode.NF4, QuantizationMode.FP4, QuantizationMode.INT4]:
                return self._quantize_4bit(layer)
            else:
                logger.warning(f"Unknown quantization mode: {self.config.mode}")
                return layer
        except Exception as e:
            logger.error(f"Quantization failed for {layer_name}: {e}")
            return layer

    def _should_skip_layer(self, layer_name: str) -> bool:
        """Check if a layer should be skipped based on configuration."""
        if self.config.llm_int8_skip_modules is None:
            return False
        return any(skip in layer_name for skip in self.config.llm_int8_skip_modules)

    def _quantize_int8(self, layer: nn.Module) -> nn.Module:
        """
        Quantize layer to 8-bit using bitsandbytes.
        
        Replaces Linear layers with Linear8bitLt.
        """
        if not BITSANDBYTES_AVAILABLE:
            logger.warning("bitsandbytes not available, skipping 8-bit quantization")
            return layer
        
        if isinstance(layer, nn.Linear):
            # Create 8-bit linear layer
            quantized = Linear8bitLt(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                has_fp16_weights=False,
                threshold=self.config.llm_int8_threshold
            )
            
            # Copy weights
            quantized.weight.data = layer.weight.data
            if layer.bias is not None:
                quantized.bias.data = layer.bias.data
            
            # Move to device and quantize
            quantized = quantized.to(layer.weight.device)
            
            return quantized
        
        # Recursively quantize child modules
        for name, child in layer.named_children():
            setattr(layer, name, self._quantize_int8(child))
        
        return layer

    def _quantize_int8_dynamic(self, layer: nn.Module) -> nn.Module:
        """
        Apply PyTorch dynamic INT8 quantization.
        
        This is the most compatible method but may have lower performance.
        """
        # Dynamic quantization works on specific layer types
        quantized = quantize_dynamic(
            layer,
            {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell, nn.RNNCell},
            dtype=torch.qint8
        )
        return quantized

    def _quantize_4bit(self, layer: nn.Module) -> nn.Module:
        """
        Quantize layer to 4-bit using bitsandbytes (NF4 or FP4).
        
        Uses Normal Float 4 (NF4) which is optimal for normally distributed weights.
        """
        if not BITSANDBYTES_AVAILABLE:
            logger.warning("bitsandbytes not available, skipping 4-bit quantization")
            return layer
        
        if isinstance(layer, nn.Linear):
            # Create 4-bit linear layer
            compute_dtype = self.config.compute_dtype
            compress_statistics = self.config.compress_statistics
            
            quantized = Linear4bit(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=compress_statistics,
                quant_type=self.config.quant_type
            )
            
            # Copy and quantize weights
            device = layer.weight.device
            state = layer.state_dict()
            
            # Quantize the weights
            if self.config.quant_type == "nf4":
                # Use NF4 quantization
                quantized_weight = self._quantize_to_nf4(state['weight'])
            else:
                # Use FP4 quantization
                quantized_weight = self._quantize_to_fp4(state['weight'])
            
            quantized.weight = quantized_weight
            if layer.bias is not None:
                quantized.bias = nn.Parameter(state['bias'])
            
            return quantized.to(device)
        
        # Recursively quantize child modules
        for name, child in layer.named_children():
            setattr(layer, name, self._quantize_4bit(child))
        
        return layer

    def _quantize_to_nf4(self, weight: torch.Tensor) -> "Optional[Params4bit]":
        """Convert weight to NF4 quantized format."""
        if not BITSANDBYTES_AVAILABLE:
            return None
        params = Params4bit(
            weight.data,
            requires_grad=False,
            quant_storage=self.config.quant_storage_dtype
        )
        return params

    def _quantize_to_fp4(self, weight: torch.Tensor) -> "Optional[Params4bit]":
        """Convert weight to FP4 quantized format."""
        if not BITSANDBYTES_AVAILABLE:
            return None
        # FP4 uses different quantization constants
        params = Params4bit(
            weight.data,
            requires_grad=False,
            quant_storage=self.config.quant_storage_dtype,
            quant_type="fp4"
        )
        return params

    def dequantize_layer(self, layer: nn.Module) -> nn.Module:
        """
        De-quantize a layer back to full precision.

        Args:
            layer: The quantized layer

        Returns:
            De-quantized layer in full precision
        """
        if not self.is_quantized(layer):
            return layer
        
        try:
            if BITSANDBYTES_AVAILABLE:
                if isinstance(layer, (Linear8bitLt, Linear4bit)):
                    return self._dequantize_bitsandbytes_layer(layer)
            
            # For PyTorch native quantization, we need to convert back
            if hasattr(layer, 'weight') and hasattr(layer.weight, 'dequantize'):
                # This handles torch.quantization quantized tensors
                layer.weight = nn.Parameter(layer.weight.dequantize())
            
            return layer
        except Exception as e:
            logger.error(f"De-quantization failed: {e}")
            return layer

    def _dequantize_bitsandbytes_layer(self, layer: nn.Module) -> nn.Module:
        """De-quantize bitsandbytes quantized layer."""
        if isinstance(layer, (Linear8bitLt, Linear4bit)):
            # Create new full precision linear layer
            fp_layer = nn.Linear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                dtype=self.config.compute_dtype
            )
            
            # Get de-quantized weights
            weight_data = layer.weight.data
            if hasattr(weight_data, 'dequantize'):
                weight_data = weight_data.dequantize()
            
            fp_layer.weight.data = weight_data.to(self.config.compute_dtype)
            if layer.bias is not None:
                fp_layer.bias.data = layer.bias.data.to(self.config.compute_dtype)
            
            # Copy to same device
            device = layer.weight.device if hasattr(layer.weight, 'device') else 'cpu'
            fp_layer = fp_layer.to(device)
            
            return fp_layer
        
        return layer

    def is_quantized(self, layer: nn.Module) -> bool:
        """Check if a layer is quantized."""
        if BITSANDBYTES_AVAILABLE:
            if isinstance(layer, (Linear8bitLt, Linear4bit)):
                return True
        
        # Check for PyTorch quantized tensors
        if hasattr(layer, 'weight'):
            weight = layer.weight
            if hasattr(weight, 'dtype'):
                if 'quantized' in str(weight.dtype):
                    return True
        
        return False

    def get_quantized_size_ratio(self, layer: nn.Module) -> float:
        """
        Get the compression ratio of a quantized layer.
        
        Returns:
            Ratio of quantized size to original size (e.g., 0.25 for 4x compression)
        """
        if not self.is_quantized(layer):
            return 1.0
        
        mode_ratios = {
            QuantizationMode.NONE: 1.0,
            QuantizationMode.INT8: 0.5,
            QuantizationMode.INT8_DYNAMIC: 0.5,
            QuantizationMode.NF4: 0.25,
            QuantizationMode.FP4: 0.25,
            QuantizationMode.INT4: 0.25
        }
        
        return mode_ratios.get(self.config.mode, 1.0)


class AdaptiveQuantizer:
    """
    Adaptive quantization that adjusts precision based on layer importance.
    
    More important layers (e.g., attention) can use higher precision,
    while less important layers can use lower precision.
    """

    def __init__(
        self,
        base_config: QuantizationConfig,
        attention_config: Optional[QuantizationConfig] = None,
        ffn_config: Optional[QuantizationConfig] = None
    ):
        """
        Initialize adaptive quantizer.

        Args:
            base_config: Default quantization config
            attention_config: Config for attention layers (higher precision)
            ffn_config: Config for FFN layers (lower precision)
        """
        self.base_config = base_config
        self.attention_config = attention_config or self._higher_precision_config(base_config)
        self.ffn_config = ffn_config or base_config

    def _higher_precision_config(self, config: QuantizationConfig) -> QuantizationConfig:
        """Create a config with higher precision (less aggressive quantization)."""
        higher = QuantizationConfig.from_dict(config.to_dict())
        # Upgrade to less aggressive quantization
        if config.mode == QuantizationMode.NF4:
            higher.mode = QuantizationMode.INT8
        elif config.mode == QuantizationMode.INT8:
            higher.mode = QuantizationMode.INT8_DYNAMIC
        elif config.mode == QuantizationMode.INT8_DYNAMIC:
            higher.mode = QuantizationMode.NONE
        return higher

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize model with different settings for different layer types.

        Args:
            model: The model to quantize

        Returns:
            Quantized model
        """
        for name, module in model.named_modules():
            config = self._get_config_for_layer(name)
            quantizer = LayerQuantizer(config)
            
            # Get parent and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
                quantized = quantizer.quantize_layer(module, name)
                setattr(parent, child_name, quantized)
        
        return model

    def _get_config_for_layer(self, layer_name: str) -> QuantizationConfig:
        """Get appropriate quantization config for a layer."""
        layer_name_lower = layer_name.lower()
        
        # Attention layers get higher precision
        if any(x in layer_name_lower for x in ['attn', 'attention', 'query', 'key', 'value', 'qkv']):
            return self.attention_config
        
        # FFN layers can use lower precision
        if any(x in layer_name_lower for x in ['ffn', 'mlp', 'feedforward', 'fc']):
            return self.ffn_config
        
        return self.base_config


class QuantizationRegistry:
    """
    Registry for storing and retrieving quantization configurations.
    """
    
    _configs: Dict[str, QuantizationConfig] = {}
    
    @classmethod
    def register(cls, name: str, config: QuantizationConfig):
        """Register a quantization config."""
        cls._configs[name] = config
        logger.info(f"Registered quantization config: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[QuantizationConfig]:
        """Get a registered quantization config."""
        return cls._configs.get(name)
    
    @classmethod
    def list_configs(cls) -> List[str]:
        """List all registered config names."""
        return list(cls._configs.keys())


# Predefined configurations
def get_int8_config() -> QuantizationConfig:
    """Get standard 8-bit quantization config."""
    return QuantizationConfig(
        mode=QuantizationMode.INT8,
        compute_dtype=torch.float16,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head", "embed_tokens"]
    )


def get_nf4_config() -> QuantizationConfig:
    """Get standard NF4 (4-bit) quantization config."""
    return QuantizationConfig(
        mode=QuantizationMode.NF4,
        compute_dtype=torch.float16,
        compress_statistics=True,
        double_quant=True,
        quant_type="nf4"
    )


def get_fp4_config() -> QuantizationConfig:
    """Get standard FP4 (4-bit) quantization config."""
    return QuantizationConfig(
        mode=QuantizationMode.FP4,
        compute_dtype=torch.float16,
        compress_statistics=True,
        double_quant=True,
        quant_type="fp4"
    )


def get_mixed_precision_config() -> AdaptiveQuantizer:
    """Get mixed precision adaptive quantizer."""
    base = get_nf4_config()
    attention = get_int8_config()
    return AdaptiveQuantizer(
        base_config=base,
        attention_config=attention,
        ffn_config=base
    )


# Register predefined configs
QuantizationRegistry.register("int8", get_int8_config())
QuantizationRegistry.register("nf4", get_nf4_config())
QuantizationRegistry.register("fp4", get_fp4_config())


# Convenience functions
def quantize_layer(
    layer: nn.Module,
    mode: Union[str, QuantizationMode] = "int8",
    **kwargs
) -> nn.Module:
    """
    Convenience function to quantize a layer.

    Args:
        layer: Layer to quantize
        mode: Quantization mode
        **kwargs: Additional config options

    Returns:
        Quantized layer
    """
    if isinstance(mode, str):
        mode = QuantizationMode(mode)
    
    config = QuantizationConfig(mode=mode, **kwargs)
    quantizer = LayerQuantizer(config)
    return quantizer.quantize_layer(layer)


def dequantize_layer(layer: nn.Module, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Convenience function to de-quantize a layer.

    Args:
        layer: Quantized layer
        compute_dtype: Target dtype

    Returns:
        De-quantized layer
    """
    config = QuantizationConfig(compute_dtype=compute_dtype)
    quantizer = LayerQuantizer(config)
    return quantizer.dequantize_layer(layer)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Quantization Module")
    print("=" * 60)
    
    # Create a sample layer
    layer = nn.Linear(1024, 4096)
    original_size = layer.weight.numel() * layer.weight.element_size()
    print(f"Original layer size: {original_size / 1e6:.2f} MB")
    
    # Test INT8 dynamic quantization (works without bitsandbytes)
    print("\n1. INT8 Dynamic Quantization:")
    config = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
    quantizer = LayerQuantizer(config)
    int8_layer = quantizer.quantize_layer(layer)
    print(f"Is quantized: {quantizer.is_quantized(int8_layer)}")
    
    # De-quantize
    dequantized = quantizer.dequantize_layer(int8_layer)
    print(f"After de-quantization: {dequantized.weight.dtype}")
    
    # Show available configs
    print("\n2. Available Quantization Configs:")
    for name in QuantizationRegistry.list_configs():
        config = QuantizationRegistry.get(name)
        print(f"  - {name}: {config.mode.value}")
    
    print("\n" + "=" * 60)
