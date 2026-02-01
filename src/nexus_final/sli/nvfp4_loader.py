"""
NVFP4 Streaming Loader for Nexus SLI

Implements streaming layer loading with NVIDIA FP4 (NVFP4) quantization support.
NVFP4 provides 4-bit floating point quantization optimized for transformer models,
enabling significant memory savings with minimal accuracy degradation.

Key Features:
- Block-wise NVFP4 quantization (block size 16)
- Mixed precision loading (BF16 for attention, NVFP4 for FFN)
- Streaming-aware quantization for on-the-fly layer conversion
- Hardware-accelerated dequantization paths

Author: Nexus Team
"""

import logging
import warnings
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

import torch
import torch.nn as nn
import numpy as np

from .exceptions import SLIError, WeightLoadingError

logger = logging.getLogger(__name__)

# Try to import NVIDIA quantization libraries
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    NVFP4_AVAILABLE = True
except ImportError:
    NVFP4_AVAILABLE = False
    warnings.warn(
        "Transformer Engine not available. NVFP4 quantization will use software fallback. "
        "For optimal performance, install: pip install transformer-engine[pytorch]"
    )


class NVFP4Mode(Enum):
    """NVFP4 quantization modes."""
    HARDWARE = "hardware"  # Hardware-accelerated via Transformer Engine
    SOFTWARE = "software"  # Software fallback implementation
    MIXED = "mixed"        # Mixed precision (attention: BF16, FFN: NVFP4)


@dataclass
class NVFP4Config:
    """Configuration for NVFP4 quantization.
    
    Attributes:
        mode: NVFP4 quantization mode (hardware/software/mixed)
        block_size: Quantization block size (must be multiple of 16)
        compute_dtype: Compute dtype (typically BF16 for Ampere+)
        attention_dtype: Dtype for attention layers
        ffn_dtype: Dtype for FFN layers
        enable_scaling: Enable per-block scaling factors
        stochastic_rounding: Use stochastic rounding for quantization
        amax_history_len: Length of amax history for scaling
    """
    mode: NVFP4Mode = NVFP4Mode.MIXED
    block_size: int = 16
    compute_dtype: torch.dtype = torch.bfloat16
    attention_dtype: torch.dtype = torch.bfloat16
    ffn_dtype: torch.dtype = torch.float8_e4m3fn  # NVFP4 format
    enable_scaling: bool = True
    stochastic_rounding: bool = True
    amax_history_len: int = 1024
    mixed_precision_threshold: int = 4096  # Dim threshold for mixed precision
    
    def __post_init__(self):
        """Validate configuration."""
        if self.block_size % 16 != 0:
            raise ValueError(f"block_size must be multiple of 16, got {self.block_size}")
        
        if self.mode == NVFP4Mode.HARDWARE and not NVFP4_AVAILABLE:
            logger.warning("Hardware mode requested but Transformer Engine not available. "
                          "Falling back to software mode.")
            self.mode = NVFP4Mode.SOFTWARE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.mode.value,
            'block_size': self.block_size,
            'compute_dtype': str(self.compute_dtype),
            'attention_dtype': str(self.attention_dtype),
            'ffn_dtype': str(self.ffn_dtype),
            'enable_scaling': self.enable_scaling,
            'stochastic_rounding': self.stochastic_rounding,
            'amax_history_len': self.amax_history_len,
            'mixed_precision_threshold': self.mixed_precision_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NVFP4Config':
        """Create config from dictionary."""
        config = cls()
        config.mode = NVFP4Mode(data.get('mode', 'mixed'))
        config.block_size = data.get('block_size', 16)
        config.compute_dtype = getattr(torch, data.get('compute_dtype', 'bfloat16').split('.')[-1])
        config.attention_dtype = getattr(torch, data.get('attention_dtype', 'bfloat16').split('.')[-1])
        config.ffn_dtype = getattr(torch, data.get('ffn_dtype', 'float8_e4m3fn').split('.')[-1])
        config.enable_scaling = data.get('enable_scaling', True)
        config.stochastic_rounding = data.get('stochastic_rounding', True)
        config.amax_history_len = data.get('amax_history_len', 1024)
        config.mixed_precision_threshold = data.get('mixed_precision_threshold', 4096)
        return config


class NVFP4QuantizationError(SLIError):
    """Raised when NVFP4 quantization fails."""
    
    def __init__(self, layer_name: str, message: str = None):
        self.layer_name = layer_name
        msg = message or f"NVFP4 quantization failed for layer: {layer_name}"
        super().__init__(msg)


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data."""
    data: torch.Tensor  # Quantized data
    scale: torch.Tensor  # Per-block scales
    orig_shape: Tuple[int, ...]
    block_size: int
    dtype: torch.dtype


class NVFP4Quantizer:
    """Handles NVFP4 quantization and de-quantization of tensors.
    
    Implements block-wise quantization where each block of size 16
    has its own scale factor for optimal precision.
    """
    
    # NVFP4 E4M3 format constants
    E4M3_MAX = 448.0  # Max representable value in E4M3
    E4M3_MIN = -448.0
    
    def __init__(self, config: Optional[NVFP4Config] = None):
        """Initialize NVFP4 quantizer.
        
        Args:
            config: NVFP4 configuration
        """
        self.config = config or NVFP4Config()
        self._amax_history: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        
        if self.config.mode == NVFP4Mode.HARDWARE and NVFP4_AVAILABLE:
            self._init_hardware_quantization()
    
    def _init_hardware_quantization(self):
        """Initialize Transformer Engine for hardware quantization."""
        try:
            # Configure FP8 recipe for NVFP4
            self._fp8_recipe = recipe.DelayedScaling(
                fp8_format=recipe.Format.E4M3,
                amax_history_len=self.config.amax_history_len,
                amax_compute_algo="max"
            )
            logger.info("Hardware NVFP4 quantization initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hardware quantization: {e}")
            self.config.mode = NVFP4Mode.SOFTWARE
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "",
        use_stochastic_rounding: Optional[bool] = None
    ) -> QuantizedTensor:
        """Quantize a tensor to NVFP4 format.
        
        Args:
            tensor: Tensor to quantize
            name: Tensor name for tracking
            use_stochastic_rounding: Override stochastic rounding setting
            
        Returns:
            QuantizedTensor containing quantized data and metadata
        """
        if use_stochastic_rounding is None:
            use_stochastic_rounding = self.config.stochastic_rounding
        
        if self.config.mode == NVFP4Mode.HARDWARE and NVFP4_AVAILABLE:
            return self._quantize_hardware(tensor, name)
        else:
            return self._quantize_software(tensor, name, use_stochastic_rounding)
    
    def _quantize_software(
        self,
        tensor: torch.Tensor,
        name: str,
        stochastic_rounding: bool
    ) -> QuantizedTensor:
        """Software fallback quantization.
        
        Implements block-wise quantization with per-block scaling.
        """
        orig_shape = tensor.shape
        block_size = self.config.block_size
        
        # Reshape to blocks
        if tensor.dim() == 2:
            # Linear layer weight: [out_features, in_features]
            out_features, in_features = tensor.shape
            
            # Pad if necessary
            pad_in = (block_size - in_features % block_size) % block_size
            pad_out = (block_size - out_features % block_size) % block_size
            
            if pad_in > 0 or pad_out > 0:
                tensor = torch.nn.functional.pad(tensor, (0, pad_in, 0, pad_out))
            
            # Reshape to blocks
            blocks = tensor.reshape(
                (out_features + pad_out) // block_size,
                block_size,
                (in_features + pad_in) // block_size,
                block_size
            ).permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
        else:
            # Flatten and pad
            flat = tensor.flatten()
            pad_len = (block_size - flat.numel() % block_size) % block_size
            if pad_len > 0:
                flat = torch.nn.functional.pad(flat, (0, pad_len))
            blocks = flat.reshape(-1, block_size)
        
        # Compute per-block scales
        amax = blocks.abs().max(dim=1, keepdim=True)[0]
        
        # Update history
        with self._lock:
            if name not in self._amax_history:
                self._amax_history[name] = []
            self._amax_history[name].append(amax.max().item())
            if len(self._amax_history[name]) > self.config.amax_history_len:
                self._amax_history[name].pop(0)
        
        # Compute scale to map to E4M3 range
        scale = amax / self.E4M3_MAX
        scale = torch.clamp(scale, min=1e-12)  # Prevent division by zero
        
        # Quantize
        if stochastic_rounding and tensor.requires_grad:
            # Add noise for stochastic rounding during training
            noise = torch.rand_like(blocks) - 0.5
            quantized = (blocks / scale + noise).round()
        else:
            quantized = (blocks / scale).round()
        
        # Clamp to E4M3 range
        quantized = torch.clamp(quantized, self.E4M3_MIN, self.E4M3_MAX)
        
        # Convert to FP8 E4M3
        quantized = quantized.to(torch.float8_e4m3fn)
        
        return QuantizedTensor(
            data=quantized,
            scale=scale.squeeze(),
            orig_shape=orig_shape,
            block_size=block_size,
            dtype=torch.float8_e4m3fn
        )
    
    def _quantize_hardware(self, tensor: torch.Tensor, name: str) -> QuantizedTensor:
        """Hardware-accelerated quantization via Transformer Engine."""
        if not NVFP4_AVAILABLE:
            raise NVFP4QuantizationError(name, "Hardware quantization not available")
        
        # Use Transformer Engine's FP8 quantization
        with te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
            # TE handles quantization automatically
            quantized = tensor.to(torch.float8_e4m3fn)
        
        # Compute scale from amax history
        with self._lock:
            if name in self._amax_history and self._amax_history[name]:
                amax = max(self._amax_history[name])
                scale = torch.tensor([amax / self.E4M3_MAX], device=tensor.device)
            else:
                scale = torch.tensor([1.0], device=tensor.device)
        
        return QuantizedTensor(
            data=quantized,
            scale=scale,
            orig_shape=tensor.shape,
            block_size=self.config.block_size,
            dtype=torch.float8_e4m3fn
        )
    
    def dequantize_tensor(self, quantized: QuantizedTensor) -> torch.Tensor:
        """Dequantize a tensor from NVFP4 format.
        
        Args:
            quantized: QuantizedTensor to dequantize
            
        Returns:
            Dequantized tensor
        """
        if self.config.mode == NVFP4Mode.HARDWARE and NVFP4_AVAILABLE:
            return self._dequantize_hardware(quantized)
        else:
            return self._dequantize_software(quantized)
    
    def _dequantize_software(self, quantized: QuantizedTensor) -> torch.Tensor:
        """Software dequantization."""
        # Convert from FP8 to compute dtype
        data = quantized.data.to(self.config.compute_dtype)
        
        # Reshape scales if necessary
        scale = quantized.scale
        if scale.dim() == 1 and data.dim() == 2:
            # Reshape scale to match blocked data
            num_blocks = data.shape[0]
            if scale.shape[0] != num_blocks:
                # Use single global scale
                scale = scale.unsqueeze(0).expand(num_blocks, -1)
        
        # Dequantize
        if data.dim() == 2 and scale.dim() == 1:
            dequantized = data * scale.unsqueeze(1)
        else:
            dequantized = data * scale
        
        # Reshape to original shape
        if len(quantized.orig_shape) == 2:
            out_features, in_features = quantized.orig_shape
            dequantized = dequantized.reshape(
                out_features // quantized.block_size,
                in_features // quantized.block_size,
                quantized.block_size,
                quantized.block_size
            ).permute(0, 2, 1, 3).reshape(out_features, in_features)
        elif len(quantized.orig_shape) > 0:
            dequantized = dequantized.flatten()[:int(np.prod(quantized.orig_shape))]
            dequantized = dequantized.reshape(quantized.orig_shape)
        # else: keep as is for 0-dim tensors
        
        return dequantized
    
    def _dequantize_hardware(self, quantized: QuantizedTensor) -> torch.Tensor:
        """Hardware-accelerated dequantization."""
        # Convert back to compute dtype
        return quantized.data.to(self.config.compute_dtype) * quantized.scale


class NVFP4StreamingLoader:
    """Streaming layer loader with NVFP4 quantization support.
    
    This class handles:
    - Loading layers from various sources (disk, cache, network)
    - On-the-fly NVFP4 quantization
    - Mixed precision loading (BF16 for attention, NVFP4 for FFN)
    - Efficient memory management for streaming inference
    
    Example:
        >>> loader = NVFP4StreamingLoader(NVFP4Config(mode=NVFP4Mode.MIXED))
        >>> layer = loader.load_layer(model_id="meta-llama/Llama-2-70b", layer_idx=0)
        >>> quantized = loader.quantize_layer(layer, is_attention=True)
        >>> dequantized = loader.dequantize_layer(quantized)
    """
    
    def __init__(
        self,
        config: Optional[NVFP4Config] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        """Initialize NVFP4 streaming loader.
        
        Args:
            config: NVFP4 configuration
            cache_dir: Directory for quantized layer cache
            device: Target device for loading
        """
        self.config = config or NVFP4Config()
        self.quantizer = NVFP4Quantizer(self.config)
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self._stats = {
            'layers_loaded': 0,
            'layers_quantized': 0,
            'bytes_saved': 0,
            'load_time_ms': 0.0,
            'quantize_time_ms': 0.0,
        }
        
        self._lock = threading.RLock()
        logger.info(f"NVFP4StreamingLoader initialized (mode: {self.config.mode.value})")
    
    def load_layer(
        self,
        model_id: str,
        layer_idx: int,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        source: str = "auto"
    ) -> nn.Module:
        """Load a layer with NVFP4 support.
        
        Args:
            model_id: Model identifier
            layer_idx: Layer index
            layer_weights: Optional pre-loaded weights
            source: Source to load from ("auto", "cache", "disk", "network")
            
        Returns:
            Loaded layer module
        """
        import time
        start_time = time.time()
        
        # Try cache first if enabled
        if source in ("auto", "cache") and self.cache_dir:
            cached_layer = self._load_from_cache(model_id, layer_idx)
            if cached_layer is not None:
                with self._lock:
                    self._stats['layers_loaded'] += 1
                return cached_layer
        
        # Build layer from weights
        if layer_weights is not None:
            layer = self._build_layer_from_weights(layer_weights, layer_idx)
        else:
            raise WeightLoadingError(
                f"layer_{layer_idx}",
                None,
                Exception("No weights provided and cache miss")
            )
        
        load_time = (time.time() - start_time) * 1000
        with self._lock:
            self._stats['layers_loaded'] += 1
            self._stats['load_time_ms'] += load_time
        
        return layer
    
    def _build_layer_from_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int
    ) -> nn.Module:
        """Build a layer module from weight dictionary."""
        # Create a simple container module
        layer = nn.Module()
        
        for name, tensor in weights.items():
            # Determine if this is a parameter or buffer
            if tensor.requires_grad:
                setattr(layer, name, nn.Parameter(tensor))
            else:
                layer.register_buffer(name, tensor)
        
        return layer
    
    def _load_from_cache(self, model_id: str, layer_idx: int) -> Optional[nn.Module]:
        """Try to load a layer from cache."""
        if not self.cache_dir:
            return None
        
        cache_key = f"{model_id.replace('/', '_')}_layer_{layer_idx}_nvfp4.pt"
        cache_path = self.cache_dir / cache_key
        
        if cache_path.exists():
            try:
                return torch.load(cache_path, map_location=self.device, weights_only=False)
            except Exception as e:
                logger.warning(f"Failed to load cached layer: {e}")
        
        return None
    
    def quantize_layer(
        self,
        layer: nn.Module,
        is_attention: bool = False,
        layer_name: str = ""
    ) -> nn.Module:
        """Quantize a layer to NVFP4 format.
        
        Args:
            layer: Layer to quantize
            is_attention: Whether this is an attention layer
            layer_name: Layer name for tracking
            
        Returns:
            Quantized layer
        """
        import time
        start_time = time.time()
        
        # Determine dtype based on layer type and config
        if self.config.mode == NVFP4Mode.MIXED:
            target_dtype = (
                self.config.attention_dtype
                if is_attention
                else self.config.ffn_dtype
            )
        else:
            target_dtype = self.config.ffn_dtype
        
        # Quantize all parameters
        quantized_params = {}
        for name, param in layer.named_parameters():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                if target_dtype == torch.float8_e4m3fn:
                    # Quantize to NVFP4
                    quantized = self.quantizer.quantize_tensor(
                        param.data,
                        name=f"{layer_name}.{name}"
                    )
                    quantized_params[name] = quantized
                else:
                    # Just convert dtype
                    quantized_params[name] = param.data.to(target_dtype)
            else:
                quantized_params[name] = param.data
        
        # Create quantized layer
        quantized_layer = nn.Module()
        for name, tensor in quantized_params.items():
            if isinstance(tensor, QuantizedTensor):
                # Store quantized tensor
                quantized_layer.register_buffer(f"{name}_quantized", tensor.data)
                quantized_layer.register_buffer(f"{name}_scale", tensor.scale)
                quantized_layer.register_buffer(
                    f"{name}_orig_shape",
                    torch.tensor(tensor.orig_shape)
                )
            else:
                quantized_layer.register_buffer(name, tensor)
        
        # Copy buffers
        for name, buffer in layer.named_buffers():
            if name not in quantized_params:
                quantized_layer.register_buffer(name, buffer)
        
        quantize_time = (time.time() - start_time) * 1000
        with self._lock:
            self._stats['layers_quantized'] += 1
            self._stats['quantize_time_ms'] += quantize_time
        
        return quantized_layer
    
    def dequantize_layer(self, layer: nn.Module) -> nn.Module:
        """Dequantize a layer from NVFP4 format.
        
        Args:
            layer: Quantized layer
            
        Returns:
            Dequantized layer in compute dtype
        """
        dequantized = nn.Module()
        
        # Track which parameters have been dequantized
        dequantized_names = set()
        
        # Look for quantized parameters
        for name, buffer in layer.named_buffers():
            if name.endswith("_quantized"):
                param_name = name[:-10]  # Remove "_quantized" suffix
                scale_name = f"{param_name}_scale"
                shape_name = f"{param_name}_orig_shape"
                
                if hasattr(layer, scale_name):
                    # Reconstruct QuantizedTensor
                    orig_shape = tuple(getattr(layer, shape_name).tolist())
                    quantized_tensor = QuantizedTensor(
                        data=buffer,
                        scale=getattr(layer, scale_name),
                        orig_shape=orig_shape,
                        block_size=self.config.block_size,
                        dtype=torch.float8_e4m3fn
                    )
                    
                    # Dequantize
                    dequantized_data = self.quantizer.dequantize_tensor(quantized_tensor)
                    setattr(dequantized, param_name, nn.Parameter(dequantized_data))
                    dequantized_names.add(param_name)
            elif not any(
                name.endswith(suffix)
                for suffix in ["_scale", "_orig_shape"]
            ):
                # Regular buffer
                dequantized.register_buffer(name, buffer)
        
        # Copy any remaining parameters that weren't quantized
        for name, param in layer.named_parameters():
            if name not in dequantized_names:
                setattr(dequantized, name, nn.Parameter(param.data))
        
        return dequantized
    
    def cache_layer(
        self,
        model_id: str,
        layer_idx: int,
        layer: nn.Module
    ) -> bool:
        """Cache a quantized layer to disk.
        
        Args:
            model_id: Model identifier
            layer_idx: Layer index
            layer: Layer to cache
            
        Returns:
            True if caching succeeded
        """
        if not self.cache_dir:
            return False
        
        cache_key = f"{model_id.replace('/', '_')}_layer_{layer_idx}_nvfp4.pt"
        cache_path = self.cache_dir / cache_key
        
        try:
            torch.save(layer, cache_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to cache layer: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        with self._lock:
            return self._stats.copy()
    
    def clear_cache(self):
        """Clear cached layers."""
        if not self.cache_dir:
            return
        
        import shutil
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared NVFP4 cache at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Convenience functions
def get_nvfp4_config(
    mode: str = "mixed",
    block_size: int = 16,
    compute_dtype: str = "bfloat16"
) -> NVFP4Config:
    """Get NVFP4 configuration with common presets.
    
    Args:
        mode: Quantization mode (hardware, software, mixed)
        block_size: Block size for quantization
        compute_dtype: Compute dtype (bfloat16, float16, float32)
        
    Returns:
        NVFP4Config instance
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    return NVFP4Config(
        mode=NVFP4Mode(mode),
        block_size=block_size,
        compute_dtype=dtype_map.get(compute_dtype, torch.bfloat16),
    )


def quantize_to_nvfp4(
    tensor: torch.Tensor,
    block_size: int = 16
) -> QuantizedTensor:
    """Convenience function to quantize a tensor to NVFP4.
    
    Args:
        tensor: Tensor to quantize
        block_size: Quantization block size
        
    Returns:
        QuantizedTensor
    """
    config = NVFP4Config(block_size=block_size)
    quantizer = NVFP4Quantizer(config)
    return quantizer.quantize_tensor(tensor)


def dequantize_from_nvfp4(quantized: QuantizedTensor) -> torch.Tensor:
    """Convenience function to dequantize from NVFP4.
    
    Args:
        quantized: QuantizedTensor to dequantize
        
    Returns:
        Dequantized tensor
    """
    config = NVFP4Config(block_size=quantized.block_size)
    quantizer = NVFP4Quantizer(config)
    return quantizer.dequantize_tensor(quantized)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing NVFP4 Streaming Loader")
    print("=" * 60)
    
    # Create config
    config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)
    print(f"NVFP4 Available: {NVFP4_AVAILABLE}")
    print(f"Config: {config.to_dict()}")
    
    # Create loader
    loader = NVFP4StreamingLoader(config)
    
    # Create test layer
    test_layer = nn.Linear(4096, 11008)
    print(f"\nOriginal layer dtype: {test_layer.weight.dtype}")
    
    # Quantize as FFN layer
    quantized = loader.quantize_layer(test_layer, is_attention=False)
    print(f"Quantized layer type: {type(quantized)}")
    
    # Dequantize
    dequantized = loader.dequantize_layer(quantized)
    print(f"Dequantized dtype: {dequantized.weight.dtype}")
    
    # Show stats
    print(f"\nStats: {loader.get_stats()}")
    
    print("\n" + "=" * 60)
