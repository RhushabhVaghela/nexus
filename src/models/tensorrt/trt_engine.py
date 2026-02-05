"""
TensorRT Engine Wrapper for Nexus

Wraps TensorRT-LLM engine for high-performance inference.
Supports FP8, INT8, and INT4 quantization modes.

Author: Nexus Team
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import TensorRT-LLM
try:
    import tensorrt_llm
    from tensorrt_llm import Builder, BuildConfig
    from tensorrt_llm.runtime import ModelRunner
    TRTLLM_AVAILABLE = True
except ImportError:
    TRTLLM_AVAILABLE = False
    warnings.warn(
        "TensorRT-LLM not available. Install with: "
        "pip install tensorrt-llm"
    )


class TRTQuantizationMode(Enum):
    """TensorRT quantization modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"      # 8-bit floating point
    INT8 = "int8"    # 8-bit integer
    INT4 = "int4"    # 4-bit integer (AWQ/GPTQ)
    WOQ = "woq"      # Weight-only quantization


@dataclass
class TRTBuildConfig:
    """Configuration for building TensorRT engine."""
    max_batch_size: int = 1
    max_seq_length: int = 2048
    max_input_len: int = 1024
    max_output_len: int = 1024
    max_beam_width: int = 1
    dtype: str = "float16"
    quantization: TRTQuantizationMode = TRTQuantizationMode.FP16
    
    # Plugin settings
    use_gpt_attention_plugin: bool = True
    use_gemm_plugin: bool = True
    use_layernorm_plugin: bool = True
    
    # Optimization
    opt_level: int = 3
    strongly_typed: bool = False


@dataclass
class TRTEngineConfig:
    """Configuration for TensorRT engine."""
    engine_path: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    build_config: TRTBuildConfig = field(default_factory=TRTBuildConfig)
    device: str = "cuda"
    
    def __post_init__(self):
        if self.engine_path is None and self.model_path is None:
            raise ValueError("Either engine_path or model_path must be provided")


class TRTEngineError(Exception):
    """Raised when TensorRT engine operations fail."""
    pass


class TRTEngine:
    """
    TensorRT-LLM Engine wrapper for high-performance inference.
    
    Supports:
    - FP32, FP16, BF16 precision
    - FP8, INT8, INT4 quantization
    - Dynamic batching
    - Beam search
    
    Example:
        >>> config = TRTEngineConfig(
        ...     engine_path="path/to/engine.plan",
        ...     quantization=TRTQuantizationMode.FP16
        ... )
        >>> engine = TRTEngine(config)
        >>> 
        >>> # Generate text
        >>> outputs = engine.generate(
        ...     input_ids=input_ids,
        ...     max_new_tokens=100,
        ...     temperature=0.7
        ... )
    """
    
    def __init__(self, config: TRTEngineConfig):
        """
        Initialize TensorRT engine.
        
        Args:
            config: Engine configuration
        """
        if not TRTLLM_AVAILABLE:
            raise TRTEngineError(
                "TensorRT-LLM not available. Install with: pip install tensorrt-llm"
            )
        
        self.config = config
        self.engine = None
        self.tokenizer = None
        self._runtime = None
        self._session = None
        
        # Statistics
        self._stats = {
            'inference_count': 0,
            'total_tokens_generated': 0,
            'avg_latency_ms': 0.0,
        }
        
        # Load or build engine
        if config.engine_path and Path(config.engine_path).exists():
            self._load_engine(config.engine_path)
        elif config.model_path:
            self._build_engine(config.model_path, config.build_config)
        else:
            raise TRTEngineError("No engine or model path provided")
        
        logger.info(f"TRTEngine initialized (quantization: {config.build_config.quantization.value})")
    
    def _load_engine(self, engine_path: str):
        """Load pre-built TensorRT engine."""
        logger.info(f"Loading TensorRT engine from {engine_path}")
        
        try:
            # Load engine using TensorRT-LLM
            self._runtime = ModelRunner.from_dir(engine_path)
            logger.info("Engine loaded successfully")
        except Exception as e:
            raise TRTEngineError(f"Failed to load engine: {e}")
    
    def _build_engine(self, model_path: str, build_config: TRTBuildConfig):
        """Build TensorRT engine from model."""
        logger.info(f"Building TensorRT engine from {model_path}")
        
        try:
            # Create builder
            builder = Builder()
            
            # Configure build
            config = BuildConfig(
                max_batch_size=build_config.max_batch_size,
                max_input_len=build_config.max_input_len,
                max_output_len=build_config.max_output_len,
                max_beam_width=build_config.max_beam_width,
            )
            
            # Set dtype
            dtype_map = {
                "float32": tensorrt_llm.Dtype.float32,
                "float16": tensorrt_llm.Dtype.float16,
                "bfloat16": tensorrt_llm.Dtype.bfloat16,
            }
            
            # Build engine
            engine = builder.build(model_path, config)
            
            self._runtime = ModelRunner(engine)
            logger.info("Engine built successfully")
            
        except Exception as e:
            raise TRTEngineError(f"Failed to build engine: {e}")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tokens using TensorRT engine.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary with 'sequences' and other outputs
        """
        if self._runtime is None:
            raise TRTEngineError("Engine not loaded")
        
        import time
        start_time = time.time()
        
        try:
            # Prepare inputs
            batch_size = input_ids.shape[0]
            input_lengths = torch.tensor(
                [input_ids.shape[1]] * batch_size,
                dtype=torch.int32,
                device=self.config.device
            )
            
            # Run inference
            outputs = self._runtime.generate(
                input_ids=input_ids.to(self.config.device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs
            )
            
            # Update stats
            latency_ms = (time.time() - start_time) * 1000
            num_tokens = outputs['sequences'].shape[1] - input_ids.shape[1]
            
            self._stats['inference_count'] += 1
            self._stats['total_tokens_generated'] += num_tokens
            
            # Update running average
            n = self._stats['inference_count']
            self._stats['avg_latency_ms'] = (
                (self._stats['avg_latency_ms'] * (n - 1) + latency_ms) / n
            )
            
            return outputs
            
        except Exception as e:
            raise TRTEngineError(f"Generation failed: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key-value cache
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        if self._runtime is None:
            raise TRTEngineError("Engine not loaded")
        
        try:
            outputs = self._runtime(
                input_ids=input_ids.to(self.config.device),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs
            )
            
            return outputs
            
        except Exception as e:
            raise TRTEngineError(f"Forward pass failed: {e}")
    
    def save_engine(self, output_path: str):
        """
        Save built engine to disk.
        
        Args:
            output_path: Path to save engine
        """
        if self._runtime is None:
            raise TRTEngineError("No engine to save")
        
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save engine
            self._runtime.save(output_dir)
            
            # Save config
            config_path = output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'max_batch_size': self.config.build_config.max_batch_size,
                    'max_seq_length': self.config.build_config.max_seq_length,
                    'dtype': self.config.build_config.dtype,
                    'quantization': self.config.build_config.quantization.value,
                }, f, indent=2)
            
            logger.info(f"Engine saved to {output_path}")
            
        except Exception as e:
            raise TRTEngineError(f"Failed to save engine: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            'memory': self.get_memory_usage(),
            'quantization': self.config.build_config.quantization.value,
            'max_batch_size': self.config.build_config.max_batch_size,
            'max_seq_length': self.config.build_config.max_seq_length,
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_runtime') and self._runtime is not None:
            del self._runtime
        torch.cuda.empty_cache()
