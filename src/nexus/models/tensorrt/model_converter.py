"""
Model Converter for PyTorch to TensorRT

Converts PyTorch models to TensorRT format with quantization support.

Author: Nexus Team
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig

logger = logging.getLogger(__name__)

# Try to import TensorRT-LLM
try:
    import tensorrt_llm
    from tensorrt_llm.builder import Builder
    from tensorrt_llm.models import MODEL_MAP
    TRTLLM_AVAILABLE = True
except ImportError:
    TRTLLM_AVAILABLE = False


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""
    model_name_or_path: str
    output_dir: str
    dtype: str = "float16"
    quantization: str = "none"  # none, fp8, int8, int4, awq, gptq
    max_batch_size: int = 1
    max_seq_length: int = 2048
    max_input_len: int = 1024
    max_output_len: int = 1024
    
    # Quantization settings
    calib_dataset: Optional[str] = None
    calib_batches: int = 512
    awq_checkpoint: Optional[str] = None
    gptq_checkpoint: Optional[str] = None
    
    # Build settings
    use_gemm_plugin: bool = True
    use_gpt_attention_plugin: bool = True
    use_layernorm_plugin: bool = True
    opt_level: int = 3


class ConversionError(Exception):
    """Raised when model conversion fails."""
    pass


class ModelConverter:
    """
    Converts PyTorch models to TensorRT format.
    
    Supports:
    - Direct PyTorch to TensorRT conversion
    - FP8, INT8, INT4 quantization
    - AWQ and GPTQ quantized models
    - Multi-GPU conversion
    
    Example:
        >>> config = ConversionConfig(
        ...     model_name_or_path="meta-llama/Llama-2-7b",
        ...     output_dir="./trt_engines/llama-7b",
        ...     dtype="float16",
        ...     quantization="fp8"
        ... )
        >>> converter = ModelConverter(config)
        >>> converter.convert()
    """
    
    def __init__(self, config: ConversionConfig):
        """
        Initialize model converter.
        
        Args:
            config: Conversion configuration
        """
        if not TRTLLM_AVAILABLE:
            raise ConversionError(
                "TensorRT-LLM not available. Install with: pip install tensorrt-llm"
            )
        
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model config
        try:
            self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
            logger.info(f"Loaded model config for {self.model_config.model_type}")
        except Exception as e:
            raise ConversionError(f"Failed to load model config: {e}")
    
    def convert(self) -> str:
        """
        Convert model to TensorRT format.
        
        Returns:
            Path to converted engine
        """
        logger.info(f"Starting conversion: {self.config.model_name_or_path}")
        
        # Determine conversion strategy based on quantization
        if self.config.quantization == "none":
            return self._convert_standard()
        elif self.config.quantization in ["fp8", "int8"]:
            return self._convert_quantized()
        elif self.config.quantization in ["int4", "awq"]:
            return self._convert_awq()
        elif self.config.quantization == "gptq":
            return self._convert_gptq()
        else:
            raise ConversionError(f"Unsupported quantization: {self.config.quantization}")
    
    def _convert_standard(self) -> str:
        """Convert model without quantization."""
        logger.info("Converting model to FP16/FP32")
        
        try:
            from tensorrt_llm.models import LLaMAForCausalLM
            
            # Create TensorRT-LLM model
            trt_llm_model = LLaMAForCausalLM.from_hugging_face(
                self.config.model_name_or_path,
                dtype=self.config.dtype,
            )
            
            # Build engine
            engine_path = self._build_engine(trt_llm_model)
            
            logger.info(f"Model converted successfully: {engine_path}")
            return engine_path
            
        except Exception as e:
            raise ConversionError(f"Standard conversion failed: {e}")
    
    def _convert_quantized(self) -> str:
        """Convert model with FP8/INT8 quantization."""
        logger.info(f"Converting model with {self.config.quantization} quantization")
        
        try:
            from tensorrt_llm.models import LLaMAForCausalLM
            from tensorrt_llm.quantization import QuantMode
            
            # Determine quantization mode
            if self.config.quantization == "fp8":
                quant_mode = QuantMode.FP8_QDQ
            elif self.config.quantization == "int8":
                quant_mode = QuantMode.INT8_SQ
            else:
                quant_mode = QuantMode.NONE
            
            # Load model
            trt_llm_model = LLaMAForCausalLM.from_hugging_face(
                self.config.model_name_or_path,
                dtype=self.config.dtype,
                quant_mode=quant_mode,
            )
            
            # Calibrate if needed
            if self.config.calib_dataset:
                self._calibrate_model(trt_llm_model)
            
            # Build engine
            engine_path = self._build_engine(trt_llm_model)
            
            logger.info(f"Quantized model converted: {engine_path}")
            return engine_path
            
        except Exception as e:
            raise ConversionError(f"Quantized conversion failed: {e}")
    
    def _convert_awq(self) -> str:
        """Convert AWQ quantized model."""
        logger.info("Converting AWQ quantized model")
        
        try:
            from tensorrt_llm.models import LLaMAForCausalLM
            from tensorrt_llm.quantization import QuantMode
            
            # Load AWQ checkpoint
            if self.config.awq_checkpoint is None:
                raise ConversionError("AWQ checkpoint not provided")
            
            trt_llm_model = LLaMAForCausalLM.from_hugging_face(
                self.config.model_name_or_path,
                dtype=self.config.dtype,
                quant_mode=QuantMode.AMAX_ACTIVATIONS_WEIGHTS,
                quant_config={
                    'checkpoint': self.config.awq_checkpoint,
                    'bits': 4,
                }
            )
            
            engine_path = self._build_engine(trt_llm_model)
            
            logger.info(f"AWQ model converted: {engine_path}")
            return engine_path
            
        except Exception as e:
            raise ConversionError(f"AWQ conversion failed: {e}")
    
    def _convert_gptq(self) -> str:
        """Convert GPTQ quantized model."""
        logger.info("Converting GPTQ quantized model")
        
        try:
            from tensorrt_llm.models import LLaMAForCausalLM
            from tensorrt_llm.quantization import QuantMode
            
            # Load GPTQ checkpoint
            if self.config.gptq_checkpoint is None:
                raise ConversionError("GPTQ checkpoint not provided")
            
            trt_llm_model = LLaMAForCausalLM.from_hugging_face(
                self.config.model_name_or_path,
                dtype=self.config.dtype,
                quant_mode=QuantMode.INT4_WEIGHTS,
                quant_config={
                    'checkpoint': self.config.gptq_checkpoint,
                    'bits': 4,
                }
            )
            
            engine_path = self._build_engine(trt_llm_model)
            
            logger.info(f"GPTQ model converted: {engine_path}")
            return engine_path
            
        except Exception as e:
            raise ConversionError(f"GPTQ conversion failed: {e}")
    
    def _build_engine(self, model) -> str:
        """Build TensorRT engine from model."""
        logger.info("Building TensorRT engine")
        
        try:
            from tensorrt_llm.builder import Builder
            
            # Create builder
            builder = Builder()
            
            # Build config
            build_config = {
                'max_batch_size': self.config.max_batch_size,
                'max_input_len': self.config.max_input_len,
                'max_output_len': self.config.max_output_len,
                'max_seq_length': self.config.max_seq_length,
                'use_gemm_plugin': self.config.use_gemm_plugin,
                'use_gpt_attention_plugin': self.config.use_gpt_attention_plugin,
                'use_layernorm_plugin': self.config.use_layernorm_plugin,
                'opt_level': self.config.opt_level,
            }
            
            # Build engine
            engine = builder.build(model, build_config)
            
            # Save engine
            engine_path = self.output_dir / "model.engine"
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            # Save config
            config_path = self.output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_type': self.model_config.model_type,
                    'dtype': self.config.dtype,
                    'quantization': self.config.quantization,
                    'max_batch_size': self.config.max_batch_size,
                    'max_seq_length': self.config.max_seq_length,
                }, f, indent=2)
            
            return str(engine_path)
            
        except Exception as e:
            raise ConversionError(f"Engine build failed: {e}")
    
    def _calibrate_model(self, model):
        """Calibrate model for quantization."""
        logger.info(f"Calibrating model on {self.config.calib_dataset}")
        
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
            
            # Load calibration dataset
            dataset = load_dataset(
                self.config.calib_dataset,
                split="train",
                streaming=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            
            # Collect calibration data
            calib_data = []
            for i, example in enumerate(dataset):
                if i >= self.config.calib_batches:
                    break
                
                text = example.get('text', example.get('content', ''))
                tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                calib_data.append(tokens.input_ids)
            
            # Calibrate
            model.calibrate(calib_data)
            
            logger.info(f"Calibration complete with {len(calib_data)} batches")
            
        except Exception as e:
            logger.warning(f"Calibration failed: {e}. Using default scales.")


class LayerConverter:
    """
    Converts individual PyTorch layers to TensorRT format.
    
    Useful for fine-grained control over conversion.
    """
    
    def __init__(self, dtype: str = "float16"):
        self.dtype = dtype
    
    def convert_linear(self, layer: nn.Linear) -> Dict[str, Any]:
        """Convert Linear layer."""
        return {
            'type': 'linear',
            'in_features': layer.in_features,
            'out_features': layer.out_features,
            'bias': layer.bias is not None,
            'weight': layer.weight.data.cpu().numpy(),
            'bias_data': layer.bias.data.cpu().numpy() if layer.bias is not None else None,
            'dtype': self.dtype,
        }
    
    def convert_layernorm(self, layer: nn.LayerNorm) -> Dict[str, Any]:
        """Convert LayerNorm layer."""
        return {
            'type': 'layernorm',
            'normalized_shape': layer.normalized_shape,
            'eps': layer.eps,
            'elementwise_affine': layer.elementwise_affine,
            'weight': layer.weight.data.cpu().numpy() if layer.weight is not None else None,
            'bias': layer.bias.data.cpu().numpy() if layer.bias is not None else None,
        }
    
    def convert_attention(
        self,
        layer: nn.Module,
        num_heads: int,
        head_dim: int
    ) -> Dict[str, Any]:
        """Convert Attention layer."""
        return {
            'type': 'attention',
            'num_heads': num_heads,
            'head_dim': head_dim,
            'qkv_weight': getattr(layer, 'qkv_proj', None),
            'o_weight': getattr(layer, 'o_proj', None),
            'dtype': self.dtype,
        }
    
    def convert_embedding(self, layer: nn.Embedding) -> Dict[str, Any]:
        """Convert Embedding layer."""
        return {
            'type': 'embedding',
            'num_embeddings': layer.num_embeddings,
            'embedding_dim': layer.embedding_dim,
            'weight': layer.weight.data.cpu().numpy(),
            'dtype': self.dtype,
        }


def convert_model(
    model_name_or_path: str,
    output_dir: str,
    dtype: str = "float16",
    quantization: str = "none",
    max_batch_size: int = 1,
    max_seq_length: int = 2048,
    **kwargs
) -> str:
    """
    Convenience function to convert a model.
    
    Args:
        model_name_or_path: Model identifier or path
        output_dir: Output directory for converted engine
        dtype: Data type (float32, float16, bfloat16)
        quantization: Quantization mode (none, fp8, int8, int4)
        max_batch_size: Maximum batch size
        max_seq_length: Maximum sequence length
        **kwargs: Additional conversion arguments
        
    Returns:
        Path to converted engine
    """
    config = ConversionConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        dtype=dtype,
        quantization=quantization,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        **kwargs
    )
    
    converter = ModelConverter(config)
    return converter.convert()
