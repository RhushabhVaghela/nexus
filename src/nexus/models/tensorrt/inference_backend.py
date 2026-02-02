"""
TensorRT-LLM Inference Backend

Unified interface for TensorRT-LLM inference with support for:
- Multiple quantization modes
- Dynamic batching
- Streaming generation
- Metrics collection

Author: Nexus Team
"""

import os
import time
import logging
from typing import Dict, Optional, Any, List, Tuple, Union, Iterator, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, AutoTokenizer

from .trt_engine import TRTEngine, TRTEngineConfig, TRTBuildConfig, TRTQuantizationMode, TRTEngineError
from .model_converter import ModelConverter, ConversionConfig

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when backend operations fail."""
    pass


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT backend."""
    model_path: str
    engine_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    quantization_mode: str = "fp16"
    max_batch_size: int = 1
    max_seq_length: int = 2048
    device: str = "cuda"
    enable_streaming: bool = False
    
    def to_engine_config(self) -> TRTEngineConfig:
        """Convert to TRTEngineConfig."""
        quant_map = {
            "fp32": TRTQuantizationMode.FP32,
            "fp16": TRTQuantizationMode.FP16,
            "bf16": TRTQuantizationMode.BF16,
            "fp8": TRTQuantizationMode.FP8,
            "int8": TRTQuantizationMode.INT8,
            "int4": TRTQuantizationMode.INT4,
        }
        
        build_config = TRTBuildConfig(
            max_batch_size=self.max_batch_size,
            max_seq_length=self.max_seq_length,
            quantization=quant_map.get(self.quantization_mode, TRTQuantizationMode.FP16),
            dtype=self.quantization_mode if self.quantization_mode in ["fp32", "fp16", "bf16"] else "float16",
        )
        
        return TRTEngineConfig(
            engine_path=self.engine_path,
            model_path=self.model_path if self.engine_path is None else None,
            build_config=build_config,
            device=self.device,
        )


@dataclass
class GenerationResult:
    """Result from text generation."""
    sequences: torch.Tensor
    scores: Optional[List[torch.Tensor]] = None
    logits: Optional[torch.Tensor] = None
    tokens_generated: int = 0
    generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0


class TensorRTBackend:
    """
    Unified TensorRT-LLM inference backend.
    
    Provides a high-level interface for:
    - Text generation
    - Batch inference
    - Streaming generation
    - Performance metrics
    
    Example:
        >>> config = TensorRTConfig(
        ...     model_path="meta-llama/Llama-2-7b",
        ...     quantization_mode="fp8",
        ...     max_batch_size=4
        ... )
        >>> backend = TensorRTBackend(config)
        >>> 
        >>> # Generate text
        >>> result = backend.generate(
        ...     prompts=["Hello, how are you?"],
        ...     max_new_tokens=100
        ... )
        >>> print(result.sequences)
    """
    
    def __init__(self, config: TensorRTConfig):
        """
        Initialize TensorRT backend.
        
        Args:
            config: Backend configuration
        """
        self.config = config
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path or config.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise BackendError(f"Failed to load tokenizer: {e}")
        
        # Initialize or load engine
        try:
            engine_config = config.to_engine_config()
            self.engine = TRTEngine(engine_config)
        except Exception as e:
            logger.warning(f"Failed to load engine: {e}. Attempting to convert model.")
            self._convert_and_load()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'total_tokens_generated': 0,
            'total_generation_time_ms': 0.0,
            'avg_latency_ms': 0.0,
            'avg_tokens_per_second': 0.0,
        }
        
        logger.info(f"TensorRTBackend initialized (quantization: {config.quantization_mode})")
    
    def _convert_and_load(self):
        """Convert model and load engine."""
        try:
            # Create temporary engine path
            engine_dir = Path(self.config.model_path) / "trt_engine"
            engine_dir.mkdir(exist_ok=True)
            
            # Convert model
            conversion_config = ConversionConfig(
                model_name_or_path=self.config.model_path,
                output_dir=str(engine_dir),
                dtype=self.config.quantization_mode if self.config.quantization_mode in ["fp32", "fp16", "bf16"] else "float16",
                quantization=self.config.quantization_mode,
                max_batch_size=self.config.max_batch_size,
                max_seq_length=self.config.max_seq_length,
            )
            
            converter = ModelConverter(conversion_config)
            engine_path = converter.convert()
            
            # Load converted engine
            self.config.engine_path = str(engine_path)
            engine_config = self.config.to_engine_config()
            self.engine = TRTEngine(engine_config)
            
        except Exception as e:
            raise BackendError(f"Failed to convert and load model: {e}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        return_logits: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from prompts.
        
        Args:
            prompts: Input prompt(s)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            return_logits: Whether to return logits
            **kwargs: Additional generation arguments
            
        Returns:
            GenerationResult with sequences and metadata
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize
        input_ids = self._tokenize(prompts, pad_token_id)
        
        # Generate
        start_time = time.time()
        
        try:
            outputs = self.engine.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_beams=num_beams,
                eos_token_id=eos_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Calculate metrics
            batch_size = len(prompts)
            tokens_generated = outputs['sequences'].shape[1] - input_ids.shape[1]
            tokens_per_second = tokens_generated / (generation_time_ms / 1000)
            
            # Update stats
            self._update_stats(generation_time_ms, tokens_generated)
            
            return GenerationResult(
                sequences=outputs['sequences'],
                scores=outputs.get('scores'),
                logits=outputs.get('logits') if return_logits else None,
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time_ms,
                tokens_per_second=tokens_per_second,
            )
            
        except Exception as e:
            raise BackendError(f"Generation failed: {e}")
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream generated text token by token.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation arguments
            
        Yields:
            Generated tokens as strings
        """
        if not self.config.enable_streaming:
            logger.warning("Streaming not enabled in config, falling back to non-streaming")
            result = self.generate(
                prompts=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )
            text = self.tokenizer.decode(result.sequences[0], skip_special_tokens=True)
            yield text[len(prompt):]
            return
        
        # Tokenize
        input_ids = self._tokenize([prompt])
        
        # Stream generation
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            try:
                outputs = self.engine.generate(
                    input_ids=generated_ids,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    **kwargs
                )
                
                new_token_id = outputs['sequences'][0, -1].item()
                generated_ids = outputs['sequences']
                
                # Decode and yield
                new_token = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
                yield new_token
                
                # Check for EOS
                if new_token_id == self.tokenizer.eos_token_id:
                    break
                    
            except Exception as e:
                logger.error(f"Streaming generation failed: {e}")
                break
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts with dynamic batching.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum new tokens
            **kwargs: Additional generation arguments
            
        Returns:
            List of GenerationResult
        """
        results = []
        batch_size = self.config.max_batch_size
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            result = self.generate(batch, max_new_tokens=max_new_tokens, **kwargs)
            
            # Split batch results
            for j in range(len(batch)):
                results.append(GenerationResult(
                    sequences=result.sequences[j:j+1],
                    scores=result.scores,
                    tokens_generated=result.tokens_generated // len(batch),
                    generation_time_ms=result.generation_time_ms,
                    tokens_per_second=result.tokens_per_second,
                ))
        
        return results
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text(s)
            
        Returns:
            Token IDs tensor
        """
        if isinstance(text, str):
            text = [text]
        
        return self._tokenize(text)
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text(s)
        """
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]
    
    def _tokenize(self, texts: List[str], pad_token_id: Optional[int] = None) -> torch.Tensor:
        """Tokenize texts."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        
        input_ids = encoded['input_ids']
        
        # Move to device
        if torch.cuda.is_available() and self.config.device == "cuda":
            input_ids = input_ids.cuda()
        
        return input_ids
    
    def _update_stats(self, generation_time_ms: float, tokens_generated: int):
        """Update generation statistics."""
        self._stats['total_requests'] += 1
        self._stats['total_tokens_generated'] += tokens_generated
        self._stats['total_generation_time_ms'] += generation_time_ms
        
        # Update averages
        n = self._stats['total_requests']
        self._stats['avg_latency_ms'] = self._stats['total_generation_time_ms'] / n
        self._stats['avg_tokens_per_second'] = (
            self._stats['total_tokens_generated'] / 
            (self._stats['total_generation_time_ms'] / 1000)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        engine_stats = self.engine.get_stats() if self.engine else {}
        
        return {
            **self._stats,
            'engine_stats': engine_stats,
            'quantization_mode': self.config.quantization_mode,
            'max_batch_size': self.config.max_batch_size,
            'max_seq_length': self.config.max_seq_length,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            'total_requests': 0,
            'total_tokens_generated': 0,
            'total_generation_time_ms': 0.0,
            'avg_latency_ms': 0.0,
            'avg_tokens_per_second': 0.0,
        }
