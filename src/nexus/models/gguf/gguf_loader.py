"""
GGUF Loader - Load and run GGUF models via llama.cpp

Supports models from unsloth, TheBloke, and other GGUF creators.
Optimized for Kimi-K2.5-GGUF and other large models.
"""

import torch
from typing import Optional, List, Dict, Any, Union, Generator
from dataclasses import dataclass
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class GGUFConfig:
    """Configuration for GGUF model loading."""
    model_path: str
    n_ctx: int = 8192
    n_batch: int = 512
    n_gpu_layers: int = -1  # -1 = offload all to GPU
    n_threads: int = -1  # -1 = use all cores
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Verbosity
    verbose: bool = False
    
    # Optional chat template
    chat_format: Optional[str] = None  # "chatml", "llama-2", etc.


class GGUfLoader:
    """
    Loader for GGUF format models via llama-cpp-python.
    
    Supports:
    - CPU inference
    - GPU offloading (CUDA, Metal, ROCm)
    - Multi-threading
    - Chat and completion modes
    """
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.model = None
        self._llama_module = None
    
    def _ensure_llama_cpp(self):
        """Ensure llama-cpp-python is installed."""
        try:
            import llama_cpp
            self._llama_module = llama_cpp
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF support. "
                "Install with: pip install llama-cpp-python"
            )
    
    def load(self) -> "GGUfLoader":
        """Load the GGUF model."""
        self._ensure_llama_cpp()
        
        logger.info(f"Loading GGUF model: {self.config.model_path}")
        
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        
        # Determine number of threads
        n_threads = self.config.n_threads
        if n_threads == -1:
            import multiprocessing
            n_threads = multiprocessing.cpu_count()
        
        # Load model
        load_kwargs = {
            "model_path": self.config.model_path,
            "n_ctx": self.config.n_ctx,
            "n_batch": self.config.n_batch,
            "verbose": self.config.verbose,
            "n_threads": n_threads,
        }
        
        # GPU offloading
        if self.config.n_gpu_layers != 0:
            load_kwargs["n_gpu_layers"] = self.config.n_gpu_layers
            logger.info(f"GPU layers: {self.config.n_gpu_layers}")
        
        # Chat format
        if self.config.chat_format:
            load_kwargs["chat_format"] = self.config.chat_format
        
        try:
            self.model = self._llama_module.Llama(**load_kwargs)
            logger.info("GGUF model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise
        
        return self
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Generate text from a prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        gen_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "top_k": top_k or self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "stream": stream,
        }
        
        if stop:
            gen_kwargs["stop"] = stop
        
        gen_kwargs.update(kwargs)
        
        if stream:
            return self._stream_generate(**gen_kwargs)
        
        output = self.model(**gen_kwargs)
        
        return {
            "text": output["choices"][0]["text"],
            "tokens": output["usage"]["completion_tokens"],
            "prompt_tokens": output["usage"]["prompt_tokens"],
        }
    
    def _stream_generate(self, **kwargs) -> Generator[str, None, None]:
        """Stream generation output."""
        kwargs["stream"] = True
        
        for chunk in self.model(**kwargs):
            text = chunk["choices"][0]["text"]
            if text:
                yield text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """Generate chat response from messages."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        gen_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature or self.config.temperature,
            "stream": stream,
        }
        gen_kwargs.update(kwargs)
        
        if stream:
            return self._stream_chat(**gen_kwargs)
        
        output = self.model.create_chat_completion(**gen_kwargs)
        
        return {
            "content": output["choices"][0]["message"]["content"],
            "role": output["choices"][0]["message"]["role"],
            "tokens": output["usage"]["completion_tokens"],
        }
    
    def _stream_chat(self, **kwargs) -> Generator[str, None, None]:
        """Stream chat output."""
        kwargs["stream"] = True
        
        for chunk in self.model.create_chat_completion(**kwargs):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"]
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.tokenize(text.encode("utf-8"))
    
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize token IDs into text."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.detokenize(tokens).decode("utf-8", errors="ignore")
    
    def get_context_size(self) -> int:
        """Get the model's context window size."""
        return self.config.n_ctx
    
    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("GGUF model unloaded")
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
    
    @staticmethod
    def list_gguf_files(directory: str) -> List[str]:
        """List all GGUF files in a directory."""
        path = Path(directory)
        return [str(f) for f in path.glob("**/*.gguf") if f.is_file()]
    
    @staticmethod
    def get_model_info(gguf_path: str) -> Dict[str, Any]:
        """Get metadata from a GGUF file."""
        try:
            from llama_cpp import Llama
            
            model = Llama(model_path=gguf_path, n_ctx=512, verbose=False)
            
            info = {
                "n_vocab": model.n_vocab(),
                "n_ctx": model.n_ctx(),
                "n_embd": model.n_embd(),
                "n_layer": model.n_layer(),
            }
            
            del model
            return info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}


class GGUFBatchLoader:
    """Load and manage multiple GGUF models."""
    
    def __init__(self):
        self.models: Dict[str, GGUfLoader] = {}
    
    def load_model(self, name: str, config: GGUFConfig) -> GGUfLoader:
        loader = GGUfLoader(config)
        loader.load()
        self.models[name] = loader
        return loader
    
    def get_model(self, name: str) -> Optional[GGUfLoader]:
        return self.models.get(name)
    
    def unload_model(self, name: str):
        if name in self.models:
            self.models[name].unload()
            del self.models[name]
    
    def unload_all(self):
        for loader in self.models.values():
            loader.unload()
        self.models.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all()
