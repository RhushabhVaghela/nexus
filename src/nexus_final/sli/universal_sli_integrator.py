"""
Universal Sequential Layer Ingestion (SLI) Integrator

Processes layers sequentially and caches activations to SSD, allowing ingestion
of massive models (1T+) on consumer hardware. Supports 130+ model architectures.

This is a drop-in replacement for the legacy SequentialLayerIntegrator with
universal architecture support.
"""

import os
import json
import gc
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .architecture_registry import (
    ArchitectureRegistry,
    ArchitectureFamily,
    get_registry
)
from .layer_factory import UniversalLayerFactory
from .weight_loader import UniversalWeightLoader
from .moe_handler import MoEHandler
from .exceptions import SLIError, UnsupportedArchitectureError


class UniversalSLIIntegrator:
    """
    Universal Sequential Layer Ingestion (SLI) Integrator.
    
    Processes layers sequentially and caches activations to SSD, allowing
    ingestion of massive models (1T+) on consumer hardware.
    
    Supports 130+ model architectures across families:
    - Llama-based (Llama, Mistral, Mixtral, Qwen2, etc.)
    - GPT-based (GPT-2, GPT-J, GPT-NeoX, etc.)
    - ChatGLM-based (ChatGLM, GLM-4, etc.)
    - T5-based (T5, FLAN-T5, UL2, etc.)
    - BLOOM-based
    - OPT-based
    - Mamba/State Space Models
    - MoE Architectures (Mixtral, DeepSeek-MoE, etc.)
    
    Usage:
        integrator = UniversalSLIIntegrator("mistralai/Mistral-7B-v0.1")
        integrator.run_sli(dataset)
    """
    
    def __init__(
        self,
        model_id: str,
        output_dir: str = "profiles/sli_profile",
        cache_dir: str = "temp_sli_shards",
        activation_cache_dir: str = "activation_cache",
        device: str = "cuda",
        trust_remote_code: bool = True,
        registry: Optional[ArchitectureRegistry] = None
    ):
        """
        Initialize the Universal SLI Integrator.
        
        Args:
            model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")
            output_dir: Output directory for profiles
            cache_dir: Directory for caching weight shards
            activation_cache_dir: Directory for caching activations
            device: Device to use ("cuda" or "cpu")
            trust_remote_code: Whether to trust remote code in models
            registry: Optional custom ArchitectureRegistry
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.activation_cache_dir = Path(activation_cache_dir)
        self.device = device
        self.trust_remote_code = trust_remote_code
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.activation_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config and tokenizer
        print(f"[SLI] Loading config for {model_id}...")
        self.config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        print(f"[SLI] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize architecture components
        self.registry = registry or get_registry()
        self.factory = UniversalLayerFactory(self.registry)
        
        # Detect architecture family
        print(f"[SLI] Detecting architecture family...")
        try:
            self.family = self.registry.detect_family(self.config)
            print(f"[SLI] Detected family: {self.family.family_name}")
        except UnsupportedArchitectureError as e:
            print(f"[SLI] Warning: {e}")
            print(f"[SLI] Falling back to Llama family")
            self.family = self.registry.get_family("llama")
        
        # Initialize weight loader
        self.weight_loader = UniversalWeightLoader(
            str(self.cache_dir),
            model_id
        )
        
        # Initialize MoE handler if applicable
        self.moe_handler = None
        if self._is_moe_model():
            print(f"[SLI] Detected MoE architecture")
            self.moe_handler = MoEHandler(self.config)
        
        # Get model info
        self.model_info = self.factory.get_model_info(self.config)
        print(f"[SLI] Model info: {self.model_info}")
    
    def _is_moe_model(self) -> bool:
        """Check if model is MoE."""
        moe_attrs = ["num_local_experts", "n_routed_experts", "moe_intermediate_size"]
        return any(hasattr(self.config, attr) for attr in moe_attrs)
    
    def run_sli(self, dataset: List[str], batch_size: int = 1):
        """
        Run Sequential Layer Ingestion (SLI) pipeline.
        
        Args:
            dataset: List of text samples to process
            batch_size: Batch size for processing
        """
        num_layers = self.model_info["num_layers"]
        print(f"\n[SLI] Starting SLI for {self.model_id}")
        print(f"[SLI] Total layers: {num_layers}")
        print(f"[SLI] Dataset size: {len(dataset)}")
        
        # Step 0: Process embeddings
        print("\n[SLI] Step 0: Processing embeddings...")
        current_act_path = self._process_embeddings(dataset)
        
        # Process each layer
        for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
            print(f"\n[SLI] Processing Layer {layer_idx+1}/{num_layers}...")
            
            # Create layer
            layer = self._create_layer(layer_idx)
            
            # Load weights
            layer_weights = self._load_layer_weights(layer_idx)
            layer.load_state_dict(layer_weights, strict=False)
            layer.to(self.device).half().eval()
            
            # Forward pass
            next_act_path = self.activation_cache_dir / f"layer_{layer_idx}.pt"
            self._forward_batch_sli(current_act_path, str(next_act_path), layer, batch_size)
            
            # Cleanup
            current_act_path = next_act_path
            del layer
            del layer_weights
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"\n[SLI] Complete!")
        print(f"[SLI] Activation cache at: {self.activation_cache_dir}")
        
        return {
            "activation_cache_dir": str(self.activation_cache_dir),
            "num_layers": num_layers,
            "model_info": self.model_info,
        }
    
    def _create_layer(self, layer_idx: int) -> nn.Module:
        """
        Create layer instance for given index.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Layer module
        """
        return self.factory.create_layer(self.config, layer_idx)
    
    def _load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load weights for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dictionary of weight tensors
        """
        return self.weight_loader.load_layer_weights(layer_idx, self.family)
    
    def _process_embeddings(self, dataset: List[str]) -> Path:
        """
        Process input texts through embedding layer.
        
        Args:
            dataset: List of text samples
            
        Returns:
            Path to saved embeddings file
        """
        # Load embedding weights
        embedding_weights = self.weight_loader.load_embedding_weights(self.family)
        
        vocab_size = self.model_info["vocab_size"]
        hidden_size = self.model_info["hidden_size"]
        
        # Create embedding layer
        embed_layer = nn.Embedding(vocab_size, hidden_size)
        embed_layer.weight.data.copy_(embedding_weights)
        embed_layer.to(self.device)
        
        # Process dataset
        all_acts = []
        for text in dataset:
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )["input_ids"]
            
            with torch.no_grad():
                acts = embed_layer(tokens.to(self.device))
                all_acts.append(acts.cpu())
        
        # Save embeddings
        embeddings = torch.cat(all_acts, dim=0)
        output_path = self.activation_cache_dir / "base_embeddings.pt"
        torch.save(embeddings, output_path)
        
        del embed_layer
        torch.cuda.empty_cache()
        
        return output_path
    
    def _forward_batch_sli(
        self,
        in_path: Path,
        out_path: str,
        layer: nn.Module,
        batch_size: int = 1
    ):
        """
        Forward pass with batching and SSD caching.
        
        Args:
            in_path: Path to input activations
            out_path: Path to save output activations
            layer: Layer module
            batch_size: Batch size for processing
        """
        x = torch.load(in_path)
        outputs = []
        
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                chunk = x[i:i+batch_size].to(self.device)
                out = layer(chunk)
                
                # Handle tuple outputs (some models return (hidden_state, ...))
                if isinstance(out, tuple):
                    out = out[0]
                
                outputs.append(out.cpu())
        
        # Save outputs
        torch.save(torch.cat(outputs, dim=0), out_path)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model configuration."""
        return {
            "model_id": self.model_id,
            "family": self.family.family_id,
            "family_name": self.family.family_name,
            "num_layers": self.model_info["num_layers"],
            "hidden_size": self.model_info["hidden_size"],
            "vocab_size": self.model_info["vocab_size"],
            "is_moe": self.moe_handler is not None,
            "moe_info": self.moe_handler.get_model_info() if self.moe_handler else None,
        }
    
    def clear_cache(self):
        """Clear all cached files."""
        import shutil
        
        # Clear weight cache
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear activation cache
        if self.activation_cache_dir.exists():
            shutil.rmtree(self.activation_cache_dir)
            self.activation_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("[SLI] Cache cleared")


# Legacy compatibility wrapper
class SequentialLayerIntegrator(UniversalSLIIntegrator):
    """
    Backward-compatible wrapper for legacy SequentialLayerIntegrator.
    
    This provides a migration path for existing code using the old API.
    The new UniversalSLIIntegrator should be used for new code.
    """
    
    def __init__(self, *args, **kwargs):
        # Warn about deprecation
        print("[SLI] Warning: SequentialLayerIntegrator is deprecated.")
        print("[SLI] Please use UniversalSLIIntegrator for new code.")
        super().__init__(*args, **kwargs)
