"""
Universal Weight Loader for SLI

Handles loading weights from multiple formats (safetensors, bin, pt)
and provides family-specific weight mapping.
"""

import os
import json
import gc
from typing import Dict, List, Optional, Set, Callable
from pathlib import Path

import torch
import requests
from safetensors.torch import load_file as safetensors_load

from .architecture_registry import ArchitectureFamily
from .exceptions import FormatDetectionError, WeightMapError, WeightLoadingError


class UniversalWeightLoader:
    """
    Universal weight loading with format detection and caching.
    
    Supports:
    - SafeTensors (.safetensors)
    - PyTorch bin files (.bin)
    - PyTorch checkpoints (.pt, .pth)
    """
    
    SUPPORTED_FORMATS = [".safetensors", ".bin", ".pt", ".pth"]
    
    def __init__(
        self,
        cache_dir: str,
        model_id: str,
        download_fn: Optional[Callable] = None
    ):
        """
        Initialize the weight loader.
        
        Args:
            cache_dir: Directory for caching downloaded weights
            model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")
            download_fn: Optional custom download function
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.download_fn = download_fn or self._default_download
        
        self.format = self._detect_format()
        self.weight_map = self._load_weight_map()
        self.loaded_shards: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def _detect_format(self) -> str:
        """
        Auto-detect weight format from available files.
        
        Returns:
            Format extension (e.g., ".safetensors")
            
        Raises:
            FormatDetectionError: If no supported format is found
        """
        # Check for index files first (sharded models)
        for ext in ["safetensors", "bin"]:
            index_name = f"model.{ext}.index.json"
            index_url = f"https://huggingface.co/{self.model_id}/resolve/main/{index_name}"
            if self._check_url_exists(index_url):
                return f".{ext}"
        
        # Check for single file weights
        for ext in self.SUPPORTED_FORMATS:
            if ext == ".safetensors":
                filenames = [
                    "model.safetensors",
                    "pytorch_model.safetensors",
                    "model-00001-of-00002.safetensors"
                ]
            elif ext == ".bin":
                filenames = [
                    "pytorch_model.bin",
                    "model.bin"
                ]
            else:
                filenames = [f"pytorch_model{ext}"]
            
            for filename in filenames:
                url = f"https://huggingface.co/{self.model_id}/resolve/main/{filename}"
                if self._check_url_exists(url):
                    return ext
        
        attempted = [f"model.{ext}.index.json" for ext in ["safetensors", "bin"]]
        attempted.extend([f"pytorch_model{ext}" for ext in self.SUPPORTED_FORMATS])
        raise FormatDetectionError(self.model_id, attempted)
    
    def _check_url_exists(self, url: str) -> bool:
        """Check if a URL exists without downloading."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _load_weight_map(self) -> Dict[str, str]:
        """
        Load the weight map from index file.
        
        Returns:
            Dictionary mapping weight names to shard filenames
        """
        # Try index file first
        index_name = f"model{self.format}.index.json"
        index_path = self.cache_dir / index_name
        
        # Download if not cached
        if not index_path.exists():
            index_url = f"https://huggingface.co/{self.model_id}/resolve/main/{index_name}"
            try:
                response = requests.get(index_url, timeout=30)
                response.raise_for_status()
                with open(index_path, 'w') as f:
                    f.write(response.text)
            except Exception as e:
                # No index file, assume single file model
                if self.format == ".safetensors":
                    default_shard = "model.safetensors"
                else:
                    default_shard = f"pytorch_model{self.format}"
                return {"__all__": default_shard}
        
        # Load index
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            return index_data.get("weight_map", {})
        except (json.JSONDecodeError, KeyError) as e:
            raise WeightMapError(self.model_id, str(index_path)) from e
    
    def _default_download(self, shard_name: str) -> str:
        """
        Download a shard file if not cached.
        
        Args:
            shard_name: Name of the shard file
            
        Returns:
            Path to the cached shard file
        """
        shard_path = self.cache_dir / shard_name
        
        if shard_path.exists():
            return str(shard_path)
        
        # Create subdirectory if needed
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        url = f"https://huggingface.co/{self.model_id}/resolve/main/{shard_name}"
        print(f"\n[SLI] Fetching shard {shard_name}...")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(shard_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r[SLI] Download progress: {progress:.1f}%", end="")
        
        print()  # New line after progress
        return str(shard_path)
    
    def _load_shard(self, shard_name: str) -> Dict[str, torch.Tensor]:
        """
        Load a shard file into memory.
        
        Args:
            shard_name: Name of the shard file
            
        Returns:
            Dictionary of weight tensors
        """
        if shard_name in self.loaded_shards:
            return self.loaded_shards[shard_name]
        
        shard_path = self.download_fn(shard_name)
        
        # Load based on format
        if shard_path.endswith(".safetensors"):
            weights = safetensors_load(shard_path, device="cpu")
        else:
            # PyTorch format
            weights = torch.load(shard_path, map_location="cpu", weights_only=True)
            # Handle pickled models that may be wrapped
            if "model" in weights and isinstance(weights["model"], dict):
                weights = weights["model"]
            elif "state_dict" in weights:
                weights = weights["state_dict"]
        
        self.loaded_shards[shard_name] = weights
        return weights
    
    def load_layer_weights(
        self,
        layer_idx: int,
        family: ArchitectureFamily
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights for a specific layer.
        
        Args:
            layer_idx: Layer index
            family: Architecture family handler
            
        Returns:
            Dictionary of weight tensors for the layer
        """
        prefix = family.get_layer_prefix(layer_idx)
        
        # Find all weights matching this prefix
        needed_shards: Set[str] = set()
        matching_weights = {}
        
        for weight_name, shard_name in self.weight_map.items():
            if weight_name.startswith(prefix):
                needed_shards.add(shard_name)
                matching_weights[weight_name] = shard_name
        
        if not needed_shards:
            # Try alternative prefix patterns for compatibility
            alt_prefixes = [
                f"transformer.h.{layer_idx}.",
                f"model.decoder.layers.{layer_idx}.",
                f"backbone.layers.{layer_idx}.",
            ]
            for alt_prefix in alt_prefixes:
                for weight_name, shard_name in self.weight_map.items():
                    if weight_name.startswith(alt_prefix):
                        needed_shards.add(shard_name)
                        matching_weights[weight_name] = shard_name
        
        # Load weights from shards
        layer_weights = {}
        for shard_name in needed_shards:
            shard_weights = self._load_shard(shard_name)
            
            for weight_name, shard in matching_weights.items():
                if shard == shard_name and weight_name in shard_weights:
                    # Strip prefix from weight name
                    normalized_name = weight_name.replace(prefix, "")
                    layer_weights[normalized_name] = shard_weights[weight_name]
        
        return layer_weights
    
    def load_embedding_weights(self, family: ArchitectureFamily) -> torch.Tensor:
        """
        Load embedding weights.
        
        Args:
            family: Architecture family handler
            
        Returns:
            Embedding weight tensor
        """
        embedding_name = family.get_embedding_name()
        weight_key = f"{embedding_name}.weight"
        
        # Find shard containing embedding
        shard_name = self.weight_map.get(weight_key)
        if not shard_name:
            # Try common variations
            variations = [
                "model.embed_tokens.weight",
                "transformer.wte.weight",
                "transformer.word_embeddings.weight",
                "shared.weight",
                "model.decoder.embed_tokens.weight",
            ]
            for var in variations:
                if var in self.weight_map:
                    shard_name = self.weight_map[var]
                    weight_key = var
                    break
        
        if not shard_name:
            raise WeightLoadingError(weight_key, None, 
                Exception("Embedding weight not found in weight map"))
        
        shard_weights = self._load_shard(shard_name)
        return shard_weights[weight_key]
    
    def clear_shards(self, shard_names: Optional[List[str]] = None):
        """
        Clear loaded shards from memory.
        
        Args:
            shard_names: Specific shards to clear, or None to clear all
        """
        if shard_names is None:
            shard_names = list(self.loaded_shards.keys())
        
        for name in shard_names:
            if name in self.loaded_shards:
                del self.loaded_shards[name]
                # Also delete cached file if desired
                shard_path = self.cache_dir / name
                if shard_path.exists():
                    os.remove(shard_path)
        
        gc.collect()
    
    def get_weight_info(self) -> Dict[str, any]:
        """Get information about loaded weights."""
        return {
            "format": self.format,
            "num_shards": len(set(self.weight_map.values())),
            "num_weights": len(self.weight_map),
            "loaded_shards": list(self.loaded_shards.keys()),
        }
