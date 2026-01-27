import torch
import torch.nn as nn
import os
import json
import gc
import requests
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

class LayerWeightLoader:
    """
    Virtual Model Map: reassembles layers split across multiple safetensor files.
    """
    def __init__(self, weight_map: Dict[str, str], cache_dir: str, download_fn: Any):
        self.weight_map = weight_map
        self.cache_dir = cache_dir
        self.download_fn = download_fn
        self.loaded_shards = {} # {shard_name: weights_dict}

    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        needed_shards = set()
        for weight_name, shard_name in self.weight_map.items():
            if weight_name.startswith(prefix):
                needed_shards.add(shard_name)
        
        layer_weights = {}
        for shard_name in needed_shards:
            if shard_name not in self.loaded_shards:
                shard_path = self.download_fn(shard_name)
                self.loaded_shards[shard_name] = load_file(shard_path, device="cpu")
            
            # Extract relevant weights
            for k, v in self.loaded_shards[shard_name].items():
                if k.startswith(prefix):
                    layer_weights[k.replace(prefix, "")] = v
        
        return layer_weights

    def clear_shards(self, shard_names: List[str]):
        for name in shard_names:
            if name in self.loaded_shards:
                del self.loaded_shards[name]
                shard_path = os.path.join(self.cache_dir, name)
                if os.path.exists(shard_path):
                    os.remove(shard_path)
        gc.collect()

class SequentialLayerIntegrator:
    """
    Generalized Sequential Layer Ingestion (SLI) Integrator.
    Processes layers sequentially and caches activations to SSD, allowing ingestion of massive models (1T+) on consumer hardware.
    """
    def __init__(
        self,
        model_id: str, # Must be provided explicitly (e.g. "deepseek-ai/DeepSeek-V3")
        output_dir: str = "profiles/sli_profile",
        cache_dir: str = "temp_sli_shards",
        activation_cache_dir: str = "activation_cache",
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.activation_cache_dir = activation_cache_dir
        self.device = device
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(activation_cache_dir, exist_ok=True)
        
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.weight_map = self._get_weight_map()
        self.loader = LayerWeightLoader(self.weight_map, self.cache_dir, self._download_shard)

    def _get_weight_map(self) -> Dict[str, str]:
        index_url = f"https://huggingface.co/{self.model_id}/resolve/main/model.safetensors.index.json"
        index_path = os.path.join(self.cache_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            r = requests.get(index_url)
            with open(index_path, 'w') as f: f.write(r.text)
        with open(index_path, 'r') as f:
            return json.load(f)["weight_map"]

    def _download_shard(self, shard_name: str) -> str:
        path = os.path.join(self.cache_dir, shard_name)
        if os.path.exists(path): return path
        print(f"\n[SLI] Fetching shard {shard_name}...")
        r = requests.get(f"https://huggingface.co/{self.model_id}/resolve/main/{shard_name}", stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
        return path

    def run_sli(self, dataset: List[str]):
        """
        Sequential Layer Ingestion (SLI) Loop.
        """
        num_layers = self.config.num_hidden_layers
        
        # 0. Embedding Step
        current_act_path = self._process_embeddings(dataset)
        
        for layer_idx in range(num_layers):
            print(f"\n[SLI] Processing Layer {layer_idx+1}/{num_layers}...")
            
            # 1. Load weights for this specific layer
            layer_weights = self.loader.load_layer_weights(layer_idx)
            
            # 2. Instantiate and move to GPU
            layer = self._create_layer(layer_idx)
            layer.load_state_dict(layer_weights, strict=False)
            layer.to(self.device).half().eval()
            
            # 3. Stream activations from SSD
            next_act_path = os.path.join(self.activation_cache_dir, f"layer_{layer_idx}.pt")
            self._forward_batch_sli(current_act_path, next_act_path, layer)
            
            # 4. Cleanup
            current_act_path = next_act_path
            del layer
            del layer_weights
            torch.cuda.empty_cache()
            
            # Clear shards that are no longer needed
            shards_used = set(self.weight_map[f"model.layers.{layer_idx}.{k}"] 
                             for k in ["input_layernorm.weight", "post_attention_layernorm.weight"] # Representative
                             if f"model.layers.{layer_idx}.{k}" in self.weight_map)
            self.loader.clear_shards(list(shards_used))
            gc.collect()

        print(f"[SLI] Complete. Activation cache at: {self.activation_cache_dir}")

    def _process_embeddings(self, dataset: List[str]) -> str:
        shard_name = self.weight_map["model.embed_tokens.weight"]
        path = self._download_shard(shard_name)
        weights = load_file(path, device="cpu")
        embed_layer = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        embed_layer.weight.data.copy_(weights["model.embed_tokens.weight"])
        
        act_path = os.path.join(self.activation_cache_dir, "base_embeddings.pt")
        all_acts = []
        for text in dataset:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
            with torch.no_grad(): all_acts.append(embed_layer(tokens).cpu())
        
        torch.save(torch.cat(all_acts, dim=0), act_path)
        del weights
        return act_path

    def _create_layer(self, idx: int) -> nn.Module:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer(self.config, layer_idx=idx)

    def _forward_batch_sli(self, in_path: str, out_path: str, layer: nn.Module):
        x = torch.load(in_path)
        batch_size = 1 # Extreme VRAM safety for 1T model activations
        outputs = []
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                chunk = x[i:i+batch_size].to(self.device)
                out = layer(chunk)
                if isinstance(out, tuple): out = out[0]
                outputs.append(out.cpu())
        torch.save(torch.cat(outputs, dim=0), out_path)
