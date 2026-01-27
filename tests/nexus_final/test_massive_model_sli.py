import pytest
import torch
import os
from unittest.mock import MagicMock, patch
from src.nexus_final.sli_integrator import LayerWeightLoader

def test_layer_weight_loader_reassembly():
    # Mock weight map: Layer 0 is split between shard_A and shard_B
    weight_map = {
        "model.layers.0.input_layernorm.weight": "shard_A.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "shard_A.safetensors",
        "model.layers.0.self_attn.k_proj.weight": "shard_B.safetensors",
        "model.layers.0.post_attention_layernorm.weight": "shard_B.safetensors"
    }
    
    cache_dir = "temp_cache"
    
    # Mock download function
    def mock_download(shard_name):
        return f"/mock/path/{shard_name}"

    # Mock load_file (safetensors)
    def mock_load_file(path, device="cpu"):
        if "shard_A" in path:
            return {
                "model.layers.0.input_layernorm.weight": torch.ones(4096),
                "model.layers.0.self_attn.q_proj.weight": torch.ones(4096, 4096)
            }
        elif "shard_B" in path:
            return {
                "model.layers.0.self_attn.k_proj.weight": torch.ones(4096, 4096),
                "model.layers.0.post_attention_layernorm.weight": torch.ones(4096)
            }
        return {}

    with patch("src.nexus_final.sli_integrator.load_file", side_effect=mock_load_file):
        loader = LayerWeightLoader(weight_map, cache_dir, mock_download)
        
        # Act
        layer_weights = loader.load_layer_weights(0)
        
        # Assert
        assert "input_layernorm.weight" in layer_weights
        assert "self_attn.q_proj.weight" in layer_weights
        assert "self_attn.k_proj.weight" in layer_weights
        assert "post_attention_layernorm.weight" in layer_weights
        assert layer_weights["input_layernorm.weight"].shape == (4096,)
        
        # Verify both shards were 'loaded' into internal state
        assert "shard_A.safetensors" in loader.loaded_shards
        assert "shard_B.safetensors" in loader.loaded_shards

if __name__ == "__main__":
    pytest.main([__file__])
