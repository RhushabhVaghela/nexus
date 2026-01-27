import pytest
import os
import torch
import torch.nn as nn
import json
from src.nexus_final.sli_integrator import SequentialLayerIntegrator, load_file

class MockResponse:
    def __init__(self, content, headers=None):
        self.content = content
        self.headers = headers or {}
        try:
            self.text = content.decode('utf-8') if isinstance(content, bytes) else content
        except UnicodeDecodeError:
            self.text = "" 
    def iter_content(self, chunk_size=1):
        yield self.content
    def __enter__(self): return self
    def __exit__(self, *args): pass

@pytest.fixture
def sli_setup(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    activation_cache_dir = tmp_path / "act_cache"
    
    # Mock requests.get for config/index
    def mock_get(url, **kwargs):
        if "index.json" in url:
            return MockResponse(json.dumps({
                "weight_map": {
                    "model.embed_tokens.weight": "shard1.safetensors",
                    "model.layers.0.input_layernorm.weight": "shard2.safetensors",
                    "model.layers.0.post_attention_layernorm.weight": "shard2.safetensors"
                }
            }).encode('utf-8'))
        
        # Fake safetensor bytes
        return MockResponse(b"fake_safetensor_data", {"content-length": "20"})

    import requests
    monkeypatch.setattr(requests, "get", mock_get)
    
    # Mock AutoConfig/AutoTokenizer
    class MockConfig:
        def __init__(self):
            self.hidden_size = 128
            self.vocab_size = 1000
            self.num_hidden_layers = 1
            self.model_type = "llama"
        def to_dict(self): return {}
        
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: MockConfig())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: type('obj', (object,), {'__call__': lambda self, x, **kpb: {"input_ids": torch.zeros(1, 5).long()}})())
    
    # Mock load_file
    def mock_load_file(path, device=None):
        if "shard1" in path:
            return {"model.embed_tokens.weight": torch.randn(1000, 128)}
        return {
            "model.layers.0.input_layernorm.weight": torch.randn(128),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(128)
        }
    
    # Update patch target to new file
    monkeypatch.setattr("src.nexus_final.sli_integrator.load_file", mock_load_file)

    integrator = SequentialLayerIntegrator(
        model_id="mock/generic-model",
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        activation_cache_dir=str(activation_cache_dir)
    )
    return integrator, output_dir, cache_dir

def test_sli_flow(sli_setup, monkeypatch):
    integrator, output_dir, cache_dir = sli_setup
    
    # Mock layer creation to return Identity
    monkeypatch.setattr(integrator, "_create_layer", lambda idx: nn.Identity())
    
    # Mock embeddings to return a dummy path
    dummy_act_path = os.path.join(integrator.activation_cache_dir, "base.pt")
    torch.save(torch.randn(1, 128), dummy_act_path)
    monkeypatch.setattr(integrator, "_process_embeddings", lambda d: dummy_act_path)
    
    # Mock forward batch to avoid loading large files
    monkeypatch.setattr(integrator, "_forward_batch_sli", lambda i, o, l: torch.save(torch.randn(1, 128), o))
    
    # Run SLI
    dummy_dataset = ["test text"]
    integrator.run_sli(dummy_dataset)
    
    # Verify shard was cleared by the integrator/loader
    # shard1 is typically handled by _process_embeddings
    # shard2 is cleared by loader.clear_shards
    assert not os.path.exists(os.path.join(cache_dir, "shard1.safetensors"))
    assert not os.path.exists(os.path.join(cache_dir, "shard2.safetensors"))

def test_loader_weight_mapping(sli_setup):
    integrator, _, _ = sli_setup
    # Verify mapping works
    assert integrator.weight_map["model.layers.0.input_layernorm.weight"] == "shard2.safetensors"
