# Universal SLI (Sequential Layer Ingestion) Module

**Version:** 1.0.0  
**Supported Architectures:** 135+ models across 12 families

The Universal SLI module enables processing of massive language models (100B - 1T+ parameters) on consumer hardware by streaming layers sequentially and caching activations to SSD.

---

## Quick Start

```python
from src.nexus_final.sli import UniversalSLIIntegrator

# Initialize for any supported model
integrator = UniversalSLIIntegrator("mistralai/Mistral-7B-v0.1")

# Process dataset
dataset = ["Sample text for processing"]
result = integrator.run_sli(dataset)

print(f"Processed {result['num_layers']} layers")
```

---

## Architecture Support

### 12 Architecture Families (135+ Models)

| Family | Count | Examples |
|--------|-------|----------|
| **Llama** | 35 | Llama, Mistral, Mixtral, Qwen2, DeepSeek |
| **GPT** | 18 | GPT-2, GPT-J, GPT-NeoX, Falcon, Pythia |
| **Qwen** | 14 | Qwen2, Qwen2.5, Qwen3, Qwen-VL, Qwen-Omni |
| **MoE** | 15 | Mixtral, DeepSeek-MoE, Qwen2-MoE, Grok |
| **Encoder** | 16 | BERT, RoBERTa, DeBERTa, ModernBERT |
| **T5** | 12 | T5, FLAN-T5, UL2, LongT5 |
| **Mamba** | 12 | Mamba, Mamba2, Jamba, Zamba, RWKV |
| **Gemma** | 8 | Gemma, Gemma2, Gemma3 |
| **ChatGLM** | 8 | ChatGLM, ChatGLM3, GLM-4, GLM-4-MoE |
| **Phi** | 6 | Phi, Phi2, Phi3, Phi4 |
| **BLOOM** | 5 | BLOOM, BLOOMZ |
| **OPT** | 6 | OPT, OPT-IML |

---

## Installation

The SLI module is part of the Nexus package:

```bash
# Clone the repository
git clone <nexus-repo>
cd nexus

# Install dependencies
pip install -r requirements.txt
```

Required dependencies:

- `torch>=2.0.0`
- `transformers>=4.35.0`
- `safetensors>=0.4.0`
- `tqdm`

---

## Usage Examples

### Basic Usage

```python
from src.nexus_final.sli import UniversalSLIIntegrator

integrator = UniversalSLIIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards",
    device="cuda"
)

dataset = ["Text sample 1", "Text sample 2"]
result = integrator.run_sli(dataset, batch_size=1)
```

### GPT Models

```python
# GPT-2
integrator = UniversalSLIIntegrator("gpt2")

# GPT-J
integrator = UniversalSLIIntegrator("EleutherAI/gpt-j-6B")

# Falcon
integrator = UniversalSLIIntegrator("tiiuae/falcon-7b")
```

### T5 Models (Encoder-Decoder)

```python
# FLAN-T5
integrator = UniversalSLIIntegrator("google/flan-t5-base")

# UL2
integrator = UniversalSLIIntegrator("google/ul2")
```

### MoE Models

```python
# Mixtral
integrator = UniversalSLIIntegrator("mistralai/Mixtral-8x7B-v0.1")

# DeepSeek-MoE
integrator = UniversalSLIIntegrator("deepseek-ai/deepseek-moe-16b-base")

# Get MoE info
summary = integrator.get_model_summary()
if summary['is_moe']:
    print(f"Experts: {summary['moe_info']['num_experts']}")
```

### Mamba/State Space Models

```python
# Mamba
integrator = UniversalSLIIntegrator("state-spaces/mamba-370m")

# Mamba2
integrator = UniversalSLIIntegrator("state-spaces/mamba2-780m")

# Jamba
integrator = UniversalSLIIntegrator("ai21labs/Jamba-v0.1")
```

### Custom Architectures (Trust Remote Code)

```python
# ChatGLM
integrator = UniversalSLIIntegrator(
    "THUDM/chatglm3-6b",
    trust_remote_code=True
)

# GLM-4
integrator = UniversalSLIIntegrator(
    "THUDM/glm-4-9b",
    trust_remote_code=True
)
```

---

## Module Components

### Core Classes

| Class | Purpose |
|-------|---------|
| [`UniversalSLIIntegrator`](universal_sli_integrator.py:33) | Main SLI pipeline orchestrator |
| [`ArchitectureRegistry`](architecture_registry.py:562) | Registry of supported architectures |
| [`UniversalLayerFactory`](layer_factory.py:20) | Creates architecture-specific layers |
| [`UniversalWeightLoader`](weight_loader.py:22) | Loads weights from multiple formats |
| [`MoEHandler`](moe_handler.py:39) | Handles MoE-specific operations |

### Architecture Family Handlers

| Handler | Family | Models |
|---------|--------|--------|
| [`LlamaFamilyHandler`](architecture_registry.py:121) | Llama | Llama, Mistral, Mixtral, Qwen2, DeepSeek |
| [`GPTFamilyHandler`](architecture_registry.py:186) | GPT | GPT-2, GPT-J, GPT-NeoX, Falcon |
| [`QwenFamilyHandler`](architecture_registry.py:155) | Qwen | Qwen2, Qwen2.5, Qwen3 |
| [`T5FamilyHandler`](architecture_registry.py:302) | T5 | T5, FLAN-T5, UL2, LongT5 |
| [`BLOOMFamilyHandler`](architecture_registry.py:331) | BLOOM | BLOOM, BLOOMZ |
| [`OPTFamilyHandler`](architecture_registry.py:354) | OPT | OPT, OPT-IML |
| [`MambaFamilyHandler`](architecture_registry.py:377) | Mamba | Mamba, Mamba2, Jamba, Zamba |
| [`MoEFamilyHandler`](architecture_registry.py:432) | MoE | Mixtral, DeepSeek-MoE, Qwen2-MoE |
| [`PhiFamilyHandler`](architecture_registry.py:514) | Phi | Phi, Phi2, Phi3, Phi4 |
| [`GemmaFamilyHandler`](architecture_registry.py:537) | Gemma | Gemma, Gemma2, Gemma3 |
| [`ChatGLMFamilyHandler`](architecture_registry.py:259) | ChatGLM | ChatGLM, GLM-4 |

### Exceptions

| Exception | Purpose |
|-----------|---------|
| [`SLIError`](exceptions.py:6) | Base exception |
| [`UnsupportedArchitectureError`](exceptions.py:11) | Unknown architecture |
| [`WeightLoadingError`](exceptions.py:24) | Weight loading failed |
| [`LayerCreationError`](exceptions.py:39) | Layer creation failed |
| [`MoEConfigurationError`](exceptions.py:52) | Invalid MoE config |

---

## Advanced Usage

### Using the Layer Factory

```python
from src.nexus_final.sli import UniversalLayerFactory, get_registry
from transformers import AutoConfig

config = AutoConfig.from_pretrained("gpt2")
factory = UniversalLayerFactory()

# Create layer
layer = factory.create_layer(config, layer_idx=0)

# Get model info
info = factory.get_model_info(config)
print(f"Layers: {info['num_layers']}")
```

### Using the Weight Loader

```python
from src.nexus_final.sli import UniversalWeightLoader, get_registry

loader = UniversalWeightLoader("temp_shards", "gpt2")

# Get format info
info = loader.get_weight_info()
print(f"Format: {info['format']}")

# Load layer weights
registry = get_registry()
config = AutoConfig.from_pretrained("gpt2")
family = registry.detect_family(config)

weights = loader.load_layer_weights(layer_idx=0, family=family)
```

### Architecture Registry

```python
from src.nexus_final.sli import get_registry

registry = get_registry()

# List families
families = registry.list_families()

# Get family info
info = registry.get_family_info()

# Detect family
family = registry.detect_family(config)
```

### Custom Architecture Registration

```python
from src.nexus_final.sli import (
    ArchitectureRegistry,
    ArchitectureFamily,
    get_registry
)

class CustomFamilyHandler(ArchitectureFamily):
    family_id = "custom"
    family_name = "Custom Architecture"
    model_types = ["custom_model"]
    architectures = ["CustomForCausalLM"]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"custom.layers.{layer_idx}."
    
    def create_layer(self, config, layer_idx: int, layer_type: str = "decoder"):
        from my_module import CustomLayer
        return CustomLayer(config, layer_idx)

# Register
registry = get_registry()
registry.register("custom", CustomFamilyHandler())
```

---

## Performance Tips

### Batch Size

```python
# Small models: larger batches
integrator.run_sli(dataset, batch_size=4)

# Large models: smaller batches
integrator.run_sli(dataset, batch_size=1)
```

### SSD Cache

```python
# Use fast SSD for activation cache
integrator = UniversalSLIIntegrator(
    model_id="...",
    activation_cache_dir="/fast_ssd/activation_cache"
)
```

### Clear Cache

```python
# Clear shard cache
integrator.weight_loader.clear_shards()

# Or clear all cache
integrator.clear_cache()
```

---

## Testing

Run the test suite:

```bash
# Run all SLI tests
pytest tests/unit/sli/ -v

# Run integration tests
pytest tests/integration/sli/ -v

# Run specific test
pytest tests/unit/sli/test_universal_sli_integrator.py -v
```

---

## Documentation

- [Universal SLI Guide](../../docs/SLI_UNIVERSAL_GUIDE.md) - Complete guide to all features
- [Migration Guide](../../docs/MIGRATION_GUIDE.md) - Migrating from SequentialLayerIntegrator
- [Technical Manual](../../docs/NEXUS_V6_TECHNICAL_MANUAL.md) - Technical implementation details

---

## Architecture Taxonomy

See [`plans/architecture_taxonomy.json`](../../plans/architecture_taxonomy.json) for the complete machine-readable list of supported architectures.

---

## License

Part of the Nexus project. See LICENSE file for details.
