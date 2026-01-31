# Migration Guide: SequentialLayerIntegrator to UniversalSLIIntegrator

**Version:** 1.0  
**Last Updated:** 2026-01-31  
**Impact:** All SLI-based pipelines

---

## Overview

The legacy [`SequentialLayerIntegrator`](src/nexus_final/sli/universal_sli_integrator.py:321) has been superseded by the new [`UniversalSLIIntegrator`](src/nexus_final/sli/universal_sli_integrator.py:33). This guide provides step-by-step instructions for migrating your code.

### Why Migrate?

| Feature | Legacy `SequentialLayerIntegrator` | New `UniversalSLIIntegrator` |
|---------|-----------------------------------|------------------------------|
| **Architecture Support** | Llama only | 135+ architectures |
| **Auto-Detection** | Manual | Automatic |
| **MoE Support** | ❌ None | ✅ Full support |
| **Weight Formats** | SafeTensors only | SafeTensors, .bin, .pt, .pth |
| **Family Handlers** | Hardcoded | Pluggable registry |
| **Maintenance** | Legacy | Active development |

---

## Quick Migration

### Before (Legacy)

```python
from src.nexus_final.sli import SequentialLayerIntegrator

integrator = SequentialLayerIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards"
)

result = integrator.run_sli(dataset)
```

### After (Universal)

```python
from src.nexus_final.sli import UniversalSLIIntegrator

integrator = UniversalSLIIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",  # Same - any architecture now works!
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards"
)

result = integrator.run_sli(dataset)
```

**The API is fully backward-compatible!** Simply change the class name.

---

## Detailed API Changes

### Class Instantiation

#### Legacy (Still Works with Deprecation Warning)

```python
from src.nexus_final.sli import SequentialLayerIntegrator

integrator = SequentialLayerIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards",
    activation_cache_dir="activation_cache",
    device="cuda"
)
```

**Output:**

```
[SLI] Warning: SequentialLayerIntegrator is deprecated.
[SLI] Please use UniversalSLIIntegrator for new code.
```

#### New (Recommended)

```python
from src.nexus_final.sli import UniversalSLIIntegrator

integrator = UniversalSLIIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",  # Now supports any architecture
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards",
    activation_cache_dir="activation_cache",
    device="cuda",
    trust_remote_code=True,  # New: for custom architectures
    registry=None  # New: optional custom registry
)
```

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trust_remote_code` | `bool` | `True` | Allow custom modeling files (e.g., ChatGLM) |
| `registry` | `ArchitectureRegistry` | `None` | Custom architecture registry |

---

## Breaking Changes

### 1. Internal Architecture

The internal structure has changed significantly:

#### Legacy Structure

```python
class SequentialLayerIntegrator:
    - Hardcoded Llama-specific layer creation
    - Hardcoded weight naming: "model.layers.{idx}."
    - Manual embedding access patterns
    - No architecture detection
```

#### New Structure

```python
class UniversalSLIIntegrator:
    - ArchitectureRegistry: Auto-detects family
    - UniversalLayerFactory: Creates correct layers
    - UniversalWeightLoader: Handles any format
    - MoEHandler: Manages expert routing
```

### 2. Layer Creation

#### Legacy (Internal - Not User-Facing)

```python
# Hardcoded in legacy integrator
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
layer = LlamaDecoderLayer(config, layer_idx=layer_idx)
```

#### New (Via Factory)

```python
# Automatic via factory
layer = self.factory.create_layer(self.config, layer_idx)
# Returns correct type: LlamaDecoderLayer, GPT2Block, T5Block, etc.
```

### 3. Weight Loading

#### Legacy (Internal)

```python
# Hardcoded prefix
prefix = f"model.layers.{layer_idx}."
# Only Llama naming convention
```

#### New (Via Family Handler)

```python
# Dynamic based on architecture family
family = self.registry.detect_family(self.config)
prefix = family.get_layer_prefix(layer_idx)
# Returns: "model.layers.{idx}." for Llama
# Returns: "transformer.h.{idx}." for GPT
# Returns: "encoder.block.{idx}." for T5 encoder
```

---

## Migration Scenarios

### Scenario 1: Simple Llama Model (Drop-in Replacement)

**Before:**

```python
from src.nexus_final.sli import SequentialLayerIntegrator

integrator = SequentialLayerIntegrator(
    model_id="meta-llama/Llama-2-7b-hf"
)
result = integrator.run_sli(dataset)
```

**After:**

```python
from src.nexus_final.sli import UniversalSLIIntegrator

integrator = UniversalSLIIntegrator(
    model_id="meta-llama/Llama-2-7b-hf"
)
result = integrator.run_sli(dataset)
```

**Changes:** Just the import and class name.

---

### Scenario 2: Using Non-Llama Architectures (New Capability)

**Before:**

```python
# Not possible with legacy integrator
# Would fail with architecture mismatch errors
```

**After:**

```python
from src.nexus_final.sli import UniversalSLIIntegrator

# GPT-2 model
integrator = UniversalSLIIntegrator("gpt2")
result = integrator.run_sli(dataset)

# T5 model
integrator = UniversalSLIIntegrator("google/flan-t5-base")
result = integrator.run_sli(dataset)

# Mamba model
integrator = UniversalSLIIntegrator("state-spaces/mamba-370m")
result = integrator.run_sli(dataset)
```

**Benefit:** Now supports 135+ architectures automatically.

---

### Scenario 3: MoE Model Processing (New Capability)

**Before:**

```python
# Not supported - would fail
```

**After:**

```python
from src.nexus_final.sli import UniversalSLIIntegrator

# Mixtral MoE model
integrator = UniversalSLIIntegrator("mistralai/Mixtral-8x7B-v0.1")
result = integrator.run_sli(dataset)

# Check MoE info
summary = integrator.get_model_summary()
print(f"Is MoE: {summary['is_moe']}")
print(f"Experts: {summary['moe_info']['num_experts']}")
```

**Benefit:** Automatic MoE detection and handling.

---

### Scenario 4: Custom Architecture with Trust Remote Code

**Before:**

```python
# Not supported - custom models would fail
```

**After:**

```python
from src.nexus_final.sli import UniversalSLIIntegrator

# ChatGLM requires trust_remote_code
integrator = UniversalSLIIntegrator(
    "THUDM/chatglm3-6b",
    trust_remote_code=True  # Required for custom modeling files
)
result = integrator.run_sli(dataset)
```

**Note:** `trust_remote_code=True` is the default in `UniversalSLIIntegrator`.

---

### Scenario 5: Accessing Model Information

**Before:**

```python
# Limited information available
# Had to manually inspect config
```

**After:**

```python
from src.nexus_final.sli import UniversalSLIIntegrator

integrator = UniversalSLIIntegrator("mistralai/Mistral-7B-v0.1")

# Rich model summary
summary = integrator.get_model_summary()
print(f"""
Model: {summary['model_id']}
Family: {summary['family_name']} ({summary['family']})
Layers: {summary['num_layers']}
Hidden Size: {summary['hidden_size']}
Vocab Size: {summary['vocab_size']}
Is MoE: {summary['is_moe']}
""")

if summary['is_moe']:
    print(f"MoE Type: {summary['moe_info']['moe_type']}")
    print(f"Experts: {summary['moe_info']['num_experts']}")
    print(f"Top-k: {summary['moe_info']['top_k']}")
```

---

## Advanced Migration: Using Component Classes

For advanced use cases, you can use the component classes directly:

### Architecture Registry

```python
from src.nexus_final.sli import get_registry

# Get global registry
registry = get_registry()

# List all supported families
families = registry.list_families()
print(f"Supported families: {families}")
# ['llama', 'gpt', 'qwen', 't5', 'bloom', 'opt', 'mamba', 'moe', 'phi', 'gemma', 'encoder_only']

# Get family information
info = registry.get_family_info()
for family_id, details in info.items():
    print(f"{family_id}: {details['name']}")
    print(f"  Model types: {details['model_types'][:3]}...")
```

### Layer Factory

```python
from src.nexus_final.sli import UniversalLayerFactory, get_registry
from transformers import AutoConfig

config = AutoConfig.from_pretrained("gpt2")
factory = UniversalLayerFactory()

# Create layer for any architecture
layer = factory.create_layer(config, layer_idx=0)
print(f"Created layer: {type(layer).__name__}")
# GPT2Block for GPT-2
# LlamaDecoderLayer for Llama
# T5Block for T5

# Get weight prefix
prefix = factory.get_weight_prefix(config, layer_idx=0)
print(f"Weight prefix: {prefix}")
# "transformer.h.0." for GPT-2
# "model.layers.0." for Llama
```

### Weight Loader

```python
from src.nexus_final.sli import UniversalWeightLoader, get_registry

loader = UniversalWeightLoader(
    cache_dir="temp_shards",
    model_id="gpt2"
)

# Auto-detect format
print(f"Format: {loader.format}")

# Get weight info
info = loader.get_weight_info()
print(f"Shards: {info['num_shards']}")
print(f"Weights: {info['num_weights']}")

# Load layer weights
registry = get_registry()
config = AutoConfig.from_pretrained("gpt2")
family = registry.detect_family(config)

weights = loader.load_layer_weights(layer_idx=0, family=family)
print(f"Loaded {len(weights)} weight tensors")
```

### MoE Handler

```python
from src.nexus_final.sli import MoEHandler, MoEConfig
from transformers import AutoConfig

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Automatic MoE detection
handler = MoEHandler(config)
info = handler.get_model_info()
print(f"MoE Type: {info['moe_type']}")
print(f"Experts: {info['num_experts']}")
print(f"Top-k: {info['top_k']}")

# Check if layer is MoE
for i in range(5):
    is_moe = handler.is_moe_layer(i)
    print(f"Layer {i}: {'MoE' if is_moe else 'Dense'}")

# Get expert weight patterns
pattern = handler.get_expert_weight_pattern(layer_idx=0, expert_idx=3)
print(f"Expert 3 pattern: {pattern}")

# Get router pattern
router_pattern = handler.get_router_weight_pattern(layer_idx=0)
print(f"Router pattern: {router_pattern}")
```

---

## Deprecation Timeline

| Phase | Date | Status |
|-------|------|--------|
| **Phase 1** | 2026-01-31 | `SequentialLayerIntegrator` deprecated but functional |
| **Phase 2** | 2026-03-31 | Deprecation warnings added |
| **Phase 3** | 2026-06-30 | `SequentialLayerIntegrator` becomes alias only |

**Recommendation:** Migrate all new code to `UniversalSLIIntegrator` immediately.

---

## Troubleshooting Migration Issues

### Issue 1: "Unsupported Architecture" Error

**Cause:** Model type not detected correctly

**Solution:**

```python
from src.nexus_final.sli import get_registry

# Check detection
config = AutoConfig.from_pretrained("your-model")
registry = get_registry()

try:
    family = registry.detect_family(config)
except UnsupportedArchitectureError:
    # Force specific family
    family = registry.get_family("llama")  # or appropriate family
    print(f"Forced family: {family.family_name}")
```

### Issue 2: Weight Loading Failures

**Cause:** Weight format not detected or corrupted cache

**Solution:**

```python
# Clear cache
integrator.clear_cache()

# Or manually
integrator.weight_loader.clear_shards()
```

### Issue 3: Custom Model with Remote Code

**Cause:** `trust_remote_code=False`

**Solution:**

```python
integrator = UniversalSLIIntegrator(
    "custom/model",
    trust_remote_code=True  # Required
)
```

### Issue 4: Memory Issues

**Cause:** Batch size too large

**Solution:**

```python
# Reduce batch size
result = integrator.run_sli(dataset, batch_size=1)
```

---

## Migration Checklist

- [ ] Update imports from `SequentialLayerIntegrator` to `UniversalSLIIntegrator`
- [ ] Test with existing Llama models (should work without changes)
- [ ] Consider supporting non-Llama architectures (new capability)
- [ ] Review `trust_remote_code` settings for custom models
- [ ] Test MoE models if applicable
- [ ] Update documentation and examples
- [ ] Remove legacy workarounds for architecture detection

---

## Code Examples

### Complete Example: Before and After

#### Complete Legacy Example

```python
"""
Legacy SequentialLayerIntegrator usage
Limited to Llama architectures only
"""
from src.nexus_final.sli import SequentialLayerIntegrator

# Only works with Llama models
integrator = SequentialLayerIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards",
    device="cuda"
)

dataset = ["Sample text for processing"]
result = integrator.run_sli(dataset)

print(f"Processed {result['num_layers']} layers")
```

#### Complete Universal Example

```python
"""
New UniversalSLIIntegrator usage
Supports 135+ architectures
"""
from src.nexus_final.sli import UniversalSLIIntegrator

# Works with ANY supported architecture
integrator = UniversalSLIIntegrator(
    model_id="meta-llama/Llama-2-7b-hf",  # Llama
    # model_id="gpt2",  # GPT
    # model_id="google/flan-t5-base",  # T5
    # model_id="mistralai/Mixtral-8x7B-v0.1",  # MoE
    output_dir="profiles/sli_profile",
    cache_dir="temp_sli_shards",
    device="cuda",
    trust_remote_code=True  # For custom architectures
)

# Get model information
summary = integrator.get_model_summary()
print(f"Model family: {summary['family_name']}")
print(f"Architecture: {summary['family']}")

dataset = ["Sample text for processing"]
result = integrator.run_sli(dataset)

print(f"Processed {result['num_layers']} layers")
print(f"Model info: {result['model_info']}")
```

---

## Summary

Migrating from `SequentialLayerIntegrator` to `UniversalSLIIntegrator` is straightforward:

1. **For basic usage:** Simply change the class name - full backward compatibility
2. **For new capabilities:** Leverage 135+ architecture support and MoE handling
3. **For advanced usage:** Access component classes (Registry, Factory, Loader)

The new Universal SLI provides:

- ✅ Full backward compatibility
- ✅ 135+ architecture support
- ✅ Automatic architecture detection
- ✅ MoE model support
- ✅ Multiple weight format support
- ✅ Extensible registry system

---

*For complete architecture documentation, see [`SLI_UNIVERSAL_GUIDE.md`](SLI_UNIVERSAL_GUIDE.md)*  
*For technical implementation details, see [`NEXUS_V6_TECHNICAL_MANUAL.md`](NEXUS_V6_TECHNICAL_MANUAL.md)*
