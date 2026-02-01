# Custom Layer Registration Documentation

## Table of Contents

- [Overview](#overview)
- [Why Custom Layers](#why-custom-layers)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Integration with ArchitectureRegistry](#integration-with-architectureregistry)

---

## Overview

The Architecture Registry's custom layer system allows you to extend Nexus SLI with your own layer types. This enables support for:

- Novel attention mechanisms
- Custom normalization layers
- Specialized expert layers (MoE)
- Architecture-specific optimizations
- Research experiments

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               Custom Layer Registry System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ArchitectureRegistry (Singleton)            │   │
│  │  ┌─────────────────┐    ┌─────────────────────────┐    │   │
│  │  │ Built-in        │    │ Custom Layers           │    │   │
│  │  │ Families        │    │ (_custom_layers)        │    │   │
│  │  │                 │    │                         │    │   │
│  │  │ • Llama         │    │ • llama_rotary_scaling  │    │   │
│  │  │ • GPT           │    │ • moe_custom_expert     │    │   │
│  │  │ • BERT          │    │ • swin_shifted_window   │    │   │
│  │  │ • ...           │    │ • your_custom_layer     │    │   │
│  │  └─────────────────┘    └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Layer Factory                         │   │
│  │                                                          │   │
│  │  def create_layer(config, layer_idx, layer_type):        │   │
│  │      factory = registry.get_layer_factory(name)          │   │
│  │      return factory(config, layer_idx)                   │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Custom Layers

### Use Cases

| Use Case | Description | Example |
|----------|-------------|---------|
| **Novel Architectures** | Add layers not in transformers library | Custom attention variants |
| **Research** | Experiment with new layer types | Modified transformer blocks |
| **Optimization** | Architecture-specific optimizations | Fused attention-FFN layers |
| **MoE Extensions** | Custom expert architectures | Task-specific experts |
| **Domain-Specific** | Layers for specific domains | Protein sequence layers |
| **Backwards Compatibility** | Legacy layer support | Old model formats |

### Benefits

1. **Extensibility**: Add any layer type without modifying core code
2. **Reusability**: Register once, use across models
3. **Versioning**: Update layer implementations independently
4. **Testing**: Isolate and test custom layers separately
5. **Collaboration**: Share layer implementations with team

---

## Quick Start

### Basic Registration

```python
from src.nexus_final.sli.architecture_registry import ArchitectureRegistry
import torch.nn as nn

# 1. Get the registry (singleton)
registry = ArchitectureRegistry()

# 2. Define a custom layer factory
def my_custom_layer_factory(config, layer_idx):
    """Create a custom layer."""
    return nn.Linear(config.hidden_size, config.hidden_size)

# 3. Register the custom layer
registry.register_custom_layer(
    layer_name="my_custom_linear",
    layer_factory=my_custom_layer_factory
)

# 4. Use the custom layer
factory = registry.get_layer_factory("my_custom_linear")
layer = factory(config, layer_idx=0)
```

### Using a Class Factory

```python
# Define a custom layer class
class CustomAttentionLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads
        )
    
    def forward(self, x):
        return self.attention(x, x, x)

# Register the class directly
registry.register_custom_layer("custom_attention", CustomAttentionLayer)

# Create instance
factory = registry.get_layer_factory("custom_attention")
layer = factory(config, layer_idx=0)
```

### Lambda Factory (Quick)

```python
# Quick registration with lambda
registry.register_custom_layer(
    "quick_linear",
    lambda config, idx: nn.Linear(config.hidden_size, config.intermediate_size)
)
```

### Full Example: Custom Rotary Embedding

```python
import torch
import torch.nn as nn
from src.nexus_final.sli.architecture_registry import get_registry

class RotaryEmbeddingWithScaling(nn.Module):
    """RoPE with learned position scaling."""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position = config.max_position_embeddings
        self.scaling = nn.Parameter(torch.ones(1))
        
    def forward(self, x, seq_len):
        # Simplified RoPE implementation
        positions = torch.arange(seq_len, device=x.device)
        freqs = torch.exp(
            -torch.arange(0, self.dim, 2).float() * 
            (torch.log(torch.tensor(10000.0)) / self.dim)
        )
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0) * self.scaling
        return angles

# Register
registry = get_registry()
registry.register_custom_layer(
    "llama_rotary_with_scaling",
    RotaryEmbeddingWithScaling
)

# Use
config = type('Config', (), {
    'hidden_size': 4096,
    'num_attention_heads': 32,
    'max_position_embeddings': 8192
})()

factory = registry.get_layer_factory("llama_rotary_with_scaling")
rope_layer = factory(config, layer_idx=0)
print(f"Created: {type(rope_layer).__name__}")
# Created: RotaryEmbeddingWithScaling
```

---

## API Reference

### ArchitectureRegistry Class

The [`ArchitectureRegistry`](src/nexus_final/sli/architecture_registry.py:709) provides methods for custom layer management.

#### register_custom_layer()

```python
def register_custom_layer(
    self, 
    layer_name: str, 
    layer_factory: Any
) -> None
```

Register a custom layer factory in the registry.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_name` | `str` | Unique identifier for the custom layer type. Should follow naming convention: `"<family>_<layer_type>"` (e.g., `"llama_custom_attn"`, `"moe_expert_layer"`) |
| `layer_factory` | `Callable` | A callable (class or factory function) that creates the layer instance. Should accept arguments `(config, layer_idx)` and return `nn.Module` |

**Raises:**

- `ValueError`: If `layer_name` is already registered or invalid
- `TypeError`: If `layer_factory` is not callable

**Example:**

```python
registry.register_custom_layer(
    "llama_rotary_with_scaling",
    lambda config, idx: RotaryEmbeddingWithScaling(config)
)
```

#### get_layer_factory()

```python
def get_layer_factory(self, layer_name: str) -> Any
```

Retrieve a registered custom layer factory by name.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_name` | `str` | The identifier of the custom layer to retrieve |

**Returns:**

- The registered layer factory (callable)

**Raises:**

- `KeyError`: If the `layer_name` is not registered. Error message includes list of available custom layers.

**Example:**

```python
factory = registry.get_layer_factory("llama_custom_attn")
layer = factory(config, layer_idx=0)
```

#### unregister_custom_layer()

```python
def unregister_custom_layer(self, layer_name: str) -> bool
```

Remove a custom layer from the registry.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_name` | `str` | The identifier of the custom layer to remove |

**Returns:**

- `True` if the layer was removed
- `False` if it didn't exist

**Example:**

```python
removed = registry.unregister_custom_layer("old_custom_layer")
print(f"Removed: {removed}")
```

#### list_custom_layers()

```python
def list_custom_layers(self) -> List[str]
```

List all registered custom layer names.

**Returns:**

- List of registered custom layer identifiers

**Example:**

```python
custom_layers = registry.list_custom_layers()
print(f"Custom layers: {custom_layers}")
# Custom layers: ['llama_rotary_scaling', 'moe_expert_layer']
```

#### clear_custom_layers()

```python
def clear_custom_layers(self) -> None
```

Clear all custom layers from the registry.

**Warning:** Use with caution. This removes all custom registrations.

**Example:**

```python
registry.clear_custom_layers()
assert len(registry.list_custom_layers()) == 0
```

### get_registry() Function

```python
def get_registry() -> ArchitectureRegistry
```

Get the global architecture registry instance (singleton).

**Returns:**

- The global [`ArchitectureRegistry`](src/nexus_final/sli/architecture_registry.py:709) instance

**Example:**

```python
from src.nexus_final.sli.architecture_registry import get_registry

registry = get_registry()
# Same instance across all imports
```

---

## Advanced Usage

### Complex Layer Factories

Create sophisticated layer factories with dependencies:

```python
class ComplexTransformerBlock(nn.Module):
    """Custom transformer block with configurable components."""
    
    def __init__(
        self, 
        config, 
        layer_idx,
        attention_impl="flash",
        norm_type="rms",
        activation="swiglu"
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Configurable attention
        if attention_impl == "flash":
            self.attn = FlashAttention(config)
        elif attention_impl == "memory_efficient":
            self.attn = MemoryEfficientAttention(config)
        else:
            self.attn = StandardAttention(config)
        
        # Configurable normalization
        if norm_type == "rms":
            self.norm1 = RMSNorm(config.hidden_size)
            self.norm2 = RMSNorm(config.hidden_size)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size)
            self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Configurable FFN
        self.ffn = self._create_ffn(config, activation)
    
    def _create_ffn(self, config, activation):
        if activation == "swiglu":
            return SwiGLUFFN(config)
        elif activation == "gelu":
            return GELUFFN(config)
        else:
            return ReLUUFFN(config)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Factory function with configuration
def create_complex_block_factory(attention_impl="flash", norm_type="rms"):
    """Create a factory for complex transformer blocks."""
    
    def factory(config, layer_idx):
        return ComplexTransformerBlock(
            config, 
            layer_idx,
            attention_impl=attention_impl,
            norm_type=norm_type
        )
    
    return factory

# Register different variants
registry.register_custom_layer(
    "transformer_flash_rms",
    create_complex_block_factory("flash", "rms")
)

registry.register_custom_layer(
    "transformer_memory_layernorm",
    create_complex_block_factory("memory_efficient", "layernorm")
)
```

### Factory with State

Create factories that maintain state:

```python
class StatefulLayerFactory:
    """Factory that tracks created layers."""
    
    def __init__(self, layer_type="standard"):
        self.layer_type = layer_type
        self.created_count = 0
        self.layer_indices = []
    
    def __call__(self, config, layer_idx):
        self.created_count += 1
        self.layer_indices.append(layer_idx)
        
        if self.layer_type == "standard":
            return StandardLayer(config, layer_idx)
        elif self.layer_type == "modified":
            return ModifiedLayer(config, layer_idx)
    
    def get_stats(self):
        return {
            "total_created": self.created_count,
            "indices": self.layer_indices
        }

# Create and register stateful factory
factory = StatefulLayerFactory("modified")
registry.register_custom_layer("stateful_modified", factory)

# Use multiple times
for i in range(5):
    layer = registry.get_layer_factory("stateful_modified")(config, i)

# Check stats
print(factory.get_stats())
# {'total_created': 5, 'indices': [0, 1, 2, 3, 4]}
```

### Callable Objects as Factories

```python
class ConfigurableLayerFactory:
    """Callable object that creates layers."""
    
    def __init__(self, use_bias=True, dropout=0.1):
        self.use_bias = use_bias
        self.dropout = dropout
    
    def __call__(self, config, layer_idx):
        layers = []
        
        # Linear layer
        layers.append(nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=self.use_bias
        ))
        
        # Activation
        layers.append(nn.GELU())
        
        # Dropout
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # Output projection
        layers.append(nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=self.use_bias
        ))
        
        return nn.Sequential(*layers)

# Register with specific configuration
no_bias_factory = ConfigurableLayerFactory(use_bias=False, dropout=0.0)
registry.register_custom_layer("ffn_no_bias", no_bias_factory)

regular_factory = ConfigurableLayerFactory(use_bias=True, dropout=0.1)
registry.register_custom_layer("ffn_regular", regular_factory)
```

### Bound Methods as Factories

```python
class LayerFactoryManager:
    """Manager class that creates different layer types."""
    
    def __init__(self, default_activation="gelu"):
        self.default_activation = default_activation
        self.created_layers = []
    
    def create_attention(self, config, layer_idx):
        """Create attention layer."""
        layer = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        self.created_layers.append(("attention", layer_idx))
        return layer
    
    def create_ffn(self, config, layer_idx):
        """Create FFN layer."""
        layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU() if self.default_activation == "gelu" else nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.created_layers.append(("ffn", layer_idx))
        return layer

# Create manager
manager = LayerFactoryManager()

# Register bound methods
registry.register_custom_layer("managed_attention", manager.create_attention)
registry.register_custom_layer("managed_ffn", manager.create_ffn)

# Use
attn = registry.get_layer_factory("managed_attention")(config, 0)
ffn = registry.get_layer_factory("managed_ffn")(config, 0)

print(manager.created_layers)
# [('attention', 0), ('ffn', 0)]
```

---

## Best Practices

### Naming Conventions

Follow the `<family>_<layer_type>` pattern:

| Pattern | Example | Description |
|---------|---------|-------------|
| `<family>_<type>` | `llama_custom_attn` | Family-specific layer |
| `moe_<type>` | `moe_expert_layer` | MoE-specific layer |
| `generic_<type>` | `generic_rms_norm` | Cross-family layer |
| `experimental_<type>` | `experimental_new_attn` | Research/experimental |

**Good Names:**

```python
"llama_rotary_with_scaling"
"moe_sparse_expert"
"gpt_neox_parallel_attn"
"bert_alibi_position"
```

**Avoid:**

```python
"layer1"           # Too generic
"custom"           # Not descriptive
"MyLayer"          # Wrong format (use snake_case)
"transformer-block" # Use underscores, not hyphens
```

### Error Handling

```python
from src.nexus_final.sli.architecture_registry import ArchitectureRegistry

registry = ArchitectureRegistry()

def safe_register(layer_name, factory):
    """Safely register with error handling."""
    try:
        registry.register_custom_layer(layer_name, factory)
        print(f"✓ Registered: {layer_name}")
        return True
    except ValueError as e:
        if "already registered" in str(e):
            print(f"⚠ {layer_name} already exists, skipping")
            return False
        raise
    except TypeError as e:
        print(f"✗ Invalid factory for {layer_name}: {e}")
        return False

# Usage
safe_register("my_layer", lambda c, i: nn.Linear(c.hidden_size, c.hidden_size))
```

### Organization

Group related layers:

```python
# layers/attention.py
class FlashAttention(nn.Module): ...
class SparseAttention(nn.Module): ...

# layers/normalization.py
class RMSNorm(nn.Module): ...
class LayerNormNoBias(nn.Module): ...

# layers/moe.py
class ExpertLayer(nn.Module): ...
class SharedExpertLayer(nn.Module): ...

# registration.py
from src.nexus_final.sli.architecture_registry import get_registry
from .layers.attention import FlashAttention, SparseAttention
from .layers.normalization import RMSNorm

def register_all_layers():
    registry = get_registry()
    
    # Attention layers
    registry.register_custom_layer("flash_attn", FlashAttention)
    registry.register_custom_layer("sparse_attn", SparseAttention)
    
    # Normalization layers
    registry.register_custom_layer("rms_norm", RMSNorm)
```

### Testing Custom Layers

```python
import pytest
import torch

class TestCustomLayer:
    """Test suite for custom layers."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        return type('Config', (), {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072
        })()
    
    def test_custom_layer_creation(self, mock_config):
        """Test layer can be created."""
        factory = registry.get_layer_factory("my_custom_layer")
        layer = factory(mock_config, 0)
        assert layer is not None
    
    def test_custom_layer_forward(self, mock_config):
        """Test layer forward pass."""
        factory = registry.get_layer_factory("my_custom_layer")
        layer = factory(mock_config, 0)
        
        x = torch.randn(2, 10, 768)  # [batch, seq, hidden]
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_custom_layer_device(self, mock_config):
        """Test layer works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        factory = registry.get_layer_factory("my_custom_layer")
        layer = factory(mock_config, 0).cuda()
        
        x = torch.randn(2, 10, 768).cuda()
        output = layer(x)
        
        assert output.is_cuda
```

---

## Examples

### Example 1: Custom MoE Expert

```python
import torch
import torch.nn as nn
from src.nexus_final.sli.architecture_registry import get_registry

class TaskSpecificExpert(nn.Module):
    """Expert layer specialized for different tasks."""
    
    def __init__(self, config, layer_idx, task_type="general"):
        super().__init__()
        self.task_type = task_type
        self.layer_idx = layer_idx
        
        # Task-specific architecture
        if task_type == "reasoning":
            # Deeper FFN for reasoning
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size * 2),
                nn.GELU(),
                nn.Linear(config.intermediate_size * 2, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
        elif task_type == "factual":
            # Wide FFN for factual recall
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size * 4),
                nn.GELU(),
                nn.Linear(config.intermediate_size * 4, config.hidden_size)
            )
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            )
    
    def forward(self, x):
        return self.ffn(x)

# Create factory for task-specific experts
def create_expert_factory(task_type="general"):
    def factory(config, layer_idx):
        return TaskSpecificExpert(config, layer_idx, task_type)
    return factory

# Register
registry = get_registry()
registry.register_custom_layer("moe_expert_reasoning", create_expert_factory("reasoning"))
registry.register_custom_layer("moe_expert_factual", create_expert_factory("factual"))
registry.register_custom_layer("moe_expert_general", create_expert_factory("general"))

# Use in MoE model
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        
        # Create experts using registry
        self.experts = nn.ModuleList([
            registry.get_layer_factory("moe_expert_general")(config, i)
            for i in range(self.num_experts)
        ])
        self.router = nn.Linear(config.hidden_size, self.num_experts)
    
    def forward(self, x):
        # Route to experts
        router_logits = self.router(x)
        weights = torch.softmax(router_logits, dim=-1)
        
        # Combine expert outputs
        output = sum(
            w.unsqueeze(-1) * expert(x)
            for w, expert in zip(weights.unbind(-1), self.experts)
        )
        return output
```

### Example 2: Custom Attention Variants

```python
import torch
import torch.nn as nn
import math

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention for efficient inference."""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # Q projects to all heads, K/V project to single head
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape Q for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand K/V for broadcasting
        k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention - balance between MHA and MQA."""
    
    def __init__(self, config, layer_idx, num_kv_groups=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim * num_kv_groups)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim * num_kv_groups)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # Repeat K/V heads to match Q
        k = k.repeat_interleave(self.num_heads // self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_groups, dim=1)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


# Register attention variants
registry = get_registry()
registry.register_custom_layer("multi_query_attention", MultiQueryAttention)
registry.register_custom_layer("grouped_query_attention", GroupedQueryAttention)

# Usage with different models
config = type('Config', (), {
    'hidden_size': 4096,
    'num_attention_heads': 32
})()

# Standard multi-head attention equivalent
mqa_layer = registry.get_layer_factory("multi_query_attention")(config, 0)
print(f"MQA parameters: {sum(p.numel() for p in mqa_layer.parameters()):,}")

# Grouped query attention (balanced)
gqa_layer = registry.get_layer_factory("grouped_query_attention")(config, 0)
print(f"GQA parameters: {sum(p.numel() for p in gqa_layer.parameters()):,}")
```

### Example 3: Complete Custom Architecture

```python
import torch
import torch.nn as nn
from src.nexus_final.sli.architecture_registry import (
    ArchitectureRegistry, 
    ArchitectureFamily
)
from transformers import PretrainedConfig

class CustomArchitectureHandler(ArchitectureFamily):
    """Handler for a custom transformer architecture."""
    
    family_id = "custom_transformer"
    family_name = "Custom Transformer Architecture"
    model_types = ["custom_transformer"]
    architectures = ["CustomTransformerForCausalLM"]
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        # Use custom layers from registry
        registry = ArchitectureRegistry()
        
        attention = registry.get_layer_factory("grouped_query_attention")(
            config, layer_idx
        )
        
        # Create custom block
        return CustomTransformerBlock(config, layer_idx, attention)


class CustomTransformerBlock(nn.Module):
    """Custom transformer block with registry layers."""
    
    def __init__(self, config, layer_idx, attention):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = attention
        
        # Get custom normalization
        registry = ArchitectureRegistry()
        norm_factory = registry.get_layer_factory("rms_norm")
        self.norm1 = norm_factory(config, layer_idx)
        self.norm2 = norm_factory(config, layer_idx)
        
        # Get custom FFN
        ffn_factory = registry.get_layer_factory("swiglu_ffn")
        self.ffn = ffn_factory(config, layer_idx)
    
    def forward(self, x):
        # Pre-norm with custom components
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Setup
registry = ArchitectureRegistry()

# Register all custom components
registry.register_custom_layer("grouped_query_attention", GroupedQueryAttention)
registry.register_custom_layer("rms_norm", lambda c, i: RMSNorm(c.hidden_size))
registry.register_custom_layer("swiglu_ffn", lambda c, i: SwiGLUFFN(c))

# Register the custom family
registry.register("custom_transformer", CustomArchitectureHandler())

# Now you can use it like any built-in family
config = PretrainedConfig(
    model_type="custom_transformer",
    architectures=["CustomTransformerForCausalLM"],
    hidden_size=2048,
    num_attention_heads=16,
    intermediate_size=8192
)

family = registry.detect_family(config)
print(f"Detected family: {family.family_id}")
layer = family.create_layer(config, 0)
print(f"Created layer: {type(layer).__name__}")
```

### Example 4: Lifecycle Management

```python
from src.nexus_final.sli.architecture_registry import ArchitectureRegistry
import atexit

class LayerRegistryManager:
    """Manage custom layer lifecycle."""
    
    def __init__(self):
        self.registry = ArchitectureRegistry()
        self.registered_layers = []
        atexit.register(self.cleanup)
    
    def register(self, name, factory, temporary=False):
        """Register a layer, optionally temporary."""
        self.registry.register_custom_layer(name, factory)
        
        if not temporary:
            self.registered_layers.append(name)
        
        print(f"Registered {'temporary ' if temporary else ''}layer: {name}")
        return name
    
    def get(self, name):
        """Get layer factory."""
        return self.registry.get_layer_factory(name)
    
    def unregister(self, name):
        """Unregister a specific layer."""
        success = self.registry.unregister_custom_layer(name)
        if success and name in self.registered_layers:
            self.registered_layers.remove(name)
        return success
    
    def cleanup(self):
        """Cleanup all registered layers on exit."""
        print(f"Cleaning up {len(self.registered_layers)} layers...")
        for name in list(self.registered_layers):
            self.unregister(name)
        print("Cleanup complete")

# Usage
manager = LayerRegistryManager()

# Register permanent layers
manager.register("permanent_attention", CustomAttention)
manager.register("permanent_ffn", CustomFFN)

# Register temporary layer (not tracked for cleanup)
manager.register("temp_experimental", ExperimentalLayer, temporary=True)

# Use layers
attn_factory = manager.get("permanent_attention")
layer = attn_factory(config, 0)

# When program exits, permanent layers are cleaned up automatically
```

---

## Integration with ArchitectureRegistry

### Custom Family with Custom Layers

```python
from src.nexus_final.sli.architecture_registry import (
    ArchitectureFamily,
    ArchitectureRegistry
)
from transformers import PretrainedConfig

class CustomFamilyWithCustomLayers(ArchitectureFamily):
    """Family that uses custom registered layers."""
    
    family_id = "hybrid"
    family_name = "Hybrid Architecture"
    model_types = ["hybrid"]
    architectures = ["HybridForCausalLM"]
    
    def __init__(self):
        super().__init__()
        self.registry = ArchitectureRegistry()
    
    def get_layer_prefix(self, layer_idx: int, layer_type: str = "decoder") -> str:
        return f"model.layers.{layer_idx}."
    
    def create_layer(self, config: PretrainedConfig, layer_idx: int,
                     layer_type: str = "decoder") -> nn.Module:
        """Create layer using custom factory if available."""
        
        # Check for custom layer configuration
        custom_layer_type = getattr(config, "custom_layer_type", "default")
        
        if custom_layer_type == "optimized":
            # Use registered custom layer
            factory = self.registry.get_layer_factory("optimized_transformer_block")
            return factory(config, layer_idx)
        else:
            # Use default
            return StandardTransformerBlock(config, layer_idx)

# Register custom optimized block
def create_optimized_block(config, layer_idx):
    """Create an optimized transformer block."""
    return OptimizedTransformerBlock(config, layer_idx)

registry = ArchitectureRegistry()
registry.register_custom_layer("optimized_transformer_block", create_optimized_block)
registry.register("hybrid", CustomFamilyWithCustomLayers())
```

### Integration with SLI Processor

```python
from src.nexus_final.sli.universal_sli import UniversalSLIProcessor

# Register custom layers before creating processor
registry = ArchitectureRegistry()
registry.register_custom_layer("custom_attn", CustomAttention)

# Create processor - it will use the registry
processor = UniversalSLIProcessor(
    model_name="custom-model",
    custom_layer_overrides={
        "attention": "custom_attn"
    }
)
```

---

## See Also

- [Architecture Registry](architecture_registry.py) - Full architecture registry documentation
- [SLI Universal Guide](SLI_UNIVERSAL_GUIDE.md) - Main SLI documentation
- [Quantization Documentation](QUANTIZATION.md) - Quantize custom layers
- [Encoder Support](ENCODER_SUPPORT.md) - Custom encoder layers

## Testing

For comprehensive test examples, see:

- [`tests/unit/test_custom_layer_registry.py`](../tests/unit/test_custom_layer_registry.py)
