# Quantization Module Documentation

## Table of Contents

- [Overview](#overview)
- [Supported Modes](#supported-modes)
- [Quick Start](#quick-start)
- [QuantizationConfig](#quantizationconfig)
- [LayerQuantizer](#layerquantizer)
- [Adaptive Quantization](#adaptive-quantization)
- [Integration with SLI](#integration-with-sli)
- [Performance Guide](#performance-guide)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Quantization is the process of reducing the precision of model weights from higher-precision formats (typically FP16 or FP32) to lower-precision formats (INT8, NF4, FP4). This technique provides significant benefits:

### Why Use Quantization?

| Benefit | Description | Typical Impact |
|---------|-------------|----------------|
| **Memory Reduction** | Store weights in fewer bits | 50-75% less memory |
| **Faster Loading** | Less data to read from disk | 2-4x faster loading |
| **Reduced VRAM** | Fit larger models on limited GPUs | Run 2-4x larger models |
| **Bandwidth Savings** | Less data transfer between CPU/GPU | 50% less PCIe traffic |

### How It Works in SLI

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLI Quantization Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Full Model  │────▶│   Quantize   │────▶│  Store to    │    │
│  │   (FP16)     │     │   Layer by   │     │  Disk Cache  │    │
│  └──────────────┘     │   Layer      │     └──────────────┘    │
│                       └──────────────┘                          │
│                               │                                 │
│                               ▼                                 │
│                       ┌──────────────┐     ┌──────────────┐    │
│                       │   On-Demand  │────▶│  Dequantize  │    │
│                       │    Load      │     │   for Use    │    │
│                       └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

```bash
# Basic quantization (PyTorch native)
pip install torch

# Advanced quantization (recommended)
pip install bitsandbytes>=0.41.0
```

---

## Supported Modes

### Mode Comparison Table

| Mode | Bits | Compression | Precision | Speed Impact | VRAM Savings | Use Case |
|------|------|-------------|-----------|--------------|--------------|----------|
| [`NONE`](src/nexus_final/sli/quantization.py:38) | 16/32 | 1x | Full | None | None | Baseline, debugging |
| [`INT8_DYNAMIC`](src/nexus_final/sli/quantization.py:40) | 8 | 2x | Good | +10-20% CPU | 50% | Compatibility, CPU inference |
| [`INT8`](src/nexus_final/sli/quantization.py:39) | 8 | 2x | Good | +5-10% GPU | 50% | Balanced performance |
| [`NF4`](src/nexus_final/sli/quantization.py:41) | 4 | 4x | Moderate | +15-25% | 75% | Maximum memory savings |
| [`FP4`](src/nexus_final/sli/quantization.py:42) | 4 | 4x | Moderate | +15-25% | 75% | Alternative 4-bit format |
| [`INT4`](src/nexus_final/sli/quantization.py:43) | 4 | 4x | Lower | +20-30% | 75% | Experimental |

### Detailed Mode Descriptions

#### NONE (FP16/FP32)

- **Description**: No quantization, full precision
- **When to use**: Development, debugging, maximum accuracy required
- **Compatibility**: Universal
- **Example**:

```python
from src.nexus_final.sli.quantization import QuantizationConfig, QuantizationMode

config = QuantizationConfig(mode=QuantizationMode.NONE)
```

#### INT8 (8-bit via bitsandbytes)

- **Description**: 8-bit integer quantization using bitsandbytes
- **When to use**: Production inference, 50% memory reduction acceptable
- **Outlier handling**: Configurable threshold for outlier values
- **Example**:

```python
config = QuantizationConfig(
    mode=QuantizationMode.INT8,
    llm_int8_threshold=6.0,  # Outlier threshold
    llm_int8_skip_modules=["lm_head", "embed_tokens"]
)
```

#### INT8_DYNAMIC (PyTorch Native)

- **Description**: Dynamic quantization at runtime using PyTorch
- **When to use**: CPU inference, maximum compatibility
- **Limitation**: Slightly slower than bitsandbytes on GPU
- **Example**:

```python
config = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
```

#### NF4 (4-bit Normal Float)

- **Description**: 4-bit quantization using Normal Float format
- **When to use**: Maximum memory savings, ~60-75% retention acceptable
- **Advantage**: Optimal for normally distributed weights
- **Example**:

```python
config = QuantizationConfig(
    mode=QuantizationMode.NF4,
    compute_dtype=torch.float16,
    double_quant=True,  # Nested quantization for better accuracy
    compress_statistics=True
)
```

#### FP4 (4-bit Float)

- **Description**: 4-bit floating point quantization
- **When to use**: Alternative to NF4 when weight distribution differs
- **Example**:

```python
config = QuantizationConfig(mode=QuantizationMode.FP4, quant_type="fp4")
```

---

## Quick Start

### Basic Usage

```python
from src.nexus_final.sli.quantization import (
    QuantizationConfig, 
    QuantizationMode,
    LayerQuantizer
)
import torch.nn as nn

# 1. Create a sample layer
layer = nn.Linear(1024, 4096)

# 2. Configure quantization (4-bit NF4)
config = QuantizationConfig(mode=QuantizationMode.NF4)

# 3. Create quantizer and quantize
quantizer = LayerQuantizer(config)
quantized_layer = quantizer.quantize_layer(layer)

# 4. Check size reduction
original_size = layer.weight.numel() * layer.weight.element_size()
ratio = quantizer.get_quantized_size_ratio(quantized_layer)
print(f"Size reduced to {ratio*100:.0f}% of original")
# Output: Size reduced to 25% of original

# 5. Dequantize when needed
full_precision_layer = quantizer.dequantize_layer(quantized_layer)
```

### Predefined Configurations

```python
from src.nexus_final.sli.quantization import (
    get_int8_config,
    get_nf4_config, 
    get_fp4_config,
    get_mixed_precision_config
)

# Standard INT8 config (recommended starting point)
int8_config = get_int8_config()

# NF4 for maximum compression
nf4_config = get_nf4_config()

# FP4 alternative
fp4_config = get_fp4_config()

# Mixed precision (attention: INT8, FFN: NF4)
mixed_quantizer = get_mixed_precision_config()
```

### Convenience Functions

```python
from src.nexus_final.sli.quantization import quantize_layer, dequantize_layer

# One-line quantization
quantized = quantize_layer(layer, mode="nf4")

# One-line dequantization
restored = dequantize_layer(quantized, compute_dtype=torch.float16)
```

---

## QuantizationConfig

### Configuration Parameters

```python
@dataclass
class QuantizationConfig:
    mode: QuantizationMode = QuantizationMode.NONE          # Quantization mode
    compute_dtype: torch.dtype = torch.float16              # Computation dtype
    compress_statistics: bool = True                        # Compress 4-bit stats
    quant_storage_dtype: torch.dtype = torch.uint8          # Storage dtype
    double_quant: bool = True                               # Nested quantization
    quant_type: str = "nf4"                                 # "nf4" or "fp4"
    llm_int8_threshold: float = 6.0                         # Outlier threshold
    llm_int8_skip_modules: Optional[List[str]] = None       # Modules to skip
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | [`QuantizationMode`](src/nexus_final/sli/quantization.py:36) | `NONE` | Quantization mode to apply |
| `compute_dtype` | `torch.dtype` | `float16` | Dtype for computation (not storage) |
| `compress_statistics` | `bool` | `True` | Compress 4-bit quantization statistics |
| `quant_storage_dtype` | `torch.dtype` | `uint8` | Underlying storage dtype |
| `double_quant` | `bool` | `True` | Use nested quantization for 4-bit |
| `quant_type` | `str` | `"nf4"` | 4-bit quantization type |
| `llm_int8_threshold` | `float` | `6.0` | Outlier threshold for 8-bit |
| `llm_int8_skip_modules` | `List[str]` | `None` | Module names to skip |

### Serialization

```python
# Convert to dictionary for saving
config = QuantizationConfig(mode=QuantizationMode.INT8)
data = config.to_dict()
# {'mode': 'int8', 'compute_dtype': 'torch.float16', ...}

# Restore from dictionary
restored = QuantizationConfig.from_dict(data)
```

---

## LayerQuantizer

### Class Overview

The [`LayerQuantizer`](src/nexus_final/sli/quantization.py:85) class handles quantization and de-quantization of model layers.

### Constructor

```python
quantizer = LayerQuantizer(config: Optional[QuantizationConfig] = None)
```

### Methods

#### quantize_layer()

```python
def quantize_layer(
    self, 
    layer: nn.Module, 
    layer_name: str = ""
) -> nn.Module
```

Quantize a layer according to the configured mode.

**Parameters:**

- `layer`: The layer to quantize
- `layer_name`: Name of the layer for skip configuration

**Returns:** Quantized layer

**Example:**

```python
quantizer = LayerQuantizer(config)
quantized_attn = quantizer.quantize_layer(attention_layer, "model.layers.0.self_attn")
```

#### dequantize_layer()

```python
def dequantize_layer(self, layer: nn.Module) -> nn.Module
```

De-quantize a layer back to full precision.

**Example:**

```python
full_precision = quantizer.dequantize_layer(quantized_layer)
```

#### is_quantized()

```python
def is_quantized(self, layer: nn.Module) -> bool
```

Check if a layer is quantized.

**Example:**

```python
if quantizer.is_quantized(layer):
    print("Layer is quantized")
```

#### get_quantized_size_ratio()

```python
def get_quantized_size_ratio(self, layer: nn.Module) -> float
```

Get the compression ratio of a quantized layer.

**Returns:** Ratio of quantized size to original size (e.g., 0.25 for 4x compression)

---

## Adaptive Quantization

### Overview

The [`AdaptiveQuantizer`](src/nexus_final/sli/quantization.py:368) applies different quantization strategies to different layer types. More important layers (attention) can use higher precision, while less important layers (FFN) can use lower precision.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveQuantizer                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Attention Layers          FFN Layers         Other Layers  │
│  ┌─────────────────┐      ┌─────────────────┐  ┌──────────┐│
│  │   Higher Prec   │      │   Lower Prec    │  │   Base   ││
│  │   (INT8/FP16)   │      │   (NF4/FP4)     │  │   Config ││
│  └─────────────────┘      └─────────────────┘  └──────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Usage

```python
from src.nexus_final.sli.quantization import AdaptiveQuantizer, QuantizationConfig

# Base config for most layers
base_config = QuantizationConfig(mode=QuantizationMode.NF4)

# Higher precision for attention
attention_config = QuantizationConfig(mode=QuantizationMode.INT8)

# Create adaptive quantizer
adaptive = AdaptiveQuantizer(
    base_config=base_config,
    attention_config=attention_config,
    ffn_config=base_config  # FFN uses base (NF4)
)

# Quantize entire model
quantized_model = adaptive.quantize_model(model)
```

### Precision Hierarchy

```python
# AdaptiveQuantizer automatically upgrades precision:
NF4 → INT8 → INT8_DYNAMIC → NONE (full precision)

# Example upgrade path
if base.mode == NF4:
    attention_config.mode = INT8
elif base.mode == INT8:
    attention_config.mode = INT8_DYNAMIC
elif base.mode == INT8_DYNAMIC:
    attention_config.mode = NONE
```

### Layer Type Detection

The quantizer automatically detects layer types by name:

```python
# Attention layers (higher precision)
if any(x in layer_name_lower for x in ['attn', 'attention', 'query', 'key', 'value', 'qkv']):
    return self.attention_config

# FFN layers (lower precision)
if any(x in layer_name_lower for x in ['ffn', 'mlp', 'feedforward', 'fc']):
    return self.ffn_config
```

---

## Integration with SLI

### Quantized Layer Caching

```python
from src.nexus_final.sli.quantization import get_nf4_config
from src.nexus_final.sli.layer_cache import LayerCache

# Create quantized layer cache
cache = LayerCache(
    cache_dir="/path/to/quantized_cache",
    max_cache_size_gb=50.0
)

# Quantize and cache layer
config = get_nf4_config()
quantizer = LayerQuantizer(config)
quantized_layer = quantizer.quantize_layer(layer)

cache.cache_layer(
    model_id="meta-llama/Llama-2-7b",
    layer_index=0,
    layer=quantized_layer,
    quantization_mode="nf4"
)
```

### SLI Processor Integration

```python
from src.nexus_final.sli.universal_sli import UniversalSLIProcessor
from src.nexus_final.sli.quantization import get_int8_config

# Create SLI processor with quantization
processor = UniversalSLIProcessor(
    model_name="meta-llama/Llama-2-7b",
    quantization_config=get_int8_config()
)

# Layers are automatically quantized on caching
# and dequantized on loading (if needed)
```

### Quantization Registry

```python
from src.nexus_final.sli.quantization import QuantizationRegistry

# Register custom configuration
QuantizationRegistry.register("my_custom_int8", custom_config)

# Retrieve configuration
config = QuantizationRegistry.get("my_custom_int8")

# List available configs
available = QuantizationRegistry.list_configs()
# ['int8', 'nf4', 'fp4', 'my_custom_int8']
```

---

## Performance Guide

### Memory Savings by Model Size

| Model Size | FP16 | INT8 | NF4/FP4 |
|------------|------|------|---------|
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |
| 405B | 810 GB | 405 GB | 202 GB |

### Speed Trade-offs

| Operation | FP16 | INT8 | NF4 |
|-----------|------|------|-----|
| Layer Loading | 1x | 2x faster | 4x faster |
| Forward Pass | 1x | 0.95x | 0.80x |
| Memory Copy | 1x | 2x faster | 4x faster |
| PCIe Transfer | 1x | 2x faster | 4x faster |

### Retention Rates

| Mode | Typical Retention | Tasks Affected |
|------|-------------------|----------------|
| INT8 | 90-95% | Minimal impact |
| NF4 | 60-75% | May affect reasoning |
| FP4 | 55-70% | May affect reasoning |
| Mixed (Adaptive) | 75-85% | Balanced approach |

### Benchmarking Example

```python
import time
import torch

# Benchmark loading speed
def benchmark_loading(cache, model_id, num_layers=10):
    times = []
    for i in range(num_layers):
        start = time.time()
        layer = cache.get_layer(model_id, i)
        times.append(time.time() - start)
    return sum(times) / len(times)

# Compare quantized vs unquantized
quantized_cache = LayerCache(cache_dir="/quantized")
regular_cache = LayerCache(cache_dir="/regular")

quantized_time = benchmark_loading(quantized_cache, "model")
regular_time = benchmark_loading(regular_cache, "model")

print(f"Speedup: {regular_time / quantized_time:.2f}x")
```

---

## API Reference

### QuantizationMode Enum

```python
class QuantizationMode(Enum):
    NONE = "none"               # No quantization
    INT8 = "int8"              # 8-bit integer
    INT8_DYNAMIC = "int8_dynamic"  # PyTorch dynamic
    NF4 = "nf4"                # 4-bit Normal Float
    FP4 = "fp4"                # 4-bit Float
    INT4 = "int4"              # 4-bit integer
```

### Convenience Functions

#### quantize_layer()

```python
def quantize_layer(
    layer: nn.Module,
    mode: Union[str, QuantizationMode] = "int8",
    **kwargs
) -> nn.Module
```

One-line layer quantization.

**Example:**

```python
quantized = quantize_layer(layer, mode="nf4", compute_dtype=torch.bfloat16)
```

#### dequantize_layer()

```python
def dequantize_layer(
    layer: nn.Module,
    compute_dtype: torch.dtype = torch.float16
) -> nn.Module
```

One-line layer dequantization.

### Predefined Config Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `get_int8_config()` | [`QuantizationConfig`](src/nexus_final/sli/quantization.py:47) | Standard INT8 with skip modules |
| `get_nf4_config()` | [`QuantizationConfig`](src/nexus_final/sli/quantization.py:47) | Standard NF4 with double quant |
| `get_fp4_config()` | [`QuantizationConfig`](src/nexus_final/sli/quantization.py:47) | Standard FP4 configuration |
| `get_mixed_precision_config()` | [`AdaptiveQuantizer`](src/nexus_final/sli/quantization.py:368) | Attention: INT8, FFN: NF4 |

---

## Best Practices

### When to Use Which Mode

#### Use NONE (FP16/FP32) When

- Debugging model behavior
- Maximum accuracy is critical
- You have sufficient VRAM
- Training or fine-tuning

#### Use INT8 When

- Production inference on modern GPUs
- 50% memory reduction is sufficient
- You need good retention (90-95%)
- Balancing speed and accuracy

#### Use NF4/FP4 When

- Maximum memory savings needed
- Running very large models on limited hardware
- Batch inference where throughput matters
- 60-75% retention is acceptable

#### Use Mixed Precision (Adaptive) When

- You want balanced performance
- Attention layers need higher precision
- FFN layers can tolerate lower precision
- Best overall retention/speed trade-off

### Recommended Configurations

```python
# High-quality inference (90%+ retention)
high_quality = get_int8_config()

# Balanced performance (75-85% retention)
balanced = get_mixed_precision_config()

# Maximum compression (60-75% retention)
maximum_compression = get_nf4_config()

# CPU inference (maximum compatibility)
cpu_inference = QuantizationConfig(mode=QuantizationMode.INT8_DYNAMIC)
```

### Layer Skip Configuration

```python
# Skip critical layers from quantization
config = QuantizationConfig(
    mode=QuantizationMode.INT8,
    llm_int8_skip_modules=[
        "lm_head",        # Output layer
        "embed_tokens",   # Input embeddings
        "norm",           # Layer normalization
        "gate"            # Gating mechanisms
    ]
)
```

### Testing Retention

```python
def test_quantization_retention(model, tokenizer, test_prompts):
    """Compare outputs before and after quantization."""
    # Original outputs
    original_outputs = []
    for prompt in test_prompts:
        output = generate(model, tokenizer, prompt)
        original_outputs.append(output)
    
    # Quantize model
    quantizer = AdaptiveQuantizer(get_mixed_precision_config())
    quantized_model = quantizer.quantize_model(model)
    
    # Quantized outputs
    quantized_outputs = []
    for prompt in test_prompts:
        output = generate(quantized_model, tokenizer, prompt)
        quantized_outputs.append(output)
    
    # Compare
    matches = sum(o == q for o, q in zip(original_outputs, quantized_outputs))
    retention = matches / len(test_prompts)
    print(f"Retention: {retention*100:.1f}%")
```

---

## Troubleshooting

### Common Issues

#### bitsandbytes Not Available

```
ImportError: bitsandbytes is required for int8 quantization
```

**Solution:**

```bash
pip install bitsandbytes>=0.41.0
```

#### CUDA Out of Memory During Quantization

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# Quantize on CPU first, then move to GPU
layer = layer.cpu()
quantized = quantizer.quantize_layer(layer)
quantized = quantized.to('cuda')
```

#### Dequantization Errors

```
AttributeError: 'Params4bit' object has no attribute 'dequantize'
```

**Solution:**

```python
# Ensure you're using the correct compute dtype
config = QuantizationConfig(compute_dtype=torch.float16)
quantizer = LayerQuantizer(config)
```

#### Poor Retention with 4-bit

**Symptoms:** Model outputs gibberish or significantly degraded quality

**Solutions:**

1. Use INT8 instead of NF4/FP4
2. Skip critical layers:

```python
config.llm_int8_skip_modules = ["lm_head", "embed_tokens", "self_attn"]
```

3. Use mixed precision (adaptive quantization)
2. Increase outlier threshold for INT8:

```python
config.llm_int8_threshold = 8.0  # Higher = more outliers in full precision
```

### Performance Issues

#### Slow Loading

**Symptoms:** Quantized layers loading slower than expected

**Solutions:**

1. Enable compression:

```python
config.compress_statistics = True
```

2. Check disk I/O (use SSD)
2. Increase memory cache size:

```python
cache = LayerCache(max_memory_cache_size_gb=4.0)
```

#### High CPU Usage

**Symptoms:** 100% CPU during quantization/dequantization

**Solutions:**

1. Use GPU for quantization:

```python
layer = layer.to('cuda')
quantized = quantizer.quantize_layer(layer)
```

2. Pre-quantize and cache layers
2. Use INT8_DYNAMIC for CPU (more efficient)

### Debugging

#### Check if Layer is Quantized

```python
from src.nexus_final.sli.quantization import BITSANDBYTES_AVAILABLE

if BITSANDBYTES_AVAILABLE:
    from bitsandbytes.nn import Linear8bitLt, Linear4bit
    print(isinstance(layer, (Linear8bitLt, Linear4bit)))
```

#### Inspect Quantization Config

```python
config = quantizer.config
print(f"Mode: {config.mode}")
print(f"Compute dtype: {config.compute_dtype}")
print(f"Compression: {config.compress_statistics}")
```

#### Verify Size Reduction

```python
def get_layer_size_mb(layer):
    total = 0
    for param in layer.parameters():
        total += param.numel() * param.element_size()
    return total / 1024 / 1024

original_size = get_layer_size_mb(layer)
quantized_size = get_layer_size_mb(quantized_layer)
print(f"Reduction: {original_size/quantized_size:.1f}x")
```

### Getting Help

If you encounter issues not covered here:

1. Check the [test suite](../tests/unit/test_quantization.py) for usage examples
2. Enable debug logging:

```python
import logging
logging.getLogger('src.nexus_final.sli.quantization').setLevel(logging.DEBUG)
```

3. Verify your environment:

```python
from src.nexus_final.sli.quantization import BITSANDBYTES_AVAILABLE
print(f"bitsandbytes available: {BITSANDBYTES_AVAILABLE}")
```

---

## See Also

- [Layer Caching Documentation](LAYER_CACHING.md)
- [SLI Universal Guide](SLI_UNIVERSAL_GUIDE.md)
- [Architecture Registry](architecture_registry.py)
