# NVFP4-QAD: Hardware-Accelerated Quantization and Distillation

**NVFP4-QAD** combines NVIDIA's 4-bit floating point (NVFP4) quantization with Quantization-Aware Distillation (QAD) to enable efficient training and inference of large models with minimal accuracy degradation.

---

## Table of Contents

1. [Overview](#overview)
2. [NVFP4 Format Technical Details](#nvfp4-format-technical-details)
3. [QAD Loss Explanation](#qad-loss-explanation)
4. [Mixed Precision Strategy](#mixed-precision-strategy)
5. [Hardware Requirements](#hardware-requirements)
6. [Accuracy vs Speed Trade-offs](#accuracy-vs-speed-trade-offs)
7. [Configuration Options](#configuration-options)
8. [Code Examples](#code-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is NVFP4?

NVFP4 (NVIDIA 4-bit Floating Point) is a 4-bit floating point format optimized for deep learning workloads. It uses the E4M3 format:

- **4 bits** for exponent
- **3 bits** for mantissa
- **1 bit** for sign

### What is QAD?

QAD (Quantization-Aware Distillation) transfers knowledge from a full-precision (FP32) teacher model to a quantized (NVFP4) student model, maintaining accuracy while reducing memory and compute requirements.

### Combined Benefits

| Metric | FP32 | INT8 | NVFP4 + QAD |
|--------|------|------|-------------|
| Memory per param | 4 bytes | 1 byte | 0.5 bytes |
| Relative Memory | 100% | 25% | 12.5% |
| Accuracy Retention | 100% | 90-95% | 95-98% |
| Hardware Support | All | Most | NVIDIA Ampere+ |

---

## NVFP4 Format Technical Details

### E4M3 Format Specification

```
Bit Layout: S EEEE MMM
- S: Sign bit (1 bit)
- E: Exponent (4 bits, biased by 7)
- M: Mantissa (3 bits)

Representable Range:
- Maximum: 448.0 (2^8 × 1.75)
- Minimum (normal): 2^-6 × 1.0 = 0.015625
- Minimum (subnormal): 2^-9 = 0.00195
```

### Block-wise Quantization

NVFP4 uses block-wise quantization for optimal precision:

```python
# Block size: typically 16 or 32 elements
block_size = 16

# Each block has its own scale factor
scale = amax / E4M3_MAX  # amax = max absolute value in block

# Quantization
quantized = round(tensor / scale)
quantized = clamp(quantized, -448, 448)
quantized = cast_to_fp4(quantized)

# Dequantization
dequantized = quantized.to_float() * scale
```

### Software vs Hardware Modes

| Mode | Implementation | Speed | Accuracy | Requirements |
|------|---------------|-------|----------|--------------|
| SOFTWARE | PyTorch fallback | Medium | High | None |
| HARDWARE | Transformer Engine | Fast | Highest | NVIDIA Ampere+ |
| MIXED | BF16 attention, NVFP4 FFN | Fast | High | None |

---

## QAD Loss Explanation

### Knowledge Distillation Basics

QAD transfers "dark knowledge" from a teacher model to a student model through soft targets:

```
Teacher (FP32) → Soft Probabilities ─┐
                                     ├──> QAD Loss → Student (NVFP4)
Ground Truth Labels ─────────────────┘
```

### Loss Components

The QAD loss combines multiple objectives:

```
L_total = α × L_distill + (1-α) × L_hard + β × L_hidden
```

#### 1. Distillation Loss (KL Divergence)

```python
# Temperature scaling softens distributions
temperature = 1.5

student_probs = softmax(student_logits / temperature)
teacher_probs = softmax(teacher_logits / temperature)

# KL divergence with temperature scaling
L_distill = KL(student_probs, teacher_probs) × temperature²
```

**Why temperature scaling?**

- Higher temperature (2.0): More uniform distribution, transfers more relationships
- Lower temperature (1.0): Sharper distribution, focuses on correct class
- Temperature² scaling: Accounts for softened gradients

#### 2. Hard Target Loss (Cross-Entropy)

```python
# Standard cross-entropy with label smoothing
L_hard = CrossEntropy(student_logits, labels, label_smoothing=0.1)
```

#### 3. Hidden State Matching Loss (Optional)

```python
# Match intermediate representations
L_hidden = MSE(
    normalize(student_hidden),
    normalize(teacher_hidden)
)
```

### Loss Weight Trade-offs

| Configuration | Alpha | Beta | Use Case |
|--------------|-------|------|----------|
| Distillation-focused | 0.9 | 0.3 | Small datasets, high teacher quality |
| Balanced | 0.7 | 0.3 | General training |
| Hard-target focused | 0.5 | 0.2 | Large datasets, diverse data |

---

## Mixed Precision Strategy

### Layer-Type Specific Precision

Different layer types have different sensitivity to quantization:

```python
# Recommended mixed precision configuration
nvfp4_config = NVFP4Config(
    mode=NVFP4Mode.MIXED,
    attention_dtype=torch.bfloat16,      # Keep attention high precision
    ffn_dtype=torch.float8_e4m3fn,        # Quantize FFN layers
    mixed_precision_threshold=4096,       # Dim threshold for decision
)
```

### Precision by Layer Type

| Layer Type | Recommended Precision | Rationale |
|------------|---------------------|-----------|
| Attention Q/K/V | BF16 | Critical for attention patterns |
| Attention Output | BF16 | Gradient flow stability |
| FFN Up/Down | NVFP4 | Large matrices, tolerant to quantization |
| Layer Norm | FP32 | Statistics precision important |
| Embeddings | BF16 | Vocabulary representation quality |

### Automatic Mixed Precision

```python
from nexus_final.sli import NVFP4Config, NVFP4Mode

# Automatic layer-type detection
config = NVFP4Config(
    mode=NVFP4Mode.MIXED,
    # Automatically applies BF16 to attention, NVFP4 to FFN
)
```

---

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any CUDA-capable | NVIDIA Ampere+ |
| VRAM | 8 GB | 24 GB |
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 200 GB NVMe |

### Optimal Hardware (Hardware Mode)

For [`NVFP4Mode.HARDWARE`](src/nexus_final/sli/nvfp4_loader.py:47):

| GPU Architecture | Tensor Cores | NVFP4 Support |
|-----------------|--------------|---------------|
| Ampere (A100) | 3rd Gen | Yes |
| Ada Lovelace (RTX 4090) | 4th Gen | Yes |
| Hopper (H100) | 4th Gen | Yes |
| Blackwell (RTX 50xx) | 5th Gen | Yes |
| Turing (RTX 20xx) | 2nd Gen | No (Software fallback) |
| Volta (V100) | 1st Gen | No (Software fallback) |

### Transformer Engine Installation

```bash
# For hardware acceleration
pip install transformer-engine[pytorch]

# Verify installation
python -c "from nexus_final.sli import NVFP4_AVAILABLE; print(NVFP4_AVAILABLE)"
```

---

## Accuracy vs Speed Trade-offs

### Quantization Mode Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Accuracy vs Speed Trade-off                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Accuracy │    HARDWARE          MIXED                         │
│     98%   │       ●                ●                            │
│     96%   │                      ●   ●   ●   SOFTWARE           │
│     94%   │                    ●       ●       ●                │
│     92%   │                  ●           ●                      │
│           └──────────────────────────────────────────────────   │
│                Slow ◄─────────────────────────────► Fast        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Mode Selection Guide

| Mode | Accuracy | Speed | Memory | Use Case |
|------|----------|-------|--------|----------|
| HARDWARE | 98% | Fastest | Lowest | Production, Ampere+ GPUs |
| MIXED | 96% | Fast | Low | Balanced workloads |
| SOFTWARE | 94% | Medium | Low | Development, older GPUs |

### Precision Levels

| Precision | Bits/Param | Memory | Accuracy | Speed |
|-----------|------------|--------|----------|-------|
| FP32 | 32 | 100% | 100% | Baseline |
| BF16 | 16 | 50% | 99% | 1.5x |
| INT8 | 8 | 25% | 93% | 2x |
| NVFP4 | 4 | 12.5% | 95% | 3x |
| NVFP4 + QAD | 4 | 12.5% | 97% | 3x |

---

## Configuration Options

### NVFP4Config

```python
from nexus_final.sli import NVFP4Config, NVFP4Mode

config = NVFP4Config(
    # Quantization mode
    mode=NVFP4Mode.MIXED,  # HARDWARE, SOFTWARE, or MIXED
    
    # Block size for quantization (must be multiple of 16)
    block_size=16,
    
    # Compute dtype for dequantized operations
    compute_dtype=torch.bfloat16,
    
    # Layer-specific dtypes (MIXED mode)
    attention_dtype=torch.bfloat16,
    ffn_dtype=torch.float8_e4m3fn,
    
    # Scaling options
    enable_scaling=True,           # Per-block scaling
    stochastic_rounding=True,      # Add noise during training
    amax_history_len=1024,         # History for scale calibration
    
    # Mixed precision threshold
    mixed_precision_threshold=4096,  # Dimension threshold
)
```

### QADLossConfig

```python
from nexus_final.sli import QADLossConfig, QADLossType

config = QADLossConfig(
    # Temperature for softening (1.0-2.0)
    temperature=1.5,
    
    # Weight for distillation vs hard target (0.0-1.0)
    alpha=0.7,
    
    # Weight for hidden state matching (0.0-1.0)
    beta=0.3,
    
    # Label smoothing factor (0.0-0.1)
    label_smoothing=0.1,
    
    # Loss type
    loss_type=QADLossType.KL_DIVERGENCE,  # or MSE, COSINE, COMBINED
    
    # Optional matching losses
    use_attention_matching=True,
    use_hidden_matching=True,
    
    # Gradient clipping
    gradient_clip=1.0,
    
    # Adaptive temperature
    adaptive_temperature=False,
    min_temperature=1.0,
    max_temperature=2.0,
)
```

### Preset Configurations

```python
from nexus_final.sli import get_nvfp4_config, get_qad_loss_config

# Fast configuration
nvfp4_fast = get_nvfp4_config(mode="software", block_size=32)
qad_fast = get_qad_loss_config(temperature=2.0, alpha=0.5)

# Quality configuration
nvfp4_quality = get_nvfp4_config(mode="mixed", block_size=16)
qad_quality = get_qad_loss_config(temperature=1.0, alpha=0.9)

# Balanced configuration (defaults)
nvfp4_balanced = get_nvfp4_config(mode="mixed", block_size=16)
qad_balanced = get_qad_loss_config(temperature=1.5, alpha=0.7)
```

---

## Code Examples

### Basic Quantization

```python
import torch
from nexus_final.sli import NVFP4StreamingLoader, NVFP4Config, NVFP4Mode

# Create loader
config = NVFP4Config(mode=NVFP4Mode.MIXED)
loader = NVFP4StreamingLoader(config, device="cuda")

# Create and quantize a layer
layer = torch.nn.Linear(4096, 11008).cuda()
print(f"Original dtype: {layer.weight.dtype}")

# Quantize (FFN layer - uses NVFP4)
quantized_layer = loader.quantize_layer(layer, is_attention=False)
print(f"Quantized successfully")

# Dequantize for inference
dequantized_layer = loader.dequantize_layer(quantized_layer)
print(f"Dequantized dtype: {dequantized_layer.weight.dtype}")

# Check compression
original_size = sum(p.numel() * p.element_size() for p in layer.parameters())
quantized_size = sum(b.numel() * b.element_size() for b in quantized_layer.buffers())
print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

### Training with QAD

```python
import torch
from nexus_final.sli import (
    QADDistillationLoss,
    QADLossConfig,
    NVFP4StreamingLoader,
    NVFP4Config,
)

# Setup
nvfp4_config = NVFP4Config(mode=NVFP4Mode.MIXED)
loader = NVFP4StreamingLoader(nvfp4_config, device="cuda")
qad_config = QADLossConfig(temperature=1.5, alpha=0.7)
qad_loss = QADDistillationLoss(qad_config)

# Models
teacher = load_fp32_teacher().cuda()  # Full precision
student = load_nvfp4_student().cuda()  # Quantized

# Training loop
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

for batch in dataloader:
    inputs, labels = batch
    inputs, labels = inputs.cuda(), labels.cuda()
    
    # Teacher forward (no gradients)
    with torch.no_grad():
        teacher_outputs = teacher(inputs)
        teacher_logits = teacher_outputs.logits
        teacher_hidden = teacher_outputs.hidden_states
    
    # Student forward
    student_outputs = student(inputs)
    student_logits = student_outputs.logits
    student_hidden = student_outputs.hidden_states
    
    # Compute QAD loss
    loss = qad_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        hidden_student=student_hidden[-1],
        hidden_teacher=teacher_hidden[-1],
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log statistics
    stats = qad_loss.get_stats()
    print(f"Step: Loss={stats['total_loss']:.4f}, "
          f"Distill={stats['distillation_loss']:.4f}, "
          f"Hard={stats['hard_target_loss']:.4f}")
```

### Mixed Precision Inference

```python
from nexus_final.sli import AdvancedSLIIntegrator, AdvancedSLIConfig

# Configure for mixed precision
config = AdvancedSLIConfig(
    enable_quantization=True,
    nvfp4_config=NVFP4Config(mode=NVFP4Mode.MIXED),
)
integrator = AdvancedSLIIntegrator(config)

# Process layers with automatic precision selection
model_id = "meta-llama/Llama-2-70b"
input_tensor = torch.randn(1, 128, 4096).cuda()

for layer_idx in range(num_layers):
    # Automatically applies BF16 to attention, NVFP4 to FFN
    is_attention = (layer_idx % 2 == 0)
    layer = integrator.load_layer(
        model_id, 
        layer_idx,
        is_attention=is_attention
    )
    input_tensor = layer(input_tensor)
```

### Per-Layer QAD Loss

```python
from nexus_final.sli import PerLayerQADLoss

# Create per-layer loss for progressive distillation
per_layer_loss = PerLayerQADLoss(
    config=QADLossConfig(temperature=1.5),
    num_layers=12,
    layer_weights=[0.1] * 10 + [0.5, 1.0],  # Higher weights for deeper layers
)

# Training with layer outputs
loss = per_layer_loss(
    layer_outputs_student=student_layer_outputs,  # List of tensors
    layer_outputs_teacher=teacher_layer_outputs,
    final_logits_student=student_logits,
    final_logits_teacher=teacher_logits,
    labels=labels,
)
```

### Custom Quantization with Stochastic Rounding

```python
from nexus_final.sli import NVFP4Quantizer, NVFP4Config

# Enable stochastic rounding for training
config = NVFP4Config(
    mode=NVFP4Mode.SOFTWARE,
    stochastic_rounding=True,  # Add noise for better training
)
quantizer = NVFP4Quantizer(config)

# Quantize tensor
tensor = torch.randn(4096, 4096).cuda()
quantized = quantizer.quantize_tensor(tensor, name="custom_tensor")

# Dequantize
dequantized = quantizer.dequantize_tensor(quantized)

# Check error
error = (tensor - dequantized).abs().mean()
print(f"Mean quantization error: {error:.6f}")
```

---

## Best Practices

### 1. Choose the Right Mode

```python
# Production inference on Ampere+
config = NVFP4Config(mode=NVFP4Mode.HARDWARE)

# Development or older GPUs
config = NVFP4Config(mode=NVFP4Mode.SOFTWARE)

# Best trade-off
config = NVFP4Config(mode=NVFP4Mode.MIXED)
```

### 2. Tune Temperature for Your Data

```python
# Small dataset / high teacher quality
config = QADLossConfig(temperature=1.0, alpha=0.9)

# Large dataset / diverse data
config = QADLossConfig(temperature=2.0, alpha=0.5)

# Balanced (recommended start)
config = QADLossConfig(temperature=1.5, alpha=0.7)
```

### 3. Use Block Size Wisely

```python
# Smaller blocks = better accuracy, more overhead
config = NVFP4Config(block_size=16)  # Default, good balance

# Larger blocks = faster, less accurate
config = NVFP4Config(block_size=32)  # For speed-critical applications
```

### 4. Enable Stochastic Rounding for Training

```python
# During training
config = NVFP4Config(stochastic_rounding=True)

# During inference
config = NVFP4Config(stochastic_rounding=False)
```

### 5. Monitor Quantization Error

```python
def check_quantization_error(original, quantized, dequantized):
    """Monitor quantization quality."""
    compression = original.numel() * 4 / (quantized.data.numel() * 0.5)
    error = (original - dequantized).abs().mean() / original.abs().mean()
    
    print(f"Compression: {compression:.2f}x")
    print(f"Relative error: {error:.4%}")
    
    if error > 0.05:  # 5% threshold
        print("Warning: High quantization error, consider smaller block size")
```

---

## Troubleshooting

### Issue: High Quantization Error

**Symptoms:** Model accuracy drops significantly after quantization

**Solutions:**

```python
# 1. Use smaller block size
config = NVFP4Config(block_size=16)  # Instead of 32

# 2. Use mixed precision
config = NVFP4Config(mode=NVFP4Mode.MIXED)

# 3. Increase distillation weight
qad_config = QADLossConfig(alpha=0.9)  # More weight on teacher

# 4. Enable hidden state matching
qad_config = QADLossConfig(
    use_hidden_matching=True,
    beta=0.5,
)
```

### Issue: Slow Quantization Speed

**Symptoms:** Quantization takes too long during training

**Solutions:**

```python
# 1. Use larger block size
config = NVFP4Config(block_size=32)  # 2x faster than 16

# 2. Use hardware mode if available
config = NVFP4Config(mode=NVFP4Mode.HARDWARE)

# 3. Cache quantized layers
loader.cache_layer(model_id, layer_idx, quantized_layer)
```

### Issue: Out of Memory

**Symptoms:** CUDA OOM during quantization

**Solutions:**

```python
# 1. Process in smaller chunks
for chunk in tensor.chunk(4):
    quantized_chunk = quantizer.quantize_tensor(chunk)

# 2. Use CPU offloading
loader = NVFP4StreamingLoader(config, device="cpu")

# 3. Clear cache regularly
loader.clear_cache()
```

### Issue: Distillation Not Converging

**Symptoms:** Student model loss plateaus early

**Solutions:**

```python
# 1. Adjust temperature
qad_config = QADLossConfig(temperature=2.0)  # Softer targets

# 2. Enable adaptive temperature
qad_config = QADLossConfig(
    adaptive_temperature=True,
    min_temperature=1.0,
    max_temperature=2.0,
)

# 3. Check teacher quality
# Ensure teacher model is well-trained

# 4. Balance distillation and hard targets
qad_config = QADLossConfig(alpha=0.5)  # Equal weight
```

---

## Further Reading

- [Advanced SLI Guide](ADVANCED_SLI.md) - Complete integration guide
- [Nested Learning Guide](NESTED_LEARNING_SLI.md) - Efficient training scheduling
- [NVIDIA FP8/FP4 Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/) - Hardware details
- [Distillation Paper](https://arxiv.org/abs/1503.02531) - Knowledge distillation theory

---

## API Reference

### Key Classes

| Class | Purpose | File |
|-------|---------|------|
| [`NVFP4StreamingLoader`](src/nexus_final/sli/nvfp4_loader.py:357) | Main loader with quantization | nvfp4_loader.py |
| [`NVFP4Quantizer`](src/nexus_final/sli/nvfp4_loader.py:136) | Low-level quantization | nvfp4_loader.py |
| [`NVFP4Config`](src/nexus_final/sli/nvfp4_loader.py:53) | Quantization configuration | nvfp4_loader.py |
| [`QADDistillationLoss`](src/nexus_final/sli/qad_loss.py:156) | Distillation loss module | qad_loss.py |
| [`QADLossConfig`](src/nexus_final/sli/qad_loss.py:39) | Loss configuration | qad_loss.py |
| [`PerLayerQADLoss`](src/nexus_final/sli/qad_loss.py:525) | Layer-wise distillation | qad_loss.py |

---

**Last Updated:** 2026-02-01  
**Version:** 1.2.0  
**Maintainer:** Nexus Team
