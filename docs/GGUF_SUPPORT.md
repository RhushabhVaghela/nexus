# GGUF Support in Nexus

Nexus provides comprehensive support for GGUF (GPT-Generated Unified Format) models through llama.cpp integration, enabling efficient CPU and GPU inference for quantized models.

## Overview

GGUF is a binary format optimized for loading and saving models for inference with llama.cpp. It supports various quantization levels that significantly reduce memory usage while maintaining model quality.

### Supported Quantizations

| Quantization | Bits/Param | Use Case |
|--------------|------------|----------|
| Q2_K | 2.5 | Maximum compression, lower quality |
| Q3_K_M | 3.5 | Balanced compression |
| Q4_K_M | 4.5 | Recommended default (good balance) |
| Q4_K_S | 4.0 | Smaller Q4 variant |
| Q5_K_M | 5.5 | Higher quality |
| Q6_K | 6.6 | Near-lossless quality |
| Q8_0 | 8.5 | Very high quality |
| F16 | 16 | Half precision |
| F32 | 32 | Full precision |

## Quick Start

### Loading a GGUF Model

```python
from nexus.models.gguf import GGUfLoader, GGUFConfig

# Configure GGUF model
config = GGUFConfig(
    model_path="/path/to/model.gguf",
    n_ctx=8192,           # Context window size
    n_batch=512,          # Batch size for prompt processing
    n_gpu_layers=-1,      # -1 = offload all to GPU, 0 = CPU only
    n_threads=-1,         # -1 = use all CPU cores
    temperature=0.7,
    top_p=0.9,
    top_k=40,
)

# Load and use model
with GGUfLoader(config) as model:
    # Text generation
    result = model.generate(
        prompt="Explain quantum computing in simple terms:",
        max_tokens=256,
        temperature=0.7
    )
    print(result["text"])
    
    # Chat completion
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    chat_result = model.chat(messages, max_tokens=256)
    print(chat_result["content"])
```

### Streaming Generation

```python
# Stream text generation
for token in model.generate(
    prompt="Write a story about",
    max_tokens=100,
    stream=True
):
    print(token, end="", flush=True)

# Stream chat
messages = [{"role": "user", "content": "Tell me a joke"}]
for token in model.chat(messages, max_tokens=100, stream=True):
    print(token, end="", flush=True)
```

## Converting Models to GGUF

### PyTorch to GGUF

```python
from nexus.models.gguf import GGUFConverter

converter = GGUFConverter()

# Convert PyTorch model to GGUF
output_path = converter.pytorch_to_gguf(
    model_path="meta-llama/Llama-2-7b",
    output_path="llama-2-7b.gguf",
    quantization="Q4_K_M",
    context_length=4096
)
```

### Using unsloth for Conversion

```python
from nexus.models.gguf import GGUFConverter

converter = GGUFConverter()

# Convert unsloth models
output_files = converter.convert_unsloth_to_gguf(
    model_name="unsloth/Llama-3.1-8B",
    output_dir="./models",
    quantizations=["Q4_K_M", "Q5_K_M", "Q6_K"]
)
```

### Size Estimation

```python
converter = GGUFConverter()

# Estimate GGUF size before conversion
size_bytes, human_readable = converter.estimate_gguf_size(
    pytorch_model_path="meta-llama/Llama-2-7b",
    quantization="Q4_K_M"
)
print(f"Estimated size: {human_readable}")

# Get quantization recommendation
quant = converter.create_quantization_config(
    target_size_gb=4.0,     # Target size
    model_params_b=7.0      # Model size in billions
)
print(f"Recommended: {quant}")
```

## Managing Multiple Models

```python
from nexus.models.gguf.gguf_loader import GGUFBatchLoader

# Load multiple models
with GGUFBatchLoader() as loader:
    # Load first model
    config1 = GGUFConfig(
        model_path="models/model1.gguf",
        n_gpu_layers=20
    )
    model1 = loader.load_model("assistant", config1)
    
    # Load second model
    config2 = GGUFConfig(
        model_path="models/model2.gguf",
        n_gpu_layers=20
    )
    model2 = loader.load_model("coder", config2)
    
    # Use models
    result1 = model1.generate("Hello")
    result2 = model2.generate("def fibonacci")
    
    # Models automatically unloaded on exit
```

## Configuration Options

### CPU-Only Inference

```python
config = GGUFConfig(
    model_path="model.gguf",
    n_gpu_layers=0,       # CPU only
    n_threads=8,          # Use 8 CPU cores
    n_ctx=4096,
)
```

### GPU Offloading

```python
config = GGUFConfig(
    model_path="model.gguf",
    n_gpu_layers=-1,      # All layers on GPU
    n_batch=1024,         # Larger batch for GPU
)
```

### Partial GPU Offloading

```python
config = GGUFConfig(
    model_path="model.gguf",
    n_gpu_layers=35,      # First 35 layers on GPU
    n_threads=4,          # Remaining on CPU with 4 threads
)
```

## Chat Formats

```python
# ChatML format
config = GGUFConfig(
    model_path="model.gguf",
    chat_format="chatml"
)

# Llama-2 format
config = GGUFConfig(
    model_path="model.gguf",
    chat_format="llama-2"
)
```

## Tokenization

```python
# Tokenize text
tokens = model.tokenize("Hello, world!")
print(f"Token count: {len(tokens)}")

# Detokenize
text = model.detokenize(tokens)
print(f"Text: {text}")

# Check context size
max_tokens = model.get_context_size()
```

## Validation and Metadata

```python
from nexus.models.gguf import GGUFConverter

converter = GGUFConverter()

# Validate GGUF file
report = converter.validate_gguf("model.gguf")
print(f"Valid: {report['valid']}")
print(f"Tensors: {report['tensor_count']}")
print(f"Size: {report['file_size']} bytes")

# Extract metadata
metadata = converter.get_gguf_metadata("model.gguf")
for key, value in metadata.items():
    print(f"{key}: {value}")
```

## Loading PyTorch State from GGUF

```python
converter = GGUFConverter()

# Extract PyTorch state dict
state_dict = converter.extract_pytorch_state(
    gguf_path="model.gguf",
    map_location="cpu"
)

# Use with PyTorch model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ... define layers

model = MyModel()
model.load_state_dict(state_dict, strict=False)
```

## Performance Tips

### For CPU Inference

- Use Q4_K_M quantization for best speed/quality balance
- Set `n_threads` to number of physical CPU cores
- Enable BLAS if available (llama.cpp compiled with OpenBLAS/MKL)

### For GPU Inference

- Use more GPU layers (`n_gpu_layers`) for better performance
- Larger batch sizes (`n_batch`) improve throughput
- Q4_K_M or Q5_K_M recommended for GPU

### Memory Management

- Use `unload()` when done with a model
- Use context managers (`with` statement) for automatic cleanup
- For multiple models, use `GGUFBatchLoader`

## Troubleshooting

### Model Won't Load

- Verify model path is correct
- Check file isn't corrupted: `converter.validate_gguf()`
- Ensure sufficient RAM/VRAM

### Slow Generation

- Increase `n_threads` for CPU
- Increase `n_gpu_layers` for GPU
- Use lower quantization (Q4_K_M vs Q6_K)

### Out of Memory

- Reduce `n_ctx` (context window)
- Reduce `n_gpu_layers` (move more to CPU)
- Use more aggressive quantization (Q3_K_M or Q2_K)

### Poor Quality

- Use higher quantization (Q5_K_M or Q6_K)
- Increase `temperature` for more creativity
- Adjust `top_p` and `top_k` for better sampling

## Popular Model Sources

### unsloth

High-quality quantized models with various sizes:

- `unsloth/Llama-3.1-8B-GGUF`
- `unsloth/Qwen2.5-14B-GGUF`
- `unsloth/Mistral-7B-v0.3-GGUF`

### TheBloke

Extensive collection of quantized models:

- `TheBloke/Llama-2-7B-GGUF`
- `TheBloke/CodeLlama-7B-GGUF`

### bartowski

Recent model quantizations:

- `bartowski/Llama-3.1-70B-GGUF`
- `bartowski/Qwen2.5-72B-GGUF`

## API Reference

### GGUfLoader

- `load()`: Load the model
- `generate()`: Generate text completion
- `chat()`: Generate chat completion
- `tokenize()`: Tokenize text
- `detokenize()`: Detokenize IDs
- `get_context_size()`: Get context window size
- `unload()`: Free memory

### GGUFConfig

- `model_path`: Path to .gguf file
- `n_ctx`: Context window size
- `n_batch`: Prompt processing batch size
- `n_gpu_layers`: GPU layer offload (-1 = all)
- `n_threads`: CPU threads (-1 = auto)
- `temperature`, `top_p`, `top_k`: Sampling parameters
- `chat_format`: Chat template format

### GGUFConverter

- `pytorch_to_gguf()`: Convert PyTorch to GGUF
- `convert_unsloth_to_gguf()`: Convert unsloth models
- `estimate_gguf_size()`: Estimate converted size
- `extract_pytorch_state()`: Extract PyTorch weights
- `get_gguf_metadata()`: Get model metadata
- `validate_gguf()`: Validate GGUF file
- `create_quantization_config()`: Recommend quantization
