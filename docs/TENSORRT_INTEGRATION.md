# TensorRT Integration Guide

Complete guide for setting up and using TensorRT with Nexus for high-performance inference.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Building Engines](#building-engines)
- [Quantization](#quantization)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Prerequisites

### System Requirements

- NVIDIA GPU with Compute Capability 7.0 or higher (V100, T4, A100, H100, RTX 30/40 series)
- CUDA 11.8 or 12.1
- Python 3.8-3.11
- 16GB+ RAM (32GB+ recommended for large models)

### Supported Models

- LLaMA/LLaMA-2 (7B, 13B, 70B)
- Mistral (7B)
- Mixtral (8x7B)
- Falcon (7B, 40B, 180B)
- GPT-J (6B)
- GPT-NeoX (20B)
- CodeLlama (7B, 13B, 34B, 70B)

## Installation

### Install TensorRT-LLM

```bash
# Install TensorRT-LLM (CUDA 12.1)
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# For CUDA 11.8
# pip install tensorrt_llm==0.7.1 --extra-index-url https://pypi.nvidia.com
```

### Verify Installation

```python
import tensorrt_llm
print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
```

### Install Additional Dependencies

```bash
# For FP8 support (H100 only)
pip install transformer-engine

# For model conversion
pip install transformers accelerate sentencepiece protobuf

# For int4 quantization
pip install auto-gptq optimum
```

## Quick Start

### Basic Inference

```python
from nexus.models.tensorrt.inference_backend import TensorRTBackend, TensorRTConfig

# Configure backend
config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b-chat-hf",
    quantization_mode="fp16",
    max_batch_size=1,
    max_seq_length=2048,
)

# Initialize backend (automatically builds/converts engine)
backend = TensorRTBackend(config)

# Generate text
result = backend.generate(
    prompts=["What is the capital of France?"],
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

print(backend.decode(result.sequences[0]))
print(f"Tokens/sec: {result.tokens_per_second:.1f}")
```

### Using Pre-built Engine

```python
config = TensorRTConfig(
    engine_path="/path/to/trt_engine",
    quantization_mode="fp16",
)

backend = TensorRTBackend(config)
```

## Configuration

### TensorRTConfig Options

```python
from nexus.models.tensorrt.inference_backend import TensorRTConfig

config = TensorRTConfig(
    # Model source (one required)
    model_path="meta-llama/Llama-2-7b",  # HuggingFace model
    engine_path=None,  # Or pre-built TensorRT engine
    
    # Tokenizer (optional, defaults to model_path)
    tokenizer_path=None,
    
    # Quantization
    quantization_mode="fp16",  # fp32, fp16, bf16, fp8, int8, int4
    
    # Sequence configuration
    max_batch_size=1,
    max_seq_length=2048,
    
    # Device
    device="cuda",
    
    # Streaming
    enable_streaming=False,
)
```

### TRTEngineConfig (Advanced)

```python
from nexus.models.tensorrt.trt_engine import TRTEngineConfig, TRTBuildConfig, TRTQuantizationMode

build_config = TRTBuildConfig(
    max_batch_size=4,
    max_seq_length=4096,
    max_input_len=2048,
    max_output_len=2048,
    dtype="float16",
    quantization=TRTQuantizationMode.FP8,
    use_gpt_attention_plugin=True,
    use_gemm_plugin=True,
    use_layernorm_plugin=True,
    opt_level=3,
)

engine_config = TRTEngineConfig(
    model_path="meta-llama/Llama-2-7b",
    build_config=build_config,
    device="cuda",
)
```

## Building Engines

### Automatic Conversion

Nexus automatically converts HuggingFace models when an engine is not found:

```python
from nexus.models.tensorrt.model_converter import ModelConverter, ConversionConfig

config = ConversionConfig(
    model_name_or_path="meta-llama/Llama-2-7b",
    output_dir="./engines/llama-7b-fp16",
    dtype="float16",
    max_batch_size=4,
    max_seq_length=2048,
)

converter = ModelConverter(config)
engine_path = converter.convert()
```

### Manual Build Script

```bash
# Save as build_engine.py
python -c "
from nexus.models.tensorrt.model_converter import ModelConverter, ConversionConfig

config = ConversionConfig(
    model_name_or_path='meta-llama/Llama-2-7b',
    output_dir='./engines/llama-7b',
    dtype='float16',
    quantization='fp16',
    max_batch_size=1,
    max_seq_length=2048,
)

converter = ModelConverter(config)
engine = converter.convert()
print(f'Engine built: {engine}')
"
```

### Build Time Estimates

| Model | Precision | Build Time | Output Size |
|-------|-----------|------------|-------------|
| 7B | FP16 | 5-10 min | ~14 GB |
| 7B | FP8 | 10-15 min | ~7 GB |
| 13B | FP16 | 10-15 min | ~26 GB |
| 13B | FP8 | 15-20 min | ~13 GB |
| 70B | FP8 | 60-90 min | ~70 GB |

## Quantization

### FP8 Quantization (H100)

Best performance on H100 GPUs:

```python
config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b",
    quantization_mode="fp8",
    max_batch_size=4,
)

backend = TensorRTBackend(config)
# Expected: 2.5-3.5x speedup vs FP16
```

### INT8 Quantization

Good balance of speed and compatibility:

```python
config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b",
    quantization_mode="int8",
)

backend = TensorRTBackend(config)
# Expected: 2-3x speedup vs FP16
```

### INT4 Quantization

Maximum compression:

```python
config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b",
    quantization_mode="int4",
)

backend = TensorRTBackend(config)
# Expected: 3-4x speedup, 75% memory reduction
```

### Quantization Accuracy

| Mode | Perplexity Impact | Use Case |
|------|-------------------|----------|
| FP16 | ~0% | Production, accuracy-critical |
| BF16 | ~0% | Training-compatible |
| FP8 | -1-2% | Balanced performance |
| INT8 | -2-3% | Maximum throughput |
| INT4 | -5-8% | Edge deployment |

## Advanced Usage

### Streaming Generation

```python
config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b",
    quantization_mode="fp16",
    enable_streaming=True,  # Enable streaming
)

backend = TensorRTBackend(config)

# Stream tokens
for token in backend.generate_stream(
    prompt="Once upon a time",
    max_new_tokens=100,
    temperature=0.7
):
    print(token, end="", flush=True)
```

### Batch Inference

```python
prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "How do neural networks work?",
    "What is the capital of Japan?",
]

config = TensorRTConfig(
    model_path="meta-llama/Llama-2-7b",
    quantization_mode="fp16",
    max_batch_size=4,  # Process all 4 at once
)

backend = TensorRTBackend(config)

# Batch generate
results = backend.batch_generate(
    prompts=prompts,
    max_new_tokens=50,
    temperature=0.7
)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Response: {backend.decode(result.sequences[0])}")
    print(f"Tokens/sec: {result.tokens_per_second:.1f}\n")
```

### Beam Search

```python
result = backend.generate(
    prompts=["The best programming language is"],
    max_new_tokens=50,
    num_beams=4,  # Use beam search
    do_sample=False,
)
```

### Custom Sampling

```python
result = backend.generate(
    prompts=["Write a poem about AI:"],
    max_new_tokens=200,
    temperature=0.9,    # Higher = more creative
    top_p=0.95,         # Nucleus sampling
    top_k=50,           # Top-k sampling
    repetition_penalty=1.2,  # Reduce repetition
)
```

### Memory Optimization

```python
# For limited GPU memory
config = TensorTEngineConfig(
    model_path="meta-llama/Llama-2-13b",
    build_config=TRTBuildConfig(
        max_batch_size=1,
        max_seq_length=1024,  # Reduce if needed
        quantization=TRTQuantizationMode.FP8,
    ),
)

backend = TensorRTBackend(config)

# Check memory usage
stats = backend.get_stats()
print(f"GPU Memory: {stats['engine_stats']['memory']['allocated_gb']:.1f} GB")
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'tensorrt_llm'`

```bash
# Solution: Install TensorRT-LLM
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
```

**Issue**: `CUDA out of memory`

```python
# Solution: Reduce batch size or sequence length
config = TensorRTConfig(
    max_batch_size=1,
    max_seq_length=1024,  # Reduce from 2048
    quantization_mode="fp8",  # Use quantization
)
```

**Issue**: `Engine build fails`

```bash
# Solution: Check CUDA version compatibility
nvcc --version
# Should match TensorRT-LLM CUDA version

# Clear cache and retry
rm -rf ~/.cache/nexus/engines/*
```

**Issue**: `Model conversion hangs`

```bash
# Solution: Increase timeout or check GPU availability
# Run with verbose logging
export NEXUS_LOG_LEVEL=DEBUG
python your_script.py
```

**Issue**: Lower accuracy than expected

```python
# Solution: Use higher precision or calibrate
config = TensorRTConfig(
    quantization_mode="fp16",  # Instead of int8/int4
)

# Or use calibration for INT8
from nexus.models.tensorrt.model_converter import CalibrationConfig

calibration_config = CalibrationConfig(
    method="entropy",
    num_samples=512,
)
```

### Performance Tuning

**Optimize for Throughput**:

```python
config = TensorRTConfig(
    max_batch_size=8,        # Maximize batching
    quantization_mode="fp8",  # Use fastest precision
)
```

**Optimize for Latency**:

```python
config = TensorRTConfig(
    max_batch_size=1,        # Minimize queuing
    max_seq_length=512,      # Reduce if acceptable
    quantization_mode="fp16", # Best accuracy-speed balance
)
```

**Optimize for Memory**:

```python
config = TensorRTConfig(
    max_batch_size=1,
    max_seq_length=1024,
    quantization_mode="int4",  # Minimum memory
)
```

## API Reference

### TensorRTBackend

```python
class TensorRTBackend:
    def __init__(self, config: TensorRTConfig)
    def generate(self, prompts, max_new_tokens, temperature, top_p, top_k, num_beams) -> GenerationResult
    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k) -> Iterator[str]
    def batch_generate(self, prompts, max_new_tokens, **kwargs) -> List[GenerationResult]
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor
    def decode(self, token_ids, skip_special_tokens=True) -> Union[str, List[str]]
    def get_stats(self) -> Dict[str, Any]
    def reset_stats(self) -> None
```

### GenerationResult

```python
@dataclass
class GenerationResult:
    sequences: torch.Tensor           # Generated token IDs
    scores: Optional[List[torch.Tensor]]  # Log probabilities
    logits: Optional[torch.Tensor]    # Output logits (if return_logits=True)
    tokens_generated: int             # Number of new tokens
    generation_time_ms: float         # Total generation time
    tokens_per_second: float          # Throughput metric
```

### TRTQuantizationMode

```python
class TRTQuantizationMode(Enum):
    FP32 = "fp32"   # 32-bit floating point
    FP16 = "fp16"   # 16-bit floating point
    BF16 = "bf16"   # BFloat16
    FP8 = "fp8"     # 8-bit floating point (H100)
    INT8 = "int8"   # 8-bit integer
    INT4 = "int4"   # 4-bit integer
    WOQ = "woq"     # Weight-only quantization
```

## Additional Resources

- [NVIDIA TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Performance Optimizations Guide](./PERFORMANCE_OPTIMIZATIONS.md)
- [Monitoring Setup Guide](./MONITORING.md)
