# Nexus Examples

This directory contains example scripts demonstrating various features of the Nexus AI platform.

## Quick Start

```bash
# Make the unified CLI executable
chmod +x scripts/nexus.sh

# Show all available commands
./scripts/nexus.sh help
```

## Examples Overview

### 1. Basic Inference ([`basic_inference.py`](basic_inference.py))

Demonstrates basic text generation using a Nexus-compatible model.

```bash
python examples/basic_inference.py --model "microsoft/Phi-3-mini-4k-instruct"
python examples/basic_inference.py --model "gpt2" --prompt "Explain AI"
```

**Features:**

- Model loading with automatic device mapping
- Chat template support
- Temperature and max tokens control

### 2. SLI Demo ([`sli_demo.py`](sli_demo.py))

Demonstrates Sequential Layer Ingestion for processing large models on consumer hardware.

```bash
# Try with different architectures
python examples/sli_demo.py --model "gpt2"
python examples/sli_demo.py --model "google/flan-t5-base"
```

**Features:**

- Automatic architecture detection
- Memory-efficient layer-by-layer processing
- Support for 11+ architecture families

### 3. Distillation Example ([`distillation_example.py`](distillation_example.py))

Shows how to distill knowledge from a teacher model to a smaller student model.

```bash
python examples/distillation_example.py \
    --teacher "microsoft/Phi-3-medium-4k-instruct" \
    --student "microsoft/Phi-3-mini-4k-instruct" \
    --dataset "openai/gsm8k"
```

**Features:**

- Logit-based distillation with temperature scaling
- Combined hard and soft target training
- Custom trainer implementation

### 4. Multimodal Example ([`multimodal_example.py`](multimodal_example.py))

Demonstrates multimodal processing (vision, audio, video).

```bash
# Text mode
python examples/multimodal_example.py --mode text --prompt "Describe AI"

# Vision mode (with sample image)
python examples/multimodal_example.py --mode vision

# Try all modes
python examples/multimodal_example.py --mode all
```

**Features:**

- Vision understanding with image Q&A
- Audio processing
- Video frame extraction

### 5. Quantization Demo ([`quantization_demo.py`](quantization_demo.py))

Demonstrates 4-bit quantization for memory-efficient inference.

```bash
# Compare FP16 vs 4-bit
python examples/quantization_demo.py --model "gpt2" --compare

# Just show 4-bit mode
python examples/quantization_demo.py --model "meta-llama/Llama-3.2-1B"
```

**Features:**

- NF4 quantization
- Memory usage comparison
- Inference speed benchmarking

## Running Full Pipelines

Use the unified CLI for complete training pipelines:

```bash
# Text training pipeline
./scripts/nexus.sh pipeline all --mode=censored

# Multimodal pipeline
./scripts/nexus.sh multimodal all --base-model=/path/to/model

# Reasoning pipeline
./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-cot

# Universal capability pipeline
./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot --enable-reasoning

# Master self-driving pipeline
./scripts/nexus.sh master --reset

# NIWT profiling
./scripts/nexus.sh niwt --model_name="microsoft/Phi-3-mini-4k-instruct"

# Run tests
./scripts/nexus.sh tests --unit-only
./scripts/nexus.sh tests --integration-only --verbose

# Cleanup
./scripts/nexus.sh cleanup
```

## Requirements

Most examples require:

```bash
pip install torch transformers accelerate
```

Specific examples may need:

```bash
# Vision/multimodal
pip install pillow

# Video processing
pip install av

# Quantization
pip install bitsandbytes

# All dependencies
pip install -r requirements.txt
```

## Architecture Support

Nexus supports 11 architecture families:

| Family | Examples |
|--------|----------|
| Llama | Llama 3, Mistral, Mixtral, Qwen2 |
| GPT | GPT-2, GPT-J, GPT-NeoX, Falcon |
| Qwen | Qwen2, Qwen2.5, Qwen3, Qwen-VL |
| MoE | Mixtral 8x7B, DeepSeek-MoE |
| T5 | T5, FLAN-T5, UL2, LongT5 |
| Mamba | Mamba, Mamba2, Jamba, RWKV |
| BERT | BERT, RoBERTa, DeBERTa |
| Gemma | Gemma, Gemma 2, Gemma 3 |
| Phi | Phi, Phi 2, Phi 3, Phi 4 |
| BLOOM | BLOOM, BLOOMZ |
| OPT | OPT, OPT-IML |

## Documentation

- [Full Usage Guide](../docs/NEXUS_USAGE_GUIDE.md)
- [Shell Scripts Reference](../docs/SHELL_SCRIPTS.md)
- [Universal SLI Guide](../docs/SLI_UNIVERSAL_GUIDE.md)
- [Testing Guide](../docs/TESTING.md)
- [Troubleshooting](../docs/TROUBLESHOOTING.md)

## License

MIT License - See [LICENSE](../LICENSE) for details.
