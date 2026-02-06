# Nexus: Universal Modular AI

![Nexus Badge](https://img.shields.io/badge/Status-Stage_6_Release-success) ![License](https://img.shields.io/badge/License-MIT-blue) ![Universal SLI](https://img.shields.io/badge/Universal_SLI-16_Families-orange) ![Research](https://img.shields.io/badge/Research_Use_Only-Lab_Experimental-red)

> **Research Project**: Nexus is an experimental research codebase for exploring Sequential Layer Ingestion (SLI) techniques. See [Performance Notes](#performance-expectations) for realistic expectations.

**Nexus** is a unified, modular AI ecosystem that distills the capabilities of **Multiple specialized "Teacher" models** into a single, efficient "Student" architecture. By leveraging advanced **Activation Anchoring (protected subspaces)** and Sparse Intent Routing, Nexus delivers state-of-the-art performance across text, vision, audio, and video—with **research-only teacher-free inference capabilities**.

> **High-Efficiency Distillation:** Nexus achieves 60-75% capability retention depending on task complexity, providing a practical balance between efficiency and performance without requiring teacher weights at runtime.

> **Universal SLI:** Process 16 architecture families (~60-70 model variants) including GPT, T5, Mamba, MoE models, and more—on consumer hardware! Encoder-only models (BERT family) have limited support for embedding extraction only.

---

## 🏆 Capability Tier Declaration

Nexus provides a tier-based capability manifest so consumers can understand the fidelity and resource requirements:

- **Tier 1 (Core):** General Language, Reasoning, Base NLP. (Optimized for <8GB VRAM, Teacher-Free)
- **Tier 2 (Pro):** Code, Tool-Use, Agent Planning. (Optimized for <12GB VRAM, Rank 512, Teacher-Free)
- **Tier 3 (Ultra):** Voice Cloning, Vision QA, Video. (Optimized for 16GB VRAM, Rank 1024, Teacher-Free)

---

## 🆕 Universal SLI (Sequential Layer Ingestion)

Nexus now features **Universal SLI**—process massive models (100B - 1T+ parameters) from **11 architecture families (~30-40 model variants)** on consumer GPUs!

### Supported Architecture Families

| Family | Count | Example Models | Notes |
|--------|-------|----------------|-------|
| **Llama** | 35 | Llama 3, Mistral, Mixtral, Qwen2, DeepSeek | Full support |
| **GPT** | 18 | GPT-2, GPT-J, GPT-NeoX, Falcon, StarCoder | Full support |
| **Qwen** | 14 | Qwen2, Qwen2.5, Qwen3, Qwen-VL, Qwen-Omni | Full support |
| **MoE** | 15 | Mixtral 8x7B, DeepSeek-MoE, Grok, Qwen2-MoE | Full support |
| **T5** | 12 | T5, FLAN-T5, UL2, LongT5 | Encoder-decoder |
| **Mamba** | 12 | Mamba, Mamba2, Jamba, Zamba, RWKV | SSM-based |
| **BERT** | 16 | BERT, RoBERTa, DeBERTa, DistilBERT | ⚠️ Encoder-only, limited |
| **Gemma** | 8 | Gemma, Gemma 2, Gemma 3 | Full support |
| **Phi** | 6 | Phi, Phi 2, Phi 3, Phi 4 | Full support |
| **BLOOM** | 5 | BLOOM, BLOOMZ | Full support |
| **OPT** | 6 | OPT, OPT-IML | Full support |
| **ChatGLM** | 8 | ChatGLM2, ChatGLM3, GLM-4 | Trust remote code |

> **Note**: While we previously claimed "135+ models," the actual implementation supports **11 distinct architecture families** covering approximately **30-40 model variants**. Encoder-only models (BERT family) have limited SLI support for embedding extraction only.

### Quick Example

```python
from src.nexus.models.sli import UniversalSLIIntegrator

# Works with ANY supported architecture!

# Llama model
integrator = UniversalSLIIntegrator("meta-llama/Llama-3.2-1B")

# GPT model
integrator = UniversalSLIIntegrator("gpt2")

# T5 model (encoder-decoder)
integrator = UniversalSLIIntegrator("google/flan-t5-base")

# MoE model
integrator = UniversalSLIIntegrator("mistralai/Mixtral-8x7B-v0.1")

# Mamba/SSM model
integrator = UniversalSLIIntegrator("state-spaces/mamba-370m")

# Run SLI pipeline
dataset = ["Sample text"]
result = integrator.run_sli(dataset)
```

### Key Features

- ✅ **Automatic Architecture Detection**—No manual configuration needed
- ✅ **MoE Support**—Native handling of Mixture of Experts
- ✅ **Multi-Format Weights**—SafeTensors, .bin, .pt, .pth
- ✅ **16 Architecture Families**—From BERT to Qwen3 (~60-70 model variants)
- ✅ **Memory Efficient**—Process massive models on consumer GPUs
- ✅ **Performance Optimizations**—Smart prefetching, activation caching
- ✅ **Resilience Patterns**—Circuit breakers, retry logic, bulkhead isolation

📚 [Universal SLI Guide](docs/SLI_UNIVERSAL_GUIDE.md) | 🔄 [Migration Guide](docs/MIGRATION_GUIDE.md) | 📖 [Technical Manual](docs/NEXUS_V6_TECHNICAL_MANUAL.md)

---

## 🚀 Advanced SLI (New in v1.2.0)

**Advanced SLI** combines three cutting-edge technologies for unprecedented performance:

| Technology | Benefit | Improvement |
|------------|---------|-------------|
| **NVFP4 Quantization** | 4-bit floating point weights | 75% memory reduction |
| **QAD Distillation** | Knowledge transfer to quantized model | 95-98% accuracy retention |
| **Nested Learning** | Multi-time-scale layer updates | 40% compute reduction |

### Key Features

- **4x faster inference** through optimized layer loading and hierarchical caching
- **75% less I/O** via three-tier caching (Hot/Warm/Cold)
- **60-75% memory reduction** with NVFP4 4-bit quantization
- **40% compute savings** through nested update scheduling
- **Production-ready** with fast/balanced/quality presets

### Quick Example

```python
from nexus.core.sli import create_advanced_integrator

# Choose your preset: fast, balanced, or quality
integrator = create_advanced_integrator(mode="balanced", device="cuda")

# Load and process with automatic optimizations
for layer_idx in range(num_layers):
    layer = integrator.load_layer("model_id", layer_idx, is_attention=(layer_idx % 2 == 0))
    output = layer(output)
    
    # Only update layers that need it (nested learning)
    if integrator.should_update(layer_idx, step):
        loss.backward()

# Compute distillation loss
loss = integrator.compute_distillation_loss(
    student_logits=student_output,
    teacher_logits=teacher_output,
    labels=labels,
)
```

### Configuration Presets

| Preset | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **Fast** | ⭐⭐⭐ | ⭐⭐ | Production inference |
| **Balanced** | ⭐⭐ | ⭐⭐⭐ | General training |
| **Quality** | ⭐ | ⭐⭐⭐ | Fine-tuning |

### Documentation

- [Advanced SLI Guide](docs/ADVANCED_SLI.md) - Complete integration guide with 600+ lines
- [NVFP4-QAD Guide](docs/NVFP4_QAD.md) - Quantization and distillation details (500+ lines)
- [Nested Learning Guide](docs/NESTED_LEARNING_SLI.md) - Multi-time-scale training (500+ lines)

### Requirements

```bash
# Base installation
pip install torch transformers

# For hardware-accelerated NVFP4 (optional, NVIDIA Ampere+)
pip install transformer-engine[pytorch]
```

---

## ⚠️ Performance Expectations

> **Important**: Nexus is a **research project** for exploring Sequential Layer Ingestion (SLI) techniques. Performance varies significantly based on hardware, model size, and configuration.

### Realistic Performance Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **SLI Baseline** | 2-8 tokens/second | Sequential Layer Ingestion |
| **Optimized SLI** | 8-16 tokens/second | With caching + prefetching |
| **Memory Efficiency** | 60-75% reduction | vs full model loading |
| **I/O Volume** | 10+ TB per run | SSD streaming required |

### Research Use Cases

✅ **Good For**:
- Researching layer-by-layer processing techniques
- Working with models that exceed VRAM capacity
- Exploring knowledge distillation architectures
- Educational purposes

❌ **Not Suitable For**:
- Production inference systems
- Real-time applications
- High-throughput workloads
- Latency-critical deployments

See [docs/PERFORMANCE_OPTIMIZATIONS.md](docs/PERFORMANCE_OPTIMIZATIONS.md) for detailed benchmarks and optimization strategies.

### Previous v6.1 Features

- 🎯 **Multimodal Training Support** - Unified embedding injection for vision, audio, video, and text with cross-modal fusion architecture
- 🎬 **Video Generation** - Stable Video Diffusion integration with memory-efficient VAE optimizations
- 🗣️ **Text-to-Speech** - Coqui TTS integration with voice cloning and streaming synthesis
- 🤖 **Multi-Agent Orchestration** - AI-powered software development with 5 specialized agents
- 📊 **3,246+ Comprehensive Tests** - Full test coverage with performance benchmarks
- 🆕 **Universal SLI** - 16 architecture family support with automatic detection

## ⚡ Performance Optimizations (v6.2)

### 8 Research-Backed Optimization Solutions

Nexus v6.2 introduces **8 cutting-edge optimization solutions** targeting the three main LLM inference bottlenecks:

| Blocker | Solutions | Speedup |
|---------|-----------|---------|
| **Sequential Dependency** | Layer Pipelining, Adaptive Skipping, Semi-Autoregressive | 2-5× |
| **Decompression Overhead** | Async Decompression, Optimized Compression | 3× |
| **Forward Pass Time** | Layer Fusion, Early Exit, Low-Rank Attention | 2-4× |

### Optimization Stack

| Optimization | Research | Speedup | Memory | Use Case |
|--------------|----------|---------|--------|----------|
| **Layer Pipelining** | EasySpec, SpecPipe, FlowSpec | 1.5-5.5× | +5% | Multi-GPU inference |
| **Adaptive Layer Skipping** | SWIFT, LayerSkip, AdaSkip | 1.8-2.2× | -15% | Variable complexity |
| **Semi-Autoregressive** | SPACE | 2-3× | +10% | Parallel generation |
| **Async Decompression** | nvCOMP | 3× | +5% | I/O bound workloads |
| **Optimized Compression** | ZSTD + Quantization | 3× | -60% | Storage efficiency |
| **Layer Fusion** | NVIDIA Blackwell | 1.3-1.5× | +2% | Kernel efficiency |
| **Early Exit Routing** | LayerSkip, DASH | 1.67× | -20% | Early termination |
| **Low-Rank Attention** | LoRA, Sparse Patterns | 2.5-4× | -30% | Long sequences |

### Quick Start

```python
from nexus.optimizations import OptimizationPipeline
from nexus.inference import OptimizedInferenceEngine

# Load optimization pipeline with all 8 solutions
pipeline = OptimizationPipeline.from_config("configs/optimization_config.yaml")

# Initialize optimized inference engine
engine = OptimizedInferenceEngine(
    model_path="meta-llama/Llama-3.1-8B",
    optimizations=pipeline,
    device="cuda"
)

# Generate with realistic expectations
output = engine.generate(
    "Explain quantum computing",
    max_new_tokens=200,
    temperature=0.7
)

print(f"Tokens/second: {engine.metrics.tokens_per_second:.1f}")
# Expect 2-16 tokens/second depending on optimization level
```

### Progressive Optimization

```python
# Enable optimizations incrementally
optimizations = [
    "async_decompression",    # Lowest risk
    "optimized_compression",
    "layer_fusion",
    "layer_skipping",
    "early_exit",
    "layer_pipelining",
    "sparse_attention",
    "semi_autoregressive",    # Highest benefit
]

for opt in optimizations:
    pipeline.enable(opt)
    # Validate accuracy remains >97%
```

📚 [Optimization Guide](docs/OPTIMIZATION_GUIDE.md) | [Performance Guide](docs/PERFORMANCE_OPTIMIZATIONS.md) | [Architecture Matrix](docs/ARCHITECTURE_COMPATIBILITY_MATRIX.md)

---

## 🌌 Universal Architecture Support

Nexus now features a **Universal Model Loader** powered by a residency-matched registry of **16 architecture families** (~60-70 model variants).

- **Any-to-Any Support**: Natively handles Qwen3-TTS, MiniCPM-V, Llama-3.2-Vision, GPT-2, T5, Mamba, MoE models, and more.
- **Robust Metadata Discovery**: Automatic extraction of hidden dims, vocab, and modality-specific configurations.
- **Unified Interface**: Standardized `UniversalSLIIntegrator` for Profiling, Distillation, and Inference.
- **Automatic Detection**: No manual architecture configuration required.

### Example: Processing Different Architectures

```python
from src.nexus.models.sli import UniversalSLIIntegrator

# GPT-2 (GPT family)
integrator = UniversalSLIIntegrator("gpt2")

# FLAN-T5 (Encoder-Decoder)
integrator = UniversalSLIIntegrator("google/flan-t5-base")

# Mixtral (MoE)
integrator = UniversalSLIIntegrator("mistralai/Mixtral-8x7B-v0.1")

# Mamba (State Space Model)
integrator = UniversalSLIIntegrator("state-spaces/mamba-370m")

# ChatGLM (requires trust_remote_code)
integrator = UniversalSLIIntegrator(
    "THUDM/chatglm3-6b",
    trust_remote_code=True
)

# All use the same API!
result = integrator.run_sli(dataset)
```

---

## 📦 Installation & Usage

- **Usage & Verification Guide**: [Full Manual](docs/NEXUS_USAGE_GUIDE.md) - Covers Live Monitoring, Inference, Benchmarking, and RAG.
- **Master Plan**: [Implementation Roadmap](implementation_roadmap.md)
- **Universal SLI Guide**: [Complete Documentation](docs/SLI_UNIVERSAL_GUIDE.md)

### 1. Development Implementation

Nexus is currently a research codebase. To run the automated pipeline, use the **Unified CLI**:

```bash
# 1. Activate Environment
conda activate nexus

# 2. Use the Unified CLI (single entry point for all operations)
./scripts/nexus.sh help                    # Show all commands
./scripts/nexus.sh master --reset          # Run full pipeline
./scripts/nexus.sh tests --unit-only       # Run tests
./scripts/nexus.sh status                  # Check pipeline status
./scripts/nexus.sh monitor                 # Real-time monitoring

# Capability pipelines
./scripts/nexus.sh pipeline all            # Text/code training
./scripts/nexus.sh multimodal all          # Multimodal training
./scripts/nexus.sh universal --enable-cot  # Universal capabilities
./scripts/nexus.sh training-suite          # Generate training scripts
./scripts/nexus.sh setup-voice             # Setup voice models

# Utility commands
./scripts/nexus.sh reset                   # Reset pipeline state
./scripts/nexus.sh cleanup                 # Clean up temporary files
```

The Unified CLI (`scripts/nexus.sh`) consolidates all previous scripts into one tool with:

- ✅ Extensive progress tracking (progress bars, ETA, spinners)
- ✅ Real-time system monitoring (GPU, memory)
- ✅ Unified command interface
- ✅ Color-coded output

See [docs/UNIFIED_CLI.md](docs/UNIFIED_CLI.md) for complete command reference.
See [examples/README.md](examples/README.md) for usage examples.

### 2. Available Options

| Option | Description |
| :--- | :--- |
| `--reset` | FULL RESET: Clear state, previous results, and checkpoints. |
| `--models <ID1,ID2>` | Filter to specific teacher models (e.g. `google_smol`) or `all`. |
| `--datasets <NAME>` | Filter datasets (e.g. `cais_mmlu`, `multimodal`), `all` (108 datasets), or specific tags. |
| `--stage <NAME>` | Run only a specific stage (profiling, extraction, training). |
| `--dry-run` | Simulate execution and verify pathing without compute. |
| `--skip-non-llm` | Skip audio/vision/multimodal teacher models. |

The pipeline will automatically:

1. **Read Registry**: Import from `nexus.core.towers.registry`.
2. **Profile Teachers (NIWT)**: Analyze activation patterns.
3. **Extract Knowledge**:
    - **Smart Download**: Automatically fetches missing datasets from Hugging Face.
    - **SLI (Massive)**: Uses "Sequential Layer Ingestion" for Teacher Models that exceed available VRAM (Memory-Aware Trigger). Supports **11 architecture families**!
4. **Train Student**: Perform multi-objective distillation with Activation Anchoring.
5. **Train Router**: Optimize the Sparse Intent Router.

---

## 🧠 The Ecosystem (Teacher Registry)

Nexus is trained on the distilled knowledge of specialized models defined in `src.nexus.models.registry`:

| **Logic & Reasoning** | Massive Reasoner (e.g. DeepSeek-70B) | Deep Reasoning capabilities | **SLI (Sequential)** |
| **Agentic** | Agent-Specialists | Long-horizon Planning | Standard |
| **Vision** | Visual-Transformers | Visual QA & Reasoning | Standard |
| **Audio** | Audio-Encoders | Speech Understanding | Standard |

---

## 🔧 Architecture

Nexus uses a **Sparse Intent Router** to dynamically activate the relevant sub-modules (Adapters) based on the input query.

- **Student Core**: **Universal Architecture** (Dynamically sized or 2B-8B) utilizing FlashAttention. Adapts to teacher dimensions.
- **The Librarian**: SSD-backed Vector Memory for infinite context lookup during training.
- **NIWT Profiler**: Neural Information-Weighted Tower for identifying critical teacher circuits.
- **Router**: Lightweight MLP for intent classification (Entropy-Regularized).
- **Universal SLI**: Process 16 architecture families (~60-70 model variants) via sequential layer ingestion.

## 📜 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

**Research Use Only**: Nexus is an experimental research project. It is not designed for production use. Performance claims are based on research benchmarks and may vary significantly in real-world applications.

- Actual throughput depends on hardware configuration
- Memory requirements vary by model size
- Some features may be incomplete or experimental
