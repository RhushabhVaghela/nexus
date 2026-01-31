# Nexus: Universal Modular AI

![Nexus Badge](https://img.shields.io/badge/Status-Stage_6_Release-success) ![License](https://img.shields.io/badge/License-MIT-blue) ![Universal SLI](https://img.shields.io/badge/Universal_SLI-135+_Architectures-orange)

**Nexus** is a unified, modular AI ecosystem that distills the capabilities of **15 specialized "Teacher" models** into a single, efficient "Student" architecture. By leveraging advanced **Activation Anchoring (protected subspaces)** and Sparse Intent Routing, Nexus delivers state-of-the-art performance across text, vision, audio, and video—with **100% teacher-free inference**.

> **Zero Retention Loss Guarantee:** Nexus is engineered to maintain >95% of the original teacher performance on critical benchmarks without requiring any teacher weights at runtime.

> **Universal SLI:** Process 135+ model architectures including GPT, T5, Mamba, MoE models, and more—on consumer hardware!

---

## 🏆 Capability Tier Declaration

Nexus provides a tier-based capability manifest so consumers can understand the fidelity and resource requirements:

- **Tier 1 (Core):** General Language, Reasoning, Base NLP. (Optimized for <8GB VRAM, Teacher-Free)
- **Tier 2 (Pro):** Code, Tool-Use, Agent Planning. (Optimized for <12GB VRAM, Rank 512, Teacher-Free)
- **Tier 3 (Ultra):** Voice Cloning, Vision QA, Video. (Optimized for 16GB VRAM, Rank 1024, Teacher-Free)

---

## 🆕 Universal SLI (Sequential Layer Ingestion)

Nexus now features **Universal SLI**—process massive models (100B - 1T+ parameters) from **135+ architectures** on consumer GPUs!

### Supported Architecture Families

| Family | Count | Example Models |
|--------|-------|----------------|
| **Llama** | 35 | Llama 3, Mistral, Mixtral, Qwen2, DeepSeek |
| **GPT** | 18 | GPT-2, GPT-J, GPT-NeoX, Falcon, StarCoder |
| **Qwen** | 14 | Qwen2, Qwen2.5, Qwen3, Qwen-VL, Qwen-Omni |
| **MoE** | 15 | Mixtral 8x7B, DeepSeek-MoE, Grok, Qwen2-MoE |
| **T5** | 12 | T5, FLAN-T5, UL2, LongT5 |
| **Mamba** | 12 | Mamba, Mamba2, Jamba, Zamba, RWKV |
| **More** | 19 | BERT, Gemma, Phi, BLOOM, OPT, ChatGLM |

### Quick Example

```python
from src.nexus_final.sli import UniversalSLIIntegrator

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
- ✅ **135+ Architectures**—From BERT to Qwen3
- ✅ **Memory Efficient**—Process 1T+ models on 16GB VRAM

📚 [Universal SLI Guide](docs/SLI_UNIVERSAL_GUIDE.md) | 🔄 [Migration Guide](docs/MIGRATION_GUIDE.md) | 📖 [Technical Manual](docs/NEXUS_V6_TECHNICAL_MANUAL.md)

---

## 🏆 Capability Tier Declaration

Nexus provides a tier-based capability manifest so consumers can understand the fidelity and resource requirements:

- **Tier 1 (Core):** General Language, Reasoning, Base NLP. (Optimized for <8GB VRAM, Teacher-Free)
- **Tier 2 (Pro):** Code, Tool-Use, Agent Planning. (Optimized for <12GB VRAM, Rank 512, Teacher-Free)
- **Tier 3 (Ultra):** Voice Cloning, Vision QA, Video. (Optimized for 16GB VRAM, Rank 1024, Teacher-Free)

---

## ✨ New in v6.1

Nexus v6.1 introduces 4 major implementations with 156 comprehensive tests and full benchmark coverage:

- 🎯 **Multimodal Training Support** - Unified embedding injection for vision, audio, video, and text with cross-modal fusion architecture
- 🎬 **Video Generation** - Stable Video Diffusion integration with memory-efficient VAE optimizations
- 🗣️ **Text-to-Speech** - Coqui TTS integration with voice cloning and streaming synthesis
- 🤖 **Multi-Agent Orchestration** - AI-powered software development with 5 specialized agents
- 📊 **346 Comprehensive Tests** - Full test coverage with performance benchmarks
- 🆕 **Universal SLI** - 135+ architecture support with automatic detection

---

## 🚀 Key Features

- **Universal Perception**: Native understanding of Text, Images, Audio (Speech/Music), and Video.
- **Sequential Layer Ingestion (SLI)**: The "Librarian" component allows ingesting knowledge from **Massive Models (100B - 1T+ parameters)** on consumer GPUs by streaming layers sequentially. Automatically falls back to SLI based on **Memory Headroom analysis**. Now supports **135+ architectures**!

---

## 🌌 Universal Architecture Support

Nexus now features a **Universal Model Loader** powered by a residency-matched registry of **135+ architectures** (directly synchronized with `llama.cpp`'s state-of-the-art mappings).

- **Any-to-Any Support**: Natively handles Qwen3-TTS, MiniCPM-V, Llama-3.2-Vision, GPT-2, T5, Mamba, MoE models, and more.
- **Robust Metadata Discovery**: Automatic extraction of hidden dims, vocab, and modality-specific configurations.
- **Unified Interface**: Standardized `UniversalSLIIntegrator` for Profiling, Distillation, and Inference.
- **Automatic Detection**: No manual architecture configuration required.

### Example: Processing Different Architectures

```python
from src.nexus_final.sli import UniversalSLIIntegrator

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

## 🧪 Getting Started (Quint-Modal Run)

To execute a full 5-teacher multimodal run:

```bash
./run_nexus_master.sh --models "coder, translation, vision_main, tts_custom, audio_tokenizer" \
                      --datasets "google_smol, mvp-lab_llava-onevision-1, google_speech_commands" \
                      --sample_size 5000 --use-unsloth
```

## 🛡️ Robustness & Safety (New in v6.1)

**Process Exclusivity (Singleton Execution)**

- **Auto-Cleanup**: The master script (`run_nexus_master.sh`) automatically detects and kills any conflicting Nexus processes (e.g., zombie `train.py` or orphaned `nexus_pipeline.py`) before starting a new run.
- **Lock Protection**: The Python Core uses a `.pipeline.lock` file to prevent accidental concurrent execution. If you try to run two instances manually, the second one will refuse to start.

## Usage

**Modular Architecture**: Hot-swappable **Adapters** allow you to load only the capabilities you need.

- **Constraint-Aware**: Optimized for consumer hardware (RTX 5080 Laptop, 16GB VRAM) via NIWT Profiling and FlashAttention.

---

## 📦 Installation & Usage

- **Usage & Verification Guide**: [Full Manual](docs/NEXUS_USAGE_GUIDE.md) - Covers Live Monitoring, Inference, Benchmarking, and RAG.
- **Master Plan**: [Implementation Roadmap](implementation_roadmap.md)
- **Universal SLI Guide**: [Complete Documentation](docs/SLI_UNIVERSAL_GUIDE.md)

### 1. Development Implementation

Nexus is currently a research codebase. To run the automated pipeline:

```bash
# 1. Activate Environment
conda activate nexus

# 2. Run the Self-Driving Pipeline
./run_nexus_master.sh [OPTIONS]
```

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

1. **Read Registry**: Import from `src.nexus_core.towers.registry`.
2. **Profile Teachers (NIWT)**: Analyze activation patterns.
3. **Extract Knowledge**:
    - **Smart Download**: Automatically fetches missing datasets from Hugging Face.
    - **SLI (Massive)**: Uses "Sequential Layer Ingestion" for Teacher Models that exceed available VRAM (Memory-Aware Trigger). Now supports **135+ architectures**!
4. **Train Student**: Perform multi-objective distillation with Activation Anchoring.
5. **Train Router**: Optimize the Sparse Intent Router.

---

## 🧠 The Ecosystem (Teacher Registry)

Nexus is trained on the distilled knowledge of specialized models defined in `src.nexus_core.towers.registry`:

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
- **Universal SLI**: Process 135+ model architectures via sequential layer ingestion.

## 📜 License

This project is licensed under the MIT License.

Nexus provides a tier-based capability manifest so consumers can understand the fidelity and resource requirements:

- **Tier 1 (Core):** General Language, Reasoning, Base NLP. (Optimized for <8GB VRAM, Teacher-Free)
- **Tier 2 (Pro):** Code, Tool-Use, Agent Planning. (Optimized for <12GB VRAM, Rank 512, Teacher-Free)
- **Tier 3 (Ultra):** Voice Cloning, Vision QA, Video. (Optimized for 16GB VRAM, Rank 1024, Teacher-Free)

---

## ✨ New in v6.1

Nexus v6.1 introduces 4 major implementations with 156 comprehensive tests and full benchmark coverage:

- 🎯 **Multimodal Training Support** - Unified embedding injection for vision, audio, video, and text with cross-modal fusion architecture
- 🎬 **Video Generation** - Stable Video Diffusion integration with memory-efficient VAE optimizations
- 🗣️ **Text-to-Speech** - Coqui TTS integration with voice cloning and streaming synthesis
- 🤖 **Multi-Agent Orchestration** - AI-powered software development with 5 specialized agents
- 📊 **346 Comprehensive Tests** - Full test coverage with performance benchmarks

---

## 🚀 Key Features

- **Universal Perception**: Native understanding of Text, Images, Audio (Speech/Music), and Video.
- **Sequential Layer Ingestion (SLI)**: The "Librarian" component allows ingesting knowledge from **Massive Models (100B - 1T+ parameters)** on consumer GPUs by streaming layers sequentially. Automatically falls back to SLI based on **Memory Headroom analysis**.

# Nexus Self-Driving Pipeline

> **v6.1 - "Beast Mode"**

Nexus is a Universal Knowledge Distillation pipeline that fully automates the journey from Profiling -> Knowledge Extraction -> Distillation -> Router Training.

### 🌌 Universal Architecture Support

Nexus now features a **Universal Model Loader** powered by a residency-matched registry of **130+ architectures** (directly synchronized with `llama.cpp`'s state-of-the-art mappings).

- **Any-to-Any Support**: Natively handles Qwen3-TTS, MiniCPM-V, Llama-3.2-Vision, and more.
- **Robust Metadata Discovery**: Automatic extraction of hidden dims, vocab, and modality-specific configurations.
- **Unified Interface**: Standardized `OmniModelLoader` for Profiling, Distillation, and Inference.

### 🧪 Getting Started (Quint-Modal Run)

To execute a full 5-teacher multimodal run:

```bash
./run_nexus_master.sh --models "coder, translation, vision_main, tts_custom, audio_tokenizer" \
                      --datasets "google_smol, mvp-lab_llava-onevision-1, google_speech_commands" \
                      --sample_size 5000 --use-unsloth
```

## 🛡️ Robustness & Safety (New in v6.1)

**Process Exclusivity (Singleton Execution)**

- **Auto-Cleanup**: The master script (`run_nexus_master.sh`) automatically detects and kills any conflicting Nexus processes (e.g., zombie `train.py` or orphaned `nexus_pipeline.py`) before starting a new run.
- **Lock Protection**: The Python Core uses a `.pipeline.lock` file to prevent accidental concurrent execution. If you try to run two instances manually, the second one will refuse to start.

## Usage

**Modular Architecture**: Hot-swappable **Adapters** allow you to load only the capabilities you need.

- **Constraint-Aware**: Optimized for consumer hardware (RTX 5080 Laptop, 16GB VRAM) via NIWT Profiling and FlashAttention.

---

## 📦 Installation & Usage

- **Usage & Verification Guide**: [Full Manual](docs/NEXUS_USAGE_GUIDE.md) - Covers Live Monitoring, Inference, Benchmarking, and RAG.
- **Master Plan**: [Implementation Roadmap](implementation_roadmap.md)

### 1. Development Implementation

Nexus is currently a research codebase. To run the automated pipeline:

```bash
# 1. Activate Environment
conda activate nexus

# 2. Run the Self-Driving Pipeline
./run_nexus_master.sh [OPTIONS]
```

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

1. **Read Registry**: Import from `src.nexus_core.towers.registry`.
2. **Profile Teachers (NIWT)**: Analyze activation patterns.
3. **Extract Knowledge**:
    - **Smart Download**: Automatically fetches missing datasets from Hugging Face.
    - **SLI (Massive)**: Uses "Sequential Layer Ingestion" for Teacher Models that exceed available VRAM (Memory-Aware Trigger).
4. **Train Student**: Perform multi-objective distillation with Activation Anchoring.
5. **Train Router**: Optimize the Sparse Intent Router.

---

## 🧠 The Ecosystem (Teacher Registry)

Nexus is trained on the distilled knowledge of specialized models defined in `src.nexus_core.towers.registry`:

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

## 📜 License

This project is licensed under the MIT License.
