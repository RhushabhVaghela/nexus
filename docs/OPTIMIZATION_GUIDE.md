# Nexus Optimization Guide

This guide details the advanced training and inference optimizations integrated into the Nexus pipeline.

## 🚀 Training Optimizations

### 1. Unsloth Integration

Nexus now supports [Unsloth](https://github.com/unslothai/unsloth) for significantly faster training and lower VRAM usage.

**Usage:**
Add the `--use_unsloth` flag to `run_nexus_master.sh` or `scripts/nexus_pipeline.py`.

```bash
./run_nexus_master.sh --use-unsloth --models qwen_main
```

**Benefits:**

- up to 3x faster training.
- 2x less VRAM usage.
- Supported architectures: Llama, Mistral, Qwen, Gemma.

### 2. Sequence Packing

Sequence packing bins multiple short sequences together to reduce padding overhead and maximize GPU utilization.

**Usage:**
Add the `--packing` flag.

```bash
./run_nexus_master.sh --packing
```

### 3. Long Context Support

Train with up to 500k context windows using optimized RoPE scaling.

**Usage:**
Specify the `--max-seq-length` parameter.

```bash
./run_nexus_master.sh --max-seq-length 32768
```

---

## 🧠 Reasoning Optimization (GRPO)

### Group Relative Policy Optimization

Distill reasoning capabilities from large thinking models (like DeepSeek-R1) into the Nexus student.

**Usage:**
Add the `--grpo` flag to activate the reasoning evolution stage.

```bash
./run_nexus_master.sh --grpo
```

---

## 🔍 Inference & Retrieval Optimizations

### FastSentenceTransformer

The `KnowledgeTower` (Librarian) now supports Unsloth's `FastSentenceTransformer` for optimized embedding and RAG performance.

**Feature Highlights:**

- **Automatic Fallback**: If `unsloth` is not installed, the system automatically reverts to standard `sentence-transformers` using `transformers`.
- **In-Memory Speed**: Significantly faster document indexing and query embedding.

**Automatic Activation:**
The system will attempt to load `FastSentenceTransformer` whenever a `KnowledgeTower` is initialized.

---

## 🛠️ Environment Setup

Ensure you are in the `nexus` conda environment:

```bash
conda activate nexus
```

To install Unsloth and its dependencies for maximum performance:

```bash
# Example installation for CUDA 12.1
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install --no-deps xformers "trl<0.13.0" peft accelerate bitsandbytes
```

> [!NOTE]
> Training and inference will still work without Unsloth, but performance will be lower (fallback mode).
