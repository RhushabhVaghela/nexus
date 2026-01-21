# Nexus Model Architecture

## 🧬 Overview

The Nexus model project is designed for modular, capability-driven training of Large Language Models. It supports multimodal inputs (Text, Vision, Audio) and advanced reasoning (Chain-of-Thought).

## 🏗️ Core Components

### 1. Training Pipeline

* **`src/24_multimodal_training.py`**: The main training engine.
  * Supports Stage 1 (Projector alignment) and Stage 2 (Full fine-tuning).
  * Uses `OmniDataset` for streamable, balanced multitask loading.
* **`src/stages/`**: Contains specialized training stages (CoT, Tools, Reasoning, Omni).

### 2. Data Layer

* **`src/data/universal_loader.py`**: Discovers and loads datasets recursively.
* **Unified Schema**: All data is normalized to a `messages` list format for consistency across capabilities.

### 3. Model Unification Framework

We provide multiple paths to a single unified model:

* **Post-Hoc Merging**: Consolidate weights using Linear, Task Arithmetic, or TIES methods (`src/omni/unify_checkpoints.py`).
* **Sequential Pipeline**: Chain training runs back-to-back (`src/omni/sequential_pipeline.py`).
* **Multitask Training**: Train on all capabilities simultaneously using balanced sampling in `OmniDataset`.

### 4. Benchmarking & Tracking

* **`src/benchmarks/benchmark_runner.py`**: Unified interface for generation and perplexity benchmarks.
* **`src/metrics_tracker.py`**: Tracks training progress and handles dataset discovery logic.

---

## 🛠️ Technology Stack

* **Framework**: PyTorch + Transformers + Accelerate
* **Quantization**: GPTQ (AutoGPTQ)
* **Environment**: Conda (`nexus`)
