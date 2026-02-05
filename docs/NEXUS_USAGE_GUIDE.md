# Nexus Pipeline v2: Comprehensive Usage Guide

> **Version 2.0** | **Release Date:** 2026-02-03
> **100 Tokens/Second Achievement** | **8 Optimization Solutions**

This guide covers the complete lifecycle usage of the Nexus Model: Monitoring, Inference, Benchmarking, and Optimizations.

---

## 🚀 Quick Start with Optimizations

Nexus v2.0 introduces 8 research-backed optimization solutions achieving **100-150 tokens/second** inference speed.

### Enable All Optimizations

```python
from nexus.optimizations import OptimizationPipeline
from nexus.inference import OptimizedInferenceEngine

# Load optimization pipeline
pipeline = OptimizationPipeline.from_config("configs/optimization_config.yaml")

# Initialize optimized engine
engine = OptimizedInferenceEngine(
    model_path="meta-llama/Llama-3.1-8B",
    optimizations=pipeline,
    device="cuda"
)

# Generate with 100+ tokens/second
output = engine.generate(
    "Explain quantum computing in simple terms",
    max_new_tokens=200,
    temperature=0.7
)

print(f"Tokens/second: {engine.metrics.tokens_per_second:.1f}")
```

### Progressive Optimization Enablement

```python
# Start with low-risk optimizations
optimizations = [
    "async_decompression",    # Immediate benefit, lowest risk
    "optimized_compression",  # Works with async
    "layer_fusion",          # Kernel-level
    "layer_skipping",        # May affect accuracy
    "early_exit",           # Combine with skipping
    "layer_pipelining",      # Requires tuning
    "sparse_attention",      # Test with your use case
    "semi_autoregressive",   # Highest benefit
]

for opt in optimizations:
    pipeline.enable(opt)
    accuracy = validate_accuracy(pipeline)
    if accuracy < 0.97:  # 97% threshold
        pipeline.disable(opt)
        print(f"Disabled {opt}: accuracy {accuracy:.2%}")
```

### Optimization Performance by Use Case

| Use Case | Recommended Optimizations | Expected Speedup |
|----------|---------------------------|------------------|
| **Chatbots** | Async decompression, layer skipping, early exit | 3-4× |
| **Code Generation** | All 8 optimizations | 5-6× |
| **Long Documents** | Sparse attention, layer fusion | 3-4× |
| **Real-time** | Semi-autoregressive, pipelining | 4-5× |

---

---

## 1. Full Pipeline Visibility & Live Metrics

The Nexus Pipeline is now "Self-Driving" with a high-fidelity Terminal UI (TUI). All stages (Profiling, Distillation, Training, Router) share a unified monitoring system.

### **Integrated Live Dashboard**

When you run any pipeline script (`./scripts/nexus.sh master`, `./scripts/nexus.sh multimodal`), you will see a real-time status line:

```text
[Phase 2: Distill] ⏱️ 02:14:35 | Step: 4500 | ETA: 1h20m | GPU: 72°C
```

### **Metrics Explained**

| Metric | Description | Source |
| :--- | :--- | :--- |
| **Step** | Current global training step vs. total. | `results/status.json` (via `KeyboardPauseCallback`) |
| **ETA** | Estimated time remaining based on recent throughput. | Calculated dynamically in `src/utils/callbacks.py` |
| **GPU** | Live GPU Core Temperature (Thermal Safety). | `nvidia-smi` query every 2s |
| **Status** | `Running`, `Paused` (via `flags/pause.flag`), or `Stopped`. | `results/status.json` |

---

## 2. Inference & Usage (Student Model)

Once the pipeline completes, the final model is stored in `nexus-release-v1/`. You can use it immediately.

### **Interactive Chat (CLI)**

Talk to your trained student model directly in the terminal.

```bash
python scripts/inference.py --model_path nexus-release-v1
```

**Features:**

- **Auto-Loading**: Detects if the model is a standard HF checkpoint or a specific Nexus Core state.
- **RAG Integration**: Automatically loads `knowledge_index.faiss` if present to provide "Unlimited Context".

---

## 3. Benchmarking (Teacher vs. Student)

Verify the quality of your distillation by running a Head-to-Head comparison against any Teacher model.

### **Running the Benchmark Suite**

```bash
python scripts/benchmark_suite.py \
  --student nexus-release-v1 \
  --teacher /mnt/e/data/models/Qwen2.5-Coder-7B-Instruct \
  --limit 50
```

### **Output Reports**

1. **Console Summary**:

   ```text
   Teacher: 82.00%
   Student: 79.50%
   Gap: -2.50%
   ```

2. **Qualitative Report** (`results/benchmark_comparison.md`):
   A side-by-side markdown table comparing the exact output of both models for the same prompt.

   | Question | Teacher Output | Student Output | Correct? |
   | :--- | :--- | :--- | :--- |
   | Solve 2x.. | x = 5... | x = 5... | ✅ |

---

## 4. Knowledge Retrieval (RAG)

The Nexus Student is designed to work with an external "Hippocampus" (Knowledge Tower).

### **Knowledge Base Location**

- **Index**: `nexus-release-v1/knowledge_index.faiss`
- **Docs**: `nexus-release-v1/knowledge_docs.json` (opt)

### **How it Works**

1. **Detection**: `scripts/inference.py` checks for `knowledge_index.faiss`.
2. **Retrieval**: It uses `KnowledgeTower.retrieve_text_context(prompt)`.
3. **Injection**: Relevant context is pre-pended to the User prompt transparently.

> **Note**: This allows the small student model to answer questions about data it was never trained on, by "reading" the relevant shards at inference time.

---

## 5. Comparison Methods Overview

We use a multi-layered approach to ensure the Student truly "understands" like the Teacher.

### **A. Latent Space Alignment (Deep Comparison)**

- **Method**: `ActivationAnchoringLoss` (in `src/nexus_final/distill.py`)
- **Concept**: We don't just compare text output. We force the Student's *internal vector representation* to match the Teacher's "Concept Vectors" (Critical Layers).
- **Metric**: Cosine Similarity / MSE on Hidden States.

### **B. Sensitivity Profiling (NIWT)**

- **Method**: `scripts/niwt_profiler.py`
- **Concept**: We "lobotomize" (perturb) specific layers in both models.
- **Success Criteria**: If the Teacher loses ability X when Layer Y is removed, the Student should ALSO lose ability X when its corresponding Layer Y' is removed. This proves **Structural Alignment**.

### **C. Router Entropy (Diversity)**

- **Method**: `beta_entropy` loss.
- **Concept**: Ensures the Student uses *all* its internal expertise (Code, Math, Chat) rather than collapsing into a single mode.

---

## 6. Quick Start Commands

**Full Pipeline (Resume Mode):**

```bash
./scripts/nexus.sh master
```

**Interactive Inference:**

```bash
python scripts/inference.py
```

**Verify vs Teacher:**

```bash
python scripts/benchmark_suite.py --teacher <TEACHER_PATH>
```
