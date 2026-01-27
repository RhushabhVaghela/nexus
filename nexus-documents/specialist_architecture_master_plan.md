# Nexus Specialist Architecture: Master Plan & Guide

## 1. Overview

The Nexus Specialist Architecture transforms a general-purpose "Teacher" model (e.g., AgentCPM-Explore) into a highly efficient "Student" ecosystem. Instead of a single monolithic model, we use a lightweight **Student Core** routed dynamically to **Specialist Towers** (Adapters) via a **Sparse Intent Router**.

### Core Philosophy

- **Activation-Guided Consolidation (AGID):** We don't just fine-tune; we *distill the internal thought process* (activations) of the teacher.
- **Sparsity:** Only activate the necessary "brain region" (Reasoning, Vision, Audio) for a given query.
- **Hardware-Aware:** Designed specifically for consumer high-end GPUs (RTX 5080, 16GB VRAM) using NF4 quantization and lazy loading.

---

## 2. Architecture Components

### A. The Student Core (`NexusStudentCore`)

- **Role:** The central "brain". Handles language understanding, routing, and integration.
- **Structure:**
  - 4096-dim Transformer Encoder/Decoder.
  - **Cross-Attention Ports:** Special layers that can "attend" to external adapters.
  - **State:** `[Student State] + [Reasoning Context] + [Vision Context]`.

### B. Specialist Adapters (`ReasoningAdapter`, `VisionAdapter`, `AudioAdapter`)

- **Role:** Specialized modules that encapsulate domain expertise.
- **Reasoning:** Distilled from CoT activations of 70B+ models. Uses **Critical Layer Projection** (only listening to the most important layers of the teacher).
- **Vision:** Wraps SigLIP/CLIP towers.
- **Audio:** Wraps Whisper/Audio encoders.

### C. Sparse Intent Router (`SparseIntentRouter`)

- **Role:** The traffic cop.
- **Mechanism:** Neural MLP classifier + Keyword Heuristic fallback.
- **Input:** User query embedding.
- **Output:** Routing mask (e.g., `[1, 0, 0]` for "Reasoning Only").

### D. The Librarian (`KnowledgeTower`)

- **Role:** Infinite Long-Term Memory.
- **Structure:** SSD-backed Vector Index (FAISS).
- **Mechanism:** **Memory Projection**. Facts are retrieved from disk and projected into the student's latent space via a `MemoryProjector`.
- **Latency:** Sub-millisecond retrieval + activation injection.

---

## 3. Implementation Pipeline

### Phase 1: Profiling (NIWT)

**Goal:** Identify *where* the Teacher "thinks".

- **Tool:** `scripts/niwt_profiler.py`
- **Metric:** Accuracy Drop per Layer when perturbed.
- **Output:** `critical_layers.json`.
- **Status:** **Complete (Initial Profile Generated).**

### Phase 2: Distillation Training

**Goal:** Transfer knowledge to Adapters.

- **Tool:** `scripts/nexus_pipeline.py` (Stage: `train_student`)
- **Method:**
    1. **Extract:** Teacher generates tokens + activations (Frozen).
    2. **Project:** Adapter projects Critical Layers -> Student Dimension.
    3. **Optimize:** Student minimizes generation loss (CE) + feature distance (MSE).

### Phase 3: Router Training

**Goal:** Teach the Router when to call whom.

- **Tool:** `scripts/nexus_pipeline.py` (Stage: `train_router`)
- **Data:** Tagged dataset of (Query, Required_Modality).

---

## 4. The Teacher Ecosystem (Full Scale)

We will integrate the following models into specialized "Towers". Each tower is frozen, quantized (NF4), and bridged to the Student Core.

| Tower Category | Primary Teachers (Frozen) | Role |
| :--- | :--- | :--- |
| **Reasoning & Logic** | **Kimi K2.5** (1T), **GLM-4.7-Flash** (30B) | Deep logic, coding, complex math (CoT). **1T Model processed via SLI.** |
| **Agentic Autonomy** | **AgentCPM-Explore** (4B) | Long-horizon planning, tool use, browsing (100+ rounds). |
| **Vision & Video** | **Step3-VL-10B**, **SigLIP2** (Enc), **VideoMAE** | Visual understanding, OCR, spatial reasoning. |
| **Audio & Speech** | **PersonaPlex-7B**, **Qwen3-TTS** (1.7B), **VibeVoice**, **Parakeet** | Full-duplex conversation, diverse TTS, cloning, ASR. |
| **Generation** | **SD3-Medium** (Img), **SVD-XT** (Vid) | High-fidelity asset generation. |
| **Analysis** | **Gemma-Scope-27B** | Interpretability and feature verification (Teacher-of-Teachers). |

### Universal Capability Matrix

1. **Thinking/CoT**: AgentCPM + GLM-4.7
2. **Omni/Streaming**: PersonaPlex + Qwen3-Tokenizer + Parakeet
3. **Vision QA**: Step3-VL + SigLIP2
4. **Creation**: SD3 + SVD + Qwen3-TTS

---

## 5. Operational Guide

- **Pause:** `Ctrl+C` triggers a graceful save.
- **Resume:** Running the script again detects the state file and prompts to continue.

### Directory Structure

```text
nexus/
├── src/nexus_core/
│   ├── adapters/      # Reasoning, Vision, Audio modules
│   ├── student/       # Core and Router
│   └── training/      # Distillation loops
├── scripts/
│   ├── niwt_core.py      # The 4-Stage Extraction Engine
│   ├── niwt_profiler.py  # (Legacy wrapper)
│   └── nexus_pipeline.py # Master automation
└── results/
    └── niwt_profiling/   # JSON profiles
```

---

## 6. Research Validity & Mitigation Strategies (The "Why")

Based on the synthesis of our 30-phase roadmap, here is how we address the fundamental failure modes of modular distillation.

### A. Hidden Representation Mismatch

- **Issue:** Different models (e.g., Qwen vs. Step3) have different LayerNorm stats and activation geometries.

- **Mitigation:** **Learned Affine Alignment per Tower**. Each Specialized Adapter includes a learnable affine transformation (scale+shift) effectively performing "Representation Calibration" (LIT) to map 30B-parameter manifolds into the 4096-Student space without distortion.

### B. Feature Dilution (The "Muddy Middle")

- **Issue:** Averaging weights or logits from opposing models (Reasoning vs. Creativity) destroys both.

- **Mitigation:** **Sparse Capability Routing**. We do not average. A Pruned Mixture-of-Experts (MoE) router ensures that for a Voice Cloning task, *only* the Audio Tower and specific "Soul" neurons are active. The Reasoning Tower remains silent.

### C. Catastrophic Forgetting

- **Issue:** Fine-tuning a student on new data overwrites old capabilities.

- **Mitigation:** **Frozen Teachers & Bridges**. The Teacher models are 4-bit quantized and **FROZEN**. The Student Core never learns the "raw task"; it only learns *how to query the teachers*. This guarantees 0% forgetfulness of the original models.

### D. The "Recovery Step" & Activation Anchoring

- **Mitigation:** **Importance-Weighted Activation Anchoring (protected subspaces)**. During Phase 2 (Distillation), we apply a specific loss term that penalizes divergence variance in "Critical Layers" (identified by NIWT) 10x more than generic layers. This forces the projection to preserve the "Soul" of the voice/logic even if it means sacrificing some generic syntax capacity.
