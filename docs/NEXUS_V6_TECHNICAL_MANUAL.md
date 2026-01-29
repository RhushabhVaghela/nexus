# 🧠 Nexus Pipeline v6.1: The Complete Technical Manual

This document consolidates all technical insights, optimizations, and design decisions implemented during the restoration and stabilization of the Nexus Self-Driving Pipeline.

---

## 🚀 1. Universal "Sticky" Persistence

The pipeline now features a fully stateful configuration engine. This ensures that a resumed run is identical to the initial run without needing repetitive CLI flags.

### What is Persisted?

- **Teacher Selection**: List of targeted experts (e.g., `coder`, `translation`).
- **Dataset Mapping**: Recursive paths to locally discovered datasets.
- **Optimization States**: `use_unsloth`, `packing`, and `max_seq_length`.
- **Hyperparameters**: `lr`, `epochs`, `sample_size`, and Router configurations.

> [!NOTE]
> Persistence is managed via `.pipeline_state.json`. Manually editing this file allows for precision "Surgery" on pipeline state.

---

## 🧪 2. Distillation Loss Theory

Nexus uses **Importance-Weighted Activation Anchoring** (protected subspaces).

### Why is the Loss "High"? (e.g., 14.0)

Unlike standard training, Nexus loss is a multi-objective sum:

1. **Output Logit Match**: Ensures the student's *answers* match the teacher.
2. **Hidden State MSE**: Ensures the student's *internal logic* matches.
3. **Critical Layer Surge**: We apply a **10x multiplier** to specific "Soul" layers of the teacher model to prevent logic drift.

### The Scaling Paradox

- **Fewer Samples**: Lower loss, but high risk of "Parroting" (overfitting).
- **More Samples**: Higher loss, but creates a "Self-Driving Specialist" that generalizes to new prompts.

---

## 🛡️ 3. Memory & Stability Guardrails

### Inference RAM Management

Traditional LLM loading greedily allocates massive RAM. We implemented:

- **Meta-Device Initialization**: The model is created as a 0-byte skeleton.
- **Direct-to-GPU Streaming**: Weights are streamed from disk directly to the RTX 5080, bypassing RAM altogether.

### Training Stability

- **Eager Attention**: Fixed `ValueError` by bypassing incompatible SDPA implementations in custom student layers.
- **CUDAGraph Safety**: Disabled `torch.compile(reduce-overhead)` for the distillation loop to prevent runtime crashes during backpropagation.

---

## ⏱️ 4. Process Integrity

### Persistent Timer Fix

A `trap` handler was integrated into `run_nexus_master.sh`.

- **Logic**: Senses `SIGINT`, `SIGTERM`, and `EXIT` signals.
- **Action**: Immediately terminates the background monitoring thread and clears the terminal progress line, preventing log leakage after the script exits.

---

## 📈 5. Scaling Roadmap

To achieve "Production Grade" intelligence, use the following scaling ladder:

| Mode | Command Flag | Benefit |
| :--- | :--- | :--- |
| **Debug** | `--sample_size 50` | Proves the math works in minutes. |
| **Standard** | `--sample_size 500` | Inherits basic stylistic traits. |
| **Production** | `--sample_size 5000+` | Full inheritance of Teacher's Reasoning IQ. |

---
*Created by Antigravity for Nexus v6.1 Implementation.*
