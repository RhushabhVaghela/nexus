# Nexus Mitigation Strategy & Research Plan

**Date:** 2025-05-20
**Target Hardware:** ASUS Zephyrus G16 (RTX 5080 16GB VRAM, Intel Ultra 9, 32GB RAM)
**Goal:** Universal, Modular, Multi-Modal, Independent, Extremely Small Student (~7B) with 97-100% Retention.

---

## 1. Core Architecture: Specialist Towers + Activation-Guided Consolidation

To achieve the conflicting goals of "Universal/Modular" and "Extremely Small/Fast" without retention loss, we abandon the naive "Unified Latent Space" and "Weight Extraction" approaches. Instead, we implement **Specialist Towers with Activation-Guided Information Distillation (AGID)**.

### The Structure

The Nexus Student is not a monolithic model but a **Dynamic Orchestrator** of frozen expert knowledge.

* **Frozen Teachers (The Experts):** Massive models (GLM-4.7, Step3-VL, Qwen3-Omni) stored in **NF4/INT8** quantization. They behave as permanent, read-only knowledge banks.
* **Specialist Towers (The Modality Drivers):** Lightweight, independent modules that encapsulate specific teacher capabilities.
  * *Reasoning Tower:* Wraps AgentCPM + GLM-4.7.
  * *Vision Tower:* Wraps Step3-VL.
  * *Speech Tower:* Wraps Qwen3-Omni/TTS.
  * *Agent Tower:* Wraps AgentCPM.
* **Activation-Guided Projections (The Bridge):** Learned matrices that map teacher-specific high-dimensional states (e.g., 8192d) into a compressed, task-normalized latent space (e.g., 4096d or 2048d) based on *actual neuron firing patterns* (NIWT).
* **Sparse Router (The Brain):** A pruned Mixture-of-Experts (MoE) gate that activates only the necessary towers for a given prompt, ensuring ultra-low latency.

---

## 2. Failure Mode Analysis & Mitigation

We explicitly address the 10 critical failure modes identified in previous research phases.

### A. Parameter Disparity (2B vs 1T)

* **Risk:** Forcing a 1T model and a 2B model into the same latent dimension causes information loss for the large model and noise amplification for the small one.
* **Mitigation:** **Capability-Normalized Latent Allocation**.
  * We do *not* force equal dimensions.
  * Reasoning Tower (High Entropy): Maintains a rich 4096d-8192d latent.
  * Voice Tower (High Temporal Precision): Uses a focused 2048d latent.
  * The Student Core learns to ingest these variable-sized inputs via flexible cross-attention, not fixed concatenation.

### B. Context Dependency (Weight Extraction Fails)

* **Risk:** Raw weights lose meaning outside their original architecture.
* **Mitigation:** **Activation-Guided Information Distillation (AGID)**.
  * We never copy weights.
  * We distill *activation trajectories*. If the Student's adapter produces the same activation geometry as the Teacher for a specific task key (e.g., "CoT Reasoning"), the function is preserved.

### C. Catastrophic Forgetting

* **Risk:** Training on new modalities overwrites old ones.
* **Mitigation:** **Frozen Teachers & Independent Towers**.
  * Teachers are read-only.
  * Towers are trained independently.
  * Adding a "Medical Tower" later touches *zero* weights in the "Vision Tower".

### D. Latency Explosion

* **Risk:** Running 15 teachers sequentially is too slow.
* **Mitigation:** **Sparse Routing & Layer Collapse**.
  * The Router selects only 1-2 towers per token.
  * NIWT Profiling allows us to "collapse" hierarchical teacher layers. We identify that for Task X, we only need the output of Layer 24 -> 48, skipping intermediate computation.

### E. Identity Crisis (Context Switching)

* **Risk:** Conflicting logic (e.g., Creative Writing vs. Strict Coding) confuses a unified model.
* **Mitigation:** **Hard Modality Separation**.
  * Different tasks route to distinct Towers.
  * They only merge at the final "Consolidation Head", preventing internal interference.

---

## 3. Hardware Alignment (RTX 5080 - 16GB VRAM)

Running this locally requires strict memory discipline.

* **Quantization Strategy:**
  * **Teachers:** Loaded in **NF4 (4-bit NormalFloat)** via `bitsandbytes`.
    * GLM-4.7 (30B) -> ~16GB (Too big alone? We might need to offload or use Qwen 30B Int4).
    * *Correction:* GLM-4.7-Flash might be an MoE where active params are low, but total size matters. We will prioritize **Qwen3-Omni-30B-A3B-Int4** (Active 2.4B) fitting easily.
  * **Student/Adapters:** Kept in **BF16** for stability.
* **Memory Management:**
  * **Lazy Loading:** Only active Towers reside in VRAM.
  * **RAM Offloading:** Inactive frozen teachers sit in system RAM (32GB is tight but workable with NVMe swap).

---

## 4. Implementation Plan

### Phase 0: Comprehensive Tower Definition

We integrate ALL models from `ModelName-Parameters-Category-BestFeature.csv` into dedicated functional towers.

**A. Cognitive Core (Reasoning & Code)**

1. **Reasoning Tower:** `AgentCPM-Explore` (4B) - *Long-horizon agentic tasks.*
2. **Logic/Code Tower:** `zai-org/GLM-4.7-Flash` (30B MoE) - *Complex coding & logic (Active params ~3B fit in VRAM).*
3. **Knowledge Tower:** `google_gemma-scope-2-27b-pt` (27B) - *Deep world knowledge backup (System RAM offload).*

**B. Perception (Vision & Audio)**
4.  **Vision Tower:** `stepfun-ai_Step3-VL-10B` (10B) - *Primary Multimodal understanding.*
5.  **Fast Perception:** `siglip2-so400m` (400M) + `MCG-NJU_videomae-large` (300M) - *High-speed embedding & retrieval.*
6.  **Hearing Tower (Command):** `parakeet-tdt-0.6b-v3` - *Ultra-fast command ASR.*
7.  **Hearing Tower (Long-form):** `microsoft_VibeVoice-ASR` - *60min+ meeting transcription.*

**C. Interaction & Creation (Speech & Generative)**
8.  **Persona Tower:** `nvidia_personaplex-7b-v1` (7B) - *Full-duplex, personality-driven conversation.*
9.  **Speech Synthesis:** `Qwen_Qwen3-TTS` (1.7B) - *High-fidelity voice generation.*
10. **Imagination Tower:** `SD3 Medium` (2B) - *Image generation.*
11. **Video Tower:** `SVD-XT-1-1` - *Video generation.*

*Note: The system will dynamically load these towers based on the intent router to stay within the 16GB VRAM limit.*

### Phase 1: NIWT Profiling Protocol

We prioritize `AgentCPM-Explore` to validate the pipeline, then apply to other towers.

* **Protocol:**
    1. **Perturbation:** Disable layers -> Check Benchmark Drop (GSM8K/MMLU).
    2. **Activation Recording:** Record outputs of critical layers on target datasets.
    3. **Spectral Analysis:** PCA/SVD to find principal components.
* **Output:** A "Feature Bitmask" JSON for each teacher.

### Phase 2: Projection Training

Train the lightweight adapters to map the "Bitmasked Features" to the Student's latent space.

### Phase 3: Consolidation & Routing

Train the Sparse Router to switch between these projections.

---

**Next Immediate Step:** Continue the **NIWT Profiling** on `AgentCPM` while designing the `Vision Tower` adapter.
