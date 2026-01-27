# Nexus Implementation Plan

## 1. Overview
This plan defines the roadmap to reach **100% implementation** of the Nexus Universal Modular AI. It focuses on replacing placeholders with rigorous, production-ready code, optimizing the NIWT profiler, and integrating the full 15-teacher ecosystem with the specified "Gradient Surgery" and "Recovery Step" mechanisms.

## 2. Project Type
**TYPE:** `BACKEND` (AI/ML Engineering)
- **Primary Agent:** `backend-specialist`
- **Secondary Agents:** `test-engineer`, `performance-optimizer`

## 3. Success Criteria
1.  **Optimization:** `niwt_profiler.py` supports batching (Batch Size > 1) and runs 5x faster than baseline.
2.  **Architecture:** `NexusStudentCore`, `SparseIntentRouter`, and all `Adapters` (Reasoning, Vision, Audio) are fully implemented (no `pass` blocks).
3.  **Integration:** All 15+ Teacher models can be loaded via a unified `TowerLoader` with NF4 quantization.
4.  **Mechanisms:** "Gradient Surgery" (Variance Penalty) and "Recovery Step" (Loss Spike Rollback) are active in the training loop.
5.  **Verification:** 100% Unit Test coverage for Core modules; End-to-End pipeline runs successfully on sample data.

## 4. Tech Stack
- **Language:** Python 3.10+
- **Framework:** PyTorch 2.x
- **Libraries:** Transformers, BitsAndBytes (NF4), PEFT (LoRA/Adapters), Accelerate
- **Data:** Parquet, JSONL
- **Hardware:** Optimized for Single/Multi-GPU (RTX 5080/A100 target)

## 5. File Structure
```
nexus/
├── src/
│   ├── nexus_core/
│   │   ├── __init__.py
│   │   ├── student/
│   │   │   ├── __init__.py
│   │   │   ├── core.py          # NexusStudentCore (Transformer)
│   │   │   ├── router.py        # SparseIntentRouter (MLP + Heuristic)
│   │   │   └── memory.py        # State management
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Abstract Adapter
│   │   │   ├── reasoning.py     # ReasoningAdapter (CoT)
│   │   │   ├── vision.py        # VisionAdapter (SigLIP/VideoMAE)
│   │   │   └── audio.py         # AudioAdapter (Whisper/TTS)
│   │   ├── towers/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py        # Unified NF4 Loader
│   │   │   └── registry.py      # Map of 15+ teachers
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── loop.py          # Main training loop (with Recovery)
│   │   │   ├── loss.py          # Gradient Surgery (Variance Penalty)
│   │   │   └── scheduler.py     # LR Scheduling
│   │   └── profiling/
│   │       ├── __init__.py
│   │       └── niwt.py          # Refactored NIWT Profiler (Batched)
│   └── nexus_final/             # Final assembly/pipeline
├── scripts/
│   ├── nexus_pipeline.py        # Master Orchestrator
│   └── run_profiling.sh
├── tests/
│   ├── unit/
│   └── integration/
└── configs/
    └── nexus_config.yaml
```

## 6. Task Breakdown

### Phase 1: Optimization & Core (NIWT)
*Focus: Speed and Data Efficiency*

- [ ] **task-opt-01:** Refactor `scripts/niwt_profiler.py` into `src/nexus_core/profiling/niwt.py` class structure.
    - **Input:** Current script. **Output:** Class-based module.
- [ ] **task-opt-02:** Implement `SmartBatchLoader` in `niwt.py` to handle large Parquet/CSV files efficiently.
    - **Input:** `pandas` read logic. **Output:** Generator-based batch loader.
- [ ] **task-opt-03:** Implement `batch_evaluate_reasoning` in `niwt.py` to run inference in batches (BS=4/8/16).
    - **Input:** Sequential loop. **Output:** Batched `model.generate`.
- [ ] **task-opt-04:** Create `scripts/run_profiling.sh` to execute the optimized profiler with CLI arguments.
    - **Input:** Python script. **Output:** Shell script with `--batch_size` support.

### Phase 2: Architecture Implementation
*Focus: The "Brain" and "Limbs"*

- [ ] **task-arch-01:** Implement `NexusStudentCore` in `src/nexus_core/student/core.py`.
    - **Input:** Specs (4096-dim, Cross-Attn). **Output:** `nn.Module`.
- [ ] **task-arch-02:** Implement `SparseIntentRouter` in `src/nexus_core/student/router.py`.
    - **Input:** Specs (MLP + Keyword Fallback). **Output:** Routing Logic.
- [ ] **task-arch-03:** Implement `ReasoningAdapter` in `src/nexus_core/adapters/reasoning.py`.
    - **Input:** Critical Layer Projection logic. **Output:** Adapter Module with Affine Transform.
- [ ] **task-arch-04:** Implement `VisionAdapter` and `AudioAdapter`.
    - **Input:** Specs. **Output:** Adapter Modules wrapping CLIP/Whisper logic.

### Phase 3: Integration & Mechanisms
*Focus: Connecting Teachers & Stability*

- [ ] **task-int-01:** Create `src/nexus_core/towers/registry.py` defining config for all 15+ teachers (AgentCPM, GLM-4, Step3, etc.).
    - **Input:** Master Plan table. **Output:** Config Dictionary.
- [ ] **task-int-02:** Implement `TowerLoader` in `src/nexus_core/towers/loader.py` for generic NF4 loading.
    - **Input:** `bitsandbytes` config. **Output:** Universal loader function.
- [ ] **task-int-03:** Implement `GradientSurgeryLoss` in `src/nexus_core/training/loss.py`.
    - **Input:** "Importance-Weighted Gradient Surgery" formula (Variance Penalty). **Output:** Custom Loss Class.
- [ ] **task-int-04:** Implement "Recovery Step" in `src/nexus_core/training/loop.py`.
    - **Input:** Training Loop. **Output:** Logic to rollback checkpoint on loss spike.
- [ ] **task-int-05:** Create `scripts/nexus_pipeline.py` to stitch Profiling -> Distillation -> Router Training.
    - **Input:** Phase 1, 2, 3 components. **Output:** End-to-end executable.

### Phase 4: Testing & Verification
*Focus: 100% Coverage*

- [ ] **task-test-01:** Create Unit Tests for `NexusStudentCore` and `Router`.
    - **Input:** Modules. **Output:** `tests/unit/test_student.py`.
- [ ] **task-test-02:** Create Unit Tests for `GradientSurgeryLoss`.
    - **Input:** Loss function. **Output:** `tests/unit/test_loss.py`.
- [ ] **task-test-03:** Run Integration Test (Mini-Pipeline).
    - **Input:** Dummy data. **Output:** Successful end-to-end run (Profile -> Train).

## 7. Phase X: Verification Checklist
- [ ] `niwt_profiler` runs successfully with Batch Size > 1.
- [ ] All 15 Teachers can be addressed by the `TowerLoader`.
- [ ] "Recovery Step" triggers correctly during simulated loss spikes.
- [ ] `GradientSurgeryLoss` computes gradients without NaN.
- [ ] Full pipeline runs from `scripts/nexus_pipeline.py`.
- [ ] Lint Check (`pylint` / `flake8`).
- [ ] Security Scan (No hardcoded paths/secrets).
