# Nexus Implementation Roadmap

**Status:** Plan Mode Active
**Source of Truth:** `execution-plan/plan-details.md`

This roadmap defines the strict task breakdown for the Nexus Execution Plan, mapping directly to the mandatory stages.

## 🛑 Global Invariants
- **Pure Distillation Only:** No teachers at inference.
- **Hardware:** RTX 5080 (16GB VRAM) target.
- **Storage:** No raw activation storage (Streaming PCA only).

---

## Stage -1: Orchestration Infrastructure
*Automation & Environment*

- [ ] **Task -1.1: Orchestrator Script Development**
    - [ ] Implement `run_nexus_master.sh` with stage resumption.
    - [ ] Add hardware-aware VRAM profiling.
- [ ] **Task -1.2: Environment Verification**
    - [ ] Verify PyTorch, BitsAndBytes, and FFmpeg dependencies.

## Stage 0: Scope Lock & Configuration
*Foundational Setup*

- [ ] **Task 0.1: Configuration Lock**
    - [ ] Create `configs/seed.txt` (Seed: 42).
    - [ ] Define immutable Dataset Paths and Capability Taxonomy.
    - [ ] Generate `configs/global_config.json`.

## Stage 1: Teacher Inventory & Retention Registry
*Machine-Readable Contracts*

- [ ] **Task 1.1: CSV Data Ingestion**
    - [ ] **Constraint:** Load metadata strictly from `ModelName-Parameters-Category-BestFeature.csv`.
    - [ ] Parse `Model Name`, `Parameters`, `Category` for every teacher.
- [ ] **Task 1.2: Registry Generation**
    - [ ] Implement `src/nexus_final/registry.py`.
    - [ ] Bind specific benchmarks (GSM8K, MMLU) to categories (Reasoning, Math).
    - [ ] Output `configs/teacher_registry.json`.

## Stage 2: NIWT Profiling (Optimization Integrated)
*Storage-Optimized Subspace Identification*

- [ ] **Task 2.1: Profiler Optimization (Speed & Memory)**
    - [ ] **Optimization:** Implement **Streaming/Incremental PCA** to avoid OOM.
    - [ ] **Constraint:** Implement batch-wise processing to keep VRAM < 16GB.
    - [ ] **Constraint:** **Never** store raw activations; discard immediately after PCA update.
- [ ] **Task 2.2: Profiling Execution**
    - [ ] Load quantized teachers (NF4/INT8).
    - [ ] Run ablation (20 samples) to find key layers.
    - [ ] Compute PCA Basis & Coefficients.
    - [ ] **Rank Audit:** Verify intrinsic dimensions against `MAX_RANK_PER_CAPABILITY`.
    - [ ] Store outputs in `niwt_stage2_output/`.

## Stage 3: Student & Adapter Architecture Synthesis
*Auto-Generated Architecture*

- [ ] **Task 3.1: Architecture Logic**
    - [ ] Implement `src/nexus_final/architect.py`.
    - [ ] Map Intrinsic Dimension (Stage 2) → Adapter Rank.
    - [ ] **Policy Enforcement:** `adapter_rank = min(intrinsic_dim, MAX_RANK)`.
    - [ ] Enforce "One Adapter per Capability" rule.
- [ ] **Task 3.2: Code Synthesis**
    - [ ] Auto-generate `src/models/nexus_student.py`.
    - [ ] Auto-generate `src/adapters/*.py`.
    - [ ] Verify dummy forward pass.

## Stage 4: Distillation with Recovery
*Training & Activation Anchoring (protected subspaces)*

- [ ] **Task 4.1: Training Loop Implementation**
    - [ ] Implement Adapter-First training (Backbone frozen).
    - [ ] Implement Loss: `L_total = L_output + α * L_activation_anchor`.
- [ ] **checkpoint: Activation Anchoring (protected subspaces)**
    - [ ] **Action:** Verify gradient health and subspace preservation.
    - [ ] **Check:** Ensure gradients flow to Adapters but NOT Backbone (if frozen).
- [ ] **Task 4.2: Recovery Mechanism**
    - [ ] Implement periodic reconstruction checks.
    - [ ] If deviation > threshold, trigger **Loss-spike rollback + anchor weight escalation**.

## Stage 5: Teacher Removal Validation
*Independence Verification*

- [ ] **Task 5.1: Teacher Deletion**
    - [ ] Implement script to physically remove/unload teacher weights.
- [ ] **checkpoint: Teacher Removal**
    - [ ] **Critical Gate:** Run inference script with NO teacher models accessible.
    - [ ] **Verify:** System must function correctly. If it crashes looking for a teacher, **FAIL**.
- [ ] **Task 5.2: Retention Verification**
    - [ ] Run GSM8K, MMLU, Voice Cosine, Vision QA.
    - [ ] **Rule:** Pass if Student Score ≥ 0.97 × Teacher Score.

## Stage 6: Packaging & Production
*Final Artifacts*

- [ ] **Task 6.1: Artifact Optimization**
    - [ ] Quantize Backbone and Adapters.
    - [ ] Prune all training-specific data.
- [ ] **Task 6.2: Bundle Assembly**
    - [ ] Create `nexus_bundle_v1/`.
    - [ ] Generate `manifest.json` with Capability Tier metadata.
    - [ ] Generate `verify_report.json`.
    - [ ] Final Size Check (Target: ~2-4GB).

## Stage 7: Cleanup and Archive
*Hygiene & Portability*

- [ ] **Task 7.1: Teacher Sanitization**
    - [ ] Securely remove all teacher weight files and training-only artifacts.
- [ ] **Task 7.2: Pipeline Archival**
    - [ ] Archive logs and intermediate PCA summaries for future reference.
