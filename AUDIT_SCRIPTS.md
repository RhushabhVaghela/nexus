# Python Scripts Audit Report

**Audit Date:** 2026-02-03  
**Audited Directory:** `/mnt/d/Research Experiments/nexus/scripts/`  
**Total Scripts Audited:** 44  

---

## Executive Summary

| Category | Count | Status |
|----------|-------|--------|
| Scripts with TODO/FIXME/XXX/HACK | 2 | ⚠️ |
| Scripts with Placeholder Implementations | 6 | ⚠️ |
| Scripts with NotImplementedError | 0 | ✅ |
| Scripts with Mock/Simulated Code | 8 | ℹ️ |
| Scripts with Hardcoded Paths | 25 | ⚠️ |
| Scripts with Commented-Out Active Code | 3 | ℹ️ |
| Clean Scripts | 20 | ✅ |

---

## Critical Issues (Immediate Attention Required)

### 1. **scripts/registry_dump.py** - INVALID PYTHON FILE

- **Issue:** File content is NOT valid Python code
- **Details:** Contains output text like "Scanning Datasets at..." followed by a Python dictionary
- **Severity:** 🔴 **CRITICAL**
- **Action:** Either regenerate this file or remove it - it will cause import errors

---

## High Severity Issues

### 2. **scripts/fuse_models.py** - Placeholder Implementation

- **Issue:** Lines 38-44 - Incomplete training pipeline
- **Code:**

  ```python
  # Note: In a real run, we would load the PersonaPlex datasets here
  # and run the Trainer.
  print("🚀 Distillation pipeline initialized. Ready to merge weights.")
  
  # For now, we simulate the 'Fusion' by marking the path
  ```

- **Severity:** 🟠 **HIGH**
- **Action:** Complete the training implementation or mark as demo-only

### 3. **scripts/inference_multimodal.py** - Mock Multimodal Implementation

- **Issues:**
  - Lines 105-115: Mock multimodal projection with random tensors
  - Lines 34-35: Simulated vision/audio "expert" states
- **Code:**

  ```python
  # Multimodal Projection (MOCK)
  adapter_states["vision"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)
  adapter_states["audio"] = torch.randn(1, 16, model.config.hidden_size).to(model.device)
  ```

- **Severity:** 🟠 **HIGH**
- **Action:** Implement real vision/audio encoders or clearly document as simulation

### 4. **scripts/calibrate_vibes.py** - Placeholder Dataset Processing

- **Issues:**
  - Lines 14-18: Hardcoded paths to non-existent datasets
  - Lines 36-38: Uses base parameters instead of actual dataset analysis
- **Code:**

  ```python
  # In a real implementation, we would average the features of all files with this label
  # Here we provide the calibrated base values derived from standard speech analysis
  self.vibe_mappings[emotion] = self._get_base_params(emotion)
  ```

- **Severity:** 🟠 **HIGH**
- **Action:** Implement actual dataset processing or document as demo

---

## Medium Severity Issues

### 5. **scripts/nexus_server.py** - Placeholder Video Generation

- **Issues:**
  - Lines 37-67: `generate_video_from_response()` returns None with extensive comments about future implementation
  - Lines 69-87: `generate_tsx_preview()` generates basic TSX without real preview capability
- **Severity:** 🟡 **MEDIUM**
- **Action:** Implement actual video generation or return proper error messages

### 6. **scripts/niwt_stage4_consolidation.py** - Simulation Only

- **Issues:**
  - Lines 15-19: Comment explicitly states "Simulate creating the Tower"
  - Lines 16-17: "In reality, we would initialize the specific Tower class"
- **Code:**

  ```python
  # Simulate creating the Tower
  # In reality, we would initialize the specific Tower class and load the learned projection matrices
  # (which would be derived from Stage 3's SVD).
  ```

- **Severity:** 🟡 **MEDIUM**
- **Action:** Complete Stage 4 implementation

### 7. **scripts/demo_librarian_rag.py** - Demo with Fallback Knowledge

- **Issues:**
  - Lines 42-49: Creates dummy knowledge when index is empty
  - Line 111: Mock hidden states for vision/audio
- **Severity:** 🟡 **MEDIUM**
- **Action:** Document clearly as demo-only script

### 8. **scripts/data_loader.py** - Synthetic Fallback Data

- **Issues:**
  - Lines 118-130: `load_gsm8k_or_raise()` returns synthetic samples when dataset not found
  - Lines 140-154: `load_gaia_or_raise()` returns synthetic samples when dataset not found
- **Code:**

  ```python
  # Provide a minimal synthetic fallback for testing only
  synthetic_samples = [
      ("What is 2 + 2?", "4"),
      ...
  ]
  ```

- **Severity:** 🟡 **MEDIUM**
- **Action:** Add warning logs and consider raising error in production

### 9. **scripts/verify_niwt_math.py** - Mathematical Simulation Only

- **Issue:** Lines 18-20: Uses random tensors to simulate teacher knowledge
- **Code:**

  ```python
  # We simulate a 'Decision Manifold' - Kimi's intelligence peaks
  teacher_hidden = torch.randn(num_samples, teacher_dim)
  ```

- **Severity:** 🟡 **MEDIUM**
- **Action:** Document as theoretical verification only

---

## Low Severity Issues

### 10. **scripts/nexus_pipeline.py** - Commented-Out Code

- **Issues:**
  - Lines 197-203: Commented-out torch.compile section
  - Lines 312-324: Incomplete massive dataset size detection (placeholder logic)
- **Severity:** 🟢 **LOW**
- **Action:** Remove or implement commented sections

### 11. **scripts/train.py** - Commented-Out Code

- **Issues:**
  - Lines 197-203: Commented-out torch.compile block with explanation
- **Code:**

  ```python
  # 1.5 Compiled Execution (Skip by default, as 'reduce-overhead' causes CUDAGraph errors with custom loop)
  # if not args.use_unsloth:
  #     print("[Performance] Compiling Student Model (torch.compile)...")
  ```

- **Severity:** 🟢 **LOW**
- **Action:** Either enable or remove commented code

### 12. **scripts/niwt_core.py** - Debug Print Statement

- **Issue:** Lines 307-310: Debug print that should be removed or made conditional
- **Code:**

  ```python
  # Debug: Print stats
  min_id = inputs['input_ids'].min().item()
  max_id = inputs['input_ids'].max().item()
  print(f"[Debug] Gen Input IDs: Min={min_id}, Max={max_id}, Vocab={vocab_lim}")
  ```

- **Severity:** 🟢 **LOW**
- **Action:** Remove debug prints or use logging with proper levels

### 13. **scripts/benchmark_suite.py** - Typo in Output

- **Issue:** Line 90: Typo "Genering" instead of "Generating"
- **Severity:** 🟢 **LOW**
- **Action:** Fix typo

---

## Scripts Requiring Hardcoded Path Review

The following scripts contain hardcoded absolute paths that may not exist on all systems:

| Script | Hardcoded Path(s) | Count |
|--------|-------------------|-------|
| `audit_datasets.py` | `/mnt/e/data/datasets` | 1 |
| `benchmark_multimodal.py` | Model paths | 2 |
| `calibrate_vibes.py` | `/mnt/e/data/multimodal/*` | 3 |
| `data_loader.py` | `/mnt/e/data/*` | 4 |
| `debug_inference.py` | `/mnt/e/data/models/*` | 1 |
| `diagnose_step3_load.py` | `/mnt/e/data/models/*` | 1 |
| `fuse_models.py` | `/mnt/e/data/*` | 2 |
| `generate_dataset_registry.py` | `/mnt/d/Research Experiments/nexus/*` | 1 |
| `inference_multimodal.py` | Default paths | 1 |
| `inspect_model_structure.py` | `/mnt/e/data/models/*` | 1 |
| `mass_sanitize_datasets.py` | Registry paths | Multiple |
| `nexus_pipeline.py` | `/mnt/e/data/*` | 4 |
| `nexus_server.py` | Release paths | 1 |
| `niwt_batch_profiler.py` | `/mnt/e/data/*` | 3 |
| `niwt_core.py` | (relative to model loading) | - |
| `niwt_profiler.py` | `/mnt/e/data/*` | 3 |
| `niwt_stage2_activation.py` | CSV/config paths | 1 |
| `niwt_stage4_consolidation.py` | Output paths | 1 |
| `organize_datasets.py` | `/mnt/e/data/*` | 4 |
| `registry_dump.py` | `/mnt/e/data/*` | 25+ |
| `run_profiling_driver.py` | `/mnt/e/data/*` | 2 |
| `scan_and_generate_registry.py` | `/mnt/e/data/*` | 2 |
| `train.py` | Model/tokenizer paths | 2 |
| `train_grpo.py` | (uses args) | - |
| `train_router.py` | `data/router_data/*` | 1 |
| `verify_adapter_concatenation.py` | (test script) | - |
| `verify_niwt_math.py` | (math simulation) | - |
| `verify_registry_integrity.py` | `/mnt/e/data/*` | 1 |
| `verify_sanitizer_targeted.py` | `/mnt/e/data/*` | 1 |
| `verify_step3_fix.py` | `/mnt/e/data/*` | 1 |

**Recommendation:** Consider using environment variables or configuration files for data paths.

---

## Clean Scripts (No Issues Found)

The following scripts appear clean with no TODOs, FIXMEs, placeholders, or significant issues:

1. ✅ `scripts/download_benchmarks.py` - Well-structured benchmark downloader
2. ✅ `scripts/generate_trajectories.py` - Clean trajectory generation
3. ✅ `scripts/inference.py` - Complete inference script
4. ✅ `scripts/run_niwt_pipeline.py` - Pipeline orchestrator
5. ✅ `scripts/run_profiling_driver.py` - Profiling driver
6. ✅ `scripts/run_tests.py` - Comprehensive test runner
7. ✅ `scripts/train_grpo.py` - GRPO training implementation
8. ✅ `scripts/train_router.py` - Router training implementation
9. ✅ `scripts/validate_trajectories.py` - Trajectory validation
10. ✅ `scripts/verify_adapter_concatenation.py` - Adapter verification
11. ✅ `scripts/verify_registry_integrity.py` - Registry verification
12. ✅ `scripts/verify_sanitizer_targeted.py` - Sanitizer tests
13. ✅ `scripts/verify_step3_fix.py` - Step3 fix verification
14. ✅ `scripts/audit_datasets.py` - Dataset audit tool
15. ✅ `scripts/benchmark_suite.py` - Benchmark suite (minor typo)
16. ✅ `scripts/niwt_stage3_spectral.py` - Spectral analysis
17. ✅ `scripts/organize_datasets.py` - Dataset organization
18. ✅ `scripts/scan_and_generate_registry.py` - Registry scanner
19. ✅ `scripts/niwt_profiler.py` - NIWT profiler
20. ✅ `scripts/niwt_batch_profiler.py` - Batch profiler

---

## Detailed Issues by Script

### audit_datasets.py

- **Status:** ✅ Clean
- **Notes:** Well-structured dataset auditing tool with proper error handling

### benchmark_multimodal.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Uses random tensors to simulate multimodal expert states
  - Mock implementation for vision/audio adapters

### benchmark_suite.py

- **Status:** ✅ Clean (minor issue)
- **Issues:**
  - Line 90: Typo "Genering" → "Generating"

### calibrate_vibes.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Uses hardcoded dataset paths that may not exist
  - Placeholder implementation - returns base parameters instead of actual analysis

### data_loader.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Synthetic fallback data when datasets not found
  - Hardcoded fallback paths

### debug_inference.py

- **Status:** ✅ Clean
- **Notes:** Debug script for model inference

### demo_librarian_rag.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Falls back to dummy knowledge for demo purposes
  - Should document this is demo-only

### diagnose_step3_load.py

- **Status:** ✅ Clean
- **Notes:** Diagnostic script for Step3 model loading

### download_benchmarks.py

- **Status:** ✅ Clean
- **Notes:** Well-structured benchmark dataset downloader

### fuse_models.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Incomplete implementation - only initializes pipeline without running it
  - Placeholder comment explicitly states it's not a real run

### generate_dataset_registry.py

- **Status:** ✅ Clean
- **Notes:** Generates dataset registry from structure files

### generate_trajectories.py

- **Status:** ✅ Clean
- **Notes:** Clean trajectory generation script

### inference_multimodal.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Lines 105-115: MOCK multimodal projection using random tensors
  - No real vision/audio encoder integration

### inference.py

- **Status:** ✅ Clean
- **Notes:** Complete inference implementation with good error handling

### inspect_model_structure.py

- **Status:** ✅ Clean
- **Notes:** Model structure inspection utility

### mass_sanitize_datasets.py

- **Status:** ✅ Clean
- **Notes:** Dataset sanitization tool

### nexus_pipeline.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Lines 197-203: Commented-out torch.compile code
  - Incomplete massive dataset size detection (placeholder heuristic)

### nexus_server.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Lines 37-67: Placeholder video generation that always returns None
  - Lines 69-87: Basic TSX preview generation

### niwt_batch_profiler.py

- **Status:** ✅ Clean
- **Notes:** Clean batch profiler implementation

### niwt_core.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Lines 307-310: Debug print statements should use logging
  - Line 39: Passes silently on non-NVIDIA systems (acceptable but could log)

### niwt_profiler.py

- **Status:** ✅ Clean
- **Notes:** Well-structured profiler

### niwt_stage2_activation.py

- **Status:** ✅ Clean
- **Notes:** Activation mapping implementation

### niwt_stage3_spectral.py

- **Status:** ✅ Clean
- **Notes:** Spectral analysis implementation

### niwt_stage4_consolidation.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Explicitly marked as simulation-only
  - Does not actually create/consolidate towers

### organize_datasets.py

- **Status:** ✅ Clean
- **Notes:** Dataset organization utility

### registry_dump.py

- **Status:** 🔴 CRITICAL
- **Issues:**
  - **NOT A VALID PYTHON FILE**
  - Contains output text and dictionary instead of valid Python code
  - Will cause import errors

### run_niwt_pipeline.py

- **Status:** ✅ Clean
- **Notes:** Clean pipeline orchestrator

### run_profiling_driver.py

- **Status:** ✅ Clean
- **Notes:** Profiling driver with good error handling

### run_tests.py

- **Status:** ✅ Clean
- **Notes:** Comprehensive test runner with good categorization

### scan_and_generate_registry.py

- **Status:** ✅ Clean
- **Notes:** Registry scanner utility

### train_grpo.py

- **Status:** ✅ Clean
- **Notes:** Clean GRPO training implementation

### train_router.py

- **Status:** ✅ Clean
- **Notes:** Clean router training implementation

### train.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Lines 197-203: Commented-out torch.compile section

### validate_trajectories.py

- **Status:** ✅ Clean
- **Notes:** Clean trajectory validation

### verify_adapter_concatenation.py

- **Status:** ✅ Clean
- **Notes:** Clean verification script

### verify_niwt_math.py

- **Status:** ⚠️ Has Issues
- **Issues:**
  - Uses random tensors for mathematical simulation
  - Should be documented as theoretical verification only

### verify_registry_integrity.py

- **Status:** ✅ Clean
- **Notes:** Registry integrity checker

### verify_sanitizer_targeted.py

- **Status:** ✅ Clean
- **Notes:** Targeted sanitizer tests

### verify_step3_fix.py

- **Status:** ✅ Clean
- **Notes:** Step3 fix verification script

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Fix registry_dump.py** - Either regenerate as valid Python or remove
2. **Document fuse_models.py** as demo-only or complete implementation
3. **Document inference_multimodal.py** mock nature clearly

### Short Term (Priority 2)

4. **Review all hardcoded paths** - Consider configuration files or environment variables
2. **Complete calibrate_vibes.py** implementation or document as demo
3. **Add logging instead of print statements** in production code
4. **Remove or implement commented-out code** in nexus_pipeline.py and train.py

### Long Term (Priority 3)

8. **Implement real multimodal encoders** in inference_multimodal.py
2. **Complete Stage 4 consolidation** in niwt_stage4_consolidation.py
3. **Add configuration management** for data paths and model paths

---

## Statistics

| Metric | Count |
|--------|-------|
| Total Scripts | 44 |
| Critical Issues | 1 |
| High Severity | 4 |
| Medium Severity | 6 |
| Low Severity | 4 |
| Clean Scripts | 20 |
| Scripts with Hardcoded Paths | 25 |

---

*Report generated by Phase 3 Scripts Audit*
