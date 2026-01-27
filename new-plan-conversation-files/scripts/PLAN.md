# Nexus Optimization & Testing Plan

## 1. Testing Strategy

### 1.1 Unit Tests (Low-Level)
- **Target**: `scripts/niwt_core.py` and `scripts/niwt_profiler.py`.
- **Scope**:
    - `ThermalProtection`: Verify check logic and cooldown simulation (mocked).
    - `NIWTCore`: Test initialization, stage execution, and helper methods.
    - `NIWTProfiler`: Test prompt formatting, model loading (mocked), and scoring logic.
- **Coverage Goal**: 100% statement coverage.

### 1.2 Integration Tests (Pipeline)
- **Target**: The flow from Data Loading -> Profiling -> Results Output.
- **Scope**:
    - Verify `niwt_profiler.py` runs end-to-end with a small dummy model/dataset.
    - Verify output JSON structure and content.
    - Verify error handling (e.g., missing files).

### 1.3 End-to-End Benchmarks (Capabilities)
- **Target**: Capabilities (CoT, Voice, Vision).
- **Scope**:
    - Benchmark script for `niwt_profiler.py` to measure samples/second.
    - Compare baseline vs. optimized (batched) performance.

## 2. Optimization Plan (`niwt_profiler.py`)

### 2.1 Batching Implementation
- **Goal**: Implement batch size 8-16 on RTX 5080.
- **Changes**:
    - Modify `evaluate_reasoning` to accept a list of questions.
    - Use `tokenizer(padding=True, return_tensors='pt')` to handle batches.
    - Use `model.generate` with batched inputs.
    - Decode and score in batch.
- **Expected Gain**: 5-10x speedup on inference.

### 2.2 Safety Protocols
- **Thermal Protection**:
    - Import/Port `ThermalProtection` class from `niwt_core.py` to `niwt_profiler.py`.
    - Inject `thermal_check()` calls inside the main profiling loops (e.g., every N batches).
- **Auto-Resume**:
    - Save intermediate state (checkpointing) to `OUTPUT_DIR`.
    - On startup, check for existing partial results and resume.

## 3. "No Retention Loss" Verification

### 3.1 Methodology (Gemma-Scope inspired)
- **Hypothesis**: If we keep critical layers, the model should retain >95% performance on the specific capability (GSM8K/Reasoning).
- **Verification Step**:
    - After profiling, perform a "Validation Run".
    - Run inference using *only* the critical layers (simulated by zeroing others).
    - Compare validation score vs. baseline score.
- **Metric**: Retention Ratio = (Validation Score / Baseline Score). Target > 0.95.

## 4. Execution Roadmap

1.  **Refactor**: Merge `ThermalProtection` into `niwt_profiler.py` and implement Batching.
2.  **Test**: Write Unit & Integration tests.
3.  **Benchmark**: Measure speedup.
4.  **Verify**: Run the "No Retention Loss" check.
