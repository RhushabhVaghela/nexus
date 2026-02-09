# Nexus v6.1.0 — Comprehensive Release Plan

**Created:** 2026-02-09
**Project Type:** BACKEND (Python ML/AI Platform)
**Status:** Phase A COMPLETE → Phases B–E READY FOR EXECUTION
**Environment:** WSL2/Linux, Python 3.12, 16GB RAM

---

## Overview

Nexus is a **Universal Modular AI — Knowledge Distillation Platform** with Sequential Layer Ingestion (SLI) supporting 16 architecture families (~60-70 model variants). The codebase has been restructured (Phase A complete, 7 commits ahead of origin/main) and now requires test stabilization, production hardening, optional NVComp-ZSTD integration, and final release tagging.

### Project Scale

| Metric | Value |
|--------|-------|
| Source files (non-init) | 267 |
| Source files (total .py) | 312 |
| Packages under `src/nexus/` | 20 |
| Test files | 239 |
| Test directories | 14 |
| Tests collected (pytest --co) | 3,813 |
| Collection errors | 9 |
| Uncommitted changes | 10 files |
| Commits ahead of origin | 7 |
| Branches | `main` (active), `audit/bootstrap` |

### Package Breakdown

| Package | Files | Purpose |
|---------|-------|---------|
| `models/` | 61 | SLI, distillation, registry, quantization |
| `core/` | 40 | Orchestration, towers, pipeline |
| `utils/` | 36 | Hardware, cache, metrics, callbacks |
| `training/` | 25 | DPO, ORPO, scripts (25 training scripts) |
| `multimodal/` | 24 | Vision, audio, video, cross-modal fusion |
| `stages/` | 17 | Pipeline stages |
| `data/` | 14 | Loaders, streaming, datasets |
| `optimizations/` | 13 | 8 research-backed optimization modules |
| `benchmarks/` | 11 | Performance benchmarking suite |
| `reasoning/` | 5 | CoT, ring attention, context extension |
| `voice_engine/` | 4 | TTS, voice cloning |
| `streaming/` | 4 | Real-time streaming |
| `security/` | 3 | Auth, audit, rate limiting |
| `optimization/` | 2 | KV cache, remotion engine |
| `monitoring/` | 2 | Prometheus, collectors |
| `config/` | 2 | Validation, memory config |
| `cli/` | 2 | Unified CLI entry point |
| `inference/` | 1 | Inference engine |
| `api/` | 1 | FastAPI explainer API |

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Test collection | 0 errors | `pytest --co -q` shows no ERROR lines |
| Test pass rate | ≥95% | `pytest -q --tb=no` → passed/(passed+failed) |
| No hanging tests | All complete in <60s each | `pytest-timeout` enforced at 30s |
| Source code clean | 0 TODOs/FIXMEs/stubs | `rg "TODO\|FIXME\|NotImplementedError"` = 0 ✅ ALREADY MET |
| Bare excepts | 0 | `rg "^\s*except\s*:"` = 0 ✅ ALREADY MET |
| Package installable | `pip install -e .` works | Import smoke test passes |
| Documentation accurate | All imports use `nexus.X` | No `from src.X` references |
| Release tagged | `v6.1.0` tag exists | `git tag -l v6.1.0` |

---

## Tech Stack

| Technology | Version | Rationale |
|------------|---------|-----------|
| Python | 3.9-3.12 | Broad ML ecosystem support |
| PyTorch | ≥2.1.0 | Core ML framework |
| Transformers | ≥4.39.0 | Model loading, 16 arch families |
| FastAPI | ≥0.109.0 | API layer |
| pytest | ≥7.0.0 | Test framework |
| pytest-mock | (MISSING) | **Must install** — mocker fixture |
| pytest-timeout | (MISSING) | **Must install** — prevent hangs |
| pytest-cov | 7.0.0 | Coverage reporting |

---

## Current State Analysis

### What's Already Clean (No Work Needed)

- ✅ **0 TODOs/FIXMEs/HACK/XXX** in source code
- ✅ **0 bare `except:` clauses** in source code
- ✅ **0 `raise NotImplementedError`** stubs in source code
- ✅ **0 bare `pass` statements** used as placeholders
- ✅ Package structure at `src/nexus/` is correct
- ✅ `pyproject.toml` properly configured
- ✅ All 20 packages have `__init__.py` files

### What Needs Work

| Issue | Count | Severity |
|-------|-------|----------|
| Collection errors | 9 | **CRITICAL** — blocks testing |
| Hanging tests | Unknown (tests timeout) | **CRITICAL** — blocks CI |
| Missing pytest plugins | 2 (mock, timeout) | **HIGH** — needed for fixes |
| Uncommitted changes | 10 files | MEDIUM |
| test_router.py metaclass conflict | 1 | HIGH |
| Root-level audit .md clutter | ~30 files | LOW |

### Collection Errors (9 files)

| # | File | Error Type | Root Cause |
|---|------|-----------|------------|
| 1 | `tests/unit/test_pipeline_orchestrator.py` | ImportError | Imports `PipelineStage` which doesn't exist in module |
| 2 | `tests/unit/test_nested_learning.py` | ImportError | Broken import path post-restructure |
| 3 | `tests/nexus_final/test_massive_model_sli.py` | ImportError | Likely broken import |
| 4 | `tests/nexus_final/test_sli_streaming.py` | ImportError | Likely broken import |
| 5 | `tests/test_generators.py` | ImportError | Root-level test, wrong import path |
| 6 | `tests/test_repetition_integration.py` | ImportError | Root-level test, wrong import path |
| 7 | `tests/test_sft_enhancements.py` | ImportError | Root-level test, wrong import path |
| 8 | `tests/test_temperature_scheduling.py` | ImportError | Root-level test, wrong import path |
| 9 | `tests/test_validators.py` | ImportError | Root-level test, wrong import path |

### Additional Collection Issue (Not in ERROR list)

| File | Error Type | Root Cause |
|------|-----------|------------|
| `tests/unit/test_router.py` | TypeError | Metaclass conflict — ABC/Mock collision |

---

## Phase Dependency Graph

```
PHASE B: Fix Test Suite ──────────────────────────────────────────┐
  B.0 (install deps)                                               │
    → B.1 (fix 9 collection errors + metaclass bug)                │
      → B.2 (identify & fix hanging tests)                         │
        → B.3 (run full suite, fix failures by directory)          │
          → B.4 (coverage report, document skips)                  │
            → B.COMMIT                                             │
                                                                   │
PHASE C: Audit & Production Readiness ◄───────────────────────────┘
  C.1 (verify source cleanliness) ← Already clean!                 │
  C.2 (architecture compatibility matrix)                          │
  C.3 (documentation audit & cleanup)                              │
  C.4 (root file cleanup)                                          │
  C.5 (coverage gap analysis)                                      │
    → C.COMMIT                                                     │
                                                                   │
PHASE D: NVComp-ZSTD Evaluation ◄─────────────────────────────────┘
  D.1 (audit existing compression_optimized.py)                    │
  D.2 (evaluate GPU-accelerated nvCOMP benefit)                    │
  D.3 (integrate or document decision)                             │
    → D.COMMIT                                                     │
                                                                   │
PHASE E: Final Release ◄──────────────────────────────────────────┘
  E.1 (commit uncommitted changes)
  E.2 (merge audit/bootstrap if needed)
  E.3 (final full test run)
  E.4 (tag v6.1.0)
  E.5 (push to origin)
```

---

## PHASE B: Fix Test Suite

**Goal:** 0 collection errors, 0 hanging tests, ≥95% pass rate.
**Estimated Time:** 4-6 hours
**Risk Level:** HIGH — cascading import failures, unknown hanging tests

### B.0 — Install Missing Dependencies

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| B.0.1 | `pip install pytest-mock pytest-timeout` | None | 2 min | LOW |
| B.0.2 | Add `pytest-mock` and `pytest-timeout` to `pyproject.toml` `[project.optional-dependencies.dev]` | None | 2 min | LOW |
| B.0.3 | Add `timeout = 30` to `pytest.ini` `[pytest]` section (global timeout) | B.0.1 | 2 min | LOW |
| B.0.4 | Verify: `python3 -m pytest --co -q tests/unit/test_multi_agent.py` collects without error | B.0.1 | 1 min | LOW |

**INPUT:** Current `pyproject.toml`, `pytest.ini`
**OUTPUT:** Both plugins installed and configured
**VERIFY:** `pip show pytest-mock pytest-timeout` succeeds

### B.1 — Fix Collection Errors (9 files + 1 metaclass bug)

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| B.1.1 | Fix `test_pipeline_orchestrator.py` — update imports to match actual exports from `pipeline_orchestrator.py` | `tests/unit/test_pipeline_orchestrator.py` | 15 min | HIGH |
| B.1.2 | Fix `test_nested_learning.py` — update import paths post-restructure | `tests/unit/test_nested_learning.py` | 10 min | MEDIUM |
| B.1.3 | Fix 5 root-level test files — update `from src.X` → proper import paths | `tests/test_generators.py`, `tests/test_repetition_integration.py`, `tests/test_sft_enhancements.py`, `tests/test_temperature_scheduling.py`, `tests/test_validators.py` | 20 min | MEDIUM |
| B.1.4 | Fix 2 nexus_final tests — update imports | `tests/nexus_final/test_massive_model_sli.py`, `tests/nexus_final/test_sli_streaming.py` | 15 min | MEDIUM |
| B.1.5 | Fix `test_router.py` metaclass conflict — refactor mock to avoid ABC collision | `tests/unit/test_router.py` | 20 min | HIGH |
| B.1.6 | Verify: `python3 -m pytest --co -q 2>&1 \| grep ERROR` returns empty | All above | 2 min | LOW |

**INPUT:** 10 broken test files
**OUTPUT:** 0 collection errors on `pytest --co`
**VERIFY:** `pytest --co -q` shows `N tests collected, 0 errors`

**Rollback:** `git checkout -- tests/` to restore original test files

### B.2 — Identify and Fix Hanging Tests

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| B.2.1 | Run `pytest tests/unit/ -v --timeout=30 --tb=short -x 2>&1` — identify first hanging test | Unknown until run | 10 min | HIGH |
| B.2.2 | For each hanging test: add `@pytest.mark.timeout(30)` or fix blocking I/O (socket, file lock, GPU wait) | To be determined | 30 min | HIGH |
| B.2.3 | Run `pytest tests/integration/ -v --timeout=30 -x` — identify hangs | Unknown | 10 min | HIGH |
| B.2.4 | Run `pytest tests/e2e/ -v --timeout=30 -x` — identify hangs | Unknown | 10 min | MEDIUM |
| B.2.5 | Add `conftest.py` with global timeout fixture if many tests hang on I/O | `tests/conftest.py` | 10 min | LOW |
| B.2.6 | Mark genuinely slow tests with `@pytest.mark.slow` and `@pytest.mark.timeout(120)` | Various | 15 min | LOW |

**INPUT:** Test suite that hangs indefinitely
**OUTPUT:** All tests complete within timeout, hangers identified and fixed/skipped
**VERIFY:** `pytest tests/ --timeout=30 -q --tb=no` completes within 10 minutes

**Risk Mitigation:** Common hanging causes in this codebase:
- Tests trying to load real models from HuggingFace → mock with `@patch`
- Tests opening sockets (FastAPI, Prometheus) → mock server or use `httpx` test client
- Tests waiting for GPU operations → skip with `@pytest.mark.gpu`
- Infinite loops in retry/resilience code → add timeout to retry logic

### B.3 — Fix Test Failures by Directory (Sequential)

| ID | Task | Directory | Est. | Risk |
|----|------|----------|------|------|
| B.3.1 | Run & fix `tests/unit/` (~80 files, ~2000 tests) | `tests/unit/` | 60 min | **HIGH** |
| B.3.2 | Run & fix `tests/security/` | `tests/security/` | 15 min | MEDIUM |
| B.3.3 | Run & fix `tests/integration/` | `tests/integration/` | 20 min | MEDIUM |
| B.3.4 | Run & fix `tests/voice/` | `tests/voice/` | 15 min | LOW |
| B.3.5 | Run & fix `tests/e2e/` | `tests/e2e/` | 15 min | LOW |
| B.3.6 | Run & fix `tests/multimodal/` | `tests/multimodal/` | 15 min | MEDIUM |
| B.3.7 | Run & fix `tests/chaos/` | `tests/chaos/` | 10 min | LOW |
| B.3.8 | Run & fix `tests/edge_cases/` | `tests/edge_cases/` | 10 min | LOW |
| B.3.9 | Run & fix `tests/regression/` | `tests/regression/` | 10 min | LOW |
| B.3.10 | Run & fix `tests/benchmarks/` | `tests/benchmarks/` | 10 min | LOW |
| B.3.11 | Run & fix `tests/load/` | `tests/load/` | 10 min | LOW |
| B.3.12 | Run & fix `tests/nexus_final/` | `tests/nexus_final/` | 10 min | LOW |
| B.3.13 | Run & fix `tests/unit_streaming/` | `tests/unit_streaming/` | 10 min | LOW |
| B.3.14 | Run & fix root-level `tests/test_*.py` | `tests/` (root) | 15 min | MEDIUM |

**Command Template:**
```bash
python3 -m pytest tests/{dir}/ -v --timeout=30 --tb=short -x 2>&1 | head -100
```

**INPUT:** Tests that fail or error
**OUTPUT:** Each directory passes or has documented skip reasons
**VERIFY:** `pytest tests/{dir}/ -q --tb=no --timeout=30` shows ≥95% pass rate per dir

**Memory Constraint:** Run ONE directory at a time. Kill process between runs if memory pressure.

### B.4 — Coverage and Summary

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| B.4.1 | Run full suite: `pytest tests/ --cov=nexus --cov-report=html --cov-report=term-missing --timeout=30 -q` | B.3.* | 15 min | LOW |
| B.4.2 | Document all skipped tests with justification in `COVERAGE_REPORT.md` | B.4.1 | 15 min | LOW |
| B.4.3 | Verify ≥95% pass rate (calculate: passed / (passed + failed)) | B.4.1 | 5 min | LOW |

**INPUT:** Full test suite
**OUTPUT:** Coverage HTML report, updated `COVERAGE_REPORT.md`
**VERIFY:** Pass rate ≥ 95%, coverage report generated

### B.COMMIT

```bash
git add pyproject.toml pytest.ini tests/ COVERAGE_REPORT.md
git commit -m "test: fix all collection errors, hanging tests, achieve 95%+ pass rate"
```

---

## PHASE C: Full Codebase Audit & Production Readiness

**Goal:** Production-quality codebase with accurate documentation.
**Estimated Time:** 3-4 hours
**Risk Level:** LOW-MEDIUM
**Dependencies:** Phase B complete (for accurate test counts in docs)

### C.1 — Source Code Cleanliness (ALREADY VERIFIED ✅)

| ID | Task | Status | Verification |
|----|------|--------|-------------|
| C.1.1 | Zero TODOs/FIXMEs | ✅ DONE | `rg "TODO\|FIXME" --type py src/nexus/` = 0 results |
| C.1.2 | Zero NotImplementedError | ✅ DONE | `rg "raise NotImplementedError" --type py src/nexus/` = 0 results |
| C.1.3 | Zero bare excepts | ✅ DONE | `rg "^\s*except\s*:" --type py src/nexus/` = 0 results |
| C.1.4 | Zero placeholder `pass` | ✅ DONE | All `pass` statements audited as clean |

**No work needed here — source is already clean.**

### C.2 — Architecture Compatibility Verification

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| C.2.1 | Verify all 20 packages import cleanly: `python3 -c "import nexus.{pkg}"` for each | None (verification only) | 15 min | MEDIUM |
| C.2.2 | Verify cross-package dependencies don't create circular imports | All `__init__.py` files | 20 min | MEDIUM |
| C.2.3 | Verify `optimization/` vs `optimizations/` don't conflict (both exist) | `src/nexus/optimization/`, `src/nexus/optimizations/` | 10 min | LOW |
| C.2.4 | Run `pip install -e .` smoke test with `python3 -c "from nexus.core import *; from nexus.models import *"` | `pyproject.toml` | 10 min | MEDIUM |
| C.2.5 | Update `docs/ARCHITECTURE_COMPATIBILITY_MATRIX.md` if any incompatibilities found | `docs/ARCHITECTURE_COMPATIBILITY_MATRIX.md` | 15 min | LOW |

**INPUT:** 20 packages, potential circular imports
**OUTPUT:** All packages import cleanly, compatibility matrix updated
**VERIFY:** `python3 -c "import nexus"` succeeds without warnings/errors

### C.3 — Documentation Audit

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| C.3.1 | Scan all docs for `from src.X` import patterns → replace with `from nexus.X` | All `docs/*.md` | 15 min | LOW |
| C.3.2 | Update `README.md` — remove duplicate content (lines 154-393 noted in audit) | `README.md` | 20 min | LOW |
| C.3.3 | Update `README.md` — verify all code examples are runnable | `README.md` | 15 min | LOW |
| C.3.4 | Update `docs/TEST_SUITE.md` with accurate test counts from Phase B | `docs/TEST_SUITE.md` | 10 min | LOW |
| C.3.5 | Update `CHANGELOG.md` with v6.1.0 entries | `CHANGELOG.md` | 15 min | LOW |
| C.3.6 | Update `docs/API_REFERENCE.md` to match current module paths | `docs/API_REFERENCE.md` | 15 min | LOW |
| C.3.7 | Verify performance claims are realistic (3-8 tok/s SLI, not 100+) | All `docs/PERFORMANCE*.md` | 10 min | MEDIUM |

**INPUT:** 80+ doc files, some with outdated information
**OUTPUT:** All docs accurate, no stale import paths, realistic claims
**VERIFY:** `rg "from src\." docs/ --type md` returns 0 results

### C.4 — Root File Cleanup

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| C.4.1 | Move ~30 root-level audit/plan `.md` files to `docs/archive/` | Root `*.md` files → `docs/archive/` | 10 min | LOW |
| C.4.2 | Keep only: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`, `ROADMAP.md` at root | Root directory | 5 min | LOW |
| C.4.3 | Remove `__pycache__/` directories from source tree | All `__pycache__/` dirs | 2 min | LOW |

**Files to archive (move to `docs/archive/`):**
```
ACTION_PLAN_100_PERCENT.md
AUDIT_CONFIGS.md, AUDIT_NEW_SOLUTIONS.md, AUDIT_SCRIPTS.md
AUDIT_SOURCE_CODE.md, AUDIT_TESTS.md
COMPLETION_REPORT.md, COMPREHENSIVE_CODEBASE_AUDIT_REPORT.md
COVERAGE_REPORT.md (old one — new one from Phase B stays)
DEPLOYMENT_INFRASTRUCTURE.md
DOCUMENTATION_AUDIT_REPORT.md
EXECUTION_PLAN_v6.1.0.md
FINAL_AUDIT_REPORT.md, FINAL_COMPREHENSIVE_AUDIT_REPORT.md
FINAL_HONEST_ASSESSMENT.md, FINAL_RIGOROUS_AUDIT_REPORT.md
IMPLEMENTATION_PLAN_ALL_BLOCKERS.md, IMPLEMENTATION_SUMMARY.md
PHASE1_COMPLETION_REPORT.md
TODO_FIXES.md, ULTIMATE_IMPLEMENTATION_PLAN.md
findings.md, progress.md, task_plan.md
session-ses_3de9.md
kilo_code_task_feb-2-2026_9-05-46-am.md
🚀 ACHIEVING 100 TOKENS_SECOND_*.md (and .pdf)
```

**INPUT:** Cluttered root directory
**OUTPUT:** Clean root with only essential files
**VERIFY:** `ls *.md` at root shows ≤6 files

### C.5 — Test Coverage Gap Analysis

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| C.5.1 | Analyze coverage report from B.4 — identify uncovered modules | Phase B.4 | 15 min | LOW |
| C.5.2 | Write tests for any module with <50% coverage (critical modules only) | C.5.1 | 45 min | MEDIUM |
| C.5.3 | Verify all 13 optimization modules have tests | `tests/` for each optimization | 10 min | LOW |

**INPUT:** Coverage report
**OUTPUT:** Critical gaps filled, ≥80% overall coverage
**VERIFY:** `pytest --cov=nexus --cov-fail-under=80`

### C.COMMIT

```bash
git add -A
git commit -m "docs: production audit, documentation update, root cleanup for v6.1.0"
```

---

## PHASE D: NVComp-ZSTD Integration Evaluation

**Goal:** Evaluate whether GPU-accelerated compression adds value to the existing compression pipeline.
**Estimated Time:** 1-2 hours
**Risk Level:** LOW (evaluation only, integration optional)
**Dependencies:** Phase C complete

### Current State

The project already has ZSTD-concept compression in `src/nexus/optimizations/compression_optimized.py`:
- ZSTD Level 22 configuration
- Quantization-aware compression (3-4× ratio)
- `QuantizedTensor` with block-wise quantization
- Delta encoding and sparsity pruning

No separate "Custom-NVComp-with-ZSTD" repository was found. The `Compression experiment` directory in the parent workspace is empty.

### D.1 — Evaluate Existing Implementation

| ID | Task | Files Affected | Est. | Risk |
|----|------|---------------|------|------|
| D.1.1 | Audit `compression_optimized.py` — verify ZSTD integration is complete (not stub) | `src/nexus/optimizations/compression_optimized.py` | 15 min | LOW |
| D.1.2 | Audit `async_decompression.py` — verify it works with compression_optimized | `src/nexus/optimizations/async_decompression.py` | 15 min | LOW |
| D.1.3 | Check if `pyzstd` or `zstandard` library is used or if ZSTD is a custom implementation | `compression_optimized.py`, `pyproject.toml` | 10 min | LOW |

### D.2 — GPU-Accelerated nvCOMP Assessment

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| D.2.1 | Research: Does nvCOMP-ZSTD provide meaningful speedup for SLI layer loading? | D.1 | 20 min | LOW |
| D.2.2 | Decision gate: Is nvCOMP integration worth the NVIDIA-only dependency? | D.2.1 | 5 min | LOW |
| D.2.3 | If YES → implement nvCOMP wrapper with fallback to CPU ZSTD | Decision | 60 min | MEDIUM |
| D.2.4 | If NO → document decision in `docs/IO_OPTIMIZATION.md` | Decision | 10 min | LOW |

**Decision Criteria:**
- nvCOMP requires CUDA toolkit and NVIDIA GPU → limits portability
- SLI is already I/O-bound at SSD level, not CPU decompression
- If decompression < 5% of total layer load time → NOT worth adding dependency
- If decompression > 20% of total layer load time → WORTH integrating

### D.3 — Validation (if integration happens)

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| D.3.1 | Run compression benchmarks before/after | D.2.3 | 15 min | LOW |
| D.3.2 | Verify all existing tests still pass | D.2.3 | 10 min | LOW |
| D.3.3 | Add nvCOMP-specific tests | D.2.3 | 20 min | LOW |

### D.COMMIT

```bash
git add -A
git commit -m "feat(compression): evaluate/integrate NVComp-ZSTD for layer decompression"
```

---

## PHASE E: Final Release

**Goal:** Clean merge, tag v6.1.0, push to remote.
**Estimated Time:** 30 minutes
**Risk Level:** LOW (all validation done in prior phases)
**Dependencies:** Phases B, C, D all complete

### E.1 — Pre-Release Checklist

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| E.1.1 | Commit all 10 currently uncommitted files | None | 5 min | LOW |
| E.1.2 | Check `audit/bootstrap` branch — merge if it has unmerged work | E.1.1 | 10 min | MEDIUM |
| E.1.3 | Verify `git status` is clean | E.1.2 | 1 min | LOW |

**Currently uncommitted files (10):**
```
pyproject.toml
src/nexus/core/towers/loader.py
src/nexus/data/streaming_trainer.py
src/nexus/data/universal_loader.py
src/nexus/models/sli/layer_cache.py
src/nexus/monitoring/collectors.py
src/nexus/multimodal/distillation.py
src/nexus/multimodal/scripts/mm_download_unified.py
src/nexus/utils/debug_dataset.py
src/nexus/utils/metrics.py
```

### E.2 — Final Validation

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| E.2.1 | Run full test suite one final time: `pytest tests/ --timeout=30 -q` | E.1 | 10 min | LOW |
| E.2.2 | Verify `pip install -e .` still works | E.1 | 3 min | LOW |
| E.2.3 | Verify `python3 -c "import nexus; print(nexus.__version__)"` prints `6.1.0` | E.2.2 | 1 min | LOW |

### E.3 — Tag and Release

| ID | Task | Dependencies | Est. | Risk |
|----|------|-------------|------|------|
| E.3.1 | Create annotated tag: `git tag -a v6.1.0 -m "Release Nexus v6.1.0 — Universal Modular AI Knowledge Distillation Platform"` | E.2 | 2 min | LOW |
| E.3.2 | Push commits: `git push origin main` | E.3.1 | 2 min | LOW |
| E.3.3 | Push tag: `git push origin v6.1.0` | E.3.2 | 2 min | LOW |

### E.COMMIT

Final release commit (before tagging):
```bash
git add -A
git commit -m "release: Nexus v6.1.0 — Universal Modular AI Knowledge Distillation Platform"
```

---

## Risk Register

| # | Risk | Severity | Likelihood | Impact | Mitigation |
|---|------|----------|-----------|--------|------------|
| R1 | Hanging tests block Phase B entirely | **CRITICAL** | **HIGH** | Cannot measure pass rate | Install pytest-timeout FIRST, set 30s global default |
| R2 | Import fixes for 9 test files cause cascade of new failures | HIGH | MEDIUM | More tests break | Fix one file at a time, run `--co` after each |
| R3 | test_router.py metaclass conflict is deep architectural issue | HIGH | MEDIUM | May require source refactor | Isolate test, mock at higher level, or skip with justification |
| R4 | Memory pressure running full 3,813 test suite | HIGH | HIGH | OOM kills, false hangs | Run by directory, 30s timeout, kill between runs |
| R5 | `optimization/` vs `optimizations/` naming confusion | MEDIUM | LOW | Import errors | Document both, consider consolidating in Phase C |
| R6 | Circular imports surface during smoke testing | MEDIUM | MEDIUM | Package unusable | Use lazy imports, fix dependency order |
| R7 | `audit/bootstrap` branch has conflicts with `main` | MEDIUM | LOW | Merge issues | Review diff before merging, use `--no-ff` |
| R8 | Coverage target of 80% may be unrealistic for 267-file codebase | MEDIUM | MEDIUM | Phase C.5 takes much longer | Focus on critical modules only (core, models, security) |
| R9 | NVComp adds heavy NVIDIA dependency | LOW | LOW | Limits portability | Make it optional with CPU fallback |

---

## Time Estimates

| Phase | Optimistic | Realistic | Pessimistic | Parallel? |
|-------|-----------|-----------|------------|-----------|
| **B: Test Suite** | 3 hr | 5 hr | 8 hr | No (sequential by dir) |
| **C: Audit & Production** | 2 hr | 3.5 hr | 5 hr | Partially (C.3 parallel with C.4) |
| **D: NVComp-ZSTD** | 30 min | 1.5 hr | 3 hr | Depends on decision |
| **E: Final Release** | 20 min | 30 min | 1 hr | No |
| **TOTAL** | **~6 hr** | **~10.5 hr** | **~17 hr** | |

**Critical Path:** B.0 → B.1 → B.2 → B.3 → B.4 → C → D → E

Phase C can partially overlap with late Phase B (C.3.1, C.4 are independent of test results).

---

## Features Documented vs. Implemented

Based on docs and README analysis, here's the feature implementation status:

| Feature (from docs/README) | Status | Evidence |
|---------------------------|--------|----------|
| Universal SLI (16 arch families) | ✅ Implemented | `src/nexus/models/sli/` — 61 files |
| Activation Anchoring | ✅ Implemented | Referenced in core modules |
| Sparse Intent Router | ✅ Implemented | `src/nexus/core/` |
| NVFP4 Quantization | ✅ Implemented | `nvfp4_loader.py` |
| QAD Distillation | ✅ Implemented | Training modules |
| Nested Learning | ✅ Implemented | `nested_learning.py` — 545 lines |
| 8 Optimization Solutions | ✅ Implemented | 13 files in `optimizations/` |
| Multimodal Training | ✅ Implemented | 24 files in `multimodal/` |
| Video Generation | ✅ Implemented | Diffusers integration in multimodal |
| Text-to-Speech | ✅ Implemented | 4 files in `voice_engine/` |
| Multi-Agent Orchestration | ✅ Implemented | Training script #20 |
| DPO Training | ✅ Implemented | `training/dpo_training.py` |
| ORPO Training | ✅ Implemented | `training/orpo_training.py` |
| Security (JWT, Rate Limiting) | ✅ Implemented | 3 files in `security/` |
| Monitoring (Prometheus) | ✅ Implemented | 2 files in `monitoring/` |
| KV Cache Optimization | ✅ Implemented | `optimization/kv_cache.py` |
| FastAPI Explainer API | ✅ Implemented | `api/explainer_api.py` |
| Unified CLI | ✅ Implemented | `cli/nexus_cli.py` |
| GPU-accelerated nvCOMP | ⚠️ Conceptual only | ZSTD config exists, no nvCOMP GPU library |
| "135 models supported" claim | ⚠️ Overstated | 16 families, ~60-70 variants realistic |
| 100+ tok/s performance | ❌ Unrealistic for SLI | 3-8 tok/s with SLI is realistic |
| Distributed training | ❌ Not implemented | Referenced but no implementation |
| Automated hyperparameter tuning | ❌ Not implemented | Referenced in assessment |

---

## Execution Checklist

```
PHASE B: FIX TEST SUITE
  [ ] B.0.1  pip install pytest-mock pytest-timeout
  [ ] B.0.2  Add to pyproject.toml dev dependencies
  [ ] B.0.3  Add timeout=30 to pytest.ini
  [ ] B.0.4  Verify collection with new plugins
  [ ] B.1.1  Fix test_pipeline_orchestrator.py imports
  [ ] B.1.2  Fix test_nested_learning.py imports
  [ ] B.1.3  Fix 5 root-level test file imports
  [ ] B.1.4  Fix 2 nexus_final test imports
  [ ] B.1.5  Fix test_router.py metaclass conflict
  [ ] B.1.6  Verify 0 collection errors
  [ ] B.2.1  Identify hanging tests in unit/
  [ ] B.2.2  Fix/skip hanging tests
  [ ] B.2.3  Identify hanging tests in integration/
  [ ] B.2.4  Identify hanging tests in e2e/
  [ ] B.2.5  Add conftest.py with global timeout
  [ ] B.2.6  Mark slow tests appropriately
  [ ] B.3.1  Run & fix tests/unit/
  [ ] B.3.2  Run & fix tests/security/
  [ ] B.3.3  Run & fix tests/integration/
  [ ] B.3.4  Run & fix tests/voice/
  [ ] B.3.5  Run & fix tests/e2e/
  [ ] B.3.6  Run & fix tests/multimodal/
  [ ] B.3.7  Run & fix tests/chaos/
  [ ] B.3.8  Run & fix tests/edge_cases/
  [ ] B.3.9  Run & fix tests/regression/
  [ ] B.3.10 Run & fix tests/benchmarks/
  [ ] B.3.11 Run & fix tests/load/
  [ ] B.3.12 Run & fix tests/nexus_final/
  [ ] B.3.13 Run & fix tests/unit_streaming/
  [ ] B.3.14 Run & fix root-level tests
  [ ] B.4.1  Full suite with coverage
  [ ] B.4.2  Document skipped tests
  [ ] B.4.3  Verify ≥95% pass rate
  [ ] B.COMMIT

PHASE C: AUDIT & PRODUCTION READINESS
  [ ] C.1.1-4 Source cleanliness (ALREADY DONE ✅)
  [ ] C.2.1  Verify all 20 packages import cleanly
  [ ] C.2.2  Check for circular imports
  [ ] C.2.3  Audit optimization/ vs optimizations/
  [ ] C.2.4  pip install -e . smoke test
  [ ] C.2.5  Update compatibility matrix
  [ ] C.3.1  Fix import paths in docs
  [ ] C.3.2  Remove README duplicate content
  [ ] C.3.3  Verify README code examples
  [ ] C.3.4  Update TEST_SUITE.md with counts
  [ ] C.3.5  Update CHANGELOG.md
  [ ] C.3.6  Update API_REFERENCE.md
  [ ] C.3.7  Verify realistic performance claims
  [ ] C.4.1  Move audit .md files to docs/archive/
  [ ] C.4.2  Clean root to essential files only
  [ ] C.4.3  Remove __pycache__ directories
  [ ] C.5.1  Analyze coverage gaps
  [ ] C.5.2  Write tests for critical uncovered modules
  [ ] C.5.3  Verify optimization module test coverage
  [ ] C.COMMIT

PHASE D: NVCOMP-ZSTD EVALUATION
  [ ] D.1.1  Audit compression_optimized.py completeness
  [ ] D.1.2  Audit async_decompression.py integration
  [ ] D.1.3  Check ZSTD library usage
  [ ] D.2.1  Research nvCOMP speedup for SLI
  [ ] D.2.2  Decision gate: integrate or skip
  [ ] D.2.3  (If YES) Implement nvCOMP wrapper
  [ ] D.2.4  (If NO) Document decision
  [ ] D.3.1  Run benchmarks (if integrated)
  [ ] D.3.2  Verify existing tests pass (if integrated)
  [ ] D.3.3  Add nvCOMP tests (if integrated)
  [ ] D.COMMIT

PHASE E: FINAL RELEASE
  [ ] E.1.1  Commit 10 uncommitted files
  [ ] E.1.2  Evaluate audit/bootstrap branch merge
  [ ] E.1.3  Verify clean git status
  [ ] E.2.1  Final full test run
  [ ] E.2.2  Verify pip install works
  [ ] E.2.3  Verify version string
  [ ] E.3.1  Create v6.1.0 tag
  [ ] E.3.2  Push commits to origin
  [ ] E.3.3  Push tag to origin
```

---

## Commands Reference

```bash
# Phase B — Install dependencies
pip install pytest-mock pytest-timeout

# Phase B — Collection check
python3 -m pytest tests/ --co -q 2>&1 | grep ERROR

# Phase B — Run by directory with timeout
python3 -m pytest tests/unit/ -v --timeout=30 --tb=short -x 2>&1 | head -100

# Phase B — Coverage
python3 -m pytest tests/ --cov=nexus --cov-report=html --timeout=30 -q

# Phase C — Find stale imports in docs
rg "from src\." docs/ --type md

# Phase C — Archive root files
mkdir -p docs/archive && mv ACTION_PLAN*.md AUDIT*.md FINAL*.md IMPLEMENTATION*.md docs/archive/

# Phase D — Check compression module
python3 -c "from nexus.optimizations.compression_optimized import CompressionConfig; print(CompressionConfig())"

# Phase E — Release
git add -A && git commit -m "release: Nexus v6.1.0"
git tag -a v6.1.0 -m "Release Nexus v6.1.0 — Universal Modular AI Knowledge Distillation Platform"
git push origin main && git push origin v6.1.0
```
