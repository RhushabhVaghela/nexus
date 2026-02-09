# Nexus v6.1.0 — Execution Plan (Steps 5c → 8)

**Created:** 2026-02-09
**Status:** READY FOR EXECUTION
**Environment:** WSL2, 16GB RAM, Python 3.9+

---

## Executive Summary

After completing Step 5a (optional import guards), the project has a **critical blocker** that must be addressed before any other step can succeed:

> **BLOCKER: Package structure mismatch.** `pyproject.toml` expects packages under `src/nexus/` (via `package-dir = {"": "src"}` + `include = ["nexus*"]`), but **`src/nexus/` does not exist.** Code lives directly at `src/core/`, `src/models/`, etc. A `pip install .` would install an **empty package.**

This changes the optimal execution order. The plan below accounts for this.

---

## Project State Snapshot

| Metric | Value |
|--------|-------|
| Source files (`.py`, excl `__init__`) | 267 |
| `__init__.py` files under `src/` | 39 |
| Test files | 248 |
| Test directories | 12 (unit, integration, e2e, voice, security, benchmarks, chaos, edge_cases, load, multimodal, regression, nexus_final) |
| Files using `from src.X` imports (source) | 91 |
| Files using `from src.X` imports (tests) | 187 |
| Docs files under `docs/` | 80+ |
| Root-level audit/plan `.md` files | ~30 (cleanup needed) |

---

## Revised Execution Order (with rationale)

```
Original:  5c → 6 → 7 → 8
Revised:   7A → 5c → 6 → 7B → 8
             │               │
             │               └─ Final packaging validation
             └─ Fix structure FIRST (changes all import paths)
```

**Why reorder?**
- Step 7A (package restructure) changes every import path in the project
- If we run tests (5c) first, we'd fix 187 test files to match the *wrong* structure
- Then Step 7 would break them all again
- Fix structure → fix tests → fix docs → validate packaging → release

---

## Phase A: Package Structure Fix (Step 7A)

**Goal:** Make `pip install -e .` produce a working `nexus` package.

### A.1 — Create `src/nexus/` directory structure
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| A.1.1 | Create `src/nexus/` directory | None | 5 min | LOW |
| A.1.2 | Move all `src/*` subdirs into `src/nexus/` | A.1.1 | 10 min | **CRITICAL** |
| A.1.3 | Move `src/__init__.py` → `src/nexus/__init__.py` | A.1.2 | 2 min | LOW |
| A.1.4 | Create new empty `src/__init__.py` (or remove it — setuptools `src` layout shouldn't have one) | A.1.3 | 2 min | MEDIUM |

### A.2 — Fix all internal import paths
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| A.2.1 | Replace `from src.X` → `from nexus.X` in all 91 source files | A.1.4 | 20 min | **HIGH** — regex must be precise |
| A.2.2 | Replace `from src.X` → `from nexus.X` in all 187 test files | A.1.4 | 20 min | **HIGH** |
| A.2.3 | Replace `import src.X` → `import nexus.X` everywhere | A.1.4 | 10 min | MEDIUM |
| A.2.4 | Update `nexus/__init__.py` shim (root-level) to alias correctly | A.2.1 | 10 min | MEDIUM |
| A.2.5 | Smoke test: `python -c "from nexus.core import ..."` | A.2.4 | 5 min | LOW |

### A.3 — Fix pyproject.toml issues
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| A.3.1 | Verify `[tool.setuptools.packages.find]` now discovers `nexus` | A.1.4 | 5 min | LOW |
| A.3.2 | Fix CLI entry points: `nexus.cli.nexus_cli:main` path | A.2.1 | 5 min | MEDIUM |
| A.3.3 | Fix `setup.py` — remove `setuptools_scm` (no git tags), use static version | None | 10 min | LOW |
| A.3.4 | Create `MANIFEST.in` for sdist | None | 10 min | LOW |
| A.3.5 | Run `pip install -e .` and verify imports work | A.3.2 | 10 min | **HIGH** |

**Phase A Total:** ~2 hours
**Phase A Risk:** **CRITICAL** — This is the highest-risk phase. A mistake here breaks everything.
**Memory Impact:** LOW — these are file operations, not compute.

### A — Risk Mitigation
```bash
# Before starting Phase A:
git stash  # or commit current state
git checkout -b v6.1.0-packaging  # work on a branch
# After Phase A passes smoke tests:
git add -A && git commit -m "refactor: restructure src/ → src/nexus/ for proper pip packaging"
```

---

## Phase B: Test Suite (Step 5c)

**Goal:** All tests pass (or are marked `skip` with justification).

### B.0 — Pre-flight
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| B.0.1 | Run `pytest --co -q` to check collection (no execution) | Phase A done | 5 min | MEDIUM — may timeout |
| B.0.2 | Count collection errors, categorize (import vs syntax vs missing fixture) | B.0.1 | 15 min | LOW |

### B.1 — Fix collection errors (batch by directory)
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| B.1.1 | Fix `tests/unit/` collection errors (~80 files) | B.0.2 | 30 min | **HIGH** |
| B.1.2 | Fix `tests/integration/` collection errors | B.0.2 | 15 min | MEDIUM |
| B.1.3 | Fix `tests/e2e/` collection errors | B.0.2 | 10 min | MEDIUM |
| B.1.4 | Fix `tests/voice/` collection errors | B.0.2 | 10 min | MEDIUM |
| B.1.5 | Fix `tests/security/` collection errors | B.0.2 | 10 min | MEDIUM |
| B.1.6 | Fix remaining dirs (benchmarks, chaos, edge_cases, load, multimodal, regression, nexus_final) | B.0.2 | 20 min | MEDIUM |

> **B.1.1–B.1.6 can run in PARALLEL** (different directories, no shared state)

### B.2 — Run tests and fix failures (sequential, by risk)
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| B.2.1 | Run `tests/unit/` — fix failures | B.1.1 | 45 min | **HIGH** — largest batch |
| B.2.2 | Run `tests/security/` — fix failures | B.1.5 | 15 min | MEDIUM |
| B.2.3 | Run `tests/integration/` — fix failures | B.1.2 | 20 min | MEDIUM |
| B.2.4 | Run `tests/voice/` — fix failures | B.1.4 | 15 min | LOW |
| B.2.5 | Run `tests/e2e/` — fix failures | B.1.3 | 15 min | LOW |
| B.2.6 | Run remaining test dirs | B.1.6 | 20 min | LOW |

> **Memory constraint:** Run ONE test directory at a time. Use `pytest --forked` if individual tests leak memory.
> **Command template:** `python -m pytest tests/unit/ -x -v --timeout=30 2>&1 | head -200`

### B.3 — Coverage & summary
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| B.3.1 | Run full suite with coverage: `pytest --cov=nexus --cov-report=html` | B.2.* | 15 min | LOW |
| B.3.2 | Document skipped tests with justifications | B.3.1 | 10 min | LOW |
| B.3.3 | Commit: "test: fix all test failures for v6.1.0" | B.3.2 | 5 min | LOW |

**Phase B Total:** ~3-4 hours
**Phase B Risk:** **HIGH** — 248 test files, likely many cascading import failures after restructure.
**Memory Impact:** MEDIUM — run test dirs one at a time, kill between runs.

---

## Phase C: Documentation (Step 6)

**Goal:** Professional-grade documentation for a v6.1.0 release.

### C.1 — Module docstrings
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| C.1.1 | Add/update docstrings in all 39 `__init__.py` files under `src/nexus/` | Phase A | 40 min | LOW |
| C.1.2 | Verify all public classes/functions have docstrings (top 20 most-used modules) | C.1.1 | 20 min | LOW |

### C.2 — README.md update
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| C.2.1 | Fix all import examples: `from src.nexus.X` → `from nexus.X` | Phase A | 15 min | LOW |
| C.2.2 | Add Installation section (`pip install nexus` / `pip install -e .`) | Phase A | 10 min | LOW |
| C.2.3 | Update Quick Start with working examples | B.3.1 (need tests passing) | 15 min | LOW |
| C.2.4 | Add v6.1.0 release highlights section | None | 10 min | LOW |

### C.3 — Documentation cleanup
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| C.3.1 | Move ~30 root-level audit/plan `.md` files → `docs/archive/` | None | 10 min | LOW |
| C.3.2 | Update `docs/API_REFERENCE.md` to match current module paths | Phase A | 20 min | LOW |
| C.3.3 | Verify Sphinx build: `cd docs && make html` | C.1.1, C.3.2 | 15 min | MEDIUM |
| C.3.4 | Update CHANGELOG.md with v6.1.0 entries | None | 15 min | LOW |

> **C.1, C.2.1–C.2.2, C.3.1, C.3.4 can run in PARALLEL** (independent files)

### C.4 — Commit
| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| C.4.1 | Commit: "docs: update documentation for v6.1.0 release" | C.3.3 | 5 min | LOW |

**Phase C Total:** ~2 hours
**Phase C Risk:** LOW — documentation doesn't break functionality.
**Memory Impact:** LOW — text editing only.

---

## Phase D: Final Packaging Validation (Step 7B)

**Goal:** `pip install .` produces a working, distributable package.

| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| D.1 | Clean build: `rm -rf build/ dist/ *.egg-info` | Phase A, B, C | 2 min | LOW |
| D.2 | Build sdist: `python -m build --sdist` | D.1 | 5 min | MEDIUM |
| D.3 | Build wheel: `python -m build --wheel` | D.1 | 5 min | MEDIUM |
| D.4 | Install wheel in fresh venv, test imports | D.3 | 10 min | **HIGH** |
| D.5 | Verify CLI entry points: `nexus --help` | D.4 | 5 min | MEDIUM |
| D.6 | Verify `nexus.config/*.yaml` included in wheel | D.4 | 5 min | LOW |
| D.7 | Check package size is reasonable (< 50MB) | D.3 | 2 min | LOW |
| D.8 | Commit: "build: validate packaging for v6.1.0" | D.7 | 5 min | LOW |

**Phase D Total:** ~45 min
**Phase D Risk:** MEDIUM — if Phase A was done correctly, this is validation only.
**Memory Impact:** LOW.

---

## Phase E: Final Commit & Release (Step 8)

| ID | Task | Dependencies | Est. Time | Risk |
|----|------|-------------|-----------|------|
| E.1 | Final `git status` — ensure no uncommitted changes | Phase D | 2 min | LOW |
| E.2 | Run full test suite one last time: `pytest tests/ -q` | Phase D | 10 min | LOW |
| E.3 | Create git tag: `git tag -a v6.1.0 -m "Release v6.1.0"` | E.2 | 2 min | LOW |
| E.4 | Final release commit with clean message | E.3 | 5 min | LOW |
| E.5 | (Optional) Push tag: `git push origin v6.1.0` | E.4 | 2 min | LOW |

**Phase E Total:** ~20 min
**Phase E Risk:** LOW — everything validated by this point.

---

## Dependency Graph

```
Phase A (Structure Fix) ──────────────────────────────────┐
  A.1 (mkdir/mv) → A.2 (fix imports) → A.3 (pyproject)   │
                                                           │
Phase B (Tests) ◄──────────────────────────────────────────┘
  B.0 (collect) → B.1 (fix collection) → B.2 (run/fix) → B.3 (coverage)
       │                   │
       │            [B.1.1-B.1.6 PARALLEL]
       │
Phase C (Docs) ◄── Phase A done (for import paths)
  [C.1, C.2.1-2, C.3.1, C.3.4 PARALLEL]  ───►  C.3.3 (sphinx) → C.4
       │
       │   NOTE: C can START after Phase A, run ALONGSIDE Phase B
       │         Only C.2.3 (working examples) needs Phase B done
       │
Phase D (Package Validation) ◄── Phase A + B + C all done
  D.1 → D.2/D.3 [PARALLEL] → D.4 → D.5/D.6/D.7 [PARALLEL] → D.8
       │
Phase E (Release) ◄── Phase D done
  E.1 → E.2 → E.3 → E.4 → E.5
```

---

## Parallelization Opportunities

| Window | Parallel Tasks | Constraint |
|--------|---------------|------------|
| After Phase A completes | Phase B (tests) + Phase C.1/C.3.1/C.3.4 (docs) | C.2.3 waits for B |
| During B.1 | B.1.1 through B.1.6 (different test dirs) | 16GB RAM — max 2 concurrent |
| During D | D.2 + D.3 (sdist + wheel) | Trivial parallelism |
| During D | D.5 + D.6 + D.7 (validation checks) | Trivial parallelism |

---

## Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Package restructure breaks deep import chains | **CRITICAL** | HIGH | Work on branch. Automated `sed` for bulk renames. Smoke test after each sub-step. |
| Test collection timeout on 16GB RAM | HIGH | MEDIUM | Run by directory, use `--timeout=30`, kill stuck processes |
| Circular imports after restructure | HIGH | MEDIUM | Test import order carefully. Use lazy imports where needed. |
| `setuptools_scm` fails without git tags | MEDIUM | HIGH | Remove `setuptools_scm` from `setup.py`, use static version |
| Sphinx build fails on updated paths | MEDIUM | MEDIUM | Fix `docs/conf.py` sys.path, regenerate API docs |
| Some tests require GPU/large models | LOW | HIGH | Mark with `@pytest.mark.gpu` or `@pytest.mark.slow`, skip gracefully |
| Root-level `nexus/` shim conflicts with installed `nexus` package | MEDIUM | MEDIUM | Remove or update shim after restructure |

---

## Total Time Estimate

| Phase | Estimated | Buffer (1.5x) |
|-------|-----------|----------------|
| A: Package Structure | 2 hr | 3 hr |
| B: Test Suite | 3.5 hr | 5 hr |
| C: Documentation | 2 hr | 3 hr |
| D: Package Validation | 45 min | 1 hr |
| E: Release | 20 min | 30 min |
| **TOTAL** | **~8.5 hr** | **~12.5 hr** |

With parallelization of C alongside B: **~7 hr** (optimistic) to **~10 hr** (realistic).

---

## Execution Checklist (copy-paste ready)

```
PHASE A: PACKAGE STRUCTURE FIX
  [ ] A.1.1  Create src/nexus/ directory
  [ ] A.1.2  Move src/* subdirs → src/nexus/
  [ ] A.1.3  Move src/__init__.py → src/nexus/__init__.py
  [ ] A.1.4  Handle src/__init__.py (remove or make empty)
  [ ] A.2.1  Fix imports in 91 source files (src.X → nexus.X)
  [ ] A.2.2  Fix imports in 187 test files (src.X → nexus.X)
  [ ] A.2.3  Fix `import src.X` statements
  [ ] A.2.4  Update root nexus/ shim
  [ ] A.2.5  Smoke test imports
  [ ] A.3.1  Verify setuptools finds nexus package
  [ ] A.3.2  Fix CLI entry points
  [ ] A.3.3  Fix setup.py (remove scm, static version)
  [ ] A.3.4  Create MANIFEST.in
  [ ] A.3.5  Test pip install -e .
  [ ] A.COMMIT  git commit "refactor: restructure to src/nexus/ layout"

PHASE B: TEST SUITE
  [ ] B.0.1  pytest --co -q (collection check)
  [ ] B.0.2  Categorize collection errors
  [ ] B.1.1  Fix tests/unit/ collection
  [ ] B.1.2  Fix tests/integration/ collection
  [ ] B.1.3  Fix tests/e2e/ collection
  [ ] B.1.4  Fix tests/voice/ collection
  [ ] B.1.5  Fix tests/security/ collection
  [ ] B.1.6  Fix remaining test dirs collection
  [ ] B.2.1  Run & fix tests/unit/
  [ ] B.2.2  Run & fix tests/security/
  [ ] B.2.3  Run & fix tests/integration/
  [ ] B.2.4  Run & fix tests/voice/
  [ ] B.2.5  Run & fix tests/e2e/
  [ ] B.2.6  Run & fix remaining test dirs
  [ ] B.3.1  Full suite with coverage
  [ ] B.3.2  Document skipped tests
  [ ] B.COMMIT  git commit "test: fix all test failures for v6.1.0"

PHASE C: DOCUMENTATION (can start after Phase A, parallel with Phase B)
  [ ] C.1.1  Module docstrings in __init__.py files
  [ ] C.1.2  Public API docstrings (top 20 modules)
  [ ] C.2.1  Fix README import examples
  [ ] C.2.2  Add README installation section
  [ ] C.2.3  Update README quick start (needs Phase B)
  [ ] C.2.4  Add v6.1.0 release highlights
  [ ] C.3.1  Move root audit .md files → docs/archive/
  [ ] C.3.2  Update docs/API_REFERENCE.md
  [ ] C.3.3  Verify Sphinx build
  [ ] C.3.4  Update CHANGELOG.md
  [ ] C.COMMIT  git commit "docs: update documentation for v6.1.0"

PHASE D: FINAL PACKAGING VALIDATION
  [ ] D.1    Clean build artifacts
  [ ] D.2    Build sdist
  [ ] D.3    Build wheel
  [ ] D.4    Install wheel in fresh venv, test imports
  [ ] D.5    Verify CLI entry points
  [ ] D.6    Verify package data included
  [ ] D.7    Check package size
  [ ] D.COMMIT  git commit "build: validate packaging for v6.1.0"

PHASE E: RELEASE
  [ ] E.1    Verify clean git status
  [ ] E.2    Final full test run
  [ ] E.3    Create git tag v6.1.0
  [ ] E.4    Final release commit
  [ ] E.5    Push tag (optional)
```

---

## Commands Reference

```bash
# Phase A — bulk import fix (example)
find src/nexus -name "*.py" -exec sed -i 's/from src\./from nexus./g' {} +
find tests -name "*.py" -exec sed -i 's/from src\./from nexus./g' {} +

# Phase B — test by directory
python -m pytest tests/unit/ -x -v --timeout=30 --tb=short 2>&1 | tee test_unit.log
python -m pytest tests/integration/ -x -v --timeout=30 --tb=short 2>&1 | tee test_integration.log

# Phase B — coverage
python -m pytest tests/ --cov=nexus --cov-report=html --cov-report=term-missing -q

# Phase D — build
pip install build
python -m build
pip install dist/nexus-6.1.0-py3-none-any.whl --force-reinstall
python -c "import nexus; print(nexus.__version__)"
nexus --help

# Phase E — tag
git tag -a v6.1.0 -m "Release Nexus v6.1.0 - Universal Modular AI Knowledge Distillation Platform"
```
