# Pytest CLI Reference

Complete reference for pytest command-line options in the Nexus Model test suite.

---

## Quick Start

```bash
# Activate conda environment
conda activate nexus

# Run all unit tests (fast)
python -m pytest tests/unit/ -v

# Run full test suite including slow tests
python -m pytest tests/ --full-tests -v

# Run benchmarks
python -m pytest tests/ --full-benchmarks -v
```

---

## Custom CLI Options

The following custom options are defined in `tests/conftest.py`:

### `--full-tests` / `-F`

**Description:** Run ALL tests including slow integration and E2E tests that are normally skipped.

**Usage:**

```bash
pytest tests/ --full-tests
pytest tests/ -F
```

**Default behavior:** Without this flag, tests marked with `@pytest.mark.slow` are skipped.

**What it runs:**

- All unit tests (fast, ~5s)
- All integration tests with real model loading (~30s)
- All E2E pipeline tests (~60s)

**Models used:**

- `Qwen2.5-0.5B` at `/mnt/e/data/models/Qwen2.5-0.5B` (text model for integration tests)
- `Qwen2.5-Omni-7B-GPTQ-Int4` at `/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4` (Omni model for specific tests)

---

### `--full-benchmarks` / `-G`

**Description:** Run ALL benchmarks (Global benchmark suite).

**Usage:**

```bash
pytest tests/ --full-benchmarks
pytest tests/ -G
```

**Default behavior:** Without this flag, tests marked with `@pytest.mark.benchmark` are skipped.

**What it runs:**

- Generation benchmarks (tokens/sec, latency)
- Perplexity benchmarks (accuracy metrics)
- Memory profiling benchmarks

**Models used:**

- `Qwen2.5-0.5B` for fast benchmarks
- Prompts sampled from `ALL_DATASETS` (real dataset samples)

**Configuration:**

- Warmup runs: 2
- Benchmark runs: 5
- Max new tokens: 100
- Batch sizes: [1, 2, 4]

**Output:**

- Results exported to `results/benchmark_metrics.csv`
- Includes: tokens/sec, latency_ms, GPU memory, perplexity

---

### `--test` / `-T`

**Description:** Filter for specific tests by name (comma-separated patterns).

**Usage:**

```bash
# Run single test
pytest tests/ --test test_model_loads

# Run multiple tests
pytest tests/ -T "test_model_loads,test_tokenizer"

# Pattern matching
pytest tests/ --test "cot,reasoning"
```

**Default:** None (runs all collected tests)

**Behavior:** Any test whose nodeid contains the pattern will be selected.

---

### `--benchmark` / `-B`

**Description:** Filter for specific benchmarks by name (comma-separated patterns).

**Usage:**

```bash
# Run specific benchmark
pytest tests/ --benchmark generation

# Run multiple benchmarks
pytest tests/ -B "generation,perplexity"
```

**Default:** None (benchmarks are skipped unless `--full-benchmarks` is set)

**Note:** This option only selects tests that have the `@pytest.mark.benchmark` marker.

---

### `--no-skip` / `-N`

**Description:** Force run ALL tests and benchmarks, including those that would normally be skipped due to missing models, files, or `@pytest.mark.slow` markers.

**Usage:**

```bash
# Run everything with no skips
pytest tests/ --no-skip
pytest tests/ -N

# Equivalent to --full-tests + --full-benchmarks + override in-test skips
pytest tests/ --no-skip -v
```

**Default:** False (tests may be skipped based on conditions)

**Behavior:**

- Implies `--full-tests` and `--full-benchmarks`
- Tests that call `pytest.skip()` due to missing models will still run (and may fail if resources unavailable)
- Use for **100% coverage testing** with all real models available

**When to use:**

- Final validation before deployment
- Ensure all tests execute with real models
- CI/CD pipelines with full model access

---

## Standard Pytest Options

These are built-in pytest options commonly used with this project:

| Option | Description | Example |
|--------|-------------|---------|
| `-v` | Verbose output | `pytest -v` |
| `-vv` | Very verbose | `pytest -vv` |
| `-s` | Show print statements | `pytest -s` |
| `-x` | Stop on first failure | `pytest -x` |
| `--tb=short` | Short traceback | `pytest --tb=short` |
| `--tb=long` | Full traceback | `pytest --tb=long` |
| `-k` | Keyword filter | `pytest -k "model"` |
| `-m` | Marker filter | `pytest -m "not slow"` |
| `--co` | Collect only (list tests) | `pytest --co` |
| `-n auto` | Parallel execution | `pytest -n auto` |
| `--cov=src` | Coverage report | `pytest --cov=src` |

---

## Test Markers

Tests can be filtered by markers with `-m`:

| Marker | Description | Command |
|--------|-------------|---------|
| `slow` | Long-running tests | `pytest -m "slow"` |
| `gpu` | GPU-required tests | `pytest -m "gpu"` |
| `real_model` | Uses real model files | `pytest -m "real_model"` |
| `integration` | Integration tests | `pytest -m "integration"` |
| `e2e` | End-to-end tests | `pytest -m "e2e"` |
| `omni` | Omni model specific | `pytest -m "omni"` |
| `benchmark` | Benchmark tests | `pytest -m "benchmark"` |

**Exclude markers:**

```bash
pytest tests/ -m "not slow"        # Skip slow tests
pytest tests/ -m "not gpu"         # Skip GPU tests
pytest tests/ -m "not benchmark"   # Skip benchmarks
```

---

## Output Locations

Tests automatically export metrics to CSV files:

| File | Description |
|------|-------------|
| `results/test_details.csv` | Per-test timing, memory, outcome |
| `results/training_metrics.csv` | Training run summaries |
| `results/benchmark_metrics.csv` | Benchmark results |
| `results/validation_metrics.csv` | Validation session summaries |

---

## Example Commands

### Development (Quick)

```bash
# Just unit tests
pytest tests/unit/ -v

# Specific file
pytest tests/unit/test_omni_loader.py -v

# Specific test
pytest tests/unit/test_omni_loader.py::TestOmniModelDetection::test_import_loader -v
```

### Full Validation

```bash
# Full test suite with real models
pytest tests/ --full-tests -v --tb=short

# Full suite with coverage
pytest tests/ --full-tests --cov=src --cov-report=html
```

### Benchmarking

```bash
# Run all benchmarks
pytest tests/ --full-benchmarks -v

# Run specific benchmark category
pytest tests/ -B generation -v
```

### CI/CD Pipeline

```bash
# Fast check (no slow tests)
pytest tests/unit/ -v --tb=short

# Full validation before merge
pytest tests/ --full-tests --full-benchmarks -v
```

---

## Extending pytest --help

To see all available options:

```bash
conda run -n nexus python -m pytest --help
```

Custom options appear in the "custom options" section:

```
custom options:
  --full-tests, -F      Run entire test suite (including slow integration/e2e)
  --full-benchmarks, -G Run all benchmarks (Global)
  --test=TEST, -T TEST  Filter for specific tests (comma-separated)
  --benchmark=BENCHMARK, -B BENCHMARK
                        Filter for specific benchmarks (comma-separated)
```

---

## Related Files

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures and CLI options |
| `pytest.ini` | Pytest configuration |
| `src/metrics_tracker.py` | CSV export and metrics |
| `src/benchmarks/benchmark_runner.py` | Benchmark implementation |
