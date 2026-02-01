# Nexus Testing Guide

This guide explains how to run the Nexus test suite effectively, categorizing tests by their requirements and providing options for different environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
  - [Using the Test Runner Script](#using-the-test-runner-script)
  - [Using pytest Directly](#using-pytest-directly)
- [Test Requirements by Category](#test-requirements-by-category)
- [Running on a Laptop](#running-on-a-laptop)
- [Running Full Test Suite](#running-full-test-suite)
- [CI/CD Integration](#cicd-integration)

## Quick Start

```bash
# Run all default tests (recommended for development)
python scripts/run_tests.py

# Run only unit tests (fastest, no external dependencies)
python scripts/run_tests.py --unit-only

# Run with verbose output
python scripts/run_tests.py -v
```

## Test Categories

Tests are organized into categories based on their requirements:

| Category | Marker | Description | Default |
|----------|--------|-------------|---------|
| **Real Model** | `real_model` | Downloads actual models from HuggingFace | Skipped |
| **Distributed** | `distributed` | Requires torch.distributed or mpi4py | Skipped |
| **GPU** | `gpu` | Requires CUDA-capable GPU | Skipped |
| **Slow** | `slow` | Tests taking >30 seconds | Skipped |
| **Integration** | `integration` | Tests multiple components together | Included |
| **E2E** | `e2e` | End-to-end pipeline tests | Skipped |
| **Benchmark** | `benchmark` | Performance benchmark tests | Skipped |
| **Chaos** | `chaos` | Fault injection and chaos tests | Skipped |

### Real Model Tests

Tests marked with `@pytest.mark.real_model` require downloading and loading actual model files from HuggingFace Hub.

**Examples:**

- `tests/integration/test_end_to_end_real_models.py` - Full pipeline tests with real models
- Tests using `conftest.py` fixtures like `real_text_model` when `--use-real-models` is passed

**Requirements:**

- Internet connection
- Sufficient disk space (~2-10GB depending on model)
- Optional: GPU for faster execution

### Distributed Tests

Tests marked with `@pytest.mark.distributed` require distributed training setup.

**Examples:**

- `tests/unit/test_orchestration_scripts_3.py` - Distributed training scripts
- `tests/unit/test_ring_attention.py` - Multi-GPU ring attention (tests world_size > 1 scenarios)

**Requirements:**

- Multiple GPUs OR
- MPI environment (mpi4py) OR
- torch.distributed setup

### GPU Tests

Tests marked with `@pytest.mark.gpu` require CUDA-capable hardware.

**Examples:**

- `tests/unit_streaming/test_streaming_trainer.py`
- `tests/integration/test_multimodal_encoders.py`

**Requirements:**

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- torch with CUDA support

### Slow Tests

Tests marked with `@pytest.mark.slow` take significant time to run (>30 seconds).

**Examples:**

- `tests/integration/test_load_performance.py`
- `tests/integration/test_end_to_end_real_models.py`

### Benchmark Tests

Performance benchmarks located in `benchmarks/` directory.

**Examples:**

- `benchmarks/test_omni_loader_benchmark.py`
- `benchmarks/test_multimodal_architect_benchmark.py`

### Chaos Tests

Fault injection tests located in `tests/chaos/`.

**Examples:**

- `tests/chaos/test_gpu_failure.py`
- `tests/chaos/test_memory_pressure.py`

## Running Tests

### Using the Test Runner Script

The recommended way to run tests is using the provided test runner script:

```bash
# Default: Run all tests except real models, distributed, GPU, slow
python scripts/run_tests.py

# Include real model tests
python scripts/run_tests.py --real-models

# Include distributed tests
python scripts/run_tests.py --distributed

# Include GPU tests
python scripts/run_tests.py --gpu

# Include slow tests
python scripts/run_tests.py --slow

# Run everything
python scripts/run_tests.py --all

# Run only unit tests (excludes integration, e2e)
python scripts/run_tests.py --unit-only

# Run only integration and e2e tests
python scripts/run_tests.py --integration-only

# Run only benchmarks
python scripts/run_tests.py --benchmark-only

# Generate coverage report
python scripts/run_tests.py --coverage

# Generate JSON report
python scripts/run_tests.py --report

# Run specific test file
python scripts/run_tests.py tests/unit/test_example.py -v

# List tests without running
python scripts/run_tests.py --collect-only
```

### Using pytest Directly

You can also run tests directly with pytest:

```bash
# Run all tests (excluding marked ones)
pytest tests/

# Skip real model tests explicitly
pytest tests/ -m "not real_model"

# Skip multiple categories
pytest tests/ -m "not (real_model or distributed or gpu or slow)"

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/ -m "not real_model"

# Run real model tests with small models
pytest tests/ -m "real_model" --use-real-models --small-model

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Requirements by Category

### Files Requiring Real Models

The following test files use real model loading:

1. **Direct real model tests:**
   - `tests/integration/test_end_to_end_real_models.py` - Uses Qwen/Qwen2.5-0.5B-Instruct
   - `tests/integration/test_e2e_pipeline.py` - Downloads small models

2. **Tests with mocked models (safe to run):**
   - Most `tests/unit/test_*.py` files use `@patch` to mock transformers
   - `tests/integration/test_model_loading.py` - Uses mocking

3. **Conditional real models (via conftest.py):**
   - Tests using `real_text_model` fixture only load real models with `--use-real-models` flag
   - Default behavior uses mocks

### Files Requiring Distributed Setup

1. **Direct distributed tests:**
   - `tests/unit/test_orchestration_scripts_3.py` - Tests distributed training
   - `tests/unit/test_ring_attention.py` - Tests multi-GPU ring attention

### Files Requiring GPU

1. **GPU-specific tests:**
   - `tests/unit_streaming/test_streaming_trainer.py` - GPU training tests
   - `tests/integration/test_multimodal_encoders.py` - Encoder loading on GPU
   - `tests/integration/test_load_performance.py` - GPU memory tests

## Running on a Laptop

For laptop/development environments without GPU or large model files:

```bash
# Recommended for laptops - unit tests only (fast, no external deps)
python scripts/run_tests.py --unit-only

# Or include integration tests that don't need real models
python scripts/run_tests.py

# Skip all resource-intensive tests explicitly
pytest tests/ -m "not (real_model or distributed or gpu or slow or benchmark or chaos)"
```

### Laptop-Friendly Test Categories

These test categories are safe to run on laptops:

✅ **Safe:**

- `tests/unit/` - Most unit tests use mocking
- `tests/integration/test_model_loading.py` - Uses mocking
- `tests/integration/test_pipeline_mini.py` - Lightweight

❌ **Avoid on Laptops:**

- Tests with `@pytest.mark.real_model` - Requires model downloads
- Tests with `@pytest.mark.gpu` - Requires CUDA
- Tests with `@pytest.mark.slow` - Long running
- `benchmarks/` - Performance tests
- `tests/chaos/` - Resource-intensive fault injection

## Running Full Test Suite

To run the complete test suite (requires appropriate hardware):

```bash
# Run everything
python scripts/run_tests.py --all

# With real models (small model variant)
python scripts/run_tests.py --all --small-model

# With coverage report
python scripts/run_tests.py --all --coverage --report

# Direct pytest equivalent
pytest tests/ --use-real-models --full-tests --cov=src
```

### Full Suite Requirements

- **CPU:** 8+ cores recommended
- **RAM:** 32GB+ recommended
- **GPU:** NVIDIA GPU with 16GB+ VRAM (optional but recommended)
- **Disk:** 50GB+ free space for models
- **Network:** Stable internet for model downloads
- **Time:** 1-2 hours for complete suite

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements/test.txt
      - name: Run unit tests
        run: |
          python scripts/run_tests.py --unit-only --junit-xml=test-results.xml
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results.xml

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements/test.txt
      - name: Run integration tests
        run: |
          python scripts/run_tests.py --integration-only

  # Run on self-hosted runner with GPU
  gpu-tests:
    runs-on: self-hosted-gpu
    if: contains(github.event.pull_request.labels.*.name, 'gpu-test')
    steps:
      - uses: actions/checkout@v3
      - name: Run GPU tests
        run: |
          python scripts/run_tests.py --gpu
```

### GitLab CI Example

```yaml
stages:
  - test

unit_tests:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements/test.txt
    - python scripts/run_tests.py --unit-only --junit-xml=junit.xml
  artifacts:
    reports:
      junit: junit.xml

integration_tests:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements/test.txt
    - python scripts/run_tests.py --integration-only

gpu_tests:
  stage: test
  tags:
    - gpu
  script:
    - pip install -r requirements/test.txt
    - python scripts/run_tests.py --gpu
  rules:
    - if: $CI_MERGE_REQUEST_LABELS =~ /gpu-test/
```

## Troubleshooting

### Tests Skipped Unexpectedly

If tests are being skipped when they shouldn't be:

```bash
# List all tests with their markers
python scripts/run_tests.py --collect-only

# Check what markers are being skipped
python scripts/run_tests.py -v --collect-only 2>&1 | grep "marker"
```

### Real Model Tests Failing

If real model tests fail to download:

```bash
# Check internet connectivity
# Set HF cache directory if needed
export HF_HOME=/path/to/large/disk

# Use small model variant
python scripts/run_tests.py --real-models --small-model
```

### Out of Memory

If tests fail with OOM:

```bash
# Skip GPU and real model tests
python scripts/run_tests.py --unit-only

# Run tests sequentially (no parallelization)
pytest tests/unit/ -p no:xdist -n0
```

## Test Markers Reference

All available pytest markers:

```ini
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU/CUDA
    real_model: marks tests as requiring real model files (downloads from HuggingFace)
    distributed: marks tests as requiring distributed setup (torch.distributed, mpi4py)
    cluster: marks tests as requiring multi-node cluster setup
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    omni: marks tests specific to Omni model
    benchmark: marks tests as performance benchmarks
    chaos: marks tests as chaos engineering/fault injection tests
    unit: marks tests as unit tests
```

## Contributing Tests

When adding new tests:

1. **Use appropriate markers:**

   ```python
   @pytest.mark.gpu
   def test_cuda_operation():
       # GPU-specific test
       pass
   ```

2. **Mock external dependencies in unit tests:**

   ```python
   @patch("transformers.AutoModel.from_pretrained")
   def test_model_loading(mock_from_pretrained):
       # Use mocks, not real models
       pass
   ```

3. **Use fixtures from conftest.py for real models:**

   ```python
   def test_with_real_model(real_text_model):
       # Only runs with --use-real-models
       pass
   ```

4. **Skip gracefully when resources unavailable:**

   ```python
   def test_gpu_feature():
       if not torch.cuda.is_available():
           pytest.skip("CUDA not available")
       # GPU test code
   ```
