# Contributing to Nexus

Thank you for your interest in contributing to Nexus! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Templates](#issue-templates)
- [Review Process](#review-process)
- [Community](#community)

---

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- CUDA-capable GPU (for development/testing)
- Conda or virtualenv

### Setting Up Development Environment

1. **Fork and clone:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/nexus.git
   cd nexus
   ```

2. **Create virtual environment:**

   ```bash
   conda create -n nexus-dev python=3.10
   conda activate nexus-dev
   ```

3. **Install in editable mode:**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**

   ```bash
   pre-commit install
   ```

### Project Structure

```
nexus/
├── src/                    # Source code
│   ├── nexus/             # Core modules
│   │   ├── optimizations/ # Optimization implementations
│   │   ├── core/          # Core modules
│   │   └── ...
│   ├── multimodal/        # Multimodal components
│   ├── utils/             # Utility functions
│   └── ...
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── benchmarks/       # Performance tests
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── configs/               # Configuration files
```

---

## Optimization Development Guidelines

Nexus includes **8 research-backed optimization solutions** for achieving 100+ tokens/second inference. This section guides contributors on adding new optimizations.

### Architecture Overview

Nexus optimizations target three main LLM inference bottlenecks:

1. **Sequential Dependency** - Layer N must complete before Layer N+1
2. **Decompression Overhead** - Loading compressed weights blocks computation
3. **Forward Pass Time** - Each layer takes significant time to compute

### Adding a New Optimization

#### 1. Research Validation

Before implementation, validate the research:

- [ ] Paper is from reputable venue (NeurIPS, ICML, ICLR, etc.)
- [ ] Claims are backed by reproducible benchmarks
- [ ] Open-source implementation exists (optional but recommended)
- [ ] Performance claims are realistic (validated on consumer hardware)

#### 2. Implementation Structure

Create a new file in `src/nexus/optimizations/`:

```python
# src/nexus/optimizations/your_optimization.py
"""
Your Optimization Name

Research: Paper Title (Year)
Problem: Brief description of the bottleneck
Solution: How this optimization solves it

Expected Performance:
- Speedup: X×
- Memory: +/- X%
- Accuracy: X%
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .base import BaseOptimizer, OptimizationConfig


class YourOptimizerConfig(OptimizationConfig):
    """Configuration for your optimization."""
    
    def __init__(
        self,
        param1: float = 0.5,
        param2: int = 4,
        enabled: bool = True
    ):
        super().__init__(enabled=enabled)
        self.param1 = param1
        self.param2 = param2


class YourOptimizer(BaseOptimizer):
    """
    Your optimization implementation.
    
    Args:
        config: Configuration object
        model: Model to optimize (optional)
    """
    
    def __init__(
        self,
        config: Optional[YourOptimizerConfig] = None,
        model: Optional[nn.Module] = None
    ):
        super().__init__(config or YourOptimizerConfig())
        self.model = model
        
    def optimize(
        self,
        model: nn.Module,
        **kwargs
    ) -> nn.Module:
        """Apply optimization to model.
        
        Args:
            model: Model to optimize
            **kwargs: Additional arguments
            
        Returns:
            Optimized model
        """
        # Implementation here
        return model
    
    def forward(
        self,
        inputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Optimized forward pass.
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
        # Implementation here
        return outputs
    
    def get_metrics(self) -> Dict[str, float]:
        """Return optimization metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "speedup": self.speedup,
            "memory_overhead": self.memory_overhead,
            "accuracy_retention": self.accuracy_retention
        }
```

#### 3. Export in `__init__.py`

Add to `src/nexus/optimizations/__init__.py`:

```python
from .your_optimization import YourOptimizer, YourOptimizerConfig

__all__ = [
    # ... existing exports ...
    "YourOptimizer",
    "YourOptimizerConfig",
]
```

#### 4. Testing Requirements

Create comprehensive tests in `tests/test_optimizations.py`:

```python
class TestYourOptimizer:
    """Test suite for YourOptimizer."""
    
    def test_initialization(self):
        """Test optimizer initializes correctly."""
        from nexus.optimizations import YourOptimizer
        
        optimizer = YourOptimizer()
        assert optimizer is not None
        assert optimizer.config.enabled
    
    def test_optimization(self):
        """Test optimization applies correctly."""
        model = create_test_model()
        optimizer = YourOptimizer()
        
        optimized = optimizer.optimize(model)
        assert optimized is not None
    
    def test_forward_pass(self):
        """Test forward pass produces correct output."""
        optimizer = YourOptimizer()
        inputs = torch.randn(1, 10, 512)
        
        outputs = optimizer.forward(inputs)
        assert outputs.shape == inputs.shape
    
    def test_performance_improvement(self):
        """Verify performance improvement."""
        model = create_test_model()
        optimizer = YourOptimizer()
        
        # Baseline
        start = time.time()
        baseline_output = model(inputs)
        baseline_time = time.time() - start
        
        # Optimized
        optimized = optimizer.optimize(model)
        start = time.time()
        optimized_output = optimized(inputs)
        optimized_time = time.time() - start
        
        # Verify speedup
        speedup = baseline_time / optimized_time
        assert speedup > 1.1, f"Expected >1.1× speedup, got {speedup:.2f}×"
    
    def test_accuracy_retention(self):
        """Verify accuracy is maintained."""
        model = create_test_model()
        optimizer = YourOptimizer()
        inputs = torch.randn(1, 10, 512)
        
        baseline_output = model(inputs)
        optimized_output = optimizer.forward(inputs)
        
        # Cosine similarity > 0.97
        similarity = torch.nn.functional.cosine_similarity(
            baseline_output.flatten(),
            optimized_output.flatten(),
            dim=0
        )
        assert similarity > 0.97, f"Accuracy below threshold: {similarity:.4f}"
```

#### 5. Configuration

Add configuration to `configs/optimization_config.yaml`:

```yaml
your_optimization_config:
  param1: 0.5
  param2: 4
  description: |
    Brief description of what this does
    and expected performance characteristics
```

#### 6. Documentation

Update documentation:

- [ ] Add entry to `docs/OPTIMIZATION_GUIDE.md`
- [ ] Include research references
- [ ] Provide usage examples
- [ ] Document configuration options
- [ ] Add troubleshooting section

#### 7. Benchmarking

Create benchmark in `scripts/benchmark_optimizations.py`:

```python
def benchmark_your_optimizer():
    """Benchmark your optimizer."""
    results = {}
    
    for model_name in ["gpt2", "meta-llama/Llama-3.1-8B"]:
        baseline = benchmark_model(model_name)
        optimized = benchmark_model(model_name, optimizer="your_optimizer")
        
        results[model_name] = {
            "baseline_tokens_per_sec": baseline["tokens_per_sec"],
            "optimized_tokens_per_sec": optimized["tokens_per_sec"],
            "speedup": optimized["tokens_per_sec"] / baseline["tokens_per_sec"],
            "accuracy": compute_accuracy(model_name, optimized["outputs"])
        }
    
    return results
```

### Code Quality Standards for Optimizations

#### Performance Requirements

| Metric | Minimum | Target |
|--------|---------|--------|
| **Speedup** | 1.1× | 1.5×+ |
| **Accuracy** | 95% | 97%+ |
| **Overhead** | <20% | <10% |

#### Code Standards

- **Type Hints**: All functions must have complete type annotations
- **Documentation**: Google-style docstrings for all public methods
- **Error Handling**: Graceful fallbacks when optimization fails
- **Logging**: Structured logging for debugging (`logger.debug()`)
- **Configuration**: YAML-based configuration support

#### Testing Standards

- **Unit Tests**: Minimum 5 test cases per optimization
- **Integration Tests**: Test with real models
- **Performance Tests**: Verify speedup claims
- **Accuracy Tests**: Verify <3% accuracy loss
- **Coverage**: Minimum 80% code coverage

### Review Checklist for Optimization PRs

Before submitting an optimization PR:

- [ ] Research paper reviewed and validated
- [ ] Implementation follows Nexus patterns
- [ ] All tests pass (`pytest tests/test_optimizations.py`)
- [ ] Performance benchmarks included
- [ ] Accuracy validation (<3% loss)
- [ ] Documentation updated
- [ ] Configuration added to YAML
- [ ] No regression in existing tests
- [ ] Code reviewed by 2+ maintainers

### Architecture Guidelines

#### Integration Patterns

1. **Composability**: Optimizations should work together
2. **Configurability**: All parameters exposed via YAML
3. **Observability**: Metrics collection built-in
4. **Fallback**: Graceful degradation on errors

#### Anti-Patterns to Avoid

❌ **Don't**:

- Break composability with other optimizations
- Hardcode parameters
- Skip error handling
- Ignore memory overhead
- Neglect accuracy validation

✅ **Do**:

- Design for composability
- Use configuration objects
- Implement fallback modes
- Monitor resource usage
- Validate accuracy rigorously

---

---

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `hotfix/description` - Critical fixes

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:

```
feat(multimodal): add video processing support

Implement video encoding using SVD with memory-efficient
attention mechanisms.

Closes #123
```

```
fix(training): resolve OOM in distributed training

Add gradient checkpointing to prevent memory overflow
when training on multiple GPUs.

Fixes #456
```

---

## Code Style Guidelines

### Python Style

We use **Black** for formatting and **isort** for import sorting:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
flake8 src/ tests/

# Type checking
mypy src/
```

### Documentation Strings

Use Google-style docstrings:

```python
def process_data(
    data: torch.Tensor,
    batch_size: int = 32,
    normalize: bool = True
) -> torch.Tensor:
    """Process input data for model training.
    
    Args:
        data: Input tensor of shape (N, ...)
        batch_size: Number of samples per batch
        normalize: Whether to normalize the data
        
    Returns:
        Processed tensor ready for training
        
    Raises:
        ValueError: If data shape is invalid
        
    Example:
        >>> data = torch.randn(100, 3, 224, 224)
        >>> result = process_data(data, batch_size=16)
        >>> print(result.shape)
        torch.Size([100, 3, 224, 224])
    """
```

### Type Hints

Always use type hints:

```python
from typing import Optional, Union, Dict, List
import torch

def forward_pass(
    inputs: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (`ModelLoader`, `DataProcessor`)
- **Functions/Variables**: `snake_case` (`load_model`, `batch_size`)
- **Constants**: `UPPER_SNAKE_CASE` (`MAX_LENGTH`, `DEFAULT_LR`)
- **Private**: `_leading_underscore` (`_internal_method`)
- **Protected**: `_single_underscore` (`_protected_attr`)

---

## Pull Request Process

### Before Submitting

1. **Run tests:**

   ```bash
   pytest tests/ -v --cov=src
   ```

2. **Check code style:**

   ```bash
   black --check src/ tests/
   flake8 src/ tests/
   ```

3. **Update documentation:**
   - Add docstrings to new functions
   - Update README if needed
   - Add to CHANGELOG.md

4. **Write tests:**
   - Unit tests for new functions
   - Integration tests for features
   - Minimum 80% code coverage

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests passed
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** must pass:
   - CI/CD pipeline
   - Code coverage
   - Linting

2. **Required reviews:**
   - Minimum 2 approvals for core changes
   - Minimum 1 approval for docs/tests

3. **Merge requirements:**
   - All conversations resolved
   - No merge conflicts
   - Up-to-date with main branch

---

## Issue Templates

### Bug Report

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run '...'
2. See error

**Expected behavior**
What you expected

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.10.0]
- PyTorch: [e.g. 2.1.0]
- CUDA: [e.g. 12.1]

**Additional context**
Add any other context
```

### Feature Request

```markdown
**Is your feature request related to a problem?**
A clear description

**Describe the solution you'd like**
What you want to happen

**Describe alternatives you've considered**
Other solutions

**Additional context**
Add any other context
```

---

## Testing Guidelines

### Unit Tests

```python
import pytest
import torch
from src.nexus_core.towers.loader import ModelLoader

def test_model_loader_initialization():
    """Test ModelLoader initializes correctly."""
    loader = ModelLoader()
    assert loader is not None
    assert hasattr(loader, 'registry')

def test_model_loading():
    """Test model can be loaded."""
    loader = ModelLoader()
    # Use small model for testing
    model = loader.load("gpt2", device_map="cpu")
    assert model is not None
```

### Integration Tests

```python
@pytest.mark.integration
def test_full_pipeline():
    """Test complete training pipeline."""
    from scripts.nexus_pipeline import run_pipeline
    
    config = {
        "model": "gpt2",
        "dataset": "test_dataset",
        "max_steps": 10
    }
    
    result = run_pipeline(config)
    assert result["status"] == "success"
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_inference_speed(benchmark):
    """Benchmark inference speed."""
    model = load_model()
    inputs = prepare_inputs()
    
    result = benchmark(model.generate, **inputs)
    assert result.stats.mean < 1.0  # Should complete in < 1s
```

---

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Include type hints
- Provide usage examples
- Document exceptions

### Guides

When adding features, update:

- `docs/NEXUS_USAGE_GUIDE.md` - Usage instructions
- `docs/TROUBLESHOOTING.md` - Common issues
- `docs/API.md` - API reference

### Changelog

Update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/):

```markdown
## [Unreleased]

### Added
- New feature X

### Changed
- Behavior of Y

### Fixed
- Bug in Z

### Deprecated
- Old API method

### Removed
- Legacy support

### Security
- Fixed vulnerability
```

---

## Release Process

1. **Version bump:**

   ```bash
   bump2version minor  # or major/patch
   ```

2. **Update changelog:**
   - Move unreleased changes to version section
   - Add release date

3. **Create PR:**
   - Title: "Release vX.Y.Z"
   - Include changelog

4. **After merge:**
   - Tag release: `git tag vX.Y.Z`
   - Push tags: `git push origin --tags`
   - Create GitHub release

---

## Community

### Communication Channels

- **GitHub Discussions:** Questions and ideas
- **Discord:** Real-time chat
- **Email:** <security@nexus.ai> (security issues only)

### Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Added to the community page

### Code Review Philosophy

- **Constructive feedback:** Focus on improvement
- **Explain why:** Don't just say "change this"
- **Be timely:** Review within 48 hours
- **Acknowledge good work:** Positive reinforcement

---

## Questions?

- Check [existing issues](https://github.com/yourusername/nexus/issues)
- Read [documentation](https://nexus.readthedocs.io)
- Ask in [Discussions](https://github.com/yourusername/nexus/discussions)

Thank you for contributing to Nexus! 🚀
