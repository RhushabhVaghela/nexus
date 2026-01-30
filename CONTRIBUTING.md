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
│   ├── nexus_core/        # Core modules
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
