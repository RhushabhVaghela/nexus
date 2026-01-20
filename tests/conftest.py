"""
conftest.py - Shared pytest fixtures for Manus Model tests.

Provides:
- Real model fixtures (Qwen2.5-0.5B for text, Qwen2.5-Omni for multimodal)
- Device fixtures (GPU/CPU detection)
- Sample data fixtures
- Test paths fixtures

Usage:
    def test_with_real_model(real_text_model):
        output = real_text_model.generate("Hello")
        assert output is not None
"""

import os
import sys
import pytest
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============ PATH CONSTANTS ============
TEST_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-0.5B"
OMNI_MODEL_PATH = "/mnt/e/data/base-model/Qwen2.5-Omni-7B-GPTQ-Int4"
VISION_ENCODER_PATH = "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512"
AUDIO_ENCODER_PATH = "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo"


# ============ MARKERS ============
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "real_model: marks tests using real model inference")


# ============ DEVICE FIXTURES ============
@pytest.fixture(scope="session")
def device() -> str:
    """Get best available device (GPU preferred)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available()


# ============ MODEL PATH FIXTURES ============
@pytest.fixture(scope="session")
def text_model_path() -> str:
    """Path to lightweight text-only model for testing."""
    return TEST_MODEL_PATH


@pytest.fixture(scope="session")
def omni_model_path() -> str:
    """Path to full Omni model."""
    return OMNI_MODEL_PATH


@pytest.fixture(scope="session")
def vision_encoder_path() -> str:
    """Path to vision encoder."""
    return VISION_ENCODER_PATH


@pytest.fixture(scope="session")
def audio_encoder_path() -> str:
    """Path to audio encoder."""
    return AUDIO_ENCODER_PATH


# ============ REAL MODEL FIXTURES ============
@pytest.fixture(scope="session")
def real_text_tokenizer():
    """Load real tokenizer from Qwen2.5-0.5B."""
    from transformers import AutoTokenizer
    
    if not Path(TEST_MODEL_PATH).exists():
        pytest.skip(f"Test model not found: {TEST_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_PATH, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="session")
def real_text_model(device):
    """Load real Qwen2.5-0.5B model for testing."""
    from transformers import AutoModelForCausalLM
    
    if not Path(TEST_MODEL_PATH).exists():
        pytest.skip(f"Test model not found: {TEST_MODEL_PATH}")
    
    model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    return model


@pytest.fixture(scope="session")
def real_model_and_tokenizer(real_text_model, real_text_tokenizer):
    """Bundle model and tokenizer together."""
    return {"model": real_text_model, "tokenizer": real_text_tokenizer}


# ============ SAMPLE DATA FIXTURES ============
@pytest.fixture
def sample_text_prompt() -> str:
    """Simple text prompt for testing."""
    return "What is 2 + 2?"


@pytest.fixture
def sample_messages() -> list:
    """Sample chat messages."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


@pytest.fixture
def sample_training_sample() -> Dict[str, Any]:
    """Sample training data in messages format."""
    return {
        "id": "test_001",
        "messages": [
            {"role": "user", "content": "Write hello world in Python"},
            {"role": "assistant", "content": "```python\nprint('Hello, World!')\n```"},
        ],
        "domain": "code",
    }


@pytest.fixture
def sample_multimodal_sample() -> Dict[str, Any]:
    """Sample multimodal training data."""
    return {
        "id": "mm_001",
        "messages": [
            {"role": "user", "content": "Describe this image"},
            {"role": "assistant", "content": "The image shows a sunset over mountains."},
        ],
        "modalities": {
            "image": [{"path": "/fake/image.png", "type": "photo"}],
            "audio": [],
            "video": [],
        },
    }


# ============ CONFIG FIXTURES ============
@pytest.fixture
def encoders_config() -> Dict[str, Any]:
    """Load encoders.yaml config."""
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "encoders.yaml"
    
    if not config_path.exists():
        pytest.skip("encoders.yaml not found")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============ TEMP DIRECTORY FIXTURES ============
@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporary output directory for test artifacts."""
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def temp_checkpoint_dir(tmp_path) -> Path:
    """Temporary checkpoint directory."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    return ckpt


# ============ SKIP CONDITIONS ============
@pytest.fixture
def skip_if_no_gpu(has_gpu):
    """Skip test if no GPU available."""
    if not has_gpu:
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_model():
    """Skip test if test model not available."""
    if not Path(TEST_MODEL_PATH).exists():
        pytest.skip(f"Test model not found: {TEST_MODEL_PATH}")


# ============ CLEANUP ============
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
