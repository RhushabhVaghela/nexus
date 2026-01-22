
import pytest
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stages.reasoning_sft import main as sft_main
from src.stages.reasoning_grpo import main as grpo_main
from src.stages.agent_finetune import main as agent_main

# Real Model Path
REAL_MODEL_PATH = "/mnt/e/data/models/Qwen2.5-0.5B"

@pytest.fixture
def clean_output(tmp_path):
    output_dir = tmp_path / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    return output_dir

@pytest.fixture
def mock_sft_args(clean_output):
    return [
        "reasoning_sft.py",
        "--model", REAL_MODEL_PATH,
        "--reasoning", 
        "--output", str(clean_output / "sft"),
        "--epochs", "1",
        "--batch-size", "1",
        "--gradient-accumulation-steps", "1",
        "--save-steps", "5", 
        "--logging-steps", "1",
        "--max-length", "128", # Short context for speed
        "--no-bf16" # Ensure compatibility if needed, or assume bf16 supported
    ]

@pytest.fixture
def mock_grpo_args(clean_output):
    return [
        "reasoning_grpo.py",
        "--model", REAL_MODEL_PATH,
        "--reasoning",
        "--output", str(clean_output / "grpo"),
        "--iterations", "1", # 1 iteration only
        "--batch-size", "1",
        "--group-size", "2",
        "--max-length", "128" # Implicitly handle via Trainer config if applicable or just run
    ]

@pytest.fixture
def mock_agent_args(clean_output):
    return [
        "agent_finetune.py",
        "--model", REAL_MODEL_PATH,
        "--tools",
        "--output", str(clean_output / "agent"),
        "--num-epochs", "1",
        "--batch-size", "1"
    ]

# We STILL patch UniversalDatasetManager to avoid needing massive datasets
# But we let Model and Trainer be REAL.

@patch("src.stages.reasoning_sft.UniversalDatasetManager")
def test_sft_pipeline_real_model(mock_manager_cls, mock_sft_args):
    """Test SFT stage with REAL Qwen2.5-0.5B model."""
    
    # Mock Data Manager to return small dummy data
    mock_manager = mock_manager_cls.return_value
    # 2 samples to allow 1 batch step
    mock_manager.get_unified_train_dataset.return_value = [{"messages": [{"role": "user", "content": "test"}]}] * 4
    
    with patch.object(sys, 'argv', mock_sft_args):
        # This will load real model and run real training steps
        # Ensure we don't run forever -> args have low epochs/steps
        sft_main()
        
    # If we reached here, training ran successfully
    assert True


@patch("src.stages.reasoning_grpo.UniversalDatasetManager")
def test_grpo_pipeline_real_model(mock_manager_cls, mock_grpo_args):
    """Test GRPO stage with REAL Qwen2.5-0.5B model."""
    
    mock_manager = mock_manager_cls.return_value
    mock_manager.get_unified_train_dataset.return_value = [{"question": "q", "answer": "a"}] * 4
    
    with patch.object(sys, 'argv', mock_grpo_args):
         grpo_main()
         
    assert True


@patch("src.stages.agent_finetune.UniversalDatasetManager")
def test_agent_pipeline_real_model(mock_manager_cls, mock_agent_args):
    """Test Agent stage with REAL Qwen2.5-0.5B model."""
    
    mock_manager = mock_manager_cls.return_value
    mock_manager.get_unified_train_dataset.return_value = [{"messages": [{"role": "user", "content": "Help me with coding"}]}] * 4
    
    with patch.object(sys, 'argv', mock_agent_args):
        agent_main()
        
    assert True
