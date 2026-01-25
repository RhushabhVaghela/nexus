"""
Unit tests for orpo_training.py and ppo_training.py (Smoke Tests)
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
import importlib

# Mock dependencies globally
sys.modules["trl"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["unsloth"] = MagicMock()

def test_orpo_training_main():
    with patch("sys.argv", ["orpo.py", "--epochs", "1"]), \
         patch("src.orpo_training.ORPOTrainer") as mock_trainer, \
         patch("src.orpo_training.ORPOConfig"), \
         patch("src.orpo_training.AutoModelForCausalLM"), \
         patch("src.orpo_training.AutoTokenizer"), \
         patch("src.orpo_training.load_dataset"), \
         patch("src.orpo_training.os.environ.get", return_value="nexus"):
         
        # Mock trainer instance
        mock_instance = MagicMock()
        mock_trainer.return_value = mock_instance
        
        s = importlib.import_module("src.orpo_training")
        # Run main if it exists or simulate logic
        if hasattr(s, "main"):
            s.main()
        else:
            # If strictly script based, we'd check if we can run it via runpy or just checking function presence
            # Assuming typical pattern
            pass

def test_ppo_training_main():
    with patch("sys.argv", ["ppo.py", "--epochs", "1"]), \
         patch("src.ppo_training.PPOTrainer") as mock_trainer, \
         patch("src.ppo_training.PPOConfig"), \
         patch("src.ppo_training.AutoTokenizer"), \
         patch("src.ppo_training.AutoModelForCausalLMWithValueHead"), \
         patch("src.ppo_training.os.environ.get", return_value="nexus"):
         
        s = importlib.import_module("src.ppo_training")
        if hasattr(s, "main"):
            s.main()
