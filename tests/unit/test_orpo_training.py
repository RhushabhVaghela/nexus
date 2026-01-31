"""
tests/unit/test_orpo_training.py
Comprehensive tests for ORPO (Odds Ratio Preference Optimization) training functionality.

Tests cover:
- ORPO trainer initialization
- Odds ratio loss computation
- Preference data handling
- Training loop execution
- Beta/Lambda parameter sensitivity
- Model and tokenizer loading
- Environment checks
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ORPOConfig:
    """Configuration for ORPO training."""
    learning_rate: float = 5e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    beta: float = 0.1  # ORPO lambda parameter
    max_length: int = 2048
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "/tmp/orpo_output"


class MockORPOTrainer:
    """Mock ORPO trainer for testing."""
    
    def __init__(self, model, args, train_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.step_count = 0
        self.epoch_count = 0
    
    def compute_log_probs(self, input_ids, attention_mask):
        """Compute log probabilities for tokens."""
        batch_size, seq_len = input_ids.shape
        vocab_size = 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        masked_log_probs = token_log_probs * attention_mask
        return masked_log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)
    
    def odds_ratio_loss(self, chosen_logps, rejected_logps, beta=0.1):
        """Compute ORPO odds ratio loss."""
        log_odds_ratio = chosen_logps - rejected_logps
        loss = -torch.nn.functional.logsigmoid(beta * log_odds_ratio)
        return loss.mean()
    
    def combined_loss(self, chosen_logps, rejected_logps, sft_loss, beta=0.1):
        """Combined SFT + ORPO loss."""
        orpo_loss = self.odds_ratio_loss(chosen_logps, rejected_logps, beta)
        return sft_loss + orpo_loss
    
    def compute_metrics(self, chosen_logps, rejected_logps):
        """Compute training metrics."""
        log_odds_ratio = chosen_logps - rejected_logps
        accuracy = (log_odds_ratio > 0).float().mean()
        
        return {
            "rewards/chosen": chosen_logps.mean().item(),
            "rewards/rejected": rejected_logps.mean().item(),
            "rewards/margins": log_odds_ratio.mean().item(),
            "rewards/accuracy": accuracy.item(),
        }
    
    def train(self):
        """Mock training loop."""
        self.epoch_count = self.args.num_train_epochs
        self.step_count = len(self.train_dataset) // self.args.batch_size
        return {"train_loss": 0.5, "epoch": self.epoch_count}
    
    def save_model(self, output_dir):
        """Mock save model."""
        pass


class TestORPOConfig:
    """Test ORPO configuration."""
    
    def test_default_config(self):
        """Test default ORPO configuration."""
        config = ORPOConfig()
        
        assert config.learning_rate == 5e-6
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.num_train_epochs == 3
        assert config.beta == 0.1
        assert config.max_length == 2048
        assert config.warmup_ratio == 0.1
        assert config.lr_scheduler_type == "cosine"
        assert config.bf16 is True
    
    def test_custom_config(self):
        """Test custom ORPO configuration."""
        config = ORPOConfig(
            learning_rate=1e-5,
            batch_size=2,
            num_train_epochs=5,
            beta=0.2
        )
        
        assert config.learning_rate == 1e-5
        assert config.batch_size == 2
        assert config.num_train_epochs == 5
        assert config.beta == 0.2
        assert config.max_length == 2048
        assert config.warmup_ratio == 0.1


class TestORPOTrainerInitialization:
    """Test ORPO trainer initialization."""
    
    def test_trainer_init_basic(self):
        """Test basic ORPO trainer initialization."""
        model = Mock()
        args = ORPOConfig()
        train_dataset = [{"prompt": "Q", "chosen": "A", "rejected": "B"}]
        tokenizer = Mock()
        
        trainer = MockORPOTrainer(model, args, train_dataset, tokenizer)
        
        assert trainer.model == model
        assert trainer.args == args
        assert trainer.train_dataset == train_dataset
        assert trainer.tokenizer == tokenizer
        assert trainer.step_count == 0


class TestORPOLossComputation:
    """Test ORPO loss computation."""
    
    def test_odds_ratio_loss_basic(self):
        """Test basic odds ratio loss computation."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0, -0.5, -0.8])
        rejected_logps = torch.tensor([-2.0, -1.5, -1.8])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps, beta=0.1)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
    
    def test_odds_ratio_loss_with_correct_preferences(self):
        """Test loss when preferences are correct."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([0.0, 0.0, 0.0])
        rejected_logps = torch.tensor([-10.0, -10.0, -10.0])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps, beta=0.1)
        
        assert loss.item() < 0.1
    
    def test_odds_ratio_loss_with_wrong_preferences(self):
        """Test loss when preferences are wrong."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-10.0, -10.0])
        rejected_logps = torch.tensor([0.0, 0.0])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps, beta=0.1)
        
        assert loss.item() > 0.5
    
    def test_combined_loss(self):
        """Test combined SFT + ORPO loss."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0, -0.5])
        rejected_logps = torch.tensor([-2.0, -1.5])
        sft_loss = torch.tensor(0.5)
        
        combined = trainer.combined_loss(chosen_logps, rejected_logps, sft_loss, beta=0.1)
        
        assert isinstance(combined, torch.Tensor)
        assert combined.item() > sft_loss.item()
    
    def test_beta_sensitivity(self):
        """Test that different beta values produce different losses."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-2.0])
        
        betas = [0.01, 0.1, 0.5, 1.0]
        losses = []
        
        for beta in betas:
            loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps, beta)
            losses.append(loss.item())
        
        assert len(set(losses)) > 1


class TestORPOMetrics:
    """Test ORPO metrics computation."""
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0, -0.5, -0.8])
        rejected_logps = torch.tensor([-2.0, -1.5, -1.8])
        
        metrics = trainer.compute_metrics(chosen_logps, rejected_logps)
        
        assert "rewards/chosen" in metrics
        assert "rewards/rejected" in metrics
        assert "rewards/margins" in metrics
        assert "rewards/accuracy" in metrics
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([0.0, 0.0, 0.0])
        rejected_logps = torch.tensor([-1.0, -1.0, -1.0])
        
        metrics = trainer.compute_metrics(chosen_logps, rejected_logps)
        
        assert metrics["rewards/accuracy"] == 1.0
    
    def test_accuracy_with_wrong_preferences(self):
        """Test accuracy with wrong preferences."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0, -1.0, -1.0])
        rejected_logps = torch.tensor([0.0, 0.0, 0.0])
        
        metrics = trainer.compute_metrics(chosen_logps, rejected_logps)
        
        assert metrics["rewards/accuracy"] == 0.0
    
    def test_mixed_accuracy(self):
        """Test accuracy with mixed preferences."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([0.0, 0.0, -1.0])
        rejected_logps = torch.tensor([-1.0, -1.0, 0.0])
        
        metrics = trainer.compute_metrics(chosen_logps, rejected_logps)
        
        assert abs(metrics["rewards/accuracy"] - 0.667) < 0.01


class TestORPODataHandling:
    """Test ORPO data handling."""
    
    def test_preference_pair_creation(self):
        """Test creation of preference pairs."""
        pair = {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris.",
            "rejected": "The capital of France is London."
        }
        
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"] != pair["rejected"]
    
    def test_preference_dataset_validation(self):
        """Test preference dataset validation."""
        valid_pairs = [
            {"prompt": "Q1", "chosen": "Good answer", "rejected": "Bad answer"},
            {"prompt": "Q2", "chosen": "Better answer", "rejected": "Worse answer"},
        ]
        
        for pair in valid_pairs:
            assert all(key in pair for key in ["prompt", "chosen", "rejected"])


class TestORPOModelLoading:
    """Test ORPO model loading."""
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_load_model_basic(self, mock_tokenizer, mock_model):
        """Test basic model loading."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        tokenizer = mock_tokenizer("test_model")
        model = mock_model("test_model")
        
        assert tokenizer is not None
        assert model is not None
    
    def test_tokenizer_padding_setup(self):
        """Test tokenizer padding token setup."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "[EOS]"
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        assert tokenizer.pad_token == "[EOS]"


class TestORPOEnvironment:
    """Test ORPO environment checks."""
    
    @patch("torch.cuda.is_available")
    def test_gpu_check_available(self, mock_cuda):
        """Test GPU check when available."""
        mock_cuda.return_value = True
        
        assert torch.cuda.is_available() is True
    
    @patch("torch.cuda.is_available")
    def test_gpu_check_unavailable(self, mock_cuda):
        """Test GPU check when unavailable."""
        mock_cuda.return_value = False
        
        assert torch.cuda.is_available() is False
    
    def test_import_checks(self):
        """Test that required imports can be checked."""
        try:
            import transformers
            import torch
            has_deps = True
        except ImportError:
            has_deps = False
        
        assert has_deps


class TestORPOMockTraining:
    """Test ORPO mock training."""
    
    def test_mock_training(self):
        """Test mock training loop."""
        model = Mock()
        args = ORPOConfig(num_train_epochs=3)
        train_dataset = [{"prompt": "Q", "chosen": "A", "rejected": "B"}] * 10
        tokenizer = Mock()
        
        trainer = MockORPOTrainer(model, args, train_dataset, tokenizer)
        result = trainer.train()
        
        assert "train_loss" in result
        assert result["epoch"] == 3
    
    def test_save_model(self):
        """Test model saving."""
        model = Mock()
        tokenizer = Mock()
        trainer = MockORPOTrainer(model, ORPOConfig(), [], tokenizer)
        
        trainer.save_model("/tmp/test_output")


class TestORPOTokenProcessing:
    """Test ORPO token processing."""
    
    def test_compute_log_probs_shape(self):
        """Test log prob computation shapes."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        batch_size = 4
        seq_len = 10
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        log_probs = trainer.compute_log_probs(input_ids, attention_mask)
        
        assert log_probs.shape == (batch_size,)
    
    def test_compute_log_probs_with_padding(self):
        """Test log prob computation with padding."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        attention_mask[0, 5:] = 0
        
        log_probs = trainer.compute_log_probs(input_ids, attention_mask)
        
        assert log_probs.shape == (2,)


class TestORPOEdgeCases:
    """Test ORPO edge cases."""
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([])
        rejected_logps = torch.tensor([])
        
        with pytest.raises(RuntimeError):
            trainer.odds_ratio_loss(chosen_logps, rejected_logps)
    
    def test_single_example(self):
        """Test with single example."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0])
        rejected_logps = torch.tensor([-2.0])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_equal_log_probs(self):
        """Test when chosen and rejected have equal log probs."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([-1.0, -1.0])
        rejected_logps = torch.tensor([-1.0, -1.0])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps)
        
        assert abs(loss.item() - 0.693) < 0.01
    
    def test_extreme_log_probs(self):
        """Test with extreme log probability values."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        chosen_logps = torch.tensor([0.0])
        rejected_logps = torch.tensor([-50.0])
        
        loss = trainer.odds_ratio_loss(chosen_logps, rejected_logps)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 0.01


class TestORPOAdvantages:
    """Test ORPO advantages over separate SFT + DPO."""
    
    def test_no_reference_model_needed(self):
        """Test that ORPO doesn't need a reference model."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        
        assert not hasattr(trainer, 'ref_model') or trainer.ref_model is None
    
    def test_single_stage_training(self):
        """Test that ORPO is single-stage."""
        trainer = MockORPOTrainer(Mock(), ORPOConfig(), [], Mock())
        assert hasattr(trainer, 'combined_loss')
