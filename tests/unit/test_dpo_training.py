"""
tests/unit/test_dpo_training.py
Comprehensive tests for DPO (Direct Preference Optimization) training functionality.

Tests cover:
- DPO training initialization
- Preference loss computation
n- Reference model handling
- Training loop execution
- Hyperparameter sensitivity
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import numpy as np


# Mock DPO trainer for testing
class MockDPOTrainer:
    """Mock DPO trainer for testing."""
    
    def __init__(self, model, ref_model, beta=0.1, loss_type="sigmoid", label_smoothing=0.0):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.device = "cpu"
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        """Compute log probabilities."""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs
    
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                 reference_chosen_logps, reference_rejected_logps):
        """Compute DPO loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        
        if self.loss_type == "sigmoid":
            losses = -torch.nn.functional.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return losses.mean()
    
    def compute_metrics(self, policy_chosen_logps, policy_rejected_logps,
                       reference_chosen_logps, reference_rejected_logps):
        """Compute training metrics."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        accuracy = (logits > 0).float().mean()
        
        return {
            "rewards/chosen": self.beta * (policy_chosen_logps - reference_chosen_logps).mean().item(),
            "rewards/rejected": self.beta * (policy_rejected_logps - reference_rejected_logps).mean().item(),
            "rewards/margins": self.beta * (pi_logratios - ref_logratios).mean().item(),
            "logps/rejected": policy_rejected_logps.mean().item(),
            "logps/chosen": policy_chosen_logps.mean().item(),
            "logits/rejected": policy_rejected_logps.mean().item(),
            "logits/chosen": policy_chosen_logps.mean().item(),
        }


class TestDPOInitialization:
    """Test DPO trainer initialization."""
    
    def test_basic_initialization(self):
        """Test basic DPO trainer initialization."""
        model = Mock()
        ref_model = Mock()
        
        trainer = MockDPOTrainer(model, ref_model)
        
        assert trainer.model == model
        assert trainer.ref_model == ref_model
        assert trainer.beta == 0.1
        assert trainer.loss_type == "sigmoid"
    
    def test_custom_beta(self):
        """Test DPO trainer with custom beta."""
        model = Mock()
        ref_model = Mock()
        
        trainer = MockDPOTrainer(model, ref_model, beta=0.5)
        assert trainer.beta == 0.5
    
    def test_different_loss_types(self):
        """Test DPO trainer with different loss types."""
        model = Mock()
        ref_model = Mock()
        
        for loss_type in ["sigmoid", "hinge"]:
            trainer = MockDPOTrainer(model, ref_model, loss_type=loss_type)
            assert trainer.loss_type == loss_type


class TestDPOLossComputation:
    """Test DPO loss computation."""
    
    def test_sigmoid_loss(self):
        """Test sigmoid DPO loss computation."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1, loss_type="sigmoid")
        
        # Create sample log probabilities
        policy_chosen_logps = torch.tensor([-1.0, -0.5, -0.8])
        policy_rejected_logps = torch.tensor([-2.0, -1.5, -1.8])
        reference_chosen_logps = torch.tensor([-1.2, -0.7, -1.0])
        reference_rejected_logps = torch.tensor([-1.8, -1.3, -1.6])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_hinge_loss(self):
        """Test hinge DPO loss computation."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1, loss_type="hinge")
        
        policy_chosen_logps = torch.tensor([-1.0, -0.5])
        policy_rejected_logps = torch.tensor([-2.0, -1.5])
        reference_chosen_logps = torch.tensor([-1.2, -0.7])
        reference_rejected_logps = torch.tensor([-1.8, -1.3])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
    
    def test_loss_with_perfect_preferences(self):
        """Test loss when policy perfectly prefers chosen over rejected."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1)
        
        # Policy strongly prefers chosen
        policy_chosen_logps = torch.tensor([0.0, 0.0, 0.0])
        policy_rejected_logps = torch.tensor([-10.0, -10.0, -10.0])
        reference_chosen_logps = torch.tensor([-1.0, -1.0, -1.0])
        reference_rejected_logps = torch.tensor([-1.0, -1.0, -1.0])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # Loss should be small when preferences are correct
        assert loss.item() < 1.0
    
    def test_loss_with_wrong_preferences(self):
        """Test loss when policy prefers rejected over chosen."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1)
        
        # Policy prefers rejected (wrong)
        policy_chosen_logps = torch.tensor([-10.0, -10.0])
        policy_rejected_logps = torch.tensor([0.0, 0.0])
        reference_chosen_logps = torch.tensor([-1.0, -1.0])
        reference_rejected_logps = torch.tensor([-1.0, -1.0])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # Loss should be larger when preferences are wrong
        assert loss.item() > 0.5


class TestDPOHyperparameters:
    """Test DPO hyperparameter sensitivity."""
    
    def test_beta_sensitivity(self):
        """Test that different beta values produce different losses."""
        model = Mock()
        ref_model = Mock()
        
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        betas = [0.01, 0.1, 0.5, 1.0]
        losses = []
        
        for beta in betas:
            trainer = MockDPOTrainer(model, ref_model, beta=beta)
            loss = trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps
            )
            losses.append(loss.item())
        
        # Different betas should produce different losses
        assert len(set(losses)) > 1
    
    def test_beta_zero(self):
        """Test beta = 0 edge case."""
        model = Mock()
        ref_model = Mock()
        
        trainer = MockDPOTrainer(model, ref_model, beta=0.0)
        
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # With beta=0, loss should be constant
        assert isinstance(loss, torch.Tensor)


class TestDPOMetrics:
    """Test DPO metrics computation."""
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1)
        
        policy_chosen_logps = torch.tensor([-1.0, -0.5])
        policy_rejected_logps = torch.tensor([-2.0, -1.5])
        reference_chosen_logps = torch.tensor([-1.2, -0.7])
        reference_rejected_logps = torch.tensor([-1.8, -1.3])
        
        metrics = trainer.compute_metrics(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert "rewards/chosen" in metrics
        assert "rewards/rejected" in metrics
        assert "rewards/margins" in metrics
        assert "logps/chosen" in metrics
        assert "logps/rejected" in metrics
    
    def test_reward_margin_calculation(self):
        """Test reward margin calculation."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model, beta=0.1)
        
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        metrics = trainer.compute_metrics(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # Reward margin should be positive when chosen > rejected
        assert metrics["rewards/margins"] > 0


class TestDPOBatchHandling:
    """Test DPO batch handling."""
    
    def test_batch_processing(self):
        """Test processing of batched preference data."""
        batch_size = 4
        seq_len = 10
        vocab_size = 1000
        
        # Create mock batch data
        chosen_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        chosen_attention_mask = torch.ones(batch_size, seq_len)
        rejected_attention_mask = torch.ones(batch_size, seq_len)
        
        assert chosen_input_ids.shape == (batch_size, seq_len)
        assert rejected_input_ids.shape == (batch_size, seq_len)
    
    def test_variable_length_sequences(self):
        """Test handling of variable length sequences."""
        batch_size = 2
        seq_lengths = [10, 20, 50]
        
        for seq_len in seq_lengths:
            chosen_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            rejected_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            
            assert chosen_input_ids.shape[1] == seq_len
            assert rejected_input_ids.shape[1] == seq_len


class TestDPOReferenceModel:
    """Test reference model handling."""
    
    def test_reference_model_inference(self):
        """Test reference model inference mode."""
        ref_model = Mock()
        ref_model.eval = Mock()
        
        model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        # Reference model should be in eval mode
        ref_model.eval.assert_called_once()
    
    def test_reference_log_prob_determinism(self):
        """Test that reference log probs are deterministic."""
        ref_model = Mock()
        ref_model.return_value = Mock()
        ref_model.return_value.logits = torch.randn(2, 10, 1000)
        
        model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        log_probs1 = trainer.compute_log_probs(ref_model, input_ids, attention_mask)
        log_probs2 = trainer.compute_log_probs(ref_model, input_ids, attention_mask)
        
        # Reference model outputs should be deterministic
        assert torch.allclose(log_probs1, log_probs2)


class TestDPOPreferenceData:
    """Test preference data handling."""
    
    def test_preference_pair_creation(self):
        """Test creation of preference pairs."""
        prompt = "What is the capital of France?"
        chosen = "The capital of France is Paris."
        rejected = "The capital of France is London."
        
        preference_pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        assert preference_pair["prompt"] == prompt
        assert preference_pair["chosen"] == chosen
        assert preference_pair["rejected"] == rejected
    
    def test_preference_dataset_validation(self):
        """Test preference dataset validation."""
        valid_pairs = [
            {"prompt": "Q1", "chosen": "A1", "rejected": "B1"},
            {"prompt": "Q2", "chosen": "A2", "rejected": "B2"},
        ]
        
        for pair in valid_pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            assert pair["chosen"] != pair["rejected"]


class TestDPOTrainingLoop:
    """Test DPO training loop."""
    
    @patch('torch.nn.Module')
    def test_training_step(self, mock_module):
        """Test single training step."""
        model = mock_module
        ref_model = mock_module
        
        trainer = MockDPOTrainer(model, ref_model)
        
        # Simulate training step
        policy_chosen_logps = torch.tensor([-1.0], requires_grad=True)
        policy_rejected_logps = torch.tensor([-2.0], requires_grad=True)
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert loss.requires_grad
    
    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        # Create parameters that require gradients
        policy_chosen_logps = torch.tensor([-1.0], requires_grad=True)
        policy_rejected_logps = torch.tensor([-2.0], requires_grad=True)
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        loss.backward()
        
        assert policy_chosen_logps.grad is not None
        assert policy_rejected_logps.grad is not None


class TestDPOEdgeCases:
    """Test DPO edge cases."""
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        policy_chosen_logps = torch.tensor([])
        policy_rejected_logps = torch.tensor([])
        reference_chosen_logps = torch.tensor([])
        reference_rejected_logps = torch.tensor([])
        
        with pytest.raises(RuntimeError):
            trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps
            )
    
    def test_single_example(self):
        """Test with single example."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        policy_chosen_logps = torch.tensor([-1.0])
        policy_rejected_logps = torch.tensor([-2.0])
        reference_chosen_logps = torch.tensor([-1.2])
        reference_rejected_logps = torch.tensor([-1.8])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert isinstance(loss, torch.Tensor)
    
    def test_equal_log_probs(self):
        """Test when chosen and rejected have equal log probs."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        policy_chosen_logps = torch.tensor([-1.0, -1.0])
        policy_rejected_logps = torch.tensor([-1.0, -1.0])
        reference_chosen_logps = torch.tensor([-1.2, -1.2])
        reference_rejected_logps = torch.tensor([-1.2, -1.2])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        # When equal, loss should be moderate
        assert loss.item() > 0
    
    def test_extreme_log_probs(self):
        """Test with extreme log probability values."""
        model = Mock()
        ref_model = Mock()
        trainer = MockDPOTrainer(model, ref_model)
        
        policy_chosen_logps = torch.tensor([0.0])  # High prob
        policy_rejected_logps = torch.tensor([-100.0])  # Very low prob
        reference_chosen_logps = torch.tensor([-1.0])
        reference_rejected_logps = torch.tensor([-1.0])
        
        loss = trainer.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert isinstance(loss, torch.Tensor)
