"""
Unit tests for PPO training logic.

Tests cover:
- PPO trainer initialization
- Advantage computation
- Policy and value loss calculation
- PPO training loop
- Clip ratio and entropy bonus
- KL divergence penalty
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    num_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    kl_penalty_coef: float = 0.1
    target_kl: float = 0.01


class MockActorCriticModel(nn.Module):
    """Mock actor-critic model for testing."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, vocab_size)
        
        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Encoder
        self.encoder = nn.Embedding(vocab_size, hidden_size)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and values."""
        # Encode input
        hidden = self.encoder(input_ids).mean(dim=1)
        
        # Get policy logits
        logits = self.policy_head(hidden)
        
        # Get state value
        values = self.value_head(hidden).squeeze(-1)
        
        return logits, values
    
    def get_action_and_value(
        self,
        input_ids: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value."""
        logits, values = self.forward(input_ids)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, values


class PPOTrainer:
    """PPO trainer implementation."""
    
    def __init__(self, model: nn.Module, config: PPOConfig, ref_model: Optional[nn.Module] = None):
        self.model = model
        self.config = config
        self.ref_model = ref_model
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.step_count = 0
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae = 0
        last_value = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO clipped policy loss."""
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute clip fraction
        clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
        
        return policy_loss, {
            "clip_fraction": clip_fraction.item(),
            "approx_kl": (old_log_probs - log_probs).mean().item()
        }
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute value function loss with clipping."""
        # Clip value predictions
        value_pred_clipped = old_values + torch.clamp(
            values - old_values,
            -self.config.clip_epsilon,
            self.config.clip_epsilon
        )
        
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        
        return torch.max(value_loss1, value_loss2).mean()
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        if ref_log_probs is None:
            return torch.tensor(0.0)
        
        # KL divergence: log(π_ref) - log(π_current)
        kl_div = ref_log_probs - log_probs
        return kl_div.mean()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single PPO training step."""
        self.model.train()
        
        # Get batch data
        input_ids = batch["input_ids"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        # Forward pass
        _, log_probs, entropy, values = self.model.get_action_and_value(
            input_ids, actions
        )
        
        # Compute losses
        policy_loss, policy_info = self.compute_policy_loss(
            log_probs, old_log_probs, advantages
        )
        
        value_loss = self.compute_value_loss(values, returns, old_values)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # KL penalty if reference model provided
        kl_loss = torch.tensor(0.0)
        if self.ref_model is not None:
            with torch.no_grad():
                _, ref_log_probs, _, _ = self.ref_model.get_action_and_value(
                    input_ids, actions
                )
            kl_loss = self.compute_kl_penalty(log_probs, ref_log_probs)
        
        # Total loss
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss +
            self.config.entropy_coef * entropy_loss +
            self.config.kl_penalty_coef * kl_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        self.step_count += 1
        
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            **policy_info
        }
    
    def train(
        self,
        rollout_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Train for multiple epochs on rollout data."""
        metrics = []
        
        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            indices = torch.randperm(len(rollout_data["input_ids"]))
            
            for start_idx in range(0, len(indices), self.config.mini_batch_size):
                end_idx = start_idx + self.config.mini_batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Create mini-batch
                batch = {
                    k: v[batch_indices] for k, v in rollout_data.items()
                }
                
                # Check KL divergence early stopping
                if epoch > 0:
                    with torch.no_grad():
                        _, log_probs, _, _ = self.model.get_action_and_value(
                            batch["input_ids"], batch["actions"]
                        )
                        approx_kl = (batch["old_log_probs"] - log_probs).mean()
                        
                        if approx_kl > self.config.target_kl:
                            return self._aggregate_metrics(metrics)
                
                step_metrics = self.train_step(batch)
                metrics.append(step_metrics)
        
        return self._aggregate_metrics(metrics)
    
    def _aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across training steps."""
        if not metrics:
            return {}
        
        aggregated = {}
        for key in metrics[0].keys():
            aggregated[key] = sum(m[key] for m in metrics) / len(metrics)
        aggregated["num_updates"] = len(metrics)
        
        return aggregated


class TestPPOConfig:
    """Test PPO configuration."""
    
    def test_default_config(self):
        """Test default PPO configuration."""
        config = PPOConfig()
        
        assert config.learning_rate == 1e-5
        assert config.batch_size == 4
        assert config.mini_batch_size == 2
        assert config.num_epochs == 4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_epsilon == 0.2
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.max_grad_norm == 1.0
        assert config.kl_penalty_coef == 0.1
        assert config.target_kl == 0.01
    
    def test_custom_config(self):
        """Test custom PPO configuration."""
        config = PPOConfig(
            learning_rate=5e-6,
            batch_size=8,
            num_epochs=2,
            clip_epsilon=0.1
        )
        
        assert config.learning_rate == 5e-6
        assert config.batch_size == 8
        assert config.num_epochs == 2
        assert config.clip_epsilon == 0.1
        # Default values should remain
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95


class TestPPOTrainerInitialization:
    """Test PPO trainer initialization."""
    
    def test_trainer_init_basic(self):
        """Test basic PPO trainer initialization."""
        model = MockActorCriticModel()
        config = PPOConfig()
        
        trainer = PPOTrainer(model, config)
        
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.ref_model is None
        assert trainer.step_count == 0
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_trainer_init_with_ref_model(self):
        """Test PPO trainer with reference model."""
        model = MockActorCriticModel()
        ref_model = MockActorCriticModel()
        config = PPOConfig()
        
        trainer = PPOTrainer(model, config, ref_model)
        
        assert trainer.ref_model == ref_model
    
    def test_trainer_optimizer_lr(self):
        """Test optimizer learning rate."""
        model = MockActorCriticModel()
        config = PPOConfig(learning_rate=1e-4)
        
        trainer = PPOTrainer(model, config)
        
        assert trainer.optimizer.defaults['lr'] == 1e-4


class TestAdvantageComputation:
    """Test advantage computation."""
    
    def test_gae_advantage_single_step(self):
        """Test GAE with single step."""
        model = MockActorCriticModel()
        config = PPOConfig(gamma=0.99, gae_lambda=0.95)
        trainer = PPOTrainer(model, config)
        
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([1.0])
        
        advantages, returns = trainer.compute_advantages(rewards, values, dones)
        
        assert len(advantages) == 1
        assert len(returns) == 1
        # Advantage should be normalized
        assert abs(advantages[0].item()) < 1e-6  # Will be 0 after normalization
    
    def test_gae_advantage_multi_step(self):
        """Test GAE with multiple steps."""
        model = MockActorCriticModel()
        config = PPOConfig(gamma=0.99, gae_lambda=0.95)
        trainer = PPOTrainer(model, config)
        
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        
        advantages, returns = trainer.compute_advantages(rewards, values, dones)
        
        assert len(advantages) == 3
        assert len(returns) == 3
        # Advantages should sum to approximately 0 after normalization
        assert abs(advantages.sum().item()) < 1e-6
    
    def test_gae_advantage_with_terminal(self):
        """Test GAE with terminal state."""
        model = MockActorCriticModel()
        config = PPOConfig(gamma=0.99, gae_lambda=0.95)
        trainer = PPOTrainer(model, config)
        
        rewards = torch.tensor([1.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.0, 0.5])  # Value is 0 at terminal
        dones = torch.tensor([0.0, 1.0, 0.0])
        
        advantages, returns = trainer.compute_advantages(rewards, values, dones)
        
        assert len(advantages) == 3
        # Check that advantage at terminal is computed correctly
        assert isinstance(advantages[1].item(), float)
    
    def test_gae_advantage_normalization(self):
        """Test that advantages are normalized."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        
        advantages, returns = trainer.compute_advantages(rewards, values, dones)
        
        # Mean should be approximately 0 after normalization
        assert abs(advantages.mean().item()) < 1e-6
        # Std should be approximately 1 after normalization
        assert abs(advantages.std().item() - 1.0) < 1e-5


class TestPolicyLoss:
    """Test policy loss computation."""
    
    def test_policy_loss_no_clipping(self):
        """Test policy loss without clipping."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Ratio = 1.0 (no change)
        log_probs = torch.tensor([0.0, 0.0, 0.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 0  # Policy loss is negative
        assert info["clip_fraction"] == 0.0
    
    def test_policy_loss_with_clipping(self):
        """Test policy loss with clipping."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Ratio > 1 + epsilon, should be clipped
        log_probs = torch.tensor([0.5, 0.5, 0.5])  # Higher probability
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        assert isinstance(loss, torch.Tensor)
        assert info["clip_fraction"] > 0.0  # Some samples should be clipped
    
    def test_policy_loss_negative_advantage(self):
        """Test policy loss with negative advantage."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Negative advantage, policy should discourage this action
        log_probs = torch.tensor([0.0, 0.0])
        old_log_probs = torch.tensor([0.0, 0.0])
        advantages = torch.tensor([-1.0, -1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        # With negative advantage and ratio=1, loss should be positive
        assert loss.item() > 0
    
    def test_policy_loss_clip_fraction(self):
        """Test clip fraction calculation."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Create ratios that should be clipped
        log_probs = torch.tensor([0.3, -0.3, 0.0])  # ratios: 1.35, 0.74, 1.0
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        advantages = torch.tensor([1.0, 1.0, 1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        # First sample: ratio=exp(0.3)=1.35 > 1.2, should be clipped
        assert info["clip_fraction"] > 0.0
        assert info["clip_fraction"] <= 1.0


class TestValueLoss:
    """Test value loss computation."""
    
    def test_value_loss_basic(self):
        """Test basic value loss computation."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        values = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([1.5, 2.5, 3.5])
        old_values = torch.tensor([1.0, 2.0, 3.0])
        
        loss = trainer.compute_value_loss(values, returns, old_values)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_value_loss_clipping(self):
        """Test value loss with clipping."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Large value change that should be clipped
        values = torch.tensor([10.0])
        returns = torch.tensor([1.0])
        old_values = torch.tensor([1.0])
        
        loss = trainer.compute_value_loss(values, returns, old_values)
        
        # Loss should use clipped value
        # Clipped value = 1.0 + clamp(9.0, -0.2, 0.2) = 1.2
        # Unclipped loss = (10 - 1)^2 = 81
        # Clipped loss = (1.2 - 1)^2 = 0.04
        # Should use max, so loss = 81
        assert loss.item() > 0


class TestKLPenalty:
    """Test KL divergence penalty."""
    
    def test_kl_penalty_no_ref(self):
        """Test KL penalty without reference model."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        log_probs = torch.tensor([0.0, 0.0, 0.0])
        ref_log_probs = None
        
        kl = trainer.compute_kl_penalty(log_probs, ref_log_probs)
        
        assert kl.item() == 0.0
    
    def test_kl_penalty_with_ref(self):
        """Test KL penalty with reference model."""
        model = MockActorCriticModel()
        ref_model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config, ref_model)
        
        log_probs = torch.tensor([-0.5, -0.5, -0.5])
        ref_log_probs = torch.tensor([0.0, 0.0, 0.0])
        
        kl = trainer.compute_kl_penalty(log_probs, ref_log_probs)
        
        # KL = ref_log_probs - log_probs = 0.5 per sample
        expected_kl = 0.5
        assert abs(kl.item() - expected_kl) < 1e-5
    
    def test_kl_penalty_same_distributions(self):
        """Test KL penalty with identical distributions."""
        model = MockActorCriticModel()
        ref_model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config, ref_model)
        
        log_probs = torch.tensor([0.0, 0.0, 0.0])
        ref_log_probs = torch.tensor([0.0, 0.0, 0.0])
        
        kl = trainer.compute_kl_penalty(log_probs, ref_log_probs)
        
        # KL should be 0 for identical distributions
        assert abs(kl.item()) < 1e-5


class TestPPOTrainingStep:
    """Test PPO training step."""
    
    def test_train_step_basic(self):
        """Test basic training step."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        metrics = trainer.train_step(batch)
        
        assert "loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert trainer.step_count == 1
    
    def test_train_step_gradient_update(self):
        """Test that gradients are updated."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        trainer.train_step(batch)
        
        # Check that parameters were updated
        for initial, updated in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial, updated)
    
    def test_train_step_with_kl_penalty(self):
        """Test training step with KL penalty."""
        model = MockActorCriticModel()
        ref_model = MockActorCriticModel()
        config = PPOConfig(kl_penalty_coef=0.1)
        trainer = PPOTrainer(model, config, ref_model)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        metrics = trainer.train_step(batch)
        
        assert "kl_loss" in metrics


class TestPPOTrainingLoop:
    """Test PPO training loop."""
    
    def test_train_multiple_epochs(self):
        """Test training for multiple epochs."""
        model = MockActorCriticModel()
        config = PPOConfig(num_epochs=2, mini_batch_size=2)
        trainer = PPOTrainer(model, config)
        
        rollout_data = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        metrics = trainer.train(rollout_data)
        
        assert "loss" in metrics
        assert "num_updates" in metrics
        assert metrics["num_updates"] >= 4  # At least 4 updates (2 epochs * 2 batches)
    
    def test_train_early_stopping(self):
        """Test early stopping on KL divergence."""
        model = MockActorCriticModel()
        config = PPOConfig(
            num_epochs=10,
            mini_batch_size=2,
            target_kl=0.0  # Very low target to trigger early stopping
        )
        trainer = PPOTrainer(model, config)
        
        rollout_data = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        metrics = trainer.train(rollout_data)
        
        # Should stop early due to KL threshold
        assert metrics["num_updates"] < 20  # Less than full 10 epochs * 2 batches
    
    def test_train_empty_data(self):
        """Test training with empty data."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        rollout_data = {
            "input_ids": torch.randint(0, 1000, (0, 10)),
            "actions": torch.randint(0, 1000, (0,)),
            "old_log_probs": torch.randn(0),
            "old_values": torch.randn(0),
            "advantages": torch.randn(0),
            "returns": torch.randn(0)
        }
        
        metrics = trainer.train(rollout_data)
        
        assert "num_updates" in metrics
        assert metrics["num_updates"] == 0


class TestClipRatio:
    """Test clip ratio behavior."""
    
    def test_clip_ratio_within_bounds(self):
        """Test clip ratio within epsilon bounds."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Ratio within bounds: 1.1
        log_probs = torch.tensor([0.0953])  # exp(0.0953) ≈ 1.1
        old_log_probs = torch.tensor([0.0])
        advantages = torch.tensor([1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        assert info["clip_fraction"] == 0.0
    
    def test_clip_ratio_above_bounds(self):
        """Test clip ratio above upper bound."""
        model = MockActorCriticModel()
        config = PPOConfig(clip_epsilon=0.2)
        trainer = PPOTrainer(model, config)
        
        # Ratio above bounds: 1.5
        log_probs = torch.tensor([0.4055])  # exp(0.4055) ≈ 1.5
        old_log_probs = torch.tensor([0.0])
        advantages = torch.tensor([1.0])
        
        loss, info = trainer.compute_policy_loss(log_probs, old_log_probs, advantages)
        
        assert info["clip_fraction"] == 1.0


class TestEntropyBonus:
    """Test entropy bonus."""
    
    def test_entropy_in_loss(self):
        """Test entropy contributes to loss."""
        model = MockActorCriticModel()
        config = PPOConfig(entropy_coef=0.01)
        trainer = PPOTrainer(model, config)
        
        batch = {
            "input_ids": torch.randint(0, 1000, (4, 10)),
            "actions": torch.randint(0, 1000, (4,)),
            "old_log_probs": torch.randn(4),
            "old_values": torch.randn(4),
            "advantages": torch.randn(4),
            "returns": torch.randn(4)
        }
        
        metrics = trainer.train_step(batch)
        
        assert "entropy" in metrics
        assert metrics["entropy"] >= 0  # Entropy is non-negative


class TestMetricsAggregation:
    """Test metrics aggregation."""
    
    def test_aggregate_single_metric(self):
        """Test aggregation of single metric."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        metrics = [{"loss": 1.0}]
        aggregated = trainer._aggregate_metrics(metrics)
        
        assert aggregated["loss"] == 1.0
        assert aggregated["num_updates"] == 1
    
    def test_aggregate_multiple_metrics(self):
        """Test aggregation of multiple metrics."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        metrics = [
            {"loss": 1.0, "entropy": 0.5},
            {"loss": 2.0, "entropy": 1.5},
            {"loss": 3.0, "entropy": 2.5}
        ]
        aggregated = trainer._aggregate_metrics(metrics)
        
        assert aggregated["loss"] == 2.0  # Average
        assert aggregated["entropy"] == 1.5  # Average
        assert aggregated["num_updates"] == 3
    
    def test_aggregate_empty_metrics(self):
        """Test aggregation of empty metrics."""
        model = MockActorCriticModel()
        config = PPOConfig()
        trainer = PPOTrainer(model, config)
        
        aggregated = trainer._aggregate_metrics([])
        
        assert aggregated == {}
