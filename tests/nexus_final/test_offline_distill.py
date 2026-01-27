import pytest
import torch
import os
from unittest.mock import MagicMock
from src.nexus_final.distill import NexusTrainer

def test_offline_distillation_loading(tmp_path):
    # Setup mock student and loaders
    student = MagicMock()
    # Mock student forward to return a dict with grad-enabled loss
    student.side_effect = lambda **kwargs: {
        "logits": torch.randn(1, 5, 32000),
        "hidden_states": torch.randn(1, 5, 4096),
        "loss": torch.tensor(1.0, requires_grad=True)
    }
    
    # Create a fake activation cache
    cache_dir = tmp_path / "activation_cache"
    cache_dir.mkdir()
    
    layer_name = "layer_0"
    batch_idx = 42
    cache_file = cache_dir / f"layer_{layer_name}_batch_{batch_idx}.pt"
    expected_feats = torch.randn(1, 5, 4096)
    torch.save(expected_feats, cache_file)
    
    # Mock data
    batch = {
        "input_ids": torch.zeros(1, 5).long(),
        "teacher_logits": torch.randn(1, 5, 32000),
        "labels": torch.zeros(1, 5).long(),
        "batch_idx": batch_idx
    }
    
    # Config
    config = {
        "offline_distillation": True,
        "activation_cache_dir": str(cache_dir),
        "critical_layers": [layer_name],
        "alpha": 1.0,
        "checkpoint_dir": str(tmp_path / "ckpts")
    }
    
    # Initialize Trainer with CPU device for testing
    optimizer = MagicMock()
    mock_adapter = MagicMock()
    # Mock adapter to return its input (identity)
    mock_adapter.side_effect = lambda x: x
    
    trainer = NexusTrainer(
        student=student,
        adapters={layer_name: mock_adapter},
        train_loader=None,
        val_loader=None,
        optimizer=optimizer,
        device="cpu", # Force CPU for test
        config=config
    )
    
    # Mock loss_fn to return a real tensor with grad
    trainer.loss_fn = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))
    metrics = trainer.training_step(batch)
    
    # Assert
    # Check if loss_fn was called with the features from SSD
    trainer.loss_fn.assert_called()
    call_args = trainer.loss_fn.call_args[1]
    # teacher_states should match expected_feats (compare on CPU)
    assert torch.allclose(call_args["teacher_states"].cpu(), expected_feats.cpu())
    assert metrics["loss"] == "1.0000"

if __name__ == "__main__":
    pytest.main([__file__])
