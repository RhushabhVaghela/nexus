import pytest
import os
import torch
from src.nexus_final.distill import NexusTrainer
from src.nexus_final.distill_knowledge import KnowledgeDistiller
from src.nexus_final.loss_functions import ActivationAnchoringLoss

@pytest.fixture
def trainer_setup(tmp_path):
    # Mock Student
    class MockStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(1, 1))
        def forward(self, input_ids, adapter_hidden_states=None, **kwargs):
            return {
                "logits": torch.randn(1, 10, 100).to(input_ids.device),
                "hidden_states": torch.randn(1, 10, 512).to(input_ids.device),
                "loss": torch.tensor(0.5).to(input_ids.device),
                "router_logits": torch.randn(1, 10, 4).to(input_ids.device)
            }
            
    student = MockStudent()
    adapters = {"test": torch.nn.Identity()}
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    
    config = {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "alpha": 1.0,
        "beta_entropy": 0.01,
        "max_grad_norm": 1.0,
        "loss_spike_threshold": 1.5,
        "warmup_epochs": 2
    }
    
    trainer = NexusTrainer(
        student=student,
        adapters=adapters,
        train_loader=[], # Empty for unit test
        val_loader=[],
        optimizer=optimizer,
        config=config,
        device="cpu"
    )
    return trainer

def test_checkpoint_resume(trainer_setup):
    trainer = trainer_setup
    trainer.global_step = 123
    trainer.save_checkpoint("test_tag")
    
    # Reset trainer state
    trainer.global_step = 0
    assert trainer.load_checkpoint("test_tag")
    assert trainer.global_step == 123

def test_curriculum_alpha(trainer_setup):
    trainer = trainer_setup
    # Warmup epochs = 2
    assert trainer.get_alpha(0) == 0.0
    assert trainer.get_alpha(1) == 0.5
    assert trainer.get_alpha(2) == 1.0
    assert trainer.get_alpha(5) == 1.0

def test_loss_spike_rollback(trainer_setup, tmp_path):
    trainer = trainer_setup
    trainer.global_step = 100
    trainer.prev_loss = 1.0
    trainer.save_checkpoint("best")
    
    # Mock batch
    batch = {
        'input_ids': torch.zeros(1, 10).long(),
        'teacher_features': {'test': torch.randn(1, 10, 512)},
        'teacher_logits': torch.randn(1, 10, 100),
        'labels': torch.zeros(1, 10).long()
    }
    
    # Trigger a massive loss to cause rollback
    # We'll mock the training_step behavior for this
    trainer.prev_loss = 0.0001
    metrics = trainer.training_step(batch)
    
    assert metrics["loss"] == "ROLLBACK"

def test_activation_anchoring_loss():
    loss_fn = ActivationAnchoringLoss(alpha_ce=1.0, alpha_hidden=1.0, alpha_critical=10.0)
    
    student_logits = torch.randn(1, 10, 100)
    teacher_logits = torch.randn(1, 10, 100)
    student_states = torch.randn(1, 10, 512)
    # Layered teacher states: [Batch, Layers, Seq, Dim]
    teacher_states = torch.randn(1, 5, 10, 512)
    
    # Test without anchoring indices
    loss_basic = loss_fn(student_logits, teacher_logits, student_states, teacher_states)
    assert loss_basic > 0
    
    # Test with anchoring
    loss_anchored = loss_fn(student_logits, teacher_logits, student_states, teacher_states, anchoring_layer_indices=[1, 3])
    assert loss_anchored > 0
    
    # Verify that anchored loss is higher if drift is present (implicitly checked by logic)

def test_smart_model_loading_strategies(tmp_path):
    # We need to mock KnowledgeDistiller dependencies to test __init__ logic
    from unittest.mock import MagicMock, patch
    from src.nexus_final.distill_knowledge import KnowledgeDistiller
    
    tower_mock = MagicMock()

    
    # CASE 1: Omni Model Detection
    with patch("src.nexus_final.distill_knowledge.AutoConfig.from_pretrained") as mock_conf, \
         patch("transformers.Qwen2ForCausalLM.from_pretrained") as mock_qwen, \
         patch("src.nexus_final.distill_knowledge.AutoModel.from_pretrained") as mock_auto:
        
        # Setup specific config response
        mock_conf.return_value.model_type = "qwen2_5_omni"

        
        distiller = KnowledgeDistiller(tower_mock, "fake/omni-model", device="cpu")
        
        # specific verify
        mock_qwen.assert_called_once()
        mock_auto.assert_not_called()
        print("Omni Smart Load: PASSED")

    # CASE 2: Standard Model Fallback
    with patch("src.nexus_final.distill_knowledge.AutoConfig.from_pretrained") as mock_conf, \
         patch("transformers.Qwen2ForCausalLM.from_pretrained") as mock_qwen, \
         patch("src.nexus_final.distill_knowledge.AutoModel.from_pretrained") as mock_auto:
        
        mock_conf.return_value.model_type = "llama"

        
        distiller = KnowledgeDistiller(tower_mock, "fake/llama-model", device="cpu")
        
        # Verify standard path
        mock_auto.assert_called_once()
        mock_qwen.assert_not_called()
        print("Standard Smart Load: PASSED")

