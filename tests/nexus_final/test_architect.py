import pytest
import shutil
import os
from src.nexus_final.architect import NeuralArchitect

@pytest.fixture
def architect_setup(tmp_path):
    output_dir = tmp_path / "test_architect"
    architect = NeuralArchitect(
        output_dir=str(output_dir)
    )
    return architect, output_dir

def test_adapter_config_logic(architect_setup):
    architect, _ = architect_setup
    
    # High intrinsic dim
    # Note: NeuralArchitect.determine_adapter_config takes (teacher_id, profile_data, ...)
    profile_data = {"teacher1": {"intrinsic_dimension": 128}}
    config_high = architect.determine_adapter_config("teacher1", profile_data, max_rank_limit=64)
    assert config_high["r"] == 64
    assert config_high["lora_alpha"] == 128
    
    # Low intrinsic dim
    profile_data_low = {"teacher1": {"intrinsic_dimension": 16}}
    config_low = architect.determine_adapter_config("teacher1", profile_data_low, max_rank_limit=64)
    assert config_low["r"] == 16
    
    # Floor check
    profile_data_floor = {"teacher1": {"intrinsic_dimension": 4}}
    config_floor = architect.determine_adapter_config("teacher1", profile_data_floor, max_rank_limit=64)
    assert config_floor["r"] == 8

def test_student_model_synthesis(architect_setup):
    architect, output_dir = architect_setup
    
    output_path = os.path.join(output_dir, "nexus_student.py")
    adapter_config = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.05}
    
    architect.synthesize_student_model(
        output_path=output_path,
        base_model_name="facebook/opt-125m",
        adapter_config=adapter_config,
        teacher_hidden_dim=768
    )
    
    # Verify file existence
    assert os.path.exists(os.path.join(output_dir, "nexus_student.py"))
    
    # Read generated file and check for key components
    with open(os.path.join(output_dir, "nexus_student.py"), "r") as f:
        content = f.read()
        assert "class NexusStudent" in content
        assert "entropy_loss" in content # Verify my router diversity fix
        assert "projected_teacher_latents" in content
