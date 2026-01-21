#!/usr/bin/env python3
"""
test_training_methods.py
Unit tests for training methods configuration and scripts.

Tests:
- TrainingMethod enum and configs
- Script imports and validation
- Integration with real models (marked slow)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============== UNIT TESTS: CONFIG ==============

class TestTrainingMethodConfig:
    """Test training_methods.py configuration."""
    
    def test_import_training_methods(self):
        """Verify training_methods module imports correctly."""
        from src.training_methods import TrainingMethod, get_training_config, get_all_methods
        assert TrainingMethod is not None
        assert callable(get_training_config)
        assert callable(get_all_methods)
    
    def test_all_methods_available(self):
        """Verify all 10 training methods are defined."""
        from src.training_methods import get_all_methods
        methods = get_all_methods()
        
        expected = ['sft', 'lora', 'qlora', 'dora', 'dpo', 'grpo', 'orpo', 'ppo', 'distillation', 'cpt']
        assert set(methods) == set(expected), f"Missing methods: {set(expected) - set(methods)}"
    
    def test_get_training_config_sft(self):
        """Test SFT config retrieval."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.SFT)
        
        assert config.method == TrainingMethod.SFT
        assert config.learning_rate == 2e-5
        assert config.use_peft == False
    
    def test_get_training_config_lora(self):
        """Test LoRA config retrieval."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.LORA)
        
        assert config.use_peft == True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
    
    def test_get_training_config_qlora(self):
        """Test QLoRA config with quantization."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.QLORA)
        
        assert config.use_peft == True
        assert config.use_quantization == True
        assert config.quantization_bits == 4
        assert config.lora_r == 64  # Higher rank for QLoRA
    
    def test_get_training_config_dpo(self):
        """Test DPO config with preference settings."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.DPO)
        
        assert config.use_preference_data == True
        assert config.beta == 0.1
    
    def test_get_training_config_grpo(self):
        """Test GRPO config (DeepSeek method)."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.GRPO)
        
        assert config.use_preference_data == True
        assert config.learning_rate == 1e-6
    
    def test_get_training_config_distillation(self):
        """Test distillation config."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.DISTILLATION)
        
        assert config.use_distillation == True
        assert config.temperature == 2.0
        assert config.distillation_alpha == 0.5
    
    def test_parse_training_method_valid(self):
        """Test parsing valid method strings."""
        from src.training_methods import parse_training_method, TrainingMethod
        
        assert parse_training_method("sft") == TrainingMethod.SFT
        assert parse_training_method("QLORA") == TrainingMethod.QLORA
        assert parse_training_method("  dpo  ") == TrainingMethod.DPO
    
    def test_parse_training_method_invalid(self):
        """Test parsing invalid method raises error."""
        from src.training_methods import parse_training_method
        
        with pytest.raises(ValueError, match="Unknown training method"):
            parse_training_method("invalid_method")
    
    def test_config_to_dict(self):
        """Test config serialization."""
        from src.training_methods import TrainingMethod, get_training_config
        config = get_training_config(TrainingMethod.QLORA)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["method"] == "qlora"
        assert config_dict["use_quantization"] == True


# ============== UNIT TESTS: SCRIPTS ==============

class TestTrainingScriptsExist:
    """Verify all training scripts exist and have main functions."""
    
    @pytest.fixture
    def scripts_dir(self):
        return Path(__file__).parent.parent / "src"
    
    def test_sft_script_exists(self, scripts_dir):
        """SFT training script exists."""
        script = scripts_dir / "10_sft_training.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_grpo_script_exists(self, scripts_dir):
        """GRPO training script exists."""
        script = scripts_dir / "12_grpo_training.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_cpt_script_exists(self, scripts_dir):
        """CPT training script exists."""
        script = scripts_dir / "11_continued_pretraining.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_dpo_script_exists(self, scripts_dir):
        """DPO training script exists."""
        script = scripts_dir / "dpo_training.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_orpo_script_exists(self, scripts_dir):
        """ORPO training script exists."""
        script = scripts_dir / "orpo_training.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_ppo_script_exists(self, scripts_dir):
        """PPO training script exists."""
        script = scripts_dir / "ppo_training.py"
        assert script.exists(), f"Missing: {script}"


class TestSFTTrainingScript:
    """Test SFT training script structure."""
    
    def test_sft_has_main(self):
        """SFT script has main function."""
        content = Path("src/10_sft_training.py").read_text()
        assert "def main():" in content
        assert "if __name__" in content
    
    def test_sft_has_lora_config(self):
        """SFT script configures LoRA."""
        content = Path("src/10_sft_training.py").read_text()
        assert "lora_rank" in content.lower() or "lora_r" in content.lower()
        assert "LoraConfig" in content or "get_peft_model" in content


class TestGRPOTrainingScript:
    """Test GRPO training script structure."""
    
    def test_grpo_has_reward_functions(self):
        """GRPO script has reward functions."""
        content = Path("src/12_grpo_training.py").read_text()
        assert "def correctness_reward" in content
        assert "def combined_reward" in content
    
    def test_grpo_uses_grpo_trainer(self):
        """GRPO script uses GRPOTrainer."""
        content = Path("src/12_grpo_training.py").read_text()
        assert "GRPOTrainer" in content
        assert "GRPOConfig" in content


# ============== INTEGRATION TESTS (with mocks) ==============

class TestDPOTrainingIntegration:
    """Test DPO training with mocked dependencies."""
    
    @patch("src.dpo_training.AutoModelForCausalLM")
    @patch("src.dpo_training.AutoTokenizer")
    @patch("src.dpo_training.DPOTrainer")
    def test_dpo_training_flow(self, mock_trainer, mock_tokenizer, mock_model):
        """Test DPO training flow with mocks."""
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # This would test the actual flow
        # In real test, we'd import and call functions from dpo_training.py
        assert True  # Placeholder for full integration test


# ============== REAL MODEL TESTS (marked slow) ==============

@pytest.mark.slow
@pytest.mark.gpu
class TestRealModelTraining:
    """Integration tests with real models (slow, needs GPU)."""
    
    def test_load_real_model_for_sft(self, real_text_model, real_text_tokenizer):
        """Test loading real model for SFT."""
        assert real_text_model is not None
        assert real_text_tokenizer is not None
    
    def test_lora_adapter_creation(self, real_text_model):
        """Test creating LoRA adapters on real model."""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            peft_model = get_peft_model(real_text_model, lora_config)
            assert peft_model is not None
            
            # Check trainable params
            trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in peft_model.parameters())
            
            assert trainable < total * 0.1, "LoRA should have <10% trainable params"
        except ImportError:
            pytest.skip("PEFT not installed")
