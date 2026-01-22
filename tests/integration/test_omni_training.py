"""
Integration tests for Omni model training.
Tests training stage, validation, and inference.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOmniStageImport:
    """Test Omni stage module imports."""
    
    def test_import_stage(self):
        """Test OmniTrainingStage can be imported."""
        from src.stages.stage_omni import OmniTrainingStage
        assert OmniTrainingStage is not None
    
    def test_import_config(self):
        """Test OmniStageConfig can be imported."""
        from src.stages.stage_omni import OmniStageConfig
        assert OmniStageConfig is not None
    
    def test_stage_has_name(self):
        """Test stage has correct name."""
        from src.stages.stage_omni import OmniTrainingStage
        assert OmniTrainingStage.name == "omni"


class TestOmniStageConfig:
    """Test OmniStageConfig configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from src.stages.stage_omni import OmniStageConfig
        
        config = OmniStageConfig()
        
        assert config.freeze_talker is True
        assert config.train_thinker_only is True
        assert config.learning_rate == 1e-6
        assert config.gradient_accumulation_steps == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from src.stages.stage_omni import OmniStageConfig
        
        config = OmniStageConfig(
            base_model="/path/to/model",
            learning_rate=2e-6,
            max_samples=100,
        )
        
        assert config.base_model == "/path/to/model"
        assert config.learning_rate == 2e-6
        assert config.max_samples == 100


class TestOmniInferenceImport:
    """Test Omni inference module imports."""
    
    def test_import_inference(self):
        """Test OmniInference can be imported."""
        from src.omni.inference import OmniInference
        assert OmniInference is not None
    
    def test_import_generation_config(self):
        """Test GenerationConfig can be imported."""
        from src.omni.inference import GenerationConfig
        assert GenerationConfig is not None


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Test default generation config."""
        from src.omni.inference import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.do_sample is True
    
    def test_custom_values(self):
        """Test custom generation config."""
        from src.omni.inference import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.5,
            stream=True,
        )
        
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.5
        assert config.stream is True


@pytest.mark.omni
class TestOmniTrainingIntegration:
    """Integration tests for Omni training (mocked)."""
    
    @pytest.fixture
    def omni_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    
    @pytest.fixture
    def sample_data_path(self):
        return Path("/mnt/e/data/datasets/kaist-ai_CoT-Collection/data/CoT_collection_en.json")
    
    def test_stage_setup_mocked(self):
        """Test Omni training stage setup (using real tiny model)."""
        from src.stages.stage_omni import OmniTrainingStage, OmniStageConfig
        from unittest.mock import patch, MagicMock
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        import torch

        config = OmniStageConfig(base_model="/fake/model")
        
        # Create a real tiny model
        model_config = GPT2Config(n_layer=1, n_head=4, n_embd=32, vocab_size=100)
        real_model = GPT2LMHeadModel(model_config)
        
        # Create a real-ish tokenizer (or mock sufficient parts)
        # Using MagicMock for tokenizer is usually safer/faster unless we need specific tokenization logic
        # But user asked for real models. Let's try to use a real tokenizer if easy, otherwise mock strictly.
        # Constructing a tokenizer from scratch is complex. Let's use a MagicMock that behaves like a tokenizer
        # or use a very simple one.
        mock_tokenizer = MagicMock(spec=GPT2Tokenizer)
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        
        with patch('src.omni.loader.OmniModelLoader.load_for_training', return_value=(real_model, mock_tokenizer)):
            stage = OmniTrainingStage(config)
            # Initially None
            assert stage.model is None
            
            # Run setup to load model
            stage.setup()
            
            assert stage.config.capability_name == "omni"
            assert stage.model is not None
            assert isinstance(stage.model, torch.nn.Module)
            # Verify it has parameters (it's a real model)
            assert len(list(stage.model.parameters())) > 0
    
    def test_prepare_data_mocked(self):
        """Test data preparation (mocked)."""
        from src.stages.stage_omni import OmniTrainingStage, OmniStageConfig
        from unittest.mock import MagicMock, patch
        from dataclasses import dataclass
        
        @dataclass
        class MockLoadResult:
            dataset = ["sample1", "sample2"]
            num_samples = 2
            format = "json"
            error = None
        
        config = OmniStageConfig(base_model="/fake/model", max_samples=5)
        stage = OmniTrainingStage(config)
        
        # Mock Path.exists to return True so we skip dynamic loading logic
        # and mock the universal loader
        with patch('src.data.universal_loader.load_dataset_universal', return_value=MockLoadResult()), \
             patch('pathlib.Path.exists', return_value=True):
            dataset = stage.prepare_data("/fake/data")
        
        assert dataset == ["sample1", "sample2"]


class TestOmniValidationPipeline:
    """Test Omni model in validation pipeline."""
    
    @pytest.fixture
    def omni_model_path(self):
        return Path("/mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4")
    
    def test_omni_model_detection_in_validation(self, omni_model_path):
        """Test validation script detects Omni model."""
        if not omni_model_path.exists():
            pytest.skip("Omni model not found")
        
        from src.omni.loader import OmniModelLoader
        
        assert OmniModelLoader.is_omni_model(omni_model_path)
