"""
Integration tests for Omni model training (MOCKED).
Tests training stage, validation, and inference propagation.
"""

import pytest
import sys
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stages.stage_omni import OmniTrainingStage, OmniStageConfig

class TestOmniStageImport:
    def test_import_stage(self):
        from src.stages.stage_omni import OmniTrainingStage
        assert OmniTrainingStage is not None
    
    def test_import_config(self):
        from src.stages.stage_omni import OmniStageConfig
        assert OmniStageConfig is not None

class TestOmniStageConfig:
    def test_default_config(self):
        config = OmniStageConfig()
        assert config.freeze_talker is True
        assert config.learning_rate == 1e-6

class TestOmniTrainingIntegration:
    def test_stage_setup_mocked(self, real_text_model, real_text_tokenizer):
        """Test Omni training stage setup with mocked components."""
        config = OmniStageConfig(base_model_path="/fake/model")
        
        # We patch the loader to return our fixtures from conftest (which are mocks by default)
        with patch('src.omni.loader.OmniModelLoader.load_for_training', return_value=(real_text_model, real_text_tokenizer)):
            stage = OmniTrainingStage(config)
            stage.setup()
            
            assert stage.model is not None
            assert hasattr(stage.model, 'parameters')

    def test_prepare_data_mocked(self):
        """Test data preparation with mocked universal loader."""
        @dataclass
        class MockLoadResult:
            dataset = ["sample1", "sample2"]
            num_samples = 2
            format = "json"
            error = None
        
        config = OmniStageConfig(base_model_path="/fake/model", sample_size=5)
        stage = OmniTrainingStage(config)
        
        with patch('src.data.universal_loader.load_dataset_universal', return_value=MockLoadResult()), \
             patch('pathlib.Path.exists', return_value=True):
            dataset = stage.prepare_data("/fake/data")
        
        assert dataset == ["sample1", "sample2"]

class TestOmniValidationPipeline:
    def test_omni_model_detection_mocked(self):
        """Test validation script detects Omni model via path name."""
        from src.omni.loader import OmniModelLoader
        # Path containing "omni" should return True even if mocked
        assert OmniModelLoader.is_omni_model("/path/to/my-omni-model") is True
