"""
Integration tests for config files.
Tests the separate config files: encoders.yaml, decoders.yaml, datasets.yaml, outputs.yaml
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEncodersConfig:
    """Test encoders.yaml configuration."""
    
    def test_encoders_config_loads(self, encoders_config):
        """Test encoders config loads successfully."""
        assert encoders_config is not None
    
    def test_encoders_has_vision(self, encoders_config):
        """Test config has vision encoder."""
        assert "encoders" in encoders_config
        assert "vision" in encoders_config["encoders"]
        assert "default" in encoders_config["encoders"]["vision"]
    
    def test_encoders_has_audio(self, encoders_config):
        """Test config has audio encoders."""
        assert "audio_input" in encoders_config["encoders"]
        assert "audio_output" in encoders_config["encoders"]
    
    def test_encoder_paths_are_local(self, encoders_config):
        """Test encoder paths use local paths."""
        vision_path = encoders_config["encoders"]["vision"]["default"]
        assert "/mnt/e/data" in vision_path
        
        audio_path = encoders_config["encoders"]["audio_input"]["default"]
        assert "/mnt/e/data" in audio_path
    
    def test_base_models_section(self, encoders_config):
        """Test base_models section exists."""
        assert "base_models" in encoders_config
        assert "default" in encoders_config["base_models"]


class TestDecodersConfig:
    """Test decoders.yaml configuration."""
    
    def test_decoders_config_loads(self, decoders_config):
        """Test decoders config loads successfully."""
        assert decoders_config is not None
    
    def test_has_vision_output(self, decoders_config):
        """Test config has vision output decoder."""
        assert "decoders" in decoders_config
        assert "vision_output" in decoders_config["decoders"]
    
    def test_has_video_output(self, decoders_config):
        """Test config has video output decoder."""
        assert "video_output" in decoders_config["decoders"]
    
    def test_decoder_has_type(self, decoders_config):
        """Test decoder entries have type field."""
        vision = decoders_config["decoders"]["vision_output"]
        assert "type" in vision
        assert vision["type"] == "diffusion"


class TestDatasetsConfig:
    """Test datasets.yaml configuration."""
    
    def test_datasets_config_loads(self, datasets_config):
        """Test datasets config loads successfully."""
        assert datasets_config is not None
    
    def test_has_datasets_section(self, datasets_config):
        """Test config has datasets section."""
        assert "datasets" in datasets_config
    
    def test_has_text_capabilities(self, datasets_config):
        """Test config has text capability datasets."""
        ds = datasets_config["datasets"]
        assert "cot" in ds
        assert "reasoning" in ds
        assert "thinking" in ds
        assert "tools" in ds
    
    def test_capability_has_primary(self, datasets_config):
        """Test each capability has primary dataset."""
        for cap in ["cot", "reasoning", "tools"]:
            assert "primary" in datasets_config["datasets"][cap]
    
    def test_has_paths_section(self, datasets_config):
        """Test config has paths section."""
        assert "paths" in datasets_config
        assert "datasets" in datasets_config["paths"]


class TestOutputsConfig:
    """Test outputs.yaml configuration."""
    
    def test_outputs_config_loads(self, outputs_config):
        """Test outputs config loads successfully."""
        assert outputs_config is not None
    
    def test_has_output_root(self, outputs_config):
        """Test config has output root."""
        assert "output" in outputs_config
        assert "root" in outputs_config["output"]
        assert "/mnt/e/data/output" in outputs_config["output"]["root"]
    
    def test_has_checkpoints(self, outputs_config):
        """Test config has checkpoints section."""
        assert "checkpoints" in outputs_config["output"]
        assert "path" in outputs_config["output"]["checkpoints"]
    
    def test_has_validation(self, outputs_config):
        """Test config has validation section."""
        assert "validation" in outputs_config["output"]
    
    def test_has_exports(self, outputs_config):
        """Test config has exports section."""
        assert "exports" in outputs_config["output"]
        assert "formats" in outputs_config["output"]["exports"]
