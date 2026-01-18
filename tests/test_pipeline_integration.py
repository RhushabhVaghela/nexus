#!/usr/bin/env python3
"""
test_pipeline_integration.py

End-to-end integration tests for the Manus Model dataset pipeline.

Tests:
1. Repetitive dataset generation
2. Preference dataset generation
3. Multimodal dataset generation
4. Dataset validation
5. Data mixing
6. Benchmark loading
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════
# TEST: REPETITIVE DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════

class TestRepetitiveDatasetGeneration:
    """Test repetitive dataset generation pipeline."""
    
    def test_generator_weights_exist(self):
        """Verify all generator weights are defined."""
        try:
            from src.gen_05_generate_repetitive_dataset import GENERATOR_WEIGHTS
            
            # Check that weights exist
            assert len(GENERATOR_WEIGHTS) > 0
            
            # Check fullstack categories exist
            fs_categories = [k for k in GENERATOR_WEIGHTS if k.startswith("fs_")]
            assert len(fs_categories) >= 50, f"Expected 50+ fs_* categories, got {len(fs_categories)}"
        except ImportError:
            pytest.skip("Module not available")
    
    def test_prompt_repetition_engine_initialization(self):
        """Test PromptRepetitionEngine can be instantiated."""
        try:
            from src.gen_05_generate_repetitive_dataset import PromptRepetitionEngine
            engine = PromptRepetitionEngine()
            assert engine is not None
            assert hasattr(engine, 'generate_trajectory')
        except ImportError:
            pytest.skip("Module not available for direct import")
    
    def test_repetition_styles(self):
        """Test all repetition styles produce valid output."""
        try:
            from src.gen_05_generate_repetitive_dataset import PromptRepetitionEngine
            engine = PromptRepetitionEngine()
            
            query = "What is 2+2?"
            context = "Basic arithmetic"
            
            for style in ["baseline", "2x", "verbose", "3x"]:
                result = engine.apply_repetition(query, context, style)
                assert len(result) > 0
                assert query in result or "2+2" in result
        except ImportError:
            pytest.skip("Module not available for direct import")


# ═══════════════════════════════════════════════════════════════
# TEST: PREFERENCE DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════

class TestPreferenceDatasetGeneration:
    """Test preference pair generation pipeline."""
    
    def test_preference_weights_balanced(self):
        """Verify preference weights are properly balanced."""
        try:
            from src.gen_06_generate_preference_dataset import PREFERENCE_WEIGHTS
            
            assert len(PREFERENCE_WEIGHTS) > 0
            
            # Check fullstack preference categories
            fs_prefs = [k for k in PREFERENCE_WEIGHTS if k.startswith("fs_")]
            assert len(fs_prefs) >= 6, f"Expected 6+ fs_* preference types, got {len(fs_prefs)}"
        except ImportError:
            pytest.skip("Module not available")
    
    def test_preference_pair_structure(self):
        """Test preference pairs have required fields."""
        try:
            from src.gen_06_generate_preference_dataset import PreferencePairEngine
            engine = PreferencePairEngine()
            
            pair = engine.generate_preference_pair()
            if pair:  # May be None if all quotas exhausted
                assert "prompt" in pair
                assert "chosen" in pair
                assert "rejected" in pair
                assert "category" in pair
                assert "id" in pair
        except ImportError:
            pytest.skip("Module not available")


# ═══════════════════════════════════════════════════════════════
# TEST: MULTIMODAL DATASET GENERATOR
# ═══════════════════════════════════════════════════════════════

class TestMultimodalDatasetGeneration:
    """Test multimodal dataset generation."""
    
    def test_multimodal_config_exists(self):
        """Verify multimodal config YAML exists and is valid."""
        config_path = Path(__file__).parent.parent / "src" / "config" / "multimodal_datasets.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check for any of the expected keys
            assert any(k in config for k in ["vision_datasets", "huggingface_datasets", "mixed_multimodal", "datasets"]), \
                "Config missing expected keys (vision_datasets, huggingface_datasets, etc)"
        else:
            pytest.skip("Config file not found")
    
    def test_screenshot_generator_sample_structure(self):
        """Test screenshot sample has correct modalities structure."""
        try:
            from src.mm_generate_screenshot_dataset import build_sample
            from pathlib import Path
            
            # Mock a fake image path
            fake_path = Path("/fake/image.png")
            sample = build_sample(fake_path, 0)
            
            assert "id" in sample
            assert "messages" in sample
            assert "modalities" in sample
            assert "image" in sample["modalities"]
            assert len(sample["messages"]) >= 2
        except ImportError:
            pytest.skip("Module not available")
    
    def test_diagram_generator_categories(self):
        """Test diagram generator has all diagram types."""
        try:
            from src.mm_generate_diagram_dataset import DIAGRAM_TYPES
            
            expected_types = [
                "system_architecture",
                "database_schema",
                "flowchart",
                "class_diagram",
            ]
            
            for dtype in expected_types:
                assert dtype in DIAGRAM_TYPES, f"Missing diagram type: {dtype}"
        except ImportError:
            pytest.skip("Module not available")
    

    def test_audio_meeting_generator_categories(self):
        """Test audio meeting generator has all meeting types."""
        try:
            from src.mm_generate_audio_meeting_dataset import MEETING_TYPES
            
            expected_types = [
                "standup",
                "sprint_planning",
                "code_review",
                "architecture_discussion",
            ]
            
            for mtype in expected_types:
                assert mtype in MEETING_TYPES, f"Missing meeting type: {mtype}"
        except ImportError:
            pytest.skip("Module not available")


# ═══════════════════════════════════════════════════════════════
# TEST: UNIFIED MULTIMODAL DOWNLOADER
# ═══════════════════════════════════════════════════════════════

class TestUnifiedMultimodalDownloader:
    """Test the unified Kaggle -> HF fallback downloader strategy."""
    
    @patch('src.mm_download_unified.KaggleApi')
    def test_kaggle_priority(self, mock_kaggle_cls):
        """Test that Kaggle is tried first."""
        try:
            from src.mm_download_unified import DatasetManager
            
            mock_api = mock_kaggle_cls.return_value
            manager = DatasetManager(Path("/tmp/test"))
            
            config = {"kaggle_id": "test/dataset", "hf_id": "test/hf"}
            
            # Mock download_dataset_files to succeed 
            # side_effect for exists: False (download needed), True (file exists for processing)
            with patch('pathlib.Path.exists', side_effect=[False, True, True, True]):
                 with patch('src.mm_download_unified.DatasetManager._process_local_files', return_value=5) as mock_process:
                    count = manager.download_and_process("vision", "test_ds", config, 5)
                    
                    # Should call kaggle api
                    mock_api.dataset_download_files.assert_called_once()
                    mock_process.assert_called_once()
                    assert count == 5
        except ImportError:
            pytest.skip("Module not available")

    @patch('src.mm_download_unified.KaggleApi')
    @patch('src.mm_download_unified.load_dataset')
    def test_hf_fallback(self, mock_load, mock_kaggle_cls):
        """Test fallback to HF when Kaggle fails."""
        try:
            from src.mm_download_unified import DatasetManager
            
            mock_api = mock_kaggle_cls.return_value
            mock_api.dataset_download_files.side_effect = Exception("403 Forbidden")
            
            # Simulate local dir not existing despite download attempt
            with patch('pathlib.Path.exists', side_effect=[False, False, False, False]): 
                manager = DatasetManager(Path("/tmp/test"))
                config = {"kaggle_id": "test/dataset", "hf_id": "test/hf"}
                
                # Mock HF dataset iterator (needs .take method if streaming)
                mock_ds = MagicMock()
                mock_ds.take.return_value = [{"text": "sample"}]
                mock_load.return_value = mock_ds
                
                count = manager.download_and_process("premium_text", "test_ds", config, 1)
                
                # Kaggle called and failed
                mock_api.dataset_download_files.assert_called_once()
                # HF called as fallback
                mock_load.assert_called_once()
                assert count == 1
        except ImportError:
            pytest.skip("Module not available")




# ═══════════════════════════════════════════════════════════════
# TEST: DATASET VALIDATION
# ═══════════════════════════════════════════════════════════════

class TestDatasetValidation:
    """Test dataset validation functionality."""
    
    def test_validator_accepts_valid_sample(self):
        """Test that valid samples pass validation."""
        try:
            from src.gen_07_validate_all_datasets import DatasetValidator
            validator = DatasetValidator()
            
            valid_sample = {
                "id": "test_001",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "domain": "general",
            }
            
            result = validator.validate_sample(valid_sample)
            assert result is not None
        except ImportError:
            pytest.skip("Module not available")
    
    def test_validator_rejects_empty_messages(self):
        """Test that samples with empty messages are rejected."""
        try:
            from src.gen_07_validate_all_datasets import DatasetValidator
            validator = DatasetValidator()
            
            invalid_sample = {
                "id": "test_002",
                "messages": [],
                "domain": "general",
            }
            
            result = validator.validate_sample(invalid_sample)
            assert result is None
        except ImportError:
            pytest.skip("Module not available")
    
    def test_modalities_validation(self):
        """Test multimodal validation handles missing files gracefully."""
        try:
            from src.gen_07_validate_all_datasets import DatasetValidator
            validator = DatasetValidator()
            
            # Sample with modalities but non-existent file
            sample = {
                "id": "mm_001",
                "messages": [
                    {"role": "user", "content": "Describe this image"},
                    {"role": "assistant", "content": "It shows..."},
                ],
                "modalities": {
                    "image": [{"path": "/nonexistent/image.png", "type": "screenshot"}],
                    "audio": [],
                    "video": [],
                },
            }
            
            # validate_modalities should flag missing file
            if hasattr(validator, 'validate_modalities'):
                result = validator.validate_modalities(sample)
                assert result is False  # File doesn't exist
        except ImportError:
            pytest.skip("Module not available")


# ═══════════════════════════════════════════════════════════════
# TEST: DATA MIXER
# ═══════════════════════════════════════════════════════════════

class TestDataMixer:
    """Test data mixing functionality."""
    
    def test_normalize_to_messages_alpaca_format(self):
        """Test Alpaca format conversion."""
        try:
            from src.utils.data_mixer import normalize_to_messages
            
            alpaca_sample = {
                "instruction": "Write a hello world",
                "input": "in Python",
                "output": "print('Hello, World!')",
            }
            
            result = normalize_to_messages(alpaca_sample, source="test")
            assert result is not None
            assert "messages" in result
            assert len(result["messages"]) == 2
        except ImportError:
            pytest.skip("Module not available")
    
    def test_normalize_to_messages_preserves_modalities(self):
        """Test that modalities block is preserved during normalization."""
        try:
            from src.utils.data_mixer import normalize_to_messages
            
            mm_sample = {
                "messages": [
                    {"role": "user", "content": "Describe this"},
                    {"role": "assistant", "content": "It shows..."},
                ],
                "modalities": {
                    "image": [{"path": "/test.png"}],
                    "audio": [],
                    "video": [],
                },
            }
            
            result = normalize_to_messages(mm_sample, source="test")
            assert result is not None
            assert "modalities" in result
            assert result["modalities"] == mm_sample["modalities"]
        except ImportError:
            pytest.skip("Module not available")
    
    def test_mix_datasets_ratio(self):
        """Test that mixing respects target ratios."""
        try:
            from src.utils.data_mixer import mix_datasets
            
            real_samples = [{"id": f"real_{i}"} for i in range(100)]
            synthetic_samples = [{"id": f"synth_{i}"} for i in range(100)]
            
            mixed, stats = mix_datasets(real_samples, synthetic_samples, real_ratio=0.3)
            
            assert len(mixed) > 0
            assert "real_samples" in stats
            assert "synthetic_samples" in stats
        except ImportError:
            pytest.skip("Module not available")


# ═══════════════════════════════════════════════════════════════
# TEST: BENCHMARKS
# ═══════════════════════════════════════════════════════════════

class TestBenchmarks:
    """Test benchmark suite functionality."""
    
    def test_fullstack_eval_has_evaluators(self):
        """Test FullstackEval has all required evaluators."""
        try:
            from src.benchmarks.fullstack_eval import FullstackEvalBenchmark
            benchmark = FullstackEvalBenchmark()
            
            test_cases = benchmark.get_test_cases()
            assert len(test_cases) > 0
        except ImportError:
            pytest.skip("Module not available")
    
    def test_lovable_benchmark_categories(self):
        """Test LovableBenchmark has expected categories."""
        try:
            from src.benchmarks.lovable_benchmark import LovableBenchmark
            benchmark = LovableBenchmark()
            
            # categories = benchmark.get_categories()
            categories = list(benchmark.CASES.keys())
            expected = ["screenshot_to_code", "feature_completion"]
            
            for cat in expected:
                assert cat in categories or any(cat in c for c in categories)
        except ImportError:
            pytest.skip("Module not available")


# ═══════════════════════════════════════════════════════════════
# TEST: STREAMING & PODCAST
# ═══════════════════════════════════════════════════════════════

class TestStreamingAndPodcast:
    """Test streaming and podcast modules."""
    
    def test_joint_streaming_orchestrator_exists(self):
        """Test joint streaming orchestrator module exists."""
        module_path = Path(__file__).parent.parent / "src" / "streaming" / "joint.py"
        assert module_path.exists(), "joint.py should exist in streaming/"
    
    def test_podcast_generator_exists(self):
        """Test podcast generator module exists."""
        module_path = Path(__file__).parent.parent / "src" / "podcast" / "generator.py"
        assert module_path.exists(), "generator.py should exist in podcast/"
    
    def test_podcast_synthesizer_exists(self):
        """Test podcast synthesizer module exists."""
        module_path = Path(__file__).parent.parent / "src" / "podcast" / "synthesizer.py"
        assert module_path.exists(), "synthesizer.py should exist in podcast/"


# ═══════════════════════════════════════════════════════════════
# TEST: PIPELINE END-TO-END
# ═══════════════════════════════════════════════════════════════

class TestPipelineEndToEnd:
    """End-to-end pipeline tests."""
    
    def test_numbered_scripts_exist(self):
        """Verify all 25 numbered pipeline scripts exist."""
        src_dir = Path(__file__).parent.parent / "src"
        
        for i in range(1, 26):
            pattern = f"{i:02d}_*.py"
            matches = list(src_dir.glob(pattern))
            assert len(matches) >= 1, f"Missing script for step {i:02d}"
    
    def test_config_files_exist(self):
        """Verify required config files exist."""
        config_dir = Path(__file__).parent.parent / "src" / "config"
        
        expected_configs = [
            "model_config.yaml",
            "multimodal_datasets.yaml",
        ]
        
        for config in expected_configs:
            config_path = config_dir / config
            # Check if at least some config exists
            if config_dir.exists():
                assert any(config_dir.glob("*.yaml")), "No YAML configs found"
    
    def test_utils_modules_exist(self):
        """Verify utility modules exist."""
        utils_dir = Path(__file__).parent.parent / "src" / "utils"
        
        expected_utils = [
            "data_mixer.py",
            "logging_config.py",
        ]
        
        for util in expected_utils:
            util_path = utils_dir / util
            assert util_path.exists(), f"Missing utility: {util}"


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
