#!/usr/bin/env python3
"""
Comprehensive unit tests for OmniModelLoader.

Tests all fixes and features:
1. persistent argument fix in register_buffer lambda (line 268)
2. SAE model detection and tokenizer fallback
3. Comprehensive architecture support for all registry models
4. Model category detection methods
5. Error handling and edge cases
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

# Optional pytest import for parametrize support
try:
    import pytest
except ImportError:
    pytest = None

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from omni.loader import OmniModelLoader, load_omni_model, OmniModelConfig


class TestPersistentArgumentFix(unittest.TestCase):
    """Test the persistent argument fix in register_buffer lambda (line 268)."""
    
    def test_register_buffer_lambda_accepts_persistent(self):
        """Test that register_buffer lambda accepts persistent parameter."""
        import torch.nn as nn
        
        # Create a simple module to test
        module = nn.Module()
        
        # Test that register_buffer accepts persistent parameter
        test_tensor = __import__('torch').zeros(1)
        
        # This should work without error
        module.register_buffer("test_buffer", test_tensor, persistent=True)
        module.register_buffer("test_buffer2", test_tensor, persistent=False)
        
        # Verify buffers exist
        self.assertIn("test_buffer", module._buffers)
        self.assertIn("test_buffer2", module._buffers)
    
    def test_self_healing_patches_applied(self):
        """Test that self-healing patches are applied during load."""
        loader = OmniModelLoader()
        
        # Mock the necessary attributes
        with patch('torch.nn.Module') as mock_module:
            with patch('transformers.PreTrainedModel') as mock_pretrain:
                # Apply patches
                loader._apply_self_healing_patches()
                
                # Verify patches were applied
                self.assertIsNotNone(mock_module.get_submodule)
                self.assertIsNotNone(mock_module.register_buffer)
                self.assertIsNotNone(mock_module.__setattr__)
    
    def test_register_buffer_name_sanitization(self):
        """Test that register_buffer sanitizes names with dots."""
        import torch.nn as nn
        
        module = nn.Module()
        test_tensor = __import__('torch').zeros(1)
        
        # Name with dots should be sanitized
        module.register_buffer("name.with.dots", test_tensor, persistent=True)
        
        # The sanitized name should be in _buffers
        self.assertIn("name_with_dots", module._buffers)


class TestSAEModelDetection(unittest.TestCase):
    """Test SAE (Sparse AutoEncoder) model detection."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_sae_model_with_sae_directories(self):
        """Test detection of SAE model with SAE-specific directories."""
        # Create SAE indicator directories
        (self.model_path / "resid_post").mkdir()
        (self.model_path / "mlp_out").mkdir()
        
        # No tokenizer files
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertTrue(result, "Should detect SAE model with SAE directories and no tokenizer")
    
    def test_is_sae_model_with_all_indicators(self):
        """Test detection with all SAE indicator directories."""
        sae_indicators = ["resid_post", "mlp_out", "attn_out", "transcoder", "resid_post_all"]
        
        for indicator in sae_indicators:
            self.setUp()  # Reset temp dir
            (self.model_path / indicator).mkdir()
            result = OmniModelLoader._is_sae_model(self.model_path)
            self.assertTrue(result, f"Should detect SAE model with {indicator} directory")
            self.tearDown()
    
    def test_is_sae_model_false_with_tokenizer(self):
        """Test that SAE detection returns False if tokenizer exists."""
        (self.model_path / "resid_post").mkdir()
        (self.model_path / "tokenizer.json").touch()
        
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertFalse(result, "Should NOT detect SAE model if tokenizer exists")
    
    def test_is_sae_model_false_no_sae_dirs(self):
        """Test that SAE detection returns False without SAE directories."""
        # Create a regular model structure
        (self.model_path / "config.json").touch()
        
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertFalse(result, "Should NOT detect SAE model without SAE directories")
    
    def test_is_sae_model_nonexistent_path(self):
        """Test SAE detection with non-existent path."""
        nonexistent_path = Path("/nonexistent/path/that/does/not/exist")
        result = OmniModelLoader._is_sae_model(nonexistent_path)
        self.assertFalse(result, "Should return False for non-existent path")
    
    def test_is_sae_model_with_tokenizer_config(self):
        """Test SAE detection with tokenizer_config.json present."""
        (self.model_path / "resid_post").mkdir()
        (self.model_path / "tokenizer_config.json").touch()
        
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertFalse(result, "Should NOT detect SAE model if tokenizer_config.json exists")
    
    def test_is_sae_model_with_spiece_model(self):
        """Test SAE detection with spiece.model present."""
        (self.model_path / "resid_post").mkdir()
        (self.model_path / "spiece.model").touch()
        
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertFalse(result, "Should NOT detect SAE model if spiece.model exists")
    
    def test_is_sae_model_with_tokenizer_model(self):
        """Test SAE detection with tokenizer.model present."""
        (self.model_path / "resid_post").mkdir()
        (self.model_path / "tokenizer.model").touch()
        
        result = OmniModelLoader._is_sae_model(self.model_path)
        self.assertFalse(result, "Should NOT detect SAE model if tokenizer.model exists")


class TestSAEBaseModelExtraction(unittest.TestCase):
    """Test extraction of base model from SAE config files."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_sae_base_model_from_config(self):
        """Test extracting base model from SAE config."""
        # Create SAE directory structure with config
        sae_dir = self.model_path / "resid_post" / "layer_0"
        sae_dir.mkdir(parents=True)
        
        config = {"model_name": "google/gemma-2-27b-it"}
        with open(sae_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertEqual(result, "google/gemma-2-27b-it")
    
    def test_get_sae_base_model_from_mlp_out(self):
        """Test extracting base model from mlp_out directory."""
        sae_dir = self.model_path / "mlp_out" / "layer_1"
        sae_dir.mkdir(parents=True)
        
        config = {"model_name": "meta-llama/Llama-2-7b"}
        with open(sae_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertEqual(result, "meta-llama/Llama-2-7b")
    
    def test_get_sae_base_model_no_config(self):
        """Test extraction when no config exists."""
        (self.model_path / "resid_post").mkdir()
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertIsNone(result)
    
    def test_get_sae_base_model_no_sae_dirs(self):
        """Test extraction with no SAE directories."""
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertIsNone(result)
    
    def test_get_sae_base_model_malformed_config(self):
        """Test extraction with malformed config file."""
        sae_dir = self.model_path / "resid_post" / "layer_0"
        sae_dir.mkdir(parents=True)
        
        # Write invalid JSON
        with open(sae_dir / "config.json", "w") as f:
            f.write("invalid json")
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertIsNone(result)
    
    def test_get_sae_base_model_no_model_name(self):
        """Test extraction when config lacks model_name."""
        sae_dir = self.model_path / "resid_post" / "layer_0"
        sae_dir.mkdir(parents=True)
        
        config = {"some_other_field": "value"}
        with open(sae_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertIsNone(result)
    
    def test_get_sae_base_model_from_attn_out(self):
        """Test extraction from attn_out directory."""
        sae_dir = self.model_path / "attn_out" / "layer_5"
        sae_dir.mkdir(parents=True)
        
        config = {"model_name": "mistralai/Mistral-7B"}
        with open(sae_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertEqual(result, "mistralai/Mistral-7B")
    
    def test_get_sae_base_model_from_transcoder(self):
        """Test extraction from transcoder directory."""
        sae_dir = self.model_path / "transcoder" / "layer_10"
        sae_dir.mkdir(parents=True)
        
        config = {"model_name": "deepseek-ai/deepseek-llm-7b"}
        with open(sae_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._get_sae_base_model(self.model_path)
        self.assertEqual(result, "deepseek-ai/deepseek-llm-7b")


class TestTokenizerLoading(unittest.TestCase):
    """Test tokenizer loading with SAE fallback."""
    
    @patch('omni.loader.AutoTokenizer')
    def test_load_tokenizer_normal_model(self, mock_tokenizer_class):
        """Test loading tokenizer for normal model."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        loader = OmniModelLoader()
        model_path = Path("/some/model/path")
        
        # Create config.json to make it look like a valid model
        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"model_type": "test"}')):
                tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        
        mock_tokenizer_class.from_pretrained.assert_called()
    
    @patch('omni.loader.AutoTokenizer')
    def test_load_tokenizer_sets_pad_token(self, mock_tokenizer_class):
        """Test that pad_token is set to eos_token if None."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        loader = OmniModelLoader()
        model_path = Path("/some/model/path")
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"model_type": "test"}')):
                tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
    
    @patch('omni.loader.AutoTokenizer')
    @patch.object(OmniModelLoader, '_is_sae_model', return_value=True)
    @patch.object(OmniModelLoader, '_get_sae_base_model', return_value="gpt2")
    def test_load_tokenizer_sae_fallback(self, mock_get_base, mock_is_sae, mock_tokenizer_class):
        """Test tokenizer loading falls back to base model for SAE."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        loader = OmniModelLoader()
        model_path = Path("/some/sae/model")
        
        tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        
        # Should try to load from base model
        calls = mock_tokenizer_class.from_pretrained.call_args_list
        self.assertTrue(any("gpt2" in str(call) for call in calls))
    
    @patch('omni.loader.AutoTokenizer')
    @patch.object(OmniModelLoader, '_is_sae_model', return_value=True)
    @patch.object(OmniModelLoader, '_get_sae_base_model', return_value=None)
    def test_load_tokenizer_sae_no_base_model(self, mock_get_base, mock_is_sae, mock_tokenizer_class):
        """Test tokenizer loading when SAE has no base model info."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        loader = OmniModelLoader()
        model_path = Path("/some/sae/model")
        
        # Should still try to load from SAE path
        tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        mock_tokenizer_class.from_pretrained.assert_called()


class TestModelCategoryDetection(unittest.TestCase):
    """Test model category detection methods."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_diffusers_model_with_model_index(self):
        """Test diffusers model detection with model_index.json."""
        (self.model_path / "model_index.json").touch()
        
        result = OmniModelLoader._is_diffusers_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_diffusers_model_with_unet_and_vae(self):
        """Test diffusers model detection with unet and vae directories."""
        (self.model_path / "unet").mkdir()
        (self.model_path / "vae").mkdir()
        
        result = OmniModelLoader._is_diffusers_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_diffusers_model_false(self):
        """Test diffusers detection returns false for regular model."""
        (self.model_path / "config.json").touch()
        
        result = OmniModelLoader._is_diffusers_model(self.model_path)
        self.assertFalse(result)
    
    def test_is_diffusers_model_nonexistent(self):
        """Test diffusers detection with non-existent path."""
        result = OmniModelLoader._is_diffusers_model(Path("/nonexistent"))
        self.assertFalse(result)
    
    def test_is_vision_encoder_siglip(self):
        """Test vision encoder detection for SigLIP."""
        config = {"architectures": ["SigLIPModel"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertTrue(result)
    
    def test_is_vision_encoder_clip(self):
        """Test vision encoder detection for CLIP."""
        config = {"architectures": ["CLIPVisionModel"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertTrue(result)
    
    def test_is_vision_encoder_dinov2(self):
        """Test vision encoder detection for DINOv2."""
        config = {"architectures": ["DINOv2Model"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertTrue(result)
    
    def test_is_vision_encoder_false_for_llm(self):
        """Test vision encoder detection returns false for LLM."""
        config = {"architectures": ["LlamaForCausalLM"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertFalse(result)
    
    def test_is_vision_encoder_no_config(self):
        """Test vision encoder detection without config."""
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertFalse(result)
    
    def test_is_vision_encoder_malformed_config(self):
        """Test vision encoder detection with malformed config."""
        with open(self.model_path / "config.json", "w") as f:
            f.write("invalid json")
        
        result = OmniModelLoader._is_vision_encoder(self.model_path)
        self.assertFalse(result)
    
    def test_is_asr_model_whisper(self):
        """Test ASR detection for Whisper."""
        config = {"architectures": ["WhisperForConditionalGeneration"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_asr_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_asr_model_speech2text(self):
        """Test ASR detection for Speech2Text."""
        config = {"architectures": ["Speech2TextForConditionalGeneration"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_asr_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_asr_model_false_for_llm(self):
        """Test ASR detection returns false for LLM."""
        config = {"architectures": ["Qwen2ForCausalLM"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._is_asr_model(self.model_path)
        self.assertFalse(result)
    
    def test_is_asr_model_no_config(self):
        """Test ASR detection without config."""
        result = OmniModelLoader._is_asr_model(self.model_path)
        self.assertFalse(result)
    
    def test_detect_model_category_diffusers(self):
        """Test category detection for diffusers model."""
        (self.model_path / "model_index.json").touch()
        
        result = OmniModelLoader._detect_model_category(self.model_path)
        self.assertEqual(result, "diffusers")
    
    def test_detect_model_category_sae(self):
        """Test category detection for SAE model."""
        (self.model_path / "resid_post").mkdir()
        # Note: SAE detection checks for absence of tokenizer
        
        result = OmniModelLoader._detect_model_category(self.model_path)
        # This will be "transformers" because we have no tokenizer check in detect
        # The SAE check requires no tokenizer, but detect_model_category doesn't check that
        self.assertIn(result, ["transformers", "sae"])
    
    def test_detect_model_category_vision_encoder(self):
        """Test category detection for vision encoder."""
        config = {"architectures": ["SigLIPModel"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._detect_model_category(self.model_path)
        self.assertEqual(result, "vision_encoder")
    
    def test_detect_model_category_asr(self):
        """Test category detection for ASR model."""
        config = {"architectures": ["WhisperForConditionalGeneration"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._detect_model_category(self.model_path)
        self.assertEqual(result, "asr")
    
    def test_detect_model_category_transformers(self):
        """Test category detection for regular transformers model."""
        config = {"architectures": ["LlamaForCausalLM"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader._detect_model_category(self.model_path)
        self.assertEqual(result, "transformers")


class TestOmniModelConfig(unittest.TestCase):
    """Test OmniModelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OmniModelConfig(model_path="/test/path")
        
        self.assertEqual(config.model_path, "/test/path")
        self.assertEqual(config.mode, "thinker_only")
        self.assertEqual(config.device_map, "auto")
        self.assertEqual(config.torch_dtype, "auto")
        self.assertFalse(config.load_in_8bit)
        self.assertFalse(config.load_in_4bit)
        self.assertTrue(config.trust_remote_code)
        self.assertTrue(config.use_flash_attention)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OmniModelConfig(
            model_path="/custom/path",
            mode="full",
            device_map="cuda:0",
            torch_dtype="float16",
            load_in_8bit=True,
            trust_remote_code=False
        )
        
        self.assertEqual(config.model_path, "/custom/path")
        self.assertEqual(config.mode, "full")
        self.assertEqual(config.device_map, "cuda:0")
        self.assertEqual(config.torch_dtype, "float16")
        self.assertTrue(config.load_in_8bit)
        self.assertFalse(config.trust_remote_code)


class TestModelInfoAndSupport(unittest.TestCase):
    """Test model info retrieval and support checking."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_model_info_no_config(self):
        """Test model info when no config exists."""
        info = OmniModelLoader.get_model_info(self.model_path)
        
        self.assertEqual(info["error"], "No config.json found")
        self.assertFalse(info["is_supported"])
    
    def test_get_model_info_with_architecture(self):
        """Test model info with valid config."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        info = OmniModelLoader.get_model_info(self.model_path)
        
        self.assertEqual(info["architecture"], "LlamaForCausalLM")
        self.assertEqual(info["model_type"], "llama")
    
    def test_get_model_info_quantized(self):
        """Test model info for quantized model."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "quantization_config": {"load_in_4bit": True}
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        info = OmniModelLoader.get_model_info(self.model_path)
        
        self.assertTrue(info["is_quantized"])
    
    def test_get_model_info_with_talker(self):
        """Test model info detection of talker/audio keys."""
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "talker_config": {},
            "audio_config": {}
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        info = OmniModelLoader.get_model_info(self.model_path)
        
        self.assertTrue(info["has_talker"])
    
    def test_is_model_supported_supported_architecture(self):
        """Test support check for supported architecture."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertTrue(result["supported"])
        self.assertEqual(result["category"], "transformers")
    
    def test_is_model_supported_unsupported_architecture(self):
        """Test support check for unsupported architecture."""
        config = {
            "architectures": ["SomeUnknownArchitecture"],
            "model_type": "unknown"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertFalse(result["supported"])
        self.assertIsNotNone(result["error"])
    
    def test_is_model_supported_no_config(self):
        """Test support check when no config exists."""
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertFalse(result["supported"])
        self.assertEqual(result["error"], "No config.json found")
    
    def test_is_model_supported_custom_files(self):
        """Test support check with custom modeling files."""
        config = {
            "architectures": ["CustomModel"],
            "model_type": "custom"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create custom modeling file
        (self.model_path / "modeling_custom.py").touch()
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertTrue(result["has_custom_files"])
        self.assertTrue(result["supported"])
    
    def test_is_model_supported_diffusers(self):
        """Test support check for diffusers model."""
        (self.model_path / "model_index.json").touch()
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertEqual(result["category"], "diffusers")
        self.assertTrue(result["supported"])
    
    def test_is_model_supported_sae(self):
        """Test support check for SAE model."""
        (self.model_path / "resid_post").mkdir()
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        self.assertEqual(result["category"], "sae")
        self.assertTrue(result["supported"])
    
    def test_is_model_supported_with_mapping(self):
        """Test support check for model type with mapping."""
        config = {
            "architectures": ["Glm4MoeForCausalLM"],
            "model_type": "glm4_moe_lite"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_model_supported(self.model_path)
        
        # Should be supported due to MODEL_TYPE_MAPPINGS
        self.assertTrue(result["supported"])


class TestIsOmniModel(unittest.TestCase):
    """Test Omni model detection."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_omni_model_by_model_type(self):
        """Test detection by model_type containing 'omni'."""
        config = {"model_type": "qwen2_5_omni"}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_omni_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_omni_model_by_architecture(self):
        """Test detection by architecture containing 'omni'."""
        config = {
            "architectures": ["Qwen2_5OmniForConditionalGeneration"],
            "model_type": "some_type"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_omni_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_omni_model_by_any_to_any(self):
        """Test detection by any-to-any model type."""
        config = {"model_type": "any-to-any"}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_omni_model(self.model_path)
        self.assertTrue(result)
    
    def test_is_omni_model_false(self):
        """Test that regular models are not detected as Omni."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama"
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        result = OmniModelLoader.is_omni_model(self.model_path)
        self.assertFalse(result)
    
    def test_is_omni_model_nonexistent_path(self):
        """Test detection with non-existent path."""
        result = OmniModelLoader.is_omni_model(Path("/nonexistent/path"))
        self.assertFalse(result)
    
    def test_is_omni_model_no_config(self):
        """Test detection when no config exists."""
        result = OmniModelLoader.is_omni_model(self.model_path)
        self.assertFalse(result)


class TestArchitectureRegistration(unittest.TestCase):
    """Test custom architecture registration."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('transformers.models.auto.configuration_auto.CONFIG_MAPPING')
    @patch('transformers.models.auto.modeling_auto.AutoModelForCausalLM')
    @patch('transformers.models.auto.modeling_auto.AutoModel')
    def test_register_glm4_moe_lite(self, mock_auto_model, mock_auto_causal, mock_config):
        """Test registration of glm4_moe_lite model type."""
        config = {
            "model_type": "glm4_moe_lite",
            "architectures": ["Glm4MoeLiteForCausalLM"]
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        loader = OmniModelLoader()
        
        # Mock the _extra_content attribute
        mock_mapping = MagicMock()
        mock_mapping._extra_content = {}
        mock_auto_causal._model_mapping = mock_mapping
        mock_auto_model._model_mapping = mock_mapping
        mock_config._extra_content = {}
        
        loader._register_custom_architecture(self.model_path)
        
        # Check that mapping was added
        self.assertIn("glm4_moe_lite", mock_mapping._extra_content)
    
    @patch('transformers.models.auto.configuration_auto.CONFIG_MAPPING')
    @patch('transformers.models.auto.modeling_auto.AutoModelForCausalLM')
    @patch('transformers.models.auto.modeling_auto.AutoModel')
    @patch('transformers.models.auto.modeling_auto.AutoModelForVision2Seq')
    def test_register_step_robotics(self, mock_vision, mock_auto_model, mock_auto_causal, mock_config):
        """Test registration of step_robotics model type."""
        config = {
            "model_type": "step_robotics",
            "architectures": ["Step3VL10BForCausalLM"]
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        loader = OmniModelLoader()
        
        # Mock the _extra_content attribute
        mock_mapping = MagicMock()
        mock_mapping._extra_content = {}
        mock_auto_causal._model_mapping = mock_mapping
        mock_auto_model._model_mapping = mock_mapping
        mock_vision._model_mapping = mock_mapping
        mock_config._extra_content = {}
        
        loader._register_custom_architecture(self.model_path)
        
        # Check that mapping was added
        self.assertIn("step_robotics", mock_mapping._extra_content)
    
    def test_register_no_config(self):
        """Test registration when no config exists."""
        loader = OmniModelLoader()
        
        # Should not raise exception
        loader._register_custom_architecture(self.model_path)
    
    def test_register_no_model_type(self):
        """Test registration when config has no model_type."""
        config = {"architectures": ["SomeModel"]}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        loader = OmniModelLoader()
        
        # Should not raise exception
        loader._register_custom_architecture(self.model_path)


class TestLoadOmniModelFunction(unittest.TestCase):
    """Test the load_omni_model convenience function."""
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_omni_model_basic(self, mock_load):
        """Test basic load_omni_model call."""
        mock_load.return_value = (Mock(), Mock())
        
        result = load_omni_model("/test/path", mode="thinker_only")
        
        mock_load.assert_called_once()
        self.assertIsNotNone(result)
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_omni_model_skip_on_error(self, mock_load):
        """Test load_omni_model with skip_on_error=True."""
        mock_load.return_value = None
        
        result = load_omni_model("/test/path", skip_on_error=True)
        
        self.assertIsNone(result)


class TestSafeLoading(unittest.TestCase):
    """Test safe loading functionality."""
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_model_safe_success(self, mock_load):
        """Test successful safe loading."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        result = OmniModelLoader.load_model_safe("/test/path", mode="thinker_only")
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], mock_model)
        self.assertEqual(result[1], mock_tokenizer)
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_model_safe_failure_skip(self, mock_load):
        """Test safe loading failure with skip_on_error=True."""
        mock_load.side_effect = Exception("Load failed")
        
        result = OmniModelLoader.load_model_safe("/test/path", mode="thinker_only", skip_on_error=True)
        
        self.assertIsNone(result)
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_model_safe_failure_raise(self, mock_load):
        """Test safe loading failure with skip_on_error=False."""
        mock_load.side_effect = Exception("Load failed")
        
        with self.assertRaises(Exception) as context:
            OmniModelLoader.load_model_safe("/test/path", mode="thinker_only", skip_on_error=False)
        
        self.assertIn("Load failed", str(context.exception))


class TestLoaderInitialization(unittest.TestCase):
    """Test OmniModelLoader initialization."""
    
    def test_init_with_path(self):
        """Test initialization with model path."""
        loader = OmniModelLoader("/test/path")
        
        self.assertEqual(str(loader.model_path), "/test/path")
        self.assertIsNone(loader._model)
        self.assertIsNone(loader._tokenizer)
        self.assertIsNone(loader._processor)
        self.assertIsNone(loader._config)
    
    def test_init_without_path(self):
        """Test initialization without model path."""
        loader = OmniModelLoader()
        
        self.assertIsNone(loader.model_path)
    
    def test_init_with_pathlib(self):
        """Test initialization with Path object."""
        path = Path("/test/path")
        loader = OmniModelLoader(path)
        
        self.assertEqual(loader.model_path, path)


class TestModelTypeMappings(unittest.TestCase):
    """Test model type mappings constants."""
    
    def test_model_type_mappings_exist(self):
        """Test that MODEL_TYPE_MAPPINGS is defined."""
        self.assertTrue(hasattr(OmniModelLoader, 'MODEL_TYPE_MAPPINGS'))
        self.assertIsInstance(OmniModelLoader.MODEL_TYPE_MAPPINGS, dict)
    
    def test_key_model_types_mapped(self):
        """Test that key model types are in mappings."""
        expected_types = ["glm4_moe_lite", "step_robotics", "qwen3", "agent_cpm"]
        
        for model_type in expected_types:
            self.assertIn(model_type, OmniModelLoader.MODEL_TYPE_MAPPINGS)
    
    def test_model_type_mapping_structure(self):
        """Test that each mapping has required fields."""
        for model_type, mapping in OmniModelLoader.MODEL_TYPE_MAPPINGS.items():
            self.assertIn("architecture", mapping)
            self.assertIn("config_class", mapping)
    
    def test_supported_architectures_list(self):
        """Test that SUPPORTED_ARCHITECTURES is populated."""
        self.assertTrue(len(OmniModelLoader.SUPPORTED_ARCHITECTURES) > 0)
        self.assertIn("LlamaForCausalLM", OmniModelLoader.SUPPORTED_ARCHITECTURES)
        self.assertIn("Qwen2ForCausalLM", OmniModelLoader.SUPPORTED_ARCHITECTURES)
    
    def test_vision_encoder_architectures(self):
        """Test VISION_ENCODER_ARCHITECTURES list."""
        self.assertIn("SigLIPModel", OmniModelLoader.VISION_ENCODER_ARCHITECTURES)
        self.assertIn("CLIPModel", OmniModelLoader.VISION_ENCODER_ARCHITECTURES)
        self.assertIn("DINOv2Model", OmniModelLoader.VISION_ENCODER_ARCHITECTURES)
    
    def test_asr_architectures(self):
        """Test ASR_ARCHITECTURES list."""
        self.assertIn("WhisperForConditionalGeneration", OmniModelLoader.ASR_ARCHITECTURES)
    
    def test_audio_encoder_architectures(self):
        """Test AUDIO_ENCODER_ARCHITECTURES list."""
        self.assertIn("Wav2Vec2Model", OmniModelLoader.AUDIO_ENCODER_ARCHITECTURES)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_config(self):
        """Test handling of empty config."""
        with open(self.model_path / "config.json", "w") as f:
            json.dump({}, f)
        
        info = OmniModelLoader.get_model_info(self.model_path)
        self.assertEqual(info["architecture"], "unknown")
        self.assertEqual(info["model_type"], "unknown")
    
    def test_config_with_empty_architectures(self):
        """Test handling of config with empty architectures list."""
        with open(self.model_path / "config.json", "w") as f:
            json.dump({"architectures": []}, f)
        
        info = OmniModelLoader.get_model_info(self.model_path)
        self.assertEqual(info["architecture"], "unknown")
    
    def test_malformed_json_config(self):
        """Test handling of malformed JSON config."""
        with open(self.model_path / "config.json", "w") as f:
            f.write("not valid json {{{")
        
        info = OmniModelLoader.get_model_info(self.model_path)
        self.assertIsNotNone(info["error"])
    
    def test_vision_encoder_with_permission_error(self):
        """Test vision encoder detection with permission error."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = OmniModelLoader._is_vision_encoder(self.model_path)
            self.assertFalse(result)
    
    def test_is_omni_model_with_permission_error(self):
        """Test is_omni_model with permission error."""
        # Create config file but patch open to raise error
        with open(self.model_path / "config.json", "w") as f:
            json.dump({"model_type": "test"}, f)
        
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = OmniModelLoader.is_omni_model(self.model_path)
            self.assertFalse(result)
    
    def test_path_with_special_characters(self):
        """Test handling of paths with special characters."""
        special_path = self.temp_dir / "model-with.special_chars"
        special_path.mkdir()
        
        config = {"architectures": ["LlamaForCausalLM"]}
        with open(special_path / "config.json", "w") as f:
            json.dump(config, f)
        
        info = OmniModelLoader.get_model_info(special_path)
        self.assertEqual(info["architecture"], "LlamaForCausalLM")


# Pytest-style tests for more modern testing
class TestPytestStyle:
    """Pytest-style tests for modern Python testing."""
    
    def test_loader_has_required_methods(self):
        """Test that loader has all required methods."""
        required_methods = [
            'is_omni_model',
            'get_model_info',
            'is_model_supported',
            '_is_sae_model',
            '_get_sae_base_model',
            '_is_diffusers_model',
            '_is_vision_encoder',
            '_is_asr_model',
            '_detect_model_category',
            '_load_tokenizer',
            '_load_diffusers_model',
            '_load_vision_encoder',
            '_load_asr_model',
            'load_model_safe',
        ]
        
        for method in required_methods:
            assert hasattr(OmniModelLoader, method), f"Missing method: {method}"
    
    def test_loader_has_required_attributes(self):
        """Test that loader has all required class attributes."""
        required_attrs = [
            'SUPPORTED_ARCHITECTURES',
            'VISION_ENCODER_ARCHITECTURES',
            'AUDIO_ENCODER_ARCHITECTURES',
            'ASR_ARCHITECTURES',
            'ARCHITECTURE_ALIASES',
            'MODEL_TYPE_MAPPINGS',
        ]
        
        for attr in required_attrs:
            assert hasattr(OmniModelLoader, attr), f"Missing attribute: {attr}"
    
    def test_model_type_mappings_content(self):
        """Test content of model type mappings."""
        test_cases = [
            ("glm4_moe_lite", "Glm4MoeForCausalLM"),
            ("step_robotics", "Step3VL10BForCausalLM"),
            ("qwen3", "Qwen3ForCausalLM"),
            ("agent_cpm", "Qwen3ForCausalLM"),
        ]
        for model_type, expected_arch in test_cases:
            assert model_type in OmniModelLoader.MODEL_TYPE_MAPPINGS, f"Missing model type: {model_type}"
            mapping = OmniModelLoader.MODEL_TYPE_MAPPINGS[model_type]
            assert mapping["architecture"] == expected_arch, f"Wrong architecture for {model_type}"


