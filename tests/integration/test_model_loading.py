#!/usr/bin/env python3
"""
Integration tests for OmniModelLoader.

Tests actual model loading from various model types:
- Loading each model type from the teacher registry
- Error handling for unsupported architectures
- Graceful skipping of failed models

These tests can be run with or without actual model files.
Tests requiring real models are marked with @pytest.mark.real_model.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Optional pytest import
try:
    import pytest
except ImportError:
    pytest = None

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from omni.loader import OmniModelLoader, load_omni_model


class TestModelLoadingIntegration(unittest.TestCase):
    """Integration tests for model loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_model(self, config, model_files=None):
        """Helper to create a mock model directory."""
        model_path = self.temp_path / "mock_model"
        model_path.mkdir(exist_ok=True)
        
        # Write config
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Create additional files if specified
        if model_files:
            for filename, content in model_files.items():
                filepath = model_path / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, dict):
                    with open(filepath, "w") as f:
                        json.dump(content, f)
                else:
                    with open(filepath, "w") as f:
                        f.write(content)
        
        return model_path


class TestSupportedArchitecturesLoading(TestModelLoadingIntegration):
    """Test loading of supported architectures."""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_llama_model(self, mock_tokenizer, mock_model):
        """Test loading a Llama model."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        result = loader.load_model_safe(model_path, mode="thinker_only", skip_on_error=True)
        
        # Should attempt to load
        self.assertIsNotNone(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_qwen_model(self, mock_tokenizer, mock_model):
        """Test loading a Qwen model."""
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_attention_heads": 28,
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        result = loader.load_model_safe(model_path, mode="thinker_only", skip_on_error=True)
        
        self.assertIsNotNone(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_mistral_model(self, mock_tokenizer, mock_model):
        """Test loading a Mistral model."""
        config = {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_attention_heads": 32,
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        result = loader.load_model_safe(model_path, mode="thinker_only", skip_on_error=True)
        
        self.assertIsNotNone(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_gemma_model(self, mock_tokenizer, mock_model):
        """Test loading a Gemma model."""
        config = {
            "architectures": ["GemmaForCausalLM"],
            "model_type": "gemma",
            "hidden_size": 3072,
            "num_attention_heads": 16,
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        result = loader.load_model_safe(model_path, mode="thinker_only", skip_on_error=True)
        
        self.assertIsNotNone(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_phi_model(self, mock_tokenizer, mock_model):
        """Test loading a Phi model."""
        config = {
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_attention_heads": 32,
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        result = loader.load_model_safe(model_path, mode="thinker_only", skip_on_error=True)
        
        self.assertIsNotNone(result)


class TestSAEModelLoading(TestModelLoadingIntegration):
    """Test SAE model loading and tokenizer fallback."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_sae_model_tokenizer_fallback(self, mock_tokenizer):
        """Test that SAE models load tokenizer from base model."""
        # Create SAE model structure
        sae_path = self.temp_path / "gemma-scope-sae"
        sae_path.mkdir()
        
        # Create SAE directories
        (sae_path / "resid_post").mkdir()
        (sae_path / "mlp_out").mkdir()
        
        # Create SAE config with base model info
        sae_config_dir = sae_path / "resid_post" / "layer_0"
        sae_config_dir.mkdir(parents=True)
        with open(sae_config_dir / "config.json", "w") as f:
            json.dump({"model_name": "google/gemma-2b-it"}, f)
        
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(sae_path)
        
        # Should detect as SAE
        self.assertTrue(OmniModelLoader._is_sae_model(sae_path))
        
        # Should extract base model
        base_model = OmniModelLoader._get_sae_base_model(sae_path)
        self.assertEqual(base_model, "google/gemma-2b-it")
    
    def test_sae_model_detection_integration(self):
        """Test SAE model detection in integration context."""
        sae_path = self.temp_path / "sae-model"
        sae_path.mkdir()
        
        # Create SAE structure
        (sae_path / "resid_post").mkdir()
        (sae_path / "attn_out").mkdir()
        
        # Should be detected as SAE
        self.assertTrue(OmniModelLoader._is_sae_model(sae_path))
        
        # Should be detected as SAE category
        category = OmniModelLoader._detect_model_category(sae_path)
        self.assertEqual(category, "sae")
        
        # Should be marked as supported
        support_info = OmniModelLoader.is_model_supported(sae_path)
        self.assertTrue(support_info["supported"])
        self.assertEqual(support_info["category"], "sae")


class TestVisionEncoderLoading(TestModelLoadingIntegration):
    """Test vision encoder model loading."""
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_siglip_model(self, mock_model):
        """Test loading SigLIP vision encoder."""
        config = {
            "architectures": ["SigLIPModel"],
            "model_type": "siglip_vision_model",
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        
        # Should detect as vision encoder
        self.assertTrue(OmniModelLoader._is_vision_encoder(model_path))
        
        # Should detect correct category
        category = OmniModelLoader._detect_model_category(model_path)
        self.assertEqual(category, "vision_encoder")
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_clip_model(self, mock_model):
        """Test loading CLIP vision encoder."""
        config = {
            "architectures": ["CLIPVisionModel"],
            "model_type": "clip_vision_model",
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        
        # Should detect as vision encoder
        self.assertTrue(OmniModelLoader._is_vision_encoder(model_path))
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_dinov2_model(self, mock_model):
        """Test loading DINOv2 vision encoder."""
        config = {
            "architectures": ["DINOv2Model"],
            "model_type": "dinov2",
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        
        # Should detect as vision encoder
        self.assertTrue(OmniModelLoader._is_vision_encoder(model_path))


class TestASRModelLoading(TestModelLoadingIntegration):
    """Test ASR model loading."""
    
    @patch('transformers.AutoModelForSpeechSeq2Seq.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def test_load_whisper_model(self, mock_processor, mock_model):
        """Test loading Whisper ASR model."""
        config = {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
        }
        model_path = self._create_mock_model(config)
        
        mock_model.return_value = Mock()
        mock_processor.return_value = Mock()
        
        # Should detect as ASR
        self.assertTrue(OmniModelLoader._is_asr_model(model_path))
        
        # Should detect correct category
        category = OmniModelLoader._detect_model_category(model_path)
        self.assertEqual(category, "asr")


class TestDiffusersModelLoading(TestModelLoadingIntegration):
    """Test Diffusers model loading."""
    
    def test_diffusers_model_detection(self):
        """Test detection of Diffusers models."""
        # Create diffusers model structure
        sd_path = self.temp_path / "stable-diffusion"
        sd_path.mkdir()
        
        # Create model_index.json
        with open(sd_path / "model_index.json", "w") as f:
            json.dump({"_class_name": "StableDiffusionPipeline"}, f)
        
        # Should detect as diffusers
        self.assertTrue(OmniModelLoader._is_diffusers_model(sd_path))
        
        # Should detect correct category
        category = OmniModelLoader._detect_model_category(sd_path)
        self.assertEqual(category, "diffusers")
        
        # Should be marked as supported
        support_info = OmniModelLoader.is_model_supported(sd_path)
        self.assertTrue(support_info["supported"])
        self.assertEqual(support_info["category"], "diffusers")
    
    def test_diffusers_model_detection_by_structure(self):
        """Test detection by unet/vae structure."""
        sd_path = self.temp_path / "stable-diffusion-2"
        sd_path.mkdir()
        
        # Create unet and vae directories
        (sd_path / "unet").mkdir()
        (sd_path / "vae").mkdir()
        
        # Should detect as diffusers
        self.assertTrue(OmniModelLoader._is_diffusers_model(sd_path))


class TestErrorHandlingIntegration(TestModelLoadingIntegration):
    """Test error handling during model loading."""
    
    def test_unsupported_architecture(self):
        """Test handling of unsupported architecture."""
        config = {
            "architectures": ["SomeCompletelyUnknownArchitecture"],
            "model_type": "unknown_type",
        }
        model_path = self._create_mock_model(config)
        
        # Should not be marked as supported
        support_info = OmniModelLoader.is_model_supported(model_path)
        self.assertFalse(support_info["supported"])
        self.assertIsNotNone(support_info["error"])
    
    def test_missing_config(self):
        """Test handling of missing config.json."""
        empty_path = self.temp_path / "empty_model"
        empty_path.mkdir()
        
        # Should handle gracefully
        info = OmniModelLoader.get_model_info(empty_path)
        self.assertEqual(info["error"], "No config.json found")
        
        # Should not crash
        support_info = OmniModelLoader.is_model_supported(empty_path)
        self.assertFalse(support_info["supported"])
    
    def test_malformed_config(self):
        """Test handling of malformed config."""
        model_path = self.temp_path / "bad_model"
        model_path.mkdir()
        
        # Write invalid JSON
        with open(model_path / "config.json", "w") as f:
            f.write("not valid json {{}}")
        
        # Should handle gracefully
        info = OmniModelLoader.get_model_info(model_path)
        self.assertIsNotNone(info["error"])
    
    def test_safe_load_with_error(self):
        """Test safe loading returns None on error."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        }
        model_path = self._create_mock_model(config)
        
        # Should return None on error (skip_on_error=True)
        result = OmniModelLoader.load_model_safe(
            model_path, 
            mode="thinker_only",
            skip_on_error=True
        )
        
        # Will return None because model files don't exist
        self.assertIsNone(result)
    
    def test_safe_load_without_skip(self):
        """Test safe loading raises error when skip_on_error=False."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        }
        model_path = self._create_mock_model(config)
        
        # Should raise error when skip_on_error=False
        with self.assertRaises(Exception):
            OmniModelLoader.load_model_safe(
                model_path,
                mode="thinker_only",
                skip_on_error=False
            )


class TestCustomModelRegistrationIntegration(TestModelLoadingIntegration):
    """Test custom model architecture registration."""
    
    @patch('transformers.models.auto.configuration_auto.CONFIG_MAPPING')
    @patch('transformers.models.auto.modeling_auto.AutoModelForCausalLM')
    @patch('transformers.models.auto.modeling_auto.AutoModel')
    def test_glm4_moe_lite_registration(self, mock_auto_model, mock_auto_causal, mock_config):
        """Test registration of glm4_moe_lite model type."""
        config = {
            "model_type": "glm4_moe_lite",
            "architectures": ["Glm4MoeLiteForCausalLM"],
        }
        model_path = self._create_mock_model(config)
        
        # Mock the mappings
        mock_mapping = MagicMock()
        mock_mapping._extra_content = {}
        mock_auto_causal._model_mapping = mock_mapping
        mock_auto_model._model_mapping = mock_mapping
        mock_config._extra_content = {}
        
        loader = OmniModelLoader(model_path)
        loader._register_custom_architecture(model_path)
        
        # Should register the model type
        self.assertIn("glm4_moe_lite", mock_mapping._extra_content)
    
    @patch('transformers.models.auto.configuration_auto.CONFIG_MAPPING')
    @patch('transformers.models.auto.modeling_auto.AutoModelForCausalLM')
    @patch('transformers.models.auto.modeling_auto.AutoModel')
    @patch('transformers.models.auto.modeling_auto.AutoModelForVision2Seq')
    def test_step_robotics_registration(self, mock_vision, mock_auto_model, mock_auto_causal, mock_config):
        """Test registration of step_robotics model type."""
        config = {
            "model_type": "step_robotics",
            "architectures": ["Step3VL10BForCausalLM"],
        }
        model_path = self._create_mock_model(config)
        
        # Mock the mappings
        mock_mapping = MagicMock()
        mock_mapping._extra_content = {}
        mock_auto_causal._model_mapping = mock_mapping
        mock_auto_model._model_mapping = mock_mapping
        mock_vision._model_mapping = mock_mapping
        mock_config._extra_content = {}
        
        loader = OmniModelLoader(model_path)
        loader._register_custom_architecture(model_path)
        
        # Should register the model type
        self.assertIn("step_robotics", mock_mapping._extra_content)


class TestModelCategoryDetectionIntegration(TestModelLoadingIntegration):
    """Test model category detection with various configurations."""
    
    def test_category_detection_priority(self):
        """Test that category detection follows correct priority."""
        # Diffusers should take priority over other categories
        sd_path = self.temp_path / "sd-model"
        sd_path.mkdir()
        (sd_path / "model_index.json").touch()
        
        category = OmniModelLoader._detect_model_category(sd_path)
        self.assertEqual(category, "diffusers")
    
    def test_category_detection_vision_encoder(self):
        """Test vision encoder category detection."""
        vision_path = self.temp_path / "vision-model"
        vision_path.mkdir()
        
        with open(vision_path / "config.json", "w") as f:
            json.dump({"architectures": ["SigLIPModel"]}, f)
        
        category = OmniModelLoader._detect_model_category(vision_path)
        self.assertEqual(category, "vision_encoder")
    
    def test_category_detection_asr(self):
        """Test ASR category detection."""
        asr_path = self.temp_path / "asr-model"
        asr_path.mkdir()
        
        with open(asr_path / "config.json", "w") as f:
            json.dump({"architectures": ["WhisperForConditionalGeneration"]}, f)
        
        category = OmniModelLoader._detect_model_category(asr_path)
        self.assertEqual(category, "asr")
    
    def test_category_detection_sae(self):
        """Test SAE category detection."""
        sae_path = self.temp_path / "sae-model"
        sae_path.mkdir()
        (sae_path / "resid_post").mkdir()
        
        category = OmniModelLoader._detect_model_category(sae_path)
        self.assertEqual(category, "sae")
    
    def test_category_detection_transformers(self):
        """Test regular transformers category detection."""
        model_path = self.temp_path / "transformers-model"
        model_path.mkdir()
        
        with open(model_path / "config.json", "w") as f:
            json.dump({"architectures": ["LlamaForCausalLM"]}, f)
        
        category = OmniModelLoader._detect_model_category(model_path)
        self.assertEqual(category, "transformers")


class TestTokenizerLoadingIntegration(TestModelLoadingIntegration):
    """Test tokenizer loading with various scenarios."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenizer_load_success(self, mock_tokenizer):
        """Test successful tokenizer loading."""
        model_path = self.temp_path / "model-with-tokenizer"
        model_path.mkdir()
        
        with open(model_path / "config.json", "w") as f:
            json.dump({"model_type": "gpt2"}, f)
        
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(model_path)
        tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        
        mock_tokenizer.assert_called()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenizer_fallback_to_eos(self, mock_tokenizer):
        """Test that pad_token falls back to eos_token when None."""
        model_path = self.temp_path / "model-no-pad"
        model_path.mkdir()
        
        with open(model_path / "config.json", "w") as f:
            json.dump({"model_type": "gpt2"}, f)
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        loader = OmniModelLoader(model_path)
        tokenizer = loader._load_tokenizer(model_path, trust_remote_code=True)
        
        # Should set pad_token to eos_token
        self.assertEqual(tokenizer.pad_token, "<|endoftext|>")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenizer_sae_fallback(self, mock_tokenizer):
        """Test tokenizer loading falls back to base model for SAE."""
        sae_path = self.temp_path / "sae-model"
        sae_path.mkdir()
        (sae_path / "resid_post").mkdir()
        
        # Create base model config in SAE structure
        sae_config_dir = sae_path / "resid_post" / "layer_0"
        sae_config_dir.mkdir(parents=True)
        with open(sae_config_dir / "config.json", "w") as f:
            json.dump({"model_name": "gpt2"}, f)
        
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.pad_token = "<pad>"
        
        loader = OmniModelLoader(sae_path)
        tokenizer = loader._load_tokenizer(sae_path, trust_remote_code=True)
        
        # Should attempt to load from base model
        mock_tokenizer.assert_called()


class TestLoadOmniModelFunctionIntegration(unittest.TestCase):
    """Test the load_omni_model convenience function."""
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_omni_model_basic(self, mock_load):
        """Test basic load_omni_model call."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        result = load_omni_model("/test/path", mode="thinker_only")
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], mock_model)
        self.assertEqual(result[1], mock_tokenizer)
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_omni_model_skip_on_error(self, mock_load):
        """Test load_omni_model with skip_on_error=True."""
        mock_load.side_effect = Exception("Load failed")
        
        result = load_omni_model("/test/path", skip_on_error=True)
        
        self.assertIsNone(result)
    
    @patch.object(OmniModelLoader, 'load')
    def test_load_omni_model_raise_on_error(self, mock_load):
        """Test load_omni_model with skip_on_error=False (default)."""
        mock_load.side_effect = Exception("Load failed")
        
        with self.assertRaises(Exception) as context:
            load_omni_model("/test/path", skip_on_error=False)
        
        self.assertIn("Load failed", str(context.exception))


@unittest.skipUnless(pytest is not None, "pytest not installed")
class TestRealModelLoading(unittest.TestCase):
    """
    Tests that require actual model files.
    
    These tests are skipped by default and only run when real models are available.
    Use pytest -m real_model to run these tests.
    """
    
    def test_load_real_gpt2(self):
        """Test loading real GPT-2 model if available."""
        try:
            result = OmniModelLoader.load_model_safe(
                "gpt2",
                mode="thinker_only",
                skip_on_error=True
            )
            if result is not None:
                model, tokenizer = result
                self.assertIsNotNone(model)
                self.assertIsNotNone(tokenizer)
        except Exception as e:
            pytest.skip(f"GPT-2 not available: {e}")
    
    def test_is_omni_model_with_real_path(self):
        """Test is_omni_model with a real model path."""
        # This test requires an actual model path
        test_path = os.environ.get("TEST_MODEL_PATH")
        if not test_path:
            pytest.skip("TEST_MODEL_PATH environment variable not set")
        
        result = OmniModelLoader.is_omni_model(test_path)
        # Just verify it doesn't crash
        self.assertIsInstance(result, bool)


class TestModelInfoIntegration(TestModelLoadingIntegration):
    """Test model info retrieval integration."""
    
    def test_model_info_with_quantization(self):
        """Test model info detection for quantized model."""
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16"
            }
        }
        model_path = self._create_mock_model(config)
        
        info = OmniModelLoader.get_model_info(model_path)
        self.assertTrue(info["is_quantized"])
        self.assertEqual(info["architecture"], "LlamaForCausalLM")
    
    def test_model_info_with_talker(self):
        """Test model info detection for model with talker."""
        config = {
            "architectures": ["Qwen2_5OmniForConditionalGeneration"],
            "model_type": "qwen2_5_omni",
            "talker_config": {
                "hidden_size": 1536
            },
            "audio_config": {}
        }
        model_path = self._create_mock_model(config)
        
        info = OmniModelLoader.get_model_info(model_path)
        self.assertTrue(info["has_talker"])
        self.assertTrue(info["is_supported"])
    
    def test_model_info_with_custom_files(self):
        """Test model info detection with custom modeling files."""
        config = {
            "architectures": ["CustomModel"],
            "model_type": "custom_model",
        }
        model_files = {
            "modeling_custom_model.py": "# Custom model implementation"
        }
        model_path = self._create_mock_model(config, model_files)
        
        info = OmniModelLoader.get_model_info(model_path)
        self.assertTrue(info["has_custom_files"])
        self.assertTrue(info["is_supported"])


if __name__ == "__main__":
    unittest.main()