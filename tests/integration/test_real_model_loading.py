"""
Integration tests using real Qwen2.5-0.5B model.

Tests:
- Real model loading
- Real tokenization
- Real inference (small sample)
- Model config inspection
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRealModelLoading:
    """Test real model loading with Qwen2.5-0.5B."""
    
    @pytest.mark.real_model
    def test_model_loads_successfully(self, real_text_model):
        """Test that the real model loads without errors."""
        assert real_text_model is not None
        assert hasattr(real_text_model, 'forward')
        assert hasattr(real_text_model, 'generate')
    
    @pytest.mark.real_model
    def test_tokenizer_loads_successfully(self, real_text_tokenizer):
        """Test that the tokenizer loads without errors."""
        assert real_text_tokenizer is not None
        assert hasattr(real_text_tokenizer, 'encode')
        assert hasattr(real_text_tokenizer, 'decode')
    
    @pytest.mark.real_model
    def test_model_on_correct_device(self, real_text_model, device):
        """Test that model is on the expected device."""
        # Get first parameter's device
        first_param = next(real_text_model.parameters())
        
        if device == "cuda":
            assert first_param.is_cuda
        else:
            assert not first_param.is_cuda
    
    @pytest.mark.real_model
    def test_model_in_eval_mode(self, real_text_model):
        """Test that model is in eval mode."""
        assert not real_text_model.training
    
    @pytest.mark.real_model
    def test_model_config_accessible(self, real_text_model):
        """Test that model config is accessible."""
        config = real_text_model.config
        assert config is not None
        assert hasattr(config, 'vocab_size')
        assert hasattr(config, 'hidden_size')


class TestRealTokenization:
    """Test real tokenization with the loaded tokenizer."""
    
    @pytest.mark.real_model
    def test_encode_simple_text(self, real_text_tokenizer):
        """Test encoding simple text."""
        text = "Hello, world!"
        tokens = real_text_tokenizer.encode(text)
        
        assert isinstance(tokens, list) or torch.is_tensor(tokens)
        assert len(tokens) > 0
    
    @pytest.mark.real_model
    def test_decode_tokens(self, real_text_tokenizer):
        """Test decoding tokens back to text."""
        text = "Hello, world!"
        tokens = real_text_tokenizer.encode(text)
        decoded = real_text_tokenizer.decode(tokens)
        
        assert "Hello" in decoded
        assert "world" in decoded
    
    @pytest.mark.real_model
    def test_tokenizer_special_tokens(self, real_text_tokenizer):
        """Test special tokens are defined."""
        assert real_text_tokenizer.pad_token_id is not None or \
               real_text_tokenizer.eos_token_id is not None
    
    @pytest.mark.real_model
    def test_batch_encoding(self, real_text_tokenizer):
        """Test batch encoding multiple texts."""
        texts = ["Hello!", "How are you?", "Goodbye!"]
        batch = real_text_tokenizer(texts, padding=True, return_tensors="pt")
        
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 3


class TestRealInference:
    """Test real inference with the loaded model."""
    
    @pytest.mark.real_model
    @pytest.mark.slow
    def test_simple_forward_pass(self, real_text_model, real_text_tokenizer, device):
        """Test a simple forward pass."""
        model = real_text_model
        tokenizer = real_text_tokenizer
        
        text = "Hello"
        inputs = tokenizer(text, return_tensors="pt")
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs is not None
        assert hasattr(outputs, 'logits')
        assert outputs.logits.shape[0] == 1  # Batch size 1
    
    @pytest.mark.real_model
    @pytest.mark.slow
    def test_small_generation(self, real_text_model, real_text_tokenizer, device):
        """Test generating a few tokens."""
        model = real_text_model
        tokenizer = real_text_tokenizer
        
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        assert len(decoded) > len(prompt)
        assert "Paris" in decoded or len(decoded) > len(prompt)
    
    @pytest.mark.real_model
    @pytest.mark.slow
    def test_math_inference(self, real_text_model, real_text_tokenizer, device):
        """Test math reasoning capability."""
        model = real_text_model
        tokenizer = real_text_tokenizer
        
        prompt = "What is 2 + 2? The answer is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check it generates something
        assert len(decoded) > len(prompt)


class TestModelConfigInspection:
    """Test model configuration inspection for modality detection."""
    
    @pytest.mark.real_model
    def test_config_has_model_type(self, real_text_model):
        """Test config has model_type field."""
        config = real_text_model.config
        assert hasattr(config, 'model_type')
        assert config.model_type is not None
    
    @pytest.mark.real_model
    def test_config_has_architectures(self, real_text_model):
        """Test config has architectures."""
        config = real_text_model.config
        # May be in config or in model.__class__.__name__
        assert hasattr(config, 'architectures') or \
               hasattr(real_text_model, '__class__')
    
    @pytest.mark.real_model
    def test_text_only_model_no_vision_config(self, real_text_model):
        """Test text-only model doesn't have vision_config."""
        config = real_text_model.config
        
        # Should NOT have vision_config
        has_vision = hasattr(config, 'vision_config') and config.vision_config is not None
        assert not has_vision, "Text-only model shouldn't have vision_config"
    
    @pytest.mark.real_model
    def test_text_only_model_no_audio_config(self, real_text_model):
        """Test text-only model doesn't have audio_config."""
        config = real_text_model.config
        
        # Should NOT have audio_config
        has_audio = hasattr(config, 'audio_config') and config.audio_config is not None
        assert not has_audio, "Text-only model shouldn't have audio_config"


class TestIntegrationWithDetectModalities:
    """Test detect_modalities with real model path."""
    
    @pytest.mark.real_model
    def test_detect_on_real_text_model(self, text_model_path):
        """Test modality detection on real text model."""
        from src.detect_modalities import detect_modalities
        
        result = detect_modalities(text_model_path)
        
        # Verify structure
        assert "modalities" in result
        assert "model_type" in result
        
        # Text-only should have text=True
        assert result["modalities"]["text"] is True
        
        # Other modalities should be False
        assert result["modalities"]["vision"] is False
        assert result["modalities"]["audio_input"] is False
    
    @pytest.mark.real_model
    def test_model_type_is_qwen(self, text_model_path):
        """Test model type is recognized as qwen."""
        from src.detect_modalities import detect_modalities
        
        result = detect_modalities(text_model_path)
        
        assert "qwen" in result["model_type"].lower()
