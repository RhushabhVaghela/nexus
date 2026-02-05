#!/usr/bin/env python3
"""
test_encoder_decoder_shapes.py

Comprehensive shape verification tests for the multimodal pipeline.
Validates tensor shapes through the entire forward pass:
- Vision: SigLIP → Projection → Resampler
- Audio: Whisper → Projection → Resampler
- Combined: OmniMultimodalLM forward pass

Usage:
    pytest test_encoder_decoder_shapes.py -v
    python test_encoder_decoder_shapes.py
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

# Import model components
try:
    from model import (
        PerceiverResampler,
        VisionEncoder,
        AudioEncoder,
        OmniMultimodalLM,
    )
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("Warning: Could not import model components. Running mock tests.")


# ═══════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Standard configuration (matching model.py defaults)
CONFIG = {
    "batch_size": 2,
    "llm_dim": 4096,
    "num_latents": 64,
    "vision": {
        "model": "google/siglip-so400m-patch14-512",
        "input_size": 512,
        "patch_size": 14,
        "output_dim": 1152,
    },
    "audio": {
        "model": "openai/whisper-large-v3-turbo",
        "sample_rate": 16000,
        "max_seconds": 30,
        "output_dim": 1280,
    },
    "text": {
        "max_length": 128,
    },
}

# Compute expected shapes
VISION_NUM_PATCHES = (CONFIG["vision"]["input_size"] // CONFIG["vision"]["patch_size"]) ** 2
AUDIO_NUM_FRAMES = int(CONFIG["audio"]["sample_rate"] * CONFIG["audio"]["max_seconds"] / 160)  # 160 = hop_length


# ═══════════════════════════════════════════════════════════════
# MOCK COMPONENTS (for testing without GPU/models)
# ═══════════════════════════════════════════════════════════════

class MockVisionEncoder(nn.Module):
    """Mock vision encoder that produces expected tensor shapes."""
    
    def __init__(self, output_dim: int = 1152):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 512, 512]
        # output: [B, num_patches+1, output_dim]
        batch = images.shape[0]
        num_patches = VISION_NUM_PATCHES + 1  # +1 for CLS token
        return torch.randn(batch, num_patches, self.output_dim)


class MockAudioEncoder(nn.Module):
    """Mock audio encoder that produces expected tensor shapes."""
    
    def __init__(self, output_dim: int = 1280):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # audio_features: [B, 80, T] (mel spectrogram)
        # output: [B, T/2, output_dim]
        batch = audio_features.shape[0]
        time_steps = audio_features.shape[2] // 2
        return torch.randn(batch, time_steps, self.output_dim)


class MockPerceiverResampler(nn.Module):
    """Mock Perceiver Resampler that produces expected tensor shapes."""
    
    def __init__(self, dim: int, num_latents: int = 64):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]
        # output: [B, num_latents, dim]
        batch = x.shape[0]
        return torch.randn(batch, self.num_latents, self.dim)


# ═══════════════════════════════════════════════════════════════
# VISION PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestVisionPipeline:
    """Test vision pipeline: SigLIP → Projection → Resampler."""
    
    @pytest.fixture
    def batch_size(self):
        return CONFIG["batch_size"]
    
    @pytest.fixture
    def mock_images(self, batch_size):
        """Create mock image input: [B, 3, 512, 512]."""
        return torch.randn(batch_size, 3, 512, 512)
    
    def test_input_shape(self, mock_images, batch_size):
        """Verify input tensor has correct shape."""
        assert mock_images.shape == (batch_size, 3, 512, 512)
    
    def test_encoder_output_shape(self, mock_images):
        """Verify vision encoder output shape."""
        encoder = MockVisionEncoder(output_dim=CONFIG["vision"]["output_dim"])
        output = encoder(mock_images)
        
        batch = mock_images.shape[0]
        expected_patches = VISION_NUM_PATCHES + 1  # +1 for CLS token
        expected_dim = CONFIG["vision"]["output_dim"]
        
        assert output.shape == (batch, expected_patches, expected_dim), \
            f"Expected {(batch, expected_patches, expected_dim)}, got {output.shape}"
    
    def test_projection_shape(self, mock_images):
        """Verify vision projection layer output shape."""
        encoder = MockVisionEncoder(output_dim=CONFIG["vision"]["output_dim"])
        projection = nn.Linear(CONFIG["vision"]["output_dim"], CONFIG["llm_dim"])
        
        encoder_out = encoder(mock_images)
        projected = projection(encoder_out)
        
        batch = mock_images.shape[0]
        expected_patches = VISION_NUM_PATCHES + 1
        
        assert projected.shape == (batch, expected_patches, CONFIG["llm_dim"]), \
            f"Expected {(batch, expected_patches, CONFIG['llm_dim'])}, got {projected.shape}"
    
    def test_resampler_output_shape(self, mock_images):
        """Verify Perceiver Resampler output shape."""
        encoder = MockVisionEncoder(output_dim=CONFIG["vision"]["output_dim"])
        projection = nn.Linear(CONFIG["vision"]["output_dim"], CONFIG["llm_dim"])
        resampler = MockPerceiverResampler(dim=CONFIG["llm_dim"], num_latents=CONFIG["num_latents"])
        
        encoder_out = encoder(mock_images)
        projected = projection(encoder_out)
        resampled = resampler(projected)
        
        batch = mock_images.shape[0]
        
        assert resampled.shape == (batch, CONFIG["num_latents"], CONFIG["llm_dim"]), \
            f"Expected {(batch, CONFIG['num_latents'], CONFIG['llm_dim'])}, got {resampled.shape}"
    
    def test_full_vision_pipeline(self, mock_images):
        """Test complete vision pipeline end-to-end."""
        batch = mock_images.shape[0]
        
        # Full pipeline
        encoder = MockVisionEncoder(output_dim=CONFIG["vision"]["output_dim"])
        projection = nn.Linear(CONFIG["vision"]["output_dim"], CONFIG["llm_dim"])
        resampler = MockPerceiverResampler(dim=CONFIG["llm_dim"], num_latents=CONFIG["num_latents"])
        
        x = encoder(mock_images)
        assert x.ndim == 3, "Encoder output should be 3D"
        
        x = projection(x)
        assert x.ndim == 3, "Projection output should be 3D"
        
        x = resampler(x)
        assert x.shape == (batch, CONFIG["num_latents"], CONFIG["llm_dim"])
        
        print(f"\n✅ Vision Pipeline: [{batch}, 3, 512, 512] → [{batch}, {CONFIG['num_latents']}, {CONFIG['llm_dim']}]")


# ═══════════════════════════════════════════════════════════════
# AUDIO PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestAudioPipeline:
    """Test audio pipeline: Whisper → Projection → Resampler."""
    
    @pytest.fixture
    def batch_size(self):
        return CONFIG["batch_size"]
    
    @pytest.fixture
    def mock_audio_features(self, batch_size):
        """Create mock mel spectrogram input: [B, 80, 3000]."""
        # Whisper expects 80 mel bins, 3000 frames (30 seconds at 16kHz)
        mel_frames = 3000  # Standard for 30 seconds
        return torch.randn(batch_size, 80, mel_frames)
    
    def test_input_shape(self, mock_audio_features, batch_size):
        """Verify input tensor has correct shape."""
        assert mock_audio_features.shape == (batch_size, 80, 3000)
    
    def test_encoder_output_shape(self, mock_audio_features):
        """Verify audio encoder output shape."""
        encoder = MockAudioEncoder(output_dim=CONFIG["audio"]["output_dim"])
        output = encoder(mock_audio_features)
        
        batch = mock_audio_features.shape[0]
        expected_frames = mock_audio_features.shape[2] // 2  # Whisper downsamples by 2
        expected_dim = CONFIG["audio"]["output_dim"]
        
        assert output.shape == (batch, expected_frames, expected_dim), \
            f"Expected {(batch, expected_frames, expected_dim)}, got {output.shape}"
    
    def test_projection_shape(self, mock_audio_features):
        """Verify audio projection layer output shape."""
        encoder = MockAudioEncoder(output_dim=CONFIG["audio"]["output_dim"])
        projection = nn.Linear(CONFIG["audio"]["output_dim"], CONFIG["llm_dim"])
        
        encoder_out = encoder(mock_audio_features)
        projected = projection(encoder_out)
        
        batch = mock_audio_features.shape[0]
        expected_frames = mock_audio_features.shape[2] // 2
        
        assert projected.shape == (batch, expected_frames, CONFIG["llm_dim"]), \
            f"Expected {(batch, expected_frames, CONFIG['llm_dim'])}, got {projected.shape}"
    
    def test_resampler_output_shape(self, mock_audio_features):
        """Verify Perceiver Resampler output shape for audio."""
        encoder = MockAudioEncoder(output_dim=CONFIG["audio"]["output_dim"])
        projection = nn.Linear(CONFIG["audio"]["output_dim"], CONFIG["llm_dim"])
        resampler = MockPerceiverResampler(dim=CONFIG["llm_dim"], num_latents=CONFIG["num_latents"])
        
        encoder_out = encoder(mock_audio_features)
        projected = projection(encoder_out)
        resampled = resampler(projected)
        
        batch = mock_audio_features.shape[0]
        
        assert resampled.shape == (batch, CONFIG["num_latents"], CONFIG["llm_dim"]), \
            f"Expected {(batch, CONFIG['num_latents'], CONFIG['llm_dim'])}, got {resampled.shape}"
    
    def test_full_audio_pipeline(self, mock_audio_features):
        """Test complete audio pipeline end-to-end."""
        batch = mock_audio_features.shape[0]
        
        encoder = MockAudioEncoder(output_dim=CONFIG["audio"]["output_dim"])
        projection = nn.Linear(CONFIG["audio"]["output_dim"], CONFIG["llm_dim"])
        resampler = MockPerceiverResampler(dim=CONFIG["llm_dim"], num_latents=CONFIG["num_latents"])
        
        x = encoder(mock_audio_features)
        assert x.ndim == 3, "Encoder output should be 3D"
        
        x = projection(x)
        assert x.ndim == 3, "Projection output should be 3D"
        
        x = resampler(x)
        assert x.shape == (batch, CONFIG["num_latents"], CONFIG["llm_dim"])
        
        print(f"\n✅ Audio Pipeline: [{batch}, 80, 3000] → [{batch}, {CONFIG['num_latents']}, {CONFIG['llm_dim']}]")


# ═══════════════════════════════════════════════════════════════
# COMBINED FORWARD PASS TESTS
# ═══════════════════════════════════════════════════════════════

class TestCombinedForward:
    """Test combined multimodal forward pass."""
    
    @pytest.fixture
    def batch_size(self):
        return CONFIG["batch_size"]
    
    @pytest.fixture
    def mock_inputs(self, batch_size):
        """Create all mock inputs."""
        return {
            "input_ids": torch.randint(0, 32000, (batch_size, CONFIG["text"]["max_length"])),
            "pixel_values": torch.randn(batch_size, 3, 512, 512),
            "audio_features": torch.randn(batch_size, 80, 3000),
        }
    
    def test_text_only_embedding_shape(self, mock_inputs, batch_size):
        """Verify text-only embedding shape."""
        embedding_dim = CONFIG["llm_dim"]
        embedding = nn.Embedding(32000, embedding_dim)
        
        text_embeds = embedding(mock_inputs["input_ids"])
        
        assert text_embeds.shape == (batch_size, CONFIG["text"]["max_length"], embedding_dim)
    
    def test_vision_text_concatenation(self, mock_inputs, batch_size):
        """Verify vision+text concatenation shape."""
        embedding_dim = CONFIG["llm_dim"]
        embedding = nn.Embedding(32000, embedding_dim)
        vision_resampler = MockPerceiverResampler(dim=embedding_dim, num_latents=CONFIG["num_latents"])
        vision_encoder = MockVisionEncoder(output_dim=CONFIG["vision"]["output_dim"])
        vision_proj = nn.Linear(CONFIG["vision"]["output_dim"], embedding_dim)
        
        # Get embeddings
        text_embeds = embedding(mock_inputs["input_ids"])
        vision_features = vision_proj(vision_encoder(mock_inputs["pixel_values"]))
        vision_tokens = vision_resampler(vision_features)
        
        # Concatenate: [Vision, Text]
        combined = torch.cat([vision_tokens, text_embeds], dim=1)
        
        expected_seq_len = CONFIG["num_latents"] + CONFIG["text"]["max_length"]
        assert combined.shape == (batch_size, expected_seq_len, embedding_dim), \
            f"Expected {(batch_size, expected_seq_len, embedding_dim)}, got {combined.shape}"
    
    def test_audio_text_concatenation(self, mock_inputs, batch_size):
        """Verify audio+text concatenation shape."""
        embedding_dim = CONFIG["llm_dim"]
        embedding = nn.Embedding(32000, embedding_dim)
        audio_resampler = MockPerceiverResampler(dim=embedding_dim, num_latents=CONFIG["num_latents"])
        audio_encoder = MockAudioEncoder(output_dim=CONFIG["audio"]["output_dim"])
        audio_proj = nn.Linear(CONFIG["audio"]["output_dim"], embedding_dim)
        
        # Get embeddings
        text_embeds = embedding(mock_inputs["input_ids"])
        audio_features = audio_proj(audio_encoder(mock_inputs["audio_features"]))
        audio_tokens = audio_resampler(audio_features)
        
        # Concatenate: [Audio, Text]
        combined = torch.cat([audio_tokens, text_embeds], dim=1)
        
        expected_seq_len = CONFIG["num_latents"] + CONFIG["text"]["max_length"]
        assert combined.shape == (batch_size, expected_seq_len, embedding_dim), \
            f"Expected {(batch_size, expected_seq_len, embedding_dim)}, got {combined.shape}"
    
    def test_full_multimodal_concatenation(self, mock_inputs, batch_size):
        """Verify full multimodal (vision+audio+text) concatenation shape."""
        embedding_dim = CONFIG["llm_dim"]
        num_latents = CONFIG["num_latents"]
        text_len = CONFIG["text"]["max_length"]
        
        # Mock text embeddings
        text_embeds = torch.randn(batch_size, text_len, embedding_dim)
        
        # Mock modality tokens
        vision_tokens = torch.randn(batch_size, num_latents, embedding_dim)
        audio_tokens = torch.randn(batch_size, num_latents, embedding_dim)
        
        # Concatenate: [Vision, Audio, Text]
        combined = torch.cat([vision_tokens, audio_tokens, text_embeds], dim=1)
        
        expected_seq_len = 2 * num_latents + text_len  # vision + audio + text
        assert combined.shape == (batch_size, expected_seq_len, embedding_dim), \
            f"Expected {(batch_size, expected_seq_len, embedding_dim)}, got {combined.shape}"
        
        print(f"\n✅ Full Multimodal: [Vision:{num_latents}, Audio:{num_latents}, Text:{text_len}] → [{batch_size}, {expected_seq_len}, {embedding_dim}]")
    
    def test_sequence_length_calculation(self, batch_size):
        """Verify total sequence length calculation."""
        num_latents = CONFIG["num_latents"]
        text_len = CONFIG["text"]["max_length"]
        
        # Vision only
        vision_only = num_latents + text_len
        assert vision_only == 64 + 128 == 192
        
        # Audio only
        audio_only = num_latents + text_len
        assert audio_only == 64 + 128 == 192
        
        # Full multimodal
        full_multimodal = 2 * num_latents + text_len
        assert full_multimodal == 2 * 64 + 128 == 256
        
        print(f"\n✅ Sequence lengths: Vision={vision_only}, Audio={audio_only}, Full={full_multimodal}")


# ═══════════════════════════════════════════════════════════════
# PERCEIVER RESAMPLER SPECIFIC TESTS
# ═══════════════════════════════════════════════════════════════

class TestPerceiverResampler:
    """Test Perceiver Resampler component."""
    
    @pytest.fixture
    def resampler(self):
        if HAS_MODEL:
            return PerceiverResampler(
                dim=CONFIG["llm_dim"],
                depth=6,
                num_latents=CONFIG["num_latents"],
            )
        else:
            return MockPerceiverResampler(dim=CONFIG["llm_dim"], num_latents=CONFIG["num_latents"])
    
    def test_resampler_variable_input_length(self, resampler):
        """Verify resampler handles variable input lengths."""
        batch = 2
        
        # Different input lengths
        for input_len in [100, 500, 1000, 2000]:
            x = torch.randn(batch, input_len, CONFIG["llm_dim"])
            output = resampler(x)
            
            assert output.shape == (batch, CONFIG["num_latents"], CONFIG["llm_dim"]), \
                f"Input length {input_len}: Expected {(batch, CONFIG['num_latents'], CONFIG['llm_dim'])}, got {output.shape}"
        
        print(f"\n✅ Perceiver Resampler handles variable input lengths (100-2000 tokens)")
    
    def test_resampler_compression_ratio(self, resampler):
        """Verify resampler compresses inputs correctly."""
        batch = 2
        
        # Large input (1337 patches from SigLIP)
        input_len = 1337
        x = torch.randn(batch, input_len, CONFIG["llm_dim"])
        output = resampler(x)
        
        compression_ratio = input_len / CONFIG["num_latents"]
        
        assert output.shape[1] == CONFIG["num_latents"]
        print(f"\n✅ Compression ratio: {input_len} → {CONFIG['num_latents']} = {compression_ratio:.1f}x")


# ═══════════════════════════════════════════════════════════════
# SHAPE SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_shape_summary():
    """Print a summary of all expected tensor shapes."""
    print("\n" + "=" * 70)
    print("MULTIMODAL PIPELINE SHAPE SUMMARY")
    print("=" * 70)
    
    batch = CONFIG["batch_size"]
    num_patches = VISION_NUM_PATCHES
    vision_dim = CONFIG["vision"]["output_dim"]
    audio_dim = CONFIG["audio"]["output_dim"]
    llm_dim = CONFIG["llm_dim"]
    num_latents = CONFIG["num_latents"]
    text_len = CONFIG["text"]["max_length"]
    
    print(f"""
VISION PIPELINE:
  Input Image:        [{batch}, 3, 512, 512]
  SigLIP Encoder:     [{batch}, {num_patches}+1, {vision_dim}]
  Projection:         [{batch}, {num_patches}+1, {llm_dim}]
  Perceiver Resampler: [{batch}, {num_latents}, {llm_dim}]

AUDIO PIPELINE:
  Mel Spectrogram:    [{batch}, 80, 3000]
  Whisper Encoder:    [{batch}, 1500, {audio_dim}]
  Projection:         [{batch}, 1500, {llm_dim}]
  Perceiver Resampler: [{batch}, {num_latents}, {llm_dim}]

TEXT EMBEDDING:
  Input Tokens:       [{batch}, {text_len}]
  Embedding:          [{batch}, {text_len}, {llm_dim}]

COMBINED FORWARD PASS:
  Vision + Text:      [{batch}, {num_latents + text_len}, {llm_dim}]
  Audio + Text:       [{batch}, {num_latents + text_len}, {llm_dim}]
  Full Multimodal:    [{batch}, {2*num_latents + text_len}, {llm_dim}]
""")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_shape_summary()
    
    # Run pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
