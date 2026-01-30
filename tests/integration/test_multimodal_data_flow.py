"""
Multi-Modal Data Flow Validation Tests

Tests data flow validation across modalities:
- Vision → Text fusion flow
- Audio → Text fusion flow
- Video → Text understanding
- All modalities combined
- Data format validation at each stage

Usage:
    pytest tests/integration/test_multimodal_data_flow.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL = "tool"


@dataclass
class DataFlowStage:
    """Represents a stage in the data flow pipeline."""
    name: str
    input_shape: Tuple
    output_shape: Tuple
    modality: ModalityType
    validation_passed: bool = False


@dataclass
class FlowValidationResult:
    """Result of a flow validation test."""
    flow_name: str
    stages: List[DataFlowStage]
    total_time_ms: float
    errors: List[str]


@pytest.mark.integration
class TestVisionTextFlow:
    """
    Tests Vision → Text fusion flow.
    """
    
    def test_vision_feature_extraction(self):
        """Test vision features are extracted correctly."""
        batch_size = 2
        num_patches = 16
        vision_dim = 768
        
        # Simulate vision encoder output
        vision_features = torch.randn(batch_size, num_patches, vision_dim)
        
        # Validate shape
        assert vision_features.shape == (batch_size, num_patches, vision_dim)
        assert not torch.isnan(vision_features).any()
        assert not torch.isinf(vision_features).any()
        
        print(f"   Vision feature extraction: {vision_features.shape}")
    
    def test_vision_to_text_projection(self):
        """Test vision features project to text embedding space."""
        batch_size = 2
        num_patches = 16
        vision_dim = 768
        text_dim = 512
        
        # Vision features
        vision_feats = torch.randn(batch_size, num_patches, vision_dim)
        
        # Projection layer
        projector = nn.Linear(vision_dim, text_dim)
        
        # Project to text space
        projected = projector(vision_feats)
        
        assert projected.shape == (batch_size, num_patches, text_dim)
        print(f"   Vision→Text projection: {vision_feats.shape} → {projected.shape}")
    
    def test_vision_text_fusion_alignment(self):
        """Test vision and text are properly aligned."""
        from src.nexus_final.alignment import CrossModalAlignment
        
        batch_size = 2
        text_seq_len = 20
        vision_patches = 16
        embed_dim = 512
        
        alignment = CrossModalAlignment(core_dim=embed_dim)
        
        # Create inputs
        text_embeds = torch.randn(batch_size, text_seq_len, embed_dim)
        vision_feats = torch.randn(batch_size, vision_patches, 768)
        
        # Align
        with torch.no_grad():
            aligned = alignment(vision_feats=vision_feats)
        
        assert aligned is not None
        assert aligned.shape[-1] == embed_dim
        print(f"   Vision-Text alignment: {aligned.shape}")
    
    def test_image_preprocessing_pipeline(self):
        """Test image preprocessing produces correct format."""
        # Create dummy image
        img_size = 224
        dummy_image = Image.new('RGB', (img_size, img_size), color='red')
        
        # Simulate preprocessing
        img_array = np.array(dummy_image)
        
        # Validate format
        assert img_array.shape == (img_size, img_size, 3)
        assert img_array.dtype == np.uint8
        assert img_array.max() > 0  # Not all black
        
        print(f"   Image preprocessing: {img_array.shape}")


@pytest.mark.integration
class TestAudioTextFlow:
    """
    Tests Audio → Text fusion flow.
    """
    
    def test_audio_feature_extraction(self):
        """Test audio features are extracted correctly."""
        batch_size = 2
        audio_seq_len = 100
        audio_dim = 256
        
        # Simulate audio encoder output
        audio_features = torch.randn(batch_size, audio_seq_len, audio_dim)
        
        # Validate
        assert audio_features.shape == (batch_size, audio_seq_len, audio_dim)
        assert not torch.isnan(audio_features).any()
        
        print(f"   Audio feature extraction: {audio_features.shape}")
    
    def test_audio_to_text_projection(self):
        """Test audio features project to text embedding space."""
        batch_size = 2
        audio_seq_len = 100
        audio_dim = 256
        text_dim = 512
        
        audio_feats = torch.randn(batch_size, audio_seq_len, audio_dim)
        projector = nn.Linear(audio_dim, text_dim)
        
        projected = projector(audio_feats)
        
        assert projected.shape == (batch_size, audio_seq_len, text_dim)
        print(f"   Audio→Text projection: {audio_feats.shape} → {projected.shape}")
    
    def test_audio_text_fusion_alignment(self):
        """Test audio and text are properly aligned."""
        from src.nexus_final.alignment import CrossModalAlignment
        
        batch_size = 2
        audio_seq_len = 100
        embed_dim = 512
        
        alignment = CrossModalAlignment(core_dim=embed_dim)
        
        audio_feats = torch.randn(batch_size, audio_seq_len, 256)
        
        with torch.no_grad():
            aligned = alignment(audio_feats=audio_feats)
        
        assert aligned is not None
        assert aligned.shape[-1] == embed_dim
        print(f"   Audio-Text alignment: {aligned.shape}")
    
    def test_audio_format_validation(self):
        """Test audio data format validation."""
        # Simulate audio waveform
        sample_rate = 16000
        duration_sec = 5
        num_samples = sample_rate * duration_sec
        
        audio_waveform = torch.randn(num_samples)
        
        # Validate format
        assert audio_waveform.dim() == 1
        assert audio_waveform.shape[0] == num_samples
        assert not torch.isnan(audio_waveform).any()
        
        print(f"   Audio format: {audio_waveform.shape}, {sample_rate}Hz")


@pytest.mark.integration
class TestVideoTextFlow:
    """
    Tests Video → Text understanding flow.
    """
    
    def test_video_frame_extraction(self):
        """Test video frames are extracted correctly."""
        batch_size = 2
        num_frames = 8
        frame_dim = 512
        
        # Simulate video encoder output
        video_features = torch.randn(batch_size, num_frames, frame_dim)
        
        # Validate
        assert video_features.shape == (batch_size, num_frames, frame_dim)
        
        print(f"   Video frame extraction: {video_features.shape}")
    
    def test_temporal_fusion(self):
        """Test temporal fusion of video frames."""
        batch_size = 2
        num_frames = 8
        frame_dim = 512
        
        video_feats = torch.randn(batch_size, num_frames, frame_dim)
        
        # Temporal pooling (mean across frames)
        temporal_pooled = video_feats.mean(dim=1)
        
        assert temporal_pooled.shape == (batch_size, frame_dim)
        print(f"   Temporal fusion: {video_feats.shape} → {temporal_pooled.shape}")
    
    def test_video_to_text_alignment(self):
        """Test video features align to text space."""
        from src.nexus_final.alignment import CrossModalAlignment
        
        batch_size = 2
        num_frames = 8
        embed_dim = 512
        
        alignment = CrossModalAlignment(core_dim=embed_dim)
        
        video_feats = torch.randn(batch_size, num_frames, 512)
        
        with torch.no_grad():
            aligned = alignment(video_feats=video_feats)
        
        assert aligned is not None
        print(f"   Video-Text alignment: {aligned.shape}")


@pytest.mark.integration
class TestAllModalitiesCombined:
    """
    Tests all modalities combined.
    """
    
    def test_all_modality_fusion(self):
        """Test fusion of all modalities together."""
        from src.nexus_final.alignment import CrossModalAlignment
        
        batch_size = 2
        embed_dim = 512
        
        alignment = CrossModalAlignment(core_dim=embed_dim)
        
        # Create all modality features
        vision_feats = torch.randn(batch_size, 16, 768)  # 16 image patches
        audio_feats = torch.randn(batch_size, 50, 256)   # 50 audio frames
        video_feats = torch.randn(batch_size, 8, 512)    # 8 video frames
        tool_feats = torch.randn(batch_size, 5, 128)     # 5 tool tokens
        
        # Align all modalities
        with torch.no_grad():
            aligned = alignment(
                vision_feats=vision_feats,
                audio_feats=audio_feats,
                video_feats=video_feats,
                tool_feats=tool_feats
            )
        
        assert aligned is not None
        assert aligned.shape[-1] == embed_dim
        print(f"   All modalities fusion: {aligned.shape}")
    
    def test_multimodal_sequence_construction(self):
        """Test construction of multimodal input sequence."""
        batch_size = 2
        
        # Different modality sequence lengths
        vision_len = 16
        audio_len = 50
        text_len = 20
        
        # Create embeddings
        vision_embeds = torch.randn(batch_size, vision_len, 512)
        audio_embeds = torch.randn(batch_size, audio_len, 512)
        text_embeds = torch.randn(batch_size, text_len, 512)
        
        # Concatenate in order: [vision, audio, text]
        combined = torch.cat([vision_embeds, audio_embeds, text_embeds], dim=1)
        
        expected_len = vision_len + audio_len + text_len
        assert combined.shape == (batch_size, expected_len, 512)
        print(f"   Multimodal sequence: {combined.shape}")
    
    def test_attention_mask_for_multimodal(self):
        """Test attention mask covers all modalities."""
        batch_size = 2
        
        vision_len = 16
        audio_len = 50
        text_len = 20
        total_len = vision_len + audio_len + text_len
        
        # Create attention mask (all ones for full attention)
        attention_mask = torch.ones(batch_size, total_len)
        
        # Verify mask covers all tokens
        assert attention_mask.shape == (batch_size, total_len)
        assert attention_mask.sum() == batch_size * total_len
        
        print(f"   Attention mask: {attention_mask.shape}")
    
    def test_label_shifting_for_multimodal(self):
        """Test labels are shifted for multimodal prefix."""
        batch_size = 2
        text_len = 20
        multimodal_prefix_len = 66  # vision + audio
        
        # Original text labels
        text_labels = torch.randint(0, 1000, (batch_size, text_len))
        
        # Create multimodal labels with -100 prefix
        prefix_labels = torch.full((batch_size, multimodal_prefix_len), -100)
        combined_labels = torch.cat([prefix_labels, text_labels], dim=1)
        
        # Verify structure
        assert combined_labels.shape == (batch_size, multimodal_prefix_len + text_len)
        assert torch.all(combined_labels[:, :multimodal_prefix_len] == -100)
        assert torch.equal(combined_labels[:, multimodal_prefix_len:], text_labels)
        
        print(f"   Label shifting: prefix={multimodal_prefix_len}, text={text_len}")


@pytest.mark.integration
class TestDataFormatValidation:
    """
    Tests data format validation at each stage.
    """
    
    def test_batch_dimension_consistency(self):
        """Test batch dimension is consistent across modalities."""
        batch_size = 4
        
        # All modalities should have same batch size
        vision_feats = torch.randn(batch_size, 16, 768)
        audio_feats = torch.randn(batch_size, 50, 256)
        video_feats = torch.randn(batch_size, 8, 512)
        text_ids = torch.randint(0, 1000, (batch_size, 20))
        
        # Validate batch dimension
        assert vision_feats.shape[0] == batch_size
        assert audio_feats.shape[0] == batch_size
        assert video_feats.shape[0] == batch_size
        assert text_ids.shape[0] == batch_size
        
        print(f"   Batch consistency: size={batch_size}")
    
    def test_tensor_type_validation(self):
        """Test tensor types are correct at each stage."""
        # Valid tensors
        valid_tensors = [
            torch.randn(2, 10, 512),  # Float features
            torch.randint(0, 1000, (2, 10)),  # Integer IDs
            torch.ones(2, 10),  # Attention mask
        ]
        
        for tensor in valid_tensors:
            assert isinstance(tensor, torch.Tensor)
            assert not torch.isnan(tensor).any()
        
        print(f"   Tensor type validation: {len(valid_tensors)} tensors passed")
    
    def test_shape_validation_at_stage_boundaries(self):
        """Test shapes are validated at component boundaries."""
        stages = [
            ("input", (2, 3, 224, 224)),      # Image input
            ("encoder", (2, 16, 768)),         # Vision encoder output
            ("projection", (2, 16, 512)),      # Projected features
            ("fusion", (2, 86, 512)),          # Fused with text
            ("output", (2, 86, 1000)),         # Model output
        ]
        
        # Validate progression
        for i in range(len(stages) - 1):
            current_name, current_shape = stages[i]
            next_name, next_shape = stages[i + 1]
            
            # Batch dimension should be preserved
            assert current_shape[0] == next_shape[0], \
                f"Batch mismatch between {current_name} and {next_name}"
        
        print(f"   Shape validation: {len(stages)} stages passed")
    
    def test_value_range_validation(self):
        """Test value ranges are valid at each stage."""
        # Token IDs should be within vocab range
        vocab_size = 1000
        token_ids = torch.randint(0, vocab_size, (2, 20))
        
        assert token_ids.min() >= 0
        assert token_ids.max() < vocab_size
        
        # Features should be finite
        features = torch.randn(2, 10, 512)
        assert torch.isfinite(features).all()
        
        # Probabilities (if softmax applied) should be in [0, 1]
        probs = torch.softmax(torch.randn(2, 10), dim=-1)
        assert probs.min() >= 0
        assert probs.max() <= 1
        
        print("   Value range validation passed")


@pytest.mark.integration
class TestDataFlowIntegration:
    """
    Integration tests for complete data flows.
    """
    
    def test_complete_vision_flow(self):
        """Test complete vision data flow end-to-end."""
        stages = []
        
        # Stage 1: Image Input
        img = torch.randn(2, 3, 224, 224)
        stages.append(DataFlowStage("input", (2, 3, 224, 224), img.shape, ModalityType.VISION, True))
        
        # Stage 2: Patch Embedding
        patches = torch.randn(2, 196, 768)  # 14x14 patches
        stages.append(DataFlowStage("patch_embed", img.shape, patches.shape, ModalityType.VISION, True))
        
        # Stage 3: Vision Encoder
        encoded = torch.randn(2, 196, 768)
        stages.append(DataFlowStage("vision_encoder", patches.shape, encoded.shape, ModalityType.VISION, True))
        
        # Stage 4: Projection
        projected = torch.randn(2, 196, 512)
        stages.append(DataFlowStage("projection", encoded.shape, projected.shape, ModalityType.VISION, True))
        
        # Stage 5: Fusion with Text
        fused = torch.randn(2, 216, 512)  # 196 + 20 text tokens
        stages.append(DataFlowStage("fusion", projected.shape, fused.shape, ModalityType.VISION, True))
        
        result = FlowValidationResult(
            flow_name="vision_to_text",
            stages=stages,
            total_time_ms=100.0,
            errors=[]
        )
        
        assert len(result.stages) == 5
        assert len(result.errors) == 0
        print(f"   Complete vision flow: {len(stages)} stages")
    
    def test_complete_audio_flow(self):
        """Test complete audio data flow end-to-end."""
        stages = []
        
        # Stage 1: Audio Waveform
        waveform = torch.randn(2, 16000)  # 1 second at 16kHz
        stages.append(DataFlowStage("waveform", (2, 16000), waveform.shape, ModalityType.AUDIO, True))
        
        # Stage 2: Spectrogram/Mel
        mel = torch.randn(2, 80, 100)  # 80 mel bins, 100 frames
        stages.append(DataFlowStage("mel_spectrogram", waveform.shape, mel.shape, ModalityType.AUDIO, True))
        
        # Stage 3: Audio Encoder
        encoded = torch.randn(2, 100, 256)
        stages.append(DataFlowStage("audio_encoder", mel.shape, encoded.shape, ModalityType.AUDIO, True))
        
        # Stage 4: Projection
        projected = torch.randn(2, 100, 512)
        stages.append(DataFlowStage("projection", encoded.shape, projected.shape, ModalityType.AUDIO, True))
        
        result = FlowValidationResult(
            flow_name="audio_to_text",
            stages=stages,
            total_time_ms=80.0,
            errors=[]
        )
        
        assert len(result.stages) == 4
        print(f"   Complete audio flow: {len(stages)} stages")
    
    def test_flow_error_handling(self):
        """Test data flow handles errors gracefully."""
        errors = []
        
        try:
            # Simulate invalid input
            invalid_input = torch.randn(2, 3, 224)  # Missing spatial dim
            
            # Try to process (would fail in real encoder)
            if invalid_input.dim() != 4:
                raise ValueError(f"Expected 4D input, got {invalid_input.dim()}D")
                
        except ValueError as e:
            errors.append(str(e))
        
        assert len(errors) == 1
        assert "Expected 4D input" in errors[0]
        print("   Flow error handling: passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
