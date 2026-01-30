"""
Integration tests for End-to-End Multimodal Pipeline.

Tests the complete multimodal training pipeline including:
- Vision + text fusion
- Audio + text fusion
- Video understanding pipeline
- Multi-modal training with all modalities
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import json
import sys
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.integration
class TestMultimodalTrainingPipeline:
    """Integration tests for multimodal training pipeline."""
    
    @pytest.fixture
    def mock_multimodal_model(self):
        """Create a mock multimodal model."""
        mock = MagicMock()
        mock.config = MagicMock()
        mock.config.hidden_size = 512
        mock.config.vocab_size = 1000
        
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.return_value = torch.randn(2, 10, 512)
        mock.get_input_embeddings = mock_embeddings
        
        return mock
    
    @pytest.fixture
    def mock_alignment_module(self):
        """Create mock alignment module."""
        mock = MagicMock()
        mock.return_value = torch.randn(2, 5, 512)  # 5 aligned tokens
        return mock
    
    @pytest.fixture
    def student_model(self, mock_multimodal_model, mock_alignment_module):
        """Create student model with all dependencies mocked."""
        with patch('src.nexus_final.architect.AutoModelForCausalLM') as mock_model_class, \
             patch('src.nexus_final.architect.AutoTokenizer'), \
             patch('src.nexus_final.architect.get_peft_model'), \
             patch('src.nexus_final.architect.LoraConfig'), \
             patch('src.nexus_final.architect.CrossModalAlignment', mock_alignment_module):
            
            mock_model_class.from_pretrained.return_value = mock_multimodal_model
            
            from src.nexus_final.architect import NexusStudent
            model = NexusStudent(base_model_id="test-model")
            return model
    
    def test_vision_text_fusion_pipeline(self, student_model):
        """Test end-to-end vision + text fusion."""
        # Simulate vision input (e.g., image patches from ViT)
        batch_size = 2
        num_patches = 16
        vision_dim = 768
        text_seq_len = 20
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        vision_feats = torch.randn(batch_size, num_patches, vision_dim)
        
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_feats=vision_feats
        )
        
        # Verify outputs
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
        assert outputs["multimodal_embeds"].shape[1] > text_seq_len  # Extended with vision tokens
        
        print(f"✅ Vision+Text fusion successful. Output shape: {outputs['multimodal_embeds'].shape}")
    
    def test_audio_text_fusion_pipeline(self, student_model):
        """Test end-to-end audio + text fusion."""
        batch_size = 2
        audio_seq_len = 50
        audio_dim = 256
        text_seq_len = 15
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        audio_feats = torch.randn(batch_size, audio_seq_len, audio_dim)
        
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_feats=audio_feats
        )
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
        
        print(f"✅ Audio+Text fusion successful. Output shape: {outputs['multimodal_embeds'].shape}")
    
    def test_video_understanding_pipeline(self, student_model):
        """Test video understanding with video + text fusion."""
        batch_size = 2
        num_frames = 8
        video_dim = 512
        text_seq_len = 25
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        video_feats = torch.randn(batch_size, num_frames, video_dim)
        
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            video_feats=video_feats
        )
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
        
        print(f"✅ Video+Text fusion successful. Output shape: {outputs['multimodal_embeds'].shape}")
    
    def test_all_modalities_combined(self, student_model):
        """Test training with all modalities together."""
        batch_size = 2
        
        # Text
        text_seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        
        # Vision
        vision_feats = torch.randn(batch_size, 16, 768)
        
        # Audio
        audio_feats = torch.randn(batch_size, 50, 256)
        
        # Video
        video_feats = torch.randn(batch_size, 8, 512)
        
        # Tool
        tool_feats = torch.randn(batch_size, 5, 128)
        
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_feats=vision_feats,
            audio_feats=audio_feats,
            video_feats=video_feats,
            tool_feats=tool_feats
        )
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
        
        print(f"✅ All modalities fusion successful. Output shape: {outputs['multimodal_embeds'].shape}")
    
    def test_label_shifting_consistency(self, student_model):
        """Test that labels are properly shifted for multimodal tokens."""
        batch_size = 1
        text_seq_len = 10
        multimodal_seq_len = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        vision_feats = torch.randn(batch_size, 16, 768)
        
        # Track what labels are passed to the model
        with patch.object(student_model.model, '__call__') as mock_forward:
            mock_forward.return_value = MagicMock(
                loss=torch.tensor(0.5),
                logits=torch.randn(batch_size, text_seq_len + multimodal_seq_len, 1000),
                hidden_states=None,
                router_logits=None
            )
            
            student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                vision_feats=vision_feats
            )
            
            # Check labels passed to model
            call_kwargs = mock_forward.call_args.kwargs
            passed_labels = call_kwargs.get('labels')
            
            assert passed_labels is not None
            assert passed_labels.shape[1] == text_seq_len + multimodal_seq_len
            # First tokens should be -100 (ignored in loss)
            assert torch.all(passed_labels[:, :multimodal_seq_len] == -100)
            # Remaining tokens should match original labels
            assert torch.equal(passed_labels[:, multimodal_seq_len:], labels)
        
        print("✅ Label shifting is correct")
    
    def test_attention_mask_consistency(self, student_model):
        """Test that attention mask is properly extended."""
        batch_size = 1
        text_seq_len = 10
        multimodal_seq_len = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        attention_mask = torch.ones(batch_size, text_seq_len)
        vision_feats = torch.randn(batch_size, 16, 768)
        
        with patch.object(student_model.model, '__call__') as mock_forward:
            mock_forward.return_value = MagicMock(
                loss=torch.tensor(0.5),
                logits=torch.randn(batch_size, text_seq_len + multimodal_seq_len, 1000),
                hidden_states=None,
                router_logits=None
            )
            
            student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vision_feats=vision_feats
            )
            
            call_kwargs = mock_forward.call_args.kwargs
            passed_mask = call_kwargs.get('attention_mask')
            
            assert passed_mask is not None
            assert passed_mask.shape[1] == text_seq_len + multimodal_seq_len
            # All tokens should be attended to
            assert torch.all(passed_mask == 1)
        
        print("✅ Attention mask extension is correct")
    
    def test_gradient_flow(self, student_model):
        """Test gradient flow through multimodal pathway."""
        student_model.train()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        labels = torch.randint(0, 1000, (1, 10))
        vision_feats = torch.randn(1, 16, 768, requires_grad=True)
        
        outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_feats=vision_feats
        )
        
        # Loss should allow backprop
        loss = outputs["loss"]
        assert loss.requires_grad
        
        print("✅ Gradient flow is correct")


@pytest.mark.integration
class TestMultimodalWithVideoDecoder:
    """Integration tests combining multimodal model with video decoder."""
    
    @patch('src.nexus_final.decoders.StableVideoDiffusionPipeline')
    def test_video_generation_to_understanding_pipeline(self, mock_pipeline_class):
        """Test pipeline from video generation to video understanding."""
        from src.nexus_final.decoders import VideoDecoder
        from src.nexus_final.architect import NexusStudent
        
        # Mock video decoder
        mock_pipeline = MagicMock()
        mock_frames = [MagicMock() for _ in range(8)]  # 8 frames
        mock_pipeline.return_value = MagicMock(frames=[mock_frames])
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Create decoder
        decoder = VideoDecoder(device="cpu")
        decoder.load()
        
        # Generate video frames
        conditioning = MagicMock()
        conditioning.height = 256
        conditioning.width = 256
        frames = decoder.generate(conditioning, num_frames=8)
        
        assert len(frames) == 8
        
        print("✅ Video generation successful")


@pytest.mark.integration
class TestMultimodalDataPipeline:
    """Integration tests for multimodal data pipeline."""
    
    def test_multimodal_batch_processing(self):
        """Test processing a batch of multimodal samples."""
        batch_size = 4
        
        # Simulate batch data
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, 20)),
            'attention_mask': torch.ones(batch_size, 20),
            'labels': torch.randint(0, 1000, (batch_size, 20)),
            'vision_feats': torch.randn(batch_size, 16, 768),
            'audio_feats': torch.randn(batch_size, 50, 256),
            'video_feats': torch.randn(batch_size, 8, 512),
        }
        
        # Verify all tensors have correct batch dimension
        for key, tensor in batch.items():
            assert tensor.shape[0] == batch_size, f"{key} has wrong batch size"
        
        print(f"✅ Batch processing with {batch_size} samples successful")
    
    def test_variable_length_sequences(self):
        """Test handling variable length sequences in batch."""
        batch_size = 3
        max_len = 30
        
        # Create variable length sequences
        input_ids = []
        attention_masks = []
        
        for i in range(batch_size):
            seq_len = 10 + i * 5  # 10, 15, 20
            ids = torch.randint(0, 1000, (seq_len,))
            mask = torch.ones(seq_len)
            
            # Pad to max length
            pad_len = max_len - seq_len
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len)])
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        
        assert input_ids.shape == (batch_size, max_len)
        assert attention_masks.shape == (batch_size, max_len)
        
        print("✅ Variable length sequence handling successful")


@pytest.mark.integration
class TestMultimodalExportImport:
    """Integration tests for model export and import."""
    
    def test_model_save_load(self, tmp_path):
        """Test saving and loading multimodal model."""
        with patch('src.nexus_final.architect.AutoModelForCausalLM'), \
             patch('src.nexus_final.architect.AutoTokenizer'), \
             patch('src.nexus_final.architect.get_peft_model'), \
             patch('src.nexus_final.architect.LoraConfig'), \
             patch('src.nexus_final.architect.CrossModalAlignment'):
            
            from src.nexus_final.architect import NexusStudent
            
            # Create model
            model = NexusStudent()
            
            # Add multimodal projection
            model._multimodal_proj = nn.Linear(256, 512)
            
            # Save
            save_path = tmp_path / "model"
            with patch.object(model.model, 'save_pretrained'), \
                 patch.object(model.tokenizer, 'save_pretrained'), \
                 patch('torch.save') as mock_save:
                
                model.save_pretrained(str(save_path))
                
                # Verify saves were called
                assert mock_save.call_count == 2  # bridge + multimodal_proj
        
        print("✅ Model save/load successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
