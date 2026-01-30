"""
Comprehensive Unit Tests for Multimodal Repetition (Paper 2512.14982).

Tests cover:
- Image repetition
- Audio repetition
- Fusion pipeline
- Different repetition styles
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.multimodal.processors import (
    MultimodalRepetitionProcessor,
    VisionPromptProcessor,
    AudioPromptProcessor,
    MultimodalFusionPipeline
)


class TestMultimodalRepetitionProcessor:
    """Test suite for MultimodalRepetitionProcessor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = MultimodalRepetitionProcessor()
        
        assert processor.image_repetition_style == "descriptor"
        assert processor.audio_repetition_style == "transcript"
        assert processor.max_image_repetitions == 2
        assert processor.max_audio_repetitions == 2
    
    def test_process_text_repetition_baseline(self):
        """Test text repetition baseline."""
        processor = MultimodalRepetitionProcessor()
        
        text = "What is this?"
        result = processor.process_text_repetition(text, repetition_factor=1)
        
        assert result == text
    
    def test_process_text_repetition_2x(self):
        """Test 2x text repetition."""
        processor = MultimodalRepetitionProcessor()
        
        text = "What is this?"
        result = processor.process_text_repetition(text, repetition_factor=2)
        
        assert result.count(text) == 2
        assert "Let me repeat that" in result
    
    def test_process_image_repetition_baseline(self):
        """Test image repetition baseline."""
        processor = MultimodalRepetitionProcessor()
        
        image_path = "test_image.jpg"
        result = processor.process_image_repetition(image_path, repetition_factor=1)
        
        assert result["images"] == [image_path]
    
    def test_process_image_repetition_2x(self):
        """Test 2x image repetition."""
        processor = MultimodalRepetitionProcessor()
        
        image_path = "test_image.jpg"
        description = "A test image"
        result = processor.process_image_repetition(
            image_path, 
            image_description=description,
            repetition_factor=2
        )
        
        assert len(result["images"]) == 2
        assert len(result["image_descriptions"]) == 2
        assert result["repetition_metadata"]["factor"] == 2
    
    def test_process_audio_repetition_baseline(self):
        """Test audio repetition baseline."""
        processor = MultimodalRepetitionProcessor()
        
        audio_path = "test_audio.wav"
        result = processor.process_audio_repetition(audio_path, repetition_factor=1)
        
        assert result["audio"] == [audio_path]
    
    def test_process_audio_repetition_2x(self):
        """Test 2x audio repetition."""
        processor = MultimodalRepetitionProcessor()
        
        audio_path = "test_audio.wav"
        transcript = "Hello world"
        result = processor.process_audio_repetition(
            audio_path,
            transcript=transcript,
            repetition_factor=2
        )
        
        assert len(result["audio"]) == 2
        assert len(result["transcripts"]) == 2


class TestVisionPromptProcessor:
    """Test suite for VisionPromptProcessor."""
    
    def test_process_visual_qa(self):
        """Test visual question answering."""
        processor = VisionPromptProcessor()
        
        result = processor.process_visual_qa(
            question="What is in this image?",
            image_path="test.jpg",
            repetition_factor=2
        )
        
        assert "text" in result
        assert "images" in result
        assert len(result["images"]) == 2


class TestAudioPromptProcessor:
    """Test suite for AudioPromptProcessor."""
    
    def test_process_audio_transcription(self):
        """Test audio transcription."""
        processor = AudioPromptProcessor()
        
        result = processor.process_audio_transcription(
            audio_path="test.wav",
            repetition_factor=2
        )
        
        assert "text" in result
        assert "audio" in result


class TestMultimodalFusionPipeline:
    """Test suite for MultimodalFusionPipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = MultimodalFusionPipeline()
        
        assert pipeline.processor is not None
        assert pipeline.vision_processor is not None
        assert pipeline.audio_processor is not None
    
    def test_create_fused_prompt_sequential(self):
        """Test creating fused prompt in sequential mode."""
        pipeline = MultimodalFusionPipeline()
        
        result = pipeline.create_fused_prompt(
            text="Analyze this:",
            images=["img1.jpg"],
            audio=["aud1.wav"],
            repetition_factor=2,
            fusion_mode="sequential"
        )
        
        assert "text" in result
        assert "images" in result
        assert "audio" in result
    
    def test_process_batch(self):
        """Test batch processing."""
        pipeline = MultimodalFusionPipeline()
        
        prompts = [
            {"text": "First prompt", "images": ["img1.jpg"]},
            {"text": "Second prompt", "audio": ["aud1.wav"]}
        ]
        
        results = pipeline.process_batch(prompts, repetition_factor=2)
        
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
