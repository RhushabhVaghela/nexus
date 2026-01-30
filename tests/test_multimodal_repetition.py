"""
Tests for multimodal repetition (Paper 2512.14982).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    
    def test_process_text_repetition_3x(self):
        """Test 3x text repetition."""
        processor = MultimodalRepetitionProcessor()
        
        text = "What is this?"
        result = processor.process_text_repetition(text, repetition_factor=3)
        
        assert result.count(text) == 3
        assert "Let me repeat that" in result
        assert "one more time" in result
    
    def test_process_image_repetition_baseline(self):
        """Test image repetition baseline."""
        processor = MultimodalRepetitionProcessor()
        
        image_path = "test_image.jpg"
        result = processor.process_image_repetition(image_path, repetition_factor=1)
        
        assert result["images"] == [image_path]
        assert "repetition_metadata" not in result
    
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
    
    def test_process_image_repetition_capped(self):
        """Test that image repetition is capped."""
        processor = MultimodalRepetitionProcessor(max_image_repetitions=2)
        
        image_path = "test_image.jpg"
        result = processor.process_image_repetition(
            image_path,
            repetition_factor=5  # Request more than max
        )
        
        # Should be capped at max_image_repetitions
        assert len(result["images"]) == 2
    
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
        assert result["repetition_metadata"]["factor"] == 2
    
    def test_process_multimodal_prompt(self):
        """Test processing multimodal prompt."""
        processor = MultimodalRepetitionProcessor()
        
        result = processor.process_multimodal_prompt(
            text="Describe this:",
            images=["image1.jpg"],
            audio=["audio1.wav"],
            repetition_factor=2
        )
        
        assert "text" in result
        assert "images" in result
        assert "audio" in result
        assert "modality_repetitions" in result
        assert result["modality_repetitions"]["image"]["factor"] == 2
        assert result["modality_repetitions"]["audio"]["factor"] == 2


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
    
    def test_process_image_captioning(self):
        """Test image captioning."""
        processor = VisionPromptProcessor()
        
        result = processor.process_image_captioning(
            image_path="test.jpg",
            style="detailed",
            repetition_factor=1
        )
        
        assert "text" in result
        assert "Describe this image in detail" in result["text"]
        assert "images" in result


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
        assert "Transcribe this audio" in result["text"]
    
    def test_process_audio_transcription_with_language(self):
        """Test audio transcription with language hint."""
        processor = AudioPromptProcessor()
        
        result = processor.process_audio_transcription(
            audio_path="test.wav",
            language_hint="French",
            repetition_factor=1
        )
        
        assert "French" in result["text"]
    
    def test_process_audio_qa(self):
        """Test audio question answering."""
        processor = AudioPromptProcessor()
        
        result = processor.process_audio_qa(
            question="What is the speaker saying?",
            audio_path="test.wav",
            repetition_factor=2
        )
        
        assert "audio" in result
        assert len(result["audio"]) == 2


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
        assert result["modality_repetitions"]["image"]["factor"] == 2
    
    def test_create_fused_prompt_parallel(self):
        """Test creating fused prompt in parallel mode."""
        pipeline = MultimodalFusionPipeline()
        
        result = pipeline.create_fused_prompt(
            text="Analyze this:",
            images=["img1.jpg", "img2.jpg"],
            audio=["aud1.wav"],
            repetition_factor=2,
            fusion_mode="parallel"
        )
        
        assert "text" in result
        assert result["fusion_mode"] == "parallel"
        assert result["image_count"] == 4  # 2 images * 2 repetitions
    
    def test_process_batch(self):
        """Test batch processing."""
        pipeline = MultimodalFusionPipeline()
        
        prompts = [
            {
                "text": "First prompt",
                "images": ["img1.jpg"],
                "audio": None
            },
            {
                "text": "Second prompt",
                "images": None,
                "audio": ["aud1.wav"]
            }
        ]
        
        results = pipeline.process_batch(prompts, repetition_factor=2)
        
        assert len(results) == 2
        assert all("text" in r for r in results)


class TestMultimodalRepetitionStyles:
    """Test different repetition styles."""
    
    def test_image_descriptor_style(self):
        """Test image descriptor repetition style."""
        processor = MultimodalRepetitionProcessor(image_repetition_style="descriptor")
        
        result = processor.process_image_repetition(
            "test.jpg",
            image_description="A cat",
            repetition_factor=2
        )
        
        assert len(result["image_descriptions"]) == 2
    
    def test_image_detail_style(self):
        """Test image detail repetition style."""
        processor = MultimodalRepetitionProcessor(image_repetition_style="detail")
        
        result = processor.process_image_repetition(
            "test.jpg",
            image_description="A cat",
            repetition_factor=2
        )
        
        # Second description should mention "again"
        assert "again" in result["image_descriptions"][1].lower() or "same" in result["image_descriptions"][1].lower()
    
    def test_audio_transcript_style(self):
        """Test audio transcript repetition style."""
        processor = MultimodalRepetitionProcessor(audio_repetition_style="transcript")
        
        result = processor.process_audio_repetition(
            "test.wav",
            transcript="Hello world",
            repetition_factor=2
        )
        
        # Second transcript should mention replay
        assert "replay" in result["transcripts"][1].lower() or result["transcripts"][1] == result["transcripts"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])