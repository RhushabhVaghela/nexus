"""
Integration Tests for Multimodal with Adaptive Repetition (Papers 2512.14982).

Tests cover:
- Full multimodal pipeline
- Vision + text with adaptive repetition
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
from src.utils.repetition import (
    AdaptiveRepetitionRouter,
    TaskType,
    TaskComplexity,
    apply_adaptive
)


class TestMultimodalAdaptiveIntegration:
    """Test integration of multimodal and adaptive repetition."""
    
    def test_vision_with_adaptive_repetition(self):
        """Test vision tasks with adaptive repetition routing."""
        # Create adaptive router
        router = AdaptiveRepetitionRouter()
        
        # Vision QA query
        query = "What objects are present in this image?"
        
        # Get repetition factor
        factor = router.get_repetition_factor(query)
        
        # Process with vision processor
        vision_processor = VisionPromptProcessor()
        result = vision_processor.process_visual_qa(
            question=query,
            image_path="test_image.jpg",
            repetition_factor=factor
        )
        
        assert "text" in result
        assert "images" in result
        assert len(result["images"]) == factor
    
    def test_audio_with_adaptive_repetition(self):
        """Test audio tasks with adaptive repetition routing."""
        # Create adaptive router
        router = AdaptiveRepetitionRouter()
        
        # Audio transcription query
        query = "Transcribe this audio file completely"
        
        # Get repetition factor
        factor = router.get_repetition_factor(query)
        
        # Process with audio processor
        audio_processor = AudioPromptProcessor()
        result = audio_processor.process_audio_transcription(
            audio_path="test_audio.wav",
            repetition_factor=factor
        )
        
        assert "text" in result
        assert "audio" in result
        assert len(result["audio"]) == factor
    
    def test_multimodal_fusion_with_adaptive(self):
        """Test multimodal fusion with adaptive repetition."""
        # Setup
        router = AdaptiveRepetitionRouter()
        pipeline = MultimodalFusionPipeline()
        
        # Complex multimodal query
        query = "Find and analyze all objects in this image while transcribing the audio"
        
        # Get adaptive repetition factor
        factor = router.get_repetition_factor(query)
        
        # Create fused prompt
        result = pipeline.create_fused_prompt(
            text=query,
            images=["image1.jpg"],
            audio=["audio1.wav"],
            repetition_factor=factor,
            fusion_mode="sequential"
        )
        
        assert "text" in result
        assert "images" in result
        assert "audio" in result


class TestTaskTypeRoutingInMultimodal:
    """Test task type routing in multimodal context."""
    
    def test_retrieval_task_multimodal(self):
        """Test retrieval task routing with multimodal input."""
        router = AdaptiveRepetitionRouter()
        
        # Retrieval-type query with image
        query = "Find all similar images to this one in the database"
        config = router.route(query)
        
        assert config.task_type == TaskType.RETRIEVAL
        assert config.repetition_factor >= 2
    
    def test_qa_task_multimodal(self):
        """Test Q&A task routing with multimodal input."""
        router = AdaptiveRepetitionRouter()
        
        # Simple Q&A query
        query = "What color is the car in this image?"
        config = router.route(query)
        
        assert config.task_type == TaskType.Q_AND_A
    
    def test_complex_reasoning_multimodal(self):
        """Test complex reasoning with multimodal input."""
        router = AdaptiveRepetitionRouter()
        
        # Complex reasoning query
        query = "Analyze the relationship between the visual elements and the audio narrative in this video"
        config = router.route(query)
        
        assert config.complexity == TaskComplexity.COMPLEX
        assert config.repetition_factor >= 2


class TestEndToEndMultimodalAdaptive:
    """End-to-end tests for multimodal adaptive repetition."""
    
    def test_full_pipeline_vision_task(self):
        """Test full pipeline for vision task."""
        # Step 1: Analyze task
        router = AdaptiveRepetitionRouter()
        query = "Describe this image in complete detail"
        config = router.route(query)
        
        # Step 2: Process with appropriate repetition
        vision_processor = VisionPromptProcessor()
        result = vision_processor.process_image_captioning(
            image_path="test.jpg",
            style="detailed",
            repetition_factor=config.repetition_factor
        )
        
        # Verify result
        assert "text" in result
        assert "images" in result
        assert result.get("repetition_applied", True)
    
    def test_full_pipeline_audio_task(self):
        """Test full pipeline for audio task."""
        # Step 1: Analyze task
        router = AdaptiveRepetitionRouter()
        query = "Transcribe and summarize this audio recording"
        config = router.route(query)
        
        # Step 2: Process with appropriate repetition
        audio_processor = AudioPromptProcessor()
        result = audio_processor.process_audio_transcription(
            audio_path="test.wav",
            language_hint="English",
            repetition_factor=config.repetition_factor
        )
        
        # Verify result
        assert "text" in result
        assert "audio" in result
    
    def test_batch_multimodal_processing(self):
        """Test batch processing of multimodal inputs."""
        pipeline = MultimodalFusionPipeline()
        router = AdaptiveRepetitionRouter()
        
        # Batch of prompts
        prompts = [
            {"text": "What is this?", "images": ["img1.jpg"]},
            {"text": "Describe the scene", "images": ["img2.jpg"]},
            {"text": "Transcribe", "audio": ["aud1.wav"]}
        ]
        
        # Process each with adaptive repetition
        results = []
        for prompt in prompts:
            factor = router.get_repetition_factor(prompt["text"])
            result = pipeline.create_fused_prompt(
                text=prompt["text"],
                images=prompt.get("images"),
                audio=prompt.get("audio"),
                repetition_factor=factor
            )
            results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert "text" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
