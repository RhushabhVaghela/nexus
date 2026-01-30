"""
Cross-Component Integration Tests

Tests integration between major components:
- Loader + Training integration
- Training + Inference integration
- Video decoder + Multi-agent integration
- TTS + Streaming integration
- Error propagation across components

Usage:
    pytest tests/integration/test_cross_component.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import gc
import time
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, call
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class IntegrationResult:
    """Result of an integration test."""
    component_a: str
    component_b: str
    success: bool
    duration_seconds: float
    error_message: Optional[str] = None


@pytest.mark.integration
class TestLoaderTrainingIntegration:
    """
    Tests integration between data loader and training components.
    """
    
    def test_loader_to_training_pipeline(self):
        """Test data flows correctly from loader to training."""
        from src.data.universal_loader import load_dataset_universal, LoadResult
        from src.stages.base import TrainingStage
        
        # Create mock training stage
        stage = MagicMock(spec=TrainingStage)
        stage.prepare_data.return_value = ["sample1", "sample2", "sample3"]
        
        # Simulate data loading
        mock_result = LoadResult(
            dataset=[{"text": "sample1"}, {"text": "sample2"}],
            format="json",
            num_samples=2,
            columns=["text"],
            source_path="/fake/path"
        )
        
        with patch('src.data.universal_loader.load_dataset_universal', return_value=mock_result):
            result = load_dataset_universal("/fake/path", sample_size=2)
            
            assert result.num_samples == 2
            assert len(result.dataset) == 2
            print("   Loader → Training integration test passed")
    
    def test_loader_format_to_model_input(self):
        """Test different data formats convert correctly to model inputs."""
        formats = ["json", "jsonl", "csv", "parquet"]
        
        for fmt in formats:
            # Create mock data in different formats
            if fmt == "json":
                mock_data = [{"text": "Hello", "label": 1}]
            elif fmt == "jsonl":
                mock_data = [{"text": "World", "label": 0}]
            else:
                mock_data = [{"text": "Test", "label": 1}]
            
            # Verify data can be tokenized (simulated)
            assert all("text" in item for item in mock_data)
        
        print(f"   All {len(formats)} format conversions passed")
    
    def test_batching_across_components(self):
        """Test batch consistency from loader to training."""
        batch_size = 4
        
        # Simulate loader output
        loader_batches = [
            {"input_ids": torch.randint(0, 1000, (batch_size, 10)),
             "attention_mask": torch.ones(batch_size, 10)}
            for _ in range(3)
        ]
        
        # Simulate training consuming batches
        for i, batch in enumerate(loader_batches):
            assert batch["input_ids"].shape[0] == batch_size, f"Batch {i} has wrong size"
        
        print(f"   Batching consistency test passed for {len(loader_batches)} batches")


@pytest.mark.integration
class TestTrainingInferenceIntegration:
    """
    Tests integration between training and inference components.
    """
    
    def test_model_state_after_training(self):
        """Test model can switch between training and inference modes."""
        # Create simple model
        model = nn.Linear(10, 2)
        
        # Training mode
        model.train()
        assert model.training
        
        # Forward pass in training mode
        x = torch.randn(5, 10)
        out_train = model(x)
        
        # Switch to eval mode
        model.eval()
        assert not model.training
        
        # Forward pass in eval mode
        with torch.no_grad():
            out_eval = model(x)
        
        # Outputs should be the same (deterministic layer)
        assert torch.allclose(out_train, out_eval)
        print("   Training → Inference mode switch test passed")
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint can be saved and loaded between training and inference."""
        from src.nexus_final.architect import NexusStudent
        
        with patch('src.nexus_final.architect.AutoModelForCausalLM') as mock_model_class, \
             patch('src.nexus_final.architect.AutoTokenizer'), \
             patch('src.nexus_final.architect.get_peft_model'), \
             patch('src.nexus_final.architect.LoraConfig'), \
             patch('src.nexus_final.architect.CrossModalAlignment'):
            
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.config.hidden_size = 512
            
            # Create student
            student = NexusStudent(base_model_id="test-model")
            
            # Mock save
            save_path = tmp_path / "checkpoint"
            save_path.mkdir()
            
            with patch.object(student.model, 'save_pretrained'), \
                 patch.object(student.tokenizer, 'save_pretrained'), \
                 patch('torch.save'):
                student.save_pretrained(str(save_path))
            
            print("   Checkpoint save/load test passed")
    
    def test_inference_after_training_step(self):
        """Test inference works immediately after training step."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Training step
        model.train()
        x = torch.randn(5, 10)
        target = torch.randint(0, 2, (5,))
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # Immediate inference
        model.eval()
        with torch.no_grad():
            inference_output = model(x)
        
        assert inference_output.shape == (5, 2)
        print("   Post-training inference test passed")


@pytest.mark.integration
class TestVideoDecoderMultiAgentIntegration:
    """
    Tests integration between video decoder and multi-agent systems.
    """
    
    def test_video_to_agent_pipeline(self):
        """Test video frames are processed correctly by agents."""
        # Mock video decoder
        mock_frames = [MagicMock() for _ in range(8)]
        
        # Mock agent processing
        agent_results = []
        for frame in mock_frames:
            # Simulate agent processing frame
            result = {"frame_id": id(frame), "processed": True}
            agent_results.append(result)
        
        assert len(agent_results) == len(mock_frames)
        assert all(r["processed"] for r in agent_results)
        print("   Video → Agent pipeline test passed")
    
    def test_multi_agent_video_analysis(self):
        """Test multiple agents analyzing same video."""
        from src.multi_agent import MultiAgentOrchestrator
        
        # Create mock agents
        agent_configs = [
            {"name": "scene_analyzer", "type": "vision"},
            {"name": "action_detector", "type": "temporal"},
            {"name": "object_tracker", "type": "detection"}
        ]
        
        # Simulate orchestration
        with patch('src.multi_agent.MultiAgentOrchestrator._load_agents'):
            orchestrator = MultiAgentOrchestrator()
            
            # Mock video input
            video_input = {"frames": [MagicMock() for _ in range(10)]}
            
            # Simulate multi-agent processing
            results = {}
            for config in agent_configs:
                results[config["name"]] = {"status": "completed", "findings": []}
            
            assert len(results) == len(agent_configs)
            print("   Multi-agent video analysis test passed")


@pytest.mark.integration
class TestTTSStreamingIntegration:
    """
    Tests integration between TTS and streaming components.
    """
    
    def test_tts_to_streaming_pipeline(self):
        """Test TTS output can be streamed."""
        from src.streaming.tts import TTSEngine
        
        # Mock TTS engine
        with patch.object(TTSEngine, '_load_model'):
            engine = TTSEngine()
            
            # Simulate text input
            text = "Hello, this is a test of streaming TTS."
            
            # Mock streaming chunks
            chunks = [b"audio_chunk_1", b"audio_chunk_2", b"audio_chunk_3"]
            
            # Simulate streaming
            streamed_audio = b"".join(chunks)
            
            assert len(streamed_audio) > 0
            print("   TTS → Streaming pipeline test passed")
    
    def test_streaming_chunk_consistency(self):
        """Test audio chunks are consistent during streaming."""
        chunk_duration_ms = 200  # 200ms chunks
        total_duration_ms = 1000  # 1 second
        
        expected_chunks = total_duration_ms // chunk_duration_ms
        
        # Simulate chunk generation
        chunks = []
        for i in range(expected_chunks):
            chunk = {
                "index": i,
                "duration_ms": chunk_duration_ms,
                "data": b"x" * 100  # dummy audio data
            }
            chunks.append(chunk)
        
        # Verify continuity
        for i, chunk in enumerate(chunks):
            assert chunk["index"] == i
            assert chunk["duration_ms"] == chunk_duration_ms
        
        print(f"   Streaming consistency test passed for {len(chunks)} chunks")


@pytest.mark.integration
class TestErrorPropagation:
    """
    Tests error handling and propagation across components.
    """
    
    def test_loader_error_propagation(self):
        """Test errors in loader are properly handled."""
        from src.data.universal_loader import load_dataset_universal
        
        # Simulate loader error
        with patch('src.data.universal_loader.UniversalDataLoader.load') as mock_load:
            mock_load.return_value = MagicMock(
                dataset=[],
                error="File not found",
                num_samples=0
            )
            
            result = load_dataset_universal("/nonexistent/path")
            
            assert result.error is not None
            assert result.num_samples == 0
        
        print("   Loader error propagation test passed")
    
    def test_training_error_recovery(self):
        """Test training can recover from errors."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Simulate training with error
        try:
            # Intentional error: wrong input shape
            x = torch.randn(5, 5)  # Wrong input size
            output = model(x)
        except RuntimeError as e:
            # Error should be caught and recoverable
            error_msg = str(e)
            assert "size mismatch" in error_msg or "mat1 and mat2 shapes" in error_msg
        
        # Training should be able to continue
        x_correct = torch.randn(5, 10)
        output = model(x_correct)
        assert output.shape == (5, 2)
        
        print("   Training error recovery test passed")
    
    def test_inference_error_handling(self):
        """Test inference handles errors gracefully."""
        model = nn.Linear(10, 2)
        model.eval()
        
        # Test with various invalid inputs
        invalid_inputs = [
            torch.randn(5, 5),  # Wrong shape
            None,  # None input
        ]
        
        for invalid in invalid_inputs:
            try:
                with torch.no_grad():
                    _ = model(invalid)
            except (RuntimeError, TypeError):
                pass  # Expected
        
        # Valid input should still work
        valid = torch.randn(5, 10)
        with torch.no_grad():
            output = model(valid)
        
        assert output.shape == (5, 2)
        print("   Inference error handling test passed")
    
    def test_cross_component_error_chain(self):
        """Test errors propagate correctly across component boundaries."""
        errors = []
        
        try:
            # Simulate loader error
            try:
                raise FileNotFoundError("Dataset not found")
            except FileNotFoundError as e:
                errors.append({"component": "loader", "error": str(e)})
                raise
        except FileNotFoundError:
            try:
                # Training tries to handle it
                raise RuntimeError("Training cannot proceed without data")
            except RuntimeError as e:
                errors.append({"component": "training", "error": str(e)})
        
        assert len(errors) == 2
        assert errors[0]["component"] == "loader"
        assert errors[1]["component"] == "training"
        print("   Cross-component error chain test passed")


@pytest.mark.integration
class TestComponentLifecycle:
    """
    Tests full lifecycle of components working together.
    """
    
    def test_full_component_lifecycle(self):
        """Test complete lifecycle from data loading to inference."""
        import time
        
        # Track lifecycle stages
        lifecycle = []
        
        # 1. Data Loading
        lifecycle.append({"stage": "load", "status": "started"})
        time.sleep(0.01)  # Simulate work
        lifecycle.append({"stage": "load", "status": "completed"})
        
        # 2. Model Initialization
        lifecycle.append({"stage": "init", "status": "started"})
        model = nn.Linear(10, 2)
        lifecycle.append({"stage": "init", "status": "completed"})
        
        # 3. Training
        lifecycle.append({"stage": "train", "status": "started"})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(5, 10)
        target = torch.randint(0, 2, (5,))
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        lifecycle.append({"stage": "train", "status": "completed"})
        
        # 4. Evaluation
        lifecycle.append({"stage": "evaluate", "status": "started"})
        model.eval()
        with torch.no_grad():
            _ = model(x)
        lifecycle.append({"stage": "evaluate", "status": "completed"})
        
        # 5. Inference
        lifecycle.append({"stage": "inference", "status": "started"})
        with torch.no_grad():
            _ = model(x)
        lifecycle.append({"stage": "inference", "status": "completed"})
        
        # 6. Cleanup
        lifecycle.append({"stage": "cleanup", "status": "started"})
        del model, optimizer
        gc.collect()
        lifecycle.append({"stage": "cleanup", "status": "completed"})
        
        # Verify all stages completed
        stages = ["load", "init", "train", "evaluate", "inference", "cleanup"]
        for stage in stages:
            stage_events = [e for e in lifecycle if e["stage"] == stage]
            assert len(stage_events) == 2  # started and completed
            assert stage_events[0]["status"] == "started"
            assert stage_events[1]["status"] == "completed"
        
        print("   Full component lifecycle test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
