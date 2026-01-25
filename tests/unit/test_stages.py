#!/usr/bin/env python3
"""
test_stages.py
Unit tests for all stage scripts.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStageConfig:
    """Test StageConfig dataclass."""
    
    def test_stage_config_import(self):
        """Test StageConfig can be imported."""
        from src.stages.base import StageConfig
        assert StageConfig is not None
    
    def test_stage_config_creation(self):
        """Test StageConfig creation with required fields."""
        from src.stages.base import StageConfig
        
        config = StageConfig(
            capability_name="test",
            base_model_path="/path/to/model",
            output_dir="/path/to/output",
        )
        
        assert config.capability_name == "test"
        assert config.base_model_path == "/path/to/model"
        assert config.epochs == 3  # default
        assert config.batch_size == 1  # default
        assert config.dry_run is False  # default
    
    def test_stage_config_to_dict(self):
        """Test StageConfig to_dict method."""
        from src.stages.base import StageConfig
        
        config = StageConfig(
            capability_name="cot",
            base_model_path="/model",
            output_dir="/output",
        )
        
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["capability_name"] == "cot"
        assert "base_model_path" in d


class TestBaseStage:
    """Test BaseStage abstract class."""
    
    def test_base_stage_import(self):
        """Test BaseStage can be imported."""
        from src.stages.base import BaseStage
        assert BaseStage is not None
    
    def test_base_stage_is_abstract(self):
        """Test BaseStage cannot be instantiated directly."""
        from src.stages.base import BaseStage, StageConfig
        from abc import ABC
        
        assert issubclass(BaseStage, ABC)
    
    def test_text_capability_stage_import(self):
        """Test TextCapabilityStage can be imported."""
        from src.stages.base import TextCapabilityStage
        assert TextCapabilityStage is not None


class TestCoTStage:
    """Test CoT stage."""
    
    def test_cot_stage_import(self):
        """Test CoTStage can be imported."""
        from src.stages.stage_cot import CoTStage
        assert CoTStage is not None
    
    def test_cot_stage_capability_name(self):
        """Test CoTStage has correct capability name."""
        from src.stages.stage_cot import CoTStage
        assert CoTStage.CAPABILITY_NAME == "cot"
    
    def test_cot_stage_has_dataset_patterns(self):
        """Test CoTStage defines dataset patterns."""
        from src.stages.stage_cot import CoTStage
        assert hasattr(CoTStage, 'DATASET_PATTERNS')
        assert len(CoTStage.DATASET_PATTERNS) > 0


class TestReasoningStage:
    """Test Reasoning stage."""
    
    def test_reasoning_stage_import(self):
        from src.stages.stage_reasoning import ReasoningStage
        assert ReasoningStage is not None
    
    def test_reasoning_stage_capability_name(self):
        from src.stages.stage_reasoning import ReasoningStage
        assert ReasoningStage.CAPABILITY_NAME == "reasoning"


class TestThinkingStage:
    """Test Thinking stage."""
    
    def test_thinking_stage_import(self):
        from src.stages.stage_thinking import ThinkingStage
        assert ThinkingStage is not None
    
    def test_thinking_stage_capability_name(self):
        from src.stages.stage_thinking import ThinkingStage
        assert ThinkingStage.CAPABILITY_NAME == "thinking"


class TestToolsStage:
    """Test Tools stage."""
    
    def test_tools_stage_import(self):
        from src.stages.stage_tools import ToolsStage
        assert ToolsStage is not None
    
    def test_tools_stage_capability_name(self):
        from src.stages.stage_tools import ToolsStage
        assert ToolsStage.CAPABILITY_NAME == "tool-calling"


class TestStreamingStage:
    """Test Streaming stage."""
    
    def test_streaming_stage_import(self):
        from src.stages.stage_streaming import StreamingStage
        assert StreamingStage is not None
    
    def test_streaming_stage_capability_name(self):
        from src.stages.stage_streaming import StreamingStage
        assert StreamingStage.CAPABILITY_NAME == "streaming"


class TestPodcastStage:
    """Test Podcast stage."""
    
    def test_podcast_stage_import(self):
        from src.stages.stage_podcast import PodcastStage
        assert PodcastStage is not None
    
    def test_podcast_stage_capability_name(self):
        from src.stages.stage_podcast import PodcastStage
        assert PodcastStage.CAPABILITY_NAME == "podcast"
    
    def test_podcast_stage_has_dataset_patterns(self):
        from src.stages.stage_podcast import PodcastStage
        assert hasattr(PodcastStage, 'DATASET_PATTERNS')
        assert len(PodcastStage.DATASET_PATTERNS) > 0


class TestVisionQAStage:
    """Test Vision QA stage."""
    
    def test_vision_qa_stage_import(self):
        from src.stages.stage_vision_qa import VisionQAStage
        assert VisionQAStage is not None
    
    def test_vision_qa_stage_capability_name(self):
        from src.stages.stage_vision_qa import VisionQAStage
        assert VisionQAStage.CAPABILITY_NAME == "vision-qa"
    
    def test_vision_qa_stage_has_dataset_patterns(self):
        from src.stages.stage_vision_qa import VisionQAStage
        assert hasattr(VisionQAStage, 'DATASET_PATTERNS')


class TestVideoUnderstandingStage:
    """Test Video Understanding stage."""
    
    def test_video_stage_import(self):
        try:
            from src.stages.stage_video import VideoUnderstandingStage
            assert VideoUnderstandingStage is not None
        except ImportError:
            pytest.skip("stage_video.py not available")
    
    def test_video_stage_capability_name(self):
        try:
            from src.stages.stage_video import VideoUnderstandingStage
            assert VideoUnderstandingStage.CAPABILITY_NAME == "video-understanding"
        except ImportError:
            pytest.skip("stage_video.py not available")


class TestTriStreamingStage:
    """Test Tri-Streaming stage."""
    
    def test_tri_streaming_stage_import(self):
        from src.stages.stage_tri_streaming import TriStreamingStage
        assert TriStreamingStage is not None
    
    def test_tri_streaming_stage_capability_name(self):
        from src.stages.stage_tri_streaming import TriStreamingStage
        assert TriStreamingStage.CAPABILITY_NAME == "tri-streaming"


class TestImageGenStage:
    """Test Image Generation stage."""
    
    def test_image_gen_stage_import(self):
        from src.stages.stage_image_gen import ImageGenStage, ImageProjector
        assert ImageGenStage is not None
        assert ImageProjector is not None
    
    def test_image_gen_stage_capability_name(self):
        from src.stages.stage_image_gen import ImageGenStage
        assert ImageGenStage.CAPABILITY_NAME == "image-generation"
    
    def test_image_projector_forward(self):
        """Test ImageProjector forward pass."""
        import torch
        from src.stages.stage_image_gen import ImageProjector
        
        projector = ImageProjector(llm_dim=4096, sd_dim=2048, num_tokens=77)
        x = torch.randn(2, 4096)
        out = projector(x)
        
        assert out.shape == (2, 77, 2048)


class TestVideoGenStage:
    """Test Video Generation stage."""
    
    def test_video_gen_stage_import(self):
        from src.stages.stage_video_gen import VideoGenStage, VideoProjector
        assert VideoGenStage is not None
        assert VideoProjector is not None
    
    def test_video_gen_stage_capability_name(self):
        from src.stages.stage_video_gen import VideoGenStage
        assert VideoGenStage.CAPABILITY_NAME == "video-generation"
    
    def test_video_projector_forward(self):
        """Test VideoProjector forward pass."""
        import torch
        from src.stages.stage_video_gen import VideoProjector
        
        projector = VideoProjector(llm_dim=4096, svd_dim=1024, num_frames=14)
        x = torch.randn(2, 4096)
        out = projector(x)
        
        assert out.shape == (2, 14, 1024)


class TestStageLogicMocked:
    """Test stage main() functions with full mocking."""
    
    @patch("src.stages.stage_cot.CoTStage.run")
    def test_cot_main_logic(self, mock_run, tmp_path):
        from src.stages.stage_cot import main
        mock_run.return_value = {"success": True}
        
        out_dir = tmp_path / "out"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        
        with patch("sys.argv", ["stage_cot.py", "--base-model", str(model_dir), "--output-dir", str(out_dir), "--epochs", "1", "--dry-run"]):
            main()
            assert mock_run.called

    @patch("src.stages.stage_thinking.ThinkingStage.run")
    def test_thinking_main_logic(self, mock_run, tmp_path):
        from src.stages.stage_thinking import main
        mock_run.return_value = {"success": True}
        
        out_dir = tmp_path / "out_think"
        model_dir = tmp_path / "model_think"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        
        with patch("sys.argv", ["stage_thinking.py", "--base-model", str(model_dir), "--output-dir", str(out_dir), "--epochs", "1", "--dry-run"]):
            main()
            assert mock_run.called

    def test_cot_has_main(self):
        from src.stages import stage_cot
        assert hasattr(stage_cot, 'main')
    
    def test_reasoning_has_main(self):
        from src.stages import stage_reasoning
        assert hasattr(stage_reasoning, 'main')
    
    def test_thinking_has_main(self):
        from src.stages import stage_thinking
        assert hasattr(stage_thinking, 'main')
    
    def test_tools_has_main(self):
        from src.stages import stage_tools
        assert hasattr(stage_tools, 'main')
    
    def test_streaming_has_main(self):
        from src.stages import stage_streaming
        assert hasattr(stage_streaming, 'main')
    
    def test_podcast_has_main(self):
        from src.stages import stage_podcast
        assert hasattr(stage_podcast, 'main')
    
    def test_vision_qa_has_main(self):
        from src.stages import stage_vision_qa
        assert hasattr(stage_vision_qa, 'main')
    
    def test_tri_streaming_has_main(self):
        from src.stages import stage_tri_streaming
        assert hasattr(stage_tri_streaming, 'main')
    
    def test_image_gen_has_main(self):
        from src.stages import stage_image_gen
        assert hasattr(stage_image_gen, 'main')
    
    def test_video_gen_has_main(self):
        from src.stages import stage_video_gen
        assert hasattr(stage_video_gen, 'main')
