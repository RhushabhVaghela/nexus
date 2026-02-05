#!/usr/bin/env python3
"""
test_pipeline_orchestrator.py
Tests for the PipelineOrchestrator class.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock required modules
mock_torch = Mock()
mock_transformers = Mock()
mock_datasets = Mock()
mock_pydantic = Mock()
sys.modules["torch"] = mock_torch
sys.modules["transformers"] = mock_transformers
sys.modules["datasets"] = mock_datasets
sys.modules["pydantic"] = mock_pydantic

from core.orchestration.pipeline_orchestrator import (
    PipelineConfig,
    PipelineStage,
    CheckpointManager,
    PipelineResult,
    PipelineProgress,
    PipelineError,
    PipelineOrchestrator,
    StageStatus,
    FailureMode,
)


class TestPipelineConfig:
    """Tests for PipelineConfig class."""

    def test_valid_config_creation(self):
        """Test creating a valid pipeline configuration."""
        config = PipelineConfig(
            stages=["reasoning", "tools", "vision"],
            dependencies={
                "reasoning": [],
                "tools": ["reasoning"],
                "vision": ["reasoning", "tools"],
            },
            checkpoint_dir="checkpoints/test",
            parallel_stages=["reasoning", "tools"],
            failure_mode="stop",
        )

        assert config.stages == ["reasoning", "tools", "vision"]
        assert config.dependencies["tools"] == ["reasoning"]
        assert config.failure_mode == "stop"

    def test_invalid_empty_stages(self):
        """Test that empty stages raises error."""
        with pytest.raises(ValueError, match="Pipeline must have at least one stage"):
            PipelineConfig(stages=[], dependencies={})

    def test_invalid_dependency_reference(self):
        """Test that invalid dependency reference raises error."""
        with pytest.raises(
            ValueError, match="Dependencies must reference only valid stages"
        ):
            PipelineConfig(
                stages=["reasoning"], dependencies={"reasoning": ["invalid_stage"]}
            )

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        with pytest.raises(ValueError, match="Circular dependency"):
            PipelineConfig(
                stages=["a", "b", "c"],
                dependencies={
                    "a": ["b"],
                    "b": ["c"],
                    "c": ["a"],  # Circular!
                },
            )

    def test_resume_from_validation(self):
        """Test resume_from validation."""
        with pytest.raises(ValueError, match="resume_from stage"):
            PipelineConfig(
                stages=["reasoning", "tools"],
                dependencies={},
                resume_from="nonexistent",
            )

    def test_failure_mode_validation(self):
        """Test failure mode validation."""
        with pytest.raises(ValueError, match="Invalid failure_mode"):
            PipelineConfig(
                stages=["reasoning"], dependencies={}, failure_mode="invalid_mode"
            )

    def test_to_dict_conversion(self):
        """Test configuration to dictionary conversion."""
        config = PipelineConfig(
            stages=["reasoning"],
            dependencies={},
            checkpoint_dir="checkpoints/test",
            enable_logging=False,
        )

        result = config.to_dict()

        assert result["stages"] == ["reasoning"]
        assert result["checkpoint_dir"] == "checkpoints/test"
        assert result["enable_logging"] is False


class TestPipelineStage:
    """Tests for PipelineStage class."""

    def test_stage_creation(self):
        """Test creating a pipeline stage wrapper."""
        # Create mock stage
        mock_stage = Mock()
        mock_stage.config = Mock()
        mock_stage.config.capability_name = "test"
        mock_stage.config.base_model_path = "/test"
        mock_stage.config.output_dir = "/output"

        # Mock pydantic models
        mock_pydantic_model = Mock()
        mock_pydantic_model.return_value = Mock()

        with patch("builtins.__import__") as mock_import:
            mock_import.side_effect = lambda name, *args, **kwargs: (
                Mock() if name == "pydantic" else __import__(name, *args, **kwargs)
            )

        # Create stage
        stage = PipelineStage(
            name="test_stage",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        assert stage.name == "test_stage"
        assert stage.stage == mock_stage

    def test_empty_stage_name_raises(self):
        """Test that empty stage name raises error."""
        mock_stage = Mock()

        with pytest.raises(ValueError, match="Stage name cannot be empty"):
            PipelineStage(
                name="", stage=mock_stage, input_schema=object, output_schema=object
            )

    def test_invalid_stage_type_raises(self):
        """Test that non-BaseStage type raises error."""
        with pytest.raises(TypeError, match="Stage must be a BaseStage instance"):
            PipelineStage(
                name="test",
                stage="not_a_stage",
                input_schema=object,
                output_schema=object,
            )


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_checkpoint_saving(self, temp_checkpoint_dir):
        """Test saving checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir, max_checkpoints=3
        )

        state = {"data": "test", "step": 1}

        checkpoint = manager.save_checkpoint(
            stage_name="test_stage", state=state, step=1, metadata={"test": True}
        )

        assert checkpoint.stage_name == "test_stage"
        assert checkpoint.step == 1
        assert os.path.exists(checkpoint.checkpoint_path)

    def test_checkpoint_loading(self, temp_checkpoint_dir):
        """Test loading checkpoints."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save checkpoint
        original_state = {"data": "test", "step": 1}
        manager.save_checkpoint("test_stage", original_state, step=1)

        # Load checkpoint
        loaded_state = manager.load_checkpoint("test_stage")

        assert loaded_state is not None
        assert loaded_state["data"] == "test"
        assert loaded_state["step"] == 1

    def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Create multiple checkpoints
        for i in range(3):
            manager.save_checkpoint("test_stage", {"data": i}, step=i)

        checkpoints = manager.list_checkpoints("test_stage")

        assert len(checkpoints) == 3
        # Should be sorted by timestamp descending (newest first)
        assert checkpoints[0].step == 2

    def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test cleanup of old checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir, max_checkpoints=2
        )

        # Create 3 checkpoints
        for i in range(3):
            manager.save_checkpoint("test_stage", {"data": i}, step=i)

        # Should have only 2 checkpoints
        checkpoints = manager.list_checkpoints("test_stage")
        assert len(checkpoints) == 2

    def test_get_latest_checkpoint(self, temp_checkpoint_dir):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Create checkpoints
        manager.save_checkpoint("test_stage", {"data": "first"}, step=1)
        manager.save_checkpoint("test_stage", {"data": "second"}, step=2)

        latest = manager.get_latest_checkpoint("test_stage")

        assert latest is not None
        assert latest.step == 2


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_stage(self):
        """Create a mock stage."""
        mock = Mock()
        mock.config = Mock()
        mock.config.capability_name = "test"
        mock.config.base_model_path = "/test"
        mock.config.output_dir = "/output"
        return mock

    def test_orchestrator_creation(self, temp_dir):
        """Test creating the pipeline orchestrator."""
        config = PipelineConfig(
            stages=["reasoning", "tools"],
            dependencies={"reasoning": [], "tools": ["reasoning"]},
            checkpoint_dir=temp_dir,
        )

        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.config == config
        assert len(orchestrator.stages) == 0
        assert orchestrator.execution_order == ["reasoning", "tools"]

    def test_add_stage(self, temp_dir, mock_stage):
        """Test adding stages to orchestrator."""
        config = PipelineConfig(
            stages=["reasoning", "tools"],
            dependencies={"reasoning": [], "tools": ["reasoning"]},
            checkpoint_dir=temp_dir,
        )

        orchestrator = PipelineOrchestrator(config)

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        assert "reasoning" in orchestrator.stages
        assert orchestrator._stage_status["reasoning"].value == "pending"

    def test_add_duplicate_stage_raises(self, temp_dir, mock_stage):
        """Test that adding duplicate stage raises error."""
        config = PipelineConfig(
            stages=["reasoning"], dependencies={}, checkpoint_dir=temp_dir
        )

        orchestrator = PipelineOrchestrator(config)

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        with pytest.raises(ValueError, match="already added"):
            orchestrator.add_stage(stage)

    def test_add_invalid_dependency_raises(self, temp_dir, mock_stage):
        """Test that adding stage with invalid dependency raises."""
        config = PipelineConfig(
            stages=["reasoning"], dependencies={}, checkpoint_dir=temp_dir
        )

        orchestrator = PipelineOrchestrator(config)

        stage2 = PipelineStage(
            name="tools", stage=mock_stage, input_schema=object, output_schema=object
        )

        with pytest.raises(ValueError, match="Dependency.*not found"):
            orchestrator.add_stage(stage2, dependencies=["nonexistent"])

    def test_remove_stage(self, temp_dir, mock_stage):
        """Test removing a stage."""
        config = PipelineConfig(
            stages=["reasoning"], dependencies={}, checkpoint_dir=temp_dir
        )

        orchestrator = PipelineOrchestrator(config)

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)
        removed = orchestrator.remove_stage("reasoning")

        assert removed is not None
        assert "reasoning" not in orchestrator.stages

    def test_get_stage(self, temp_dir, mock_stage):
        """Test getting a stage by name."""
        config = PipelineConfig(
            stages=["reasoning"], dependencies={}, checkpoint_dir=temp_dir
        )

        orchestrator = PipelineOrchestrator(config)

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        retrieved = orchestrator.get_stage("reasoning")

        assert retrieved == stage

    def test_validate_dag(self, temp_dir):
        """Test DAG validation."""
        config = PipelineConfig(
            stages=["a", "b", "c"],
            dependencies={"a": [], "b": ["a"], "c": ["a", "b"]},
            checkpoint_dir=temp_dir,
        )

        orchestrator = PipelineOrchestrator(config)

        assert orchestrator.validate_dag() is True

    def test_get_progress_initial(self, temp_dir):
        """Test getting progress before execution."""
        config = PipelineConfig(
            stages=["reasoning"], dependencies={}, checkpoint_dir=temp_dir
        )

        orchestrator = PipelineOrchestrator(config)
        progress = orchestrator.get_progress()

        assert progress.total_stages == 0
        assert progress.completed_stages == []
        assert progress.progress_percentage == 0.0

    def test_execute_dry_run(self, temp_dir, mock_stage):
        """Test executing pipeline in dry-run mode."""
        config = PipelineConfig(
            stages=["reasoning", "tools"],
            dependencies={"reasoning": [], "tools": ["reasoning"]},
            checkpoint_dir=temp_dir,
            enable_logging=False,
        )

        orchestrator = PipelineOrchestrator(config)

        stage1 = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage1)

        result = orchestrator.execute_pipeline({"input_data": "test"}, dry_run=True)

        assert result.success is True
        assert result.duration >= 0

    def test_execute_with_real_stages(self, temp_dir, mock_stage):
        """Test executing pipeline with real (mocked) stages."""
        config = PipelineConfig(
            stages=["reasoning"],
            dependencies={},
            checkpoint_dir=temp_dir,
            enable_logging=False,
        )

        orchestrator = PipelineOrchestrator(config)

        # Setup mock to return output
        mock_stage.run.return_value = {"result": "test_output"}

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        result = orchestrator.execute_pipeline({"input_data": "test"}, dry_run=False)

        mock_stage.run.assert_called_once()
        assert result.success is True
        assert "reasoning" in result.outputs

    def test_error_handling(self, temp_dir, mock_stage):
        """Test error handling during execution."""
        config = PipelineConfig(
            stages=["reasoning"],
            dependencies={},
            checkpoint_dir=temp_dir,
            failure_mode="continue",
            enable_logging=False,
        )

        orchestrator = PipelineOrchestrator(config)

        # Create mock stage that fails
        mock_stage.run.side_effect = Exception("Test error")

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        result = orchestrator.execute_pipeline({"input_data": "test"}, dry_run=False)

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "Exception"

    def test_progress_tracking(self, temp_dir, mock_stage):
        """Test progress tracking during execution."""
        config = PipelineConfig(
            stages=["reasoning"],
            dependencies={},
            checkpoint_dir=temp_dir,
            enable_logging=False,
        )

        orchestrator = PipelineOrchestrator(config)

        # Setup mock to return output
        mock_stage.run.return_value = {"result": "test"}

        stage = PipelineStage(
            name="reasoning",
            stage=mock_stage,
            input_schema=object,
            output_schema=object,
        )

        orchestrator.add_stage(stage)

        # Check progress before execution
        progress_before = orchestrator.get_progress()
        assert progress_before.progress_percentage == 0.0

        # Execute
        orchestrator.execute_pipeline({"input_data": "test"})

        # Check progress after execution
        progress_after = orchestrator.get_progress()
        assert progress_after.progress_percentage == 100.0
        assert "reasoning" in progress_after.completed_stages

    def test_reset(self, temp_dir):
        """Test resetting the orchestrator."""
        config = PipelineConfig(
            stages=["reasoning"],
            dependencies={},
            checkpoint_dir=temp_dir,
            enable_logging=False,
        )

        orchestrator = PipelineOrchestrator(config)

        # Set some state
        orchestrator._stage_outputs = {"reasoning": "test"}
        orchestrator._errors = ["error"]

        # Reset
        orchestrator.reset()

        assert orchestrator._stage_outputs == {}
        assert orchestrator._errors == []
        assert orchestrator._start_time is None


class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        progress = PipelineProgress(
            current_stage=None,
            total_stages=1,
            completed_stages=["test"],
            failed_stages=[],
            skipped_stages=[],
            current_step=1,
            total_steps=1,
            start_time=None,
            elapsed_seconds=10.0,
            last_checkpoint=None,
        )

        result = PipelineResult(
            success=True,
            outputs={"test": "output"},
            checkpoints=[],
            duration=10.0,
            errors=[],
            progress=progress,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["outputs"]["test"] == "output"
        assert result_dict["duration"] == 10.0
        assert result_dict["progress"]["progress_percentage"] == 100.0


class TestPipelineProgress:
    """Tests for PipelineProgress class."""

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = PipelineProgress(
            current_stage=None,
            total_stages=4,
            completed_stages=["a", "b"],
            failed_stages=[],
            skipped_stages=[],
            current_step=2,
            total_steps=4,
            start_time=None,
            elapsed_seconds=10.0,
            last_checkpoint=None,
        )

        assert progress.progress_percentage == 50.0

    def test_is_complete(self):
        """Test completion detection."""
        # Not complete
        progress = PipelineProgress(
            current_stage=None,
            total_stages=2,
            completed_stages=["a"],
            failed_stages=[],
            skipped_stages=[],
            current_step=1,
            total_steps=2,
            start_time=None,
            elapsed_seconds=10.0,
            last_checkpoint=None,
        )

        assert progress.is_complete is False

        # Complete
        progress = PipelineProgress(
            current_stage=None,
            total_stages=2,
            completed_stages=["a", "b"],
            failed_stages=[],
            skipped_stages=[],
            current_step=2,
            total_steps=2,
            start_time=None,
            elapsed_seconds=20.0,
            last_checkpoint=None,
        )

        assert progress.is_complete is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
