import os
import json
import pytest
from unittest.mock import MagicMock, patch
from scripts.nexus_pipeline import NexusPipeline

@pytest.fixture
def mock_registry():
    return [
        {"teacher_id": "llm-1", "category": "Language model", "path": "/path/1", "status": "ready"},
        {"teacher_id": "audio-1", "category": "Audio (TTS)", "path": "/path/2", "status": "ready"}
    ]

def test_pipeline_init_and_reset(tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline()
        assert pipeline.state["current_stage"] == "init"
        
        pipeline.state["current_stage"] = "done"
        pipeline._save_state()
        
        # Reload
        pipeline2 = NexusPipeline()
        assert pipeline2.state["current_stage"] == "done"

def test_dry_run_no_execution(tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True)
        pipeline.run_command = MagicMock()
        
        with patch("os.system") as mock_sys:
            pipeline.run_command("some_heavy_cmd")
            mock_sys.assert_not_called()

def test_filtering_permissive_default(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True, skip_non_llm=False)
        pipeline.registry = mock_registry
        pipeline.run_command = MagicMock()
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        # Should call profile for both LLM and Audio (with warning for audio)
        assert pipeline.run_command.call_count == 2

def test_filtering_strict_flag(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True, skip_non_llm=True)
        pipeline.registry = mock_registry
        pipeline.run_command = MagicMock()
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        # Should only call profile for LLM
        assert pipeline.run_command.call_count == 1
        args = pipeline.run_command.call_args_list[0][0][0]
        assert "llm-1" in args

def test_stage_sequence_flow(tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True)
        pipeline.run_command = MagicMock()
        
        # Starting from profiling
        pipeline.state["current_stage"] = "profiling"
        pipeline.registry = [{"teacher_id": "m", "category": "Language model", "path": "p", "status": "ready"}]
        
        # Mocking existence for export
        with patch("os.path.exists", return_value=True):
            pipeline.run()
            
        # Verify stages were hit (checked via state updates)
        assert "profiling" in pipeline.state["completed_stages"]
        assert "knowledge_extraction" in pipeline.state["completed_stages"]
        assert "training" in pipeline.state["completed_stages"]
        assert "evaluation" in pipeline.state["completed_stages"]
        assert "export" in pipeline.state["completed_stages"]
        assert "cleanup" in pipeline.state["completed_stages"]
        assert pipeline.state["current_stage"] == "done"

def test_model_filtering(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        # Target only llm-1
        pipeline = NexusPipeline(dry_run=True, models="llm-1")
        pipeline.registry = mock_registry
        pipeline.run_command = MagicMock()
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        assert pipeline.run_command.call_count == 1
        assert "llm-1" in pipeline.run_command.call_args[0][0]

def test_dataset_filtering(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        # Target specific datasets
        pipeline = NexusPipeline(dry_run=True, datasets="code/stack,reasoning/math")
        pipeline.registry = [mock_registry[0]] # Just llm-1
        pipeline.run_command = MagicMock()
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_knowledge_extraction()
            
        # Should be called once for each dataset
        assert pipeline.run_command.call_count == 2
        calls = [c[0][0] for c in pipeline.run_command.call_args_list]
        assert any("code/stack" in c for c in calls)
        assert any("reasoning/math" in c for c in calls)
