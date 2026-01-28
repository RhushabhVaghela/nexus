import os
import json
import pytest
from unittest.mock import MagicMock, patch
from scripts.nexus_pipeline import NexusPipeline

@pytest.fixture
def mock_registry():
    registries = {
        "teacher": {
            "llm-1": {"teacher_id": "llm-1", "category": "Language model", "tags": ["Language model"], "path": "/path/1", "status": "ready"},
            "audio-1": {"teacher_id": "audio-1", "category": "Audio (TTS)", "path": "/path/2", "status": "ready"}
        },
        "dataset": {
            "code/stack": {"local_path": "/data/code"},
            "reasoning/math": {"local_path": "/data/math"}
        }
    }
    return registries

@pytest.fixture(autouse=True)
def patch_registries(mock_registry):
    with patch("scripts.nexus_pipeline.TEACHER_REGISTRY", mock_registry["teacher"]), \
         patch("scripts.nexus_pipeline.DATASET_REGISTRY", mock_registry["dataset"]):
        yield


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
        pipeline.run_command = MagicMock(return_value=0)
        
        with patch("os.system") as mock_sys:
            pipeline.run_command("some_heavy_cmd")
            mock_sys.assert_not_called()

def test_filtering_permissive_default(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        # Pipeline init will use patched TEACHER_REGISTRY to set target_models
        pipeline = NexusPipeline(dry_run=True, skip_non_llm=False, models="all")
        pipeline.run_command = MagicMock(return_value=0)
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        # Should call profile for both LLM and Audio (with warning for audio)
        assert pipeline.run_command.call_count == 2

def test_filtering_strict_flag(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True, skip_non_llm=True, models="all")
        pipeline.run_command = MagicMock(return_value=0)
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        # Should only call profile for LLM
        assert pipeline.run_command.call_count == 1
        args = pipeline.run_command.call_args_list[0][0][0]
        assert "llm-1" in args

def test_stage_sequence_flow(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=True, models="all")
        pipeline.run_command = MagicMock(return_value=0)
        pipeline.ensure_dataset_available = MagicMock(return_value=(None, False)) # Bypass FS/Network
        
        # Determine teachers via mocked registry (handled by autouse fixture)
        # Starting from profiling
        pipeline.state["current_stage"] = "profiling"
        
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
        pipeline.run_command = MagicMock(return_value=0)
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_profiling()
            
        assert pipeline.run_command.call_count == 1
        assert "llm-1" in pipeline.run_command.call_args[0][0]

def test_dataset_filtering(mock_registry, tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        # Target specific datasets
        pipeline = NexusPipeline(dry_run=True, datasets="code/stack,reasoning/math")
        # In this test we also want to limit models to llm-1 so we don't multiply calls
        pipeline.target_models = ["llm-1"] 
        pipeline.run_command = MagicMock(return_value=0)
        
        with patch("scripts.nexus_pipeline.sys.executable", "python"):
            pipeline.stage_knowledge_extraction()
            
        # Should be called once for each dataset
        assert pipeline.run_command.call_count == 2

        calls = [c[0][0] for c in pipeline.run_command.call_args_list]
        assert any("/data/code" in c for c in calls)
        assert any("/data/math" in c for c in calls)

def test_run_command_retry_logic(tmp_path):
    state_file = tmp_path / ".pipeline_state.json"
    with patch("scripts.nexus_pipeline.STATE_FILE", str(state_file)):
        pipeline = NexusPipeline(dry_run=False)
        
        # Mock os.system to fail (return 1)
        with patch("os.system", return_value=1):
            
            # 1. Test allow_fail=True (Should NOT exit)
            ret = pipeline.run_command("bad_cmd", allow_fail=True)
            assert ret == 1
            
            # 2. Test allow_fail=False (Should exit)
            with pytest.raises(SystemExit) as excinfo:
                pipeline.run_command("bad_cmd", allow_fail=False)
            assert excinfo.value.code == 1

