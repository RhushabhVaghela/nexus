import pytest
import os
import shutil
import json
import time
from unittest.mock import MagicMock
from src.utils.callbacks import KeyboardPauseCallback

class TestMonitoring:
    
    @pytest.fixture
    def monitor_env(self):
        """Setup temporary environment for flags/results."""
        flags = "test_flags"
        results = "test_results"
        os.makedirs(flags, exist_ok=True)
        os.makedirs(results, exist_ok=True)
        yield flags, results
        # Cleanup
        shutil.rmtree(flags)
        shutil.rmtree(results)
        
    def test_status_json_generation(self, monitor_env):
        """Verify status.json is written correctly."""
        flags, results = monitor_env
        
        callback = KeyboardPauseCallback(
            flag_dir=flags,
            output_dir=results,
            status_update_interval=0.1
        )
        
        # Simulate Training State
        mock_state = MagicMock()
        mock_state.global_step = 42
        
        # Trigger explicit write (avoid waiting for thread in unit test)
        callback._write_status(state=mock_state)
        
        # Check file
        status_path = os.path.join(results, "status.json")
        assert os.path.exists(status_path)
        
        with open(status_path, 'r') as f:
            data = json.load(f)
            
        assert data['step'] == 42
        assert data['status'] == "training"
        assert 'eta' in data
        assert 'gpu_temp' in data
        
    def test_pause_logic(self, monitor_env):
        """Verify pause flag detection."""
        flags, results = monitor_env
        
        callback = KeyboardPauseCallback(flag_dir=flags, output_dir=results)
        
        # Create Pause Flag
        with open(os.path.join(flags, "pause.flag"), 'w') as f:
            f.write("1")
            
        # Run Hook
        mock_control = MagicMock()
        mock_state = MagicMock()
        callback.on_step_end(None, mock_state, mock_control)
        
        # Should request stop
        assert mock_control.should_training_stop is True
        assert mock_control.should_save is True
