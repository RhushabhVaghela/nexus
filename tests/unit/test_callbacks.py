import pytest
import os
import json
import time
from unittest.mock import MagicMock, patch
from src.utils.callbacks import KeyboardPauseCallback

@pytest.fixture
def temp_dirs(tmp_path):
    flag_dir = tmp_path / "flags"
    output_dir = tmp_path / "output"
    flag_dir.mkdir()
    output_dir.mkdir()
    return str(flag_dir), str(output_dir)

def test_callback_init(temp_dirs):
    flag_dir, output_dir = temp_dirs
    cb = KeyboardPauseCallback(flag_dir, output_dir)
    assert cb.flag_dir == flag_dir
    assert cb.output_dir == output_dir

def test_callback_step_end_pause(temp_dirs):
    flag_dir, output_dir = temp_dirs
    cb = KeyboardPauseCallback(flag_dir, output_dir)
    
    # Create pause flag
    with open(cb.pause_file, "w") as f: f.write("1")
    
    state = MagicMock(global_step=100)
    control = MagicMock(should_save=False, should_training_stop=False)
    
    cb.on_step_end(None, state, control)
    
    assert control.should_save is True
    assert control.should_training_stop is True
    assert cb.is_paused is True

def test_callback_step_end_next(temp_dirs):
    flag_dir, output_dir = temp_dirs
    cb = KeyboardPauseCallback(flag_dir, output_dir)
    
    # Create next flag
    with open(cb.next_file, "w") as f: f.write("1")
    
    state = MagicMock(global_step=100)
    control = MagicMock(should_training_stop=False)
    
    cb.on_step_end(None, state, control)
    
    assert control.should_training_stop is True
    assert not os.path.exists(cb.next_file)

def test_status_writing(temp_dirs):
    flag_dir, output_dir = temp_dirs
    cb = KeyboardPauseCallback(flag_dir, output_dir)
    
    state = MagicMock(global_step=50)
    cb._write_status(state=state)
    
    assert os.path.exists(cb.status_file)
    with open(cb.status_file, "r") as f:
        data = json.load(f)
        assert data["step"] == 50
        assert data["status"] == "training"

def test_eta_calculation(temp_dirs):
    flag_dir, output_dir = temp_dirs
    cb = KeyboardPauseCallback(flag_dir, output_dir)
    
    # Mock trainer state for total steps
    cb.trainer_ref = MagicMock()
    cb.trainer_ref.state = MagicMock(max_steps=1000)
    
    # Add fake step times
    cb.step_times.append((10, 1000.0))
    cb.step_times.append((20, 1010.0)) # 10 steps in 10s -> 1s/step
    
    eta = cb._calc_eta_seconds()
    # 1000 - 20 = 980 steps left -> 980s
    assert eta == 980.0
