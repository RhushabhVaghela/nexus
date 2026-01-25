import pytest
import os
from unittest.mock import patch
from src.utils.control_training import create_flag_file, remove_flag_file, main

def test_flag_file_ops(tmp_path):
    flag_dir = str(tmp_path)
    flag_name = "test.flag"
    
    create_flag_file(flag_dir, flag_name)
    assert os.path.exists(os.path.join(flag_dir, flag_name))
    
    remove_flag_file(flag_dir, flag_name)
    assert not os.path.exists(os.path.join(flag_dir, flag_name))

def test_control_training_main_pause(tmp_path):
    flag_dir = str(tmp_path)
    
    # Mock input to send 'p' then 'q'
    with patch("builtins.input", side_effect=["p", "q"]), \
         patch("sys.argv", ["control_training.py", "--flag-dir", flag_dir]):
        main()
        
    assert os.path.exists(os.path.join(flag_dir, "pause.flag"))

def test_control_training_main_resume(tmp_path):
    flag_dir = str(tmp_path)
    # Pre-create pause flag
    create_flag_file(flag_dir, "pause.flag")
    
    with patch("builtins.input", side_effect=["r", "q"]), \
         patch("sys.argv", ["control_training.py", "--flag-dir", flag_dir]):
        main()
        
    assert not os.path.exists(os.path.join(flag_dir, "pause.flag"))

def test_control_training_main_next(tmp_path):
    flag_dir = str(tmp_path)
    
    with patch("builtins.input", side_effect=["n", "q"]), \
         patch("sys.argv", ["control_training.py", "--flag-dir", flag_dir]):
        main()
        
    assert os.path.exists(os.path.join(flag_dir, "next.flag"))
