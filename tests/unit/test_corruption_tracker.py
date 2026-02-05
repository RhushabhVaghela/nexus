import pytest
import os
from pathlib import Path
from src.utils.corruption_tracker import CorruptionTracker

def test_corruption_tracker(tmp_path):
    log_file = tmp_path / "corrupted.log"
    tracker = CorruptionTracker(log_path=str(log_file))
    
    tracker.log_corrupted("test_file.png", "Invalid header")
    
    assert log_file.exists()
    content = log_file.read_text()
    assert "CORRUPTED: test_file.png" in content
    assert "ERROR: Invalid header" in content
