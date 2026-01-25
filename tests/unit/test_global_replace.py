import pytest
from pathlib import Path
from src.utils.global_replace import replace_in_files

def test_replace_in_files(tmp_path):
    # Create test files
    f1 = tmp_path / "test1.txt"
    f1.write_text("Hello World")
    
    f2 = tmp_path / "test2.py"
    f2.write_text("old_path = '/data/old'")
    
    replacements = {
        "World": "Nexus",
        "/data/old": "/data/new"
    }
    
    replace_in_files(str(tmp_path), replacements)
    
    assert f1.read_text() == "Hello Nexus"
    assert f2.read_text() == "old_path = '/data/new'"
