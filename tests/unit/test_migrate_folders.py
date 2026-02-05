import pytest
import shutil
from pathlib import Path
from src.utils.migrate_to_mode_folders import simplify

def test_simplify_folders(tmp_path):
    # Setup structure
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    
    censored = datasets / "censored"
    censored.mkdir()
    
    (censored / "item1.txt").write_text("content1")
    (censored / "dir1").mkdir()
    (censored / "dir1" / "sub1.txt").write_text("subcontent")
    
    # Target already has dir1 but not sub1.txt
    (datasets / "dir1").mkdir()
    
    simplify(str(tmp_path))
    
    assert (datasets / "item1.txt").exists()
    assert (datasets / "dir1" / "sub1.txt").exists()
    assert not censored.exists()
