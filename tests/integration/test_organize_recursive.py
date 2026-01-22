"""
Integration test for deep-recursive dataset organization.
"""

import pytest
import os
import shutil
from pathlib import Path
from src.utils.organize_datasets import organize

@pytest.fixture
def complex_data_root(tmp_path):
    root = tmp_path / "data"
    datasets = root / "datasets"
    datasets.mkdir(parents=True)
    
    # 1. Misplaced uncensored folder deep in code
    deep_path = datasets / "code" / "misc" / "hidden_uncensored"
    deep_path.mkdir(parents=True)
    (deep_path / "data.json").write_text("{}")
    
    # 2. File with uncensored name in general
    general = datasets / "general"
    general.mkdir()
    (general / "lovable_dataset.json").write_text("{}")
    
    # 3. Already correct uncensored file
    uncensored = datasets / "uncensored"
    uncensored.mkdir()
    (uncensored / "keep_me.json").write_text("{}")
    
    return root

def test_deep_recursive_reorganization(complex_data_root):
    # Run organization
    organize(str(complex_data_root), move=True)
    
    datasets_root = complex_data_root / "datasets"
    target_uncensored = datasets_root / "uncensored"
    
    # 1. Verify deep folder was moved
    assert (target_uncensored / "hidden_uncensored").exists()
    assert not (datasets_root / "code" / "misc" / "hidden_uncensored").exists()
    
    # 2. Verify individual file was moved
    assert (target_uncensored / "lovable_dataset.json").exists()
    assert not (datasets_root / "general" / "lovable_dataset.json").exists()
    
    # 3. Verify original stayed
    assert (target_uncensored / "keep_me.json").exists()
