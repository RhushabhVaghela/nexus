"""
Unit tests for UniversalDatasetManager and mode-based dataset extension logic.
"""

import pytest
import os
import shutil
from pathlib import Path
from src.data.universal_manager import UniversalDatasetManager

@pytest.fixture
def mock_data_root(tmp_path):
    # Setup mock structure
    data_root = tmp_path / "data"
    datasets_dir = data_root / "datasets"
    
    # Core categories
    (datasets_dir / "code").mkdir(parents=True)
    (datasets_dir / "general").mkdir(parents=True)
    
    # Uncensored sibling
    (datasets_dir / "uncensored").mkdir(parents=True)
    
    # Add dummy files
    (datasets_dir / "code" / "sample.jsonl").write_text('{"messages": []}')
    (datasets_dir / "general" / "sample.jsonl").write_text('{"messages": []}')
    (datasets_dir / "uncensored" / "hidden.jsonl").write_text('{"messages": []}')
    
    return data_root

class TestUniversalManagerModes:
    
    def test_default_mode_excludes_uncensored(self, mock_data_root):
        manager = UniversalDatasetManager(mode="default", data_root=str(mock_data_root))
        
        from unittest.mock import MagicMock
        from datasets import Dataset
        manager.load_category = MagicMock(return_value=[Dataset.from_list([{"messages": []}])])
        
        # Logic: check enabled categories
        ds = manager.get_unified_train_dataset(enabled_categories=["code"])
        
        # Verify it loaded ONLY code (one call)
        assert manager.load_category.call_count == 1
        assert manager.load_category.call_args[0][0] == "code"
        
    def test_uncensored_mode_includes_extension(self, mock_data_root):
        # We need to ensure manager.get_unified_train_dataset adds "uncensored"
        manager = UniversalDatasetManager(mode="uncensored", data_root=str(mock_data_root))
        
        # Mock load_category to track what's being loaded
        from unittest.mock import MagicMock
        from datasets import Dataset
        manager.load_category = MagicMock(return_value=[Dataset.from_list([{"messages": []}])])
        
        manager.get_unified_train_dataset(enabled_categories=["code"])
        
        # Verify it loaded BOTH code and uncensored
        calls = [call[0][0] for call in manager.load_category.call_args_list]
        assert "code" in calls
        assert "uncensored" in calls

    def test_organizer_priority_logic(self, mock_data_root):
        # Test keyword priority for 'uncensored'
        # Even if it looks like code, it should go to uncensored
        from pathlib import Path
        dummy_file = Path("code_uncensored_samples.jsonl")
        
        # Import correctly
        from src.utils.organize_datasets import TRAINING_CATEGORIES
        
        filename = dummy_file.name.lower()
        keyword_category = None
        if any(kw in filename for kw in TRAINING_CATEGORIES["uncensored"]):
            keyword_category = "uncensored"
            
        assert keyword_category == "uncensored"
