
import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.data.universal_manager import UniversalDatasetManager

# Mock data structure for testing
@pytest.fixture
def mock_data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    
    # Create datasets/reasoning/dataset1
    datasets = root / "datasets"
    datasets.mkdir()
    
    # Category: Reasoning
    reasoning = datasets / "reasoning"
    reasoning.mkdir()
    
    d1 = reasoning / "dataset1"
    d1.mkdir()
    (d1 / "data.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}], "text": "hello"}\n')
    
    # Category: Math
    math = datasets / "math"
    math.mkdir()
    
    d2 = math / "dataset2"
    d2.mkdir()
    (d2 / "data.json").write_text('[{"question": "1+1", "answer": "2"}]')
    
    # Benchmarks
    benchmarks = root / "benchmarks"
    benchmarks.mkdir()
    
    b1 = benchmarks / "benchmark1"
    b1.mkdir()
    (b1 / "data.csv").write_text("col1,col2\nval1,val2")
    
    return root

def test_dataset_manager_init(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    assert manager.datasets_dir.exists()
    assert manager.benchmarks_dir.exists()

def test_detect_format(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    
    d1 = mock_data_root / "datasets/reasoning/dataset1"
    assert manager._detect_format(d1) == "json"
    
    b1 = mock_data_root / "benchmarks/benchmark1"
    assert manager._detect_format(b1) == "csv"

def test_load_dataset_by_name_nested(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    
    with patch("src.data.universal_manager.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 1
        mock_load.return_value = mock_ds
        
        # Load by leaf name
        ds = manager.load_dataset_by_name("dataset1")
        
        assert ds is not None
        assert len(ds) == 1
        # Verify it found the path
        args, kwargs = mock_load.call_args
        assert "dataset1" in kwargs.get("data_dir", "") or "dataset1" in args[0] if args else False

        # Load benchmark
        mock_load.reset_mock()
        mock_load.return_value = mock_ds
        ds_bench = manager.load_dataset_by_name("benchmark1")
        assert ds_bench is not None
        assert len(ds_bench) == 1

def test_load_category(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    
    with patch("src.data.universal_manager.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 1
        mock_load.return_value = mock_ds
        
        # Load all reasoning
        ds_list = manager.load_category("reasoning")
        assert len(ds_list) == 1
        assert len(ds_list[0]) == 1

def test_unified_train_dataset(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    
    with patch("src.data.universal_manager.load_dataset") as mock_load, \
         patch("src.data.universal_manager.concatenate_datasets") as mock_concat:
        
        # Mock dataset with 'messages' feature
        mock_ds = MagicMock()
        mock_ds.features = {"messages": {}}
        mock_ds.select_columns.return_value = mock_ds
        # Fix MagicMock len() issue by creating a mock that looks like a Dataset
        # MagicMock.__len__ is tricky. We'll verify identity instead.
        
        mock_load.return_value = mock_ds
        mock_concat.return_value = mock_ds
        
        ds = manager.get_unified_train_dataset(enabled_categories=["reasoning"])
        # Verify it returns the concatenated result
        assert ds is mock_ds

def test_split_dataset(mock_data_root):
    manager = UniversalDatasetManager(data_root=str(mock_data_root))
    
    # Create a larger dataset for splitting
    large_ds_path = mock_data_root / "datasets/general/large_ds"
    large_ds_path.mkdir(parents=True)
    
    # We mock load_dataset, so file content doesn't matter much but path existence does for discovery
    (large_ds_path / "data.jsonl").touch()
    
    with patch("src.data.universal_manager.load_dataset") as mock_load:
        # Mock split behavior
        mock_ds = MagicMock()
        mock_train = MagicMock()
        mock_train.__len__.return_value = 80
        mock_test = MagicMock()
        mock_test.__len__.return_value = 10
        mock_val = MagicMock()
        mock_val.__len__.return_value = 10
        
        # ds.train_test_split returns {'train': ..., 'test': ...}
        mock_ds.train_test_split.return_value = {'train': mock_train, 'test': mock_ds} 
        # Second split on 'test' returns {'train': val, 'test': test}
        mock_ds.train_test_split.side_effect = [
            {'train': mock_train, 'test': mock_ds}, # First split (train/rest)
            {'train': mock_val, 'test': mock_test}   # Second split (val/test)
        ]
        
        mock_load.return_value = mock_ds
        
        ds = manager.load_dataset_by_name("large_ds")
        splits = manager.split_dataset(ds, test_size=0.1, val_size=0.1)
        
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
