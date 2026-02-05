
import pytest
from src.data.universal_manager import UniversalDatasetManager

# This test requires the reorganization to have run, OR it mocks appropriately.
# Since we haven't run reorganization yet, we should use a mock root or skip if on real data not found.
# But for integration, we want to see if it works with *some* real folder structure if it existed.
# Given we are in the middle of refactoring, let's use a temporary integration test 
# that creates a mini-structure to test the *integration* of components without relying on E:/data yet.

@pytest.fixture
def integration_root(tmp_path):
    d = tmp_path / "datasets" / "reasoning" / "mock_cot"
    d.mkdir(parents=True)
    
    # Create valid huggingface-like json
    data = [{"instruction": "solve", "output": "42"}] * 100
    import json
    with open(d / "dataset_dict.json", "w") as f:
        # Mocking a HF dictionary structure roughly or just raw files
        pass 
    
    with open(d / "train.json", "w") as f:
        json.dump(data, f)
        
    return tmp_path

def test_manager_integration_flow(integration_root):
    """
    Simulates the flow:
    1. Manager initialized
    2. Loads a dataset
    3. Splits it
    4. Prepares for training (get_unified)
    """
    manager = UniversalDatasetManager(data_root=str(integration_root))
    
    # 1. Load dictionary style
    # We call it 'mock_cot'
    ds = manager.load_dataset_by_name("mock_cot")
    assert ds is not None
    assert len(ds) == 100
    
    # 2. Split
    splits = manager.split_dataset(ds)
    # Default split 0.05 test, 0.05 val -> 0.1 total test size
    # 100 items -> 90 train, 10 test_val
    # 10 test_val -> split 50/50 (0.05/0.1) -> 5 val, 5 test
    assert len(splits['train']) == 90
    assert len(splits['validation']) == 5
    assert len(splits['test']) == 5
    
    # 3. Unified (simulate training input)
    # The mocks have 'instruction'/'output' but no messages. 
    # Real integration test should check if normalized logic works, 
    # but our current 'universal_manager' only supports 'messages' or 'conversations'
    # Let's see if it handles non-compliant features gracefully (warns/skips) 
    # or if we need to mock 'messages' column.
    
    # Let's update the mock data to have messages
    d = integration_root / "datasets" / "reasoning" / "mock_cot"
    import json
    data = [{"messages": [{"role": "user", "content": "x"}]}] * 100
    with open(d / "train.json", "w") as f:
        json.dump(data, f)
        
    # Reload and unify
    ds_new = manager.load_dataset_by_name("mock_cot")
    unified = manager.get_unified_train_dataset(included_datasets=["mock_cot"])
    assert len(unified) == 100
    assert "messages" in unified.features

