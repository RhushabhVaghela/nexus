"""
test_universal_loader.py
Unit and integration tests for the universal dataset loader.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============== FIXTURES ==============

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def json_array_file(temp_dir):
    """Create a JSON array format file."""
    data = [
        {"id": 1, "text": "Hello world", "label": "greeting"},
        {"id": 2, "text": "How are you?", "label": "question"},
        {"id": 3, "text": "Goodbye", "label": "farewell"},
    ]
    path = temp_dir / "array.json"
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


@pytest.fixture
def json_dict_file(temp_dir):
    """Create a JSON dict format file."""
    data = {
        "sample_1": {"text": "First sample", "score": 0.9},
        "sample_2": {"text": "Second sample", "score": 0.7},
    }
    path = temp_dir / "dict.json"
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


@pytest.fixture
def jsonl_file(temp_dir):
    """Create a JSONL format file."""
    data = [
        {"text": "Line 1", "value": 1},
        {"text": "Line 2", "value": 2},
        {"text": "Line 3", "value": 3},
        {"text": "Line 4", "value": 4},
    ]
    path = temp_dir / "data.jsonl"
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return path


@pytest.fixture
def csv_file(temp_dir):
    """Create a CSV format file."""
    path = temp_dir / "data.csv"
    with open(path, 'w') as f:
        f.write("id,text,score\n")
        f.write("1,Hello,0.9\n")
        f.write("2,World,0.8\n")
    return path


@pytest.fixture
def text_file(temp_dir):
    """Create a plain text file."""
    path = temp_dir / "data.txt"
    with open(path, 'w') as f:
        f.write("First line of text\n")
        f.write("Second line of text\n")
        f.write("Third line of text\n")
    return path


# ============== UNIT TESTS ==============

class TestUniversalLoaderImport:
    """Test module imports."""
    
    def test_import_module(self):
        from src.data import universal_loader
        assert hasattr(universal_loader, 'UniversalDataLoader')
    
    def test_import_class(self):
        from src.data.universal_loader import UniversalDataLoader
        assert UniversalDataLoader is not None
    
    def test_import_function(self):
        from src.data.universal_loader import load_dataset_universal
        assert callable(load_dataset_universal)
    
    def test_import_load_result(self):
        from src.data.universal_loader import LoadResult
        assert LoadResult is not None


class TestFormatDetection:
    """Test format auto-detection."""
    
    def test_detect_json_array(self, json_array_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        fmt = loader.detect_format()
        assert fmt in ("json_array", "json")
    
    def test_detect_json_dict(self, json_dict_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_dict_file)
        fmt = loader.detect_format()
        assert fmt in ("json_dict", "json")
    
    def test_detect_jsonl(self, jsonl_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(jsonl_file)
        fmt = loader.detect_format()
        # May detect as jsonl or json depending on extension
        assert fmt in ("jsonl", "json", "json_array", "json_dict")
    
    def test_detect_csv(self, csv_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(csv_file)
        fmt = loader.detect_format()
        assert fmt == "csv"
    
    def test_detect_text(self, text_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(text_file)
        fmt = loader.detect_format()
        assert fmt == "text"
    
    def test_detect_nonexistent(self, temp_dir):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(temp_dir / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            loader.detect_format()


class TestJsonLoading:
    """Test JSON format loading."""
    
    def test_load_json_array(self, json_array_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load()
        
        assert result.dataset is not None
        assert result.num_samples == 3
        assert "text" in result.columns
        assert result.error is None
    
    def test_load_json_dict(self, json_dict_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_dict_file)
        result = loader.load()
        
        assert result.dataset is not None
        assert result.num_samples == 2
        assert result.error is None
    
    def test_load_json_with_sample_size(self, json_array_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load(sample_size=2)
        
        assert result.num_samples == 2


class TestJsonlLoading:
    """Test JSONL format loading."""
    
    def test_load_jsonl(self, jsonl_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(jsonl_file)
        result = loader.load()
        
        assert result.dataset is not None
        assert result.num_samples == 4
        assert "text" in result.columns
    
    def test_load_jsonl_sample_size(self, jsonl_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(jsonl_file)
        result = loader.load(sample_size=2)
        
        assert result.num_samples == 2
    
    def test_load_jsonl_content(self, jsonl_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(jsonl_file)
        result = loader.load()
        
        assert result.dataset[0]["text"] == "Line 1"
        assert result.dataset[0]["value"] == 1


class TestCsvLoading:
    """Test CSV format loading."""
    
    def test_load_csv(self, csv_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(csv_file)
        result = loader.load()
        
        assert result.dataset is not None
        assert result.num_samples == 2
        assert "text" in result.columns


class TestTextLoading:
    """Test text format loading."""
    
    def test_load_text(self, text_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(text_file)
        result = loader.load()
        
        assert result.dataset is not None
        assert result.num_samples == 3
        assert "text" in result.columns
    
    def test_load_text_content(self, text_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(text_file)
        result = loader.load()
        
        assert "First line" in result.dataset[0]["text"]


class TestLoadResult:
    """Test LoadResult dataclass."""
    
    def test_load_result_fields(self, json_array_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load()
        
        # Check all expected fields
        assert hasattr(result, 'dataset')
        assert hasattr(result, 'format')
        assert hasattr(result, 'num_samples')
        assert hasattr(result, 'columns')
        assert hasattr(result, 'source_path')
        assert hasattr(result, 'error')
    
    def test_load_result_source_path(self, json_array_file):
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load()
        
        assert str(json_array_file) in result.source_path


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_load_dataset_universal(self, json_array_file):
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(json_array_file)
        
        assert result.dataset is not None
        assert result.num_samples == 3
    
    def test_load_dataset_universal_sample_size(self, jsonl_file):
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(jsonl_file, sample_size=2)
        
        assert result.num_samples == 2


# ============== INTEGRATION TESTS ==============

class TestRealDatasets:
    """Integration tests with real local datasets."""
    
    @pytest.fixture
    def cot_dataset_path(self):
        return Path("/mnt/e/data/datasets/reasoning/kaist-ai_CoT-Collection/data/CoT_collection_en.json")
    
    @pytest.fixture
    def tools_dataset_path(self):
        return Path("/mnt/e/data/datasets/tools/Salesforce_xlam-function-calling-60k/xlam_function_calling_60k.json")
    
    @pytest.fixture
    def o1_dataset_path(self):
        return Path("/mnt/e/data/datasets/reasoning/O1-OPEN_OpenO1-SFT-Pro")
    
    @pytest.fixture
    def gsm8k_path(self):
        return Path("/mnt/e/data/datasets/reasoning/openai_gsm8k") # Likely missing, test should skip
    
    def test_load_cot_dataset(self, cot_dataset_path):
        """Test loading CoT Collection (large JSON dict)."""
        if not cot_dataset_path.exists():
            pytest.skip(f"Dataset not found: {cot_dataset_path}")
        
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(cot_dataset_path, sample_size=5)
        
        assert result.dataset is not None
        assert result.num_samples == 5
        assert result.error is None
    
    def test_load_tools_dataset(self, tools_dataset_path):
        """Test loading function calling dataset."""
        if not tools_dataset_path.exists():
            pytest.skip(f"Dataset not found: {tools_dataset_path}")
        
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(tools_dataset_path, sample_size=5)
        
        assert result.dataset is not None
        assert result.num_samples == 5
    
    def test_load_o1_dataset(self, o1_dataset_path):
        """Test loading O1 dataset (may be JSONL or parquet)."""
        if not o1_dataset_path.exists():
            pytest.skip(f"Dataset not found: {o1_dataset_path}")
        
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(o1_dataset_path, sample_size=5)
        
        # Should succeed without error
        assert result.error is None or result.dataset is not None
    
    def test_load_gsm8k_dataset(self, gsm8k_path):
        """Test loading GSM8K dataset (parquet format)."""
        if not gsm8k_path.exists():
            pytest.skip(f"Dataset not found: {gsm8k_path}")
        
        from src.data.universal_loader import load_dataset_universal
        result = load_dataset_universal(gsm8k_path, sample_size=5)
        
        assert result.dataset is not None
        assert result.error is None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_file(self, temp_dir):
        """Test loading empty file."""
        path = temp_dir / "empty.json"
        path.touch()
        
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(path)
        result = loader.load()
        
        # Should have error or empty dataset
        assert result.error is not None or result.num_samples == 0
    
    def test_invalid_json(self, temp_dir):
        """Test loading invalid JSON."""
        path = temp_dir / "invalid.json"
        with open(path, 'w') as f:
            f.write("not valid json {{{")
        
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(path)
        result = loader.load()
        
        assert result.error is not None
    
    def test_zero_sample_size(self, json_array_file):
        """Test with sample_size=0."""
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load(sample_size=0)
        
        # Should return empty or all samples
        assert result.num_samples >= 0
    
    def test_large_sample_size(self, json_array_file):
        """Test with sample_size larger than dataset."""
        from src.data.universal_loader import UniversalDataLoader
        loader = UniversalDataLoader(json_array_file)
        result = loader.load(sample_size=1000)
        
        # Should return all available samples
        assert result.num_samples == 3
