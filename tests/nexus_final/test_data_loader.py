import os
import json
import pytest
import shutil
from src.nexus_final.data_loader import UniversalDataLoader

@pytest.fixture
def mock_data_env(tmp_path):
    """
    Creates a temporary directory with various dataset formats.
    """
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    
    # 1. Translation (Google Smol)
    smol_dir = datasets_dir / "general" / "google_smol"
    smol_dir.mkdir(parents=True)
    with open(smol_dir / "train.jsonl", "w") as f:
        f.write(json.dumps({"src": "Afar", "trgs": ["Qafar"], "sl": "en", "tl": "aa"}) + "\n")
        f.write(json.dumps({"src": "Hello", "trgs": ["Hola"], "sl": "en", "tl": "es"}) + "\n")

    # 2. Reasoning (Problem/Solution)
    math_dir = datasets_dir / "reasoning" / "math"
    math_dir.mkdir(parents=True)
    with open(math_dir / "data.jsonl", "w") as f:
        f.write(json.dumps({"problem": "2+2", "solution": "4"}) + "\n")

    # 3. Code (Contents)
    code_dir = datasets_dir / "code" / "stack"
    code_dir.mkdir(parents=True)
    with open(code_dir / "code.jsonl", "w") as f:
        f.write(json.dumps({"contents": "print('hello')"}) + "\n")

    # 4. Audio CSV
    audio_dir = datasets_dir / "multimodal" / "common_voice"
    audio_dir.mkdir(parents=True)
    with open(audio_dir / "meta.csv", "w") as f:
        f.write("id,sentence\n1,Hello world\n")

    return str(datasets_dir)

def test_normalization_translation(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("general/google_smol", limit=1))
    
    assert len(samples) == 1
    msg = samples[0]["messages"]
    assert "Translate from en to aa" in msg[0]["content"]
    assert msg[1]["content"] == "Qafar"

def test_normalization_reasoning(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("reasoning/math", limit=1))
    
    assert len(samples) == 1
    msg = samples[0]["messages"]
    assert msg[0]["content"] == "2+2"
    assert msg[1]["content"] == "4"

def test_normalization_code(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("code/stack", limit=1))
    
    assert len(samples) == 1
    msg = samples[0]["messages"]
    assert "Complete the following code" in msg[0]["content"]
    assert msg[1]["content"] == "print('hello')"

def test_limit_functionality(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("general/google_smol", limit=1))
    assert len(samples) == 1

def test_missing_dataset(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("non/existent"))
    assert len(samples) == 0

def test_list_datasets(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    datasets = loader.list_available_datasets()
    assert "general/google_smol" in datasets
    assert "reasoning/math" in datasets
    assert "code/stack" in datasets
    assert "multimodal/common_voice" in datasets
    assert len(datasets) == 4

def test_normalization_csv(mock_data_env):
    loader = UniversalDataLoader(data_root=mock_data_env)
    samples = list(loader.load_dataset("multimodal/common_voice", limit=1))
    assert len(samples) == 1
    msg = samples[0]["messages"]
    assert "Transcribe" in msg[0]["content"]
    assert msg[1]["content"] == "Hello world"
