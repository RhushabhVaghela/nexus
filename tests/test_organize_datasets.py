import os
import pytest
import json
import shutil
from pathlib import Path
from src.utils.organize_datasets import inspect_content, get_destination, organize_folder, MODEL_CATEGORIES

# Fixture for creating temporary test files
@pytest.fixture
def temp_data_env(tmp_path):
    """
    Creates a temporary directory structure mimicking the data root.
    Returns the root path.
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    
    (data_root / "datasets").mkdir()
    (data_root / "benchmarks").mkdir()
    (data_root / "encoders").mkdir()
    (data_root / "decoders").mkdir()
    
    return data_root

def create_json_file(path, content):
    with open(path, 'w') as f:
        json.dump(content, f)

def create_jsonl_file(path, content_list):
    with open(path, 'w') as f:
        for item in content_list:
            f.write(json.dumps(item) + '\n')

# --- Tests for inspect_content ---

def test_inspect_content_cot(tmp_path):
    p = tmp_path / "cot.json"
    create_json_file(p, [{"messages": [{"role": "user", "content": "hi"}]}])
    assert inspect_content(p) == "cot"

def test_inspect_content_tools(tmp_path):
    p = tmp_path / "tools.jsonl"
    create_jsonl_file(p, [{"input": "some input", "tool_calls": ["call1"]}])
    assert inspect_content(p) == "tools"

def test_inspect_content_vision(tmp_path):
    p = tmp_path / "vision.json"
    create_json_file(p, {"image": "base64...", "text": "desc"})
    assert inspect_content(p) == "vision-qa"

def test_inspect_content_reasoning(tmp_path):
    p = tmp_path / "reasoning.jsonl"
    create_jsonl_file(p, [{"problem": "1+1", "solution": "2"}])
    assert inspect_content(p) == "reasoning"

def test_inspect_content_empty_or_invalid(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("not json")
    assert inspect_content(p) is None

# --- Tests for get_destination ---

def test_get_destination_benchmark():
    # Keyword override
    p = Path("/data/datasets/gsm8k_test.json")
    base = Path("/data")
    top, cat, sub = get_destination(p, base)
    assert top == "benchmarks"
    assert cat == "reasoning"

def test_get_destination_dataset_keyword():
    p = Path("/data/datasets/alpaca_cot.json")
    base = Path("/data")
    top, cat, sub = get_destination(p, base)
    assert top == "datasets"
    assert cat == "cot"

def test_get_destination_encoder_vision():
    p = Path("/data/encoders/clip-vit-large.bin")
    base = Path("/data")
    top, cat, sub = get_destination(p, base)
    assert top == "encoders"
    assert sub == "vision-encoders"

def test_get_destination_encoder_audio():
    p = Path("/data/encoders/whisper-v3.bin")
    base = Path("/data")
    top, cat, sub = get_destination(p, base)
    assert top == "encoders"
    assert sub == "audio-encoders"

def test_get_destination_decoder_audio():
    p = Path("/data/decoders/music-gen.bin")
    base = Path("/data")
    top, cat, sub = get_destination(p, base)
    assert top == "decoders"
    assert sub == "audio-decoders"

# --- Integration Tests ---

def test_organize_folder_integration(temp_data_env):
    root = temp_data_env
    
    # 1. Setup Files
    
    # Dataset - CoT (Content based)
    cot_file = root / "datasets" / "my_chat_data.json"
    create_json_file(cot_file, [{"messages": []}])
    
    # Dataset - Tools (Content based)
    tool_file = root / "datasets" / "functions.jsonl"
    create_jsonl_file(tool_file, [{"tool_calls": []}])
    
    # Benchmark
    bench_file = root / "datasets" / "mmlu_eval.json"
    create_json_file(bench_file, {"test": True})
    
    # Encoder - Vision
    vision_enc = root / "encoders" / "siglip.model"
    vision_enc.touch()
    
    # 2. Run Organize
    # We test organize_folder individually or the main optimize function if importable.
    # organize_folder is cleaner for unit testing.
    
    # Organize Datasets
    organize_folder(root / "datasets", root, move=True, is_model_dir=False)
    
    # Organize Encoders
    organize_folder(root / "encoders", root, move=True, is_model_dir=True)
    
    # 3. Verify Moves
    
    # CoT should be in datasets/cot
    assert (root / "datasets/cot/my_chat_data.json").exists()
    assert not cot_file.exists()
    
    # Tools should be in datasets/tools
    assert (root / "datasets/tools/functions.jsonl").exists()
    
    # Benchmark should be in benchmarks/uncategorized (mmlu keyword matches benchmark list, but maybe not a specific category if just 'mmlu' unless mapped)
    # Check mappings: mmlu is in BENCHMARK_KEYWORDS. 
    # 'mmlu' is NOT in TRAINING_CATEGORIES key list? 
    # Wait, TRAINING_CATEGORIES has "reasoning": [..., "gsm8k",...]. 
    # "mmlu" is only in BENCHMARK_KEYWORDS?
    # Let's check logic: if is_benchmark=True, best_category is keyword matched from TRAINING_CATEGORIES.
    # If "mmlu" is not in TRAINING_CATEGORIES, it returns "uncategorized".
    assert (root / "benchmarks/uncategorized/mmlu_eval.json").exists()
    
    # Vision Encoder
    assert (root / "encoders/vision-encoders/siglip.model").exists()
