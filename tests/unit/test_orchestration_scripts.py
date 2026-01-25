import importlib
import pytest
from unittest.mock import MagicMock, patch

def test_script_01_normalize_to_messages():
    s01 = importlib.import_module("src.01_download_real_datasets")
    sample = {"instruction": "say hi", "output": "hi"}
    res = s01.normalize_to_messages(sample)
    assert res["messages"][0]["role"] == "user"

def test_script_01_main():
    s01 = importlib.import_module("src.01_download_real_datasets")
    with patch("sys.argv", ["01.py", "--limit", "1"]), \
         patch("src.01_download_real_datasets.load_config", return_value={"predistilled": [{"name": "t", "source": "s", "type": "huggingface"}]}), \
         patch("src.01_download_real_datasets.process_dataset", return_value=True):
        s01.main()

def test_script_02_main():
    s02 = importlib.import_module("src.02_download_benchmarks")
    with patch("sys.argv", ["02.py"]), \
         patch("src.02_download_benchmarks.os.environ.get", return_value="nexus"), \
         patch("src.02_download_benchmarks.load_dataset"):
        s02.main()

def test_script_03_main():
    s03 = importlib.import_module("src.03_load_premium_datasets")
    # Patch PremiumDatasetLoader instead of load_dataset_universal
    with patch("sys.argv", ["03.py", "--mode", "censored", "--limit", "1"]), \
         patch("src.03_load_premium_datasets.PremiumDatasetLoader") as mock_loader_cls, \
         patch("src.03_load_premium_datasets.os.environ.get", return_value="nexus"):
        mock_loader = mock_loader_cls.return_value
        mock_loader.load_all_datasets.return_value = MagicMock()
        s03.main()

def test_script_04_main():
    s04 = importlib.import_module("src.04_process_real_datasets")
    # Patch RealDataProcessor
    with patch("sys.argv", ["04.py"]), \
         patch("src.04_process_real_datasets.os.environ.get", return_value="nexus"), \
         patch("src.04_process_real_datasets.RealDataProcessor") as mock_proc_cls:
        mock_proc = mock_proc_cls.return_value
        mock_proc.process_category.return_value = 0
        s04.main()

def test_script_05_main():
    s05 = importlib.import_module("src.05_generate_repetitive_dataset")
    # Patch object directly
    with patch("sys.argv", ["05.py"]), \
         patch("src.05_generate_repetitive_dataset.os.environ.get", return_value="nexus"), \
         patch.object(s05, "CONFIG", {"target_samples": 1, "samples_per_file": 1, "output_dir": "/tmp"}), \
         patch("src.05_generate_repetitive_dataset.PromptRepetitionEngine") as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.generate_trajectory.return_value = {"prompt": "p", "messages": [{"role": "user", "content": "c"}]}
        s05.main()

def test_script_06_main():
    s06 = importlib.import_module("src.06_generate_preference_dataset")
    with patch("sys.argv", ["06.py", "--mode=censored"]), \
         patch("src.06_generate_preference_dataset.os.environ.get", return_value="nexus"), \
         patch.object(s06, "CONFIG", {"target_samples": 1, "samples_per_file": 1, "output_dir": "/tmp/preference-pairs-censored", "mode": "censored"}), \
         patch("src.06_generate_preference_dataset.PreferencePairEngine") as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.generate_preference_pair.return_value = {"prompt": "p", "chosen": "c", "rejected": "r"}
        s06.main()

def test_script_07_main():
    s07 = importlib.import_module("src.07_validate_all_datasets")
    with patch("sys.argv", ["07.py"]), \
         patch("src.07_validate_all_datasets.os.environ.get", return_value="nexus"), \
         patch("src.07_validate_all_datasets.os.walk", return_value=[("/tmp", [], ["f.jsonl"])]), \
         patch("src.07_validate_all_datasets.validate_jsonl_file", return_value={"summary": {"valid": 1, "total_checked": 1}}):
        s07.main()

def test_script_08_main():
    s08 = importlib.import_module("src.08_validate_benchmarks")
    with patch("sys.argv", ["08.py"]), \
         patch("src.08_validate_benchmarks.os.environ.get", return_value="nexus"), \
         patch("src.08_validate_benchmarks.os.walk", return_value=[("/tmp", [], ["f.jsonl"])]), \
         patch("src.08_validate_benchmarks.validate_benchmark_file", return_value={"valid_samples": 1}):
        s08.main()

def test_script_09_main():
    s09 = importlib.import_module("src.09_validate_premium_datasets")
    with patch("sys.argv", ["09.py"]), \
         patch("src.09_validate_premium_datasets.os.environ.get", return_value="nexus"), \
         patch("src.09_validate_premium_datasets.os.path.exists", return_value=True), \
         patch("src.09_validate_premium_datasets.PremiumDatasetValidator") as mock_val_cls:
        mock_val = mock_val_cls.return_value
        mock_val.get_summary.return_value = {"total_valid": 1, "total_checked": 1, "valid_rate": "100%", "source_distribution": {}}
        s09.main()

def test_script_10_main():
    s10 = importlib.import_module("src.10_sft_training")
    if hasattr(s10, "torch"):
        s10.torch.cuda.is_available = MagicMock(return_value=True)
    with patch("sys.argv", ["10.py", "--epochs", "1"]), \
         patch("src.10_sft_training.os.environ.get", return_value="nexus"), \
         patch("src.10_sft_training.SFTTrainer"), \
         patch("src.10_sft_training.AutoModelForCausalLM"), \
         patch("src.10_sft_training.AutoTokenizer"):
        s10.main()
