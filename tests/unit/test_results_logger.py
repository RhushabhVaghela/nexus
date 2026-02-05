import pytest
import csv
from pathlib import Path
from src.utils.results_logger import ResultsLogger

def test_results_logger(tmp_path):
    csv_file = tmp_path / "results.csv"
    logger = ResultsLogger(str(csv_file))
    
    assert csv_file.exists()
    
    result = {
        "experiment_name": "test_exp",
        "final_train_loss": 0.123
    }
    logger.log_result(result)
    
    results = logger.get_all_results()
    assert len(results) == 1
    assert results[0]["experiment_name"] == "test_exp"
    assert results[0]["final_train_loss"] == "0.123"
    assert "timestamp" in results[0]
