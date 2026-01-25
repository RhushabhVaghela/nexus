import pytest
import json
from unittest.mock import patch, MagicMock
from src.utils.validate_dataset_diversity import validate_diversity

def test_validate_diversity(tmp_path):
    # Create mock dataset
    data_file = tmp_path / "dataset.jsonl"
    samples = [
        {"category": "c1", "output": "NexusTimeline"},
        {"category": "c1", "output": "NexusChart"},
        {"category": "c1", "output": "Unknown"},
    ]
    with open(data_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    # Mock rich Console to avoid printing during test
    with patch("src.utils.validate_dataset_diversity.Console") as mock_console_cls:
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console
        
        validate_diversity(str(data_file))
        
        # Verify table was added to console
        assert mock_console.print.called
        # Check if table was passed
        args, kwargs = mock_console.print.call_args
        from rich.table import Table
        assert isinstance(args[0], Table)
