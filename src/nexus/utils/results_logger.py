#!/usr/bin/env python3
"""
CSV Results Logger for tracking training experiments
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ResultsLogger:
    """Log training results to CSV for comparison."""
    
    def __init__(self, csv_path: str = "training_results.csv"):
        self.csv_path = Path(csv_path)
        self.fieldnames = [
            "timestamp",
            "experiment_name",
            "sample_size",
            "optimization_level",  # optimized or ultra
            "train_samples",
            "val_samples",
            "test_samples",
            "training_time_minutes",
            "final_train_loss",
            "final_val_loss",
            "final_test_loss",
            "peak_vram_gb",
            "peak_ram_gb",
            "throughput_samples_per_sec",
            "model_size_gb",
            "config_used"
        ]
        
        # Create CSV with headers if doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log_result(self, result: Dict[str, Any]):
        """Append a result to the CSV."""
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        # Ensure all fields present
        row = {field: result.get(field, "N/A") for field in self.fieldnames}
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
        
        print(f"✓ Results logged to {self.csv_path}")
    
    def get_all_results(self):
        """Read all results from CSV."""
        results = []
        if self.csv_path.exists():
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                results = list(reader)
        return results
