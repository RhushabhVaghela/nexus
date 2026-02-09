import os
from pathlib import Path
import datetime
from nexus.config.paths import OUTPUT_DIR

class CorruptionTracker:
    """
    Logs corrupted file paths to a central log file.
    """
    def __init__(self, log_path: str = f"{OUTPUT_DIR}/corrupted_files.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_corrupted(self, file_path: str, error: str):
        """Log a corrupted file with timestamp and error message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] CORRUPTED: {file_path} | ERROR: {error}\n"
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(f"⚠️ Logged corrupted file: {file_path}")

# Global instance
tracker = CorruptionTracker()
