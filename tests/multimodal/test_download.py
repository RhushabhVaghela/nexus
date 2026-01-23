
import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

class TestMultimodalDownload(unittest.TestCase):
    
    def setUp(self):
        # Import the module to test
        from src.multimodal import download
        self.module = download
        
        # Store original objects
        self.original_load_dataset = getattr(self.module, 'load_dataset', None)
        self.original_Dataset = getattr(self.module, 'Dataset', None)
        
        # Create mocks
        self.mock_load_dataset = MagicMock()
        self.mock_Dataset = MagicMock()
        
        # Patch the module attributes
        self.module.load_dataset = self.mock_load_dataset
        self.module.Dataset = self.mock_Dataset

    def tearDown(self):
        # Restore original objects
        if self.original_load_dataset:
            self.module.load_dataset = self.original_load_dataset
        if self.original_Dataset:
            self.module.Dataset = self.original_Dataset

    def test_download_vision(self):
        mock_ds = MagicMock()
        self.mock_load_dataset.return_value = mock_ds
        # Setup mock for Dataset.from_list().save_to_disk() chain
        mock_static_ds = MagicMock()
        self.mock_Dataset.from_list.return_value = mock_static_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_vision_data(output_dir)
        
        # Check call arguments
        # The script calls it without keywords often or with different ones
        self.mock_load_dataset.assert_called()
        # Verify iteration and saving
        mock_ds.take.assert_called()
        self.mock_Dataset.from_list.assert_called()
        mock_static_ds.save_to_disk.assert_called()

    def test_download_audio(self):
        mock_ds = MagicMock()
        self.mock_load_dataset.return_value = mock_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_audio_data(output_dir)
        
        self.mock_load_dataset.assert_called()

    def test_download_video(self):
        mock_ds = MagicMock()
        self.mock_load_dataset.return_value = mock_ds
        
        output_dir = "/tmp/test_output"
        self.module.download_video_data(output_dir)
        
        self.mock_load_dataset.assert_called()

if __name__ == "__main__":
    unittest.main()
