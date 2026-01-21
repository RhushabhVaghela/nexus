#!/usr/bin/env python3
import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.streaming_trainer import StreamingDatasetLoader

class TestLoaderDelegation(unittest.TestCase):
    """Test that StreamingDatasetLoader delegates giant/media files correctly."""
    
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        
    def tearDown(self):
        self.tmp_dir.cleanup()
    
    def test_delegates_media_file(self):
        """Test delegation of .mp4 file."""
        # Create a mock mp4
        media_file = self.tmp_path / "test.mp4"
        media_file.touch()
        
        loader = StreamingDatasetLoader(media_file)
        
        # We access the internal generator to verify delegation
        with patch('src.data.streaming_trainer.ChunkedSampleProcessor') as MockProcessor:
            MockProcessor.return_value.stream_chunks.return_value = iter([{'mock': 'chunk'}])
            
            # Consume one sample
            gen = loader._stream_path(media_file)
            sample = next(gen)
            
            # Assert ChunkedSampleProcessor was initialized and used
            # We assume it is called with the path
            MockProcessor.assert_called()
            args, _ = MockProcessor.call_args
            assert str(args[0]) == str(media_file)
            assert sample == {'mock': 'chunk'}
            print("✓ test_delegates_media_file passed")
            
    def test_delegates_giant_file(self):
        """Test delegation of >1GB file."""
        giant_file = self.tmp_path / "giant.txt"
        giant_file.touch()
        
        loader = StreamingDatasetLoader(giant_file)
        
        
        # Mocking stat to return > 1GB and valid file mode
        import stat
        with patch('pathlib.Path.stat') as mock_stat_method:
            mock_stat_method.return_value.st_size = 2 * 1024**3 # 2GB
            mock_stat_method.return_value.st_mode = stat.S_IFREG | 0o644 # Regular file
            
            with patch('src.data.streaming_trainer.ChunkedSampleProcessor') as MockProcessor:
                MockProcessor.return_value.stream_chunks.return_value = iter([{'mock': 'chunk'}])
                
                gen = loader._stream_path(giant_file)
                sample = next(gen)
                
                MockProcessor.assert_called()
                assert sample == {'mock': 'chunk'}
                print("✓ test_delegates_giant_file passed")

if __name__ == '__main__':
    unittest.main()
