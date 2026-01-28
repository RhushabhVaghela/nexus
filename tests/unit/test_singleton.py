import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock
from scripts.nexus_pipeline import NexusPipeline

class TestSingletonLock(unittest.TestCase):
    @patch("scripts.nexus_pipeline.os.path.exists")
    @patch("scripts.nexus_pipeline.open", new_callable=mock_open, read_data="12345")
    @patch("scripts.nexus_pipeline.os.kill")
    def test_singleton_conflict_detected(self, mock_kill, mock_file, mock_exists):
        """
        Test that NexusPipeline detects an existing lock file with a live PID and exits.
        """
        # Setup: Lock file exists
        mock_exists.return_value = True
        
        # Setup: PID 12345 is alive (os.kill returns None)
        mock_kill.return_value = None

        # Expect SystemExit(1)
        with self.assertRaises(SystemExit) as cm:
            NexusPipeline(dry_run=True)
        
        self.assertEqual(cm.exception.code, 1)
        mock_kill.assert_called_with(12345, 0)

    @patch("scripts.nexus_pipeline.os.path.exists")
    @patch("scripts.nexus_pipeline.open", new_callable=mock_open, read_data="99999")
    @patch("scripts.nexus_pipeline.os.kill")
    def test_singleton_stale_lock_cleanup(self, mock_kill, mock_file, mock_exists):
        """
        Test that NexusPipeline overrides the lock if the PID is dead.
        """
        # Setup: Lock file exists
        # One check for init read, one check for atexit remove? 
        # Actually NexusPipeline checks existence first.
        mock_exists.side_effect = [True, False] 
        
        # Setup: PID 99999 is dead -> ProcessLookupError (OSError)
        mock_kill.side_effect = OSError
        
        # Should NOT exit, should acquire lock
        try:
            pipeline = NexusPipeline(dry_run=True)
            # Verify it tried to write new PID
            mock_file.assert_called() 
            handle = mock_file()
            handle.write.assert_called()
        except SystemExit:
            self.fail("NexusPipeline exited on stale lock!")

if __name__ == "__main__":
    unittest.main()
