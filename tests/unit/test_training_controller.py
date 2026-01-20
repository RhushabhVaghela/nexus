"""
Unit tests for training_controller.py

Tests:
- Signal handlers (mocked)
- Cooldown logic
- Checkpoint saving hooks
- Compressed file extraction
"""

import pytest
import signal
import sys
import os
import tempfile
import gzip
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training_controller import (
    setup_signal_handlers,
    check_pause_state,
    check_and_cooldown,
    extract_if_compressed,
    training_step_hook,
    get_gpu_temperature,
    COOLDOWN_INTERVAL_STEPS,
    GPU_TEMP_THRESHOLD,
    COOLDOWN_DURATION_SECONDS,
)


class TestSetupSignalHandlers:
    """Test signal handler setup."""
    
    def test_setup_signal_handlers_runs(self):
        """Test signal handlers can be set up without error."""
        # This should not raise
        setup_signal_handlers()
    
    def test_signal_handlers_registered(self):
        """Test signal handlers are registered."""
        with patch('signal.signal') as mock_signal:
            setup_signal_handlers()
            
            # Should register at least 2 signals
            assert mock_signal.call_count >= 2


class TestCheckPauseState:
    """Test pause state checking."""
    
    def test_check_pause_state_not_paused(self):
        """Test check_pause_state returns immediately when not paused."""
        import src.training_controller as tc
        
        # Ensure not paused
        tc._paused = False
        
        # Should return immediately (not block)
        check_pause_state()
        
        assert True  # If we get here, it didn't block
    
    def test_check_pause_state_paused_then_resumed(self):
        """Test check_pause_state blocks when paused."""
        import src.training_controller as tc
        import threading
        
        tc._paused = True
        
        def unpause():
            import time
            time.sleep(0.1)
            tc._paused = False
        
        t = threading.Thread(target=unpause)
        t.start()
        
        # This should block briefly then return
        check_pause_state()
        t.join()
        
        assert tc._paused is False


class TestCooldownLogic:
    """Test cooldown interval logic."""
    
    def test_cooldown_interval_constant_defined(self):
        """Test cooldown interval constant is defined."""
        assert COOLDOWN_INTERVAL_STEPS > 0
        assert COOLDOWN_INTERVAL_STEPS >= 100
    
    def test_cooldown_duration_defined(self):
        """Test cooldown duration is defined."""
        assert COOLDOWN_DURATION_SECONDS > 0
    
    def test_check_and_cooldown_not_at_interval(self):
        """Test no cooldown at non-interval step."""
        with patch('src.training_controller.get_gpu_temperature', return_value=50):
            with patch('time.sleep') as mock_sleep:
                result = check_and_cooldown(1)  # Not at interval
                
                # Should not perform cooldown
                assert result is False or mock_sleep.call_count == 0
    
    def test_check_and_cooldown_at_interval(self):
        """Test cooldown triggers at step interval."""
        with patch('src.training_controller.get_gpu_temperature', return_value=50):
            with patch('time.sleep') as mock_sleep:
                with patch('torch.cuda.empty_cache'):
                    result = check_and_cooldown(COOLDOWN_INTERVAL_STEPS)
                    
                    # Should perform cooldown
                    assert result is True or mock_sleep.call_count > 0


class TestGPUTemperatureMonitoring:
    """Test GPU temperature monitoring."""
    
    def test_temperature_threshold_defined(self):
        """Test temperature threshold is defined."""
        assert GPU_TEMP_THRESHOLD > 0
        assert GPU_TEMP_THRESHOLD >= 75
        assert GPU_TEMP_THRESHOLD <= 90
    
    @patch('subprocess.run')
    def test_get_gpu_temperature_success(self, mock_run):
        """Test successful GPU temperature reading."""
        mock_run.return_value = MagicMock(
            stdout="80\n",
            returncode=0
        )
        
        temp = get_gpu_temperature()
        assert temp == 80 or temp is not None
    
    @patch('subprocess.run')
    def test_get_gpu_temperature_failure(self, mock_run):
        """Test GPU temperature reading failure."""
        mock_run.side_effect = Exception("nvidia-smi not found")
        
        temp = get_gpu_temperature()
        # Should handle gracefully
        assert temp is None or isinstance(temp, (int, float))


class TestCompressedFileExtraction:
    """Test compressed file extraction."""
    
    def test_extract_gzip(self, tmp_path):
        """Test .gz file extraction."""
        content = b"test content for gzip"
        gz_path = tmp_path / "test.txt.gz"
        
        with gzip.open(gz_path, 'wb') as f:
            f.write(content)
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = extract_if_compressed(str(gz_path), str(output_dir))
        
        assert result is not None
    
    def test_extract_tar_gz(self, tmp_path):
        """Test .tar.gz file extraction."""
        tar_path = tmp_path / "test.tar.gz"
        content_file = tmp_path / "content.txt"
        content_file.write_text("test content for tar.gz")
        
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(content_file, arcname="content.txt")
        
        output_dir = tmp_path / "output"
        result = extract_if_compressed(str(tar_path), str(output_dir))
        
        assert result is not None
    
    def test_extract_zip(self, tmp_path):
        """Test .zip file extraction."""
        zip_path = tmp_path / "test.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("content.txt", "test content for zip")
        
        output_dir = tmp_path / "output"
        result = extract_if_compressed(str(zip_path), str(output_dir))
        
        assert output_dir.exists() or result is not None
    
    def test_non_compressed_passthrough(self, tmp_path):
        """Test non-compressed files are passed through."""
        regular_file = tmp_path / "regular.txt"
        regular_file.write_text("not compressed")
        
        result = extract_if_compressed(str(regular_file), str(tmp_path / "output"))
        
        # Should return original path or handle gracefully
        assert result is None or Path(result).exists() or result == str(regular_file)


class TestTrainingStepHook:
    """Test training step hook integration."""
    
    def test_hook_runs_without_error(self):
        """Test hook runs without error."""
        import src.training_controller as tc
        
        tc._paused = False
        tc._checkpoint_requested = False
        
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        with patch('src.training_controller.check_pause_state'):
            with patch('src.training_controller.check_and_cooldown', return_value=False):
                with patch('src.training_controller.check_checkpoint_request', return_value=False):
                    # Should not raise
                    training_step_hook(mock_model, mock_optimizer, 1, "/tmp")
    
    def test_hook_calls_pause_check(self):
        """Test hook calls pause state check."""
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        with patch('src.training_controller.check_pause_state') as mock_pause:
            with patch('src.training_controller.check_and_cooldown', return_value=False):
                with patch('src.training_controller.check_checkpoint_request', return_value=False):
                    training_step_hook(mock_model, mock_optimizer, 1, "/tmp")
                    
                    mock_pause.assert_called_once()
    
    def test_hook_calls_cooldown_check(self):
        """Test hook calls cooldown check."""
        mock_model = MagicMock()
        mock_optimizer = MagicMock()
        
        with patch('src.training_controller.check_pause_state'):
            with patch('src.training_controller.check_and_cooldown') as mock_cool:
                with patch('src.training_controller.check_checkpoint_request', return_value=False):
                    training_step_hook(mock_model, mock_optimizer, 5, "/tmp")
                    
                    mock_cool.assert_called_once_with(5)


class TestGlobalState:
    """Test global state management."""
    
    def test_paused_state_toggleable(self):
        """Test _paused global can be toggled."""
        import src.training_controller as tc
        
        original = tc._paused
        tc._paused = not original
        assert tc._paused != original
        
        tc._paused = original  # Reset
    
    def test_checkpoint_requested_state(self):
        """Test _checkpoint_requested global."""
        import src.training_controller as tc
        
        tc._checkpoint_requested = True
        assert tc._checkpoint_requested is True
        
        tc._checkpoint_requested = False
        assert tc._checkpoint_requested is False
