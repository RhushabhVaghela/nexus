"""
Unit tests for hardware_optimizer.py
"""

import pytest
from unittest.mock import MagicMock, patch
import torch
import sys

from src.utils.hardware_optimizer import optimize_for_hardware

class TestHardwareOptimizer:
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(8, 6)) # Ampere
    @patch("os.environ", {})
    def test_optimize_ampere(self, mock_cap, mock_avail):
        # Mock both cuda and cudnn backends
        with patch("torch.backends.cuda") as mock_cuda, \
             patch("torch.backends.cudnn") as mock_cudnn:
            
            profile = optimize_for_hardware()
            
            assert profile["device"] == "cuda"
            assert profile["flash_attention"] is True
            # Verify attributes were set on the mock
            assert mock_cuda.matmul.allow_tf32 is True
            assert mock_cudnn.allow_tf32 is True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_capability", return_value=(7, 0)) # Volta
    @patch("os.environ", {})
    def test_optimize_volta(self, mock_cap, mock_avail):
        profile = optimize_for_hardware()
        assert profile["device"] == "cuda"
        assert profile["flash_attention"] is False

    @patch("torch.cuda.is_available", return_value=False)
    def test_optimize_cpu(self, mock_avail):
        profile = optimize_for_hardware()
        assert profile["device"] == "cpu"
        assert profile["flash_attention"] is False
