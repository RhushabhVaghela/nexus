"""
Unit tests for top-level orchestration scripts (21-26).
(MOCKED)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib

# Mock missing modules
sys.modules["unsloth"] = MagicMock()
sys.modules["mpi4py"] = MagicMock()
sys.modules["deepspeed"] = MagicMock()

def test_script_21_main():
    with patch("sys.argv", ["21.py"]), \
         patch("src.21_deployment_configs.check_env", return_value=True):
        s21 = importlib.import_module("src.21_deployment_configs")
        s21.main()

def test_script_22_main():
    with patch("sys.argv", ["22.py", "--phase", "test"]), \
         patch("src.22_multimodal_pipeline.os.environ.get", return_value="nexus"), \
         patch("src.22_multimodal_pipeline.DATASET_REGISTRY", {}), \
         patch("src.22_multimodal_pipeline.DatasetManager"), \
         patch("src.22_multimodal_pipeline.get_test_prompts", return_value={}):
        s22 = importlib.import_module("src.22_multimodal_pipeline")
        s22.main()

def test_script_23_main():
    with patch("sys.argv", ["23.py"]), \
         patch("src.23_multimodal_distillation.MultimodalDataProcessor"), \
         patch("src.23_multimodal_distillation.os.environ.get", return_value="nexus"), \
         patch("src.23_multimodal_distillation.Path.exists", return_value=True):
        s23 = importlib.import_module("src.23_multimodal_distillation")
        s23.main()

def test_script_24_main():
    # Force clean sys.modules for deepspeed to avoid contamination from previous tests
    with patch.dict(sys.modules):
        if "deepspeed" in sys.modules:
            del sys.modules["deepspeed"]
            
        with patch("sys.argv", ["24.py", "--data-path", "fake"]), \
             patch("src.multimodal.model.OmniMultimodalLM"), \
             patch("transformers.Trainer"), \
             patch("transformers.TrainingArguments"), \
             patch("src.24_multimodal_training.os.environ.get", return_value="nexus"), \
             patch("src.24_multimodal_training.OmniDataset"), \
             patch("src.24_multimodal_training.DynamicDataCollator"), \
             patch("importlib.util.find_spec", return_value=None):
            
            # Re-import to ensure mocks apply
            if "src.24_multimodal_training" in sys.modules:
                importlib.reload(sys.modules["src.24_multimodal_training"])
            else:
                importlib.import_module("src.24_multimodal_training")
            
            s24 = sys.modules["src.24_multimodal_training"]
            s24.main()

def test_script_25_main():
    with patch.dict(sys.modules):
        if "deepspeed" in sys.modules:
            del sys.modules["deepspeed"]
            
        with patch("sys.argv", ["25.py"]), \
             patch("src.streaming.memory.StreamingMemory"), \
             patch("src.streaming.tts.TTSStreamer"), \
             patch("src.streaming.vision.VisionStreamBuffer"), \
             patch("src.multimodal.model.OmniMultimodalLM"), \
             patch("importlib.util.find_spec", return_value=None):
            
            if "src.25_realtime_streaming" in sys.modules:
                importlib.reload(sys.modules["src.25_realtime_streaming"])
            else:
                importlib.import_module("src.25_realtime_streaming")
                
            s25 = sys.modules["src.25_realtime_streaming"]
            s25.main()

def test_script_26_main():
    # Mock transformers entirely to avoid torch kernel registration conflicts
    transformers_mock = MagicMock()
    
    # Create mocks for dependencies
    deepspeed_mock = MagicMock()
    deepspeed_mock.comm = MagicMock()
    deepspeed_mock.comm.comm = MagicMock() # For mpi_discovery patch
    
    with patch.dict(sys.modules):
        # PREVENT real deepspeed import by populating sys.modules with a mock
        sys.modules["deepspeed"] = deepspeed_mock
        sys.modules["deepspeed.comm"] = deepspeed_mock.comm
        sys.modules["deepspeed.comm.comm"] = deepspeed_mock.comm.comm
        
        sys.modules["transformers"] = transformers_mock
        sys.modules["safetensors"] = MagicMock()
        sys.modules["safetensors.torch"] = MagicMock()
            
        with patch("sys.argv", ["26.py", "--epochs", "1", "--model-path", "fake"]), \
             patch("src.26_distributed_training.setup_distributed", return_value=0), \
             patch("importlib.util.find_spec", return_value=None):
            
            # We don't need to patch deepspeed.initialize via string because we injected the mock module
            # But the script imports it. 
            
            if "src.26_distributed_training" in sys.modules:
                importlib.reload(sys.modules["src.26_distributed_training"])
            else:
                importlib.import_module("src.26_distributed_training")
                
            s26 = sys.modules["src.26_distributed_training"]
            
            # Mock the imported deepspeed in the script namespace if needed, 
            # but since we mocked sys.modules, import deepspeed should return our mock.
            
            # Script 26 might check for torch.distributed.is_initialized()
            with patch("torch.distributed.is_initialized", return_value=True), \
                 patch("torch.distributed.get_rank", return_value=0), \
                 patch("torch.distributed.get_world_size", return_value=1):
                try:
                    s26.main()
                except SystemExit:
                    pass

