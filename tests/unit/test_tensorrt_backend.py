"""
Unit tests for TensorRT Backend

This module contains comprehensive tests for the TensorRT integration including:
- TRTEngine configuration and initialization
- TRTQuantizationMode
- TensorRTBackend generation methods
- Model conversion
- Error handling

Author: Nexus Team
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import json

import torch
import numpy as np


# Mock TensorRT-LLM availability before importing
import sys
sys.modules['tensorrt_llm'] = MagicMock()
sys.modules['tensorrt_llm.runtime'] = MagicMock()

from src.models.tensorrt.trt_engine import (
    TRTQuantizationMode,
    TRTBuildConfig,
    TRTEngineConfig,
    TRTEngineError,
    TRTEngine,
    TRTLLM_AVAILABLE,
)

from src.models.tensorrt.inference_backend import (
    BackendError,
    TensorRTConfig,
    GenerationResult,
    TensorRTBackend,
)


class TestTRTQuantizationMode(unittest.TestCase):
    """Test TRTQuantizationMode enum."""
    
    def test_quantization_values(self):
        """Test quantization mode values."""
        self.assertEqual(TRTQuantizationMode.FP32.value, "fp32")
        self.assertEqual(TRTQuantizationMode.FP16.value, "fp16")
        self.assertEqual(TRTQuantizationMode.BF16.value, "bf16")
        self.assertEqual(TRTQuantizationMode.FP8.value, "fp8")
        self.assertEqual(TRTQuantizationMode.INT8.value, "int8")
        self.assertEqual(TRTQuantizationMode.INT4.value, "int4")
        self.assertEqual(TRTQuantizationMode.WOQ.value, "woq")


class TestTRTBuildConfig(unittest.TestCase):
    """Test TRTBuildConfig dataclass."""
    
    def test_default_values(self):
        """Test default build configuration."""
        config = TRTBuildConfig()
        
        self.assertEqual(config.max_batch_size, 1)
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.max_input_len, 1024)
        self.assertEqual(config.max_output_len, 1024)
        self.assertEqual(config.max_beam_width, 1)
        self.assertEqual(config.dtype, "float16")
        self.assertEqual(config.quantization, TRTQuantizationMode.FP16)
        self.assertTrue(config.use_gpt_attention_plugin)
        self.assertTrue(config.use_gemm_plugin)
        self.assertTrue(config.use_layernorm_plugin)
        self.assertEqual(config.opt_level, 3)
        self.assertFalse(config.strongly_typed)
    
    def test_custom_values(self):
        """Test custom build configuration."""
        config = TRTBuildConfig(
            max_batch_size=8,
            max_seq_length=4096,
            quantization=TRTQuantizationMode.FP8,
            dtype="float16"
        )
        
        self.assertEqual(config.max_batch_size, 8)
        self.assertEqual(config.max_seq_length, 4096)
        self.assertEqual(config.quantization, TRTQuantizationMode.FP8)


class TestTRTEngineConfig(unittest.TestCase):
    """Test TRTEngineConfig dataclass."""
    
    def test_engine_path_only(self):
        """Test config with engine path."""
        config = TRTEngineConfig(engine_path="/path/to/engine")
        
        self.assertEqual(config.engine_path, "/path/to/engine")
        self.assertIsNone(config.model_path)
    
    def test_model_path_only(self):
        """Test config with model path."""
        config = TRTEngineConfig(model_path="/path/to/model")
        
        self.assertIsNone(config.engine_path)
        self.assertEqual(config.model_path, "/path/to/model")
    
    def test_neither_path_raises(self):
        """Test that ValueError is raised if neither path provided."""
        with self.assertRaises(ValueError):
            TRTEngineConfig()
    
    def test_default_device(self):
        """Test default device."""
        config = TRTEngineConfig(engine_path="/path/to/engine")
        
        self.assertEqual(config.device, "cuda")


class TestTRTEngineError(unittest.TestCase):
    """Test TRTEngineError exception."""
    
    def test_error_message(self):
        """Test error message."""
        error = TRTEngineError("Test error message")
        
        self.assertEqual(str(error), "Test error message")
    
    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        error = TRTEngineError("Test")
        
        self.assertIsInstance(error, Exception)


class TestTRTEngine(unittest.TestCase):
    """Test TRTEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TRTEngineConfig(
            engine_path=f"{self.temp_dir}/test_engine",
            build_config=TRTBuildConfig()
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', False)
    def test_init_without_trtllm(self):
        """Test initialization without TensorRT-LLM."""
        with self.assertRaises(TRTEngineError) as context:
            TRTEngine(self.config)
        
        self.assertIn("TensorRT-LLM not available", str(context.exception))
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_init_with_engine_path(self, mock_runner):
        """Test initialization with engine path."""
        mock_runner.from_dir.return_value = MagicMock()
        
        # Create engine directory
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        mock_runner.from_dir.assert_called_once()
        self.assertIsNotNone(engine._runtime)
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.Builder')
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_init_with_model_path(self, mock_runner, mock_builder):
        """Test initialization with model path."""
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance
        mock_runner.return_value = MagicMock()
        
        config = TRTEngineConfig(
            model_path="/path/to/model",
            build_config=TRTBuildConfig()
        )
        
        with patch.object(Path, 'exists', return_value=False):
            engine = TRTEngine(config)
        
        self.assertIsNotNone(engine._runtime)
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    def test_init_no_paths(self):
        """Test initialization with no valid paths."""
        config = TRTEngineConfig(engine_path="/nonexistent/path")
        
        with self.assertRaises(TRTEngineError) as context:
            TRTEngine(config)
        
        self.assertIn("No engine or model path", str(context.exception))
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_generate_success(self, mock_runner):
        """Test successful generation."""
        mock_runtime = MagicMock()
        mock_runtime.generate.return_value = {
            'sequences': torch.tensor([[1, 2, 3, 4, 5]])
        }
        mock_runner.from_dir.return_value = mock_runtime
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        input_ids = torch.tensor([[1, 2, 3]])
        outputs = engine.generate(input_ids, max_new_tokens=2)
        
        self.assertIn('sequences', outputs)
        mock_runtime.generate.assert_called_once()
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_generate_runtime_not_loaded(self, mock_runner):
        """Test generation without loaded runtime."""
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        engine._runtime = None
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        with self.assertRaises(TRTEngineError) as context:
            engine.generate(input_ids)
        
        self.assertIn("Engine not loaded", str(context.exception))
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_forward_success(self, mock_runner):
        """Test successful forward pass."""
        mock_runtime = MagicMock()
        mock_runtime.return_value = {'logits': torch.randn(1, 10, 1000)}
        mock_runner.from_dir.return_value = mock_runtime
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        input_ids = torch.tensor([[1, 2, 3]])
        outputs = engine.forward(input_ids)
        
        self.assertIn('logits', outputs)
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_get_memory_usage(self, mock_runner):
        """Test memory usage retrieval."""
        mock_runner.from_dir.return_value = MagicMock()
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=1000000000):
                with patch('torch.cuda.memory_reserved', return_value=2000000000):
                    with patch('torch.cuda.max_memory_allocated', return_value=1500000000):
                        mem_usage = engine.get_memory_usage()
        
        self.assertIn('allocated_gb', mem_usage)
        self.assertIn('reserved_gb', mem_usage)
        self.assertIn('max_allocated_gb', mem_usage)
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_get_memory_usage_no_cuda(self, mock_runner):
        """Test memory usage when CUDA not available."""
        mock_runner.from_dir.return_value = MagicMock()
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        with patch('torch.cuda.is_available', return_value=False):
            mem_usage = engine.get_memory_usage()
        
        self.assertEqual(mem_usage, {})
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_get_stats(self, mock_runner):
        """Test getting engine statistics."""
        mock_runner.from_dir.return_value = MagicMock()
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        engine._stats['inference_count'] = 10
        engine._stats['total_tokens_generated'] = 100
        
        stats = engine.get_stats()
        
        self.assertEqual(stats['inference_count'], 10)
        self.assertEqual(stats['total_tokens_generated'], 100)
        self.assertEqual(stats['quantization'], 'fp16')
    
    @patch('nexus.models.tensorrt.trt_engine.TRTLLM_AVAILABLE', True)
    @patch('nexus.models.tensorrt.trt_engine.ModelRunner')
    def test_save_engine(self, mock_runner):
        """Test saving engine."""
        mock_runtime = MagicMock()
        mock_runner.from_dir.return_value = mock_runtime
        
        engine_dir = Path(self.temp_dir) / "test_engine"
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        engine = TRTEngine(self.config)
        
        output_path = f"{self.temp_dir}/saved_engine"
        engine.save_engine(output_path)
        
        mock_runtime.save.assert_called_once()


class TestTensorRTConfig(unittest.TestCase):
    """Test TensorRTConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = TensorRTConfig(model_path="meta-llama/Llama-2-7b")
        
        self.assertEqual(config.model_path, "meta-llama/Llama-2-7b")
        self.assertIsNone(config.engine_path)
        self.assertIsNone(config.tokenizer_path)
        self.assertEqual(config.quantization_mode, "fp16")
        self.assertEqual(config.max_batch_size, 1)
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.device, "cuda")
        self.assertFalse(config.enable_streaming)
    
    def test_to_engine_config(self):
        """Test conversion to TRTEngineConfig."""
        config = TensorRTConfig(
            model_path="test_model",
            quantization_mode="fp8",
            max_batch_size=4,
            max_seq_length=4096
        )
        
        engine_config = config.to_engine_config()
        
        self.assertEqual(engine_config.model_path, "test_model")
        self.assertEqual(engine_config.build_config.max_batch_size, 4)
        self.assertEqual(engine_config.build_config.max_seq_length, 4096)
        self.assertEqual(engine_config.build_config.quantization, TRTQuantizationMode.FP8)
    
    def test_quantization_mapping(self):
        """Test quantization mode mapping."""
        test_cases = [
            ("fp32", TRTQuantizationMode.FP32),
            ("fp16", TRTQuantizationMode.FP16),
            ("bf16", TRTQuantizationMode.BF16),
            ("fp8", TRTQuantizationMode.FP8),
            ("int8", TRTQuantizationMode.INT8),
            ("int4", TRTQuantizationMode.INT4),
        ]
        
        for mode_str, expected_mode in test_cases:
            config = TensorRTConfig(model_path="test", quantization_mode=mode_str)
            engine_config = config.to_engine_config()
            self.assertEqual(engine_config.build_config.quantization, expected_mode)


class TestGenerationResult(unittest.TestCase):
    """Test GenerationResult dataclass."""
    
    def test_default_values(self):
        """Test default result values."""
        sequences = torch.tensor([[1, 2, 3]])
        result = GenerationResult(sequences=sequences)
        
        self.assertTrue(torch.equal(result.sequences, sequences))
        self.assertIsNone(result.scores)
        self.assertIsNone(result.logits)
        self.assertEqual(result.tokens_generated, 0)
        self.assertEqual(result.generation_time_ms, 0.0)
        self.assertEqual(result.tokens_per_second, 0.0)
    
    def test_custom_values(self):
        """Test custom result values."""
        sequences = torch.tensor([[1, 2, 3, 4, 5]])
        result = GenerationResult(
            sequences=sequences,
            tokens_generated=2,
            generation_time_ms=100.0,
            tokens_per_second=20.0
        )
        
        self.assertEqual(result.tokens_generated, 2)
        self.assertEqual(result.generation_time_ms, 100.0)
        self.assertEqual(result.tokens_per_second, 20.0)


class TestTensorRTBackend(unittest.TestCase):
    """Test TensorRTBackend class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TensorRTConfig(
            model_path="test_model",
            quantization_mode="fp16"
        )
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_initialization(self, mock_engine, mock_tokenizer):
        """Test backend initialization."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_engine.return_value = MagicMock()
        
        backend = TensorRTBackend(self.config)
        
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_engine.assert_called_once()
        self.assertIsNotNone(backend.tokenizer)
        self.assertIsNotNone(backend.engine)
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    def test_initialization_tokenizer_failure(self, mock_tokenizer):
        """Test initialization with tokenizer failure."""
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")
        
        with self.assertRaises(BackendError) as context:
            TensorRTBackend(self.config)
        
        self.assertIn("Failed to load tokenizer", str(context.exception))
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_generate_single_prompt(self, mock_engine, mock_tokenizer):
        """Test generation with single prompt."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = {
            'sequences': torch.tensor([[1, 2, 3, 4, 5]])
        }
        mock_engine.return_value = mock_engine_instance
        
        backend = TensorRTBackend(self.config)
        result = backend.generate("Hello", max_new_tokens=2)
        
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.tokens_generated, 2)
        mock_engine_instance.generate.assert_called_once()
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_generate_batch_prompts(self, mock_engine, mock_tokenizer):
        """Test generation with batch prompts."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = {
            'sequences': torch.tensor([[1, 2, 5], [3, 4, 6]])
        }
        mock_engine.return_value = mock_engine_instance
        
        backend = TensorRTBackend(self.config)
        result = backend.generate(["Hello", "World"], max_new_tokens=1)
        
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.tokens_generated, 2)  # 1 token x 2 prompts
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_generate_streaming_disabled(self, mock_engine, mock_tokenizer):
        """Test streaming when disabled."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.decode.return_value = "Hello World"
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = {
            'sequences': torch.tensor([[1, 2, 3, 4, 5]])
        }
        mock_engine.return_value = mock_engine_instance
        
        backend = TensorRTBackend(self.config)
        tokens = list(backend.generate_stream("Hello", max_new_tokens=3))
        
        # Should yield single string when streaming disabled
        self.assertEqual(len(tokens), 1)
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_batch_generate(self, mock_engine, mock_tokenizer):
        """Test batch generation."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.eos_token_id = 2
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.generate.return_value = {
            'sequences': torch.tensor([[1, 2, 5], [3, 4, 6]])
        }
        mock_engine.return_value = mock_engine_instance
        
        backend = TensorRTBackend(self.config)
        results = backend.batch_generate(["Hello", "World"], max_new_tokens=1)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], GenerationResult)
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_encode(self, mock_engine, mock_tokenizer):
        """Test text encoding."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_engine.return_value = MagicMock()
        
        backend = TensorRTBackend(self.config)
        encoded = backend.encode("Hello World")
        
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.shape, (1, 3))
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_decode_single(self, mock_engine, mock_tokenizer):
        """Test decoding single sequence."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.decode.return_value = "Hello World"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_engine.return_value = MagicMock()
        
        backend = TensorRTBackend(self.config)
        decoded = backend.decode(torch.tensor([1, 2, 3]))
        
        self.assertEqual(decoded, "Hello World")
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_decode_batch(self, mock_engine, mock_tokenizer):
        """Test decoding batch sequences."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer_instance.decode.side_effect = ["Hello", "World"]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_engine.return_value = MagicMock()
        
        backend = TensorRTBackend(self.config)
        decoded = backend.decode(torch.tensor([[1, 2], [3, 4]]))
        
        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0], "Hello")
        self.assertEqual(decoded[1], "World")
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_get_stats(self, mock_engine, mock_tokenizer):
        """Test getting backend statistics."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.get_stats.return_value = {
            'inference_count': 10,
            'total_tokens_generated': 100
        }
        mock_engine.return_value = mock_engine_instance
        
        backend = TensorRTBackend(self.config)
        backend._stats['total_requests'] = 5
        
        stats = backend.get_stats()
        
        self.assertEqual(stats['total_requests'], 5)
        self.assertEqual(stats['engine_stats']['inference_count'], 10)
    
    @patch('nexus.models.tensorrt.inference_backend.AutoTokenizer')
    @patch('nexus.models.tensorrt.inference_backend.TRTEngine')
    def test_reset_stats(self, mock_engine, mock_tokenizer):
        """Test resetting statistics."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_engine.return_value = MagicMock()
        
        backend = TensorRTBackend(self.config)
        backend._stats['total_requests'] = 100
        
        backend.reset_stats()
        
        self.assertEqual(backend._stats['total_requests'], 0)


if __name__ == '__main__':
    unittest.main()
