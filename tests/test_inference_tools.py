import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add scripts and src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "scripts")
SRC_DIR = os.path.join(os.path.dirname(BASE_DIR), "src")

sys.path.append(SCRIPTS_DIR)
sys.path.append(SRC_DIR)

# Import locally to ensure they are registered in sys.modules
try:
    import inference
    import benchmark_suite
except ImportError:
    pass

class TestInferenceTools:
    
    @patch("inference.AutoModelForCausalLM")
    @patch("inference.AutoTokenizer")
    def test_inference_loading(self, mock_tok, mock_model):
        """Verify inference.py model loading logic."""
        import inference
        
        # Mock HF objects
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tok.from_pretrained.return_value = MagicMock()
        
        # Test 1: Load Student
        model, tokenizer = inference.load_student("dummy_path")
        assert model is not None
        mock_model.from_pretrained.assert_called()
        
    @patch("inference.load_student")
    @patch("builtins.input", side_effect=["Hello", "exit"])
    @patch("sys.argv", ["inference.py"])
    def test_inference_interactive_loop(self, mock_input, mock_load):
        """Verify the interactive loop logic."""
        import inference
        
        # Mock dependencies
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "World"
        mock_load.return_value = (mock_model, mock_tok)
        
        # Run main (should exit after one loop)
        inference.main()
        
        # Check generation called
        assert mock_model.generate.called

class TestBenchmarkSuite:
    
    @patch("benchmark_suite.AutoModelForCausalLM")
    @patch("benchmark_suite.AutoTokenizer")
    def test_benchmark_evaluation(self, mock_tok, mock_model):
        """Verify benchmark comparison logic."""
        import benchmark_suite
        
        # Setup Mocks
        mock_hf_model = MagicMock()
        mock_model.from_pretrained.return_value = mock_hf_model
        
        mock_hf_tok = MagicMock()
        mock_hf_tok.decode.return_value = "The answer is 100" # Contains target "100"
        mock_tok.from_pretrained.return_value = mock_hf_tok
        
        questions = ["What is 25 + 75?"]
        targets = ["100"]
        
        # Run Evaluation
        score, results = benchmark_suite.evaluate(mock_hf_model, mock_hf_tok, questions, targets, device="cpu")
        
        # Assertions
        assert score == 1.0 # Should be correct
        assert results[0]['correct'] is True
        assert results[0]['response'] == "The answer is 100"
