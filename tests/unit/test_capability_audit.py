import pytest
import torch
from unittest.mock import MagicMock, patch
from src.capability_audit import audit_capabilities

@patch("src.capability_audit.AutoTokenizer.from_pretrained")
@patch("src.capability_audit.OmniMultimodalLM")
@patch("src.capability_audit.ReasoningWrapper")
def test_audit_capabilities_success(mock_reasoner_cls, mock_omni_cls, mock_tokenizer_fn):
    # Mock model and tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer_fn.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_omni_cls.return_value = mock_model
    
    # Mock reasoner
    mock_reasoner = MagicMock()
    mock_reasoner_cls.return_value = mock_reasoner
    mock_reasoner.generate.return_value = "<think>3+2=5</think> You have 4 apples."
    
    # Mock tool use generation
    mock_tokenizer.apply_chat_template.return_value = "chat text"
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.input_ids.shape = [1, 5]
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.decode.return_value = '{"name": "calculator", "arguments": {"expression": "12345*67890"}}'
    
    mock_model.wrapper.llm.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
    mock_model.wrapper.llm.device = "cpu"
    
    # Run audit
    audit_capabilities()
    
    assert mock_reasoner.generate.called
    assert mock_model.wrapper.llm.generate.called

@patch("src.capability_audit.AutoTokenizer.from_pretrained")
def test_audit_capabilities_load_failure(mock_tokenizer_fn):
    mock_tokenizer_fn.side_effect = Exception("Load failed")
    # Should not raise exception, just print error
    audit_capabilities()
