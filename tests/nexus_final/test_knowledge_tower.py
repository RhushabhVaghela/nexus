import pytest
import torch
import os
from unittest.mock import MagicMock, patch
from src.nexus_final.knowledge import KnowledgeTower, FileMemoryManager

def test_knowledge_tower_projection():
    # Mock Tokenizer and Model
    with patch("src.nexus_final.knowledge.AutoTokenizer") as mock_tokenizer, \
         patch("src.nexus_final.knowledge.AutoModel") as mock_model:
        
        # Setup mock outputs
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        tower = KnowledgeTower(student_dim=128, embedding_dim=384)
        
        # Mock the encoder behavior during forward
        tower.encoder = MagicMock()
        tower.tokenizer = MagicMock()
        
        # Helper to create mock output
        def mock_encode_and_pass(**kwargs):
            # The dict from tokenizer is passed as kwargs
            # We assume batch size is 1 for tower forward call or len of input_ids
            batch_size = 1
            if "input_ids" in kwargs:
                batch_size = kwargs["input_ids"].shape[0]
            mock_out = MagicMock()
            mock_out.last_hidden_state = torch.randn(batch_size, 5, 384)
            return mock_out

        tower.encoder.side_effect = mock_encode_and_pass
        tower.tokenizer.return_value = {"attention_mask": torch.ones(1, 5).long().to(tower.device)}
        
        # Build a fake index
        documents = ["Hello world", "Nexus is awesome"]
        tower.index = MagicMock()
        tower.index.search.return_value = (None, torch.tensor([[0, 1]]))
        tower.documents = documents
        
        # Test forward
        projected = tower("test query", top_k=2)
        
        assert projected.shape == (1, 2, 128)
        assert isinstance(projected, torch.Tensor)

def test_file_memory_manager(tmp_path):
    mem_root = str(tmp_path / "memory")
    manager = FileMemoryManager(memory_root=mem_root)
    
    manager.store_observation("test_key", "test content")
    assert manager.load_observation("test_key") == "test content"
    assert "test_key.txt" in manager.list_memory_files()

def test_student_read_from_memory():
    from src.nexus_core.student.core import NexusStudentCore, NexusStudentConfig
    
    config = NexusStudentConfig(num_hidden_layers=1, hidden_size=128)
    student = NexusStudentCore(config)
    
    mock_tower = MagicMock()
    mock_tower.return_value = torch.randn(1, 3, 128)
    
    context = student.read_from_memory("tell me about nexus", mock_tower)
    
    assert context.shape == (1, 3, 128)
    mock_tower.assert_called_once()
