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
        assert projected.shape == (1, 2, 128)
        assert isinstance(projected, torch.Tensor)

def test_init_custom_model():
    tower = KnowledgeTower(student_dim=128, embedding_model="custom/model")
    assert tower.embedding_model_id == "custom/model"

def test_text_retrieval():
    # Mock Tokenizer and Model for retrieval
    with patch("src.nexus_final.knowledge.AutoTokenizer.from_pretrained") as mock_tok_cls, \
         patch("src.nexus_final.knowledge.AutoModel.from_pretrained") as mock_model_cls:
        
        tower = KnowledgeTower(student_dim=128)
        # Mock lazy init components
        tower.tokenizer = MagicMock()
        tower.tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]), 
            "attention_mask": torch.tensor([[1, 1]])
        }
        
        tower.encoder = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 2, 384) # Batch 1, Seq 2, Dim 384
        tower.encoder.return_value = mock_output
        
        # Mock Index
        tower.index = MagicMock()
        # Assume 2 docs found: doc 0 and doc 2
        tower.index.search.return_value = (None, torch.tensor([[0, 2]]))
        tower.index.ntotal = 10
        
        tower.documents = ["Doc 0", "Doc 1", "Doc 2"]
        
        # Test retrieval
        results = tower.retrieve_text_context("query", top_k=2)
        
        assert len(results) == 2
        assert results[0] == "Doc 0"
        assert results[1] == "Doc 2"
        assert isinstance(results[0], str)

def test_file_memory_manager(tmp_path):
    mem_root = str(tmp_path / "memory")
    manager = FileMemoryManager(memory_root=mem_root)
    
    manager.store_observation("test_key", "test content")
    assert manager.load_observation("test_key") == "test content"
    assert "test_key.txt" in manager.list_memory_files()

def test_session_persistence(tmp_path):
    # Mocking memory root via KnowledgeTower
    with patch("src.nexus_final.knowledge.FileMemoryManager") as MockMem:
        mem_instance = MockMem.return_value
        tower = KnowledgeTower(student_dim=128)
        tower.memory_manager = mem_instance
        
        context = ["Doc A", "Doc B"]
        tower.save_desk("sess_1", context)
        
        mem_instance.save_context_session.assert_called_with("sess_1", context)
        
        tower.load_desk("sess_1")
        mem_instance.load_context_session.assert_called_with("sess_1")

def test_real_file_persistence(tmp_path):
    # Integration test for FileMemoryManager real file IO
    from src.nexus_final.knowledge import FileMemoryManager
    mem_root = str(tmp_path / "memory")
    manager = FileMemoryManager(memory_root=mem_root)
    
    context = ["Real Doc 1", "Real Doc 2"]
    manager.save_context_session("real_sess", context)
    
    loaded = manager.load_context_session("real_sess")
    assert loaded == context
    assert os.path.exists(os.path.join(mem_root, "sessions", "real_sess.json"))

def test_student_read_from_memory():
    from src.nexus_core.student.core import NexusStudentCore, NexusStudentConfig
    
    config = NexusStudentConfig(num_hidden_layers=1, hidden_size=128)
    student = NexusStudentCore(config)
    
    mock_tower = MagicMock()
    mock_tower.return_value = torch.randn(1, 3, 128)
    
    context = student.read_from_memory("tell me about nexus", mock_tower)
    
    assert context.shape == (1, 3, 128)
    mock_tower.assert_called_once()
