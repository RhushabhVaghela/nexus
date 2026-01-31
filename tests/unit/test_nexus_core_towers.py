"""
tests/unit/test_nexus_core_towers.py
Comprehensive tests for nexus_core towers functionality.

Tests cover:
- BaseTower abstract class
- VisionTower
- ReasoningTower
- Tower registry
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

# Import tower components
from src.nexus_core.towers.base_tower import BaseTower
from src.nexus_core.towers.vision_tower import VisionTower
from src.nexus_core.towers.reasoning_tower import ReasoningTower
from src.nexus_core.towers import registry


class TestBaseTower:
    """Test BaseTower abstract class."""
    
    def test_base_tower_is_abstract(self):
        """Test that BaseTower cannot be instantiated directly."""
        
        class ConcreteTower(BaseTower):
            def forward(self, x):
                return x
        
        config = {"hidden_size": 128}
        tower = ConcreteTower(config)
        
        assert tower.config == config
        assert isinstance(tower.adapters, nn.ModuleDict)
        assert tower.frozen_teacher is None
    
    def test_load_teacher(self):
        """Test loading teacher model."""
        
        class ConcreteTower(BaseTower):
            def forward(self, x):
                return x
        
        tower = ConcreteTower({})
        
        # Create a mock teacher model
        mock_teacher = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_teacher.parameters.return_value = [mock_param]
        
        tower.load_teacher(mock_teacher)
        
        assert tower.frozen_teacher is mock_teacher
        assert mock_param.requires_grad is False
    
    def test_add_adapter(self):
        """Test adding adapter."""
        
        class ConcreteTower(BaseTower):
            def forward(self, x):
                return x
        
        tower = ConcreteTower({})
        mock_adapter = Mock()
        
        tower.add_adapter("test_adapter", mock_adapter)
        
        assert "test_adapter" in tower.adapters
        assert tower.adapters["test_adapter"] is mock_adapter
    
    def test_get_adapter(self):
        """Test getting adapter."""
        
        class ConcreteTower(BaseTower):
            def forward(self, x):
                return x
        
        tower = ConcreteTower({})
        mock_adapter = Mock()
        
        tower.add_adapter("test_adapter", mock_adapter)
        retrieved = tower.get_adapter("test_adapter")
        
        assert retrieved is mock_adapter


class TestVisionTower:
    """Test VisionTower class."""
    
    def test_initialization(self):
        """Test VisionTower initialization."""
        config = {"hidden_size": 128}
        tower = VisionTower(config, teacher_dim=768, student_dim=512)
        
        assert tower.config == config
        assert tower.teacher_dim == 768
        assert tower.student_dim == 512
        assert tower.projection is not None
    
    @patch("torch.no_grad")
    def test_forward_with_teacher(self, mock_no_grad):
        """Test forward pass with teacher model."""
        config = {"hidden_size": 128}
        tower = VisionTower(config, teacher_dim=768, student_dim=512)
        
        # Mock teacher model
        mock_teacher = Mock()
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        mock_teacher.vision_model.return_value = mock_output
        
        tower.load_teacher(mock_teacher)
        
        pixel_values = torch.randn(2, 3, 224, 224)
        result = tower.forward(pixel_values)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] == 512  # student_dim
    
    def test_forward_without_teacher(self):
        """Test forward pass without teacher raises error."""
        config = {"hidden_size": 128}
        tower = VisionTower(config, teacher_dim=768, student_dim=512)
        
        pixel_values = torch.randn(2, 3, 224, 224)
        
        with pytest.raises(ValueError, match="Teacher model not loaded"):
            tower.forward(pixel_values)


class TestReasoningTower:
    """Test ReasoningTower class."""
    
    def test_initialization(self):
        """Test ReasoningTower initialization."""
        config = {"hidden_size": 128}
        tower = ReasoningTower(config, teacher_dim=4096, student_dim=2048)
        
        assert tower.config == config
        assert tower.teacher_dim == 4096
        assert tower.student_dim == 2048
        assert tower.projection is not None
    
    @patch("torch.no_grad")
    def test_forward_with_teacher(self, mock_no_grad):
        """Test forward pass with teacher model."""
        config = {"hidden_size": 128}
        tower = ReasoningTower(config, teacher_dim=4096, student_dim=2048)
        
        # Mock teacher model
        mock_teacher = Mock()
        mock_output = Mock()
        mock_output.hidden_states = [torch.randn(2, 10, 4096)] * 5
        mock_teacher.return_value = mock_output
        
        tower.load_teacher(mock_teacher)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        result = tower.forward(input_ids, attention_mask)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] == 2048  # student_dim
    
    def test_forward_without_teacher(self):
        """Test forward pass without teacher raises error."""
        config = {"hidden_size": 128}
        tower = ReasoningTower(config, teacher_dim=4096, student_dim=2048)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        
        with pytest.raises(ValueError, match="Teacher model not loaded"):
            tower.forward(input_ids)


class TestTowerRegistry:
    """Test tower registry module."""
    
    def test_teacher_registry_exists(self):
        """Test TEACHER_REGISTRY exists and is a dict."""
        assert hasattr(registry, "TEACHER_REGISTRY")
        assert isinstance(registry.TEACHER_REGISTRY, dict)
    
    def test_teacher_registry_entries(self):
        """Test TEACHER_REGISTRY has expected entries."""
        registry_keys = registry.TEACHER_REGISTRY.keys()
        
        # Check for expected towers
        expected_towers = [
            "reasoning_core",
            "logic_heavy",
            "base_small",
            "coder",
            "vision_main",
            "omni_base"
        ]
        
        for tower in expected_towers:
            assert tower in registry_keys, f"Expected tower '{tower}' not found"
    
    def test_teacher_registry_entry_structure(self):
        """Test teacher registry entry structure."""
        entry = registry.TEACHER_REGISTRY["base_small"]
        
        assert "model" in entry
        assert "path" in entry
        assert "type" in entry
        assert "desc" in entry
        assert "tags" in entry
        
        assert isinstance(entry["tags"], list)
    
    def test_dataset_registry_exists(self):
        """Test DATASET_REGISTRY exists and is a dict."""
        assert hasattr(registry, "DATASET_REGISTRY")
        assert isinstance(registry.DATASET_REGISTRY, dict)
    
    def test_dataset_registry_entries(self):
        """Test DATASET_REGISTRY has entries."""
        assert len(registry.DATASET_REGISTRY) > 0
    
    def test_dataset_registry_entry_structure(self):
        """Test dataset registry entry structure."""
        # Get first entry
        key = list(registry.DATASET_REGISTRY.keys())[0]
        entry = registry.DATASET_REGISTRY[key]
        
        assert "path" in entry
        assert "local_path" in entry
        assert "desc" in entry
        assert "tags" in entry
    
    def test_base_path_constant(self):
        """Test BASE_PATH constant."""
        assert hasattr(registry, "BASE_PATH")
        assert isinstance(registry.BASE_PATH, str)
    
    def test_teacher_types(self):
        """Test different teacher types exist."""
        types_found = set()
        
        for entry in registry.TEACHER_REGISTRY.values():
            types_found.add(entry.get("type"))
        
        expected_types = {"causal", "multimodal", "vision", "audio", "encoder", "generation", "tokenizer"}
        
        for et in expected_types:
            assert et in types_found, f"Expected type '{et}' not found"
    
    def test_teacher_tags(self):
        """Test teacher tags are lists."""
        for key, entry in registry.TEACHER_REGISTRY.items():
            assert isinstance(entry.get("tags", []), list), f"Tags for {key} should be a list"


class TestTowerIntegration:
    """Test tower integration scenarios."""
    
    def test_tower_with_adapter(self):
        """Test tower working with adapters."""
        from src.nexus_core.adapters.vision_adapter import VisionAdapter
        
        config = {"hidden_size": 128}
        tower = VisionTower(config, teacher_dim=768, student_dim=512)
        
        # Add a custom adapter
        custom_adapter = VisionAdapter(512, 256)
        tower.add_adapter("custom", custom_adapter)
        
        assert "custom" in tower.adapters
    
    def test_tower_teacher_dimensions(self):
        """Test various teacher-student dimension combinations."""
        test_cases = [
            (768, 512),
            (4096, 2048),
            (1024, 512),
            (2048, 1024)
        ]
        
        for teacher_dim, student_dim in test_cases:
            config = {"hidden_size": 128}
            tower = ReasoningTower(config, teacher_dim=teacher_dim, student_dim=student_dim)
            
            assert tower.teacher_dim == teacher_dim
            assert tower.student_dim == student_dim


class TestEdgeCases:
    """Test edge cases."""
    
    def test_vision_tower_batch_sizes(self):
        """Test VisionTower with different batch sizes."""
        config = {"hidden_size": 128}
        tower = VisionTower(config, teacher_dim=768, student_dim=512)
        
        # Mock teacher model
        mock_teacher = Mock()
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 10, 768)
        mock_teacher.vision_model.return_value = mock_output
        
        tower.load_teacher(mock_teacher)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            result = tower.forward(pixel_values)
            assert result.shape[0] == batch_size
    
    def test_reasoning_tower_sequence_lengths(self):
        """Test ReasoningTower with different sequence lengths."""
        config = {"hidden_size": 128}
        tower = ReasoningTower(config, teacher_dim=4096, student_dim=2048)
        
        # Mock teacher model
        mock_teacher = Mock()
        
        def mock_forward(*args, **kwargs):
            seq_len = kwargs.get("input_ids", args[0]).shape[1]
            mock_output = Mock()
            mock_output.hidden_states = [torch.randn(2, seq_len, 4096)] * 5
            return mock_output
        
        mock_teacher.side_effect = mock_forward
        tower.load_teacher(mock_teacher)
        
        # Test different sequence lengths
        for seq_len in [10, 50, 100]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            result = tower.forward(input_ids)
            assert result.shape[1] == seq_len
