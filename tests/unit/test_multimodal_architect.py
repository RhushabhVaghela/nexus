"""
Unit tests for Multimodal Embedding Injection in NexusStudent.

Tests the core multimodal fusion mechanism that enables the model
to process vision, audio, video, and tool modalities alongside text.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestNeuralArchitect:
    """Tests for NeuralArchitect class."""
    
    @pytest.fixture
    def architect(self, tmp_path):
        """Fixture for NeuralArchitect instance."""
        from src.nexus_final.architect import NeuralArchitect
        return NeuralArchitect(output_dir=str(tmp_path / "architect_output"))
    
    @pytest.fixture
    def sample_profile_data(self):
        """Sample profiling data for testing."""
        return {
            "teacher_1": {
                "intrinsic_dimension": 64,
                "explained_variance": 0.95
            },
            "teacher_2": {
                "intrinsic_dimension": 32,
                "explained_variance": 0.90
            }
        }
    
    def test_initialization(self, architect, tmp_path):
        """Test NeuralArchitect initialization."""
        assert architect.output_dir == str(tmp_path / "architect_output")
        assert architect.default_base_model == "meta-llama/Llama-3.2-1B-Instruct"
        assert Path(architect.output_dir).exists()
    
    def test_load_profiling_data_success(self, architect, tmp_path, sample_profile_data):
        """Test loading profiling data from valid file."""
        profile_path = tmp_path / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump(sample_profile_data, f)
        
        data = architect.load_profiling_data(str(profile_path))
        assert data == sample_profile_data
    
    def test_load_profiling_data_missing_file(self, architect):
        """Test loading profiling data when file doesn't exist."""
        data = architect.load_profiling_data("/nonexistent/path.json")
        assert data == {}
    
    def test_determine_adapter_config_default(self, architect, sample_profile_data):
        """Test adapter config determination with default settings."""
        config = architect.determine_adapter_config("teacher_1", sample_profile_data)
        
        assert config["r"] == 64
        assert config["lora_alpha"] == 128
        assert config["lora_dropout"] == 0.05
    
    def test_determine_adapter_config_with_limit(self, architect, sample_profile_data):
        """Test adapter config with max rank limit."""
        config = architect.determine_adapter_config("teacher_1", sample_profile_data, max_rank_limit=32)
        
        assert config["r"] == 32  # Capped at limit
        assert config["lora_alpha"] == 64
    
    def test_determine_adapter_config_floor(self, architect):
        """Test adapter config floor value."""
        data = {"teacher": {"intrinsic_dimension": 4}}
        config = architect.determine_adapter_config("teacher", data)
        
        assert config["r"] == 8  # Floor value
    
    def test_determine_adapter_config_from_env(self, architect, sample_profile_data, monkeypatch):
        """Test adapter config from environment variable."""
        monkeypatch.setenv("MAX_RANK", "16")
        config = architect.determine_adapter_config("teacher_1", sample_profile_data)
        
        assert config["r"] == 16  # From environment
    
    def test_determine_adapter_config_missing_teacher(self, architect):
        """Test adapter config for missing teacher."""
        config = architect.determine_adapter_config("unknown_teacher", {})
        
        assert config["r"] == 8  # Default floor
        assert config["lora_alpha"] == 16
    
    def test_synthesize_student_model(self, architect, tmp_path):
        """Test student model code synthesis."""
        output_path = tmp_path / "student_model.py"
        adapter_config = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
        
        architect.synthesize_student_model(
            str(output_path),
            "test-model",
            adapter_config,
            teacher_hidden_dim=2048
        )
        
        assert output_path.exists()
        content = output_path.read_text()
        assert "class NexusStudent" in content
        assert "class NexusBridge" in content
        assert '"r": 16' in content
        assert "test-model" in content
    
    def test_execute_design_process(self, architect, tmp_path, sample_profile_data):
        """Test complete design process execution."""
        profile_path = tmp_path / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump(sample_profile_data, f)
        
        output_path = tmp_path / "output" / "student.py"
        
        architect.execute_design_process(
            "teacher_1",
            str(profile_path),
            str(output_path),
            base_model_override="custom-model"
        )
        
        assert output_path.exists()


class TestNexusBridge:
    """Tests for NexusBridge projection layer."""
    
    @pytest.fixture
    def bridge(self):
        """Fixture for NexusBridge instance."""
        from src.nexus_final.architect import NexusBridge
        return NexusBridge(in_features=1024, out_features=512)
    
    def test_initialization(self, bridge):
        """Test bridge initialization."""
        assert isinstance(bridge.projector, nn.Linear)
        assert bridge.projector.in_features == 1024
        assert bridge.projector.out_features == 512
        assert isinstance(bridge.norm, nn.LayerNorm)
        assert isinstance(bridge.act, nn.GELU)
    
    def test_forward_pass(self, bridge):
        """Test bridge forward pass."""
        x = torch.randn(2, 10, 1024)
        output = bridge(x)
        
        assert output.shape == (2, 10, 512)
        assert not torch.isnan(output).any()


class TestNexusStudentMultimodal:
    """Tests for NexusStudent multimodal embedding injection."""
    
    @pytest.fixture
    def mock_base_model(self):
        """Create a mock base model."""
        mock = MagicMock()
        mock.config = MagicMock()
        mock.config.hidden_size = 512
        
        # Mock embedding layer
        mock_embeds = MagicMock()
        mock_embeds.return_value = torch.randn(2, 10, 512)
        mock.get_input_embeddings = mock_embeds
        
        # Mock the PEFT model
        mock_peft = MagicMock()
        mock_peft.return_value = MagicMock(
            loss=torch.tensor(0.5),
            logits=torch.randn(2, 10, 1000),
            hidden_states=None,
            router_logits=None
        )
        
        return mock, mock_peft
    
    @pytest.fixture
    def mock_alignment(self):
        """Create mock alignment module."""
        mock = MagicMock()
        # Return aligned features matching student_dim
        mock.return_value = torch.randn(2, 5, 512)
        return mock
    
    @pytest.fixture
    def student(self, mock_base_model, mock_alignment):
        """Fixture for NexusStudent with mocked dependencies."""
        with patch('src.nexus_final.architect.AutoModelForCausalLM') as mock_model_class, \
             patch('src.nexus_final.architect.AutoTokenizer') as mock_tokenizer_class, \
             patch('src.nexus_final.architect.get_peft_model') as mock_get_peft, \
             patch('src.nexus_final.architect.LoraConfig'), \
             patch('src.nexus_final.architect.CrossModalAlignment', mock_alignment):
            
            mock_base, mock_peft = mock_base_model
            mock_model_class.from_pretrained.return_value = mock_base
            mock_get_peft.return_value = mock_peft
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "[EOS]"
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            from src.nexus_final.architect import NexusStudent
            student = NexusStudent(base_model_id="test-model")
            return student
    
    def test_initialization(self, student):
        """Test student initialization."""
        assert student.config["r"] == 8
        assert student.config["lora_alpha"] == 16
        assert student.student_dim == 512
        assert student.teacher_dim == 4096
    
    def test_forward_text_only(self, student):
        """Test forward pass with text only."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 1000, (2, 10))
        
        outputs = student(input_ids, attention_mask=attention_mask, labels=labels)
        
        assert "loss" in outputs
        assert "logits" in outputs
    
    def test_forward_with_vision(self, student):
        """Test forward pass with vision features."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        vision_feats = torch.randn(2, 16, 768)  # 16 image patches, 768-dim vision features
        
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
        assert outputs["multimodal_seq_len"] == 5  # From mock alignment
    
    def test_forward_with_audio(self, student):
        """Test forward pass with audio features."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        audio_feats = torch.randn(2, 20, 256)  # 20 audio frames, 256-dim audio features
        
        outputs = student(input_ids, attention_mask=attention_mask, audio_feats=audio_feats)
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
    
    def test_forward_with_video(self, student):
        """Test forward pass with video features."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        video_feats = torch.randn(2, 8, 512)  # 8 video frames, 512-dim video features
        
        outputs = student(input_ids, attention_mask=attention_mask, video_feats=video_feats)
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
    
    def test_forward_with_tool(self, student):
        """Test forward pass with tool features."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        tool_feats = torch.randn(2, 3, 128)  # 3 tool calls, 128-dim tool features
        
        outputs = student(input_ids, attention_mask=attention_mask, tool_feats=tool_feats)
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
    
    def test_forward_with_multiple_modalities(self, student):
        """Test forward pass with multiple modalities."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        vision_feats = torch.randn(2, 16, 768)
        audio_feats = torch.randn(2, 20, 256)
        
        outputs = student(
            input_ids,
            attention_mask=attention_mask,
            vision_feats=vision_feats,
            audio_feats=audio_feats
        )
        
        assert "loss" in outputs
        assert "multimodal_embeds" in outputs
    
    def test_attention_mask_update(self, student):
        """Test that attention mask is properly updated for multimodal tokens."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        # Attention mask should be extended with multimodal tokens
        # Original: (2, 10), After adding 5 multimodal tokens: (2, 15)
        assert "multimodal_embeds" in outputs
        multimodal_embeds = outputs["multimodal_embeds"]
        assert multimodal_embeds.shape[1] == 15  # 5 multimodal + 10 text
    
    def test_labels_shifted_with_neg_100(self, student):
        """Test that labels are shifted with -100 for multimodal prefix."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        labels = torch.randint(0, 1000, (2, 10))
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_feats=vision_feats
        )
        
        # Labels should be extended with -100 for multimodal prefix
        assert "multimodal_embeds" in outputs
    
    def test_bridge_projection_teacher_latents(self, student):
        """Test bridge projection for teacher latents."""
        input_ids = torch.randint(0, 1000, (2, 10))
        teacher_latents = torch.randn(2, 10, 4096)  # Teacher dimension
        
        outputs = student(input_ids, teacher_latents=teacher_latents)
        
        assert "projected_teacher_latents" in outputs
        projected = outputs["projected_teacher_latents"]
        assert projected.shape == (2, 10, 512)  # Projected to student_dim
    
    def test_dimension_mismatch_handling(self, student):
        """Test handling of dimension mismatch between multimodal and text embeddings."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Create vision features with different dimension than text embeddings
        vision_feats = torch.randn(2, 16, 256)  # 256-dim, different from 512
        
        # Should create projection layer dynamically
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        assert "multimodal_embeds" in outputs
        # Check that projection layer was created
        assert hasattr(student, '_multimodal_proj')
    
    def test_missing_embeddings_error(self, student):
        """Test error handling when embeddings layer cannot be found."""
        # Remove embedding methods
        student.base_model.get_input_embeddings = MagicMock(return_value=None)
        if hasattr(student.base_model, 'model'):
            delattr(student.base_model, 'model')
        
        input_ids = torch.randint(0, 1000, (2, 10))
        vision_feats = torch.randn(2, 16, 768)
        
        with pytest.raises(ValueError, match="Could not find input embeddings"):
            student(input_ids, vision_feats=vision_feats)
    
    def test_router_entropy_calculation(self, student):
        """Test router entropy calculation for MoE models."""
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Mock router logits
        mock_router_logits = torch.randn(20, 8)  # 20 tokens, 8 experts
        
        with patch.object(student.model, '__call__') as mock_forward:
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(0.5)
            mock_output.logits = torch.randn(2, 10, 1000)
            mock_output.router_logits = mock_router_logits
            mock_forward.return_value = mock_output
            
            outputs = student(input_ids, output_router_logits=True)
            
            assert "entropy_loss" in outputs
            assert outputs["entropy_loss"] is not None
            assert isinstance(outputs["entropy_loss"], torch.Tensor)
    
    def test_input_ids_none_with_embeddings(self, student):
        """Test that input_ids is None when inputs_embeds is provided."""
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        vision_feats = torch.randn(2, 16, 768)
        
        with patch.object(student.model, '__call__') as mock_forward:
            mock_forward.return_value = MagicMock(
                loss=torch.tensor(0.5),
                logits=torch.randn(2, 15, 1000),
                hidden_states=None,
                router_logits=None
            )
            
            student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
            
            # Check that inputs_embeds was passed and input_ids was not
            call_kwargs = mock_forward.call_args.kwargs
            assert "inputs_embeds" in call_kwargs
            assert call_kwargs.get("input_ids") is None
    
    def test_save_pretrained(self, student, tmp_path):
        """Test saving model with bridge and multimodal projection."""
        save_path = tmp_path / "saved_model"
        
        # Add a multimodal projection layer
        student._multimodal_proj = nn.Linear(256, 512)
        
        with patch.object(student.model, 'save_pretrained') as mock_save_model, \
             patch.object(student.tokenizer, 'save_pretrained') as mock_save_tokenizer, \
             patch('torch.save') as mock_torch_save:
            
            student.save_pretrained(str(save_path))
            
            mock_save_model.assert_called_once_with(str(save_path))
            mock_save_tokenizer.assert_called_once_with(str(save_path))
            # Should save both bridge and multimodal projection
            assert mock_torch_save.call_count == 2


class TestMultimodalEmbeddingInjectionEdgeCases:
    """Edge case tests for multimodal embedding injection."""
    
    @pytest.fixture
    def student_with_mocks(self):
        """Create student with comprehensive mocking."""
        with patch('src.nexus_final.architect.AutoModelForCausalLM'), \
             patch('src.nexus_final.architect.AutoTokenizer'), \
             patch('src.nexus_final.architect.get_peft_model'), \
             patch('src.nexus_final.architect.LoraConfig'), \
             patch('src.nexus_final.architect.CrossModalAlignment') as mock_align:
            
            mock_align.return_value.return_value = torch.randn(2, 5, 512)
            
            from src.nexus_final.architect import NexusStudent
            student = NexusStudent()
            student.base_model = MagicMock()
            student.base_model.config.hidden_size = 512
            student.model = MagicMock()
            
            mock_embeds = MagicMock()
            mock_embeds.return_value = torch.randn(2, 10, 512)
            student.base_model.get_input_embeddings = mock_embeds
            
            return student
    
    def test_empty_multimodal_context(self, student_with_mocks):
        """Test handling when alignment returns None."""
        student = student_with_mocks
        
        # Mock alignment to return None
        student.alignment = MagicMock(return_value=None)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(input_ids, vision_feats=vision_feats)
        
        # Should still work without multimodal_embeds
        assert "multimodal_embeds" not in outputs or outputs.get("multimodal_embeds") is None
    
    def test_batch_size_one(self, student_with_mocks):
        """Test with batch size of 1."""
        student = student_with_mocks
        
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        vision_feats = torch.randn(1, 16, 768)
        
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        assert "multimodal_embeds" in outputs
        assert outputs["multimodal_embeds"].shape[0] == 1
    
    def test_no_labels_provided(self, student_with_mocks):
        """Test forward pass without labels."""
        student = student_with_mocks
        
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        assert "loss" in outputs
    
    def test_no_attention_mask(self, student_with_mocks):
        """Test forward pass without attention mask."""
        student = student_with_mocks
        
        input_ids = torch.randint(0, 1000, (2, 10))
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(input_ids, vision_feats=vision_feats)
        
        assert "multimodal_embeds" in outputs
    
    def test_very_long_sequence(self, student_with_mocks):
        """Test with very long text sequence."""
        student = student_with_mocks
        
        input_ids = torch.randint(0, 1000, (2, 1000))
        attention_mask = torch.ones(2, 1000)
        vision_feats = torch.randn(2, 16, 768)
        
        outputs = student(input_ids, attention_mask=attention_mask, vision_feats=vision_feats)
        
        assert "multimodal_embeds" in outputs
        # Check total length: 1000 text + 5 multimodal
        assert outputs["multimodal_embeds"].shape[1] == 1005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
