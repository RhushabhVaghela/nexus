import pytest
import torch
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.omni.unify_checkpoints import merge_checkpoints

class TestUnifyCheckpoints:
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("safetensors.torch.load_file")
    def test_linear_merge(self, mock_load_file, mock_from_pretrained, tmp_path):
        # Setup base model
        mock_base = MagicMock()
        mock_base.state_dict.return_value = {
            "layer1.weight": torch.ones(2, 2)
        }
        mock_from_pretrained.return_value = mock_base
        
        # Setup checkpoint paths
        ckpt1 = tmp_path / "ckpt1"
        ckpt1.mkdir()
        (ckpt1 / "model.safetensors").touch()
        
        ckpt2 = tmp_path / "ckpt2"
        ckpt2.mkdir()
        (ckpt2 / "model.safetensors").touch()
        
        # Mock load_file for ckpt1 and ckpt2
        def side_effect(path):
            if "ckpt1" in path:
                return {"layer1.weight": torch.ones(2, 2) * 2}
            return {"layer1.weight": torch.ones(2, 2) * 4}
        mock_load_file.side_effect = side_effect
        
        output = tmp_path / "unified"
        
        # Weights 0.5 each: (2*0.5 + 4*0.5) = 3
        merge_checkpoints(
            base_model_path="base",
            checkpoint_paths=[str(ckpt1), str(ckpt2)],
            output_path=str(output),
            method="linear",
            weights=[0.5, 0.5]
        )
        
        # Verify load_state_dict called with expected average
        call_args = mock_base.load_state_dict.call_args[0][0]
        assert torch.allclose(call_args["layer1.weight"], torch.ones(2, 2).to(torch.bfloat16) * 3)
        assert mock_base.save_pretrained.called

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("safetensors.torch.load_file")
    def test_task_arithmetic_merge(self, mock_load_file, mock_from_pretrained, tmp_path):
        mock_base = MagicMock()
        # Base weight is 1
        mock_base.state_dict.return_value = {
            "layer1.weight": torch.ones(2, 2)
        }
        mock_from_pretrained.return_value = mock_base
        
        ckpt1 = tmp_path / "ckpt1"
        ckpt1.mkdir()
        (ckpt1 / "model.safetensors").touch()
        
        # Ckpt1 weight is 1.5 -> delta is 0.5
        mock_load_file.return_value = {"layer1.weight": torch.ones(2, 2) * 1.5}
        
        output = tmp_path / "unified"
        
        # Task arithmetic: Base + Delta*Weight = 1 + 0.5*1.0 = 1.5
        merge_checkpoints(
            base_model_path="base",
            checkpoint_paths=[str(ckpt1)],
            output_path=str(output),
            method="task_arithmetic",
            weights=[1.0]
        )
        
        call_args = mock_base.load_state_dict.call_args[0][0]
        assert torch.allclose(call_args["layer1.weight"], torch.ones(2, 2).to(torch.bfloat16) * 1.5)
