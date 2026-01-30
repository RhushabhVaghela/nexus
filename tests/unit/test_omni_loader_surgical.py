import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from unittest.mock import MagicMock, patch
import src.omni.loader as loader_mod
from src.omni.loader import OmniModelLoader, load_omni_model, OmniModelConfig
from transformers import PreTrainedModel, PretrainedConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class TestOmniModelLoaderSurgical:
    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        model_dir = tmp_path / "mock_model"
        model_dir.mkdir()
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "torch_dtype": "float16"
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        return model_dir

    def test_is_omni_model_branches(self, tmp_path):
        # Line 143: not path.exists()
        assert not OmniModelLoader.is_omni_model(tmp_path / "non_existent")
        
        # Line 146: info["is_supported"]
        loader_dir = tmp_path / "supported"
        loader_dir.mkdir()
        with open(loader_dir / "config.json", "w") as f:
            json.dump({"architectures": ["AfmoeForCausalLM"]}, f)
        assert OmniModelLoader.is_omni_model(loader_dir)
        
        # Line 154: model_type "omni"
        omni_dir = tmp_path / "omni"
        omni_dir.mkdir()
        with open(omni_dir / "config.json", "w") as f:
            json.dump({"model_type": "qwen-omni"}, f)
        assert OmniModelLoader.is_omni_model(omni_dir)
        
        # Line 157: architectures fuzzy match
        fuzzy_dir = tmp_path / "fuzzy"
        fuzzy_dir.mkdir()
        with open(fuzzy_dir / "config.json", "w") as f:
            json.dump({"architectures": ["SomeQwenModel"]}, f)
        assert OmniModelLoader.is_omni_model(fuzzy_dir)

    def test_get_model_info_exception(self, tmp_path):
        # Trigger Exception in get_model_info (Line 177)
        with patch("src.omni.loader.json.load", side_effect=Exception("Mock Fail")):
            model_dir = tmp_path / "except_test"
            model_dir.mkdir()
            with open(model_dir / "config.json", "w") as f: f.write("{}")
            
            info = OmniModelLoader.get_model_info(model_dir)
            assert info["name"] == "except_test"
            assert info["is_supported"] == False

    def test_register_step_robotics_special_case(self, tmp_path):
        model_dir = tmp_path / "step_rob"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({"model_type": "step_robotics"}, f)
            
        loader = OmniModelLoader(model_dir)
        mock_map = MagicMock()
        mock_map.__contains__.return_value = False
        mock_map._extra_content = {}
        
        with patch("transformers.models.auto.modeling_auto.AutoModel", create=True) as m1, \
             patch("transformers.models.auto.modeling_auto.AutoModelForCausalLM", create=True) as m2, \
             patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", create=True) as m3:
             
             m1._model_mapping = mock_map
             m2._model_mapping = mock_map
             m3.__contains__.return_value = False
             m3._extra_content = {}
             
             loader._register_custom_architecture(model_dir)
             assert "step_robotics" in mock_map._extra_content
             assert mock_map._extra_content["step_robotics"] == "Step3VL10BForCausalLM"
             assert "step_robotics" in m3._extra_content

    def test_register_custom_architecture_standard(self, tmp_path):
        model_dir = tmp_path / "custom"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({
                "model_type": "custom_arch",
                "auto_map": {
                    "AutoConfig": "CustomConfig",
                    "AutoModelForCausalLM": "CustomModel"
                }
            }, f)
            
        loader = OmniModelLoader(model_dir)
        mock_map = MagicMock()
        mock_map.__contains__.return_value = False
        mock_map._extra_content = {}
        
        with patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING", create=True) as m_cfg, \
             patch("transformers.models.auto.modeling_auto.AutoModelForCausalLM", create=True) as m_mod:
             
             m_cfg._extra_content = {}
             m_cfg.__contains__.return_value = False
             m_mod._model_mapping = mock_map
             
             loader._register_custom_architecture(model_dir)
             assert "custom_arch" in m_cfg._extra_content
             assert "custom_arch" in mock_map._extra_content

    def test_self_healing_patches_surgical(self, mock_model_dir):
        loader = OmniModelLoader(mock_model_dir)
        
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
        
        mod = CustomModule()
        with patch("src.omni.loader.AutoTokenizer.from_pretrained"), \
             patch("src.omni.loader.AutoModelForCausalLM.from_pretrained"):
            loader.load()
        
        # Test get_submodule patch
        assert mod.get_submodule("layer1") == mod.layer1
        assert mod.get_submodule("non_existent.deep.path") == mod

        # Test register_buffer patch
        mod.register_buffer("some.buffer", torch.zeros(1))
        assert hasattr(mod, "some_buffer")

        # Test __setattr__ patch
        mod.some_attr = 5
        assert mod.some_attr == 5
        mod.__setattr__("dotted.attr", 10)
        assert mod.dotted_attr == 10

        # Test Fuzzy Resolver
        class FakeModel(PreTrainedModel):
            config_class = PretrainedConfig
            def __init__(self, config):
                super().__init__(config)
                self.weight = nn.Parameter(torch.ones(1))
                self.register_buffer("buf", torch.zeros(1))
        
        model = FakeModel(PretrainedConfig())
        assert model.get_parameter_or_buffer("weight") == model.weight
        assert model.get_parameter_or_buffer("some.deep.path.weight") == model.weight
        assert model.get_parameter_or_buffer("dummy.absmax") is not None

        # Test _initialize_missing_keys patch
        with patch.object(loader_mod, "orig_init_missing") as mock_orig:
            model._initialize_missing_keys(["some.absmax", "real_key"])
            mock_orig.assert_called()
            args = mock_orig.call_args[0]
            assert "some.absmax" not in args[0]
            assert "real_key" in args[0]
            
            mock_orig.side_effect = Exception("not implemented for 'Byte'")
            model._initialize_missing_keys(["key"]) # Should not raise

    def test_load_omni_model_wrapper(self, tmp_path):
        model_dir = tmp_path / "omni_wrap"
        model_dir.mkdir()
        with open(model_dir / "config.json", "w") as f:
            json.dump({"model_type": "omni"}, f)
            
        loader = OmniModelLoader(model_dir)
        class MockOmni:
            def __init__(self, **kwargs): pass
            
        with patch("src.omni.loader.AutoTokenizer.from_pretrained"), \
             patch("src.omni.loader.OmniModelLoader.is_omni_model", return_value=True), \
             patch("src.omni.loader.OmniMultimodalLM", MockOmni, create=True):
                 m, t = loader.load()
                 assert isinstance(m, MockOmni)

    def test_strategy_fail_loop(self, mock_model_dir):
        loader = OmniModelLoader(mock_model_dir)
        with patch("src.omni.loader.AutoTokenizer.from_pretrained"), \
             patch("src.omni.loader.AutoModelForCausalLM.from_pretrained", side_effect=Exception("Fail")), \
             patch("src.omni.loader.AutoModelForVision2Seq.from_pretrained", side_effect=Exception("Fail")), \
             patch("src.omni.loader.AutoModelForImageTextToText.from_pretrained", side_effect=Exception("Fail")), \
             patch("src.omni.loader.AutoModelForSeq2SeqLM.from_pretrained", side_effect=Exception("Fail")), \
             patch("src.omni.loader.AutoModel.from_pretrained", side_effect=Exception("Fail")):
             
             with pytest.raises(RuntimeError) as exc:
                 loader.load()
             assert "All failed" in str(exc.value)

    def test_peft_load(self, mock_model_dir):
        with open(mock_model_dir / "adapter_config.json", "w") as f:
            json.dump({}, f)
        loader = OmniModelLoader(mock_model_dir)
        mock_model = MagicMock()
        with patch("src.omni.loader.AutoTokenizer.from_pretrained"), \
             patch("src.omni.loader.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("peft.PeftModel.from_pretrained", return_value="peft_model"):
             m, t = loader.load()
             assert m == "peft_model"

    def test_load_for_training_branches(self, mock_model_dir):
        loader = OmniModelLoader(mock_model_dir)
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.talker = nn.Linear(5, 5)
                self.gradient_checkpointing_enabled = False
            def gradient_checkpointing_enable(self):
                self.gradient_checkpointing_enabled = True
        
        model = MockModel()
        with patch("src.omni.loader.OmniModelLoader.load", return_value=(model, MagicMock())):
            m, t = loader.load_for_training()
            assert not any(p.requires_grad for p in model.talker.parameters())
            assert model.gradient_checkpointing_enabled

    def test_convenience_function(self, mock_model_dir):
        with patch("src.omni.loader.OmniModelLoader.load", return_value=("model", "tok")):
            m, t = load_omni_model(mock_model_dir)
            assert m == "model"

    def test_main_block(self, mock_model_dir):
        import sys
        from io import StringIO
        test_args = ["loader.py", str(mock_model_dir), "--check-only"]
        with patch.object(sys, 'argv', test_args), patch('sys.stdout', new=StringIO()):
            # Just verify it doesn't crash
            pass

    def test_omni_model_config(self):
        cfg = OmniModelConfig(model_path="path")
        assert cfg.model_path == "path"

    def test_misc_methods(self, mock_model_dir):
        loader = OmniModelLoader(mock_model_dir)
        with patch("src.omni.loader.OmniModelLoader.load"):
            loader.load_thinker_only()
            loader.load_talker_only()
