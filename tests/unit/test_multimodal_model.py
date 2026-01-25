import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.multimodal.model import OmniMultimodalLM, PerceiverResampler, ModularMultimodalWrapper, VisionEncoder, AudioEncoder, VideoDecoder, SpeechDecoder

class TestMultimodalComponents:
    @patch("src.multimodal.model.AutoModel.from_pretrained")
    def test_vision_encoder(self, mock_from):
        mock_model = MagicMock()
        mock_model.vision_model.return_value = MagicMock(last_hidden_state=torch.randn(1, 10, 1152))
        mock_from.return_value = mock_model
        enc = VisionEncoder(load_in_8bit=True)
        out = enc(torch.randn(1, 3, 512, 512))
        assert out.shape == (1, 10, 1152)

    @patch("src.multimodal.model.WhisperModel.from_pretrained")
    def test_audio_encoder(self, mock_from):
        mock_model = MagicMock()
        mock_model.encoder.return_value = MagicMock(last_hidden_state=torch.randn(1, 10, 1280))
        mock_from.return_value = mock_model
        enc = AudioEncoder(load_in_8bit=True)
        out = enc(torch.randn(1, 80, 3000))
        assert out.shape == (1, 10, 1280)

    @patch("src.multimodal.model.AutoModel.from_pretrained")
    def test_speech_decoder_fail(self, mock_from):
        mock_from.side_effect = Exception("Fail")
        with pytest.raises(Exception):
            SpeechDecoder()

    def test_save_pretrained(self, tmp_path):
        base = MagicMock()
        with patch("src.multimodal.model.VisionEncoder"), \
             patch("src.multimodal.model.AudioEncoder"), \
             patch("src.multimodal.model.VideoDecoder"), \
             patch("src.multimodal.model.SpeechDecoder"):
            wrapper = ModularMultimodalWrapper(base, inject_vision=True, inject_audio=True, llm_dim=512)
            wrapper.save_pretrained(str(tmp_path))
            assert base.save_pretrained.called

    @patch("src.multimodal.model.AutoModel.from_pretrained")
    def test_video_decoder(self, mock_from):
        mock_model = MagicMock()
        mock_model.config.hidden_size = 512
        mock_model.generate.return_value = torch.zeros(1, 10)
        mock_from.return_value = mock_model
        dec = VideoDecoder()
        out = dec(torch.randn(1, 512))
        assert out.shape == (1, 10)

    def test_perceiver_resampler(self):
        resampler = PerceiverResampler(dim=512, depth=1, num_latents=8)
        x = torch.randn(2, 20, 512)
        out = resampler(x)
        assert out.shape == (2, 8, 512)

    def test_modular_wrapper_forward_full(self, device):
        base = MagicMock()
        base.config = MagicMock(hidden_size=512)
        base.get_input_embeddings.return_value = MagicMock(return_value=torch.randn(1, 5, 512).to(device, dtype=torch.float16))
        
        mock_out = MagicMock()
        mock_out.loss = torch.tensor(1.0).to(device, dtype=torch.float16)
        mock_out.last_hidden_state = torch.randn(1, 10, 512).to(device, dtype=torch.float16)
        base.return_value = mock_out
        
        with patch("src.multimodal.model.VisionEncoder") as mv, \
             patch("src.multimodal.model.AudioEncoder") as ma, \
             patch("src.multimodal.model.VideoDecoder") as mvd, \
             patch("src.multimodal.model.SpeechDecoder") as msd:
            
            mv.return_value.output_dim = 1152
            mv.return_value.return_value = torch.randn(1, 10, 1152).to(device, dtype=torch.float16)
            ma.return_value.output_dim = 1280
            ma.return_value.return_value = torch.randn(1, 10, 1280).to(device, dtype=torch.float16)
            
            mvd.return_value.hidden_dim = 256
            mvd.return_value.return_value = torch.zeros(1, 10)
            msd.return_value.hidden_dim = 256
            msd.return_value.return_value = torch.zeros(1, 10)

            wrapper = ModularMultimodalWrapper(
                base, inject_vision=True, inject_audio=True, llm_dim=512, 
                use_dfm=False, visual_repetition_factor=2, audio_repetition_factor=2
            )
            wrapper.to(device)
            
            px = torch.randn(1, 3, 512, 512).to(device, dtype=torch.float16)
            ax = torch.randn(1, 80, 3000).to(device, dtype=torch.float16)
            
            out = wrapper(
                input_ids=torch.zeros((1, 5), dtype=torch.long).to(device),
                pixel_values=px,
                audio_features=ax,
                output_modality="video"
            )
            assert out is not None
            
            out_s = wrapper(
                input_ids=torch.zeros((1, 5), dtype=torch.long).to(device),
                output_modality="speech"
            )
            assert out_s is not None

class TestOmniMultimodalLM:
    @patch("src.multimodal.model.AutoConfig.from_pretrained")
    @patch("src.multimodal.model.AutoModelForCausalLM.from_pretrained")
    @patch("src.multimodal.model.AutoModel.from_pretrained")
    @patch("src.multimodal.model.VideoDecoder")
    @patch("src.multimodal.model.SpeechDecoder")
    def test_init_with_injection(self, mock_speech, mock_video, mock_vision_fn, mock_llm_fn, mock_config_fn, device):
        mock_llm = MagicMock()
        mock_llm.config = MagicMock(hidden_size=4096)
        mock_llm.device = torch.device(device)
        mock_llm_fn.return_value = mock_llm
        
        mock_vision = MagicMock()
        mock_vision.config = MagicMock(hidden_size=1024)
        mock_vision_fn.return_value = mock_vision
        
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        if hasattr(mock_config, "vision_config"): del mock_config.vision_config
        if hasattr(mock_config, "audio_config"): del mock_config.audio_config
        mock_config_fn.return_value = mock_config
        
        model = OmniMultimodalLM(llm_name="fake_llm", inject_vision=True, inject_audio=False)
        assert hasattr(model.wrapper, "vision_encoder")

    @patch("src.multimodal.model.AutoConfig.from_pretrained")
    @patch("src.multimodal.model.AutoModelForCausalLM.from_pretrained")
    @patch("src.multimodal.model.VideoDecoder")
    @patch("src.multimodal.model.SpeechDecoder")
    def test_forward_text_only(self, mock_speech, mock_video, mock_llm_fn, mock_config_fn, device):
        mock_llm = MagicMock()
        mock_llm.config = MagicMock(hidden_size=4096)
        mock_llm.device = torch.device(device)
        mock_llm.get_input_embeddings.return_value = MagicMock(return_value=torch.randn(1, 10, 4096).to(device, dtype=torch.float16))
        mock_llm.return_value = MagicMock(loss=torch.tensor(1.0).to(device, dtype=torch.float16))
        mock_llm_fn.return_value = mock_llm
        
        mock_config = MagicMock()
        mock_config.model_type = "llama"
        mock_config_fn.return_value = mock_config
        
        model = OmniMultimodalLM(llm_name="fake_llm", inject_vision=False, inject_audio=False)
        model.to(device)
        
        input_ids = torch.randint(0, 1000, (1, 10)).to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        assert outputs.loss == 1.0

    @patch("src.multimodal.model.AutoConfig.from_pretrained")
    @patch("src.multimodal.model.AutoModelForCausalLM.from_pretrained")
    @patch("src.multimodal.model.AutoModel.from_pretrained")
    @patch("src.multimodal.model.VideoDecoder")
    @patch("src.multimodal.model.SpeechDecoder")
    @patch("src.multimodal.model.VisionEncoder")
    @patch("src.multimodal.model.AudioEncoder")
    def test_transplantation_success(self, mock_audio, mock_vision, mock_speech, mock_video, mock_auto_model, mock_llm_fn, mock_config_fn, device):
        # Setup AutoModel mock to behave correctly for different calls
        def auto_side_effect(name, **kwargs):
            if name == "fail_native": raise Exception("AutoModel Fail")
            m = MagicMock()
            m.config.hidden_size = 512
            m.vision_model.return_value = MagicMock(last_hidden_state=torch.randn(1, 10, 1152))
            return m
        mock_auto_model.side_effect = auto_side_effect

        def llm_side_effect(name, **kwargs):
            if 'config' in kwargs: # Successful transplantation call
                m = MagicMock()
                m.config.hidden_size = 512
                return m
            raise Exception("Fail")
        mock_llm_fn.side_effect = llm_side_effect
        
        mock_orig_cfg = MagicMock()
        mock_orig_cfg.vocab_size = 1000
        mock_orig_cfg.hidden_size = 512
        mock_orig_cfg.num_hidden_layers = 2
        mock_orig_cfg.num_attention_heads = 4
        mock_orig_cfg.intermediate_size = 1024
        mock_orig_cfg.quantization_config = MagicMock()
        mock_config_fn.return_value = mock_orig_cfg
        
        # Ensure mocks return objects with expected attributes
        mock_vision.return_value.output_dim = 1152
        mock_audio.return_value.output_dim = 1280
        
        model = OmniMultimodalLM(llm_name="fail_native")
        assert model.wrapper is not None

    @patch("src.multimodal.model.AutoModelForCausalLM.from_pretrained")
    @patch("src.multimodal.model.AutoConfig.from_pretrained")
    def test_gptq_fix(self, mock_cfg, mock_llm_fn):
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.qzeros = torch.zeros(1, dtype=torch.float32)
        mock_module.qweight = torch.zeros(1, dtype=torch.float32)
        mock_module.scales = torch.zeros(1, dtype=torch.float32) # hit scales path
        mock_model.named_modules.return_value = [("layer1", mock_module)]
        mock_llm_fn.return_value = mock_model
        mock_cfg.return_value.model_type = "qwen2"
        
        with patch("src.multimodal.model.VideoDecoder"), patch("src.multimodal.model.SpeechDecoder"):
            model = OmniMultimodalLM(llm_name="gptq_model")
            assert mock_module.qzeros.dtype == torch.int32
            assert mock_module.qweight.dtype == torch.int32

    def test_input_schema_full(self):
        # Create a real wrapper but mock the encoders
        base = MagicMock()
        base.config.hidden_size = 512
        with patch("src.multimodal.model.VisionEncoder"), \
             patch("src.multimodal.model.AudioEncoder"), \
             patch("src.multimodal.model.VideoDecoder"), \
             patch("src.multimodal.model.SpeechDecoder"), \
             patch("src.multimodal.model.AutoModelForCausalLM.from_pretrained") as ml, \
             patch("src.multimodal.model.AutoConfig.from_pretrained") as mc:
            
            ml.return_value = base
            # Ensure it is NOT treated as native omni so injection happens
            mc.return_value.model_type = "llama"
            if hasattr(mc.return_value, "vision_config"): del mc.return_value.vision_config
            if hasattr(mc.return_value, "audio_config"): del mc.return_value.audio_config
            
            model = OmniMultimodalLM(llm_name="f", inject_vision=True)
            schema = model.get_input_schema()
            assert schema["requires_vision_input"] is True
            # When injected, it expects 'pixel_values'
            # If native (qwen2_vl), it expects 'images' or 'pixel_values' depending on config
            assert schema["vision_key"] == "pixel_values"
            
    def test_save_pretrained_omni(self, tmp_path):
        model = MagicMock(spec=OmniMultimodalLM)
        model.wrapper = MagicMock()
        OmniMultimodalLM.save_pretrained(model, str(tmp_path))
        assert model.wrapper.save_pretrained.called
            
    def test_forward_alias(self):
        model = MagicMock(spec=OmniMultimodalLM)
        model.wrapper = MagicMock()
        OmniMultimodalLM.forward(model, "arg")
        assert model.wrapper.called
