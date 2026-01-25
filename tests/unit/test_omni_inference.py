import pytest
import torch
from unittest.mock import MagicMock, patch
from src.omni.inference import OmniInference, GenerationConfig

class TestOmniInference:
    @patch("src.omni.loader.OmniModelLoader.load_for_inference")
    def test_init(self, mock_load):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        inference = OmniInference("/fake/path", enable_audio=False)
        assert inference.model == mock_model
        assert inference.tokenizer == mock_tokenizer
        assert not inference.enable_audio

    @patch("src.omni.loader.OmniModelLoader.load_for_inference")
    def test_generate(self, mock_load):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        mock_tokenizer.return_value = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}
        mock_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        mock_tokenizer.decode.return_value = "Hello"
        
        inference = OmniInference("/fake/path")
        res = inference.generate("Hi")
        
        assert res == "Hello"
        assert mock_model.generate.called

    @patch("src.omni.loader.OmniModelLoader.load_for_inference")
    @patch("transformers.TextIteratorStreamer")
    def test_generate_stream(self, mock_streamer_cls, mock_load):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        mock_streamer = MagicMock()
        mock_streamer.__iter__.return_value = iter(["Hel", "lo"])
        mock_streamer_cls.return_value = mock_streamer
        
        inference = OmniInference("/fake/path")
        stream = inference.generate_stream("Hi")
        
        results = list(stream)
        assert results == ["Hel", "lo"]

    @patch("src.omni.loader.OmniModelLoader.load_for_inference")
    def test_generate_with_audio(self, mock_load):
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Mock talker
        mock_model.talker = MagicMock()
        mock_model.talker.generate.return_value = MagicMock(audio_values="fake_audio")
        
        inference = OmniInference("/fake/path", enable_audio=True)
        # Mock generate text part
        with patch.object(inference, 'generate', return_value="Hello"):
            res = inference.generate_with_audio("Hi")
            assert res["text"] == "Hello"
            assert res["audio"] == "fake_audio"

    @patch("src.omni.loader.OmniModelLoader.load_for_inference")
    def test_chat(self, mock_load):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        inference = OmniInference("/fake/path")
        with patch.object(inference, 'generate', return_value="Hi back"):
            res = inference.chat([{"role": "user", "content": "Hi"}])
            assert res == "Hi back"
