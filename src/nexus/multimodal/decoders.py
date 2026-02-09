"""
Multimodal Decoders (SOTA 2026)
SigLIP 2 (512px) and Whisper V3 Turbo
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Mock torch/transformers if not present for CI/Tests
try:
    import torch
    from transformers import AutoProcessor
except ImportError:
    torch = None
    AutoProcessor = None


class ContentDecoder(ABC):
    """Base class for content decoders that process different modalities."""

    @abstractmethod
    def decode(self, file_path: str) -> Dict[str, Any]:
        """
        Decode content from a file path. Must be implemented by subclasses.

        Args:
            file_path: Path to the file to decode

        Returns:
            Dictionary containing decoded content and metadata
        """
        pass


class ImageDecoder(ContentDecoder):
    """
    SigLIP 2 Processor (512px)
    """

    def __init__(
        self,
        model_id: str = "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512",
    ):
        self.processor = None
        self.model_id = model_id
        if AutoProcessor:
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                logger.info(f"Loaded image processor: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load Image Processor from {model_id}: {e}")

    def decode(self, file_path: str) -> Dict[str, Any]:
        """
        Decode image from file path.

        Args:
            file_path: Path to image file

        Returns:
            Dictionary with modality info and pixel values tensor
        """
        if not Path(file_path).exists():
            return {
                "modality": "image",
                "tensor_type": "pixel_values",
                "processor_id": self.model_id,
                "warning": f"File not found: {file_path}",
            }

        from PIL import Image

        try:
            image = Image.open(file_path).convert("RGB")
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                return {
                    "modality": "image",
                    "tensor_type": "pixel_values",
                    "processor_id": self.model_id,
                    "pixel_values": inputs["pixel_values"],
                }
        except Exception as e:
            raise RuntimeError(f"Failed to process image {file_path}: {e}")

        return {
            "modality": "image",
            "tensor_type": "pixel_values",
            "processor_id": self.model_id,
        }


class AudioDecoder(ContentDecoder):
    """
    Whisper V3 Turbo
    """

    def __init__(
        self,
        model_id: str = "/mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo",
    ):
        self.processor = None
        self.model_id = model_id
        if AutoProcessor:
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                logger.info(f"Loaded audio processor: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load Audio Processor from {model_id}: {e}")

    def decode(self, file_path: str) -> Dict[str, Any]:
        """
        Decode audio from file path.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with modality info and input features tensor
        """
        if not Path(file_path).exists():
            return {
                "modality": "audio",
                "tensor_type": "input_features",
                "processor_id": self.model_id,
                "warning": f"File not found: {file_path}",
            }

        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Resample if needed (Whisper expects 16000Hz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                if torch is None:
                    raise RuntimeError(
                        "torch is required for audio mono conversion but is not available"
                    )
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if self.processor:
                inputs = self.processor(
                    waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
                )
                return {
                    "modality": "audio",
                    "tensor_type": "input_features",
                    "processor_id": self.model_id,
                    "input_features": inputs["input_features"],
                }
        except Exception as e:
            raise RuntimeError(f"Failed to process audio {file_path}: {e}")

        return {
            "modality": "audio",
            "tensor_type": "input_features",
            "processor_id": self.model_id,
        }


class VideoDecoder(ContentDecoder):
    """
    SigLIP 2 (Temporal Pooling)
    """

    def __init__(
        self,
        model_id: str = "/mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512",
    ):
        self.processor = None
        self.model_id = model_id
        if AutoProcessor:
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                logger.info(f"Loaded video processor: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load Video Processor from {model_id}: {e}")

    def decode(self, file_path: str) -> Dict[str, Any]:
        """
        Decode video from file path.

        Args:
            file_path: Path to video file

        Returns:
            Dictionary with modality info and video metadata
        """
        if not Path(file_path).exists():
            return {
                "modality": "video",
                "tensor_type": "pixel_values_stacked",
                "processor_id": self.model_id,
                "warning": f"File not found: {file_path}",
            }

        # Simple Temporal Pooling Strategy:
        # Load video, sample frames, process as batch of images.
        # For this implementation, we will verify the file and prepare metadata.
        # Actual loading would use decord or cv2, which might not be in the minimal env.

        return {
            "modality": "video",
            "tensor_type": "pixel_values_stacked",
            "processor_id": self.model_id,
            "file_path": file_path,
            "strategy": "temporal_pooling",
        }


class TextDecoder(ContentDecoder):
    """
    Text decoder/tokenizer for text generation outputs.
    Handles text decoding from model embeddings or token IDs.
    """

    def __init__(self, model_id: str = "gpt2"):
        self.tokenizer = None
        self.model_id = model_id
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info(f"Loaded text decoder tokenizer: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to load text tokenizer from {model_id}: {e}")

    def decode(self, file_path: str) -> Dict[str, Any]:
        """
        Decode text from a file path.

        Args:
            file_path: Path to text file to decode

        Returns:
            Dictionary with modality info and text content
        """
        if not Path(file_path).exists():
            return {
                "modality": "text",
                "tensor_type": "input_ids",
                "tokenizer_id": self.model_id,
                "warning": f"File not found: {file_path}",
            }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            result = {
                "modality": "text",
                "tensor_type": "input_ids",
                "tokenizer_id": self.model_id,
                "text": text_content,
            }

            # Tokenize if tokenizer is available
            if self.tokenizer:
                inputs = self.tokenizer(
                    text_content,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                result["input_ids"] = inputs["input_ids"]
                result["attention_mask"] = inputs.get("attention_mask")

            return result
        except Exception as e:
            logger.error(f"Error decoding text file {file_path}: {e}")
            return {
                "modality": "text",
                "tensor_type": "input_ids",
                "tokenizer_id": self.model_id,
                "error": str(e),
            }


class MultiModalDecoder:
    """
    Unified Multi-Modal Decoder for handling multiple modalities.
    Provides consistent interface for decoding various content types.
    """

    def __init__(self):
        self.image = ImageDecoder()
        self.audio = AudioDecoder()
        self.video = VideoDecoder()
        self.text = TextDecoder()

    def decode(self, file_path: str, modality: str) -> Dict[str, Any]:
        """
        Decode file based on specified modality.

        Args:
            file_path: Path to file
            modality: Type of content ('image', 'audio', 'video', 'text')

        Returns:
            Dictionary with decoded content
        """
        modality_map = {
            "vision": self.image,
            "image": self.image,
            "audio": self.audio,
            "video": self.video,
            "text": self.text,
        }

        if modality not in modality_map:
            raise ValueError(
                f"Unknown modality: {modality}. Supported: {list(modality_map.keys())}"
            )

        return modality_map[modality].decode(file_path)

    def decode_batch(
        self, file_paths: list[str], modalities: list[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Decode multiple files with different modalities.

        Args:
            file_paths: List of file paths
            modalities: List of modality types corresponding to files

        Returns:
            Dictionary mapping file paths to decode results
        """
        if len(file_paths) != len(modalities):
            raise ValueError("file_paths and modalities must have same length")

        results = {}
        for path, modality in zip(file_paths, modalities):
            results[path] = self.decode(path, modality)

        return results


# Backward compatibility alias
OmniDecoder = MultiModalDecoder
