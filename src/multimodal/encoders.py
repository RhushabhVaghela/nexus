"""
multimodal/encoders.py
Multimodal encoders for vision, audio, and text modalities.
Integrates with repetition pipeline for enhanced feature extraction (Paper 2512.14982).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput:
    """Standardized output from multimodal encoders."""
    embeddings: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    modality: str = "unknown"
    repetition_metadata: Optional[Dict[str, Any]] = None


class ModalityEncoder:
    """Base class for modality-specific encoders."""
    
    def __init__(self, modality: str, embedding_dim: int = 768):
        self.modality = modality
        self.embedding_dim = embedding_dim
    
    def encode(self, data: Any, apply_repetition: bool = False, 
               repetition_factor: int = 1) -> EncoderOutput:
        """Encode data to embeddings. Override in subclasses."""
        raise NotImplementedError


class VisionEncoder(ModalityEncoder):
    """
    Vision encoder with repetition support for image inputs.
    Applies repetition at the feature level for vision-language tasks.
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",
                 embedding_dim: int = 768,
                 use_repetition_features: bool = True):
        super().__init__("vision", embedding_dim)
        self.model_name = model_name
        self.use_repetition_features = use_repetition_features
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the vision model."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            logger.info(f"Loaded vision encoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load vision model: {e}")
    
    def encode(self, 
               images: Union[List[str], List[torch.Tensor]], 
               apply_repetition: bool = False,
               repetition_factor: int = 1) -> EncoderOutput:
        """
        Encode images with optional repetition.
        
        Args:
            images: List of image paths or tensors
            apply_repetition: Whether to apply repetition
            repetition_factor: Number of times to repeat features
            
        Returns:
            EncoderOutput with image embeddings
        """
        if self.model is None:
            logger.error("Vision model not loaded")
            return EncoderOutput(
                embeddings=torch.zeros(1, self.embedding_dim),
                modality="vision"
            )
        
        # Process images
        if isinstance(images[0], str):
            # Image paths - load and process
            try:
                from PIL import Image
                images = [Image.open(path).convert("RGB") for path in images]
            except Exception as e:
                logger.error(f"Error loading images: {e}")
                return EncoderOutput(
                    embeddings=torch.zeros(1, self.embedding_dim),
                    modality="vision"
                )
        
        # Get embeddings
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        embeddings = outputs
        
        # Apply repetition at feature level
        if apply_repetition and repetition_factor > 1:
            embeddings = self._apply_feature_repetition(embeddings, repetition_factor)
        
        return EncoderOutput(
            embeddings=embeddings,
            modality="vision",
            repetition_metadata={
                "factor": repetition_factor if apply_repetition else 1,
                "num_images": len(images)
            }
        )
    
    def _apply_feature_repetition(self, embeddings: torch.Tensor, 
                                  factor: int) -> torch.Tensor:
        """
        Apply repetition to image embeddings.
        Repeats embeddings to reinforce visual features.
        """
        # Repeat embeddings along batch dimension
        repeated = embeddings.repeat_interleave(factor, dim=0)
        
        # Add slight noise to repeated features to maintain diversity
        if factor > 1:
            noise = torch.randn_like(repeated) * 0.01
            repeated = repeated + noise
        
        return repeated


class AudioEncoder(ModalityEncoder):
    """
    Audio encoder with repetition support for audio inputs.
    Applies repetition at the feature level for audio-language tasks.
    """
    
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 embedding_dim: int = 768,
                 use_repetition_features: bool = True):
        super().__init__("audio", embedding_dim)
        self.model_name = model_name
        self.use_repetition_features = use_repetition_features
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the audio model."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            logger.info(f"Loaded audio encoder: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load audio model: {e}")
    
    def encode(self,
               audio: Union[List[str], torch.Tensor],
               apply_repetition: bool = False,
               repetition_factor: int = 1) -> EncoderOutput:
        """
        Encode audio with optional repetition.
        
        Args:
            audio: List of audio file paths or audio tensor
            apply_repetition: Whether to apply repetition
            repetition_factor: Number of times to repeat features
            
        Returns:
            EncoderOutput with audio embeddings
        """
        if self.model is None:
            logger.error("Audio model not loaded")
            return EncoderOutput(
                embeddings=torch.zeros(1, self.embedding_dim),
                modality="audio"
            )
        
        # Load audio if paths provided
        if isinstance(audio[0], str):
            try:
                import librosa
                audio_arrays = []
                for path in audio:
                    waveform, sample_rate = librosa.load(path, sr=16000)
                    audio_arrays.append(waveform)
                audio = audio_arrays
            except Exception as e:
                logger.error(f"Error loading audio: {e}")
                return EncoderOutput(
                    embeddings=torch.zeros(1, self.embedding_dim),
                    modality="audio"
                )
        
        # Process audio
        inputs = self.processor(audio, return_tensors="pt", padding=True, 
                               sampling_rate=16000)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean of hidden states as embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Apply repetition at feature level
        if apply_repetition and repetition_factor > 1:
            embeddings = self._apply_feature_repetition(embeddings, repetition_factor)
        
        return EncoderOutput(
            embeddings=embeddings,
            modality="audio",
            repetition_metadata={
                "factor": repetition_factor if apply_repetition else 1,
                "num_audio": len(audio) if isinstance(audio, list) else 1
            }
        )
    
    def _apply_feature_repetition(self, embeddings: torch.Tensor,
                                  factor: int) -> torch.Tensor:
        """Apply repetition to audio embeddings."""
        repeated = embeddings.repeat_interleave(factor, dim=0)
        
        # Add temporal modulation to repeated features
        if factor > 1:
            for i in range(1, factor):
                idx = slice(i, None, factor)
                repeated[idx] = repeated[idx] * (1.0 - 0.05 * i)  # Attenuate slightly
        
        return repeated


class MultimodalEncoder:
    """
    Unified multimodal encoder that combines vision, audio, and text encoders.
    Implements repetition across all modalities for improved performance.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 enable_vision: bool = True,
                 enable_audio: bool = True):
        self.embedding_dim = embedding_dim
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(embedding_dim=embedding_dim) if enable_vision else None
        self.audio_encoder = AudioEncoder(embedding_dim=embedding_dim) if enable_audio else None
        
        # Projection layer to unify dimensions
        self.projection = nn.Linear(embedding_dim * 3, embedding_dim)  # Vision + Audio + Text
    
    def encode_multimodal(self,
                         text: Optional[str] = None,
                         images: Optional[List[str]] = None,
                         audio: Optional[List[str]] = None,
                         apply_repetition: bool = False,
                         repetition_factor: int = 1) -> Dict[str, EncoderOutput]:
        """
        Encode multimodal inputs with optional repetition.
        
        Args:
            text: Text input
            images: List of image paths
            audio: List of audio paths
            apply_repetition: Whether to apply repetition
            repetition_factor: Repetition factor
            
        Returns:
            Dictionary of encoder outputs by modality
        """
        outputs = {}
        
        # Encode vision
        if images and self.vision_encoder:
            outputs["vision"] = self.vision_encoder.encode(
                images, apply_repetition, repetition_factor
            )
        
        # Encode audio
        if audio and self.audio_encoder:
            outputs["audio"] = self.audio_encoder.encode(
                audio, apply_repetition, repetition_factor
            )
        
        # Text encoding would use the text encoder (simplified here)
        if text:
            # In practice, use the text encoder model
            text_embedding = torch.randn(1, self.embedding_dim)  # Placeholder
            outputs["text"] = EncoderOutput(
                embeddings=text_embedding,
                modality="text",
                repetition_metadata={"factor": repetition_factor if apply_repetition else 1}
            )
        
        return outputs
    
    def fuse_embeddings(self, 
                       outputs: Dict[str, EncoderOutput],
                       fusion_mode: str = "concat") -> torch.Tensor:
        """
        Fuse embeddings from multiple modalities.
        
        Args:
            outputs: Dictionary of encoder outputs
            fusion_mode: Fusion strategy ("concat", "sum", "attention")
            
        Returns:
            Fused embedding tensor
        """
        embeddings = [out.embeddings for out in outputs.values()]
        
        if fusion_mode == "concat":
            # Concatenate all embeddings
            fused = torch.cat(embeddings, dim=-1)
            # Project to target dimension
            return self.projection(fused)
        elif fusion_mode == "sum":
            # Sum embeddings (requires same dimensions)
            # Pad or project to same size if needed
            max_dim = max(e.shape[-1] for e in embeddings)
            padded = [nn.functional.pad(e, (0, max_dim - e.shape[-1])) for e in embeddings]
            return torch.stack(padded).sum(dim=0)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")


class RepetitionAwareEncoder:
    """
    Encoder that is aware of repetition and optimizes encoding accordingly.
    Implements KV-cache optimization for repeated inputs.
    """
    
    def __init__(self, base_encoder: MultimodalEncoder):
        self.encoder = base_encoder
        self.kv_cache = {}
        self.repetition_counter = {}
    
    def encode_with_cache(self,
                         text: Optional[str] = None,
                         images: Optional[List[str]] = None,
                         audio: Optional[List[str]] = None,
                         repetition_factor: int = 1,
                         cache_key: Optional[str] = None) -> Dict[str, EncoderOutput]:
        """
        Encode with KV-cache optimization for repeated content.
        Only encodes the first repetition, caches it, and reuses for subsequent repetitions.
        
        Args:
            text: Text input
            images: Image paths
            audio: Audio paths
            repetition_factor: Number of repetitions
            cache_key: Optional key for caching
            
        Returns:
            Encoder outputs with cached values
        """
        if cache_key is None:
            cache_key = f"{text}_{images}_{audio}"
        
        # Check cache for repeated content
        if cache_key in self.kv_cache and repetition_factor > 1:
            # Return cached encoding for second repetition onwards
            logger.debug(f"Using cached encoding for {cache_key}")
            cached_output = self.kv_cache[cache_key]
            
            # Update repetition counter
            self.repetition_counter[cache_key] = self.repetition_counter.get(cache_key, 1) + 1
            
            # If we've seen this multiple times, use cache
            if self.repetition_counter[cache_key] >= 2:
                return cached_output
        
        # Encode fresh
        outputs = self.encoder.encode_multimodal(
            text=text,
            images=images,
            audio=audio,
            apply_repetition=repetition_factor > 1,
            repetition_factor=repetition_factor
        )
        
        # Cache the output for future repetitions
        if repetition_factor > 1:
            self.kv_cache[cache_key] = outputs
            self.repetition_counter[cache_key] = 1
        
        return outputs
    
    def clear_cache(self):
        """Clear the KV cache."""
        self.kv_cache.clear()
        self.repetition_counter.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.kv_cache),
            "repetition_counts": self.repetition_counter.copy()
        }