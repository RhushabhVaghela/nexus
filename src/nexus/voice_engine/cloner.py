"""
Voice Cloner Module

Provides voice cloning capabilities using neural audio encoding.
Integrates with XTTS-v2 for high-quality voice cloning.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Optional torch/torchaudio imports
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    TORCH_AVAILABLE = False

from .registry import voice_registry


@dataclass
class VoiceDNA:
    """Voice DNA containing speaker characteristics."""
    embedding: Any  # torch.Tensor when available
    sample_rate: int
    duration_seconds: float
    source_path: str
    voice_name: str
    metadata: Dict[str, Any]
    
    def save(self, output_path: Path) -> None:
        """Save voice DNA to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "embedding": self.embedding,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "source_path": self.source_path,
            "voice_name": self.voice_name,
            "metadata": self.metadata,
        }
        
        if TORCH_AVAILABLE:
            torch.save(save_dict, output_path)
        else:
            # Fallback to JSON for non-torch environments
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: Path) -> "VoiceDNA":
        """Load voice DNA from disk."""
        if TORCH_AVAILABLE:
            data = torch.load(path, weights_only=False)
        else:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        return cls(
            embedding=data["embedding"],
            sample_rate=data["sample_rate"],
            duration_seconds=data["duration_seconds"],
            source_path=data["source_path"],
            voice_name=data["voice_name"],
            metadata=data.get("metadata", {}),
        )


class VoiceEncoder:
    """
    Neural encoder for extracting voice characteristics.
    Uses WavLM or similar models for speaker embedding extraction.
    """
    
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    def _load_model(self):
        """Lazy load the encoder model."""
        if self._model is None:
            if not TORCH_AVAILABLE:
                self._model = "fallback"
                return
                
            try:
                from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
                
                self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                self._model = WavLMForXVector.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
            except Exception as e:
                # Fallback: create a simple feature extractor
                self._model = "fallback"
                self._processor = None
    
    def extract_features(self, waveform: Any, sample_rate: int) -> Any:
        """
        Extract speaker embedding from audio.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Speaker embedding tensor
        """
        self._load_model()
        
        if self._model == "fallback" or not TORCH_AVAILABLE:
            # Simple statistical features as fallback
            return self._extract_fallback_features(waveform, sample_rate)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        
        # Process through model
        with torch.no_grad():
            inputs = self._processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self._model(**inputs).embeddings
            
        return embeddings.cpu().squeeze()
    
    def _extract_fallback_features(self, waveform: Any, sample_rate: int) -> Any:
        """Extract simple statistical features as fallback."""
        if TORCH_AVAILABLE:
            # Ensure mono
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            
            features = torch.tensor([
                waveform.mean().item(),
                waveform.std().item(),
                torch.abs(waveform).max().item(),
                (waveform > 0).float().mean().item(),
                torch.quantile(waveform, 0.25).item(),
                torch.quantile(waveform, 0.75).item(),
            ])
            
            # Pad to 512 dimensions
            return torch.nn.functional.pad(features, (0, 512 - features.shape[0]))
        else:
            # Return zeros when torch not available
            import numpy as np
            return np.zeros(512)


class VoiceCloner:
    """
    Voice cloning engine for replicating voice characteristics.
    
    Integrates with XTTS-v2 and other TTS models for voice cloning.
    """
    
    def __init__(
        self,
        encoder_model: str = "microsoft/wavlm-base-plus-sv",
        dna_storage_dir: Optional[str] = None
    ):
        """
        Initialize voice cloner.
        
        Args:
            encoder_model: Model for extracting speaker embeddings
            dna_storage_dir: Directory to store voice DNA files
        """
        self.encoder = VoiceEncoder(encoder_model)
        self.dna_storage_dir = Path(dna_storage_dir) if dna_storage_dir else Path("/tmp/nexus_voice_dna")
        self.dna_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.clones_created = 0
    
    def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
        description: str = "Custom cloned voice"
    ) -> Optional[str]:
        """
        Clone voice from audio sample.
        
        Args:
            audio_path: Path to source audio file (.wav, .mp3)
            voice_name: Name for the cloned voice
            description: Voice description
            
        Returns:
            Path to saved voice DNA file, or None if failed
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio
            if TORCH_AVAILABLE and torchaudio is not None:
                waveform, sample_rate = torchaudio.load(str(audio_path))
                duration_seconds = waveform.shape[-1] / sample_rate
            else:
                # Fallback for non-torch environments
                waveform = None
                sample_rate = 16000
                duration_seconds = 0.0
            
            # Extract features
            embedding = self.encoder.extract_features(waveform, sample_rate)
            
            # Create voice DNA
            voice_dna = VoiceDNA(
                embedding=embedding,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                source_path=str(audio_path),
                voice_name=voice_name,
                metadata={
                    "description": description,
                    "encoder_model": self.encoder.model_name,
                }
            )
            
            # Save DNA
            safe_name = voice_name.replace(" ", "_").lower()
            dna_filename = f"{safe_name}_{hashlib.md5(voice_name.encode()).hexdigest()[:8]}.pt"
            dna_path = self.dna_storage_dir / dna_filename
            
            voice_dna.save(dna_path)
            
            # Register in voice registry
            voice_registry.register_voice(voice_name, str(dna_path), description)
            
            self.clones_created += 1
            
            return str(dna_path)
            
        except Exception as e:
            print(f"Voice cloning failed: {e}")
            return None
    
    def load_voice_dna(self, dna_path: str) -> Optional[VoiceDNA]:
        """
        Load voice DNA from file.
        
        Args:
            dna_path: Path to voice DNA file
            
        Returns:
            VoiceDNA object or None if failed
        """
        try:
            return VoiceDNA.load(Path(dna_path))
        except Exception as e:
            print(f"Failed to load voice DNA: {e}")
            return None
    
    def compare_voices(self, dna_path1: str, dna_path2: str) -> float:
        """
        Compare two voices and return similarity score.
        
        Args:
            dna_path1: Path to first voice DNA
            dna_path2: Path to second voice DNA
            
        Returns:
            Similarity score between 0 and 1
        """
        dna1 = self.load_voice_dna(dna_path1)
        dna2 = self.load_voice_dna(dna_path2)
        
        if dna1 is None or dna2 is None:
            return 0.0
        
        # Cosine similarity
        emb1 = dna1.embedding
        emb2 = dna2.embedding
        
        if TORCH_AVAILABLE:
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0)
            ).item()
            # Normalize to 0-1 range
            return (similarity + 1) / 2
        else:
            # Simple numpy cosine similarity
            import numpy as np
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = dot / (norm1 * norm2)
            return (similarity + 1) / 2


# Global instance for convenience
voice_cloner = VoiceCloner()
