
import torch
import torchaudio
from pathlib import Path
from typing import Optional
from .registry import voice_registry

class VoiceCloner:
    """
    Interface for PersonaPlex Zero-Shot Voice Cloning.
    Extracts a 'Voice DNA' (embedding) from a short audio clip.
    """
    
    def __init__(self, model_path: str = "/mnt/e/data/models/personaplex-7b-v1"):
        self.model_path = Path(model_path)
        self.encoder = None  # Lazy loaded
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_encoder(self):
        """Load the PersonaPlex Reference Encoder with 4-bit quantization."""
        if self.encoder is not None:
            return
        
        from transformers import AutoModel, BitsAndBytesConfig
        
        print(f"🚀 Loading PersonaPlex Encoder from {self.model_path} (4-bit mode)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        try:
            # We load the base model to extract the reference encoder weights
            # PersonaPlex 7B typically has a specialized head for audio embedding
            self.encoder = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.encoder.eval()
            print("✅ PersonaPlex Encoder loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load real model (likely missing weights): {e}")
            print("   Falling back to simulation mode for testing.")
            self.encoder = "SIMULATED_ENCODER"

    def clone_voice(self, audio_path_str: str, voice_name: str, description: str = "") -> Optional[str]:
        """
        Processes a 5-10s audio clip and saves the embedding DNA.
        
        Args:
            audio_path_str: Path to the audio file to clone
            voice_name: Name to give the cloned voice
            description: Optional description of the voice
            
        Returns:
            Path to the saved voice DNA file, or None if cloning failed
        """
        self._load_encoder()
        
        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            # 1. Load and preprocess audio
            print(f"Loading audio from {audio_path}...")
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed (standard for voice embeddings)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to 5-10 seconds (80k-160k samples at 16kHz)
            target_length = 16000 * 5  # 5 seconds
            if waveform.shape[1] > target_length * 2:
                # If longer than 10 seconds, take the middle 5 seconds
                start = (waveform.shape[1] - target_length) // 2
                waveform = waveform[:, start:start + target_length]
            elif waveform.shape[1] < target_length:
                # If shorter than 5 seconds, pad with zeros
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            print(f"Audio shape after preprocessing: {waveform.shape}")
            
            # 2. Extract Embedding (Voice DNA)
            print("Extracting voice DNA...")
            
            if self.encoder == "SIMULATED_ENCODER":
                # Fallback: Create a simple spectrogram-based embedding
                print("Using simulated encoder (spectrogram-based embedding)...")
                spec_transform = torchaudio.transforms.Spectrogram(n_fft=512)
                spectrogram = spec_transform(waveform)
                # Create a compact embedding by averaging frequency bands
                dna = torch.mean(spectrogram, dim=2).squeeze()
                # Normalize
                dna = dna / (torch.norm(dna) + 1e-8)
            else:
                # Use the real encoder
                with torch.no_grad():
                    waveform = waveform.to(self.device)
                    
                    # Try different encoder methods
                    if hasattr(self.encoder, 'extract_features'):
                        dna = self.encoder.extract_features(waveform)
                    elif hasattr(self.encoder, 'encode'):
                        dna = self.encoder.encode(waveform)
                    else:
                        # Fallback: use hidden states
                        outputs = self.encoder(waveform)
                        if hasattr(outputs, 'last_hidden_state'):
                            dna = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
                        else:
                            dna = outputs[0].squeeze()
            
            print(f"Extracted voice DNA with shape: {dna.shape}")
            
            # 3. Save DNA to storage
            save_dir = Path("/mnt/e/data/models/voice_dna")
            save_dir.mkdir(parents=True, exist_ok=True)
            dna_path = save_dir / f"{voice_name}.pt"
            
            # Save the DNA tensor and metadata
            save_dict = {
                'dna': dna.cpu(),
                'voice_name': voice_name,
                'description': description,
                'sample_rate': sample_rate,
                'original_path': str(audio_path),
            }
            torch.save(save_dict, dna_path)
            print(f"Successfully saved voice DNA to {dna_path}")
            
            # 4. Register in the voice registry
            voice_registry.register_voice(voice_name, str(dna_path), description)
            print(f"Registered voice '{voice_name}' in registry")
            
            return str(dna_path)
            
        except Exception as e:
            print(f"❌ Voice cloning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_voice_similarity(self, voice_path1: str, voice_path2: str) -> float:
        """
        Calculate similarity between two cloned voices.
        
        Args:
            voice_path1: Path to first voice DNA
            voice_path2: Path to second voice DNA
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            dna1 = torch.load(voice_path1)['dna']
            dna2 = torch.load(voice_path2)['dna']
            
            # Ensure same shape
            min_len = min(dna1.shape[0], dna2.shape[0])
            dna1 = dna1[:min_len]
            dna2 = dna2[:min_len]
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                dna1.unsqueeze(0),
                dna2.unsqueeze(0)
            )
            return similarity.item()
        except Exception as e:
            print(f"Error calculating voice similarity: {e}")
            return 0.0

voice_cloner = VoiceCloner()
