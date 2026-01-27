import torch
import torch.nn as nn
from transformers import SpeechT5HifiGan, AutoModelForCausalLM
from diffusers import AutoencoderKL

class NexusDecoders:
    """
    Factory for initializing multimodal decoders at inference time.
    These are standard, lightweight components included in the bundle.
    """
    
    @staticmethod
    def load_audio_vocoder(model_id="microsoft/speecht5_hifigan", device="cuda"):
        """
        Loads the HiFi-GAN Vocoder for Text-to-Speech synthesis.
        Converts Mel Spectrograms (from Audio Adapter) -> Waveform.
        """
        print(f"Loading Audio Vocoder: {model_id}")
        vocoder = SpeechT5HifiGan.from_pretrained(model_id).to(device)
        vocoder.eval()
        return vocoder

    @staticmethod
    def load_image_decoder(model_id="stabilityai/sd-vae-ft-mse", device="cuda"):
        """
        Loads the VAE Decoder for Image Generation.
        Converts Latent Images (from Vision Adapter) -> RGB Pixels.
        """
        print(f"Loading Image VAE: {model_id}")
        vae = AutoencoderKL.from_pretrained(model_id).to(device)
        vae.eval()
        return vae

    @staticmethod
    def load_video_decoder(model_id="stabilityai/stable-video-diffusion-img2vid-xt", device="cuda"):
        """
        Loads SVD Decoder for Video Generation.
        """
        # SVD is complex; usually we load the full pipeline but here we might just want the VAE/UNet
        # For compactness, we might reuse the Image VAE and a temporal UNet.
        # Placeholder for V1.
        print(f"Loading Video Decoder: {model_id}")
        return None 
