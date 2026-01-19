"""
Omni-Modal Architecture (SOTA 2026)
Reference Implementation: GPT-OSS-20B -> Omni
Encoders: SigLIP 2 (Vision), Whisper V3 Turbo (Audio)
Connector: Perceiver Resampler
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoProcessor,
    WhisperModel,
    WhisperProcessor,
    AutoModelForCausalLM
)

try:
    from .connectors.dfm import DFMConnector
    DFM_AVAILABLE = True
except ImportError:
    DFM_AVAILABLE = False
    print("⚠️  DFM connector not available, using Perceiver Resampler")

# ═══════════════════════════════════════════════════════════════
# PERCEIVER RESAMPLER
# ═══════════════════════════════════════════════════════════════

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: Compresses variable-length modality features to fixed tokens.
    """
    def __init__(
        self,
        dim: int,                    # LLM dimension
        depth: int = 6,              # Resampler depth
        num_latents: int = 64,       # Output tokens per modality
        dim_head: int = 128,
        heads: int = 16,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            PerceiverAttentionBlock(dim, dim_head, heads, ff_mult)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        batch = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            latents = layer(latents, x)
        return self.norm(latents)

class PerceiverAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, ff_mult):
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim_head, heads)
        self.ff = FeedForward(dim, ff_mult)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, latents, context):
        latents = latents + self.cross_attn(self.norm1(latents), context)
        latents = latents + self.ff(self.norm2(latents))
        return latents

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = q.view(*q.shape[:2], h, -1).transpose(1, 2)
        k = k.view(*k.shape[:2], h, -1).transpose(1, 2)
        v = v.view(*v.shape[:2], h, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(*x.shape[:2], -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
    def forward(self, x):
        return self.net(x)

# ═══════════════════════════════════════════════════════════════
# ENCODERS
# ═══════════════════════════════════════════════════════════════

class VisionEncoder(nn.Module):
    """SigLIP 2 Vision Encoder (Feb 2025)"""
    def __init__(self, model_name="google/siglip-so400m-patch14-512", output_dim=1152):
        super().__init__()
        print(f"Loading Vision Encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.output_dim = output_dim
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        outputs = self.encoder.vision_model(pixel_values=images)
        return outputs.last_hidden_state

class AudioEncoder(nn.Module):
    """Whisper Large V3 Turbo Audio Encoder"""
    def __init__(self, model_name="openai/whisper-large-v3-turbo", output_dim=1280):
        super().__init__()
        print(f"Loading Audio Encoder: {model_name}")
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        self.output_dim = output_dim
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, audio_features):
        outputs = self.encoder(audio_features)
        return outputs.last_hidden_state

class VideoDecoder(nn.Module):
    """PaDT OVD 3B Video Decoder"""
    def __init__(self, model_name="/mnt/d/Research Experiments/manus_model/base-model/PaDT_OVD_3B"):
        super().__init__()
        print(f"Loading Video Decoder: {model_name}")
        self.decoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_dim = self.decoder.config.hidden_size
        # Freeze initially, will unfreeze in training
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def forward(self, embeddings):
        """Convert LLM embeddings to video frames"""
        return self.decoder.generate(embeddings)

class SpeechDecoder(nn.Module):
    """Parakeet TDT Speech Decoder"""
    def __init__(self, model_name="/mnt/d/Research Experiments/manus_model/base-model/parakeet-tdt-0.6b-v3"):
        super().__init__()
        print(f"Loading Speech Decoder: {model_name}")
        self.decoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_dim = self.decoder.config.hidden_size if hasattr(self.decoder.config, 'hidden_size') else 768
        # Freeze initially
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def forward(self, embeddings):
        """Convert LLM embeddings to speech audio"""
        return self.decoder.generate(embeddings)

# ═══════════════════════════════════════════════════════════════
# OMNI MODEL
# ═══════════════════════════════════════════════════════════════

class OmniMultimodalLM(nn.Module):
    def __init__(
        self,
        llm_name: str,
        llm_dim: int = 4096,
        num_latents: int = 64,
        vision_name: str = "google/siglip-so400m-patch14-512",
        audio_name: str = "openai/whisper-large-v3-turbo",
        enable_decoders: bool = True,
        use_dfm: bool = True,
        device_map: str = "auto",  # NEW: CPU/GPU hybrid
        load_in_8bit: bool = True   # NEW: Quantization for memory
    ):
        super().__init__()
        
        # LLM with CPU offloading + quantization
        print(f"Loading Base LLM: {llm_name}")
        if load_in_8bit:
            print("  Using 4-bit quantization for ultra speed (QLoRA)")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4-bit instead of 8-bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",  # NormalFloat4
                bnb_4bit_use_double_quant=True  # Double quantization
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, 
                device_map="cpu",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, 
                dtype=torch.float16, 
                device_map=device_map,
                trust_remote_code=True
            )
        self.llm_dim = self.llm.config.hidden_size
        
        # Encoders
        self.vision_encoder = VisionEncoder(model_name=vision_name)
        self.audio_encoder = AudioEncoder(model_name=audio_name)
        
        # Input Projections (encoder → LLM)
        self.vision_proj = nn.Linear(self.vision_encoder.output_dim, self.llm_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.output_dim, self.llm_dim)
        
        # Connectors (DFM for SOTA or Perceiver for fallback)
        self.use_dfm = use_dfm and DFM_AVAILABLE
        if self.use_dfm:
            print(f"Using DFM Connectors (SOTA) with {num_latents} latents...")
            from .connectors.dfm import DFMConnector
            self.vision_connector = DFMConnector(dim=self.llm_dim, num_latents=num_latents)
            self.audio_connector = DFMConnector(dim=self.llm_dim, num_latents=num_latents)
        else:
            print(f"Using Perceiver Resamplers with {num_latents} latents...")
            self.vision_connector = PerceiverResampler(dim=self.llm_dim, num_latents=num_latents)
            self.audio_connector = PerceiverResampler(dim=self.llm_dim, num_latents=num_latents)
        
        # Decoders (for any-to-any)
        self.enable_decoders = enable_decoders
        if enable_decoders:
            self.video_decoder = VideoDecoder()
            self.speech_decoder = SpeechDecoder()
            
            # Output Projections (LLM → decoder)
            self.video_proj_out = nn.Linear(self.llm_dim, self.video_decoder.hidden_dim)
            self.speech_proj_out = nn.Linear(self.llm_dim, self.speech_decoder.hidden_dim)

    def encode_vision(self, images):
        features = self.vision_encoder(images)
        features = self.vision_proj(features)
        if self.use_dfm:
            tokens, _ = self.vision_connector(features)
        else:
            tokens = self.vision_connector(features)
        return tokens

    def encode_audio(self, audio_features):
        features = self.audio_encoder(audio_features)
        features = self.audio_proj(features)
        if self.use_dfm:
            tokens, _ = self.audio_connector(features)
        else:
            tokens = self.audio_connector(features)
        return tokens

    def forward(self, input_ids, pixel_values=None, audio_features=None, output_modality="text", **kwargs):
        """Forward pass with any-to-any support.
        
        Args:
            output_modality: "text" | "video" | "speech" - target output modality
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        multimodal_embeds = []
        
        if pixel_values is not None:
            multimodal_embeds.append(self.encode_vision(pixel_values))
        
        if audio_features is not None:
            multimodal_embeds.append(self.encode_audio(audio_features))
            
        if multimodal_embeds:
            # Concatenate [Vision, Audio, Text]
            all_embeds = torch.cat(multimodal_embeds + [text_embeds], dim=1)
        else:
            all_embeds = text_embeds
        
        # Get LLM outputs
        llm_outputs = self.llm(inputs_embeds=all_embeds, **kwargs)
        
        # Route to appropriate decoder
        if output_modality == "text":
            return llm_outputs
        elif output_modality == "video" and self.enable_decoders:
            # Project LLM hidden states to video decoder
            video_embeds = self.video_proj_out(llm_outputs.last_hidden_state)
            return self.video_decoder(video_embeds)
        elif output_modality == "speech" and self.enable_decoders:
            # Project LLM hidden states to speech decoder
            speech_embeds = self.speech_proj_out(llm_outputs.last_hidden_state)
            return self.speech_decoder(speech_embeds)
        else:
            return llm_outputs

    def save_pretrained(self, path):
        self.llm.save_pretrained(f"{path}/llm")
        torch.save(self.vision_resampler.state_dict(), f"{path}/vision_adapter.pt")
        torch.save(self.audio_resampler.state_dict(), f"{path}/audio_adapter.pt")
