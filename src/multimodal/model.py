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
    def __init__(self, model_name="google/siglip-so400m-patch14-512", output_dim=1152, load_in_8bit=False):
        super().__init__()
        print(f"Loading Vision Encoder: {model_name}")
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.encoder = AutoModel.from_pretrained(
                model_name, 
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.output_dim = output_dim
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        outputs = self.encoder.vision_model(pixel_values=images)
        return outputs.last_hidden_state

class AudioEncoder(nn.Module):
    """Whisper Large V3 Turbo Audio Encoder"""
    def __init__(self, model_name="openai/whisper-large-v3-turbo", output_dim=1280, load_in_8bit=False):
        super().__init__()
        print(f"Loading Audio Encoder: {model_name}")
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            whisper_model = WhisperModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.encoder = whisper_model.encoder
        else:
            self.encoder = WhisperModel.from_pretrained(model_name).encoder
        self.output_dim = output_dim
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, audio_features):
        outputs = self.encoder(audio_features)
        return outputs.last_hidden_state

import warnings
import logging
# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*CUDA extension not installed.*")
warnings.filterwarnings("ignore", message=".*Unrecognized keys in `rope_scaling`.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="auto_gptq")
logging.getLogger("auto_gptq").setLevel(logging.ERROR)

class VideoDecoder(nn.Module):
    """PaDT OVD 3B Video Decoder"""
    def __init__(self, model_name="/mnt/e/data/base-model/PaDT_OVD_3B"):
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
    def __init__(self, model_name="/mnt/e/data/base-model/parakeet-tdt-0.6b-v3"):
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

from transformers import AutoConfig

# ═══════════════════════════════════════════════════════════════
# MODULAR WRAPPER (The "Frankenstein" Builder)
# ═══════════════════════════════════════════════════════════════

class ModularMultimodalWrapper(nn.Module):
    """
    Wraps a Base LLM and injects ONLY the missing modality encoders.
    """
    def __init__(
        self,
        base_model,
        inject_vision: bool = False,
        inject_audio: bool = False,
        vision_name: str = "google/siglip-so400m-patch14-512",
        audio_name: str = "openai/whisper-large-v3-turbo",
        llm_dim: int = 4096,
        num_latents: int = 64,
        use_dfm: bool = True,
        enable_decoders: bool = True
    ):
        super().__init__()
        self.llm = base_model
        self.llm_dim = llm_dim
        self.inject_vision = inject_vision
        self.inject_audio = inject_audio
        self.use_dfm = use_dfm and DFM_AVAILABLE
        self.enable_decoders = enable_decoders
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 1. Vision Injection
        if self.inject_vision:
            print(f"  👁️  Injecting Vision Module ({vision_name})...")
            self.vision_encoder = VisionEncoder(model_name=vision_name, load_in_8bit=True)
            self.vision_proj = nn.Linear(self.vision_encoder.output_dim, self.llm_dim).to(device)
            
            if self.use_dfm:
                self.vision_connector = DFMConnector(dim=self.llm_dim, num_latents=num_latents).to(device)
            else:
                self.vision_connector = PerceiverResampler(dim=self.llm_dim, num_latents=num_latents).to(device)
        else:
            print("  ✅ Base model handles Vision natively. Skipping injection.")

        # 2. Audio Injection
        if self.inject_audio:
            print(f"  👂  Injecting Audio Module ({audio_name})...")
            self.audio_encoder = AudioEncoder(model_name=audio_name, load_in_8bit=True)
            self.audio_proj = nn.Linear(self.audio_encoder.output_dim, self.llm_dim).to(device)
            
            if self.use_dfm:
                self.audio_connector = DFMConnector(dim=self.llm_dim, num_latents=num_latents).to(device)
            else:
                self.audio_connector = PerceiverResampler(dim=self.llm_dim, num_latents=num_latents).to(device)
        else:
            print("  ✅ Base model handles Audio natively. Skipping injection.")

        # 3. Output Decoders (Optional / Any-to-Any)
        # Assuming if we inject encoders, we probably need decoders, OR if user explicitly asks.
        # But if base model is Omni, it might have talker heads?
        # For safety/consistency with user request, we inject these if enabled.
        if self.enable_decoders:
             self.video_decoder = VideoDecoder()
             self.speech_decoder = SpeechDecoder()
             self.video_proj_out = nn.Linear(self.llm_dim, self.video_decoder.hidden_dim).to(device)
             self.speech_proj_out = nn.Linear(self.llm_dim, self.speech_decoder.hidden_dim).to(device)

    def encode_vision(self, images):
        if not self.inject_vision: return None
        features = self.vision_encoder(images)
        features = self.vision_proj(features)
        if self.use_dfm:
            tokens, _ = self.vision_connector(features)
        else:
            tokens = self.vision_connector(features)
        return tokens

    def encode_audio(self, audio_features):
        if not self.inject_audio: return None
        features = self.audio_encoder(audio_features)
        features = self.audio_proj(features)
        if self.use_dfm:
            tokens, _ = self.audio_connector(features)
        else:
            tokens = self.audio_connector(features)
        return tokens

    def forward(self, input_ids, pixel_values=None, audio_features=None, output_modality="text", **kwargs):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        multimodal_embeds = []
        
        # Only encode if injected AND provided
        if self.inject_vision and pixel_values is not None:
            multimodal_embeds.append(self.encode_vision(pixel_values))
        
        if self.inject_audio and audio_features is not None:
            multimodal_embeds.append(self.encode_audio(audio_features))
            
        if multimodal_embeds:
            all_embeds = torch.cat(multimodal_embeds + [text_embeds], dim=1)
        else:
            all_embeds = text_embeds
        
        # For Native models (not injected), inputs like 'pixel_values' might need to be passed kwargs?
        # But here we assume if NOT injected, the Base LLM handles it via its own forward() logic?
        # Actually, if we wrap it, we usually control the embedding concatenation.
        # If Base is Native, we shouldn't be doing embedding concat here manually unless we are MIXING native + custom.
        # For now, simplistic concat logic for injected modalities.
        
        llm_outputs = self.llm(inputs_embeds=all_embeds, **kwargs)
        
        if output_modality == "text":
            return llm_outputs
        elif output_modality == "video" and self.enable_decoders:
            video_embeds = self.video_proj_out(llm_outputs.last_hidden_state)
            return self.video_decoder(video_embeds)
        elif output_modality == "speech" and self.enable_decoders:
            speech_embeds = self.speech_proj_out(llm_outputs.last_hidden_state)
            return self.speech_decoder(speech_embeds)
        else:
            return llm_outputs
            
    def save_pretrained(self, path):
         self.llm.save_pretrained(f"{path}/llm")
         if self.inject_vision:
             torch.save(self.vision_connector.state_dict(), f"{path}/vision_adapter.pt")
         if self.inject_audio:
             torch.save(self.audio_connector.state_dict(), f"{path}/audio_adapter.pt")

# ═══════════════════════════════════════════════════════════════
# SMART DETECTING FACTORY
# ═══════════════════════════════════════════════════════════════

from transformers import Qwen2Config

class OmniMultimodalLM(nn.Module):
    def __init__(self, llm_name: str, **kwargs):
        super().__init__()
        print(f"\n🧠 INTELLIGENT MODEL LOAD: {llm_name}")
        
        # 1. Analyze Config for Capabilities
        self.capabilities = {"vision": False, "audio": False}
        self.native_input_keys = {"vision": "images", "audio": "audios"} # Defaults for Qwen
        
        try:
            config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
            self.capabilities["vision"] = hasattr(config, "vision_config") or "vision" in str(config).lower()
            self.capabilities["audio"] = hasattr(config, "audio_config") or "audio" in str(config).lower()
            
            # Special check for Qwen-Omni series
            if config.architectures and "Omni" in config.architectures[0]:
                 self.capabilities["vision"] = True
                 self.capabilities["audio"] = True
                 
            print(f"  🔍 Detected Native Capabilities: {self.capabilities}")
            
        except Exception:
            print("  ⚠️ Only Basic Config Detected.")

        # 2. Attempt Native Load
        base_model = None
        native_success = False
        
        try:
            # Try loading as is (Native)
            base_model = AutoModelForCausalLM.from_pretrained(
                llm_name, 
                device_map="auto", 
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            native_success = True
            print("  ✅ Base Model Loaded Successfully (Native/Text).")
            
        except Exception as e:
            print(f"  ❌ Native Load Failed: {e}")
            print("  ⚠️ Fallback: Performing Architecture Transplantation (Omni -> Qwen2 Native)...")
            
            try:
                # TRANSPLANTATION: Create a healthy Qwen2 body for the weights
                original_config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
                
                # Helper to dig for config attributes in nested Omni structure
                def get_cfg_attr(cfg, attr, default=None):
                    # Check root
                    val = getattr(cfg, attr, None)
                    if val is not None: return val
                    # Check thinker_config.text_config (Omni standard)
                    if hasattr(cfg, "thinker_config") and hasattr(cfg.thinker_config, "text_config"):
                         val = getattr(cfg.thinker_config.text_config, attr, None)
                         if val is not None: return val
                    return default

                # Copy critical architectural genes
                vocab_size = get_cfg_attr(original_config, "vocab_size", 152064)
                hidden_size = get_cfg_attr(original_config, "hidden_size", 3584)
                
                compatible_config = Qwen2Config(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=get_cfg_attr(original_config, "intermediate_size", 18944),
                    num_hidden_layers=get_cfg_attr(original_config, "num_hidden_layers", 28),
                    num_attention_heads=get_cfg_attr(original_config, "num_attention_heads", 28),
                    num_key_value_heads=get_cfg_attr(original_config, "num_key_value_heads", 4),
                    max_position_embeddings=get_cfg_attr(original_config, "max_position_embeddings", 32768),
                    rms_norm_eps=get_cfg_attr(original_config, "rms_norm_eps", 1e-6),
                    tie_word_embeddings=False, 
                    torch_dtype="float16"
                )
                
                # CRITICAL: Preserve Quantization Config to avoid blowing up RAM (loading Int4 as FP16)
                if hasattr(original_config, "quantization_config"):
                    compatible_config.quantization_config = original_config.quantization_config
                    print(f"  💾 Preserved Quantization Config: {compatible_config.quantization_config.get('quant_method', 'unknown')}")
                
                print(f"  🧬 Synthesized Compatible Config: Qwen2Config (Vocab={vocab_size}, L={compatible_config.num_hidden_layers}, H={compatible_config.hidden_size})")
                
                # Load weights into the compatible shell
                # We rely on default permissive loading to ignore the extra "thinker/talker" keys
                base_model = AutoModelForCausalLM.from_pretrained(
                    llm_name, 
                    config=compatible_config,
                    device_map="auto", 
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Since we transplanted, we effectively stripped native capabilities
                self.capabilities["vision"] = False
                self.capabilities["audio"] = False
                print("  ✅ Transplantation Successful! Loaded as Text-Only Qwen2 Backbone.")
                
            except Exception as e2:
                print(f"  ❌ Transplantation Failed: {e2}")
                raise e2

        # 3. Determine Injections
        inject_vision = not self.capabilities["vision"]
        inject_audio = not self.capabilities["audio"]
        
        print(f"  🛠️  Final Architecture Plan:")
        print(f"      - Vision: Native={self.capabilities['vision']} -> Inject={inject_vision}")
        print(f"      - Audio:  Native={self.capabilities['audio']}  -> Inject={inject_audio}")

        # 4. Wrap
        self.wrapper = ModularMultimodalWrapper(
            base_model=base_model,
            inject_vision=inject_vision,
            inject_audio=inject_audio,
            llm_dim=base_model.config.hidden_size,
            **kwargs
        )
        
    def forward(self, *args, **kwargs):
        return self.wrapper(*args, **kwargs)

    def save_pretrained(self, path):
        self.wrapper.save_pretrained(path)
        
    def get_input_schema(self):
        """
        Returns the data schema required by this specific model instance.
        Used by the Trainer to prepare batches dynamically.
        """
        schema = {
            "requires_vision_input": True, # We always want vision data available
            "requires_audio_input": True,  # We always want audio data available
            "vision_key": "pixel_values" if getattr(self.wrapper, "inject_vision", False) else self.native_input_keys["vision"],
            "audio_key": "audio_features" if getattr(self.wrapper, "inject_audio", False) else self.native_input_keys["audio"],
            "text_key": "input_ids"
        }
        return schema
