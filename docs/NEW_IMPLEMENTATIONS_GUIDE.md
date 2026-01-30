# Nexus v6.1: New Implementations Guide

A comprehensive guide to the 4 major new implementations in Nexus v6.1:

1. Multimodal Embedding Injection
2. Video Generation Pipeline
3. Text-to-Speech Engine
4. Multi-Agent Orchestration

---

## Table of Contents

1. [Section 1: Multimodal Embedding Injection](#section-1-multimodal-embedding-injection)
2. [Section 2: Video Generation](#section-2-video-generation)
3. [Section 3: Text-to-Speech](#section-3-text-to-speech)
4. [Section 4: Multi-Agent Orchestration](#section-4-multi-agent-orchestration)

---

## Section 1: Multimodal Embedding Injection

### Architecture Overview

The Multimodal Embedding Injection system enables Nexus to seamlessly integrate embeddings from multiple modalities (vision, audio, video, text, and tools) into a unified representation space. This architecture is the foundation for true multimodal understanding and generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multimodal Fusion Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│  Vision    Audio    Video    Text    Tools                      │
│    │         │        │        │       │                        │
│    ▼         ▼        ▼        ▼       ▼                        │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                   │
│ │SigLIP│ │Whispr│ │VidMAE│ │Token ││ Tool │                   │
│ │ 512d │ │ 768d │ │1024d │ │ 768d │ │ 512d │                   │
│ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                   │
│    │         │        │        │       │                        │
│    ▼         ▼        ▼        ▼       ▼                        │
│ ┌──────────────────────────────────────────────────┐           │
│ │        NeuralArchitect Projection Layers          │           │
│ │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │           │
│ │  │512→ │ │768→ │ │1024→│ │768→ │ │512→ │        │           │
│ │  │2048 │ │2048 │ │2048 │ │2048 │ │2048 │        │           │
│ │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘        │           │
│ └──────────────────────────────────────────────────┘           │
│                      │                                           │
│                      ▼                                           │
│           ┌─────────────────────┐                               │
│           │  Modality Fusion    │                               │
│           │  (Cross-Attention)  │                               │
│           └─────────────────────┘                               │
│                      │                                           │
│                      ▼                                           │
│           ┌─────────────────────┐                               │
│           │  Unified Embedding  │                               │
│           │  (2048d → 4096d)    │                               │
│           └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### How Embedding Injection Works

The embedding injection process follows a three-stage pipeline:

#### Stage 1: Modality-Specific Encoding

Each modality has its dedicated encoder:

| Modality | Encoder | Output Dim | Use Case |
|----------|---------|------------|----------|
| Vision | SigLIP | 512 | Image understanding |
| Audio | Whisper | 768 | Speech recognition |
| Video | VideoMAE | 1024 | Video understanding |
| Text | Token Embedding | 768 | Language processing |
| Tools | Tool Adapter | 512 | Function calling |

#### Stage 2: Cross-Modal Projection

The `NeuralArchitect` class manages learnable projection layers that map modality-specific embeddings to a unified space:

```python
from src.multimodal.architect import NeuralArchitect

# Initialize the architect
architect = NeuralArchitect(
    target_dim=4096,  # Unified embedding dimension
    num_heads=16,     # Cross-attention heads
    dropout=0.1
)

# Project embeddings from different modalities
projected = architect.project_embeddings(
    vision_emb=vision_output,      # [batch, 196, 512]
    audio_emb=audio_output,        # [batch, 300, 768]
    text_emb=text_output,          # [batch, 77, 768]
    modality_types=['vision', 'audio', 'text']
)
```

#### Stage 3: Modality Fusion

The fusion layer combines all projected embeddings using cross-attention:

```python
# Fuse multiple modalities into unified representation
fused = architect.fuse_modalities(
    embeddings=projected,
    fusion_type='attention',  # or 'weighted_sum', 'concat'
    return_attention_weights=True
)
```

### API Reference

#### `NeuralArchitect`

The main class for managing multimodal embedding injection.

**Constructor:**

```python
NeuralArchitect(
    target_dim: int = 4096,
    num_heads: int = 16,
    dropout: float = 0.1,
    use_layer_norm: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dim` | int | 4096 | Dimension of unified embedding space |
| `num_heads` | int | 16 | Number of attention heads for fusion |
| `dropout` | float | 0.1 | Dropout rate for regularization |
| `use_layer_norm` | bool | True | Whether to use layer normalization |

**Methods:**

##### `project_embeddings()`

```python
def project_embeddings(
    self,
    **modality_embeddings: torch.Tensor
) -> Dict[str, torch.Tensor]
```

Projects embeddings from multiple modalities to the unified space.

**Parameters:**

- `**modality_embeddings`: Keyword arguments mapping modality names to tensors
  - Each tensor shape: `[batch_size, seq_len, modality_dim]`

**Returns:**

- Dictionary mapping modality names to projected embeddings
  - Each tensor shape: `[batch_size, seq_len, target_dim]`

**Example:**

```python
projected = architect.project_embeddings(
    vision=vision_emb,  # [batch, 196, 512] → [batch, 196, 4096]
    audio=audio_emb,    # [batch, 300, 768] → [batch, 300, 4096]
    text=text_emb       # [batch, 77, 768] → [batch, 77, 4096]
)
```

##### `fuse_modalities()`

```python
def fuse_modalities(
    self,
    embeddings: Dict[str, torch.Tensor],
    fusion_type: str = 'attention',
    return_attention_weights: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

Fuses multiple modality embeddings into a single representation.

**Parameters:**

- `embeddings`: Dictionary of projected embeddings
- `fusion_type`: Fusion strategy ('attention', 'weighted_sum', 'concat')
- `return_attention_weights`: Whether to return attention weights

**Returns:**

- Fused embedding tensor of shape `[batch_size, total_seq_len, target_dim]`
- Optionally attention weights if `return_attention_weights=True`

#### `NexusBridge`

Handles the connection between multimodal embeddings and the language model.

**Constructor:**

```python
NexusBridge(
    llm_dim: int = 4096,
    projection_dim: int = 2048,
    num_projection_layers: int = 2
)
```

**Methods:**

##### `inject_to_llm()`

```python
def inject_to_llm(
    self,
    multimodal_emb: torch.Tensor,
    text_tokens: torch.Tensor,
    injection_points: List[str] = ['input', 'hidden']
) -> torch.Tensor
```

Injects multimodal embeddings into the LLM at specified points.

### Usage Examples

#### Example 1: Vision-Language Fusion

```python
from src.multimodal.architect import NeuralArchitect, NexusBridge
from src.omni.loader import load_omni_model
import torch

# Load models
vision_encoder = load_omni_model("google/siglip-base-patch16-224", mode="vision")
llm = load_omni_model("nexus-student-2b", mode="thinker_only")

# Initialize multimodal components
architect = NeuralArchitect(target_dim=llm.config.hidden_size)
bridge = NexusBridge(llm_dim=llm.config.hidden_size)

# Process image
image = load_image("example.jpg")  # [3, 224, 224]
with torch.no_grad():
    vision_emb = vision_encoder(image.unsqueeze(0))

# Inject into LLM
projected = architect.project_embeddings(vision=vision_emb)
fused = architect.fuse_modalities(projected)

# Generate with multimodal context
output = bridge.inject_to_llm(fused, text_input_ids)
response = llm.generate(output)
```

#### Example 2: Audio-Text Fusion

```python
from src.multimodal.architect import NeuralArchitect
import torch

# Initialize
architect = NeuralArchitect(target_dim=4096)

# Load audio
audio = load_audio("speech.wav")  # [1, 16000]
with torch.no_grad():
    audio_emb = whisper_encoder(audio)  # [1, 300, 768]

# Process text query
text_tokens = tokenizer("What is being said in this audio?")
text_emb = llm.embed_tokens(text_tokens)  # [1, 10, 768]

# Fuse modalities
fused_emb = architect.project_and_fuse(
    audio=audio_emb,
    text=text_emb
)

# Generate response
response = llm.generate(inputs_embeds=fused_emb)
```

#### Example 3: Video Understanding

```python
from src.multimodal.architect import NeuralArchitect
from src.video.decoder import VideoDecoder

# Initialize
architect = NeuralArchitect(target_dim=4096)
decoder = VideoDecoder(model_id="stabilityai/stable-video-diffusion")

# Load video frames
video_frames = load_video("example.mp4", num_frames=16)  # [16, 3, 224, 224]

# Extract video embeddings
video_emb = decoder.encode_frames(video_frames)  # [1, 400, 1024]

# Create multimodal query
query_text = tokenizer("Describe what happens in this video")
query_emb = llm.embed_tokens(query_tokens)

# Inject and generate
fused = architect.project_and_fuse(
    video=video_emb,
    text=query_emb,
    fusion_weights={'video': 0.6, 'text': 0.4}
)

description = llm.generate(inputs_embeds=fused)
```

#### Example 4: Tool Fusion

```python
from src.multimodal.architect import NeuralArchitect
from src.tools.adapter import ToolAdapter

# Initialize components
architect = NeuralArchitect(target_dim=4096)
tool_adapter = ToolAdapter(available_tools=['search', 'calculator', 'database'])

# User query
query = "What's the weather in Tokyo and calculate 15% tip on $85?"
query_emb = llm.embed_tokens(tokenizer(query))

# Get tool embeddings
tool_descriptions = tool_adapter.get_tool_embeddings(query)
tool_emb = tool_adapter.encode_tools(tool_descriptions)  # [1, 3, 512]

# Fuse with query
fused = architect.project_and_fuse(
    text=query_emb,
    tools=tool_emb,
    fusion_type='attention_with_bias',
    bias_weights={'tools': 1.5}  # Emphasize tool selection
)

# Generate with tool awareness
tool_calls = llm.generate(inputs_embeds=fused, max_new_tokens=100)
```

### Performance Characteristics

Based on benchmark results:

| Operation | 512→768 | 768→1024 | 1024→2048 | 2048→4096 |
|-----------|---------|----------|-----------|-----------|
| Projection Latency | 0.5ms | 0.8ms | 1.5ms | 3.0ms |
| Memory (per batch) | 50MB | 100MB | 400MB | 1000MB |
| Throughput (FPS) | 1000 | 800 | 600 | 300 |

**Optimization Tips:**

1. **Batch Processing**: Process multiple samples together to maximize GPU utilization
2. **Mixed Precision**: Use FP16/BF16 for 2x speedup with minimal accuracy loss
3. **Gradient Checkpointing**: Enable for large models to trade compute for memory
4. **Caching**: Cache projected embeddings for repeated inputs

### Troubleshooting

#### Issue: Out of Memory During Projection

**Symptoms:**

- CUDA out of memory errors during multimodal processing
- System slowdown during fusion

**Solutions:**

```python
# 1. Reduce batch size
architect.project_embeddings(
    vision=vision_emb[:4]  # Process 4 at a time instead of 16
)

# 2. Enable gradient checkpointing
architect.enable_gradient_checkpointing()

# 3. Use VAE slicing for video
video_decoder.enable_vae_slicing()

# 4. Process modalities sequentially
for modality in ['vision', 'audio', 'text']:
    projected[modality] = architect.project_single(
        embeddings[modality], 
        modality_type=modality
    )
```

#### Issue: Misaligned Embedding Dimensions

**Symptoms:**

- `RuntimeError: size mismatch` during projection
- Unexpected tensor shapes in fusion

**Solutions:**

```python
# Check dimensions before projection
print(f"Vision: {vision_emb.shape}")  # Should be [batch, seq, 512]
print(f"Audio: {audio_emb.shape}")    # Should be [batch, seq, 768]

# Use dimension adapter
from src.multimodal.utils import DimensionAdapter

adapter = DimensionAdapter(
    input_dims={'vision': 512, 'audio': 768},
    output_dim=4096
)

# Normalize dimensions
normalized = adapter.normalize(embeddings)
projected = architect.project_embeddings(**normalized)
```

#### Issue: Slow Fusion with Many Modalities

**Symptoms:**

- Fusion takes >100ms with 4+ modalities
- Low throughput in production

**Solutions:**

```python
# Use weighted sum for faster fusion
fused = architect.fuse_modalities(
    embeddings=projected,
    fusion_type='weighted_sum',  # Faster than attention
    weights={'vision': 0.3, 'audio': 0.3, 'text': 0.4}
)

# Or use hierarchical fusion
fused = architect.fuse_hierarchical(
    embeddings=projected,
    fusion_order=[['vision', 'text'], ['audio']]  # Fuse in stages
)
```

---

## Section 2: Video Generation

### Stable Video Diffusion Integration

Nexus integrates Stable Video Diffusion (SVD) for high-quality video generation from images and text prompts.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Video Generation Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Input     │───▶│   VAE       │───▶│   UNet      │         │
│  │  (Image/    │    │  Encoder    │    │  Denoising  │         │
│  │   Text)     │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                │
│  ┌─────────────┐    ┌─────────────┐           │                │
│  │   Output    │◀───│   VAE       │◀──────────┘                │
│  │   Video     │    │  Decoder    │                            │
│  │  (MP4/GIF)  │    │             │                            │
│  └─────────────┘    └─────────────┘                            │
│                                                                  │
│  Optimization Options:                                           │
│  - VAE Slicing (memory efficient)                                │
│  - VAE Tiling (high resolution)                                  │
│  - Frame Batching                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Setup and Configuration

#### Installation

```bash
# Install video dependencies
pip install -r requirements-video.txt

# Download SVD model
python -c "from src.video.decoder import VideoDecoder; VideoDecoder.download_model()"
```

#### Configuration File

Create `config/video_generation.yaml`:

```yaml
video_generation:
  model_id: "stabilityai/stable-video-diffusion-img2vid-xt"
  variant: "fp16"
  torch_dtype: "float16"
  
  # VAE Optimization
  vae:
    enable_slicing: true
    enable_tiling: true
    tile_sample_min_size: 512
  
  # Generation Parameters
  generation:
    num_frames: 16
    fps: 8
    motion_bucket_id: 127
    noise_aug_strength: 0.02
  
  # Memory Management
  memory:
    cpu_offload: true
    enable_vae_slicing: true
    max_batch_size: 1
```

### Generation Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_frames` | int | 16 | Number of video frames to generate |
| `fps` | int | 8 | Frames per second in output video |
| `motion_bucket_id` | int | 127 | Controls motion intensity (0-255) |
| `noise_aug_strength` | float | 0.02 | Noise augmentation for diversity |
| `min_guidance_scale` | float | 1.0 | Minimum guidance scale |
| `max_guidance_scale` | float | 3.0 | Maximum guidance scale |
| `num_inference_steps` | int | 50 | Denoising steps (more = better quality) |
| `decode_chunk_size` | int | 4 | VAE decoding chunks (memory control) |

**Motion Bucket Guide:**

- 0-50: Minimal motion (static camera)
- 51-127: Moderate motion (handheld feel)
- 128-200: High motion (action scenes)
- 201-255: Extreme motion (sports/action)

### API Reference

#### `VideoDecoder`

Main class for video generation and decoding.

**Constructor:**

```python
VideoDecoder(
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | "stabilityai/stable-video-diffusion-img2vid-xt" | HuggingFace model ID |
| `device` | str | "auto" | Device to use ("cuda", "cpu", "auto") |
| `dtype` | torch.dtype | torch.float16 | Data type for inference |
| `enable_vae_slicing` | bool | True | Enable VAE slicing for memory efficiency |
| `enable_vae_tiling` | bool | False | Enable VAE tiling for high resolution |

**Methods:**

##### `generate_from_image()`

```python
def generate_from_image(
    self,
    image: Union[str, Image.Image, torch.Tensor],
    num_frames: int = 16,
    fps: int = 8,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    num_inference_steps: int = 50,
    output_path: Optional[str] = None
) -> Union[torch.Tensor, str]
```

Generates video from an input image.

**Parameters:**

- `image`: Input image (path, PIL Image, or tensor)
- `num_frames`: Number of frames to generate
- `fps`: Frames per second
- `motion_bucket_id`: Motion intensity (0-255)
- `noise_aug_strength`: Noise augmentation strength
- `num_inference_steps`: Number of denoising steps
- `output_path`: Optional path to save video

**Returns:**

- Video tensor of shape `[num_frames, channels, height, width]` or path if `output_path` specified

##### `generate_from_text()`

```python
def generate_from_text(
    self,
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_frames: int = 16,
    fps: int = 8,
    output_path: Optional[str] = None
) -> Union[torch.Tensor, str]
```

Generates video from a text prompt (requires text-to-video model).

##### `export_video()`

```python
def export_video(
    self,
    video: torch.Tensor,
    output_path: str,
    fps: int = 8,
    format: str = "mp4",
    quality: str = "high",
    codec: Optional[str] = None
) -> str
```

Exports video tensor to various formats.

**Supported Formats:**

- `mp4`: H.264/HEVC codec, best compatibility
- `webm`: VP9 codec, web-optimized
- `gif`: Animation GIF, limited to 256 colors
- `avi`: Legacy format

**Quality Presets:**

- `low`: CRF 28, ultrafast preset
- `medium`: CRF 23, medium preset
- `high`: CRF 18, slow preset
- `lossless`: CRF 0, veryslow preset

### Usage Examples

#### Example 1: Image-to-Video

```python
from src.video.decoder import VideoDecoder
from PIL import Image

# Initialize decoder
decoder = VideoDecoder(
    model_id="stabilityai/stable-video-diffusion-img2vid-xt",
    enable_vae_slicing=True
)

# Load image
image = Image.open("input_image.jpg")

# Generate video with moderate motion
video = decoder.generate_from_image(
    image=image,
    num_frames=16,
    fps=8,
    motion_bucket_id=127,  # Moderate motion
    noise_aug_strength=0.02,
    num_inference_steps=50,
    output_path="output_video.mp4"
)

print(f"Video saved to: {video}")
```

#### Example 2: Text-to-Video

```python
from src.video.decoder import VideoDecoder

# Initialize text-to-video model
decoder = VideoDecoder(
    model_id="damo-vilab/text-to-video-ms-1.7b"
)

# Generate from text prompt
video = decoder.generate_from_text(
    prompt="A serene lake surrounded by mountains at sunset, gentle ripples on water",
    width=512,
    height=512,
    num_frames=16,
    fps=8,
    output_path="sunset_lake.mp4"
)
```

#### Example 3: Batch Processing with Memory Optimization

```python
from src.video.decoder import VideoDecoder
import torch

# Initialize with optimizations
decoder = VideoDecoder(
    enable_vae_slicing=True,  # Essential for large batches
    enable_vae_tiling=True,   # Enable for 1024x1024+
)

# Process multiple images
images = ["scene1.jpg", "scene2.jpg", "scene3.jpg"]

for i, img_path in enumerate(images):
    # Clear cache between generations
    if i > 0:
        torch.cuda.empty_cache()
    
    video = decoder.generate_from_image(
        image=img_path,
        num_frames=16,
        motion_bucket_id=150,  # Higher motion
        output_path=f"output_{i}.mp4"
    )
    print(f"Generated: {video}")
```

#### Example 4: Custom Export Settings

```python
from src.video.decoder import VideoDecoder

# Generate video
decoder = VideoDecoder()
video_tensor = decoder.generate_from_image("input.jpg", num_frames=24)

# Export with different quality settings
formats = [
    ("output_high.mp4", "high", "h264"),
    ("output_web.mp4", "low", "h264"),
    ("output_prores.mov", "lossless", "prores"),
]

for path, quality, codec in formats:
    decoder.export_video(
        video=video_tensor,
        output_path=path,
        fps=24,
        quality=quality,
        codec=codec
    )
```

### Memory Optimization Tips

#### For 16GB VRAM (RTX 4080/4090)

```python
decoder = VideoDecoder(
    enable_vae_slicing=True,
    enable_vae_tiling=False,
    dtype=torch.float16
)

# Max settings: 512x512, 16 frames
video = decoder.generate_from_image(
    image=input_img,
    num_frames=16,
    num_inference_steps=50
)
```

#### For 8GB VRAM (RTX 3070/4060)

```python
decoder = VideoDecoder(
    enable_vae_slicing=True,
    enable_vae_tiling=True,  # Required for memory
    dtype=torch.float16
)

# Conservative settings
video = decoder.generate_from_image(
    image=input_img,
    num_frames=8,  # Fewer frames
    num_inference_steps=25  # Fewer steps
)
```

#### For CPU Inference

```python
decoder = VideoDecoder(
    device="cpu",
    dtype=torch.float32,  # CPU doesn't support float16 well
    enable_vae_slicing=True
)

# Very conservative settings
video = decoder.generate_from_image(
    image=input_img,
    num_frames=8,
    num_inference_steps=25,
    height=256,  # Lower resolution
    width=256
)
```

### Performance Benchmarks

Based on [test_video_decoder_benchmark.py](../benchmarks/test_video_decoder_benchmark.py):

| Resolution | Frames | Time | FPS | Memory |
|------------|--------|------|-----|--------|
| 256x256 | 16 | 2.0s | 8 | 2GB |
| 512x512 | 16 | 8.0s | 2 | 6GB |
| 1024x1024 | 16 | 30.0s | 0.5 | 16GB |

**VAE Optimization Impact:**

| Resolution | No Opt | With Slicing | With Tiling |
|------------|--------|--------------|-------------|
| 256x256 | 100% | 110% | 105% |
| 512x512 | 100% | 95% | 90% |
| 1024x1024 | 100% | 85% | 80% |

---

## Section 3: Text-to-Speech

### Coqui TTS Integration

Nexus integrates Coqui TTS for high-quality speech synthesis with voice cloning capabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TTS Engine Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │    Text     │───▶│  Acoustic   │───▶│   Vocoder   │         │
│  │   Input     │    │    Model    │    │  (HiFiGAN)  │         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘         │
│                            │                   │                │
│                     ┌──────┴──────┐            │                │
│                     │   Speaker   │            │                │
│                     │  Embedding  │────────────┘                │
│                     │  (Optional) │                             │
│                     └─────────────┘                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Output Formats: WAV, MP3, OGG, FLAC                    │    │
│  │  Sample Rates: 16kHz, 22.05kHz, 44.1kHz, 48kHz         │    │
│  │  Bit Rates: 128kbps - 320kbps (MP3)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Setup and Voice Cloning

#### Installation

```bash
# Install TTS dependencies
pip install -r requirements-tts.txt

# Download default model
python -c "from src.tts.engine import TTSEngine; TTSEngine.download_model()"
```

#### Model Options

| Model | Quality | Speed | VRAM | Best For |
|-------|---------|-------|------|----------|
| `tts_models/en/vctk/vits` | ⭐⭐⭐ | Fast | 2GB | General use |
| `tts_models/multilingual/multi-dataset/xtts_v2` | ⭐⭐⭐⭐⭐ | Medium | 4GB | Voice cloning |
| `tts_models/en/ljspeech/tacotron2-DDC` | ⭐⭐⭐⭐ | Medium | 3GB | Narration |
| `tts_models/en/ljspeech/glow-tts` | ⭐⭐⭐ | Fast | 2GB | Real-time |

#### Voice Cloning Setup

```python
from src.tts.engine import TTSEngine

# Initialize XTTS for voice cloning
engine = TTSEngine(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
)

# Clone voice from reference audio
engine.clone_voice(
    reference_audio="reference_speaker.wav",  # 3-30 seconds
    speaker_name="custom_speaker"
)

# Synthesize with cloned voice
engine.synthesize(
    text="Hello, this is my cloned voice speaking.",
    speaker="custom_speaker",
    output_path="cloned_output.wav"
)
```

### Streaming vs Batch Synthesis

#### Batch Synthesis

Best for offline content generation:

```python
from src.tts.engine import TTSEngine

engine = TTSEngine()

# Process entire text at once
audio = engine.synthesize(
    text="This is a long text that will be processed entirely before any audio is returned.",
    output_path="batch_output.wav"
)
```

**Characteristics:**

- Higher quality (full context available)
- Lower latency for short texts
- Better prosody and intonation
- Requires complete text upfront

#### Streaming Synthesis

Best for real-time applications:

```python
from src.tts.engine import TTSStreamer

streamer = TTSStreamer(
    model_name="tts_models/en/vctk/vits",
    chunk_size=50  # phonemes per chunk
)

# Start streaming
text_stream = "This is a long text stream..."
for audio_chunk in streamer.synthesize_stream(text_stream):
    # Play chunk immediately
    play_audio(audio_chunk)
```

**Characteristics:**

- Lower time-to-first-audio
- Suitable for real-time applications
- Slightly lower quality
- Can handle infinite-length text

### API Reference

#### `TTSEngine`

Main class for text-to-speech synthesis.

**Constructor:**

```python
TTSEngine(
    model_name: str = "tts_models/en/vctk/vits",
    device: str = "auto",
    use_gpu: bool = True,
    cache_dir: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "tts_models/en/vctk/vits" | Coqui TTS model name |
| `device` | str | "auto" | Device ("cuda", "cpu", "auto") |
| `use_gpu` | bool | True | Whether to use GPU |
| `cache_dir` | str | None | Directory to cache models |

**Methods:**

##### `synthesize()`

```python
def synthesize(
    self,
    text: str,
    speaker: Optional[str] = None,
    language: str = "en",
    output_path: Optional[str] = None,
    format: str = "wav",
    sample_rate: int = 22050,
    speed: float = 1.0
) -> Union[np.ndarray, str]
```

Synthesizes speech from text.

**Parameters:**

- `text`: Text to synthesize
- `speaker`: Speaker ID (for multi-speaker models)
- `language`: Language code ("en", "zh", "ja", etc.)
- `output_path`: Path to save audio (returns array if None)
- `format`: Output format ("wav", "mp3", "ogg", "flac")
- `sample_rate`: Output sample rate
- `speed`: Speech speed multiplier (0.5 - 2.0)

**Returns:**

- Audio array or path to saved file

##### `clone_voice()`

```python
def clone_voice(
    self,
    reference_audio: Union[str, Path],
    speaker_name: str,
    reference_text: Optional[str] = None
) -> bool
```

Clones a voice from reference audio.

##### `list_speakers()`

```python
def list_speakers(self) -> List[str]
```

Returns available speaker IDs for multi-speaker models.

#### `TTSStreamer`

For real-time streaming synthesis.

**Constructor:**

```python
TTSStreamer(
    model_name: str = "tts_models/en/vctk/vits",
    chunk_size: int = 50,
    buffer_size: int = 3,
    device: str = "auto"
)
```

**Methods:**

##### `synthesize_stream()`

```python
def synthesize_stream(
    self,
    text: str,
    speaker: Optional[str] = None,
    language: str = "en"
) -> Generator[np.ndarray, None, None]
```

Yields audio chunks as they are synthesized.

#### `VoiceConfig`

Configuration class for voice characteristics.

```python
@dataclass
class VoiceConfig:
    pitch_shift: float = 0.0  # -12 to +12 semitones
    speed: float = 1.0        # 0.5 to 2.0
    volume: float = 1.0       # 0.0 to 2.0
    emphasis: str = "normal"  # "low", "normal", "high"
    
    # Advanced settings
    top_k: int = 50
    top_p: float = 0.8
    temperature: float = 0.7
```

### Usage Examples

#### Example 1: Basic Synthesis

```python
from src.tts.engine import TTSEngine

# Initialize engine
engine = TTSEngine()

# Synthesize simple text
audio = engine.synthesize(
    text="Hello, welcome to Nexus text-to-speech system.",
    output_path="welcome.wav"
)

# Synthesize with different speaker
audio = engine.synthesize(
    text="This is a different voice.",
    speaker="p226",  # VCTK speaker ID
    output_path="different_speaker.wav"
)
```

#### Example 2: Voice Cloning

```python
from src.tts.engine import TTSEngine

# Initialize XTTS model
engine = TTSEngine(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
)

# Clone voice from sample
engine.clone_voice(
    reference_audio="my_voice_sample.wav",
    speaker_name="my_voice"
)

# Synthesize with cloned voice
sentences = [
    "This is my voice speaking.",
    "Voice cloning technology is amazing.",
    "I can now generate speech in my own voice."
]

for i, text in enumerate(sentences):
    engine.synthesize(
        text=text,
        speaker="my_voice",
        output_path=f"cloned_{i}.wav"
    )
```

#### Example 3: Streaming Synthesis

```python
from src.tts.engine import TTSStreamer
import sounddevice as sd

# Initialize streamer
streamer = TTSStreamer(chunk_size=30)

# Real-time synthesis and playback
text = "This is a long text that will be synthesized in chunks and played in real time."

for audio_chunk in streamer.synthesize_stream(text):
    sd.play(audio_chunk, samplerate=22050)
    sd.wait()
```

#### Example 4: Multi-Language Synthesis

```python
from src.tts.engine import TTSEngine

engine = TTSEngine(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2"
)

# Synthesize in different languages
texts = {
    "en": "Hello, how are you?",
    "zh": "你好，你好吗？",
    "ja": "こんにちは、お元気ですか？",
    "es": "¡Hola! ¿Cómo estás?",
    "de": "Hallo, wie geht es dir?",
}

for lang, text in texts.items():
    engine.synthesize(
        text=text,
        language=lang,
        output_path=f"greeting_{lang}.wav"
    )
```

### Language Support and Performance

#### Supported Languages

| Language | Code | Model | Quality | Notes |
|----------|------|-------|---------|-------|
| English | en | All models | ⭐⭐⭐⭐⭐ | Best support |
| Chinese | zh | XTTS | ⭐⭐⭐⭐ | Good support |
| Japanese | ja | XTTS | ⭐⭐⭐⭐ | Good support |
| Spanish | es | XTTS | ⭐⭐⭐⭐ | Good support |
| German | de | XTTS | ⭐⭐⭐⭐ | Good support |
| French | fr | XTTS | ⭐⭐⭐⭐ | Good support |
| Italian | it | XTTS | ⭐⭐⭐ | Limited |
| Portuguese | pt | XTTS | ⭐⭐⭐ | Limited |
| Dutch | nl | XTTS | ⭐⭐⭐ | Limited |

#### Performance Benchmarks

Based on [test_tts_benchmark.py](../benchmarks/test_tts_benchmark.py):

| Text Length | Characters | Latency | RTF |
|-------------|------------|---------|-----|
| Short | <50 | 50ms | 0.05x |
| Medium | ~200 | 200ms | 0.1x |
| Long | ~1000 | 800ms | 0.2x |

**Voice Cloning Setup:**

| Reference Duration | Setup Time |
|-------------------|------------|
| 3 seconds | 100ms |
| 10 seconds | 300ms |
| 30 seconds | 800ms |

**Streaming Throughput:**

| Chunk Size | Throughput |
|------------|------------|
| 1 phoneme | 50 chunks/sec |
| 5 phonemes | 20 chunks/sec |
| 10 phonemes | 10 chunks/sec |

---

## Section 4: Multi-Agent Orchestration

### System Architecture

The Multi-Agent Orchestration system enables collaborative AI agents to work together on complex software development tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                 Multi-Agent Orchestration System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                 │
│  │   User      │                                                 │
│  │   Request   │                                                 │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              AgentOrchestrator                          │    │
│  │         (Workflow Manager & Coordinator)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│    ┌────┴────┬──────────┬──────────┬──────────┐                 │
│    ▼         ▼          ▼          ▼          ▼                 │
│ ┌──────┐ ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│ │Planning│ │Backend │  │Frontend│  │Review  │  │Testing │               │
│ │ Agent  │ │ Agent  │  │ Agent  │  │ Agent  │  │ Agent  │               │
│ │        │ │        │  │        │  │        │  │        │               │
│ │- Arch  │ │- API   │  │- UI    │  │- Code  │  │- Unit  │               │
│ │- Design│ │- DB    │  │- React │  │ Review │  │- E2E   │               │
│ │- Specs │ │- Logic │  │- CSS   │  │- Sec.  │  │- Perf  │               │
│ └──────┘ └──────┘  └──────┘  └──────┘  └──────┘               │
│    │         │          │          │          │                 │
│    └─────────┴──────────┴──────────┴──────────┘                 │
│                      │                                           │
│                      ▼                                           │
│           ┌─────────────────────┐                               │
│           │   Context Store     │                               │
│           │   (Shared Memory)   │                               │
│           └─────────────────────┘                               │
│                                                                  │
│  Features:                                                       │
│  - Retry mechanisms with exponential backoff                     │
│  - Concurrent execution with ThreadPool                          │
│  - Context passing between agents                                │
│  - Workflow checkpointing                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Types and Roles

#### 1. Planning Agent

**Purpose:** Architecture design and task decomposition

**Responsibilities:**

- Analyze user requirements
- Design system architecture
- Create implementation plans
- Decompose tasks for other agents

**Output:**

```json
{
  "architecture": "microservices",
  "components": ["api_gateway", "auth_service", "database"],
  "plan": [
    {"step": 1, "agent": "backend", "task": "implement_auth"},
    {"step": 2, "agent": "frontend", "task": "implement_login"}
  ]
}
```

#### 2. Backend Agent

**Purpose:** Server-side code generation

**Responsibilities:**

- API endpoint implementation
- Database schema design
- Business logic implementation
- Authentication/authorization

**Supported Frameworks:**

- FastAPI
- Flask
- Django
- Express.js
- Spring Boot

#### 3. Frontend Agent

**Purpose:** Client-side code generation

**Responsibilities:**

- UI component development
- State management
- API integration
- Responsive design

**Supported Frameworks:**

- React
- Vue.js
- Angular
- Svelte
- Vanilla JS

#### 4. Review Agent

**Purpose:** Code quality assurance

**Responsibilities:**

- Static analysis
- Security audit
- Performance review
- Best practices verification

**Checks:**

- Code style compliance
- Security vulnerabilities
- Performance bottlenecks
- Documentation completeness

#### 5. Testing Agent

**Purpose:** Test generation and execution

**Responsibilities:**

- Unit test generation
- Integration test creation
- E2E test scenarios
- Performance benchmarks

### Workflow Customization

#### Defining Custom Workflows

```python
from src.agents.orchestrator import AgentOrchestrator, Workflow

# Define custom workflow
workflow = Workflow(
    name="api_development",
    steps=[
        {
            "agent": "planning",
            "action": "design_api",
            "output_keys": ["api_spec", "schema"]
        },
        {
            "agent": "backend",
            "action": "implement_api",
            "input_keys": ["api_spec", "schema"],
            "output_keys": ["backend_code"]
        },
        {
            "agent": "review",
            "action": "review_code",
            "input_keys": ["backend_code"],
            "output_keys": ["review_report"],
            "condition": "review_report.score > 0.8"  # Conditional flow
        }
    ]
)

# Execute workflow
orchestrator = AgentOrchestrator()
result = orchestrator.execute_workflow(
    workflow=workflow,
    initial_context={"requirement": "Create a user API"}
)
```

#### Parallel Execution

```python
from src.agents.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Execute agents in parallel
results = orchestrator.execute_parallel(
    tasks=[
        ("backend", {"task": "implement_auth"}),
        ("frontend", {"task": "implement_login_ui"}),
        ("backend", {"task": "implement_user_crud"})
    ],
    max_workers=3
)
```

#### Retry Configuration

```python
from src.agents.orchestrator import AgentOrchestrator, RetryConfig

retry_config = RetryConfig(
    max_retries=3,
    backoff_strategy="exponential",  # or "fixed", "linear"
    initial_delay=1.0,
    max_delay=30.0,
    retry_on=["timeout", "rate_limit"]
)

orchestrator = AgentOrchestrator(retry_config=retry_config)
```

### API Reference

#### `AgentOrchestrator`

Main coordinator for multi-agent workflows.

**Constructor:**

```python
AgentOrchestrator(
    llm_client: Optional[LLMClient] = None,
    retry_config: Optional[RetryConfig] = None,
    context_store: Optional[ContextStore] = None,
    max_workers: int = 4
)
```

**Methods:**

##### `execute_workflow()`

```python
def execute_workflow(
    self,
    workflow: Workflow,
    initial_context: Dict[str, Any],
    checkpoint_dir: Optional[str] = None
) -> WorkflowResult
```

Executes a defined workflow across multiple agents.

**Parameters:**

- `workflow`: Workflow definition
- `initial_context`: Starting context data
- `checkpoint_dir`: Directory to save checkpoints

**Returns:**

- `WorkflowResult` containing final context and execution metadata

##### `execute_parallel()`

```python
def execute_parallel(
    self,
    tasks: List[Tuple[str, Dict[str, Any]]],
    max_workers: Optional[int] = None
) -> List[AgentResult]
```

Executes multiple tasks concurrently.

##### `register_agent()`

```python
def register_agent(
    self,
    name: str,
    agent: BaseAgent,
    priority: int = 0
) -> None
```

Registers a custom agent with the orchestrator.

#### `PlanningAgent`

```python
PlanningAgent(
    llm_client: LLMClient,
    tools: Optional[List[Tool]] = None,
    max_iterations: int = 5
)

# Methods
def plan(self, requirement: str) -> Plan
def decompose(self, task: str) -> List[Subtask]
def estimate_effort(self, plan: Plan) -> Estimate
```

#### `BackendAgent`

```python
BackendAgent(
    llm_client: LLMClient,
    framework: str = "fastapi",
    database: str = "postgresql",
    include_tests: bool = True
)

# Methods
def generate_api(self, spec: APISpec) -> Code
def generate_models(self, schema: Schema) -> Code
def implement_logic(self, requirements: str) -> Code
```

#### `FrontendAgent`

```python
FrontendAgent(
    llm_client: LLMClient,
    framework: str = "react",
    styling: str = "tailwind",
    include_responsive: bool = True
)

# Methods
def generate_component(self, design: Design) -> Code
def implement_page(self, layout: Layout) -> Code
def integrate_api(self, endpoints: List[Endpoint]) -> Code
```

#### `ReviewAgent`

```python
ReviewAgent(
    llm_client: LLMClient,
    check_security: bool = True,
    check_performance: bool = True,
    check_style: bool = True
)

# Methods
def review_code(self, code: Code) -> ReviewReport
def check_security(self, code: Code) -> SecurityReport
def suggest_improvements(self, code: Code) -> List[Suggestion]
```

### Usage Examples

#### Example 1: Full Development Workflow

```python
from src.agents.orchestrator import AgentOrchestrator
from src.agents.types import PlanningAgent, BackendAgent, FrontendAgent

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# Register agents
orchestrator.register_agent("planning", PlanningAgent())
orchestrator.register_agent("backend", BackendAgent(framework="fastapi"))
orchestrator.register_agent("frontend", FrontendAgent(framework="react"))

# Define requirement
requirement = """
Create a task management application with:
- User authentication
- CRUD operations for tasks
- Due date tracking
- Priority levels
"""

# Execute full workflow
result = orchestrator.execute_workflow(
    workflow={
        "steps": [
            {"agent": "planning", "action": "design_system"},
            {"agent": "backend", "action": "implement_api"},
            {"agent": "frontend", "action": "implement_ui"},
            {"agent": "review", "action": "review_all"}
        ]
    },
    initial_context={"requirement": requirement}
)

print(f"Generated files: {result.outputs}")
print(f"Quality score: {result.metadata['quality_score']}")
```

#### Example 2: Custom Agent Implementation

```python
from src.agents.base import BaseAgent
from typing import Dict, Any

class DevOpsAgent(BaseAgent):
    """Custom agent for DevOps tasks."""
    
    def __init__(self):
        super().__init__(name="devops")
        self.supported_actions = ["create_dockerfile", "setup_ci_cd", "deploy"]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        action = input_data.get("action")
        
        if action == "create_dockerfile":
            return self._create_dockerfile(input_data)
        elif action == "setup_ci_cd":
            return self._setup_cicd(input_data)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def _create_dockerfile(self, data: Dict) -> Dict:
        app_type = data.get("app_type", "python")
        
        dockerfile = f"""
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
        return {"dockerfile": dockerfile}

# Register custom agent
orchestrator.register_agent("devops", DevOpsAgent())

# Use in workflow
result = orchestrator.execute_workflow(
    workflow={
        "steps": [
            {"agent": "backend", "action": "implement_api"},
            {"agent": "devops", "action": "create_dockerfile", 
             "input_map": {"app_type": "backend"}}
        ]
    },
    initial_context={}
)
```

#### Example 3: Integration with OmniModelLoader

```python
from src.agents.orchestrator import AgentOrchestrator
from src.agents.types import PlanningAgent, BackendAgent
from src.omni.loader import load_omni_model

# Load specialized LLM for code generation
llm = load_omni_model("deepseek-coder-33b", mode="thinker_only")

# Create agents with loaded LLM
planning_agent = PlanningAgent(llm_client=llm)
backend_agent = BackendAgent(
    llm_client=llm,
    framework="fastapi"
)

# Initialize orchestrator
orchestrator = AgentOrchestrator()
orchestrator.register_agent("planning", planning_agent)
orchestrator.register_agent("backend", backend_agent)

# Execute with powerful LLM
result = orchestrator.execute_workflow(
    workflow={
        "steps": [
            {"agent": "planning", "action": "design_architecture"},
            {"agent": "backend", "action": "implement_microservices"}
        ]
    },
    initial_context={
        "requirement": "Build a scalable e-commerce platform"
    }
)
```

### Performance Benchmarks

Based on [test_multi_agent_benchmark.py](../benchmarks/test_multi_agent_benchmark.py):

| Agent Type | Init Time | Response Time |
|------------|-----------|---------------|
| Planning | 50ms | 100-1500ms |
| Backend | 30ms | 100-2000ms |
| Frontend | 30ms | 100-2000ms |
| Review | 20ms | 50-500ms |

**Workflow Execution:**

| Workflow | Steps | Time |
|----------|-------|------|
| Simple | 2 | 150ms |
| Standard | 3 | 400ms |
| Complex | 6 | 1000ms |

**Concurrent Execution:**

| Concurrent Agents | Throughput |
|-------------------|------------|
| 2 | 66 ops/sec |
| 4 | 114 ops/sec |
| 8 | 178 ops/sec |
| 16 | 246 ops/sec |

---

## Summary

This guide covered the four major new implementations in Nexus v6.1:

1. **Multimodal Embedding Injection**: Enables unified representation of vision, audio, video, text, and tools through projection layers and cross-modal fusion.

2. **Video Generation**: Integrates Stable Video Diffusion for high-quality video generation with memory-efficient VAE optimizations.

3. **Text-to-Speech**: Provides high-quality speech synthesis with voice cloning, streaming support, and multi-language capabilities.

4. **Multi-Agent Orchestration**: Coordinates specialized AI agents for collaborative software development with workflow customization and parallel execution.

Each implementation includes comprehensive benchmarking, optimization strategies, and troubleshooting guidance. For more details, refer to the individual implementation files and benchmark results.

---

*Document Version: 1.0*
*Last Updated: 2026-01-30*
*Maintainer: Nexus Documentation Team*
