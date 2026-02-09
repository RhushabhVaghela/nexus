# Multimodal Module

The `multimodal` module provides encoding, decoding, and processing capabilities for vision, audio, and video modalities in the Nexus ecosystem. It implements state-of-the-art multimodal processing with support for repetition-based enhancement (Paper 2512.14982).

## Overview

This module enables Nexus to understand and generate content across multiple modalities:

- **Vision**: Image understanding and processing (SigLIP 2, CLIP)
- **Audio**: Speech recognition and audio processing (Whisper V3 Turbo, Wav2Vec2)
- **Video**: Video understanding and generation (Stable Video Diffusion)
- **Text**: Cross-modal alignment with text

## Module Structure

```
multimodal/
├── encoders.py              # Multimodal encoders with repetition support
├── decoders.py              # Content decoders for image, audio, video
├── processors.py            # Multimodal data processors
├── model.py                 # Core multimodal model implementations
├── distillation.py          # Knowledge distillation for multimodal models
├── tools.py                 # Multimodal tool utilities
├── reasoning.py             # Multimodal reasoning components
├── connectors/              # Connector modules
│   └── dfm.py              # Domain fusion module
├── datasets/                # Dataset loaders
│   ├── emm1_loader.py      # EMM1 dataset loader
│   └── unified_loader.py   # Unified multimodal loader
└── tests/                   # Module-specific tests
    └── test_encoder_decoder_shapes.py
```

## Encoders

### VisionEncoder

Encodes images into embeddings with optional repetition support.

```python
from src.multimodal.encoders import VisionEncoder

# Initialize encoder
encoder = VisionEncoder(
    model_name="openai/clip-vit-base-patch32",
    embedding_dim=768,
    use_repetition_features=True
)

# Encode images
output = encoder.encode(
    images=["path/to/image.jpg"],
    apply_repetition=True,
    repetition_factor=2
)

print(output.embeddings.shape)  # torch.Size([2, 768])
```

### AudioEncoder

Encodes audio into embeddings with repetition support.

```python
from src.multimodal.encoders import AudioEncoder

# Initialize encoder
encoder = AudioEncoder(
    model_name="facebook/wav2vec2-base",
    embedding_dim=768
)

# Encode audio
output = encoder.encode(
    audio=["path/to/audio.wav"],
    apply_repetition=True,
    repetition_factor=2
)
```

### MultimodalEncoder

Unified encoder that combines vision, audio, and text encodings.

```python
from src.multimodal.encoders import MultimodalEncoder

# Initialize multimodal encoder
encoder = MultimodalEncoder(
    embedding_dim=768,
    enable_vision=True,
    enable_audio=True
)

# Encode multimodal inputs
outputs = encoder.encode_multimodal(
    text="Describe this scene",
    images=["image1.jpg", "image2.jpg"],
    audio=["audio.wav"],
    apply_repetition=True,
    repetition_factor=2
)

# Fuse embeddings
fused = encoder.fuse_embeddings(outputs, fusion_mode="concat")
```

## Decoders

### ImageDecoder

Processes images using SigLIP 2 (512px).

```python
from src.multimodal.decoders import ImageDecoder

decoder = ImageDecoder()
result = decoder.decode("path/to/image.jpg")

print(result["modality"])        # "image"
print(result["tensor_type"])     # "pixel_values"
```

### AudioDecoder

Processes audio using Whisper V3 Turbo.

```python
from src.multimodal.decoders import AudioDecoder

decoder = AudioDecoder()
result = decoder.decode("path/to/audio.wav")

print(result["modality"])        # "audio"
print(result["input_features"].shape)
```

### VideoDecoder

Processes video using temporal pooling with SigLIP 2.

```python
from src.multimodal.decoders import VideoDecoder

decoder = VideoDecoder()
result = decoder.decode("path/to/video.mp4")

print(result["modality"])        # "video"
print(result["strategy"])        # "temporal_pooling"
```

### OmniDecoder

Unified decoder entry point for all modalities.

```python
from src.multimodal.decoders import OmniDecoder

decoder = OmniDecoder()

# Decode any modality
image_result = decoder.decode("image.jpg", modality="vision")
audio_result = decoder.decode("audio.wav", modality="audio")
video_result = decoder.decode("video.mp4", modality="video")
```

## Repetition-Aware Encoding

The module implements KV-cache optimization for repeated multimodal content (Paper 2512.14982).

```python
from src.multimodal.encoders import MultimodalEncoder, RepetitionAwareEncoder

base_encoder = MultimodalEncoder()
rep_encoder = RepetitionAwareEncoder(base_encoder)

# First encoding - caches result
output1 = rep_encoder.encode_with_cache(
    text="Query",
    images=["image.jpg"],
    repetition_factor=3,
    cache_key="query_1"
)

# Second encoding - uses cache
output2 = rep_encoder.encode_with_cache(
    text="Query",
    images=["image.jpg"],
    repetition_factor=3,
    cache_key="query_1"
)

# Check cache stats
stats = rep_encoder.get_cache_stats()
print(stats)  # {'cache_size': 1, 'repetition_counts': {...}}
```

## Supported Models

### Vision Encoders

| Model | ID | Resolution | Use Case |
|-------|-----|------------|----------|
| SigLIP 2 | `siglip2-so400m-patch16-512` | 512px | General vision |
| CLIP | `openai/clip-vit-base-patch32` | 224px | Vision-language |
| DINOv2 | `facebook/dinov2-base` | 518px | Visual features |

### Audio Encoders

| Model | ID | Sample Rate | Use Case |
|-------|-----|-------------|----------|
| Whisper V3 Turbo | `whisper-large-v3-turbo` | 16kHz | Speech recognition |
| Wav2Vec2 | `facebook/wav2vec2-base` | 16kHz | Audio features |
| Parakeet | `parakeet-tdt-0.6b-v3` | 16kHz | TTS/ASR |

### Video Models

| Model | Type | Resolution | Use Case |
|-------|------|------------|----------|
| Stable Video Diffusion | Generation | 512x512 | Video generation |
| VideoMAE | Understanding | 224x224 | Video understanding |

## Configuration

Encoder paths are configured in `configs/encoders.yaml`:

```yaml
encoders:
  vision:
    default: /mnt/e/data/encoders/vision-encoders/siglip2-so400m-patch16-512
  audio_input:
    default: /mnt/e/data/encoders/audio-encoders/whisper-large-v3-turbo
  audio_output:
    default: /mnt/e/data/encoders/audio-encoders/parakeet-tdt-0.6b-v3

decoders:
  vision_output:
    default: stabilityai/stable-diffusion-3.5-large
  video_output:
    default: stabilityai/stable-video-diffusion-img2vid-xt
```

## Testing

Run module-specific tests:

```bash
# Test encoder/decoder shapes
python -m pytest src/multimodal/tests/test_encoder_decoder_shapes.py -v

# Run all multimodal tests
python -m pytest tests/ -k "multimodal" -v
```

## Integration

The multimodal module integrates with:

- **Training Pipeline**: [`src/23_multimodal_distillation.py`](src/23_multimodal_distillation.py)
- **Inference**: [`src/inference/`](src/inference/)
- **Core**: [`src/nexus_core/`](src/nexus_core/)

## References

- Paper 2512.14982: Prompt Repetition for Multimodal Enhancement
- SigLIP 2: [Google Research](https://arxiv.org/abs/2303.15343)
- Whisper V3: [OpenAI](https://openai.com/research/whisper)

## License

MIT License - See [LICENSE](../../LICENSE) for details.
