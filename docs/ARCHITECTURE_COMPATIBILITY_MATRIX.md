# Nexus Architecture Compatibility Matrix

This document provides a comprehensive overview of supported model architectures and their compatibility with Nexus components.

## Legend

- ✅ Fully Supported
- ⚠️ Partial Support
- ❌ Not Supported
- 🔄 In Development

## Language Models

| Architecture | Type | Registry Key | Training | Inference | GGUF | Notes |
|--------------|------|--------------|----------|-----------|------|-------|
| Llama 2/3 | Causal | `llama-*` | ✅ | ✅ | ✅ | Native support |
| Mistral | Causal | `mistral-*` | ✅ | ✅ | ✅ | v0.1, v0.2, v0.3 |
| Mixtral | MoE | `mixtral-*` | ✅ | ✅ | ✅ | 8x7B, 8x22B |
| Qwen 2/2.5 | Causal | `qwen2-*` | ✅ | ✅ | ✅ | 0.5B to 72B |
| Gemma 2/3 | Causal | `gemma-*` | ✅ | ✅ | ✅ | 2B, 4B, 9B, 27B |
| Phi 3/4 | Causal | `phi3-*`, `phi4-*` | ✅ | ✅ | ✅ | Mini, Small, Medium |
| Trinity-Large | Reasoning | `trinity_large` | ✅ | ✅ | ✅ | Added in v6.1 |
| LongCat | Long Context | `longcat-*` | ✅ | ✅ | ✅ | 8B, 16B variants |
| Command-R | Causal | `command-r-*` | ✅ | ✅ | ⚠️ | Limited testing |
| DBRX | MoE | `dbrx-*` | ⚠️ | ✅ | ✅ | Inference only |

## Vision Models

| Architecture | Type | Registry Key | Training | Inference | Notes |
|--------------|------|--------------|----------|-----------|-------|
| CLIP | Encoder | `clip-*` | ✅ | ✅ | All variants |
| SigLIP | Encoder | `siglip-*` | ✅ | ✅ | SigLIP 1/2 |
| ViT | Encoder | `vit-*` | ✅ | ✅ | Vision Transformer |
| Step-VL | Vision-Language | `vision_main` | ⚠️ | ✅ | 10B parameter |
| OVD | Detection | `object_detection` | ⚠️ | ✅ | Open vocabulary |

## Diffusion Models (Image)

| Architecture | Type | Pipeline | Text2Img | Img2Img | Inpaint | ControlNet |
|--------------|------|----------|----------|---------|---------|------------|
| SD 1.5/2.1 | Diffusion | `sd` | ✅ | ✅ | ✅ | ✅ |
| SDXL | Diffusion | `sdxl` | ✅ | ✅ | ✅ | ✅ |
| SDXL Turbo | Diffusion | `sdxl` | ✅ | ✅ | ❌ | ❌ |
| SD 3/3.5 | Diffusion | `sd3` | ✅ | ✅ | ✅ | ⚠️ |
| FLUX Dev | Flow | `flux` | ✅ | ✅ | ⚠️ | ❌ |
| FLUX Schnell | Flow | `flux` | ✅ | ❌ | ❌ | ❌ |
| FLUX Fill | Flow | `flux-fill` | ❌ | ❌ | ✅ | ❌ |
| Z-Image | Diffusion | `z-image` | ✅ | ✅ | ⚠️ | ❌ |
| Z-Image Turbo | Diffusion | `z-image-turbo` | ✅ | ❌ | ❌ | ❌ |
| HunyuanDiT | Diffusion | `hunyuan` | ✅ | ✅ | ⚠️ | ❌ |

## Video Models

| Architecture | Type | Pipeline | T2V | I2V | Consistency | Notes |
|--------------|------|----------|-----|-----|-------------|-------|
| LTX-Video | Video Diffusion | `ltx-video` | ✅ | ✅ | ✅ | Lightricks |
| SVD | Video Diffusion | `svd` | ❌ | ✅ | ✅ | Stability AI |
| SVD-XT | Video Diffusion | `svd-xt` | ❌ | ✅ | ✅ | Extended frames |
| CogVideoX | Video Diffusion | `cogvideo` | ✅ | ✅ | ✅ | 2B, 5B variants |
| HunyuanVideo | Video Diffusion | `hunyuan-video` | ✅ | ⚠️ | ✅ | Tencent |

## Audio Models

| Architecture | Type | Registry Key | Training | Inference | Notes |
|--------------|------|--------------|----------|-----------|-------|
| Whisper | ASR | `whisper-*` | ✅ | ✅ | All sizes |
| Parakeet | ASR | `asr_fast` | ⚠️ | ✅ | Fast ASR |
| VibeVoice | ASR | `asr_long` | ⚠️ | ✅ | Long-form |
| PersonaPlex | Speech | `omni_speech` | ⚠️ | ✅ | Conversational |
| Qwen TTS | TTS | `tts_*` | ⚠️ | ✅ | Voice cloning |
| AudioCraft | Audio Gen | `audiocraft-*` | ⚠️ | ✅ | MusicGen, AudioGen |

## Multimodal Models

| Architecture | Type | Registry Key | Text | Vision | Audio | Notes |
|--------------|------|--------------|------|--------|-------|-------|
| Qwen-Omni | Multimodal | `omni_*` | ✅ | ✅ | ✅ | 7B, 30B |
| CLIP | Vision-Lang | `vision_enc` | ✅ | ✅ | ❌ | Embeddings |
| VideoMAE | Video | `video_enc` | ❌ | ✅ | ❌ | Temporal |

## GGUF Support Matrix

| Architecture | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 | Notes |
|--------------|--------|--------|------|------|-----|-------|
| Llama 2/3 | ✅ | ✅ | ✅ | ✅ | ✅ | All variants |
| Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | v0.1-0.3 |
| Mixtral | ✅ | ✅ | ✅ | ✅ | ✅ | MoE supported |
| Qwen 2/2.5 | ✅ | ✅ | ✅ | ✅ | ✅ | All sizes |
| Gemma 2 | ✅ | ✅ | ✅ | ✅ | ✅ | Native support |
| Phi 3/4 | ✅ | ✅ | ✅ | ✅ | ✅ | All sizes |
| Trinity-Large | ✅ | ✅ | ✅ | ✅ | ✅ | Recommended: Q5 |
| LongCat | ✅ | ✅ | ✅ | ✅ | ✅ | Recommended: Q4 |

## Unknown Architecture Auto-Detection

Nexus v6.1+ includes automatic architecture detection for unknown models:

```python
from nexus.core.towers.registry import detect_architecture

# Auto-detect model type
type = detect_architecture("some/unknown-model")
# Returns: 'causal', 'vision', 'audio', 'generation', 'encoder', or 'unknown'
```

### Detection Patterns

| Pattern | Detected Type |
|---------|---------------|
| `vision`, `clip`, `siglip`, `vit` | Vision |
| `audio`, `whisper`, `wav2vec` | Audio |
| `diffusion`, `sd`, `sdxl`, `flux` | Generation |
| `embedding`, `sentence-transformer` | Encoder |
| `llama`, `mistral`, `qwen`, `gemma` | Causal |
| `trinity` | Reasoning |
| `longcat` | Long Context |

## Component Compatibility

### Knowledge Distillation

| Source | Target | Adapter Support | Notes |
|--------|--------|-----------------|-------|
| Diffusion UNet | Student | ✅ | Via DiffusionAdapter |
| Vision Encoder | Student | ✅ | Via VisionAdapter |
| LLM Hidden | Student | ✅ | Via BaseAdapter |

### Quantization Support

| Format | Load | Save | Training | Inference |
|--------|------|------|----------|-----------|
| PyTorch (FP32) | ✅ | ✅ | ✅ | ✅ |
| PyTorch (FP16/BF16) | ✅ | ✅ | ✅ | ✅ |
| BitsAndBytes (8-bit) | ✅ | ❌ | ⚠️ | ✅ |
| BitsAndBytes (4-bit/NF4) | ✅ | ❌ | ⚠️ | ✅ |
| GGUF (all quant) | ✅ | ⚠️ | ❌ | ✅ |

### Device Support

| Device | Training | Inference | Diffusion | Video | GGUF |
|--------|----------|-----------|-----------|-------|------|
| CUDA | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROCm | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| MPS (Apple) | ⚠️ | ✅ | ✅ | ✅ | ⚠️ |
| CPU | ⚠️ | ✅ | ✅ | ⚠️ | ✅ |
| Intel GPU | ❌ | ⚠️ | ❌ | ❌ | ⚠️ |

## Version Compatibility

| Nexus Version | Python | PyTorch | Transformers | Diffusers | Notes |
|---------------|--------|---------|--------------|-----------|-------|
| 6.1.0 | 3.9-3.12 | 2.1+ | 4.39+ | 0.26+ | Current |
| 6.0.x | 3.9-3.11 | 2.0+ | 4.30+ | 0.20+ | Previous |

## Deprecated Features

| Feature | Deprecated | Removal | Replacement |
|---------|------------|---------|-------------|
| Old registry format | v6.0 | v7.0 | New dict format |
| Static tower loading | v6.0 | v7.0 | TowerLoader |
| Manual adapter init | v6.1 | v7.0 | BaseAdapter |

## Planned Support (v6.2)

| Architecture | Type | Status |
|--------------|------|--------|
| Janus | Multimodal | 🔄 In Development |
| InternVL | Vision-Language | 🔄 In Development |
| Mamba | State Space | 🔄 Research |
| RetNet | Alternative Arch | 🔄 Research |

## Troubleshooting Compatibility

### Model Won't Load
1. Check architecture is in compatibility matrix
2. Verify model files are complete
3. Check Nexus version supports the architecture
4. Review error logs for specific incompatibilities

### Performance Issues
1. Use recommended quantization for your hardware
2. Enable appropriate optimizations (VAE slicing, CPU offloading)
3. Check device compatibility matrix

### Training Failures
1. Ensure adapter dimensions match
2. Verify gradient checkpointing compatibility
3. Check mixed precision support for your device

## Contributing New Architectures

To add support for a new architecture:

1. Add registry entry in `src/nexus/core/towers/registry.py`
2. Create loader in appropriate module
3. Add tests in `tests/unit/`
4. Update this compatibility matrix
5. Submit PR with documentation

See `CONTRIBUTING.md` for detailed guidelines.
