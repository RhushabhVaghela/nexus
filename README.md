# Manus Universal Omni Model

**Production-ready Any-to-Any Multimodal Model with Full Capability Training Pipeline**

> 🎯 **Goal**: Train a single model that can understand and generate Text, Images, Audio, and Video.

---

## 🆕 Latest Updates (January 2026)

### New Pipeline Architecture

- **Modality Detection**: Auto-detect model capabilities from config
- **Capability Registry**: 12 trainable capabilities with modality gates
- **Decoder Support**: Image generation (SD3) and Video generation (SVD)
- **Training Controller**: Pause/resume, emergency checkpoints, cooldown intervals

### Supported Capabilities

| Category | Capability | Required Modalities |
|----------|------------|---------------------|
| **Text** | tool-calling, cot, reasoning, thinking, streaming | text |
| **Audio** | podcast | text + audio_in + audio_out |
| **Vision** | vision-qa, video-understanding | text + vision + video |
| **Omni** | tri-streaming | ALL modalities |
| **Generation** | image-generation, video-generation | text + vision_output/video_output |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Base LLM: Qwen2.5-Omni-7B-GPTQ-Int4                           │
├─────────────────────────────────────────────────────────────────┤
│  ENCODERS (Input)                │  DECODERS (Output)           │
│  ├─ SigLIP2 (Vision) 512x512     │  ├─ SD3 Medium (Images)      │
│  ├─ Whisper V3 Turbo (Audio)     │  ├─ SVD 1.1 (Video)          │
│  └─ Parakeet TDT (ASR)           │  └─ Built-in TTS (Audio)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
E:/data/
├── base-model/
│   └── Qwen2.5-Omni-7B-GPTQ-Int4/
├── encoders/
│   ├── vision encoders/siglip2-so400m-patch16-512/
│   └── audio encoders/whisper-large-v3-turbo/
├── decoders/
│   ├── vision-decoders/stabilityai_stable-diffusion-3-medium-diffusers/
│   └── audio-decoders/stabilityai_stable-video-diffusion-img2vid-xt-1-1/
└── datasets/
    ├── JourneyDB-GoT/          # Image generation
    ├── Laion-Aesthetics-GoT/   # Image generation
    ├── OmniEdit-GoT/           # Image editing
    ├── VideoCoF-50k/           # Video generation
    ├── olewave_OleSpeech-IV/   # Podcast
    ├── O1-OPEN_OpenO1-SFT-Ultra/ # Reasoning
    └── ... (42 total datasets)
```

---

## 🚀 Quick Start

### 1. Detect Model Capabilities

```bash
python src/detect_modalities.py /mnt/e/data/base-model/Qwen2.5-Omni-7B-GPTQ-Int4
```

### 2. View Available Capabilities

```bash
python src/capability_registry.py
```

### 3. Run Training

```bash
# Text capabilities (safe, low VRAM)
./run_pipeline.sh --enable-cot --enable-tools

# Full Omni with image generation
./run_pipeline.sh --enable-omni --enable-image-generation
```

---

## ⚙️ Training Features

### Pause/Resume Training

```bash
# Get training PID
ps aux | grep python

# Pause training
kill -USR1 <PID>

# Resume training
kill -USR1 <PID>
```

### Emergency Checkpoint

```bash
# Force immediate checkpoint save
kill -USR2 <PID>
```

### Automatic Cooldown

- Every 500 steps: 1 minute cooldown
- GPU temp > 83°C: Auto cooldown
- Configurable in `src/training_controller.py`

---

## 💾 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 16GB | RTX 4090 24GB |
| RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |
| VRAM (Training) | 14GB | 20GB |

### Your Setup (RTX 5080 16GB)

- ✅ Text capabilities: Full speed
- ✅ Podcast, Vision-QA: Full speed
- ⚠️ Image/Video generation: Use gradient checkpointing
- ⚠️ Tri-streaming: Reduce batch size to 1

---

## 📊 Dataset Summary

| Category | Datasets | Total Samples |
|----------|----------|---------------|
| Reasoning | CoT-Collection, O1-SFT-Pro/Ultra | ~500K |
| Tool-Calling | Gorilla, XLAM, Hermes | ~150K |
| Podcast | OleSpeech-IV, SPoRC, Cornell | ~200K |
| Image Gen | JourneyDB-GoT, Laion-GoT, OmniEdit-GoT | ~220K |
| Video Gen | VideoCoF-50k, MSR-VTT, VaTeX | ~100K |

---

## 🔧 Configuration

### `configs/encoders.yaml`

Central configuration for all encoder/decoder paths and dataset mappings.

### Key Files

| File | Purpose |
|------|---------|
| `src/detect_modalities.py` | Probe model for native capabilities |
| `src/capability_registry.py` | Define capability requirements |
| `src/training_controller.py` | Pause/resume/checkpoint/cooldown |
| `src/24_multimodal_training.py` | Main Omni training script |

---

## 📈 Expected Results (Hypothetical)

After training on all datasets with proper hyperparameters:

| Benchmark | Expected Score | Comparison |
|-----------|----------------|------------|
| MMLU | ~75-78% | Above Llama-3.1-8B |
| HellaSwag | ~82-85% | Competitive with GPT-4o-mini |
| Function Calling | ~85-88% | Near Gorilla-Openfunctions |
| Vision QA | ~70-72% | Below GPT-4V, above LLaVA |
| Audio Understanding | ~65-70% | Competitive tier |

**Positioning**: Upper-mid tier open-source multimodal model.

---

## ✅ Checklist

- [x] Modality detection system
- [x] Capability registry with gates
- [x] Encoder/decoder path configuration
- [x] Training controller with safety features
- [ ] Orchestrator script rewrite
- [ ] Image/Video generation projector training
- [ ] Full benchmark evaluation

---

## 📚 Documentation

- [Pipeline Architecture](./docs/pipeline_architecture_plan.md)
- [Dataset Catalog](./dataset%20and%20performance%20suggestions.md)
- [Training Suite](./training-suite/README.md)
