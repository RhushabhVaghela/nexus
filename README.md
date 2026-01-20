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
- **Universal Orchestrator**: Single script to train any combination of capabilities

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

### 3. Run Training with Universal Orchestrator

```bash
# Text-only capabilities (any model)
./run_universal_pipeline.sh --enable-cot --enable-tools

# Full Omni with podcast
./run_universal_pipeline.sh --enable-full-omni

# Image generation (requires SD3 decoder)
./run_universal_pipeline.sh --enable-image-generation
```

### 4. Available Capability Flags

| Flag | Description |
|------|-------------|
| `--enable-omni` | Convert text→Omni |
| `--enable-cot` | Chain-of-Thought |
| `--enable-reasoning` | Multi-level reasoning |
| `--enable-tools` | Function calling |
| `--enable-podcast` | NotebookLM-style |
| `--enable-vision-qa` | Image understanding |
| `--enable-tri-streaming` | Gemini Live-style |
| `--enable-image-generation` | Text→Image (SD3) |
| `--enable-video-generation` | Text→Video (SVD) |
| `--enable-all-text` | All text capabilities |
| `--enable-full-omni` | Everything |

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

## 📁 Key Files

| File | Purpose |
|------|---------|
| `run_universal_pipeline.sh` | ⭐ Main orchestrator |
| `src/detect_modalities.py` | Probe model capabilities |
| `src/capability_registry.py` | 12 capability definitions |
| `src/training_controller.py` | Pause/checkpoint/cooldown |
| `configs/encoders.yaml` | All encoder/decoder paths |

---

## 📚 Documentation

- **[Complete Guide](docs/GUIDE.md)** - Full documentation
- **[Training Suite](training-suite/README.md)** - Sample-based training
- **[Dataset Catalog](docs/multimodal/datasets.md)** - All 42 datasets

---

## 💾 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 16GB | RTX 4090 24GB |
| RAM | 32GB | 64GB |
| Storage | 500GB SSD | 1TB NVMe |

---

## 🛡️ Training Safety

```bash
# Pause/Resume: kill -USR1 <PID>
# Emergency Checkpoint: kill -USR2 <PID>
# Auto-cooldown every 500 steps
```

---

*For archived legacy documentation, see `docs/archive/`*
