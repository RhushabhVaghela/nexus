# Multimodal Datasets for Manus Prime

## 🖼️ Vision (Image-to-Code)

### 1. WebSight (BEST) ⭐

- **Size**: 2M HTML/screenshot pairs
- **Quality**: High (Tailwind CSS, real images)
- **License**: Apache 2.0
- **HuggingFace**: `HuggingFaceM4/WebSight`

```python
from datasets import load_dataset
vision_ds = load_dataset("HuggingFaceM4/WebSight", split="train")
```

### 2. LLaVA-Instruct-150K (General Vision)

- **Size**: 150K instruction pairs
- **Use**: General vision-language alignment
- **HuggingFace**: `liuhaotian/LLaVA-Instruct-150K`

---

## 🎤 Audio (Speech-to-Text/Code)

### 1. Common Voice (BEST for ASR) ⭐

- **Size**: 19K+ hours multilingual
- **Use**: Speech recognition training
- **HuggingFace**: `mozilla-foundation/common_voice_17_0`

```python
from datasets import load_dataset
audio_ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train")
```

### 2. LibriSpeech (Clean English)

- **Size**: 1000 hours clean speech
- **Use**: High-quality ASR
- **HuggingFace**: `openslr/librispeech_asr`

---

## 🎬 Video Understanding

### 1. FineVideo (BEST) ⭐

- **Size**: 43K videos, 3425 hours
- **Quality**: Rich metadata, scene annotations
- **HuggingFace**: `HuggingFaceM4/FineVideo`

```python
from datasets import load_dataset
video_ds = load_dataset("HuggingFaceM4/FineVideo", split="train")
```

### 2. Video-MME Benchmark

- **Use**: Evaluation of video understanding
- **HuggingFace**: `lmms-lab/Video-MME`

---

## 📥 Quick Download Script

```bash
# Install datasets library
pip install datasets

# Download vision data (2M samples, ~50GB)
python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceM4/WebSight', split='train[:100000]'); ds.save_to_disk('datasets/vision')"

# Download audio data (sample)
python -c "from datasets import load_dataset; ds = load_dataset('mozilla-foundation/common_voice_17_0', 'en', split='train[:10000]'); ds.save_to_disk('datasets/audio')"

# Download video data (sample)
python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceM4/FineVideo', split='train[:1000]'); ds.save_to_disk('datasets/video')"
```

---

## Training Priority

| Modality | Dataset | Samples | Priority |
|----------|---------|---------|----------|
| **Vision** | WebSight | 100K | 🔴 HIGH |
| **Audio** | Common Voice | 10K | 🟡 MEDIUM |
| **Video** | FineVideo | 1K | 🟢 LOW |
