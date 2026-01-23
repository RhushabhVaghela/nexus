# Nexus Universal Omni Model - Complete Guide

> **Last Updated:** January 2026  
> **Hardware:** RTX 5080+ 16GB VRAM  
> **Status:** Ready for Training

---

## 🎯 Overview

This project trains a **Universal Any-to-Any Multimodal Model** with:

- 12 trainable capabilities (CoT, Tools, Podcast, Image Gen, etc.)
- Automatic modality detection and validation
- Unified orchestrator with capability flags

---

## 📁 Project Structure

```
nexus_model/
├── src/
│   ├── detect_modalities.py     # Probe model capabilities
│   ├── capability_registry.py   # 12 capability definitions
│   ├── training_controller.py   # Pause/resume/checkpoint
│   ├── 24_multimodal_training.py # Main Omni training
│   ├── 10_sft_training.py       # Text SFT training
│   └── multimodal/
│       ├── model.py             # Encoders, wrappers
│       └── decoders.py          # Decoder interfaces
├── configs/
│   └── encoders.yaml            # All encoder/decoder paths
├── run_universal_pipeline.sh    # ⭐ MAIN ORCHESTRATOR
├── run_pipeline.sh              # Text-only pipeline
└── run_multimodal_pipeline.sh   # Omni conversion
```

---

## 📊 Performance Benchmarks

Run comprehensive benchmarks:

```bash
python src/benchmarks/benchmark_runner.py \
  --model /mnt/e/data/models/Qwen2.5-0.5B \
  --output results/benchmark.csv
```

**Sample Results (Qwen2.5-0.5B):**

| Metric | Value |
|--------|-------|
| Tokens/sec | 60.3 |
| Latency | 773ms |
| Perplexity | 11.06 |
| GPU Peak | 997MB |

---

## 🔄 Omni Model Support

### Custom Omni Loader

```python
from src.omni.loader import OmniModelLoader

# For training (freezes talker)
loader = OmniModelLoader()
model, tokenizer = loader.load_for_training("/path/to/omni")

# For inference
model, tokenizer = loader.load_for_inference("/path/to/omni")
```

### Omni Training Stage

```bash
python -m src.stages.stage_omni \
  --base-model /mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --data-dir /mnt/e/data/datasets/kaist-ai_CoT-Collection
```

---

## 🚀 Quick Start

### 1. Detect Your Model's Capabilities

```bash
python src/detect_modalities.py /mnt/e/data/models/Qwen2.5-Omni-7B-GPTQ-Int4
```

### 2. View Available Training Capabilities

```bash
python src/capability_registry.py
```

### 3. Run Training with Orchestrator

```bash
# Text-only (any model)
./run_universal_pipeline.sh --enable-cot --enable-tools

# Full Omni training
./run_universal_pipeline.sh --enable-full-omni

# With image generation
./run_universal_pipeline.sh --enable-omni --enable-image-generation
```

---

## 🎛️ Capability Flags

| Flag | Requires | Description |
|------|----------|-------------|
| `--enable-omni` | text | Convert to full Omni |
| `--enable-cot` | text | Chain-of-Thought reasoning |
| `--enable-reasoning` | text | Multi-level reasoning |
| `--enable-tools` | text | Function calling |
| `--enable-podcast` | Omni | NotebookLM-style audio |
| `--enable-vision-qa` | vision | Image understanding |
| `--enable-tri-streaming` | ALL | Gemini Live-style |
| `--enable-image-generation` | SD3 decoder | Text→Image |
| `--enable-video-generation` | SVD decoder | Text→Video |
| `--enable-remotion-explainer`| text | 3B1B-style Video |
| `--enable-all-text` | text | All text capabilities |
| `--enable-full-omni` | text | Everything |

### NexusLib Components
*   `NexusMath`: Animated LaTeX formulas.
*   `NexusGraph`: Dynamic function plotting.
*   `NexusFlow`: Animated flowcharts.
*   `NexusAnnotator`: Image annotation and labeling.
*   `NexusAudio`: Synchronized narration support.
*   `Nexus3D`: Three.js-powered 3D math visualizations.

---

## 🖥️ Interactive Dashboard & API

The project includes a web-based dashboard and a FastAPI backend for real-time video generation.

### 1. Start the API Server
```bash
python src/api/explainer_api.py
```
The server will be available at `http://localhost:8000`.

### 2. Start the Dashboard (Frontend)
```bash
cd dashboard
npm run dev
```

### 3. Generate via API
You can also trigger generation via `curl`:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the Pythagorean Theorem"}'
```

---

## 📂 Data Locations

```
E:/data/
├── base-model/
│   └── Qwen2.5-Omni-7B-GPTQ-Int4/     # Base LLM
├── encoders/
│   ├── vision encoders/siglip2-so400m-patch16-512/
│   └── audio encoders/whisper-large-v3-turbo/
├── decoders/
│   ├── vision-decoders/stabilityai_stable-diffusion-3-medium/
│   └── vision-decoders/stable-video-diffusion-img2vid-xt-1-1/
├── datasets/
│   ├── JourneyDB-GoT/          # Image generation
│   ├── Laion-Aesthetics-GoT/   # Image generation
│   ├── OmniEdit-GoT/           # Image editing
│   ├── VideoCoF-50k/           # Video generation
│   └── ... (42 total datasets)
└── models/
    ├── checkpoints/
    └── final/
```

---

## 🗄️ Dataset Management

The project includes an intelligent **Dataset Organizer** that automatically categorizes your raw data.

### Automation

- **Auto-Run**: The tool runs automatically at the start of `run_universal_pipeline.sh`.
- **Content Detection**: It inspects JSON/JSONL files for keys (e.g., `prompt`, `messages`, `tool_calls`) to determine the capability type.
- **Physical Organization**: Moves files into structured directories.

### Manual Usage

```bash
# Preview changes (Dry Run)
python src/utils/organize_datasets.py --base-path /mnt/e/data --dry-run

# Apply changes
python src/utils/organize_datasets.py --base-path /mnt/e/data --move
```

### Supported Categories

| Category | Keywords / Content Signals |
|----------|----------------------------|
| **cot** | `messages`, `prompt`+`response`, `instruction`+`output` |
| **reasoning** | `problem`+`solution`, `question`+`answer`, keywords: "math", "gsm8k" |
| **tools** | `tool_calls`, `function` |
| **vision-qa** | `image`, `visual` |
| **benchmarks** | keywords: "benchmark", "eval", "mmlu", "humaneval" |

### Model Organization

The tool also organizes model components:

- **Encoders**: `encoders/audio-encoders`, `encoders/vision-encoders`
- **Decoders**: `decoders/audio-decoders`, `decoders/vision-decoders`

---

## ⚙️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 14GB | 16GB+ |
| RAM | 32GB | 64GB |
| Storage | 500GB | 1TB NVMe |

### Capability VRAM Usage

| Capability | VRAM | Time |
|------------|------|------|
| CoT, Tools | ~8GB | 2h |
| Vision-QA | ~10GB | 3h |
| Podcast | ~12GB | 4h |
| Tri-Streaming | ~14GB | 6h |
| Image Gen | ~14GB | 6h |
| Remotion Explainer | ~12GB | 8h |

---

## 🛡️ Training Safety Features

### Pause/Resume Training

```bash
# Get PID
ps aux | grep python

# Pause: kill -USR1 <PID>
# Resume: kill -USR1 <PID>
# Emergency Checkpoint: kill -USR2 <PID>
```

### Automatic Cooldown

- Every 500 steps: 1 min cooldown
- GPU > 83°C: Auto cooldown
- Configurable in `src/training_controller.py`

---

## 📊 Datasets by Capability

| Capability | Datasets |
|------------|----------|
| **Reasoning** | CoT-Collection, O1-SFT-Pro/Ultra |
| **Tool-Calling** | Gorilla, XLAM-60K, Hermes |
| **Podcast** | OleSpeech-IV, SPoRC, Cornell |
| **Image Gen** | JourneyDB-GoT, Laion-GoT, OmniEdit-GoT |
| **Video Gen** | VideoCoF-50k, MSR-VTT, VaTeX |
| **Remotion Explainer** | Remotion-1M-Synthetic |

---

## 📝 Key Scripts

| Script | Purpose |
|--------|---------|
| `run_universal_pipeline.sh` | ⭐ Main orchestrator |
| `src/detect_modalities.py` | Probe model capabilities |
| `src/capability_registry.py` | Capability definitions |
| `src/training_controller.py` | Pause/checkpoint/cooldown |
| `src/24_multimodal_training.py` | Omni training |

---

## 🔧 Configuration

### `configs/encoders.yaml`

Central config for all paths:

- Encoder locations
- Decoder locations
- Dataset mappings
- Output directories

---

## ❓ FAQ

**Q: Which capabilities can I train with 16GB VRAM?**  
A: All of them with batch_size=1 and gradient checkpointing.

**Q: Do I need to download extra datasets for image generation?**  
A: Yes - JourneyDB-GoT, Laion-GoT recommended (~100GB total).

**Q: Can I pause training mid-way?**  
A: Yes - use `kill -USR1 <PID>` to pause/resume.

**Q: What if GPU overheats?**  
A: Auto-cooldown triggers at 83°C. Can adjust in training_controller.py.

---

## 🎓 Recommended Training Order

1. **Start small:** `--enable-cot` (test setup)
2. **Add tools:** `--enable-cot --enable-tools`
3. **Add vision:** `--enable-vision-qa`
4. **Full Omni:** `--enable-full-omni`
5. **Generation:** `--enable-image-generation`

---

*For archived legacy documentation, see `docs/archive/`*
