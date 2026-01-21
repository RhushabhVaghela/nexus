# Shell Scripts Reference

Complete documentation for all shell scripts in the Nexus Model pipeline.

---

## run_universal_pipeline.sh

**Location:** `/mnt/d/Research Experiments/nexus_model/run_universal_pipeline.sh`

The main orchestrator script for training any combination of capabilities.

### Synopsis

```bash
./run_universal_pipeline.sh --base-model PATH [CAPABILITIES] [OPTIONS]
```

### Description

This script orchestrates the complete training pipeline:

1. **Detects** model modalities (text, vision, audio, video)
2. **Validates** capability requirements against available modalities
3. **Sequences** training stages in optimal order
4. **Executes** each stage with proper model chaining

### Capability Flags

| Flag | Description | Required Modalities |
|------|-------------|---------------------|
| `--enable-omni` | Convert text model to Omni | text |
| `--enable-cot` | Chain-of-Thought reasoning | text |
| `--enable-reasoning` | Multi-level reasoning | text |
| `--enable-thinking` | Extended thinking/reflection | text |
| `--enable-tools` | Function/tool calling | text |
| `--enable-streaming` | Token streaming output | text |
| `--enable-podcast` | NotebookLM-style podcast | text, audio_input, audio_output |
| `--enable-vision-qa` | Image understanding | text, vision |
| `--enable-video-understanding` | Video comprehension | text, vision, video |
| `--enable-tri-streaming` | Real-time multimodal streaming | ALL modalities |
| `--enable-image-generation` | Text-to-image generation | text, vision_output |
| `--enable-video-generation` | Text-to-video generation | text, video_output |

### Convenience Flags

| Flag | Expands To |
|------|------------|
| `--enable-all-text` | cot, reasoning, thinking, tools, streaming |
| `--enable-full-omni` | omni + all text + podcast, vision-qa, tri-streaming |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model PATH` | (required) | Path to base model |
| `--output-dir PATH` | `/mnt/e/data/models/trained` | Output directory |
| `--sample-size N` | 0 (all) | Limit samples per dataset |
| `--batch-size N` | 1 | Training batch size |
| `--epochs N` | 3 | Training epochs |
| `--dry-run` | false | Simulate without training |

### Examples

```bash
# Train CoT and reasoning on text model
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-cot \
    --enable-reasoning

# Convert to Omni, then add podcast
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-omni \
    --enable-podcast

# Full pipeline with all text capabilities
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-all-text \
    --sample-size=1000 \
    --epochs=5

# Dry-run to preview stages
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-0.5B \
    --enable-all-text \
    --dry-run

# Image generation training
./run_universal_pipeline.sh \
    --base-model=/mnt/e/data/models/Qwen2.5-Omni-7B \
    --enable-image-generation \
    --sample-size=500
```

### Output Structure

```
/mnt/e/data/models/trained/
├── cot/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── reasoning/
├── thinking/
└── final_model/
    └── ...
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Modality validation failed |
| 1 | Base model not found |
| 1 | No capabilities enabled |

---

## Stage Scripts (`src/stages/`)

Individual training stage implementations.

### Common Interface

All stage scripts accept these arguments:

```bash
python src/stages/stage_<name>.py \
    --base-model PATH \
    --output-dir PATH \
    --sample-size N \
    --batch-size N \
    --epochs N \
    --dry-run
```

### stage_cot.py

Chain-of-Thought reasoning training.

**Datasets:** OpenThoughts-114k, OpenMathReasoning

```bash
python src/stages/stage_cot.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/cot \
    --epochs 3
```

### stage_reasoning.py

Multi-level reasoning with math focus.

**Datasets:** NuminaMath-CoT, OpenMathReasoning

```bash
python src/stages/stage_reasoning.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/reasoning
```

### stage_thinking.py

Extended thinking and reflection.

**Datasets:** simplescaling/s1K-1.1

```bash
python src/stages/stage_thinking.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/thinking
```

### stage_tools.py

Function and tool calling.

**Datasets:** xlam-function-calling-60k, Synth-APIGen

```bash
python src/stages/stage_tools.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/tools
```

### stage_streaming.py

Token streaming output capability.

**Note:** Primarily a runtime feature, minimal training required.

```bash
python src/stages/stage_streaming.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/streaming
```

### stage_image_gen.py

Image generation with LLM → SD3 projector.

**Architecture:**

- Freezes LLM backbone
- Trains ImageProjector (LLM hidden → SD3 conditioning)
- Projects to 77 tokens × 2048 dimensions

**Datasets:** naruto-blip-captions, lora-training-dataset

```bash
python src/stages/stage_image_gen.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/image-gen \
    --sample-size 1000
```

### stage_video_gen.py

Video generation with LLM → SVD projector.

**Architecture:**

- Freezes LLM backbone  
- Trains VideoProjector (LLM hidden → SVD conditioning)
- Projects to 14 frames × 1024 dimensions
- Includes temporal smoothness regularization

**Datasets:** webvid (streaming)

```bash
python src/stages/stage_video_gen.py \
    --base-model /mnt/e/data/models/Qwen2.5-0.5B \
    --output-dir /mnt/e/data/models/trained/video-gen \
    --sample-size 500
```

---

## Pause/Resume Training

All training stages integrate with the training controller for runtime control.

### Signal-Based Control

```bash
# Find training process PID
ps aux | grep "stage_cot.py"
# Example output: user 12345 ... python stage_cot.py

# PAUSE training
kill -USR1 12345

# RESUME training (same signal toggles)
kill -USR1 12345

# Emergency CHECKPOINT
kill -USR2 12345
```

### Automatic Cooldown

Training automatically pauses every 500 steps for 60 seconds to:

- Clear CUDA cache
- Allow GPU to cool
- Reduce thermal throttling

### GPU Temperature Protection

If GPU temperature exceeds 83°C:

- Training pauses automatically
- Waits for temperature to drop
- Resumes when safe

### Monitoring

```bash
# Watch training logs
tail -f logs/train_cot.log

# Monitor GPU
watch -n 1 nvidia-smi

# Check pause state
# Training will print "⏸️ Paused... (SIGUSR1 to resume)"
```

---

## Environment

All scripts expect:

- Conda environment: `nexus`
- CUDA available (optional, falls back to CPU)
- Model paths under `/mnt/e/data/`

```bash
# Activate environment
conda activate nexus

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Logging

All scripts log to:

```
/mnt/d/Research Experiments/nexus_model/logs/
├── train_cot.log
├── train_reasoning.log
├── train_omni.log
└── ...
```

Log format:

```
[2026-01-20 17:00:00] [stage_cot] INFO: Starting training...
[2026-01-20 17:00:01] [stage_cot] INFO: Epoch 1/3
[2026-01-20 17:00:10] [stage_cot] INFO: Step 100, Avg Loss: 1.234
```
