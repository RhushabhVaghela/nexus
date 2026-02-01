# Shell Scripts Reference

Complete documentation for the Nexus unified CLI and shell scripts.

---

## Unified CLI (`scripts/nexus.sh`)

**Location:** `/mnt/d/Research Experiments/nexus/scripts/nexus.sh`

The unified CLI provides a single entry point for all Nexus operations.

### Synopsis

```bash
./scripts/nexus.sh [COMMAND] [OPTIONS]
```

### Commands

| Command | Description |
|---------|-------------|
| `pipeline` | Run the main text/code training pipeline |
| `multimodal` | Run the multimodal pipeline (vision/audio/video) |
| `reasoning` | Run the reasoning training pipeline |
| `universal` | Run the universal capability pipeline |
| `master` | Run the master self-driving pipeline |
| `distillation` | Run knowledge distillation |
| `niwt` | Run NIWT profiling pipeline |
| `profiling` | Run performance profiling |
| `tests` | Run the test suite |
| `cleanup` | Clean up temporary files and caches |
| `help` | Show help message |

### Quick Start

```bash
# Make executable
chmod +x scripts/nexus.sh

# Show help
./scripts/nexus.sh help

# Run tests
./scripts/nexus.sh tests --unit-only

# Clean up
./scripts/nexus.sh cleanup
```

---

## Pipeline Command

**Usage:** `./scripts/nexus.sh pipeline [PHASE] [OPTIONS]`

### Phases

| Phase | Description |
|-------|-------------|
| `download` | Download text datasets (01-03) |
| `process` | Process text data (04-06) |
| `validate` | Validate text datasets (07-09) |
| `train` | Run training pipeline (10-15) |
| `distill` | Run distillation from teacher model |
| `all` | Run complete TEXT pipeline (01-15) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode=censored|uncensored` | censored | Training mode |
| `--target-samples=N` | 100000 | Target samples for premium datasets |
| `--training-method=METHOD` | sft | Training method: sft, lora, qlora, dpo, grpo, orpo, distillation |
| `--teacher-model=PATH` | - | Teacher model for distillation |
| `--distillation-alpha=FLOAT` | 0.5 | Distillation alpha |

### Examples

```bash
# Run complete pipeline
./scripts/nexus.sh pipeline all

# Run training only with QLoRA
./scripts/nexus.sh pipeline train --training-method=qlora --mode=censored

# Run distillation
./scripts/nexus.sh pipeline distill --teacher-model=/path/to/teacher
```

---

## Multimodal Command

**Usage:** `./scripts/nexus.sh multimodal [PHASE] [OPTIONS]`

Convert ANY text model to OMNI (any-to-any multimodal).

### Phases

| Phase | Description |
|-------|-------------|
| `download` | Download multimodal data |
| `distill` | Distill multimodal data |
| `train` | Train Omni-Modal model |
| `all` | Run full pipeline |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model=PATH` | (required) | Base model path |
| `--modality=vision|audio|video` | vision | Modality to train |
| `--stage=1|2` | 1 | Training stage |
| `--teacher=mock-teacher|gpt-4v` | mock-teacher | Teacher model |
| `--force` | false | Force training even if already Omni |
| `--limit=N` | 1000 | Dataset sample limit |

### Examples

```bash
# Run full multimodal pipeline
./scripts/nexus.sh multimodal all --base-model=/path/to/model

# Train specific stage
./scripts/nexus.sh multimodal train --base-model=/path/to/model --stage=1

# Force training
./scripts/nexus.sh multimodal all --base-model=/path/to/model --force
```

---

## Reasoning Command

**Usage:** `./scripts/nexus.sh reasoning [OPTIONS]`

Complete pipeline for training models with advanced reasoning capabilities.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model PATH` | (required) | Path to base model |
| `--output-dir PATH` | checkpoints/reasoning | Output directory |
| `--enable-cot` | false | Enable CoT dataset generation |
| `--enable-context` | false | Enable context extension |
| `--skip-sft` | false | Skip SFT stage |
| `--skip-grpo` | false | Skip GRPO stage |
| `--cot-type TYPE` | math | Reasoning type: math, code, logic |
| `--target-context N` | 32768 | Target context length |

### Examples

```bash
# Enable CoT generation and train
./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-cot

# Enable context extension
./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-context --target-context 131072

# Skip GRPO, only do SFT
./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-cot --skip-grpo
```

---

## Universal Command

**Usage:** `./scripts/nexus.sh universal [OPTIONS]`

Unified pipeline for training any combination of capabilities on any base model.

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
| `--enable-remotion-explainer` | 3Blue1Brown-style video generation | text, vision_output |

### Convenience Flags

| Flag | Expands To |
|------|------------|
| `--enable-all-text` | cot, reasoning, thinking, tools, streaming |
| `--enable-full-omni` | omni + all text + podcast, vision-qa, tri-streaming |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model PATH` | (required) | Path to base model |
| `--output-dir PATH` | /mnt/e/data/models/trained | Output directory |
| `--sample-size N` | 0 (all) | Limit samples per dataset |
| `--batch-size N` | 1 | Training batch size |
| `--epochs N` | 3 | Training epochs |
| `--training-method METHOD` | sft | sft, lora, qlora, dpo, grpo, orpo, distillation |
| `--dry-run` | false | Simulate without training |
| `--organize` | false | Auto-organize datasets before training |

### Examples

```bash
# Train CoT and reasoning
./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot --enable-reasoning

# Convert to Omni, then add podcast
./scripts/nexus.sh universal --base-model=/path/to/model --enable-omni --enable-podcast

# Full pipeline with all text capabilities
./scripts/nexus.sh universal --base-model=/path/to/model --enable-all-text --sample-size=1000

# Dry-run to preview stages
./scripts/nexus.sh universal --base-model=/path/to/model --enable-all-text --dry-run

# Image generation training
./scripts/nexus.sh universal --base-model=/path/to/model --enable-image-generation
```

---

## Master Command

**Usage:** `./scripts/nexus.sh master [OPTIONS]`

Run the master self-driving pipeline.

### Options

| Option | Description |
|--------|-------------|
| `--reset` | Full reset: clear state and checkpoints |
| `--dry-run` | Simulate execution without compute |
| `--skip-non-llm` | Skip audio/vision/multimodal models |
| `--stage NAME` | Run only specific stage |
| `--models ID1,ID2` | Filter to specific teacher models |
| `--datasets NAME` | Filter datasets |
| `--sample_size N` | Sample size for training |
| `--epochs N` | Training epochs |
| `--use-unsloth` | Use Unsloth for faster training |

### Examples

```bash
# Full reset and run
./scripts/nexus.sh master --reset

# Run with specific models
./scripts/nexus.sh master --models "coder,vision_main" --sample_size 5000

# Dry run to check configuration
./scripts/nexus.sh master --dry-run
```

---

## Distillation Command

**Usage:** `./scripts/nexus.sh distillation [OPTIONS]`

Run knowledge distillation from teacher to student model.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--teacher PATH` | /mnt/e/data/models/Qwen2.5-Omni | Teacher model path |
| `--student PATH` | /mnt/e/data/models/Qwen2.5-0.5B | Student model path |
| `--data PATH` | /mnt/e/data/multimodal | Data directory |
| `--alpha FLOAT` | 0.5 | Distillation alpha |
| `--temperature FLOAT` | 2.0 | Temperature for soft targets |

### Examples

```bash
# Run with default models
./scripts/nexus.sh distillation

# Specify custom models
./scripts/nexus.sh distillation --teacher=/path/to/teacher --student=/path/to/student

# Adjust distillation parameters
./scripts/nexus.sh distillation --alpha=0.7 --temperature=3.0
```

---

## NIWT Command

**Usage:** `./scripts/nexus.sh niwt [OPTIONS]`

Run NIWT (Neural Information-Weighted Tower) profiling pipeline.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_name NAME` | (required) | Model name from registry |
| `--batch_size N` | 8 | Batch size |
| `--samples N` | 50 | Number of samples |

### Examples

```bash
# Profile a model
./scripts/nexus.sh niwt --model_name="microsoft/Phi-3-mini-4k-instruct"

# Adjust batch size and samples
./scripts/nexus.sh niwt --model_name="gpt2" --batch_size=4 --samples=100
```

---

## Profiling Command

**Usage:** `./scripts/nexus.sh profiling [OPTIONS]`

Run performance profiling on a model.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model PATH` | /mnt/e/data/models/AgentCPM-Explore | Model path |
| `--batch-size N` | 4 | Batch size |

### Examples

```bash
# Profile default model
./scripts/nexus.sh profiling

# Profile specific model with custom batch size
./scripts/nexus.sh profiling --model=/path/to/model --batch-size=8
```

---

## Tests Command

**Usage:** `./scripts/nexus.sh tests [OPTIONS]`

Run the test suite with intelligent categorization.

### Options

| Option | Description |
|--------|-------------|
| `--unit-only` | Run only unit tests |
| `--integration-only` | Run only integration tests |
| `--real-models` | Include tests requiring real models |
| `--distributed` | Include distributed tests |
| `--gpu` | Include GPU tests |
| `--slow` | Include slow tests |
| `--benchmark` | Include benchmark tests |
| `--coverage` | Generate coverage report |
| `--report` | Generate JSON test report |
| `-v, --verbose` | Verbose output |
| `--all` | Run all tests including real models |

### Examples

```bash
# Run unit tests only
./scripts/nexus.sh tests --unit-only

# Run integration tests with verbose output
./scripts/nexus.sh tests --integration-only --verbose

# Generate coverage report
./scripts/nexus.sh tests --coverage --report

# Run everything
./scripts/nexus.sh tests --all
```

---

## Cleanup Command

**Usage:** `./scripts/nexus.sh cleanup`

Clean up temporary files, caches, and old artifacts.

### What It Cleans

- Python bytecode (`__pycache__`, `*.pyc`, `*.pyo`)
- Temporary files (`*.tmp`, `.DS_Store`, `Thumbs.db`)
- Lock files (`*.lock`)
- Pipeline state files

### Examples

```bash
# Clean up
./scripts/nexus.sh cleanup
```

---

## Legacy Scripts (Consolidated)

The following individual scripts have been **consolidated into the unified CLI**:

| Old Script (Removed) | New Command |
|----------------------|-------------|
| `run_pipeline.sh` | `./scripts/nexus.sh pipeline` |
| `run_multimodal_pipeline.sh` | `./scripts/nexus.sh multimodal` |
| `run_reasoning_pipeline.sh` | `./scripts/nexus.sh reasoning` |
| `run_universal_pipeline.sh` | `./scripts/nexus.sh universal` |
| `run_nexus_master.sh` | `./scripts/nexus.sh master` |
| `scripts/run_nexus_master.sh` | `./scripts/nexus.sh master` |
| `scripts/run_distillation.sh` | `./scripts/nexus.sh distillation` |
| `scripts/run_profiling.sh` | `./scripts/nexus.sh profiling` |
| `scripts/setup_voice_models.sh` | `./scripts/nexus.sh setup-voice` |
| `generate_training_scripts.sh` | `./scripts/nexus.sh training-suite` |
| `cleanup.sh` | `./scripts/nexus.sh cleanup` |

---

## Environment

All commands expect:

- Conda environment: `nexus`
- CUDA available (optional, falls back to CPU)
- Model paths under `/mnt/e/data/` (configurable)

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
/mnt/d/Research Experiments/nexus/logs/
├── 01_download.log
├── 02_benchmarks.log
├── 03_premium.log
├── 04_process.log
├── 05_repetitive.log
├── 06_preferences.log
├── 07_validate.log
├── 08_validate_benchmarks.log
├── 09_validate_premium.log
├── 10_sft.log
├── 12_grpo.log
├── 13_safety.log
├── 14_antirefusal.log
├── 22_multimodal_dl.log
├── 23_distill_vision.log
├── 24_train_stage1.log
└── ...
```

Log format:

```
[2026-01-20 17:00:00] [INFO] Starting training...
[2026-01-20 17:00:01] [INFO] Epoch 1/3
[2026-01-20 17:00:10] [INFO] Step 100, Avg Loss: 1.234
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error / validation failed |
| 130 | Interrupted by user (Ctrl+C) |

---

## See Also

- [Testing Guide](TESTING.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Universal SLI Guide](SLI_UNIVERSAL_GUIDE.md)
- [Examples](../examples/README.md)
