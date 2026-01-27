# Nexus Shell Script Reference

> **Primary Entry Point:** `run_nexus_master.sh`
> This document details the usage of the "Self-Driving" Nexus Master Pipeline.

---

## 🚀 The Master Script: `run_nexus_master.sh`

This script is the **Cockpit** for the Nexus pipeline. It handles environment validation, colorful logging, and orchestration of the underlying Python engine.

### Basic Usage

```bash
./run_nexus_master.sh
```

* **Behavior**: Resumes the pipeline from where it left off (saved in `.pipeline_state.json`).
* **Checks**: Verifies `nexus` conda env, `torch`, and `registry.json` existence.

### Safe Reset

```bash
./run_nexus_master.sh --reset
```

* **Behavior**: Wipes the state file and starts fresh from Stage 1 (Profiling).
* **Use Case**: When you've changed the teacher registry or codebase and want a clean run.

### Target Specific Stages

You can surgically execute just one part of the pipeline:

```bash
# Only run the Profiler (NIWT)
./run_nexus_master.sh --stage profiling

# Only run Knowledge Extraction (SLI / Distillation)
./run_nexus_master.sh --stage knowledge_extraction

# Only train the Student
./run_nexus_master.sh --stage training

# Only train the Router
./run_nexus_master.sh --stage router_training
```

### Simulation Mode

```bash
# Verify the flow without actually training (Fast)
./run_nexus_master.sh --dry-run
```

---

## 🛠️ Diagnostics & Utilities

### `scripts/run_profiling.sh`

**Manual Profiler.**
Runs ONLY the NIWT profiling stage on a specific model path.

```bash
./scripts/run_profiling.sh --model "/path/to/model" --output "results/profiling"
```

### `scripts/run_distillation.sh`

**Manual Distiller.**
Runs ONLY the distillation training loop (NexusTrainer) with explicit paths.

```bash
./scripts/run_distillation.sh --teacher "/path/to/teacher" --student "/path/to/student"
```

### `scripts/setup_voice_models.sh`

**Dependency Installer.**
Downloads/Setups dependencies specifically for Whisper/Qwen-Audio adapters.

```bash
./scripts/setup_voice_models.sh
```

---

## ⚠️ Legacy Scripts (Archived)

> These scripts are from the "Manual Phase" and are kept for reference. They are effectively superseded by the Master Pipeline.

* `run_universal_pipeline.sh`: The old orchestrator.
* `run_multimodal_pipeline.sh`: Old vision/audio script.
* `training-suite/*.sh`: Old batch runners.

---

## 🎨 Log Output Guide

The Master Pipeline uses color-coded logs to help you monitor progress:

* [INFO] (Blue): General system status.
* [✓] (Green): Successful completion of a step.
* [⚠] (Yellow): Warnings (e.g., thermal throttling, missing registry items).
* [✗] (Red): Critical errors.
* [STAGE] (Purple): Major phase transitions.
