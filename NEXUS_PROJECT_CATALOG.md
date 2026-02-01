# Nexus Project - Comprehensive Codebase Catalog

**Generated:** 2026-01-31  
**Version:** v6.1 "Beast Mode"  
**Project Type:** Universal Modular AI / Knowledge Distillation Framework  

---

## Executive Summary

Nexus is a massive AI/ML project implementing a unified, modular ecosystem that distills capabilities from 15+ specialized "Teacher" models into a single efficient "Student" architecture. It supports **135+ model architectures** via Universal SLI (Sequential Layer Ingestion) and handles text, vision, audio, and video modalities.

---

## Table of Contents

1. [Project Statistics](#project-statistics)
2. [Main Entry Points](#main-entry-points)
3. [Directory Structure Overview](#directory-structure-overview)
4. [Core Source Code (src/)](#core-source-code-src)
5. [Scripts Directory](#scripts-directory)
6. [Configuration Files](#configuration-files)
7. [Test Suite](#test-suite)
8. [Documentation](#documentation)
9. [Benchmarks](#benchmarks)
10. [Deployment](#deployment)
11. [Specialized Components](#specialized-components)
12. [File Count by Category](#file-count-by-category)

---

## Project Statistics

| Metric | Count |
|--------|-------|
| **Total Python Files** | ~400+ |
| **Total Test Files** | ~150+ |
| **Shell Scripts** | ~20+ |
| **Configuration Files** | ~30+ |
| **Documentation Files** | ~80+ |
| **Architecture Support** | 135+ models |
| **Test Coverage** | 346+ tests |

---

## Main Entry Points

### Primary Orchestration Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| [`run_nexus_master.sh`](run_nexus_master.sh) | **Master pipeline orchestrator** - Self-driving pipeline for complete distillation workflow | Root |
| [`run_pipeline.sh`](run_pipeline.sh) | Standard pipeline runner | Root |
| [`run_multimodal_pipeline.sh`](run_multimodal_pipeline.sh) | Multimodal-specific pipeline | Root |
| [`run_reasoning_pipeline.sh`](run_reasoning_pipeline.sh) | Reasoning-focused pipeline | Root |
| [`run_universal_pipeline.sh`](run_universal_pipeline.sh) | Universal architecture pipeline | Root |
| [`nexus_explain.py`](nexus_explain.py) | CLI for generating explainer videos | Root |

### Core Pipeline Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| [`scripts/nexus_pipeline.py`](scripts/nexus_pipeline.py) | Main pipeline implementation | scripts/ |
| [`scripts/nexus_server.py`](scripts/nexus_server.py) | Server deployment | scripts/ |
| [`scripts/train.py`](scripts/train.py) | Core training script | scripts/ |
| [`scripts/inference.py`](scripts/inference.py) | Inference engine | scripts/ |
| [`scripts/train_grpo.py`](scripts/train_grpo.py) | GRPO training | scripts/ |
| [`scripts/train_router.py`](scripts/train_router.py) | Router training | scripts/ |

---

## Directory Structure Overview

```
nexus/
├── src/                          # Core source code (130+ files)
│   ├── nexus_core/              # Core framework components
│   ├── nexus_final/             # Final implementation modules
│   ├── multimodal/              # Multimodal processing
│   ├── stages/                  # Pipeline stages
│   ├── reasoning/               # Reasoning capabilities
│   ├── streaming/               # Streaming/TTS components
│   ├── omni/                    # Omni-model loader
│   ├── utils/                   # Utility modules
│   └── ...
├── scripts/                     # Utility scripts (30+ files)
├── tests/                       # Test suite (150+ files)
├── docs/                        # Documentation (80+ files)
├── configs/                     # Configuration files
├── config/                      # Additional configs
├── benchmarks/                  # Benchmark scripts
├── deployment/                  # Deployment configs
├── training-suite/              # Training configurations
├── dashboard/                   # Web dashboard (React/TS)
├── remotion/                    # Video generation components
├── plans/                       # Architecture plans
├── enhanced-plan/               # Enhanced planning docs
├── conversation/                # Conversation history
├── flags/                       # Feature flags
└── ...
```

---

## Core Source Code (src/)

### 1. Nexus Core (`src/nexus_core/`)

**Purpose:** Foundation framework for knowledge distillation

| Submodule | Files | Description |
|-----------|-------|-------------|
| **towers/** | 9 files | Teacher model tower implementations |
| **student/** | 3 files | Student model core, router, sparse router |
| **training/** | 5 files | Training loop, loss functions, data loader |
| **adapters/** | 4 files | Audio, vision, reasoning adapters |
| **profiling/** | 1 file | NIWT profiler |
| **data/** | 1 file | Data sanitizer |
| **utils/** | 1 file | Universal inspector |

**Key Files:**

- [`src/nexus_core/towers/registry.py`](src/nexus_core/towers/registry.py) - Teacher model registry
- [`src/nexus_core/student/core.py`](src/nexus_core/student/core.py) - Student model core
- [`src/nexus_core/student/router.py`](src/nexus_core/student/router.py) - Intent router
- [`src/nexus_core/training/loop.py`](src/nexus_core/training/loop.py) - Training loop
- [`src/nexus_core/config.py`](src/nexus_core/config.py) - Core configuration

### 2. Nexus Final (`src/nexus_final/`)

**Purpose:** Final production-ready implementations

| Submodule | Files | Description |
|-----------|-------|-------------|
| **Root** | 18 files | Core final modules |
| **sli/** | 7 files | Sequential Layer Ingestion (Universal SLI) |
| **utils/** | 1 file | Memory utilities |

**Key Files:**

- [`src/nexus_final/architect.py`](src/nexus_final/architect.py) - Architecture management
- [`src/nexus_final/distill.py`](src/nexus_final/distill.py) - Knowledge distillation
- [`src/nexus_final/sli/universal_sli_integrator.py`](src/nexus_final/sli/universal_sli_integrator.py) - Universal SLI
- [`src/nexus_final/sli/architecture_registry.py`](src/nexus_final/sli/architecture_registry.py) - 135+ architecture registry
- [`src/nexus_final/profiler.py`](src/nexus_final/profiler.py) - NIWT profiling
- [`src/nexus_final/data_loader.py`](src/nexus_final/data_loader.py) - Data loading

### 3. Multimodal (`src/multimodal/`)

**Purpose:** Multimodal processing (vision, audio, video)

| Submodule | Files | Description |
|-----------|-------|-------------|
| **Root** | 9 files | Core multimodal modules |
| **connectors/** | 1 file | DFM connector |
| **datasets/** | 2 files | EMM1 loader, unified loader |
| **tests/** | 2 files | Test utilities |

**Key Files:**

- [`src/multimodal/model.py`](src/multimodal/model.py) - Multimodal model
- [`src/multimodal/encoders.py`](src/multimodal/encoders.py) - Multimodal encoders
- [`src/multimodal/decoders.py`](src/multimodal/decoders.py) - Multimodal decoders
- [`src/multimodal/processors.py`](src/multimodal/processors.py) - Data processors
- [`src/multimodal/distillation.py`](src/multimodal/distillation.py) - Multimodal distillation

### 4. Stages (`src/stages/`)

**Purpose:** Pipeline stage implementations (18 files)

| File | Purpose |
|------|---------|
| [`src/stages/base.py`](src/stages/base.py) | Base stage class |
| [`src/stages/stage_cot.py`](src/stages/stage_cot.py) | Chain-of-thought stage |
| [`src/stages/stage_reasoning.py`](src/stages/stage_reasoning.py) | Reasoning stage |
| [`src/stages/stage_omni.py`](src/stages/stage_omni.py) | Omni-model stage |
| [`src/stages/stage_streaming.py`](src/stages/stage_streaming.py) | Streaming stage |
| [`src/stages/stage_video.py`](src/stages/stage_video.py) | Video processing stage |
| [`src/stages/stage_vision_qa.py`](src/stages/stage_vision_qa.py) | Vision QA stage |
| [`src/stages/stage_tools.py`](src/stages/stage_tools.py) | Tool use stage |
| [`src/stages/stage_podcast.py`](src/stages/stage_podcast.py) | Podcast generation stage |
| [`src/stages/stage_thinking.py`](src/stages/stage_thinking.py) | Thinking mode stage |
| [`src/stages/reasoning_grpo.py`](src/stages/reasoning_grpo.py) | GRPO reasoning |
| [`src/stages/reasoning_sft.py`](src/stages/reasoning_sft.py) | SFT reasoning |
| [`src/stages/agent_finetune.py`](src/stages/agent_finetune.py) | Agent fine-tuning |

### 5. Reasoning (`src/reasoning/`)

**Purpose:** Advanced reasoning capabilities

| File | Purpose |
|------|---------|
| [`src/reasoning/cot_generator.py`](src/reasoning/cot_generator.py) | Chain-of-thought generation |
| [`src/reasoning/reward_functions.py`](src/reasoning/reward_functions.py) | RL reward functions |
| [`src/reasoning/ring_attention.py`](src/reasoning/ring_attention.py) | Ring attention mechanism |
| [`src/reasoning/context_extension.py`](src/reasoning/context_extension.py) | Context length extension |
| [`src/reasoning/bookmark_indexation.py`](src/reasoning/bookmark_indexation.py) | Bookmark indexing |

### 6. Streaming (`src/streaming/`)

**Purpose:** Streaming and TTS components

| File | Purpose |
|------|---------|
| [`src/streaming/joint.py`](src/streaming/joint.py) | Joint streaming |
| [`src/streaming/memory.py`](src/streaming/memory.py) | Memory management |
| [`src/streaming/tts.py`](src/streaming/tts.py) | Text-to-speech |
| [`src/streaming/vision.py`](src/streaming/vision.py) | Vision streaming |

### 7. Omni (`src/omni/`)

**Purpose:** Universal model loading

| File | Purpose |
|------|---------|
| [`src/omni/loader.py`](src/omni/loader.py) | Universal model loader |
| [`src/omni/inference.py`](src/omni/inference.py) | Omni inference |
| [`src/omni/sequential_pipeline.py`](src/omni/sequential_pipeline.py) | Sequential pipeline |
| [`src/omni/unify_checkpoints.py`](src/omni/unify_checkpoints.py) | Checkpoint unification |

### 8. Utils (`src/utils/`)

**Purpose:** Utility modules (30+ files)

**Key Files:**

- [`src/utils/hardware_optimizer.py`](src/utils/hardware_optimizer.py) - Hardware optimization
- [`src/utils/cache_manager.py`](src/utils/cache_manager.py) - Cache management
- [`src/utils/asset_manager.py`](src/utils/asset_manager.py) - Asset management
- [`src/utils/data_mixer.py`](src/utils/data_mixer.py) - Data mixing
- [`src/utils/repetition.py`](src/utils/repetition.py) - Repetition handling
- [`src/utils/metrics.py`](src/utils/metrics.py) - Metrics tracking
- [`src/utils/circuit_breaker.py`](src/utils/circuit_breaker.py) - Circuit breaker pattern
- [`src/utils/organize_datasets.py`](src/utils/organize_datasets.py) - Dataset organization

### 9. Other Core Modules

| Module | Files | Purpose |
|--------|-------|---------|
| **Benchmarks** | 9 files | Evaluation suites |
| **CLI** | 2 files | Command-line interface |
| **API** | 1 file | Explainer API |
| **Config** | 6 files | Configuration management |
| **Data** | 6 files | Data loaders and managers |
| **Inference** | 2 files | KV cache, Remotion engine |
| **Podcast** | 2 files | Podcast generation |
| **Security** | 1 file | Security audit |
| **Voice Engine** | 6 files | Voice cloning, TTS engine |

---

## Scripts Directory

**Location:** [`scripts/`](scripts/)

**Purpose:** Utility scripts for various tasks (35+ files)

| Category | Scripts |
|----------|---------|
| **Pipeline** | `nexus_pipeline.py`, `nexus_server.py`, `run_niwt_pipeline.py` |
| **Training** | `train.py`, `train_grpo.py`, `train_router.py` |
| **Inference** | `inference.py`, `inference_multimodal.py`, `debug_inference.py` |
| **Data** | `data_loader.py`, `organize_datasets.py`, `mass_sanitize_datasets.py`, `generate_dataset_registry.py` |
| **NIWT** | `niwt_core.py`, `niwt_profiler.py`, `niwt_batch_profiler.py`, `niwt_stage2_activation.py`, `niwt_stage3_spectral.py`, `niwt_stage4_consolidation.py` |
| **Benchmarking** | `benchmark_suite.py`, `benchmark_multimodal.py` |
| **Verification** | `verify_niwt_math.py`, `verify_registry_integrity.py`, `verify_step3_fix.py`, `verify_adapter_concatenation.py`, `verify_sanitizer_targeted.py` |
| **Utilities** | `inspect_model_structure.py`, `registry_dump.py`, `fuse_models.py`, `demo_librarian_rag.py` |
| **Shell Scripts** | `run_distillation.sh`, `run_profiling.sh`, `run_nexus_master.sh`, `setup_voice_models.sh` |

---

## Configuration Files

### Primary Configs (`configs/`)

| File | Purpose |
|------|---------|
| [`configs/datasets.yaml`](configs/datasets.yaml) | Dataset configurations |
| [`configs/decoders.yaml`](configs/decoders.yaml) | Decoder configurations |
| [`configs/encoders.yaml`](configs/encoders.yaml) | Encoder configurations |
| [`configs/global_config.json`](configs/global_config.json) | Global settings |
| [`configs/outputs.yaml`](configs/outputs.yaml) | Output configurations |
| [`configs/teacher_registry.json`](configs/teacher_registry.json) | Teacher model registry |

### Secondary Configs (`config/`)

| File | Purpose |
|------|---------|
| [`config/ds_config.json`](config/ds_config.json) | DeepSpeed config |
| [`config/ds_config_ultra.json`](config/ds_config_ultra.json) | DeepSpeed ultra config |
| [`config/model_config.yaml`](config/model_config.yaml) | Model configuration |
| [`config/training_config.yaml`](config/training_config.yaml) | Training settings |
| [`config/production.yaml`](config/production.yaml) | Production config |

### Source Configs (`src/config/`)

| File | Purpose |
|------|---------|
| [`src/config/datasets.yaml`](src/config/datasets.yaml) | Dataset definitions |
| [`src/config/model_config.yaml`](src/config/model_config.yaml) | Model architecture config |
| [`src/config/multimodal_datasets.yaml`](src/config/multimodal_datasets.yaml) | Multimodal datasets |
| [`src/config/memory_config.py`](src/config/memory_config.py) | Memory settings |
| [`src/config/validator.py`](src/config/validator.py) | Config validation |

---

## Test Suite

**Location:** [`tests/`](tests/)

**Total Tests:** 346+ comprehensive tests

### Test Structure

| Category | Count | Location |
|----------|-------|----------|
| **Root Tests** | 20 | [`tests/`](tests/) |
| **Unit Tests** | 95 | [`tests/unit/`](tests/unit/) |
| **Integration Tests** | 30 | [`tests/integration/`](tests/integration/) |
| **E2E Tests** | 5 | [`tests/e2e/`](tests/e2e/) |
| **Nexus Final Tests** | 10 | [`tests/nexus_final/`](tests/nexus_final/) |
| **Multimodal Tests** | 5 | [`tests/multimodal/`](tests/multimodal/) |
| **SLI Tests** | 8 | [`tests/unit/sli/`](tests/unit/sli/) |
| **Streaming Tests** | 3 | [`tests/unit_streaming/`](tests/unit_streaming/) |
| **Benchmarks** | 6 | [`tests/benchmarks/`](tests/benchmarks/) |

### Key Test Files

| File | Purpose |
|------|---------|
| [`tests/run_all_tests.py`](tests/run_all_tests.py) | Test runner |
| [`tests/test_integration.py`](tests/test_integration.py) | Integration tests |
| [`tests/test_repetition_logic.py`](tests/test_repetition_logic.py) | Repetition logic tests |
| [`tests/test_reasoning.py`](tests/test_reasoning.py) | Reasoning tests |

---

## Documentation

**Location:** [`docs/`](docs/)

**Total Documents:** 80+ files

### Main Guides

| Document | Purpose |
|----------|---------|
| [`docs/NEXUS_V6_TECHNICAL_MANUAL.md`](docs/NEXUS_V6_TECHNICAL_MANUAL.md) | Complete technical manual |
| [`docs/NEXUS_USAGE_GUIDE.md`](docs/NEXUS_USAGE_GUIDE.md) | Usage guide |
| [`docs/NEXUS_ULTIMATE_TECHNICAL_GUIDE.md`](docs/NEXUS_ULTIMATE_TECHNICAL_GUIDE.md) | Ultimate guide |
| [`docs/SLI_UNIVERSAL_GUIDE.md`](docs/SLI_UNIVERSAL_GUIDE.md) | Universal SLI guide |
| [`docs/OMNI_LOADER_GUIDE.md`](docs/OMNI_LOADER_GUIDE.md) | Omni loader guide |
| [`docs/REASONING_TRAINING.md`](docs/REASONING_TRAINING.md) | Reasoning training guide |
| [`docs/TRAINING_METHODS.md`](docs/TRAINING_METHODS.md) | Training methods |
| [`docs/OPTIMIZATION_GUIDE.md`](docs/OPTIMIZATION_GUIDE.md) | Optimization guide |
| [`docs/MIGRATION_GUIDE.md`](docs/MIGRATION_GUIDE.md) | Migration guide |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Troubleshooting |

### API Documentation (`docs/api/`)

| File | Purpose |
|------|---------|
| [`docs/api/nexus_core.rst`](docs/api/nexus_core.rst) | Nexus Core API |
| [`docs/api/multimodal.rst`](docs/api/multimodal.rst) | Multimodal API |
| [`docs/api/omni.rst`](docs/api/omni.rst) | Omni API |
| [`docs/api/reasoning.rst`](docs/api/reasoning.rst) | Reasoning API |
| [`docs/api/utils.rst`](docs/api/utils.rst) | Utils API |

### Deployment Docs (`docs/deployment/`)

| File | Purpose |
|------|---------|
| [`docs/deployment/aws.rst`](docs/deployment/aws.rst) | AWS deployment |
| [`docs/deployment/docker.rst`](docs/deployment/docker.rst) | Docker deployment |
| [`docs/deployment/gcp.rst`](docs/deployment/gcp.rst) | GCP deployment |
| [`docs/deployment/kubernetes.rst`](docs/deployment/kubernetes.rst) | K8s deployment |
| [`docs/deployment/local_development.rst`](docs/deployment/local_development.rst) | Local dev |
| [`docs/deployment/production_checklist.rst`](docs/deployment/production_checklist.rst) | Production checklist |

---

## Benchmarks

**Location:** [`benchmarks/`](benchmarks/)

| File | Purpose |
|------|---------|
| [`benchmarks/LOADER_BENCHMARK_REPORT.md`](benchmarks/LOADER_BENCHMARK_REPORT.md) | Loader performance report |
| [`benchmarks/PERFORMANCE_BASELINES.md`](benchmarks/PERFORMANCE_BASELINES.md) | Performance baselines |
| [`benchmarks/test_omni_loader_benchmark.py`](benchmarks/test_omni_loader_benchmark.py) | Omni loader benchmarks |
| [`benchmarks/test_multimodal_architect_benchmark.py`](benchmarks/test_multimodal_architect_benchmark.py) | Multimodal benchmarks |
| [`benchmarks/test_tts_benchmark.py`](benchmarks/test_tts_benchmark.py) | TTS benchmarks |
| [`benchmarks/test_video_decoder_benchmark.py`](benchmarks/test_video_decoder_benchmark.py) | Video decoder benchmarks |

---

## Deployment

**Location:** [`deployment/`](deployment/)

| File | Purpose |
|------|---------|
| [`deployment/Dockerfile`](deployment/Dockerfile) | Docker container |
| [`deployment/docker-compose.yml`](deployment/docker-compose.yml) | Docker Compose |
| [`deployment/k8s_deployment.yaml`](deployment/k8s_deployment.yaml) | Kubernetes deployment |
| [`deployment/vllm_config.json`](deployment/vllm_config.json) | vLLM configuration |
| [`deployment/packaging_spec.md`](deployment/packaging_spec.md) | Packaging specification |

---

## Specialized Components

### 1. Dashboard (`dashboard/`)

**Purpose:** Web-based monitoring dashboard (React + TypeScript)

| File | Purpose |
|------|---------|
| [`dashboard/index.html`](dashboard/index.html) | HTML entry |
| [`dashboard/package.json`](dashboard/package.json) | NPM config |
| [`dashboard/src/App.tsx`](dashboard/src/App.tsx) | Main React app |
| [`dashboard/src/main.tsx`](dashboard/src/main.tsx) | React entry |

### 2. Remotion (`remotion/`)

**Purpose:** Video generation library for explainer videos

| File | Purpose |
|------|---------|
| [`remotion/package.json`](remotion/package.json) | NPM config |
| [`remotion/src/index.ts`](remotion/src/index.ts) | Entry point |
| [`remotion/src/Root.tsx`](remotion/src/Root.tsx) | Root component |
| [`remotion/src/NexusLib/`](remotion/src/NexusLib/) | Component library (12 components) |

### 3. Training Suite (`training-suite/`)

**Purpose:** Training configuration scripts (20 files)

- Optimized and ultra configs for 1K, 10K, 50K, 100K, 500K, 1M, 5M, 10M, FULL datasets

### 4. Plans (`plans/`)

| File | Purpose |
|------|---------|
| [`plans/architecture_taxonomy.json`](plans/architecture_taxonomy.json) | Architecture taxonomy |
| [`plans/sli_universal_architecture_design.md`](plans/sli_universal_architecture_design.md) | SLI design doc |
| [`plans/UNIVERSAL_SLI_ROADMAP.md`](plans/UNIVERSAL_SLI_ROADMAP.md) | SLI roadmap |

### 5. Enhanced Plan (`enhanced-plan/`)

| File | Purpose |
|------|---------|
| Design documents (DOCX, MD, PDF) | Architecture planning |
| Analysis documents | File analysis and synthesis |

---

## File Count by Category

| Category | File Count | Percentage |
|----------|------------|------------|
| **Core Source Code** | ~150 | 35% |
| **Tests** | ~150 | 35% |
| **Scripts** | ~35 | 8% |
| **Documentation** | ~80 | 18% |
| **Configuration** | ~30 | 4% |
| **Total** | **~445** | **100%** |

---

## Key Architectural Components

### 1. Universal SLI (Sequential Layer Ingestion)

**Location:** [`src/nexus_final/sli/`](src/nexus_final/sli/)

**Purpose:** Process 135+ model architectures on consumer hardware

**Key Files:**

- [`universal_sli_integrator.py`](src/nexus_final/sli/universal_sli_integrator.py) - Main integrator
- [`architecture_registry.py`](src/nexus_final/sli/architecture_registry.py) - 135+ architecture registry
- [`layer_factory.py`](src/nexus_final/sli/layer_factory.py) - Layer factory
- [`weight_loader.py`](src/nexus_final/sli/weight_loader.py) - Weight loading
- [`moe_handler.py`](src/nexus_final/sli/moe_handler.py) - MoE handling

### 2. Sparse Intent Router

**Location:** [`src/nexus_core/student/`](src/nexus_core/student/)

**Key Files:**

- [`router.py`](src/nexus_core/student/router.py) - Intent router
- [`sparse_router.py`](src/nexus_core/student/sparse_router.py) - Sparse routing
- [`core.py`](src/nexus_core/student/core.py) - Student core

### 3. NIWT Profiler

**Location:** [`src/nexus_core/profiling/`](src/nexus_core/profiling/)

**Key Files:**

- [`niwt.py`](src/nexus_core/profiling/niwt.py) - Neural Information-Weighted Tower

### 4. Multimodal Processing

**Location:** [`src/multimodal/`](src/multimodal/)

**Components:**

- Encoders for vision/audio
- Decoders for generation
- Processors for data handling
- Distillation for knowledge transfer

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXUS PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. PROFILE (NIWT)                                          │
│     └── scripts/niwt_profiler.py                            │
│                                                             │
│  2. EXTRACT KNOWLEDGE                                       │
│     └── src/nexus_final/distill.py                          │
│     └── src/nexus_final/sli/universal_sli_integrator.py    │
│                                                             │
│  3. TRAIN STUDENT                                           │
│     └── scripts/train.py                                    │
│     └── src/nexus_core/training/loop.py                     │
│                                                             │
│  4. TRAIN ROUTER                                            │
│     └── scripts/train_router.py                             │
│     └── src/nexus_core/student/router.py                    │
│                                                             │
│  5. BENCHMARK                                               │
│     └── src/benchmarks/                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

The Nexus project is a comprehensive AI/ML framework with:

- **445+ files** across multiple categories
- **135+ supported model architectures** via Universal SLI
- **346+ comprehensive tests** ensuring reliability
- **15+ teacher models** for knowledge distillation
- **Multimodal support** (text, vision, audio, video)
- **End-to-end pipeline** from profiling to deployment
- **Production-ready** with Docker, K8s, and monitoring

This catalog serves as the foundation for Phase 2 deep-dive analysis.

---

*Generated by Nexus Codebase Explorer - Phase 1 Catalog*
