"""
Nexus Centralized Path Configuration
=====================================

All filesystem paths used across the Nexus project are defined here.
Instead of hardcoding machine-specific paths (e.g. /mnt/e/data), every module
should import from this file.

Environment Variables
---------------------
NEXUS_DATA_ROOT       Base data directory         (default: /mnt/e/data)
NEXUS_MODELS_ROOT     Models directory            (default: $NEXUS_DATA_ROOT/models)
NEXUS_OUTPUT_ROOT     Output directory            (default: $NEXUS_DATA_ROOT/output)
NEXUS_DATASETS_ROOT   Datasets directory          (default: $NEXUS_DATA_ROOT/datasets)

Example usage::

    from nexus.config.paths import DATA_ROOT, DEFAULT_LLM_MODEL, DATASETS_DIR
"""

import os
from pathlib import Path

# ── Root Directories ──────────────────────────────────────────────────────────

DATA_ROOT = os.environ.get("NEXUS_DATA_ROOT", "/mnt/e/data")
MODELS_DIR = os.environ.get("NEXUS_MODELS_ROOT", os.path.join(DATA_ROOT, "models"))
OUTPUT_DIR = os.environ.get("NEXUS_OUTPUT_ROOT", os.path.join(DATA_ROOT, "output"))
DATASETS_DIR = os.environ.get(
    "NEXUS_DATASETS_ROOT", os.path.join(DATA_ROOT, "datasets")
)

# ── Derived Directories ──────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
TRAINING_DIR = os.path.join(DATA_ROOT, "training")
ENCODERS_DIR = os.path.join(DATA_ROOT, "encoders")
UNIFIED_MULTIMODAL_DIR = os.path.join(DATA_ROOT, "unified_multimodal")
MIXED_TRAINING_DIR = os.path.join(DATA_ROOT, "mixed-training")
MULTIMODAL_DIR = os.path.join(DATA_ROOT, "multimodal")
DOWNLOADED_DIR = os.path.join(DATA_ROOT, "downloaded")
PREMIUM_DIR = os.path.join(DATA_ROOT, "premium")
REAL_DATASETS_DIR = os.path.join(DATA_ROOT, "real-datasets")
MM_RAW_DIR = os.path.join(DATA_ROOT, "mm_raw")
MM_RAW_VIDEOS_DIR = os.path.join(MM_RAW_DIR, "videos")

# ── Model Paths ──────────────────────────────────────────────────────────────

DEFAULT_LLM_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-Omni-7B-GPTQ-Int4")
DEFAULT_STUDENT_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-0.5B")
DEFAULT_OMNI_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-Omni")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

# ── Encoder Paths ────────────────────────────────────────────────────────────

VISION_ENCODERS_DIR = os.path.join(ENCODERS_DIR, "vision-encoders")
AUDIO_ENCODERS_DIR = os.path.join(ENCODERS_DIR, "audio-encoders")
DEFAULT_VISION_ENCODER = os.path.join(VISION_ENCODERS_DIR, "siglip2-so400m-patch16-512")
DEFAULT_AUDIO_ENCODER = os.path.join(AUDIO_ENCODERS_DIR, "whisper-large-v3-turbo")

# ── Dataset Paths ────────────────────────────────────────────────────────────

REMOTION_DATASET_DIR = os.path.join(DATASETS_DIR, "remotion")
COMMON_VOICE_DIR = os.path.join(DATASETS_DIR, "Mozilla_Common-Voice")

# ── Output Paths ─────────────────────────────────────────────────────────────

TRAINED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "trained")
REMOTION_EXPLAINER_DIR = os.path.join(TRAINED_OUTPUT_DIR, "remotion-explainer")
DPO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dpo")
ORPO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "orpo")
PPO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ppo")

# ── Standalone Output Dirs (outside DATA_ROOT) ───────────────────────────────

_models_output_base = os.environ.get(
    "NEXUS_MODELS_OUTPUT",
    os.path.dirname(DATA_ROOT),  # e.g. /mnt/e
)
NEXUS_PRIME_DIR = os.path.join(_models_output_base, "models", "nexus-prime")
NEXUS_PRIME_CPT_DIR = os.path.join(_models_output_base, "models", "nexus-prime-cpt")

# ── Synthetic Dataset Dirs ───────────────────────────────────────────────────

FINETUNED_FULLSTACK_DIR = os.path.join(DATA_ROOT, "finetuned-fullstack-dataset")
REPETITIVE_PROMPT_DIR = os.path.join(DATA_ROOT, "repetitive-prompt-dataset")
REPETITIVE_QUERY_DIR = os.path.join(DATA_ROOT, "repetitive-query-dataset")
ARCHITECTURE_REASONING_DIR = os.path.join(DATA_ROOT, "architecture-reasoning-dataset")
QA_ENGINEERING_DIR = os.path.join(DATA_ROOT, "qa-engineering-dataset")
UIUX_DESIGN_DIR = os.path.join(DATA_ROOT, "uiux-design-dataset")
DEVOPS_ENGINEERING_DIR = os.path.join(DATA_ROOT, "devops-engineering-dataset")
MULTIMODAL_FULLSTACK_DIR = os.path.join(DATA_ROOT, "multimodal-fullstack-dataset")
MULTIMODAL_PODCAST_DIR = os.path.join(DATA_ROOT, "multimodal-podcast-dataset")
SAFETY_ALIGNMENT_DIR = os.path.join(DATA_ROOT, "safety-alignment")
ANTI_REFUSAL_DIR = os.path.join(DATA_ROOT, "anti-refusal")
PREFERENCE_PAIRS_DIR = os.path.join(DATA_ROOT, "preference-pairs")

# ── Corruption tracker log ───────────────────────────────────────────────────

CORRUPTED_FILES_LOG = os.path.join(OUTPUT_DIR, "corrupted_files.log")


__all__ = [
    # Roots
    "DATA_ROOT",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "DATASETS_DIR",
    # Derived
    "PROCESSED_DIR",
    "TRAINING_DIR",
    "ENCODERS_DIR",
    "UNIFIED_MULTIMODAL_DIR",
    "MIXED_TRAINING_DIR",
    "MULTIMODAL_DIR",
    "DOWNLOADED_DIR",
    "PREMIUM_DIR",
    "REAL_DATASETS_DIR",
    "MM_RAW_DIR",
    "MM_RAW_VIDEOS_DIR",
    # Models
    "DEFAULT_LLM_MODEL",
    "DEFAULT_STUDENT_MODEL",
    "DEFAULT_OMNI_MODEL",
    "CHECKPOINT_DIR",
    # Encoders
    "VISION_ENCODERS_DIR",
    "AUDIO_ENCODERS_DIR",
    "DEFAULT_VISION_ENCODER",
    "DEFAULT_AUDIO_ENCODER",
    # Datasets
    "REMOTION_DATASET_DIR",
    "COMMON_VOICE_DIR",
    # Output
    "TRAINED_OUTPUT_DIR",
    "REMOTION_EXPLAINER_DIR",
    "DPO_OUTPUT_DIR",
    "ORPO_OUTPUT_DIR",
    "PPO_OUTPUT_DIR",
    "NEXUS_PRIME_DIR",
    "NEXUS_PRIME_CPT_DIR",
    # Synthetic datasets
    "FINETUNED_FULLSTACK_DIR",
    "REPETITIVE_PROMPT_DIR",
    "REPETITIVE_QUERY_DIR",
    "ARCHITECTURE_REASONING_DIR",
    "QA_ENGINEERING_DIR",
    "UIUX_DESIGN_DIR",
    "DEVOPS_ENGINEERING_DIR",
    "MULTIMODAL_FULLSTACK_DIR",
    "MULTIMODAL_PODCAST_DIR",
    "SAFETY_ALIGNMENT_DIR",
    "ANTI_REFUSAL_DIR",
    "PREFERENCE_PAIRS_DIR",
    # Misc
    "CORRUPTED_FILES_LOG",
]
