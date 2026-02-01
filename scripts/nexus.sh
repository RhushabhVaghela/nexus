#!/bin/bash
# =============================================================================
# NEXUS UNIFIED CLI SCRIPT
# =============================================================================
# A unified command-line interface for all Nexus operations.
#
# Usage: ./scripts/nexus.sh [COMMAND] [OPTIONS]
#
# Commands:
#   pipeline         Run the main text/code training pipeline
#   multimodal       Run the multimodal pipeline (vision/audio/video)
#   reasoning        Run the reasoning training pipeline
#   universal        Run the universal capability pipeline
#   master           Run the master self-driving pipeline
#   distillation     Run knowledge distillation
#   niwt             Run NIWT profiling pipeline
#   profiling        Run performance profiling
#   tests            Run the test suite
#   cleanup          Clean up temporary files and caches
#   help             Show this help message
#
# Examples:
#   ./scripts/nexus.sh pipeline --mode=censored
#   ./scripts/nexus.sh multimodal --base-model=/path/to/model
#   ./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot
#   ./scripts/nexus.sh master --reset
#   ./scripts/nexus.sh tests --unit-only
#
# =============================================================================

set -e

# ===================== GLOBAL CONFIGURATION =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"

# Ensure logs directory exists
mkdir -p "${LOG_DIR}"

# ===================== COLORS =====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { echo -e "${PURPLE}[STAGE]${NC} $1"; }

# ===================== ENVIRONMENT CHECK =====================
check_environment() {
    if [[ "$CONDA_DEFAULT_ENV" != "nexus" ]]; then
        DESIRED_PYTHON="${CONDA_PREFIX:-$HOME/miniconda3}/envs/nexus/bin/python"
        if [ -f "$DESIRED_PYTHON" ]; then
            export PATH="$(dirname "$DESIRED_PYTHON"):$PATH"
            PYTHON_CMD="$DESIRED_PYTHON"
            log_warn "Not in 'nexus' environment. Using: $DESIRED_PYTHON"
        else
            log_error "Error: Must be run in 'nexus' conda environment."
            echo "    Current: ${CONDA_DEFAULT_ENV:-None}"
            echo "    Please run: conda activate nexus"
            exit 1
        fi
    else
        PYTHON_CMD="python"
    fi
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
}

# ===================== HELP MESSAGE =====================
show_help() {
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                    NEXUS UNIFIED CLI - v6.1                               ║
║              Universal Modular AI - Knowledge Distillation                ║
╚═══════════════════════════════════════════════════════════════════════════╝

USAGE:
    ./scripts/nexus.sh [COMMAND] [OPTIONS]

COMMANDS:
    pipeline        Run the main text/code training pipeline
    multimodal      Run the multimodal pipeline (adds vision/audio/video)
    reasoning       Run the reasoning training pipeline (CoT, GRPO)
    universal       Run the universal capability pipeline
    master          Run the master self-driving pipeline
    distillation    Run knowledge distillation from teacher to student
    niwt            Run NIWT (Neural Information-Weighted Tower) profiling
    profiling       Run performance profiling on a model
    tests           Run the test suite
    cleanup         Clean up temporary files and caches
    help            Show this help message

─────────────────────────────────────────────────────────────────────────────
COMMAND DETAILS:
─────────────────────────────────────────────────────────────────────────────

▶ pipeline [download|process|validate|train|distill|all] [OPTIONS]

  Run the main text/code training pipeline.

  Options:
    --mode=censored|uncensored    Training mode (default: censored)
    --target-samples=N            Target samples for premium datasets
    --training-method=METHOD      sft|lora|qlora|dpo|grpo|orpo|distillation
    --teacher-model=PATH          Teacher model for distillation
    --distillation-alpha=FLOAT    Distillation alpha (default: 0.5)

  Examples:
    ./scripts/nexus.sh pipeline all
    ./scripts/nexus.sh pipeline train --training-method=qlora
    ./scripts/nexus.sh pipeline distill --teacher-model=/path/to/teacher

▶ multimodal [download|distill|train|all] [OPTIONS]

  Convert any text model to multimodal (Omni) capabilities.

  Options:
    --base-model=PATH             Base model path (required for train)
    --modality=vision|audio|video Modality to train
    --stage=1|2                   Training stage
    --teacher=mock-teacher|gpt-4v Teacher model for distillation
    --force                       Force training even if already Omni
    --limit=N                     Dataset sample limit

  Examples:
    ./scripts/nexus.sh multimodal all --base-model=/path/to/model
    ./scripts/nexus.sh multimodal train --base-model=/path/to/model --stage=1

▶ reasoning [OPTIONS]

  Train models with advanced reasoning capabilities (CoT, GRPO).

  Options:
    --base-model PATH             Path to base model (required)
    --output-dir PATH             Output directory
    --enable-cot                  Enable CoT dataset generation
    --enable-context              Enable context extension
    --skip-sft                    Skip SFT stage
    --skip-grpo                   Skip GRPO stage
    --cot-type TYPE               Reasoning type: math|code|logic
    --target-context N            Target context length (default: 32768)

  Examples:
    ./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-cot
    ./scripts/nexus.sh reasoning --base-model=/path/to/model --enable-context

▶ universal [OPTIONS]

  Universal pipeline for training any combination of capabilities.

  Capability Flags:
    --enable-omni                 Convert text model to Omni
    --enable-cot                  Chain-of-Thought reasoning
    --enable-reasoning            Multi-level reasoning
    --enable-thinking             Extended thinking/reflection
    --enable-tools                Function/tool calling
    --enable-streaming            Token streaming output
    --enable-podcast              NotebookLM-style podcast
    --enable-vision-qa            Image understanding
    --enable-video-understanding  Video comprehension
    --enable-tri-streaming        Real-time multimodal streaming
    --enable-image-generation     Text-to-image generation
    --enable-video-generation     Text-to-video generation
    --enable-remotion-explainer   3Blue1Brown-style video generation
    --enable-all-text             Enable all text-only capabilities
    --enable-full-omni            Enable Omni + all capabilities

  Options:
    --base-model PATH             Base model path (required)
    --output-dir PATH             Output directory
    --sample-size N               Limit samples per dataset (0=all)
    --batch-size N                Training batch size (default: 1)
    --epochs N                    Training epochs (default: 3)
    --training-method METHOD      sft|lora|qlora|dpo|grpo|orpo|distillation
    --dry-run                     Simulate training without executing
    --organize                    Auto-organize datasets before training

  Examples:
    ./scripts/nexus.sh universal --base-model=/path/to/model --enable-cot
    ./scripts/nexus.sh universal --base-model=/path/to/model --enable-full-omni
    ./scripts/nexus.sh universal --base-model=/path/to/model --enable-omni --enable-podcast

▶ master [OPTIONS]

  Run the master self-driving pipeline.

  Options:
    --reset                       Full reset: clear state and checkpoints
    --dry-run                     Simulate execution without compute
    --skip-non-llm                Skip audio/vision/multimodal models
    --stage NAME                  Run only specific stage
    --models ID1,ID2              Filter to specific teacher models
    --datasets NAME               Filter datasets
    --sample_size N               Sample size for training
    --epochs N                    Training epochs
    --use-unsloth                 Use Unsloth for faster training

  Examples:
    ./scripts/nexus.sh master --reset
    ./scripts/nexus.sh master --models "coder,vision_main" --sample_size 5000

▶ distillation [OPTIONS]

  Run knowledge distillation from teacher to student model.

  Options:
    --teacher PATH                Teacher model path
    --student PATH                Student model path
    --data PATH                   Data directory
    --alpha FLOAT                 Distillation alpha (default: 0.5)
    --temperature FLOAT           Temperature for soft targets (default: 2.0)

  Examples:
    ./scripts/nexus.sh distillation --teacher=/path/to/teacher --student=/path/to/student

▶ niwt [OPTIONS]

  Run NIWT (Neural Information-Weighted Tower) profiling pipeline.

  Options:
    --model_name NAME             Model name from registry (required)
    --batch_size N                Batch size (default: 8)
    --samples N                   Number of samples (default: 50)

  Examples:
    ./scripts/nexus.sh niwt --model_name="microsoft/Phi-3-mini-4k-instruct"

▶ profiling [OPTIONS]

  Run performance profiling on a model.

  Options:
    --model PATH                  Model path (default: /mnt/e/data/models/AgentCPM-Explore)
    --batch-size N                Batch size (default: 4)

  Examples:
    ./scripts/nexus.sh profiling --model=/path/to/model --batch-size=8

▶ tests [OPTIONS]

  Run the test suite with intelligent categorization.

  Options:
    --unit-only                   Run only unit tests
    --integration-only            Run only integration tests
    --real-models                 Include tests requiring real models
    --distributed                 Include distributed tests
    --gpu                         Include GPU tests
    --slow                        Include slow tests
    --benchmark                   Include benchmark tests
    --coverage                    Generate coverage report
    --report                      Generate JSON test report
    -v, --verbose                 Verbose output
    --all                         Run all tests including real models

  Examples:
    ./scripts/nexus.sh tests --unit-only
    ./scripts/nexus.sh tests --integration-only --verbose
    ./scripts/nexus.sh tests --coverage --report

▶ cleanup

  Clean up temporary files, caches, and old artifacts.

  Examples:
    ./scripts/nexus.sh cleanup

▶ help

  Show this help message.

─────────────────────────────────────────────────────────────────────────────
ENVIRONMENT:
─────────────────────────────────────────────────────────────────────────────

All commands expect the 'nexus' conda environment to be active:

    conda activate nexus

Logs are written to: ./logs/

─────────────────────────────────────────────────────────────────────────────
DOCUMENTATION:
─────────────────────────────────────────────────────────────────────────────

• Full Usage Guide: docs/NEXUS_USAGE_GUIDE.md
• Shell Scripts: docs/SHELL_SCRIPTS.md
• Universal SLI: docs/SLI_UNIVERSAL_GUIDE.md
• Testing: docs/TESTING.md
• Troubleshooting: docs/TROUBLESHOOTING.md

═══════════════════════════════════════════════════════════════════════════
EOF
}

# ===================== COMMAND: PIPELINE =====================
cmd_pipeline() {
    check_environment
    log_step "Running Nexus Pipeline"
    
    shift  # Remove 'pipeline' from args
    
    if [ $# -eq 0 ]; then
        set -- "all"
    fi
    
    PHASE="$1"
    shift
    
    # Default values
    MODE="censored"
    TARGET_SAMPLES=100000
    SAMPLE_SIZE=200000
    TRAINING_METHOD="sft"
    TEACHER_MODEL=""
    DISTILLATION_ALPHA=0.5
    
    # Parse remaining arguments
    for arg in "$@"; do
        case $arg in
            --mode=*) MODE="${arg#*=}" ;;
            --target-samples=*) TARGET_SAMPLES="${arg#*=}" ;;
            --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
            --training-method=*) TRAINING_METHOD="${arg#*=}" ;;
            --teacher-model=*) TEACHER_MODEL="${arg#*=}" ;;
            --distillation-alpha=*) DISTILLATION_ALPHA="${arg#*=}" ;;
        esac
    done
    
    # Colors for logging
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    NC='\033[0m'
    
    log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
    log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
    
    run_download() {
        log_info "Phase 1: Downloading text datasets..."
        $PYTHON_CMD "${SRC_DIR}/01_download_real_datasets.py" --limit="${SAMPLE_SIZE}" 2>&1 | tee "${LOG_DIR}/01_download.log"
        $PYTHON_CMD "${SRC_DIR}/02_download_benchmarks.py" 2>&1 | tee "${LOG_DIR}/02_benchmarks.log"
        $PYTHON_CMD "${SRC_DIR}/03_load_premium_datasets.py" --mode="${MODE}" --target-samples="${TARGET_SAMPLES}" 2>&1 | tee "${LOG_DIR}/03_premium.log"
        log_success "Download phase complete"
    }
    
    run_process() {
        log_info "Phase 2: Processing data..."
        $PYTHON_CMD "${SRC_DIR}/04_process_real_datasets.py" 2>&1 | tee "${LOG_DIR}/04_process.log"
        $PYTHON_CMD "${SRC_DIR}/05_generate_repetitive_dataset.py" 2>&1 | tee "${LOG_DIR}/05_repetitive.log"
        $PYTHON_CMD "${SRC_DIR}/06_generate_preference_dataset.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/06_preferences.log"
        log_success "Process phase complete"
    }
    
    run_validate() {
        log_info "Phase 3: Validating datasets..."
        $PYTHON_CMD "${SRC_DIR}/07_validate_all_datasets.py" 2>&1 | tee "${LOG_DIR}/07_validate.log"
        $PYTHON_CMD "${SRC_DIR}/08_validate_benchmarks.py" 2>&1 | tee "${LOG_DIR}/08_validate_benchmarks.log"
        $PYTHON_CMD "${SRC_DIR}/09_validate_premium_datasets.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/09_validate_premium.log"
        log_success "Validation phase complete"
    }
    
    run_train() {
        log_info "Phase 4: Training (mode: ${MODE}, method: ${TRAINING_METHOD})..."
        $PYTHON_CMD "${SRC_DIR}/10_sft_training.py" --mode="${MODE}" --training-method="${TRAINING_METHOD}" 2>&1 | tee "${LOG_DIR}/10_sft.log"
        $PYTHON_CMD "${SRC_DIR}/12_grpo_training.py" --mode="${MODE}" --training-method="${TRAINING_METHOD}" 2>&1 | tee "${LOG_DIR}/12_grpo.log"
        
        if [ "${MODE}" = "censored" ]; then
            $PYTHON_CMD "${SRC_DIR}/13_safety_finetuning.py" 2>&1 | tee "${LOG_DIR}/13_safety.log"
        else
            $PYTHON_CMD "${SRC_DIR}/14_anti_refusal_training.py" 2>&1 | tee "${LOG_DIR}/14_antirefusal.log"
        fi
        log_success "Training complete"
    }
    
    run_distill() {
        if [ -z "${TEACHER_MODEL}" ]; then
            log_info "Skipping distillation: --teacher-model not specified"
            return
        fi
        log_info "Phase 5: Distillation from teacher: ${TEACHER_MODEL}"
        $PYTHON_CMD "${SRC_DIR}/distillation.py" --mode="${MODE}" --teacher-model="${TEACHER_MODEL}" --distillation-alpha="${DISTILLATION_ALPHA}" 2>&1 | tee "${LOG_DIR}/distillation.log"
        log_success "Distillation complete"
    }
    
    echo "==============================================="
    echo "  NEXUS PRIME PIPELINE"
    echo "  Phase: ${PHASE} | Mode: ${MODE}"
    echo "  Training Method: ${TRAINING_METHOD}"
    echo "==============================================="
    
    case "${PHASE}" in
        download)   run_download ;;
        process)    run_process ;;
        validate)   run_validate ;;
        train)      run_train ;;
        distill)    run_distill ;;
        all)
            run_download
            run_process
            run_validate
            run_train
            run_distill
            ;;
        *)
            log_error "Unknown phase: ${PHASE}"
            echo "Usage: ./scripts/nexus.sh pipeline [download|process|validate|train|distill|all] [options]"
            exit 1
            ;;
    esac
    
    log_success "Pipeline execution finished!"
}

# ===================== COMMAND: MULTIMODAL =====================
cmd_multimodal() {
    check_environment
    log_step "Running Multimodal Pipeline"
    
    shift  # Remove 'multimodal' from args
    
    if [ $# -eq 0 ]; then
        set -- "all"
    fi
    
    PHASE="$1"
    shift
    
    # Defaults
    MODALITY="vision"
    STAGE=1
    TEACHER="mock-teacher"
    LIMIT=1000
    SAMPLE_SIZE=0
    BASE_MODEL=""
    FORCE_TRAIN=false
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --base-model=*) BASE_MODEL="${arg#*=}" ;;
            --force) FORCE_TRAIN=true ;;
            --modality=*) MODALITY="${arg#*=}" ;;
            --stage=*) STAGE="${arg#*=}" ;;
            --teacher=*) TEACHER="${arg#*=}" ;;
            --limit=*) LIMIT="${arg#*=}" ;;
            --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
        esac
    done
    
    run_download() {
        log_info "Phase 1: Download Multimodal Data (Limit: ${LIMIT})..."
        $PYTHON_CMD "${SRC_DIR}/22_multimodal_pipeline.py" --phase download --limit "${LIMIT}" 2>&1 | tee "${LOG_DIR}/22_multimodal_dl.log"
    }
    
    run_distill() {
        log_info "Phase 2: Distill Data (Teacher: ${TEACHER}, Modality: ${MODALITY})..."
        $PYTHON_CMD "${SRC_DIR}/23_multimodal_distillation.py" --modality "${MODALITY}" --teacher "${TEACHER}" 2>&1 | tee "${LOG_DIR}/23_distill_${MODALITY}.log"
    }
    
    run_train() {
        DATA_PATH="/mnt/e/data/datasets"
        log_info "Phase 3: Train Omni-Modal Model (Stage: ${STAGE})..."
        $PYTHON_CMD "${SRC_DIR}/24_multimodal_training.py" --stage "${STAGE}" --data-path "${DATA_PATH}" --sample-size "${SAMPLE_SIZE}" 2>&1 | tee "${LOG_DIR}/24_train_stage${STAGE}.log"
    }
    
    case "${PHASE}" in
        download) run_download ;;
        distill)  run_distill ;;
        train)    run_train ;;
        all)
            run_download
            run_distill
            run_train
            ;;
        *)
            log_error "Unknown phase: ${PHASE}"
            echo "Usage: ./scripts/nexus.sh multimodal [download|distill|train|all] [options]"
            exit 1
            ;;
    esac
    
    log_success "Multimodal pipeline execution finished!"
}

# ===================== COMMAND: REASONING =====================
cmd_reasoning() {
    check_environment
    log_step "Running Reasoning Pipeline"
    
    shift  # Remove 'reasoning' from args
    
    # Defaults
    BASE_MODEL=""
    OUTPUT_DIR="checkpoints/reasoning"
    DATA_DIR="data/reasoning"
    ENABLE_COT_GENERATION=false
    ENABLE_SFT=true
    ENABLE_GRPO=true
    ENABLE_CONTEXT_EXTENSION=false
    COT_TYPE="math"
    TARGET_CONTEXT_LENGTH=32768
    CONDA_ENV="nexus"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --base-model) BASE_MODEL="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --enable-cot) ENABLE_COT_GENERATION=true; shift ;;
            --enable-context) ENABLE_CONTEXT_EXTENSION=true; shift ;;
            --skip-sft) ENABLE_SFT=false; shift ;;
            --skip-grpo) ENABLE_GRPO=false; shift ;;
            --cot-type) COT_TYPE="$2"; shift 2 ;;
            --target-context) TARGET_CONTEXT_LENGTH="$2"; shift 2 ;;
            --conda-env) CONDA_ENV="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$BASE_MODEL" ]]; then
        log_error "Error: --base-model is required"
        exit 1
    fi
    
    echo "============================================================"
    echo "       Nexus Reasoning Pipeline"
    echo "============================================================"
    echo "Base Model: $BASE_MODEL"
    echo "Output: $OUTPUT_DIR"
    echo "Stages: CoT=$ENABLE_COT_GENERATION SFT=$ENABLE_SFT GRPO=$ENABLE_GRPO"
    echo "============================================================"
    
    mkdir -p "$OUTPUT_DIR" "$DATA_DIR"
    CURRENT_MODEL="$BASE_MODEL"
    COT_OUTPUT_PATH="${DATA_DIR}/cot_dataset.jsonl"
    
    # Stage 0: CoT Generation
    if [[ "$ENABLE_COT_GENERATION" == "true" ]]; then
        log_step "Stage 0: CoT Dataset Generation"
        conda run -n "$CONDA_ENV" python -m src.reasoning.cot_generator \
            --synthetic --output "$COT_OUTPUT_PATH" --type "$COT_TYPE" --num-samples 10000
    fi
    
    # Stage 1: SFT
    if [[ "$ENABLE_SFT" == "true" ]]; then
        log_step "Stage 1: Reasoning SFT"
        SFT_OUTPUT="${OUTPUT_DIR}/sft"
        SFT_DATASET="${COT_OUTPUT_PATH}"
        
        SFT_ARGS=""
        [[ "$ENABLE_CONTEXT_EXTENSION" == "true" ]] && SFT_ARGS="--extend-context --target-context $TARGET_CONTEXT_LENGTH"
        
        conda run -n "$CONDA_ENV" python -m src.stages.reasoning_sft \
            --model "$CURRENT_MODEL" --dataset "$SFT_DATASET" --output "$SFT_OUTPUT" \
            --epochs 3 --batch-size 2 --lr "2e-5" $SFT_ARGS
        
        CURRENT_MODEL="$SFT_OUTPUT"
    fi
    
    # Stage 2: GRPO
    if [[ "$ENABLE_GRPO" == "true" ]]; then
        log_step "Stage 2: GRPO Training"
        GRPO_OUTPUT="${OUTPUT_DIR}/grpo"
        GRPO_DATASET="${DATA_DIR}/grpo_problems.jsonl"
        [[ ! -f "$GRPO_DATASET" ]] && GRPO_DATASET="$COT_OUTPUT_PATH"
        
        conda run -n "$CONDA_ENV" python -m src.stages.reasoning_grpo \
            --model "$CURRENT_MODEL" --dataset "$GRPO_DATASET" --output "$GRPO_OUTPUT" \
            --iterations 1000 --batch-size 4
        
        CURRENT_MODEL="$GRPO_OUTPUT"
    fi
    
    echo "============================================================"
    log_success "Pipeline Complete! Final model: $CURRENT_MODEL"
    echo "============================================================"
}

# ===================== COMMAND: UNIVERSAL =====================
cmd_universal() {
    check_environment
    log_step "Running Universal Capability Pipeline"
    
    shift  # Remove 'universal' from args
    
    # Run the original universal pipeline script if it exists
    if [ -f "${PROJECT_DIR}/run_universal_pipeline.sh" ]; then
        bash "${PROJECT_DIR}/run_universal_pipeline.sh" "$@"
    else
        log_error "Universal pipeline script not found"
        exit 1
    fi
}

# ===================== COMMAND: MASTER =====================
cmd_master() {
    check_environment
    log_step "Running Nexus Master Pipeline"
    
    shift  # Remove 'master' from args
    
    # Run the original master pipeline script if it exists
    if [ -f "${PROJECT_DIR}/run_nexus_master.sh" ]; then
        bash "${PROJECT_DIR}/run_nexus_master.sh" "$@"
    else
        log_error "Master pipeline script not found"
        exit 1
    fi
}

# ===================== COMMAND: DISTILLATION =====================
cmd_distillation() {
    check_environment
    log_step "Running Knowledge Distillation"
    
    shift  # Remove 'distillation' from args
    
    # Defaults
    TEACHER_MODEL="/mnt/e/data/models/Qwen2.5-Omni"
    STUDENT_MODEL="/mnt/e/data/models/Qwen2.5-0.5B"
    DATA_DIR="/mnt/e/data/multimodal"
    ALPHA=0.5
    TEMPERATURE=2.0
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --teacher) TEACHER_MODEL="$2"; shift 2 ;;
            --student) STUDENT_MODEL="$2"; shift 2 ;;
            --data) DATA_DIR="$2"; shift 2 ;;
            --alpha) ALPHA="$2"; shift 2 ;;
            --temperature) TEMPERATURE="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    echo "Starting Multimodal Knowledge Distillation..."
    echo "Teacher: $TEACHER_MODEL"
    echo "Student: $STUDENT_MODEL"
    echo "Data:    $DATA_DIR"
    echo "Alpha:   $ALPHA"
    echo "Temperature: $TEMPERATURE"
    
    $PYTHON_CMD src/multimodal/distillation.py \
        --data-dir "$DATA_DIR" \
        --distill \
        --distill-teacher "$TEACHER_MODEL" \
        --distill-student "$STUDENT_MODEL" \
        --alpha "$ALPHA" \
        --temperature "$TEMPERATURE"
    
    if [ $? -eq 0 ]; then
        log_success "Distillation pipeline finished successfully."
    else
        log_error "Distillation pipeline failed."
        exit 1
    fi
}

# ===================== COMMAND: NIWT =====================
cmd_niwt() {
    check_environment
    log_step "Running NIWT Profiling Pipeline"
    
    shift  # Remove 'niwt' from args
    
    # Defaults
    MODEL_NAME=""
    BATCH_SIZE=8
    SAMPLES=50
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_name) MODEL_NAME="$2"; shift 2 ;;
            --batch_size) BATCH_SIZE="$2"; shift 2 ;;
            --samples) SAMPLES="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [[ -z "$MODEL_NAME" ]]; then
        log_error "Error: --model_name is required"
        echo "Usage: ./scripts/nexus.sh niwt --model_name='microsoft/Phi-3-mini-4k-instruct'"
        exit 1
    fi
    
    $PYTHON_CMD "${SCRIPT_DIR}/run_niwt_pipeline.py" \
        --model_name "$MODEL_NAME" \
        --batch_size "$BATCH_SIZE" \
        --samples "$SAMPLES"
}

# ===================== COMMAND: PROFILING =====================
cmd_profiling() {
    check_environment
    log_step "Running Performance Profiling"
    
    shift  # Remove 'profiling' from args
    
    # Defaults
    BATCH_SIZE=4
    MODEL_PATH="/mnt/e/data/models/AgentCPM-Explore"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) MODEL_PATH="$2"; shift 2 ;;
            --batch-size) BATCH_SIZE="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    echo "=== Nexus NIWT Profiler ==="
    echo "Batch Size: $BATCH_SIZE"
    echo "Model: $MODEL_PATH"
    
    # Run the profiling inline
    $PYTHON_CMD << EOFPYTHON
import os
import sys
sys.path.insert(0, '${SRC_DIR}')

from nexus_core.profiling.niwt import NIWTCore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd

MODEL_PATH = '${MODEL_PATH}'
DATA_PATH = '/mnt/e/data/benchmarks/math/openai_gsm8k/main/test-00000-of-00001.parquet'

print(f'Loading model: {MODEL_PATH}')
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True
    )
except Exception as e:
    print(f'Failed to load model: {e}')
    sys.exit(1)

# Load Data
print(f'Loading data from {DATA_PATH}')
try:
    df = pd.read_parquet(DATA_PATH)
    test_cases = []
    for _, row in df.head(20).iterrows():
        q = row['question']
        a = row['answer'].split('####')[-1].strip()
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{'role': 'user', 'content': q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f'Question: {q}\nLet\'s think step by step.\nAnswer:'
        test_cases.append((prompt, a))
except Exception as e:
    print(f'Failed to load data: {e}')
    test_cases = [('What is 2+2?', '4'), ('Who is the president?', 'Biden')]

# Run NIWT
config = {'batch_size': ${BATCH_SIZE}}
niwt = NIWTCore(model, tokenizer, config)
niwt.run_stage_1_perturbation(test_cases)
niwt.run_stage_2_activation_analysis([t[0] for t in test_cases])
print("Profiling complete!")
EOFPYTHON
}

# ===================== COMMAND: TESTS =====================
cmd_tests() {
    check_environment
    log_step "Running Test Suite"
    
    shift  # Remove 'tests' from args
    
    # Run the test script
    $PYTHON_CMD "${SCRIPT_DIR}/run_tests.py" "$@"
}

# ===================== COMMAND: CLEANUP =====================
cmd_cleanup() {
    log_step "Cleaning up Nexus codebase"
    
    echo "🧹 Cleaning codebase..."
    echo "======================="
    
    # 1. Remove Python bytecode
    echo "Removing Python bytecode..."
    find "${PROJECT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -type f -name "*.pyo" -delete 2>/dev/null || true
    echo "✓ Python bytecode removed"
    
    # 2. Remove temporary files
    echo "Removing temporary files..."
    find "${PROJECT_DIR}" -name "*.tmp" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -name ".DS_Store" -delete 2>/dev/null || true
    find "${PROJECT_DIR}" -name "Thumbs.db" -delete 2>/dev/null || true
    echo "✓ Temporary files removed"
    
    # 3. Remove lock files
    echo "Removing lock files..."
    find "${PROJECT_DIR}" -type f -name "*.lock" -delete 2>/dev/null || true
    rm -f "${PROJECT_DIR}/.pipeline_state.json" 2>/dev/null || true
    rm -f /tmp/nexus_master.pid 2>/dev/null || true
    echo "✓ Lock files removed"
    
    # 4. Ensure directories exist
    echo "Organizing directory structure..."
    mkdir -p "${PROJECT_DIR}/logs"
    mkdir -p "${PROJECT_DIR}/results"
    mkdir -p "${PROJECT_DIR}/checkpoints"
    echo "✓ Directories organized"
    
    echo ""
    log_success "Cleanup complete!"
}

# ===================== MAIN DISPATCH =====================
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    COMMAND="$1"
    
    case "$COMMAND" in
        pipeline)       cmd_pipeline "$@" ;;
        multimodal)     cmd_multimodal "$@" ;;
        reasoning)      cmd_reasoning "$@" ;;
        universal)      cmd_universal "$@" ;;
        master)         cmd_master "$@" ;;
        distillation)   cmd_distillation "$@" ;;
        niwt)           cmd_niwt "$@" ;;
        profiling)      cmd_profiling "$@" ;;
        tests)          cmd_tests "$@" ;;
        test)           cmd_tests "$@" ;;
        cleanup)        cmd_cleanup "$@" ;;
        help|--help|-h) show_help ;;
        *)
            log_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
