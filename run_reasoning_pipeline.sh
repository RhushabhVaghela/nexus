#!/bin/bash
# =============================================================================
# Reasoning Training Pipeline - Nexus
# =============================================================================
# Complete pipeline for training models with advanced reasoning capabilities.
#
# Stages:
#   1. CoT Dataset Generation
#   2. Reasoning SFT
#   3. GRPO Training
#   4. Evaluation
#
# Usage:
#   ./run_reasoning_pipeline.sh --base-model /path/to/model --enable-cot
# =============================================================================

set -e

# Default Configuration
BASE_MODEL=""
OUTPUT_DIR="checkpoints/reasoning"
DATA_DIR="data/reasoning"

ENABLE_COT_GENERATION=false
ENABLE_SFT=true
ENABLE_GRPO=true
ENABLE_CONTEXT_EXTENSION=false
ENABLE_EVAL=true

COT_DATASET_PATH=""
COT_OUTPUT_PATH="${DATA_DIR}/cot_dataset.jsonl"
COT_NUM_SAMPLES=10000
COT_TYPE="math"

SFT_EPOCHS=3
SFT_BATCH_SIZE=2
SFT_LR="2e-5"
SFT_MAX_LENGTH=4096
SFT_LORA_R=64

GRPO_ITERATIONS=1000
GRPO_BATCH_SIZE=4
GRPO_GROUP_SIZE=4
GRPO_LR="1e-6"
GRPO_KL_COEF="0.1"

TARGET_CONTEXT_LENGTH=32768
SCALING_TYPE="yarn"

CONDA_ENV="nexus"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Reasoning Training Pipeline - Nexus

OPTIONS:
    --base-model PATH       Path to base model (required)
    --output-dir PATH       Output directory (default: checkpoints/reasoning)
    --enable-cot            Enable CoT dataset generation
    --enable-context        Enable context extension
    --skip-sft              Skip SFT stage
    --skip-grpo             Skip GRPO stage
    --cot-type TYPE         Reasoning type: math|code|logic (default: math)
    --target-context N      Target context length (default: 32768)
    --conda-env NAME        Conda environment (default: nexus)
    -h, --help              Show this help

EXAMPLES:
    $(basename "$0") --base-model /path/to/Qwen2.5-7B --enable-cot
    $(basename "$0") --base-model /path/to/model --enable-context --target-context 131072
EOF
}

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
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown: $1"; show_help; exit 1 ;;
    esac
done

if [[ -z "$BASE_MODEL" ]]; then
    echo "Error: --base-model required"
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

# Stage 0: CoT Generation
if [[ "$ENABLE_COT_GENERATION" == "true" ]]; then
    echo "[Stage 0] CoT Dataset Generation"
    conda run -n "$CONDA_ENV" python -m src.reasoning.cot_generator \
        --synthetic --output "$COT_OUTPUT_PATH" --type "$COT_TYPE" --num-samples "$COT_NUM_SAMPLES"
fi

# Stage 1: SFT
if [[ "$ENABLE_SFT" == "true" ]]; then
    echo "[Stage 1] Reasoning SFT"
    SFT_OUTPUT="${OUTPUT_DIR}/sft"
    SFT_DATASET="${COT_OUTPUT_PATH:-$COT_DATASET_PATH}"
    
    SFT_ARGS=""
    [[ "$ENABLE_CONTEXT_EXTENSION" == "true" ]] && SFT_ARGS="--extend-context --target-context $TARGET_CONTEXT_LENGTH"
    
    conda run -n "$CONDA_ENV" python -m src.stages.reasoning_sft \
        --model "$CURRENT_MODEL" --dataset "$SFT_DATASET" --output "$SFT_OUTPUT" \
        --epochs "$SFT_EPOCHS" --batch-size "$SFT_BATCH_SIZE" --lr "$SFT_LR" $SFT_ARGS
    
    CURRENT_MODEL="$SFT_OUTPUT"
fi

# Stage 2: GRPO
if [[ "$ENABLE_GRPO" == "true" ]]; then
    echo "[Stage 2] GRPO Training"
    GRPO_OUTPUT="${OUTPUT_DIR}/grpo"
    GRPO_DATASET="${DATA_DIR}/grpo_problems.jsonl"
    [[ ! -f "$GRPO_DATASET" ]] && GRPO_DATASET="$COT_OUTPUT_PATH"
    
    conda run -n "$CONDA_ENV" python -m src.stages.reasoning_grpo \
        --model "$CURRENT_MODEL" --dataset "$GRPO_DATASET" --output "$GRPO_OUTPUT" \
        --iterations "$GRPO_ITERATIONS" --batch-size "$GRPO_BATCH_SIZE"
    
    CURRENT_MODEL="$GRPO_OUTPUT"
fi

echo "============================================================"
echo "Pipeline Complete! Final model: $CURRENT_MODEL"
echo "============================================================"
