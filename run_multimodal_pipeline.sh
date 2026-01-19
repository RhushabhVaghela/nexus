#!/bin/bash
# =============================================================================
# MANUS PRIME - Multimodal Pipeline Script
# =============================================================================
# 
# Stages:
#   1. DOWNLOAD (Script 22) - Fetch raw Vision/Audio data
#   2. DISTILL  (Script 23) - Use Teacher VLM to label data
#   3. TRAIN    (Script 24) - Train Omni-Modal Projectors (Stage 1) or Full Model (Stage 2)
#
# Usage:
#   ./run_multimodal_pipeline.sh [phase] [options]
#
# Phases:
#   download      - Run Script 22
#   distill       - Run Script 23 (requires --modality)
#   train         - Run Script 24 (requires --stage)
#   all           - Run full pipeline (Vision default)
#
# Options:
#   --modality=vision|audio|video
#   --stage=1|2
#   --teacher=mock-teacher|gpt-4v|gemini-pro
#
# =============================================================================

set -e

# Enforce 'manus' conda environment
if [ "$CONDA_DEFAULT_ENV" != "manus" ]; then
    echo -e "\033[0;31m[ERROR] This script must be run in the 'manus' conda environment.\033[0m"
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: conda activate manus"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${PROJECT_DIR}/src"
LOG_DIR="${PROJECT_DIR}/logs"

PHASE="${1:-all}"
MODALITY="vision"
STAGE=1
TEACHER="mock-teacher"
LIMIT=1000
SAMPLE_SIZE=0

for arg in "$@"; do
    case $arg in
        --modality=*) MODALITY="${arg#*=}" ;;
        --stage=*) STAGE="${arg#*=}" ;;
        --teacher=*) TEACHER="${arg#*=}" ;;
        --limit=*) LIMIT="${arg#*=}" ;;
        --sample-size=*) SAMPLE_SIZE="${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# 1. DOWNLOAD
run_download() {
    log_info "Phase 1: Download Multimodal Data (Limit: ${LIMIT})..."
    python "${SRC_DIR}/22_multimodal_pipeline.py" --phase download --limit "${LIMIT}" 2>&1 | tee "${LOG_DIR}/22_multimodal_dl.log"
}

# 2. DISTILL
run_distill() {
    log_info "Phase 2: Distill Data (Teacher: ${TEACHER}, Modality: ${MODALITY})..."
    python "${SRC_DIR}/23_multimodal_distillation.py" \
        --modality "${MODALITY}" \
        --teacher "${TEACHER}" \
        2>&1 | tee "${LOG_DIR}/23_distill_${MODALITY}.log"
}

# 3. TRAIN
run_train() {
    # Determine data path based on modality default
    DATA_PATH="/mnt/e/data/datasets"
    
    log_info "Phase 3: Train Omni-Modal Model (Stage: ${STAGE})..."
    python "${SRC_DIR}/24_multimodal_training.py" \
        --stage "${STAGE}" \
        --data-path "${DATA_PATH}" \
        --sample-size "${SAMPLE_SIZE}" \
        2>&1 | tee "${LOG_DIR}/24_train_stage${STAGE}.log"
}

# MAIN
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
        echo "Usage: ./run_multimodal_pipeline.sh [download|distill|train|all] [options]"
        exit 1
        ;;
esac

log_success "Multimodal pipeline execution finished!"
