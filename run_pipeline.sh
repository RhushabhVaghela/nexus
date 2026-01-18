#!/bin/bash
# =============================================================================
# MANUS PRIME - Master Pipeline Script
# =============================================================================
# 
# Main Pipeline (Text/Code):
#   01-03: Downloads (Real data, Benchmarks, Premium RLHF)
#   04:    Process real data
#   05-06: Generate synthetic data
#   07-09: Validate datasets
#   10-15: Training
#   16-21: Tools
#
#
# Usage:
#   ./run_pipeline.sh [phase] [options]
#
# Phases:
#   download    - Download text datasets (01-03)
#   process     - Process text data (04-06)
#   validate    - Validate text datasets (07-09)
#   train       - Run training pipeline (10-15)
#   all         - Run complete TEXT pipeline (01-15)
#
# Options:
#   --mode=censored|uncensored  - Training mode (default: censored)
#   --target-samples=N          - For premium datasets (default: 100000)
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
MODE="censored"
TARGET_SAMPLES=100000

for arg in "$@"; do
    case $arg in
        --mode=*) MODE="${arg#*=}" ;;
        --target-samples=*) TARGET_SAMPLES="${arg#*=}" ;;
    esac
done

mkdir -p "${LOG_DIR}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# =============================================================================
# PHASE 1: DOWNLOAD (Text Only)
# =============================================================================
run_download() {
    log_info "Phase 1: Downloading text datasets..."
    
    python "${SRC_DIR}/01_download_real_datasets.py" 2>&1 | tee "${LOG_DIR}/01_download.log"
    python "${SRC_DIR}/02_download_benchmarks.py" 2>&1 | tee "${LOG_DIR}/02_benchmarks.log"
    python "${SRC_DIR}/03_load_premium_datasets.py" --mode="${MODE}" --target-samples="${TARGET_SAMPLES}" 2>&1 | tee "${LOG_DIR}/03_premium.log"
    
    log_success "Text download phase complete"
}

# =============================================================================
# PHASE 2: PROCESS
# =============================================================================
run_process() {
    log_info "Phase 2: Processing data..."
    
    python "${SRC_DIR}/04_process_real_datasets.py" 2>&1 | tee "${LOG_DIR}/04_process.log"
    python "${SRC_DIR}/05_generate_repetitive_dataset.py" 2>&1 | tee "${LOG_DIR}/05_repetitive.log"
    python "${SRC_DIR}/06_generate_preference_dataset.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/06_preferences.log"
    
    log_success "Process phase complete"
}

# =============================================================================
# PHASE 3: VALIDATE
# =============================================================================
run_validate() {
    log_info "Phase 3: Validating datasets..."
    
    python "${SRC_DIR}/07_validate_all_datasets.py" 2>&1 | tee "${LOG_DIR}/07_validate.log"
    python "${SRC_DIR}/08_validate_benchmarks.py" 2>&1 | tee "${LOG_DIR}/08_validate_benchmarks.log"
    python "${SRC_DIR}/09_validate_premium_datasets.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/09_validate_premium.log"
    
    log_success "Validation phase complete"
}

# =============================================================================
# PHASE 4: TRAIN
# =============================================================================
run_train() {
    log_info "Phase 4: Training (mode: ${MODE})..."
    
    python "${SRC_DIR}/10_sft_training.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/10_sft.log"
    python "${SRC_DIR}/12_grpo_training.py" --mode="${MODE}" 2>&1 | tee "${LOG_DIR}/12_grpo.log"
    
    if [ "${MODE}" = "censored" ]; then
        python "${SRC_DIR}/13_safety_finetuning.py" 2>&1 | tee "${LOG_DIR}/13_safety.log"
    else
        python "${SRC_DIR}/14_anti_refusal_training.py" 2>&1 | tee "${LOG_DIR}/14_antirefusal.log"
    fi
    
    log_success "Training complete"
}

# =============================================================================
# MAIN
# =============================================================================
echo "=============================================="
echo "  MANUS PRIME PIPELINE"
echo "  Phase: ${PHASE} | Mode: ${MODE}"
echo "=============================================="

case "${PHASE}" in
    download)   run_download ;;
    process)    run_process ;;
    validate)   run_validate ;;
    train)      run_train ;;
    all)
        run_download
        run_process
        run_validate
        run_train
        ;;
    *)
        echo "Usage: ./run_pipeline.sh [download|process|validate|train|multimodal|all] [options]"
        exit 1
        ;;
esac

log_success "Pipeline execution finished!"
