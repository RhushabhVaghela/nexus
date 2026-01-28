#!/bin/bash
# =============================================================================
# NEXUS MASTER PIPELINE ORCHESTRATOR
# =============================================================================
#
# The "Self-Driving" engine for Universal Knowledge Distillation.
# Automates: Profiling -> Extraction (SLI) -> Distillation -> Router Training
#
# Usage:
#   ./run_nexus_master.sh [OPTIONS]
#
# =============================================================================

set -e

# ============ ENVIRONMENT CHECK ============
if [[ "$CONDA_DEFAULT_ENV" != "nexus" ]]; then
    # Try to locate the environment explicitly
    DESIRED_PYTHON=$(conda info --base)/envs/nexus/bin/python
    if [ -f "$DESIRED_PYTHON" ]; then
        export PATH="$(dirname "$DESIRED_PYTHON"):$PATH"
    else
        echo -e "\033[0;31m[Error] Must be run in 'nexus' conda environment.\033[0m"
        echo "       Please run: conda activate nexus"
        exit 1
    fi
fi

# ============ COLORS ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
log_step() { echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}\n${PURPLE}[STAGE]${NC} $1\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"; }

# ============ CONFIGURATION ============
RESET_STATE=false
DRY_RUN=false
SKIP_NON_LLM=false
TARGET_STAGE=""
SELECTED_MODELS="all"
SELECTED_DATASETS=""

# ============ HELP ============
print_usage() {
    echo -e "Usage: ./run_nexus_master.sh [OPTIONS]"
    echo ""
    echo "Control Options:"
    echo "  --reset             Delete saved pipeline state and start fresh."
    echo "  --stage <NAME>      Run only specific stage (profiling, knowledge_extraction, training, router_training)"
    echo "  --dry-run           Simulate execution without running heavy compute."
    echo "  --skip-non-llm      Skip profiling of non-LLM models."
    echo "  --models <NAMES>    Comma-separated list of teacher models or 'all' (default: all)"
    echo "  --datasets <NAMES>  Comma-separated list of datasets (e.g. 'code/stack-smol') or 'all'"
    echo "  --sample_size <N>   Number of samples for profiling/distillation (default: 50)"
    echo "  --epochs <N>        Number of training epochs (default: 1)"
    echo "  --help              Show this help message."
    echo ""
    exit 0
}

# ============ PARSE ARGUMENTS ============
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset) RESET_STATE=true ;;
        --dry-run) DRY_RUN=true ;;
        --skip-non-llm) SKIP_NON_LLM=true ;;
        --stage) TARGET_STAGE="$2"; shift ;;
        --models) SELECTED_MODELS="$2"; shift ;;
        --datasets) SELECTED_DATASETS="$2"; shift ;;
        --sample_size) SAMPLE_SIZE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --help|-h) print_usage ;;
        *) log_error "Unknown parameter: $1"; print_usage ;;
    esac
    shift
done

# ============ HEADER ============
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              NEXUS SELF-DRIVING PIPELINE v6.1                 ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Environment: ${GREEN}${CONDA_DEFAULT_ENV}${NC}"
MODE_LABEL=""
if $RESET_STATE; then
    MODE_LABEL="${RED}RESET${NC}"
elif [ -f ".pipeline_state.json" ]; then
    MODE_LABEL="${GREEN}RESUME${NC}"
else
    MODE_LABEL="${BLUE}FRESH START${NC}"
fi
echo -e "  Mode:        $MODE_LABEL"
if [ -n "$TARGET_STAGE" ]; then
    echo -e "  Target:      ${YELLOW}$TARGET_STAGE${NC}"
fi
echo ""

# ============ PRE-FLIGHT CHECKS ============
log_info "Performing system health check..."

# 1. Check Python Dependencies
python -c "import torch; import faiss; import transformers; import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    log_error "Critical dependencies missing (torch, faiss, transformers, huggingface_hub)."
    exit 1
fi
log_success "Dependencies Verified."

# 2. Check Registry Import
python -c "from src.nexus_core.towers.registry import TEACHER_REGISTRY, DATASET_REGISTRY; print(f'Loaded {len(TEACHER_REGISTRY)} models, {len(DATASET_REGISTRY)} datasets')" 2>/dev/null
if [ $? -ne 0 ]; then
    log_warn "Could not import Python Registry. Pipeline might fallback to empty defaults."
else
    log_success "Python Registry Verified."
fi

# ============ EXECUTION ============
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

if $RESET_STATE; then
    log_warn "Reset flag detected. Pipeline will clear previous artifacts in scripts/nexus_pipeline.py."
fi

CMD="python scripts/nexus_pipeline.py"

if $RESET_STATE; then
    CMD="$CMD --reset"
fi

if $DRY_RUN; then
    CMD="$CMD --dry-run"
fi

if $SKIP_NON_LLM; then
    CMD="$CMD --skip-non-llm"
fi

if [ -n "$TARGET_STAGE" ]; then
    CMD="$CMD --stage $TARGET_STAGE"
fi

if [ -n "$SELECTED_DATASETS" ]; then
    CMD="$CMD --datasets '$SELECTED_DATASETS'"
fi

if [ "$SELECTED_MODELS" != "all" ]; then
    CMD="$CMD --models '$SELECTED_MODELS'"
fi

if [ -n "$SAMPLE_SIZE" ]; then
    CMD="$CMD --sample_size $SAMPLE_SIZE"
fi

if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

log_step "Handing control to Python Orchestrator"
echo -e "${YELLOW}> Executing: $CMD${NC}"
echo ""

# Execute
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   MISSION ACCOMPLISHED                        ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_success "Nexus Pipeline finished successfully."
else
    log_error "Pipeline encountered an error."
    exit 1
fi
