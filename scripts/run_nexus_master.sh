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
TARGET_STAGE=""

# ============ HELP ============
print_usage() {
    echo -e "Usage: ./run_nexus_master.sh [OPTIONS]"
    echo ""
    echo "Control Options:"
    echo "  --reset             Delete saved pipeline state and start fresh."
    echo "  --stage <NAME>      Run only specific stage (profiling, knowledge_extraction, training, router_training)"
    echo "  --dry-run           Simulate execution without running heavy compute."
    echo "  --help              Show this help message."
    echo ""
    exit 0
}

# ============ PARSE ARGUMENTS ============
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset) RESET_STATE=true ;;
        --dry-run) DRY_RUN=true ;;
        --stage) TARGET_STAGE="$2"; shift ;;
        --help|-h) print_usage ;;
        *) log_error "Unknown parameter: $1"; print_usage ;;
    esac
    shift
done

# ============ HEADER ============
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              NEXUS SELF-DRIVING PIPELINE v6.0                 ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Environment: ${GREEN}${CONDA_DEFAULT_ENV}${NC}"
echo -e "  Mode:        $(if $RESET_STATE; then echo "${RED}RESET${NC}"; else echo "${GREEN}RESUME${NC}"; fi)"
if [ -n "$TARGET_STAGE" ]; then
    echo -e "  Target:      ${YELLOW}$TARGET_STAGE${NC}"
fi
echo ""

# ============ PRE-FLIGHT CHECKS ============
log_info "Performing system health check..."

# 1. Check Python Dependencies
python -c "import torch; import faiss; import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    log_error "Critical dependencies missing (torch, faiss, transformers)."
    exit 1
fi
log_success "Dependencies Verified."

# 2. Check Registry
REGISTRY_FILE="src/nexus_final/registry.json"
if [ ! -f "$REGISTRY_FILE" ]; then
    log_warn "Registry not found at $REGISTRY_FILE. Pipeline may fail."
else
    TEACHER_COUNT=$(grep -c "teacher_id" "$REGISTRY_FILE" || true)
    log_success "Registry Loaded ($TEACHER_COUNT teachers found)."
fi

# ============ EXECUTION ============
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

CMD="python scripts/nexus_pipeline.py"

if $RESET_STATE; then
    CMD="$CMD --reset"
fi

if $DRY_RUN; then
    CMD="$CMD --dry-run"
fi

if [ -n "$TARGET_STAGE" ]; then
    CMD="$CMD --stage $TARGET_STAGE"
fi

log_step "Handing control to Python Orchestrator"
echo -e "${YELLOW}> Executing: $CMD${NC}"
echo ""

# Execute
$CMD

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
