#!/bin/bash
# =============================================================================
# NEXUS MASTER PIPELINE ORCHESTRATOR
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

# ===================== GLOBALS =====================
LOCKFILE="/tmp/nexus_master.pid"

# ===================== ENVIRONMENT CHECK =====================
if [[ "$CONDA_DEFAULT_ENV" != "nexus" ]]; then
    DESIRED_PYTHON=$(conda info --base)/envs/nexus/bin/python
    if [ -f "$DESIRED_PYTHON" ]; then
        export PATH="$(dirname "$DESIRED_PYTHON"):$PATH"
    else
        echo -e "\033[0;31m[Error] Must be run in 'nexus' conda environment.\033[0m"
        exit 1
    fi
fi

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
log_warn()    { echo -e "${YELLOW}[ ⚠ ]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { echo -e "\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}\n${PURPLE}[STAGE]${NC} $1\n${PURPLE}═══════════════════════════════════════════════════════════════${NC}"; }

# ===================== CONFIGURATION =====================
RESET_STATE=false
DRY_RUN=false
SKIP_NON_LLM=false
TARGET_STAGE=""
SELECTED_MODELS="all"
SELECTED_DATASETS=""

# ===================== SAFE KILL TREE =====================
kill_tree() {
    local pid="$1"
    [[ -z "$pid" || "$pid" -le 1 || "$pid" -eq "$$" ]] && return
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_tree "$child"
    done
    kill -TERM "$pid" 2>/dev/null || true
}

force_kill_tree() {
    local pid="$1"
    [[ -z "$pid" || "$pid" -le 1 || "$pid" -eq "$$" ]] && return
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        force_kill_tree "$child"
    done
    kill -KILL "$pid" 2>/dev/null || true
}

# ===================== RESET-ONLY CLEANUP =====================
cleanup_existing_processes() {
    log_warn "RESET mode active — safe Nexus cleanup"

    if [[ -f "$LOCKFILE" ]]; then
        old_pid=$(cat "$LOCKFILE" || true)
        if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
            log_warn "Terminating existing Nexus instance (PID $old_pid)"
            kill_tree "$old_pid"
            sleep 2
            ps -p "$old_pid" >/dev/null 2>&1 && force_kill_tree "$old_pid"
        fi
        rm -f "$LOCKFILE"
    fi

    log_warn "Removing lock & cache files"
    find . -type f -name "*.lock" -delete 2>/dev/null || true
    rm -rf .cache __pycache__ logs/* .pipeline_state.json 2>/dev/null || true

    log_success "RESET cleanup complete"
}

# ===================== HELP =====================
print_usage() {
    echo "Usage: ./run_nexus_master.sh [OPTIONS]"
    exit 0
}

# ===================== PARSE ARGUMENTS =====================
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
        --lr) LR="$2"; shift ;;
        --router-epochs) ROUTER_EPOCHS="$2"; shift ;;
        --router-lr) ROUTER_LR="$2"; shift ;;
        --embedding-model) EMBEDDING_MODEL="$2"; shift ;;
        --use-unsloth) USE_UNSLOTH=true ;;
        --packing) PACKING=true ;;
        --max-seq-length) MAX_SEQ_LENGTH="$2"; shift ;;
        --grpo) USE_GRPO=true ;;
        --help|-h) print_usage ;;
        *) log_error "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ===================== HEADER =====================
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              NEXUS SELF-DRIVING PIPELINE v6.1                 ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"

# ===================== RESET (ONLY HERE) =====================
if $RESET_STATE; then
    cleanup_existing_processes
else
    log_info "Reset not requested — skipping process cleanup"
fi

# ===================== INSTANCE LOCK =====================
if [[ -f "$LOCKFILE" ]]; then
    log_error "Another Nexus instance is running. Use --reset to replace it."
    exit 1
fi
echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# ===================== MONITOR UTILS =====================
MONITOR_SCRIPT="scripts/utils/monitor_utils.sh"
if [ -f "$MONITOR_SCRIPT" ]; then
    source "$MONITOR_SCRIPT"
else
    start_monitor() { echo "Starting monitor..."; }
    stop_monitor() { :; }
fi

# ===================== HEALTH CHECK =====================
log_info "Performing system health check..."
python -c "import torch, faiss, transformers, huggingface_hub" 2>/dev/null || {
    log_error "Critical dependencies missing"
    exit 1
}
log_success "Dependencies verified"

# ===================== PYTHONPATH (SAFE) =====================
PROJECT_ROOT="$(pwd)"

if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi


# ===================== EXECUTION =====================
CMD="python scripts/nexus_pipeline.py"
$RESET_STATE && CMD="$CMD --reset"
$DRY_RUN && CMD="$CMD --dry-run"
$SKIP_NON_LLM && CMD="$CMD --skip-non-llm"
[[ -n "$TARGET_STAGE" ]] && CMD="$CMD --stage $TARGET_STAGE"

log_step "Handing control to Python Orchestrator"
echo -e "${YELLOW}> Executing: $CMD${NC}"

start_monitor "Nexus Pipeline"
MONITOR_PID=$(cat .monitor_pid 2>/dev/null || true)

trap "stop_monitor $MONITOR_PID" EXIT SIGINT SIGTERM
eval $CMD
EXIT_CODE=$?
stop_monitor $MONITOR_PID

# ---------- FIX: Reset-only lock cleanup ----------
if $RESET_STATE && [ $EXIT_CODE -eq 0 ]; then
    log_info "Reset-only run completed — releasing instance lock"
    rm -f "$LOCKFILE"
fi


if [ $EXIT_CODE -eq 0 ]; then
    log_success "Nexus Pipeline finished successfully."
else
    log_error "Pipeline encountered an error."
    exit 1
fi
