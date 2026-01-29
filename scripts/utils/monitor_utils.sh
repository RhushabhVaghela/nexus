#!/bin/bash
# =============================================================================
# NEXUS MONITORING UTILITIES v2.0
# Advanced TUI with Live Metrics (GPU, ETA, Step) via status.json
# =============================================================================

# Colors for TUI
M_RED='\033[0;31m'
M_GREEN='\033[0;32m'
M_YELLOW='\033[1;33m'
M_BLUE='\033[0;34m'
M_CYAN='\033[0;36m'
M_NC='\033[0m'

STATUS_FILE="results/status.json"

# Formatting helper
display_time() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Background Monitor Function
# Usage: start_monitor "Stage Name"
# Returns: PID of monitor process (store in MONITOR_PID)
start_monitor() {
    local stage_name="$1"
    local start_time=$(date +%s)
    
    # Run in background
    (
        while true; do
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - start_time))
            ELAPSED_STR=$(display_time $ELAPSED)
            
            # Default Status
            METRICS=""
            
            # Read status.json if available
            if [ -f "$STATUS_FILE" ]; then
                # Use python one-liner to parse robustly to avoid jq dependency requirement
                
                # We read into a variable carefully to avoid partial reads
                JSON_DATA=$(cat "$STATUS_FILE" 2>/dev/null || echo "{}")
                
                # Parse key metrics
                GPU_TEMP=$(echo "$JSON_DATA" | python3 -c "import sys, json; print(json.load(sys.stdin).get('gpu_temp', -1))" 2>/dev/null || echo "-1")
                ETA=$(echo "$JSON_DATA" | python3 -c "import sys, json; print(json.load(sys.stdin).get('eta', 'N/A'))" 2>/dev/null || echo "N/A")
                STEP=$(echo "$JSON_DATA" | python3 -c "import sys, json; print(json.load(sys.stdin).get('step', 0))" 2>/dev/null || echo "0")
                STATUS=$(echo "$JSON_DATA" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'running'))" 2>/dev/null || echo "running")
                
                # Format Metrics String
                METRICS=" | ${M_CYAN}Step:${M_NC} $STEP | ${M_GREEN}ETA:${M_NC} $ETA"
                
                if [ "$GPU_TEMP" -ne "-1" ]; then
                     METRICS="$METRICS | ${M_RED}GPU:${M_NC} ${GPU_TEMP}°C"
                fi
                
                if [ "$STATUS" == "paused" ]; then
                    METRICS="$METRICS | ${M_YELLOW}[PAUSED]${M_NC}"
                fi
            fi
            
            # Print TUI Line
            # \r = carriage return, \033[K = clear line
            echo -ne "\r${M_BLUE}[${stage_name}]${M_NC} ${M_YELLOW}⏱️  $ELAPSED_STR${M_NC}$METRICS" >&2
            
            sleep 1
        done
    ) &
    
    echo $! > .monitor_pid
}

# Stop Monitor Function
# Usage: stop_monitor $MONITOR_PID
stop_monitor() {
    local pid=$1
    if [ -n "$pid" ] && [ "$pid" -ne 0 ]; then
        # Use kill -9 if it doesn't stop gracefully, and kill the whole group if possible
        kill "$pid" 2>/dev/null || true
        sleep 0.2
        kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    # Clear the progress line with a final status or newline
    echo -ne "\r\033[K" >&2
    rm -f .monitor_pid
}
