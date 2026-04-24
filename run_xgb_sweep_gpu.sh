#!/usr/bin/env bash
# GPU-parallel XGBoost shift sweep — distributes radii across 3 GPUs.
# B=500 bootstraps per radius. Expected: ~1 hour for all 20 radii.

set -u
cd "$(dirname "$0")"

B=${XGB_B:-500}
STAMP=$(date +%Y%m%d_%H%M%S)

# Partition radii across GPUs. Panels grow with radius so alternating
# assignment balances compute. Also skip radii we already have (they'll
# be overwritten since the GPU sweep is fast).
GPU0_RADII=(1 4 7 10 13 16 19)
GPU1_RADII=(2 5 8 11 14 17 20)
GPU2_RADII=(3 6 9 12 15 18)

# Build inline command for each GPU stream. No function embedding —
# variable expansion happens here so $B is baked into the string.
run_cmd() {
    local GPU=$1
    shift
    local RADII=("$@")
    local CMD="cd ~/induced-seismicity && source .venv/bin/activate"
    for R in "${RADII[@]}"; do
        CMD+=" && echo '[GPU${GPU}] Starting R=${R} at '\$(date +%H:%M:%S) && "
        CMD+="CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=2 .venv/bin/python run_xgb_shift_gpu.py --radius ${R} --B ${B} 2>&1 | tee xgb_r${R}_gpu${GPU}.log"
        CMD+=" && echo '[GPU${GPU}] Finished R=${R} at '\$(date +%H:%M:%S)"
    done
    CMD+=" ; exec bash"
    echo "$CMD"
}

tmux kill-session -t xgb-gpu0 2>/dev/null
tmux kill-session -t xgb-gpu1 2>/dev/null
tmux kill-session -t xgb-gpu2 2>/dev/null

tmux new-session -d -s xgb-gpu0 "$(run_cmd 0 "${GPU0_RADII[@]}")"
tmux new-session -d -s xgb-gpu1 "$(run_cmd 1 "${GPU1_RADII[@]}")"
tmux new-session -d -s xgb-gpu2 "$(run_cmd 2 "${GPU2_RADII[@]}")"

echo "Launched 3 GPU streams (B=${B} each). Monitor:"
echo "  tmux attach -t xgb-gpu0   # or gpu1, gpu2"
echo "  watch 'ls -la xgb_shift_*km.csv | wc -l'"
