#!/usr/bin/env bash
# GPU-parallel XGBoost DOSE-RESPONSE sweep: evaluates E[Y_a] at 5 dose levels
# across all 20 radii. Uses 3 GPUs. Expected ~1-2 hours.

set -u
cd "$(dirname "$0")"

B=${XGB_B:-500}
STAMP=$(date +%Y%m%d_%H%M%S)

GPU0_RADII=(1 4 7 10 13 16 19)
GPU1_RADII=(2 5 8 11 14 17 20)
GPU2_RADII=(3 6 9 12 15 18)

run_cmd() {
    local GPU=$1
    shift
    local RADII=("$@")
    local CMD="cd ~/induced-seismicity && source .venv/bin/activate"
    for R in "${RADII[@]}"; do
        CMD+=" && echo '[GPU${GPU}-dose] R=${R} start '\$(date +%H:%M:%S) && "
        CMD+="CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=2 .venv/bin/python run_xgb_dose_response_gpu.py --radius ${R} --B ${B} 2>&1 | tee xgb_dose_r${R}_gpu${GPU}.log"
        CMD+=" && echo '[GPU${GPU}-dose] R=${R} done '\$(date +%H:%M:%S)"
    done
    CMD+=" ; exec bash"
    echo "$CMD"
}

tmux kill-session -t xgb-dose-gpu0 2>/dev/null
tmux kill-session -t xgb-dose-gpu1 2>/dev/null
tmux kill-session -t xgb-dose-gpu2 2>/dev/null

tmux new-session -d -s xgb-dose-gpu0 "$(run_cmd 0 "${GPU0_RADII[@]}")"
tmux new-session -d -s xgb-dose-gpu1 "$(run_cmd 1 "${GPU1_RADII[@]}")"
tmux new-session -d -s xgb-dose-gpu2 "$(run_cmd 2 "${GPU2_RADII[@]}")"

echo "Launched 3 GPU streams (dose-response, B=${B} each)."
