#!/usr/bin/env bash
# XGBoost-GPU seed-sensitivity sweep: for each of 5 seeds (seed 2..6),
# run a B=200 bootstrap at all 20 radii. Compares CI coverage across
# seeds. Expected: ~2 hours on 3 GPUs.
#
# This starts AFTER the dose-response sweep finishes to avoid GPU contention.
# We poll for dose-response completion first.

set -u
cd "$(dirname "$0")"

B=200
STAMP=$(date +%Y%m%d_%H%M%S)

# Wait for dose-response sweep to finish by watching tmux sessions
wait_dose_done() {
    while true; do
        ACTIVE=$(tmux ls 2>/dev/null | grep -c "xgb-dose-gpu")
        if [ "$ACTIVE" -eq 0 ]; then
            return 0
        fi
        sleep 60
    done
}

echo "Waiting for dose-response sweep to complete..."
wait_dose_done
echo "Dose-response done. Starting seed-sensitivity sweep at $(date)"

# Cycle through seeds 2-6, each running a 20-radius sweep across 3 GPUs.
# Within each seed, same GPU assignment as the shift sweep.
GPU0_RADII=(1 4 7 10 13 16 19)
GPU1_RADII=(2 5 8 11 14 17 20)
GPU2_RADII=(3 6 9 12 15 18)

for SEED in 2 3 4 5 6; do
    echo "═══ Seed $SEED starting at $(date +%H:%M:%S) ═══"

    seed_cmd() {
        local GPU=$1
        shift
        local RADII=("$@")
        local CMD="cd ~/induced-seismicity && source .venv/bin/activate"
        for R in "${RADII[@]}"; do
            CMD+=" && echo '[seed${SEED} gpu${GPU}] R=${R}' && "
            CMD+="CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=2 .venv/bin/python run_xgb_shift_gpu.py --radius ${R} --B ${B} --seed ${SEED} 2>&1 | tee xgb_seed${SEED}_r${R}.log && "
            CMD+="mv xgb_shift_${R}km.csv xgb_shift_${R}km_seed${SEED}.csv"
        done
        echo "$CMD"
    }

    tmux new-session -d -s xgb-s${SEED}-g0 "$(seed_cmd 0 "${GPU0_RADII[@]}"); exec bash"
    tmux new-session -d -s xgb-s${SEED}-g1 "$(seed_cmd 1 "${GPU1_RADII[@]}"); exec bash"
    tmux new-session -d -s xgb-s${SEED}-g2 "$(seed_cmd 2 "${GPU2_RADII[@]}"); exec bash"

    # Wait for this seed's 3 GPU streams to finish before moving to next seed
    while true; do
        ACTIVE=$(tmux ls 2>/dev/null | grep -c "xgb-s${SEED}-g")
        if [ "$ACTIVE" -eq 0 ]; then
            break
        fi
        sleep 60
    done
    echo "═══ Seed $SEED done at $(date +%H:%M:%S) ═══"
done

echo "All 5 seed sweeps complete at $(date)"
