#!/usr/bin/env bash
# Parallel undersmoothed-HAL shift sweep across all 20 radii.
#
# Each radius runs in its own tmux session with a single Python worker.
# Workers fire at launch time but limit concurrency via tmux session count
# plus a simple batch-size throttle (NBATCH).
#
# Memory: ~10-15 GB per worker (HAL on full n). NBATCH=6 keeps peak at
# ~90 GB on minitim, leaving headroom.
#
# Runtime estimate per radius: fit ~7 min + B bootstraps × ~5 min each.
# At B=50 and 6-way parallel, 20 radii finish in ~15 hours.
#
# Run from ~/induced-seismicity/ on minitim:
#   bash run_hal_sweep.sh

set -u
cd "$(dirname "$0")"

B=${HAL_B:-50}
NBATCH=${HAL_NBATCH:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="hal_sweep_${STAMP}.log"

RADII=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "Undersmoothed-HAL shift sweep started $(date)" | tee -a "$MASTER_LOG"
echo "  B = $B bootstraps, NBATCH = $NBATCH parallel workers" | tee -a "$MASTER_LOG"
echo "  radii: ${RADII[*]}" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

# Throttle: keep at most NBATCH HAL workers alive at any time.
# Each radius runs in its own tmux session named hal-r{R}.

wait_for_slot() {
    while true; do
        ACTIVE=$(ps aux | grep run_undersmoothed_hal.py | grep -v grep | wc -l)
        if [ "$ACTIVE" -lt "$NBATCH" ]; then
            return 0
        fi
        sleep 30
    done
}

for R in "${RADII[@]}"; do
    OUT="hal_shift_${R}km.csv"
    if [ -s "$OUT" ]; then
        echo "  R=$R already has output ($OUT), skipping" | tee -a "$MASTER_LOG"
        continue
    fi
    wait_for_slot
    echo "  Launching R=$R at $(date +%H:%M:%S)" | tee -a "$MASTER_LOG"
    tmux new-session -d -s "hal-r$R" \
        "cd ~/induced-seismicity && source .venv/bin/activate && OMP_NUM_THREADS=1 .venv/bin/python run_undersmoothed_hal.py --radius $R --B $B 2>&1 | tee hal_r${R}_${STAMP}.log; exec bash"
    sleep 5  # stagger launches so R initialization doesn't collide
done

echo "" | tee -a "$MASTER_LOG"
echo "All 20 radii dispatched. Wait for completion via: watch 'ls -la hal_shift_*km.csv | wc -l'" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
