#!/usr/bin/env bash
# Parallel regHAL-TMLE sweep across all 20 radii.
# Each radius runs in its own tmux session (NBATCH concurrent workers).
# Expected: ~5 min per radius with line-search Newton, 4 workers in
# parallel → ~25 min for full sweep. CPU-only (no GPU involvement).

set -u
cd "$(dirname "$0")"

MAX_N=${REGHAL_MAX_N:-50000}
MAX_ITER=${REGHAL_MAX_ITER:-50}
NBATCH=${REGHAL_NBATCH:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="reghal_sweep_${STAMP}.log"

RADII=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "regHAL-TMLE sweep started $(date)" | tee -a "$MASTER_LOG"
echo "  max_n = $MAX_N, NBATCH = $NBATCH, max_iter = $MAX_ITER" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

wait_for_slot() {
    while true; do
        ACTIVE=$(ps aux | grep run_reghal_tmle.py | grep -v grep | wc -l)
        if [ "$ACTIVE" -lt "$NBATCH" ]; then
            return 0
        fi
        sleep 15
    done
}

for R in "${RADII[@]}"; do
    OUT="reghal_shift_${R}km.csv"
    if [ -s "$OUT" ]; then
        # Overwrite — the earlier r=1,7,13,19 runs used the pre-line-search
        # code, so we want fresh results under the current targeting loop.
        :
    fi
    wait_for_slot
    echo "  Launching R=$R at $(date +%H:%M:%S)" | tee -a "$MASTER_LOG"
    tmux new-session -d -s "reghal-r$R" \
        "cd ~/induced-seismicity && source .venv/bin/activate && OMP_NUM_THREADS=2 .venv/bin/python run_reghal_tmle.py --radius $R --max-n $MAX_N --max-iter $MAX_ITER 2>&1 | tee reghal_r${R}_sweep_${STAMP}.log; exec bash"
    sleep 4
done

echo "" | tee -a "$MASTER_LOG"
echo "All 20 dispatched. Watch: 'ls -la reghal_shift_*km.csv | wc -l'" | tee -a "$MASTER_LOG"
