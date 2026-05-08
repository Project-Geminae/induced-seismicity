#!/usr/bin/env bash
# Tmux-free parallel regHAL-TMLE sweep across 20 radii.
# Uses background processes + jobs control to limit concurrency.

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

source .venv/bin/activate

run_one() {
    local R=$1
    echo "  [$(date +%H:%M:%S)] R=$R START" | tee -a "$MASTER_LOG"
    OMP_NUM_THREADS=2 python run_reghal_tmle.py \
        --radius $R --max-n $MAX_N --max-iter $MAX_ITER \
        > "reghal_r${R}_sweep_${STAMP}.log" 2>&1
    echo "  [$(date +%H:%M:%S)] R=$R DONE" | tee -a "$MASTER_LOG"
}

# Slot-limited parallel dispatch.
for R in "${RADII[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$NBATCH" ]; do
        sleep 5
    done
    run_one $R &
done
wait

echo "" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "regHAL-TMLE sweep finished $(date)" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
