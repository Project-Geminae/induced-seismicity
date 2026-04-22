#!/usr/bin/env bash
# Middle-range CV-TMLE serial sweep — third worker complementing the
# forward (1-10 minus 7) and reverse (11-20) sweeps. Works on the
# mid-range radii first to reduce collision risk with the reverse worker
# when it eventually reaches these.

set -u
cd "$(dirname "$0")"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export TMLE_SKIP_GBM=1
export TMLE_CV_DENSITY=haldensify
export TMLE_USE_HAL=1

STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="cvtmle_serial_middle_${STAMP}.log"

# Middle radii — pick from the queue that reverse is least likely to
# reach soon (reverse is currently on 18, will take ~27 hours to reach
# 13). Work 13 -> 11.
RADII=(13 12 11)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
echo "CV-TMLE middle serial sweep started $(date)" | tee -a "$SWEEP_LOG"
echo "  radii: ${RADII[*]}" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"

for R in "${RADII[@]}"; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "────────────────── Radius ${R} km (started $(date +%H:%M:%S)) ──────────────────" | tee -a "$SWEEP_LOG"

    PER_R_OUT="tmle_shift_cvtmle_${R}km_mid_${STAMP}.csv"

    .venv/bin/python tmle_run_parallel.py shift \
        --window 365 --shift 0.10 \
        --radii "$R" \
        --cv-tmle --max-n 50000 \
        --workers 1 \
        --output "$PER_R_OUT" 2>&1 | tee -a "$SWEEP_LOG"

    EXIT=${PIPESTATUS[0]}
    if [ $EXIT -ne 0 ]; then
        echo "⚠️  Radius ${R} km failed (exit $EXIT). Continuing." | tee -a "$SWEEP_LOG"
        continue
    fi

    [ -s "$PER_R_OUT" ] && echo "✅  Radius ${R} km done ($(date +%H:%M:%S))" | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "Middle serial sweep complete $(date)" | tee -a "$SWEEP_LOG"
