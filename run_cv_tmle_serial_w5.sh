#!/usr/bin/env bash
# Fifth CV-TMLE serial worker — picks up radius 9 so the forward worker
# can retire after radius 8 finishes.

set -u
cd "$(dirname "$0")"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export TMLE_SKIP_GBM=1
export TMLE_CV_DENSITY=haldensify
export TMLE_USE_HAL=1

STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="cvtmle_serial_w5_${STAMP}.log"

RADII=(9)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
echo "CV-TMLE w5 serial sweep started $(date)" | tee -a "$SWEEP_LOG"
echo "  radii: ${RADII[*]}" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"

for R in "${RADII[@]}"; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "────────────────── Radius ${R} km (started $(date +%H:%M:%S)) ──────────────────" | tee -a "$SWEEP_LOG"

    PER_R_OUT="tmle_shift_cvtmle_${R}km_w5_${STAMP}.csv"

    .venv/bin/python tmle_run_parallel.py shift \
        --window 365 --shift 0.10 \
        --radii "$R" \
        --cv-tmle --max-n 50000 \
        --workers 1 \
        --output "$PER_R_OUT" 2>&1 | tee -a "$SWEEP_LOG"

    EXIT=${PIPESTATUS[0]}
    [ -s "$PER_R_OUT" ] && echo "✅  Radius ${R} km done ($(date +%H:%M:%S))" | tee -a "$SWEEP_LOG"
done

echo "w5 serial sweep complete $(date)" | tee -a "$SWEEP_LOG"
