#!/usr/bin/env bash
# Reverse serial CV-TMLE + haldensify + HAL sweep — second worker that
# processes radii from 20 down toward the middle, running in parallel
# with the forward serial sweep (radii 1..10 minus 7). Two workers × 1
# core each = ~8 GB peak RAM, leaves plenty of headroom for other
# services on minitim.

set -u
cd "$(dirname "$0")"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export TMLE_SKIP_GBM=1
export TMLE_CV_DENSITY=haldensify
export TMLE_USE_HAL=1

STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="cvtmle_serial_reverse_${STAMP}.log"
SUMMARY_CSV="tmle_shift_cvtmle_serial_reverse_${STAMP}.csv"

# Radii to process (20 down to 11 — forward sweep handles 1-10 minus 7)
RADII=(20 19 18 17 16 15 14 13 12 11)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
echo "CV-TMLE reverse serial sweep started $(date)" | tee -a "$SWEEP_LOG"
echo "  radii: ${RADII[*]}" | tee -a "$SWEEP_LOG"
echo "  output: $SUMMARY_CSV" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"

for R in "${RADII[@]}"; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "────────────────── Radius ${R} km (started $(date +%H:%M:%S)) ──────────────────" | tee -a "$SWEEP_LOG"

    PER_R_OUT="tmle_shift_cvtmle_${R}km_rev_${STAMP}.csv"

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

    if [ -s "$PER_R_OUT" ]; then
        if [ ! -f "$SUMMARY_CSV" ]; then
            cp "$PER_R_OUT" "$SUMMARY_CSV"
        else
            tail -n +2 "$PER_R_OUT" >> "$SUMMARY_CSV"
        fi
        echo "✅  Radius ${R} km done ($(date +%H:%M:%S))" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "Reverse serial sweep complete $(date)" | tee -a "$SWEEP_LOG"
