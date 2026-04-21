#!/usr/bin/env bash
# Serial CV-TMLE + haldensify + HAL sweep — runs one radius at a time to
# avoid memory contention with other services on minitim (embedding
# models, openclaw gateway, etc.).
#
# Skips radius 7 (running separately in tmux cv-tmle-vdl).
# Writes per-radius CSVs so a partial sweep still has useful output.

set -u
cd "$(dirname "$0")"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export TMLE_SKIP_GBM=1
export TMLE_CV_DENSITY=haldensify
export TMLE_USE_HAL=1

STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="cvtmle_serial_sweep_${STAMP}.log"
SUMMARY_CSV="tmle_shift_cvtmle_serial_${STAMP}.csv"

# Radii to process (skip 7 — already running)
RADII=(1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
echo "CV-TMLE serial sweep started $(date)" | tee -a "$SWEEP_LOG"
echo "  radii: ${RADII[*]}" | tee -a "$SWEEP_LOG"
echo "  output: $SUMMARY_CSV" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"

for R in "${RADII[@]}"; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "────────────────── Radius ${R} km (started $(date +%H:%M:%S)) ──────────────────" | tee -a "$SWEEP_LOG"

    PER_R_OUT="tmle_shift_cvtmle_${R}km_${STAMP}.csv"

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

    # Append this radius's result to the combined summary CSV
    if [ -s "$PER_R_OUT" ]; then
        if [ ! -f "$SUMMARY_CSV" ]; then
            cp "$PER_R_OUT" "$SUMMARY_CSV"
        else
            tail -n +2 "$PER_R_OUT" >> "$SUMMARY_CSV"
        fi
        echo "✅  Radius ${R} km done ($(date +%H:%M:%S)). Appended to $SUMMARY_CSV" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
echo "Serial sweep complete $(date)" | tee -a "$SWEEP_LOG"
echo "Combined output: $SUMMARY_CSV" | tee -a "$SWEEP_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$SWEEP_LOG"
