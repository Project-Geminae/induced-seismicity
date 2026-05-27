#!/usr/bin/env bash
# Parallel regHAL-TMLE sweep against an APRIL-ONLY filtered panel.
# Used to disambiguate whether the May 2026 Estimator A weakening is
# driven by data shift (May rows) or by something else (subsample seed,
# panel size). Output CSVs are reghal_shift_<R>km_apriltest.csv.

set -u
cd "$(dirname "$0")"

MAX_N=${REGHAL_MAX_N:-49519}
MAX_ITER=${REGHAL_MAX_ITER:-50}
NBATCH=${REGHAL_NBATCH:-8}
MAX_DATE=${REGHAL_MAX_DATE:-2026-04-08}
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="reghal_sweep_aprilonly_${STAMP}.log"

RADII=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "regHAL-TMLE APRIL-ONLY sweep started $(date)" | tee -a "$MASTER_LOG"
echo "  max_n = $MAX_N, NBATCH = $NBATCH, max_iter = $MAX_ITER" | tee -a "$MASTER_LOG"
echo "  max_date = $MAX_DATE" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"

source .venv/bin/activate

run_one() {
    local R=$1
    echo "  [$(date +%H:%M:%S)] R=$R START" | tee -a "$MASTER_LOG"
    OMP_NUM_THREADS=2 python run_reghal_tmle.py \
        --radius $R --max-n $MAX_N --max-iter $MAX_ITER \
        --max-date "$MAX_DATE" \
        --out "reghal_shift_${R}km_apriltest.csv" \
        > "reghal_r${R}_aprilonly_${STAMP}.log" 2>&1
    echo "  [$(date +%H:%M:%S)] R=$R DONE" | tee -a "$MASTER_LOG"
}

for R in "${RADII[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$NBATCH" ]; do
        sleep 5
    done
    run_one $R &
done
wait

echo "" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "regHAL-TMLE APRIL-ONLY sweep finished $(date)" | tee -a "$MASTER_LOG"
echo "═══════════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
