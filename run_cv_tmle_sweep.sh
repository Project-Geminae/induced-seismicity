#!/usr/bin/env bash
# Parallel CV-TMLE + haldensify + HAL sweep across radii.
#
# Phase 1: tiny test (2 radii, n=5000, 2 workers) to verify the parallel
#          driver + nuisance stack work end-to-end.
# Phase 2: full sweep (all 20 radii minus 7, n=50000, 16 workers).
#
# Phase 2 only runs if Phase 1 succeeds.
#
# Run from ~/induced-seismicity/ on minitim:
#   bash run_cv_tmle_sweep.sh
#
# Output CSVs:
#   tmle_shift_test_cvtmle_<timestamp>.csv   (Phase 1)
#   tmle_shift_full_cvtmle_<timestamp>.csv   (Phase 2)

set -u
cd "$(dirname "$0")"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export TMLE_SKIP_GBM=1
export TMLE_CV_DENSITY=haldensify
export TMLE_USE_HAL=1

TEST_STAMP="test_cvtmle_$(date +%Y%m%d_%H%M%S)"
FULL_STAMP="full_cvtmle_$(date +%Y%m%d_%H%M%S)"
TEST_OUT="tmle_shift_${TEST_STAMP}.csv"
FULL_OUT="tmle_shift_${FULL_STAMP}.csv"

echo "════════════════════════════════════════════════════════════════════"
echo "PHASE 1: Tiny test — 2 radii (1, 2 km), n=5000, 2 workers"
echo "════════════════════════════════════════════════════════════════════"

.venv/bin/python tmle_run_parallel.py shift \
    --window 365 --shift 0.10 \
    --radii 1 2 \
    --cv-tmle --max-n 5000 \
    --workers 2 \
    --output "$TEST_OUT" 2>&1 | tee "${TEST_STAMP}.log"

TEST_EXIT=${PIPESTATUS[0]}
if [ $TEST_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Phase 1 FAILED (exit $TEST_EXIT). Not proceeding to Phase 2."
    exit 1
fi

if [ ! -s "$TEST_OUT" ]; then
    echo ""
    echo "❌ Phase 1 produced no output CSV. Not proceeding to Phase 2."
    exit 1
fi

N_ROWS=$(wc -l < "$TEST_OUT")
if [ "$N_ROWS" -lt 3 ]; then  # header + 2 data rows
    echo ""
    echo "❌ Phase 1 CSV has < 2 data rows ($N_ROWS lines total)."
    exit 1
fi

echo ""
echo "✅ Phase 1 passed ($N_ROWS lines in $TEST_OUT)"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "PHASE 2: Full sweep — radii 1-6, 8-20 (skipping 7, already running),"
echo "         n=50000, 16 workers"
echo "════════════════════════════════════════════════════════════════════"

.venv/bin/python tmle_run_parallel.py shift \
    --window 365 --shift 0.10 \
    --radii 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --cv-tmle --max-n 50000 \
    --workers 16 \
    --output "$FULL_OUT" 2>&1 | tee "${FULL_STAMP}.log"

echo ""
echo "Done. Output: $FULL_OUT"
