#!/bin/bash
R=$1; GPU=$2
cd /opt/induced-seismicity
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "[$(date +%H:%M:%S)] R=$R START on GPU $GPU"
for f in 1 2 3 4 5; do
  echo "[$(date +%H:%M:%S)] R=$R fold $f"
  OMP_NUM_THREADS=3 python3 -u run_hurdle_full_n_cv.py --radius $R --n-folds 5 --max-irls 15 --fold-only $f > hurdle_cv_R${R}_fold${f}.log 2>&1
done
echo "[$(date +%H:%M:%S)] R=$R aggregate"
OMP_NUM_THREADS=3 python3 -u run_hurdle_full_n_cv.py --radius $R --n-folds 5 --max-irls 15 --aggregate > hurdle_cv_R${R}_aggregate.log 2>&1
echo "[$(date +%H:%M:%S)] R=$R DONE"
