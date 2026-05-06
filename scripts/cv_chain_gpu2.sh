#!/bin/bash
echo "=== GPU 2 chain start: $(date) ==="
for R in 8 10 12 14 16 18; do
  /opt/induced-seismicity/scripts/run_one_radius.sh $R 2
done
echo "=== GPU 2 chain done: $(date) ==="
