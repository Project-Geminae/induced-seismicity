#!/bin/bash
echo "=== GPU 0 chain start: $(date) ==="
for R in 7 9 11 13 15 17 19; do
  /opt/induced-seismicity/scripts/run_one_radius.sh $R 0
done
echo "=== GPU 0 chain done: $(date) ==="
