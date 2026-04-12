# Migrating the induced-seismicity TMLE pipeline to minitim

The full TMLE sweep is CPU-bound (Super Learner stacking + cluster
bootstrap, ~50 minutes wall-clock for 20 radii × 3 drivers running in
parallel on the M-series Mac). Lewis prefers compute-heavy iterations on
**minitim** (Lambda Vector workstation, Ubuntu 24.04, Tailscale
`100.65.23.59`) which has substantially more cores per socket. This file is
the recipe.

## What needs to move

| Item | How | Notes |
|---|---|---|
| Source code (`.py` files, `requirements.txt`, `CHANGES.md`, etc.) | `git push` from local, `git clone` on minitim | Or `rsync -av --exclude=.venv --exclude='*.csv' ~/induced-seismicity minitim:~/` |
| Raw SWD CSV (`swd_data.csv`, ~292 MB) | `rsync -av` directly | Currently a symlink → `~/Desktop/IndividualWellwithNeighbourInducedSeismicity/swd_data.csv` |
| Raw TexNet CSV (`texnet_events.csv`, ~6 MB) | `rsync -av` | Same source dir |
| Fault shapefile bundle (`Horne_et_al._2023_MB_BSMT_FSP_V1.{shp,dbf,shx,prj}`) | Already in repo | Lives at the repo root, no separate sync needed |

## One-shot recipe

```bash
# 1. From local Mac
cd ~/induced-seismicity

# Push code (assumes a feature branch is checked out)
git add -A && git commit -m "TMLE pipeline" && git push origin HEAD

# Sync the raw data CSVs separately (they're symlinked locally; rsync
# follows the symlinks with -L)
rsync -avL \
  ~/Desktop/IndividualWellwithNeighbourInducedSeismicity/swd_data.csv \
  ~/Desktop/IndividualWellwithNeighbourInducedSeismicity/texnet_events.csv \
  minitim:~/induced-seismicity-data/

# 2. SSH into minitim
ssh minitim

# 3. On minitim
cd ~
git clone <repo-url> induced-seismicity   # or git pull if already cloned
cd induced-seismicity

# Symlink the raw data into the repo so the scripts find it
ln -sf ~/induced-seismicity-data/swd_data.csv      swd_data.csv
ln -sf ~/induced-seismicity-data/texnet_events.csv texnet_events.csv

# Build the venv
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 4. Run the full pipeline under tmux so it survives disconnect
tmux new -s induced
.venv/bin/python run_all.py 2>&1 | tee pipeline_run.log
# Ctrl-b d to detach; tmux attach -t induced to reattach

# 5. When done, sync results back to local Mac
# (run from local Mac)
rsync -av minitim:~/induced-seismicity/{causal_*,dowhy_*,tmle_*,plots/,*.png,*.mp4,CHANGES.md,pipeline_run.log} \
  ~/induced-seismicity/
```

## Tuning for minitim's hardware (validated 2026-04-11)

Minitim has 32 cores and 125 GB RAM. The recommended TMLE configuration,
based on the first successful escalation run:

```bash
# 1. SSH in
ssh minitim@100.65.23.59

# 2. On minitim — make sure the panel files exist (~10 min if not)
cd ~/induced-seismicity
.venv/bin/python run_all.py --only 0 1 2 3 4    # ingest + panels + faults

# 3. Launch all 3 TMLE drivers in parallel under tmux.
#    The exact env vars matter — see "Lessons learned" below for why.
tmux new-session -d -s tmle_shift "bash -lc \"cd ~/induced-seismicity && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMLE_N_FOLDS=5 TMLE_BIG_LIBRARY=0 TMLE_SKIP_GBM=1 TMLE_XGB_N=300 TMLE_WORKERS=10 \
  .venv/bin/python -u tmle_run_parallel.py shift --window 365 --shift 0.10 \
  &> minitim_tmle_shift.log\""

tmux new-session -d -s tmle_dose "bash -lc \"cd ~/induced-seismicity && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMLE_N_FOLDS=5 TMLE_BIG_LIBRARY=0 TMLE_SKIP_GBM=1 TMLE_XGB_N=300 TMLE_WORKERS=10 \
  .venv/bin/python -u tmle_run_parallel.py dose --window 365 --grid 1e4 1e5 1e6 1e7 1e8 \
  &> minitim_tmle_dose.log\""

tmux new-session -d -s tmle_med "bash -lc \"cd ~/induced-seismicity && \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TMLE_N_FOLDS=5 TMLE_BIG_LIBRARY=0 TMLE_SKIP_GBM=1 TMLE_XGB_N=300 TMLE_WORKERS=10 \
  .venv/bin/python -u tmle_run_parallel.py mediation --window 365 --n-iter-boot 200 \
  &> minitim_tmle_med.log\""

# 4. Watch progress
tail -f minitim_tmle_*.log    # NOT grep — see "Lessons learned"

# 5. Sync back when done
# (run from local Mac)
rsync -av minitim@100.65.23.59:~/induced-seismicity/tmle_*_365d_*.csv \
  ~/induced-seismicity/
```

**Wall-clock budget at this configuration (validated):**
- shift driver:  ~26 min for 20 radii
- dose driver:   ~28 min for 20 radii × 5 grid points
- mediation:     ~50 min for 20 radii × 200 bootstrap iters
- All three in parallel: ~50 min total wall-clock (mediation is the long pole)

### Lessons learned (the hard way)

1. **`OMP_NUM_THREADS=1` is mandatory.** Without it, xgboost defaults to
   `n_jobs=-1` and each worker tries to use all 32 cores, fighting the other
   29 workers from the parallel sweep. Single-threaded xgboost per worker is
   the right pattern for `ProcessPoolExecutor`-based outer parallelism.

2. **`TMLE_SKIP_GBM=1` is mandatory.** sklearn's `GradientBoostingClassifier`
   and `GradientBoostingRegressor` are single-threaded and have no `n_jobs`
   parameter. On a 345k-row panel with 5-fold cross-fitting × hurdle stages
   × SL CV, they take ~30 min per worker per radius and dominate runtime.
   XGBoost provides the same boosted-tree representation with multi-threading
   that we want disabled for outer parallelism.

3. **`TMLE_BIG_LIBRARY=1` is NOT worth it.** Adding `MLPRegressor`,
   `KNeighborsRegressor`, and `ExtraTreesRegressor` to the SL stack — which
   is what the env var was designed for — makes per-worker runtime balloon
   without meaningfully changing the point estimates. The Ridge + XGBoost
   stack with TMLE targeting is enough at this sample size.

4. **`&> file.log` not `| tee file.log`.** `tee` buffers stdout in line
   chunks; for a 30-minute parallel run with deferred completion lines,
   `grep` against the log returns nothing for the first ~20 minutes even
   though all the workers are progressing. Use `&>` and `tail -f` instead.

5. **`TMLE_WORKERS=10` is the sweet spot for 3-driver parallel.** That's 30
   total workers on 32 cores, with two cores left for the OS and the parent
   `ProcessPoolExecutor` schedulers. Going higher causes load >40 and CPU
   contention slowdowns; going lower wastes hardware.

6. **5-fold cross-fitting + 200 trees is the publish-quality config.** vs the
   local Mac 3-fold + 120 trees. Point estimates agree to within 5% across
   all 20 radii at this scale; CIs from the minitim run are 10-30% tighter,
   which is the bias-variance gain you'd expect from more cross-fit folds.

---

## Dashboard deployment on minitim

The web dashboard (`dashboard/server.py` + `dashboard/templates/index.html`)
runs as a long-running uvicorn service on minitim, accessed from any device
on the Tailnet via Tailscale. **Recipe:**

```bash
# 1. From local Mac — sync code + run the link-table generator on minitim
rsync -av ~/induced-seismicity/spatiotemporal_join.py \
          ~/induced-seismicity/dashboard/server.py \
          ~/induced-seismicity/dashboard/__init__.py \
          minitim@100.65.23.59:~/induced-seismicity/dashboard/
rsync -av ~/induced-seismicity/dashboard/templates/ \
          minitim@100.65.23.59:~/induced-seismicity/dashboard/templates/

# 2. SSH in and install dashboard deps if not already there
ssh minitim@100.65.23.59
cd ~/induced-seismicity
.venv/bin/pip install fastapi uvicorn pyarrow

# 3. Generate the event-well link tables (one-time, ~70 sec)
.venv/bin/python spatiotemporal_join.py --links-only
# This writes:
#   event_well_links_{1..20}km.csv         (per-radius CSVs, ~150 MB total)
#   event_well_links.parquet               (consolidated across radii)
#   event_index.json                       (per-event metadata + per-radius well counts)

# 4. Launch the dashboard in tmux so it survives disconnect
tmux new-session -d -s dashboard "bash -lc \"cd ~/induced-seismicity && \
  .venv/bin/uvicorn dashboard.server:app \
    --host 0.0.0.0 --port 8765 --log-level info \
  &> dashboard.log\""

# 5. Verify
curl -s http://100.65.23.59:8765/api/health
# {"status":"ok","loaded_at":"...","n_events":5233,"n_panel":689213,"n_links":2401199,...}
```

**Access from any device on the Tailnet:**

| URL                                            | When |
|---|---|
| `http://100.65.23.59:8765/`                    | Direct Tailscale IP |
| `http://minitim-lambda-vector:8765/`           | Tailscale MagicDNS hostname |
| `http://minitim-lambda-vector:8765/docs`       | Auto-generated FastAPI OpenAPI docs |
| `http://minitim-lambda-vector:8765/api/health` | Liveness check |

**Memory footprint:** ~500 MB resident on minitim startup (events table,
event index, link parquet, panel CSV, TMLE summary CSVs). Negligible on
minitim's 125 GB.

**Endpoints exposed:**
- `GET /` — renders the Leaflet HTML
- `GET /api/health` — liveness + row counts
- `GET /api/events?since=YYYY-MM-DD&until=YYYY-MM-DD&min_ml=2.0&limit=10000`
- `GET /api/event/{event_id}` — single event metadata
- `GET /api/event/{event_id}/wells?radius_km=7` — wells within R km, with
  full panel features (cum vol at 4 windows, vw pressure, BHP, formation,
  perf depth, days_active) on the event date
- `GET /api/tmle/summary?radius_km=7` — population-level TMLE numbers
  (TE, NDE, NIE, dose-response @ 1e7 BBL) at the chosen radius
- `GET /api/tmle/all` — full per-radius TMLE summary table
- `GET /api/wells/{api_number}/timeseries?around=YYYY-MM-DD&days=180`
  — daily volume + pressure + cum_365d for a single well around a date
  (intended for a future "well timeline" panel; currently unused by the UI)

**Restart / debug:**
```bash
ssh minitim@100.65.23.59
tmux ls                          # confirm 'dashboard' session is up
tmux attach -t dashboard         # see uvicorn live; Ctrl-b d to detach
tail -f ~/induced-seismicity/dashboard.log
tmux kill-session -t dashboard   # kill cleanly
```

**Stopping:**
```bash
ssh minitim@100.65.23.59 "tmux kill-session -t dashboard"
```

## What NOT to migrate

- The `.venv/` directory — do not rsync it. macOS arm64 wheels won't run
  on Ubuntu x86_64. Rebuild fresh from `requirements.txt`.
- The old `pipeline_output_20250604_*` directories — they're stale local
  artifacts from the OLD pipeline.
- The shapefile auxiliary files (`Horne_et_al._2023_MB_BSMT_FSP_V1.cpg`,
  `.qmd`, `.sbn`, `.sbx`) — already in the repo.

## Sanity check after migration

```bash
# On minitim, after the venv is built
.venv/bin/python -c "
import causal_core, tmle_core
import pandas as pd
panel = causal_core.load_panel('panel_with_faults_5km.csv', radius_km=5)
print(f'Panel loaded: {len(panel):,} rows, {panel.shape[1]} cols')
print('TMLE primitives importable, ready to run.')
"
```
