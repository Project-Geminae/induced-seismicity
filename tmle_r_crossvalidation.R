#!/usr/bin/env Rscript
# tmle_r_crossvalidation.R
# ────────────────────────
# Cross-validate the pure-Python TMLE shift implementation against the
# canonical R `tlverse` reference (sl3 + tmle3 + txshift).
#
# Run on minitim:
#   cd ~/induced-seismicity
#   Rscript tmle_r_crossvalidation.R [RADIUS_KM] [WINDOW_DAYS]
#
# Default: RADIUS_KM = 7, WINDOW_DAYS = 365
#
# Output: tmle_r_crossvalidation_<radius>km_<timestamp>.csv
#
# Compares against the most recent tmle_shift_*.csv produced by the Python
# pipeline at the same radius. The two should agree within bootstrap noise
# — if they diverge by more than ~2σ on the point estimate, the Python
# implementation has a bug.

# ──────────────────── Args ────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
RADIUS_KM   <- ifelse(length(args) >= 1, as.numeric(args[1]), 7)
WINDOW_DAYS <- ifelse(length(args) >= 2, as.numeric(args[2]), 365)
SHIFT_PCT   <- 0.10

cat(sprintf("R cross-validation: radius=%dkm window=%dd shift=%.0f%%\n",
            RADIUS_KM, WINDOW_DAYS, SHIFT_PCT * 100))

# ──────────────────── Libraries ───────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(sl3)
  library(tmle3)
  library(txshift)
})

# ──────────────────── Load panel ──────────────────────
panel_path <- sprintf("panel_with_faults_%dkm.csv", RADIUS_KM)
if (!file.exists(panel_path)) {
  stop(sprintf("Panel file not found: %s", panel_path))
}
cat(sprintf("Loading %s …\n", panel_path))
panel <- fread(panel_path)
cat(sprintf("  %d rows, %d cols\n", nrow(panel), ncol(panel)))

# ──────────────────── Aggregate to (date, cluster) ────
cat("Aggregating to (date, ~5km cluster) cells …\n")
LAT_BIN <- 0.05
LON_BIN <- 0.05
panel[, lat_bin := round(`Surface Latitude`  / LAT_BIN) * LAT_BIN]
panel[, lon_bin := round(`Surface Longitude` / LON_BIN) * LON_BIN]

W_col <- sprintf("cum_vol_%dd_BBL", WINDOW_DAYS)
P_col <- sprintf("bhp_vw_avg_%dd",  WINDOW_DAYS)
seg_col <- sprintf("Fault Segments <= %d km", RADIUS_KM)

# Volume-weighted pressure aggregation across wells in same cluster-day
agg <- panel[, .(
    well_count        = .N,
    cum_vol_w         = sum(get(W_col), na.rm = TRUE),
    pv_w              = sum(get(P_col) * get(W_col), na.rm = TRUE),
    nearest_fault_km  = min(`Nearest Fault Dist (km)`, na.rm = TRUE),
    fault_segs        = mean(get(seg_col), na.rm = TRUE),
    perf_depth_ft     = mean(perf_depth_ft, na.rm = TRUE),
    days_active       = mean(days_active, na.rm = TRUE),
    formation         = names(sort(table(formation), decreasing = TRUE))[1],
    Y                 = max(outcome_max_ML, na.rm = TRUE)
  ),
  by = .(`Date of Injection`, lat_bin, lon_bin)]

# Volume-weighted mean pressure
agg[, vw_press := ifelse(cum_vol_w > 0, pv_w / cum_vol_w, NA_real_)]
agg[, pv_w := NULL]

# Drop rows missing the treatment / mediator
agg <- agg[!is.na(cum_vol_w) & !is.na(vw_press) & !is.na(Y)]
cat(sprintf("  %d cluster-day cells (%d with Y > 0)\n",
            nrow(agg), sum(agg$Y > 0)))

# Cluster id (for clustering in tmle3)
agg[, cluster_id := paste0(round(lat_bin, 2), "_", round(lon_bin, 2))]

# One-hot formation (top-K + OTHER)
top_forms <- names(sort(table(agg$formation), decreasing = TRUE))[1:6]
agg[, form := ifelse(formation %in% top_forms, formation, "OTHER")]
form_dummies <- model.matrix(~ form - 1, data = agg)
agg <- cbind(agg, form_dummies)

# ──────────────────── txshift TMLE ────────────────────
cat(sprintf("Fitting txshift TMLE: shift +%.0f%%\n", SHIFT_PCT * 100))

# Multiplicative shift function
shift_fn   <- function(tx, delta) tx * (1 + delta)
shift_inv  <- function(tx, delta) tx / (1 + delta)

# Super Learner library — match the Python stack as closely as practical
sl_lrnrs <- Stack$new(
  Lrnr_glm$new(),
  Lrnr_xgboost$new(nrounds = 200, max_depth = 4, eta = 0.05),
  Lrnr_ranger$new(num.trees = 200, max.depth = 12)
)
sl_meta <- Lrnr_nnls$new()
sl_Q <- Lrnr_sl$new(learners = sl_lrnrs, metalearner = sl_meta)
sl_g <- Lrnr_sl$new(learners = sl_lrnrs, metalearner = sl_meta)

# Confounder columns (drop the first dummy to avoid collinearity)
form_cols <- colnames(form_dummies)[-1]
W_cols <- c("nearest_fault_km", "fault_segs", "perf_depth_ft",
            "days_active", form_cols)

t0 <- Sys.time()
fit <- tryCatch({
  txshift(
    W      = as.matrix(agg[, ..W_cols]),
    A      = agg$cum_vol_w,
    Y      = agg$Y,
    delta  = SHIFT_PCT,
    g_exp_fit_args = list(fit_type = "sl",
                           sl_learners_density = sl_g),
    Q_fit_args     = list(fit_type = "sl",
                           sl_learners = sl_Q),
    eif_reg_type    = "hal",
    estimator       = "tmle"
  )
}, error = function(e) {
  cat("ERROR in txshift:\n")
  cat(conditionMessage(e), "\n")
  NULL
})

elapsed <- as.numeric(Sys.time() - t0, units = "secs")

if (is.null(fit)) {
  cat("R cross-validation failed.\n")
  quit(status = 1)
}

cat(sprintf("\nR txshift TMLE result (radius=%dkm, window=%dd, shift=+%.0f%%):\n",
            RADIUS_KM, WINDOW_DAYS, SHIFT_PCT * 100))
print(fit)

# ──────────────────── Compare to Python result ────────
shift_files <- list.files(pattern = "tmle_shift_.*\\.csv$")
shift_files <- shift_files[order(file.info(shift_files)$mtime, decreasing = TRUE)]
if (length(shift_files) > 0) {
  py <- fread(shift_files[1])
  py_R <- py[radius_km == RADIUS_KM & abs(shift_pct - SHIFT_PCT) < 0.001]
  if (nrow(py_R) > 0) {
    cat(sprintf("\nPython TMLE result (from %s):\n", shift_files[1]))
    cat(sprintf("  psi    = %+.4e\n", py_R$psi[1]))
    cat(sprintf("  CI low = %+.4e\n", py_R$ci_low[1]))
    cat(sprintf("  CI hi  = %+.4e\n", py_R$ci_high[1]))
    cat(sprintf("  p      = %.4f\n", py_R$pval[1]))
  }
}

# ──────────────────── Save ────────────────────────────
out_path <- sprintf("tmle_r_crossvalidation_%dkm_%s.csv",
                     RADIUS_KM,
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
result_df <- data.frame(
  radius_km     = RADIUS_KM,
  window_days   = WINDOW_DAYS,
  shift_pct     = SHIFT_PCT,
  psi           = fit$psi,
  se            = fit$se,
  ci_low        = fit$lwr_ci,
  ci_high       = fit$upr_ci,
  elapsed_sec   = elapsed,
  implementation = "R_txshift_tlverse"
)
fwrite(result_df, out_path)
cat(sprintf("\n✅  Saved %s\n", out_path))
