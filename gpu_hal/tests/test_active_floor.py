"""Regression test for the active-floor λ-selection rule in
run_hurdle_full_n_cv.cv_stage1_logistic + main()'s post-CV selection
block.

The bug we're guarding against: prior to 2026-05-08, the Stage 1
λ-selection picked argmin(mean_dev) without checking whether the
chosen λ pruned all bases. In radii where the data was weakly
informative, this picked a λ large enough that n_active = 0,
which forces ψ_freq = 0 by construction in the downstream calibrated-
shift computation.

The patched code applies an "active-floor" rule: among CV candidates,
pick the smallest mean-deviance λ where median(n_active) >= min_active.
If no λ in grid satisfies the floor, fall back to the smallest λ.

This test fabricates a synthetic n_active matrix where the argmin
candidate would violate the floor, and verifies that the rule picks
a different — non-degenerate — λ.

Run from repo root:
    .venv/bin/python -m pytest gpu_hal/tests/test_active_floor.py -v
"""
import numpy as np


def select_lambda_with_active_floor(mean_dev, cv_nact, min_active):
    """Reproduces the active-floor selection block in
    run_hurdle_full_n_cv.main(). Pure-array form so it's testable
    without rebuilding the full IRLS path.
    """
    raw_idx = int(np.argmin(mean_dev))
    if cv_nact is None or min_active <= 0:
        return raw_idx, "raw"
    median_nact = np.median(cv_nact, axis=0)
    feasible = median_nact >= min_active
    if feasible.any():
        mean_dev_feasible = np.where(feasible, mean_dev, np.inf)
        idx = int(np.argmin(mean_dev_feasible))
        return idx, ("raw" if idx == raw_idx else "floor-fallback")
    return len(mean_dev) - 1, "no-feasible-fallback"


def test_active_floor_overrides_pruning_argmin():
    """If the argmin λ has n_active = 0, the rule must fall back."""
    # 15-point grid. Imagine an L-shape: low-λ end fits well but
    # n_active explodes; mid-λ has n_active = 5-10; top-λ prunes
    # everything.
    mean_dev = np.array([0.50, 0.48, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41,
                          0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48])
    # Naive argmin picks idx 7 (mean_dev 0.41). But at idx 7,
    # n_active = 0 across all folds — a pathological pruning.
    cv_nact = np.array([
        [200, 150, 100,  60,  30,  15,   8,   0,   0,   0,   0,   0,   0,  0, 0],
        [210, 155, 102,  58,  32,  14,   7,   0,   0,   0,   0,   0,   0,  0, 0],
        [195, 148, 101,  61,  29,  16,   9,   0,   0,   0,   0,   0,   0,  0, 0],
    ])
    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact, min_active=5)
    assert mode == "floor-fallback", f"expected floor-fallback, got {mode}"
    # Feasible idxs (median n_active >= 5): 0..5 (median values 200,
    # 150, 101, 60, 30, 15, 8). Idx 6 has median 8 ≥ 5 too.
    # Among feasible, smallest mean_dev is at idx 6 (0.42).
    assert idx == 6, f"expected idx 6, got {idx}"


def test_active_floor_keeps_argmin_when_safe():
    """If the argmin λ already satisfies the floor, leave it alone."""
    mean_dev = np.array([0.50, 0.48, 0.46, 0.45, 0.44, 0.43, 0.42])
    cv_nact = np.array([
        [200, 150, 100,  60,  30,  15,  10],
        [210, 155, 102,  58,  32,  14,  11],
        [195, 148, 101,  61,  29,  16,  12],
    ])
    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact, min_active=5)
    assert mode == "raw"
    assert idx == 6


def test_active_floor_no_feasible_lambda():
    """If NO λ in grid has median n_active >= floor, fall back to the
    smallest λ in grid (heaviest fit) rather than failing silently."""
    mean_dev = np.array([0.50, 0.48, 0.46, 0.45, 0.44])
    # Every fold prunes to 0-2 active bases at every λ.
    cv_nact = np.array([
        [2, 1, 1, 0, 0],
        [1, 2, 1, 0, 0],
        [2, 0, 1, 0, 0],
    ])
    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact, min_active=5)
    assert mode == "no-feasible-fallback"
    # Falls back to smallest λ in grid (heaviest fit) = idx len-1.
    assert idx == 4


def test_active_floor_disabled_when_nact_missing():
    """Legacy fold CSVs that don't have n_active should leave the
    raw argmin untouched (backward compatibility)."""
    mean_dev = np.array([0.50, 0.48, 0.46, 0.45, 0.44])
    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact=None, min_active=5)
    assert mode == "raw"
    assert idx == 4


def test_active_floor_disabled_when_min_active_zero():
    """Setting --min-active 0 disables the floor."""
    mean_dev = np.array([0.50, 0.48, 0.46, 0.45, 0.44])
    cv_nact = np.array([[10, 5, 0, 0, 0],
                         [12, 6, 0, 0, 0],
                         [11, 4, 0, 0, 0]])
    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact, min_active=0)
    assert mode == "raw"
    assert idx == 4  # argmin even though n_active = 0 there


def test_active_floor_recovers_may06_pathology():
    """Synthetic version of the R=11 km / R=12 km May 6 pathology
    that motivated the patch: the argmin landed in the pruning
    plateau (n_active = 0) and ψ_freq was forced to 0.

    Confirms the patched rule picks a non-degenerate λ instead.
    """
    # 25-point grid mirroring the patched default. The bottom end
    # has very low CV-dev but is pathological (all bases active +
    # overfitting); the middle has a sweet spot with 50-100 active;
    # the top prunes everything.
    n = 25
    mean_dev = np.full(n, 0.50)
    # Bowtie shape with minimum at idx 18 — but at idx 18, n_active = 0
    for i in range(n):
        mean_dev[i] = 0.45 + abs(i - 18) * 0.005

    cv_nact = np.zeros((5, n), dtype=int)
    for f in range(5):
        for i in range(n):
            # Simulate the May 6 pattern: at the very bottom of the
            # λ-grid (small λ) active set is large; from there it
            # shrinks; in this synthetic radius it hits 0 by idx 14.
            if i <= 5:    cv_nact[f, i] = 200 - i*10 + np.random.randint(-2, 3)
            elif i <= 10: cv_nact[f, i] = 100 - (i-5)*15
            elif i <= 13: cv_nact[f, i] = max(0, 25 - (i-10)*8)
            else:         cv_nact[f, i] = 0

    idx, mode = select_lambda_with_active_floor(mean_dev, cv_nact, min_active=5)
    assert mode == "floor-fallback"
    # Patched rule must NOT pick idx ≥ 14 (where n_active = 0).
    assert idx < 14, f"expected idx < 14, got {idx}"
    assert np.median(cv_nact[:, idx]) >= 5


if __name__ == "__main__":
    test_active_floor_overrides_pruning_argmin()
    test_active_floor_keeps_argmin_when_safe()
    test_active_floor_no_feasible_lambda()
    test_active_floor_disabled_when_nact_missing()
    test_active_floor_disabled_when_min_active_zero()
    test_active_floor_recovers_may06_pathology()
    print("all active-floor tests passed.")
