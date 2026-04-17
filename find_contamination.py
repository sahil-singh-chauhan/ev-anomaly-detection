"""
find_contamination.py
=====================
Elbow Method (Inflection Point Analysis) for determining the optimal
contamination parameter for the Isolation Forest anomaly detector.

Rather than hardcoding contamination=0.01, this script:

1. Runs the full feature-engineering pipeline to reproduce X_scaled exactly
   as model_training.py does.
2. Trains a baseline IsolationForest with contamination='auto' so the
   decision_function is unbiased by any preset threshold.
3. Computes the anomaly score for every row via decision_function().
   (sklearn convention: lower/negative = more anomalous, higher/positive = normal)
4. Sorts scores and plots the full distribution curve.
5. Applies two complementary elbow-detection methods:
   - Second-derivative (numpy gradient): finds where the curve's rate of
     change accelerates most sharply — the mathematical inflection point.
   - Kneedle geometry: projects the sorted curve onto a unit line and finds
     the point of maximum perpendicular distance — robust to scale.
6. Prints the recommended contamination rate and saves the annotated plot
   to data/anomaly_scores_elbow.png.

Usage
-----
Run from the project root::

    python find_contamination.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_processing import run_pipeline
from src.model_training import TRAINING_FEATURES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join(_PROJECT_ROOT, "data", "charging_logs.csv")
PLOT_PATH  = os.path.join(_PROJECT_ROOT, "data", "anomaly_scores_elbow.png")

# Smoothing window for the second-derivative to reduce noise sensitivity
_SMOOTH_WINDOW = 500


# ---------------------------------------------------------------------------
# Helper: moving average smoother
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving-average of *arr* with the given *window* size.

    Edges are handled by shrinking the window (same as 'valid' convolution
    padded back to full length), so the output always has the same shape as
    the input.

    Parameters
    ----------
    arr : np.ndarray
        1-D array to smooth.
    window : int
        Number of points in the moving average.

    Returns
    -------
    np.ndarray
        Smoothed array of the same length as *arr*.
    """
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ---------------------------------------------------------------------------
# Helper: Kneedle algorithm (geometry-based elbow detection)
# ---------------------------------------------------------------------------

def _kneedle_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Find the elbow index using the Kneedle geometric method.

    The algorithm normalises (x, y) to the unit square, constructs the
    straight line from the first to the last point, and returns the index
    of the point with the maximum perpendicular distance from that line.
    This is the 'knee' — the point where the curve bends most sharply.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (e.g. row indices).
    y : np.ndarray
        y-coordinates (e.g. sorted anomaly scores).

    Returns
    -------
    int
        Index into *x* / *y* of the detected elbow.
    """
    # Normalise both axes to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Direction vector of the line from first to last normalised point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    line_len = np.hypot(dx, dy)

    # Perpendicular distance from each point to the line
    distances = np.abs(dy * x_norm - dx * y_norm + x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0])
    distances /= line_len

    return int(np.argmax(distances))


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def find_optimal_contamination() -> None:
    """Run the full elbow-method analysis and save the annotated plot.

    Steps
    -----
    1. Load and process data via the existing pipeline.
    2. Fit imputer + scaler (same as model_training.py).
    3. Train IsolationForest with contamination='auto' to avoid biasing
       decision_function() toward any preset threshold.
    4. Compute decision_function scores for every row.
    5. Sort scores ascending (most anomalous first).
    6. Detect elbow via two methods:
       - Second derivative (numpy gradient on smoothed curve)
       - Kneedle geometry (max perpendicular distance)
    7. Plot the sorted score curve with both elbows annotated.
    8. Print recommended contamination rates.
    """
    t0 = time.perf_counter()

    # ── Step 1: Feature engineering ─────────────────────────────────────────
    log.info("Loading and processing data from: %s", DATA_PATH)
    df = run_pipeline(DATA_PATH)
    log.info("Pipeline complete — shape: %s", df.shape)

    X_raw = df[TRAINING_FEATURES].copy()

    # ── Step 2: Impute + Scale (mirroring model_training.py exactly) ─────────
    log.info("Fitting imputer ...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    log.info("Fitting scaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    log.info("Preprocessing complete — X_scaled shape: %s", X_scaled.shape)

    # ── Step 3: Train baseline IsolationForest ───────────────────────────────
    # contamination='auto' sets the internal threshold to 0.1 internally but
    # does NOT distort decision_function() scores — they remain the raw average
    # path length scores, unaffected by any contamination setting.
    log.info("Training baseline IsolationForest (contamination='auto') ...")
    model = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    log.info("Training complete.")

    # ── Step 4: Compute anomaly scores ───────────────────────────────────────
    log.info("Computing decision_function scores ...")
    scores = model.decision_function(X_scaled)
    # sklearn: lower (more negative) = more anomalous, higher = more normal

    # Sort ascending: index 0 = most anomalous row, index N-1 = most normal
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    x_idx = np.arange(n)
    log.info(
        "Score range: min=%.4f  max=%.4f  mean=%.4f",
        sorted_scores.min(), sorted_scores.max(), sorted_scores.mean(),
    )

    # ── Step 5a: Elbow via second derivative ─────────────────────────────────
    log.info("Computing second derivative elbow (window=%d) ...", _SMOOTH_WINDOW)
    smoothed = _smooth(sorted_scores, _SMOOTH_WINDOW)
    first_deriv  = np.gradient(smoothed, x_idx)
    second_deriv = np.gradient(first_deriv, x_idx)

    # We only look at the LEFT TAIL (first 10% of rows) where anomalies live.
    # The right 90% is flat normal data — its second derivative is noisy but
    # irrelevant.
    tail_limit   = int(0.10 * n)
    second_deriv_tail = second_deriv[:tail_limit]

    # The elbow is where the second derivative is maximally positive in the
    # tail — i.e., where the curve transitions from steep to flat most abruptly.
    elbow_2deriv_idx   = int(np.argmax(second_deriv_tail))
    elbow_2deriv_score = sorted_scores[elbow_2deriv_idx]
    elbow_2deriv_pct   = elbow_2deriv_idx / n * 100

    log.info(
        "Second-derivative elbow: index=%d, score=%.4f, contamination=%.4f%%",
        elbow_2deriv_idx, elbow_2deriv_score, elbow_2deriv_pct,
    )

    # ── Step 5b: Elbow via Kneedle geometry ──────────────────────────────────
    log.info("Computing Kneedle geometric elbow ...")
    # Run Kneedle only on the left tail to keep it focused on the anomaly region
    tail_scores = sorted_scores[:tail_limit]
    tail_x      = x_idx[:tail_limit].astype(float)
    kneedle_idx   = _kneedle_elbow(tail_x, tail_scores)
    kneedle_score = sorted_scores[kneedle_idx]
    kneedle_pct   = kneedle_idx / n * 100

    log.info(
        "Kneedle geometric elbow: index=%d, score=%.4f, contamination=%.4f%%",
        kneedle_idx, kneedle_score, kneedle_pct,
    )

    # ── Step 6: Plot ─────────────────────────────────────────────────────────
    log.info("Building plot ...")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"hspace": 0.45})

    # ── Top panel: full sorted score curve ───────────────────────────────────
    ax1 = axes[0]
    ax1.plot(x_idx, sorted_scores, lw=0.6, color="steelblue", alpha=0.8, label="Anomaly score")
    ax1.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.6, label="Score = 0 (sklearn boundary)")

    # Mark both elbows
    ax1.axvline(elbow_2deriv_idx, color="red",    ls="--", lw=1.4,
                label=f"2nd-derivative elbow  (idx={elbow_2deriv_idx:,}, score={elbow_2deriv_score:.4f}, {elbow_2deriv_pct:.2f}%)")
    ax1.axvline(kneedle_idx,      color="darkorange", ls="--", lw=1.4,
                label=f"Kneedle elbow         (idx={kneedle_idx:,}, score={kneedle_score:.4f}, {kneedle_pct:.2f}%)")

    ax1.set_title("Sorted Anomaly Scores — Full Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Row index (sorted: most anomalous → most normal)", fontsize=11)
    ax1.set_ylabel("Anomaly score (decision_function)", fontsize=11)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax1.legend(fontsize=9, loc="lower right")
    plt.setp(ax1.spines.values(), linewidth=0.6)

    # ── Bottom panel: zoomed left tail (anomaly region) ──────────────────────
    ax2 = axes[1]
    zoom_end = tail_limit
    ax2.plot(x_idx[:zoom_end], sorted_scores[:zoom_end],
             lw=1.2, color="steelblue", label="Anomaly score (tail zoom)")
    ax2.plot(x_idx[:zoom_end], smoothed[:zoom_end],
             lw=1.5, color="navy", alpha=0.7, label=f"Smoothed (window={_SMOOTH_WINDOW})")

    ax2.axvline(elbow_2deriv_idx, color="red",    ls="--", lw=1.4,
                label=f"2nd-deriv elbow  idx={elbow_2deriv_idx:,} ({elbow_2deriv_pct:.2f}%)")
    ax2.axvline(kneedle_idx,      color="darkorange", ls="--", lw=1.4,
                label=f"Kneedle elbow    idx={kneedle_idx:,} ({kneedle_pct:.2f}%)")

    ax2.set_title(
        f"Zoomed Left Tail (first 10% = {tail_limit:,} rows) — Anomaly Region",
        fontsize=13, fontweight="bold",
    )
    ax2.set_xlabel("Row index (sorted: most anomalous → most normal)", fontsize=11)
    ax2.set_ylabel("Anomaly score (decision_function)", fontsize=11)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.legend(fontsize=9, loc="lower right")
    plt.setp(ax2.spines.values(), linewidth=0.6)

    fig.suptitle(
        "Isolation Forest — Elbow Method for Optimal Contamination",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    log.info("Plot saved -> %s", PLOT_PATH)
    plt.close(fig)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    # Count rows below each threshold
    n_below_2deriv  = int((scores < elbow_2deriv_score).sum())
    n_below_kneedle = int((scores < kneedle_score).sum())

    print()
    print("=" * 62)
    print("   Elbow Method — Optimal Contamination Analysis")
    print("=" * 62)
    print(f"  Total rows analysed         : {n:>10,}")
    print()
    print("  Method 1 — Second Derivative (mathematical inflection)")
    print(f"    Elbow score threshold      : {elbow_2deriv_score:>10.4f}")
    print(f"    Rows below threshold       : {n_below_2deriv:>10,}")
    print(f"    Recommended contamination  : {elbow_2deriv_pct:>9.4f}%")
    print()
    print("  Method 2 — Kneedle Geometry (max perpendicular distance)")
    print(f"    Elbow score threshold      : {kneedle_score:>10.4f}")
    print(f"    Rows below threshold       : {n_below_kneedle:>10,}")
    print(f"    Recommended contamination  : {kneedle_pct:>9.4f}%")
    print()
    print("  Current hardcoded value      :     1.0000%")
    print()
    print(f"  Plot saved to              : {PLOT_PATH}")
    print(f"  Elapsed                    : {elapsed:.2f}s")
    print("=" * 62)
    print()
    print("  NEXT STEP: Inspect data/anomaly_scores_elbow.png.")
    print("  Update contamination= in src/model_training.py to")
    print("  whichever elbow percentage best matches the visual bend.")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    find_optimal_contamination()
