"""
model_training.py
=================
Trains an unsupervised anomaly-detection pipeline on EV charging session logs
and persists all fitted artefacts to disk for use by the inference pipeline.

Pipeline stages
---------------
1. Load & process raw logs via :func:`src.data_processing.run_pipeline`.
2. Subset to ``TRAINING_FEATURES`` — labels and identifiers are deliberately
   excluded to keep the approach fully unsupervised.
3. Impute any residual NaNs with a median strategy.
4. Standardise features to zero mean / unit variance.
5. Train an Isolation Forest on the scaled data.
6. Persist ``imputer.joblib``, ``scaler.joblib``, and
   ``isolation_forest.joblib`` to ``models/``.

Usage
-----
Run from the project root::

    python -m src.model_training

or::

    python src/model_training.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when the module is run directly
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_processing import run_pipeline  # noqa: E402  (after path fix)

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
# Constants
# ---------------------------------------------------------------------------

#: Features fed to the model.  All labels, IDs, and timestamps are excluded
#: so the model has zero access to ground-truth fault information.
#:
#: Electrical features use a two-signal design for voltage and temperature:
#:
#:   ``voltage_rolling_mean``  — the smooth session-level trend (the level)
#:   ``voltage_deviation``     — instantaneous reading minus that trend (the spike)
#:
#:   ``temperature_c_rolling_mean`` — smooth thermal trend within the session
#:   ``temp_deviation``             — instantaneous thermal spike above/below trend
#:
#: This decomposition gives the Isolation Forest two orthogonal dimensions
#: per sensor: one capturing gradual drift and one capturing sudden spikes.
#: Both are meaningful anomaly signals but carry different information, so
#: they deserve independent feature columns rather than a single raw reading.
TRAINING_FEATURES: list[str] = [
    "power_mismatch",
    "voltage_rolling_mean",
    "voltage_deviation",
    "current",
    "temperature_c_rolling_mean",
    "temp_deviation",
    "hour_sin",
    "hour_cos",
    "station_avg_temp",
    "session_energy_total",
]

#: Columns that must never appear in X_train.
_FORBIDDEN_FEATURES: set[str] = {
    "is_error",
    "error_code",
    "session_id",
    "station_id",
    "timestamp",
}

DATA_PATH: str = os.path.join(_PROJECT_ROOT, "data", "charging_logs.csv")
MODELS_DIR: str = os.path.join(_PROJECT_ROOT, "models")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_features(features: list[str]) -> None:
    """Raise ``ValueError`` if any forbidden column sneaked into the feature list.

    Parameters
    ----------
    features : list[str]
        The feature list to validate.

    Raises
    ------
    ValueError
        If any element of *features* is in ``_FORBIDDEN_FEATURES``.
    """
    leaks = _FORBIDDEN_FEATURES.intersection(features)
    if leaks:
        raise ValueError(
            f"Label-leakage detected — the following forbidden columns are "
            f"present in TRAINING_FEATURES: {sorted(leaks)}.  "
            f"Remove them to keep the model strictly unsupervised."
        )


def _ensure_models_dir(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
    log.info("Model output directory: %s", path)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(
    data_path: str = DATA_PATH,
    models_dir: str = MODELS_DIR,
    contamination: float = 0.01,
    random_state: int = 42,
    rolling_window: int = 5,
) -> dict[str, object]:
    """Run the full training pipeline and save all artefacts.

    Steps
    -----
    1. **Load & process** – calls :func:`run_pipeline` to produce a clean,
       feature-rich DataFrame from the raw CSV.
    2. **Feature selection** – subsets to ``TRAINING_FEATURES`` after
       asserting that no forbidden columns are present.
    3. **Imputation** – fits a :class:`~sklearn.impute.SimpleImputer` with
       ``strategy='median'`` to handle any NaNs that survive the pipeline
       (e.g. the first few rows of very short sessions where the rolling
       window never reaches size *rolling_window* even after bfill/ffill).
    4. **Scaling** – fits a :class:`~sklearn.preprocessing.StandardScaler`
       so all features contribute equally to the anomaly score regardless
       of their physical units.
    5. **Modelling** – trains an
       :class:`~sklearn.ensemble.IsolationForest` with *contamination*
       controlling the expected fraction of outliers.  The forest isolates
       anomalies by repeatedly partitioning the feature space at random;
       points that are isolated quickly (short average path length) are
       scored as outliers.
    6. **Persistence** – saves ``imputer.joblib``, ``scaler.joblib``, and
       ``isolation_forest.joblib`` to *models_dir*.

    Parameters
    ----------
    data_path : str, optional
        Path to the raw ``charging_logs.csv``.  Defaults to
        ``<project_root>/data/charging_logs.csv``.
    models_dir : str, optional
        Directory where fitted artefacts will be saved.  Defaults to
        ``<project_root>/models/``.
    contamination : float, optional
        Expected proportion of anomalies in the dataset (passed directly
        to :class:`~sklearn.ensemble.IsolationForest`).  Defaults to
        ``0.01`` (1 %).
    random_state : int, optional
        Seed for reproducibility.  Defaults to ``42``.
    rolling_window : int, optional
        Window size forwarded to :func:`run_pipeline`.  Defaults to ``5``.

    Returns
    -------
    dict
        A mapping with keys ``"imputer"``, ``"scaler"``, and ``"model"``
        pointing to the fitted scikit-learn objects.

    Raises
    ------
    FileNotFoundError
        If *data_path* does not exist.
    ValueError
        If ``TRAINING_FEATURES`` contains any forbidden column.
    """
    t0 = time.perf_counter()

    # ── Guard: no label leakage ─────────────────────────────────────────────
    log.info("Validating feature list for label leakage …")
    _validate_features(TRAINING_FEATURES)
    log.info("Feature list OK — %d features, all unsupervised.", len(TRAINING_FEATURES))

    # ── Stage 1: load & process ─────────────────────────────────────────────
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    log.info("Loading and processing data from: %s", data_path)
    df = run_pipeline(data_path, rolling_window=rolling_window)
    log.info("Pipeline complete — DataFrame shape: %s", df.shape)

    # ── Extract and freeze station baselines ────────────────────────────────
    # Compute once from the full training set and persist so inference never
    # recomputes baselines from a potentially small or skewed batch.
    station_baselines: dict[str, float] = (
        df.groupby("station_id")["temperature_c"].mean().to_dict()
    )
    log.info(
        "Station baselines extracted — %d stations, global mean=%.2f °C.",
        len(station_baselines),
        sum(station_baselines.values()) / len(station_baselines),
    )

    # ── Stage 2: feature selection ──────────────────────────────────────────
    missing_cols = [c for c in TRAINING_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following TRAINING_FEATURES are missing from the processed "
            f"DataFrame: {missing_cols}"
        )

    X = df[TRAINING_FEATURES].copy()
    log.info("Training feature matrix shape: %s", X.shape)
    log.info(
        "NaN counts before imputation:\n%s",
        X.isna().sum().to_string(),
    )

    # ── Stage 3: imputation ─────────────────────────────────────────────────
    log.info("Fitting SimpleImputer (strategy='median') …")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    log.info("Imputation complete — remaining NaNs: %d", pd.isna(X_imputed).sum())

    # ── Stage 4: scaling ────────────────────────────────────────────────────
    log.info("Fitting StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    log.info(
        "Scaling complete — mean≈0 check: max abs mean = %.6f",
        abs(X_scaled.mean(axis=0)).max(),
    )

    # ── Stage 5: model training ─────────────────────────────────────────────
    log.info(
        "Training IsolationForest (contamination=%.3f, random_state=%d) …",
        contamination,
        random_state,
    )
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    log.info("IsolationForest training complete.")

    # Quick sanity check on training-set predictions
    raw_preds = model.predict(X_scaled)
    n_anomalies = int((raw_preds == -1).sum())
    log.info(
        "Training-set self-prediction: %d anomalies (%.2f%% of %d rows).",
        n_anomalies,
        n_anomalies / len(raw_preds) * 100,
        len(raw_preds),
    )

    # ── Stage 6: persist artefacts ──────────────────────────────────────────
    _ensure_models_dir(models_dir)

    artefacts = {
        "imputer":           (imputer,           "imputer.joblib"),
        "scaler":            (scaler,            "scaler.joblib"),
        "model":             (model,             "isolation_forest.joblib"),
        "station_baselines": (station_baselines, "station_baselines.joblib"),
    }

    for key, (obj, filename) in artefacts.items():
        path = os.path.join(models_dir, filename)
        joblib.dump(obj, path)
        log.info("Saved %-20s -> %s", key, path)

    elapsed = time.perf_counter() - t0
    log.info("Training pipeline finished in %.2f seconds.", elapsed)

    return {
        "imputer":           imputer,
        "scaler":            scaler,
        "model":             model,
        "station_baselines": station_baselines,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
