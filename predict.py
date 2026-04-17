"""
predict.py
==========
Hybrid Anomaly Detection inference pipeline for EV Charging Station logs.

Architecture: Two-Gate System
------------------------------
Every row passes through **two independent gates** before a final verdict is
issued.  A row is flagged as an anomaly if **either** gate raises a flag:

* **Gate 1 — Rules Engine** (:func:`apply_rules_engine`)
  Hard-coded physics and hardware constraints that are always true regardless
  of the data distribution.  Catches absolute violations the ML model might
  miss because they are too rare to learn from.

* **Gate 2 — ML Engine** (Isolation Forest)
  Statistical anomaly detection trained on the full feature space.  Catches
  subtle, multivariate deviations that no single rule can express.

Final verdict::

    is_anomaly = rule_anomaly  OR  ml_anomaly

Usage
-----
::

    python predict.py --input data/charging_logs.csv --output predictions.csv

Arguments
---------
--input   Path to the input CSV file (raw charging logs format).
--output  Path where the output CSV with predictions will be written.

Exit codes
----------
0 — success
1 — file-not-found or missing model artefact
2 — unexpected runtime error
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable when script is called directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_processing import run_pipeline        # noqa: E402
from src.model_training import TRAINING_FEATURES    # noqa: E402

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
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

_ARTEFACT_FILES = {
    "imputer":           "imputer.joblib",
    "scaler":            "scaler.joblib",
    "model":             "isolation_forest.joblib",
    "station_baselines": "station_baselines.joblib",
}

# ---------------------------------------------------------------------------
# Rule thresholds (single source of truth — easy to tune without touching logic)
# ---------------------------------------------------------------------------
_VOLTAGE_MIN:     float = 150.0   # V   — below this is an under-voltage fault
_VOLTAGE_MAX:     float = 280.0   # V   — above this is an over-voltage fault
_TEMP_MAX:        float = 70.0    # °C  — thermal runaway threshold


# ---------------------------------------------------------------------------
# Gate 1 — Rules Engine
# ---------------------------------------------------------------------------

def apply_rules_engine(df: pd.DataFrame) -> pd.Series:
    """Gate 1: Heuristics engine for absolute physical violations.

    This engine encodes hard physical and operational constraints that are
    *always* true, regardless of the statistical distribution of the data.
    It is intentionally designed to complement the ML model (Gate 2):

    * The ML model excels at detecting *subtle, multivariate* deviations that
      are hard to express as rules.
    * The Rules Engine excels at catching *absolute violations* (e.g., negative
      current, which is physically impossible) that are so rare in the training
      data that a statistical model might not learn to flag them reliably.

    Note on ``error_code``
    ----------------------
    ``error_code`` (and the derived ``is_error`` column) are the ground-truth
    fault labels produced by the charging station's own firmware.  They are
    intentionally **excluded** from the rules engine so that the evaluation in
    ``evaluate_model.py`` Section C is non-circular: the rules engine and the
    ML engine both act as independent detectors, and their combined recall of
    hardware-reported faults is a genuine test of coverage.

    Rules applied (any single rule triggers a flag)
    ------------------------------------------------
    1. **Voltage out of bounds** — ``voltage < 150 V`` or ``voltage > 280 V``
       Outside this band, the charger is operating outside its safe electrical
       rating.  Sustained under-voltage degrades battery health; over-voltage
       can cause insulation breakdown or fire.

    2. **Thermal runaway threshold** — ``temperature_c > 70 °C``
       Above 70 °C, lithium-ion cell chemistry becomes unstable.  This is a
       hard safety limit regardless of what the station's historical baseline is.

    3. **Impossible physics: negative current** — ``current < 0``
       Current in a DC charging circuit cannot be negative.  A negative reading
       always indicates a sensor failure or a wiring fault.

    4. **Impossible physics: negative energy** — ``energy_kwh < 0``
       Energy delivered in a time step is always non-negative.  A negative
       value means the meter is running backwards — a critical metering fault.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame as returned by :func:`src.data_processing.run_pipeline`.
        Must contain columns: ``voltage``, ``temperature_c``, ``current``,
        ``energy_kwh``.

    Returns
    -------
    pd.Series
        Integer Series (dtype int64) aligned with *df*'s index where
        ``1`` = anomaly flagged by at least one rule, ``0`` = no rule triggered.
    """
    voltage_fault    = (df["voltage"]       < _VOLTAGE_MIN) | (df["voltage"] > _VOLTAGE_MAX)
    temp_fault       = (df["temperature_c"] > _TEMP_MAX)
    negative_current = df["current"]        < 0
    negative_energy  = df["energy_kwh"]     < 0

    rule_flag = (
        voltage_fault
        | temp_fault
        | negative_current
        | negative_energy
    ).astype(int)

    n_flagged = int(rule_flag.sum())
    log.info(
        "Gate 1 (Rules Engine) complete — %d rows flagged  "
        "(voltage: %d | temp: %d | neg_current: %d | neg_energy: %d)",
        n_flagged,
        int(voltage_fault.sum()),
        int(temp_fault.sum()),
        int(negative_current.sum()),
        int(negative_energy.sum()),
    )
    return rule_flag


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with ``--input`` and ``--output`` arguments.
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Hybrid Two-Gate EV Charging Anomaly Detector.\n"
            "Gate 1: Rules Engine (physics violations + hardware faults).\n"
            "Gate 2: Isolation Forest (statistical multivariate anomalies)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py --input data/charging_logs.csv "
            "--output predictions.csv\n"
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to the input CSV file containing raw charging logs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path where the output CSV with the is_anomaly column will be saved.",
    )
    return parser


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_artefacts(models_dir: str) -> tuple:
    """Load the fitted imputer, scaler, and IsolationForest from disk.

    Parameters
    ----------
    models_dir : str
        Directory containing the three ``.joblib`` files produced by
        :mod:`src.model_training`.

    Returns
    -------
    tuple
        ``(imputer, scaler, model, station_baselines)`` — three fitted
        scikit-learn objects plus the frozen station baseline dict.

    Raises
    ------
    FileNotFoundError
        If *models_dir* does not exist or any expected ``.joblib`` file is
        missing.  Suggests running ``python -m src.model_training`` first.
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(
            f"Models directory not found: '{models_dir}'.  "
            f"Run 'python -m src.model_training' first to train the model."
        )

    loaded = {}
    for key, filename in _ARTEFACT_FILES.items():
        path = os.path.join(models_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model artefact missing: '{path}'.  "
                f"Run 'python -m src.model_training' to regenerate it."
            )
        loaded[key] = joblib.load(path)
        log.info("Loaded %-10s <- %s", key, path)

    return loaded["imputer"], loaded["scaler"], loaded["model"], loaded["station_baselines"]


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def predict(input_path: str, output_path: str) -> None:
    """Run the full Two-Gate hybrid inference pipeline and save results.

    Workflow
    --------
    1. Validate that *input_path* exists.
    2. Load the frozen station baselines from ``models/station_baselines.joblib``.
    3. Run :func:`run_pipeline` passing those baselines so ``station_avg_temp``
       reflects training-set values, not the inference-batch distribution.
    4. **Gate 1** — :func:`apply_rules_engine` produces ``rule_anomaly``.
    5. **Gate 2** — Apply imputer + scaler, run Isolation Forest, map
       ``-1 → 1`` (anomaly) and ``+1 → 0`` (normal) into ``ml_anomaly``.
    6. Combine gates: ``is_anomaly = rule_anomaly | ml_anomaly`` (bitwise OR
       on boolean-cast Series, then cast back to int).
    7. Append ``rule_anomaly``, ``ml_anomaly``, and ``is_anomaly`` to *df*.
    8. Write output CSV (no index) to *output_path*.
    9. Print a detailed summary to the console.

    Parameters
    ----------
    input_path : str
        Path to the raw input CSV (must follow the ``charging_logs.csv``
        schema produced by the data-collection system).
    output_path : str
        Destination path for the output CSV with all prediction columns
        appended.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist or any model artefact is missing.
    KeyError
        If the processed DataFrame is missing an expected feature column.
    """
    t0 = time.perf_counter()

    # ── Validate input ──────────────────────────────────────────────────────
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: '{input_path}'")
    log.info("Input  file : %s", input_path)
    log.info("Output file : %s", output_path)

    # ── Load artefacts first (needed before pipeline for station baselines) ─
    log.info("Loading model artefacts from: %s", MODELS_DIR)
    imputer, scaler, model, station_baselines = _load_artefacts(MODELS_DIR)
    log.info(
        "Station baselines loaded — %d stations known from training.",
        len(station_baselines),
    )

    # ── Feature engineering ─────────────────────────────────────────────────
    # Pass frozen station_baselines so station_avg_temp uses training-set
    # means rather than being recomputed from this (potentially small) batch.
    log.info("Running feature-engineering pipeline ...")
    df = run_pipeline(input_path, station_baselines=station_baselines)
    log.info("Pipeline complete — %d rows, %d columns.", *df.shape)

    # ── Gate 1: Rules Engine ────────────────────────────────────────────────
    log.info("Running Gate 1: Rules Engine ...")
    df["rule_anomaly"] = apply_rules_engine(df)

    # ── Gate 2: ML Engine ───────────────────────────────────────────────────
    log.info("Running Gate 2: ML Engine (Isolation Forest) ...")

    missing = [c for c in TRAINING_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following expected feature columns are absent from the "
            f"processed DataFrame: {missing}"
        )

    # Pass as DataFrame (not .values) so sklearn recognises the feature names
    # and suppresses the "fitted with feature names" UserWarning.
    X = df[TRAINING_FEATURES]
    log.info("Applying imputer ...")
    X = imputer.transform(X)

    log.info("Applying scaler ...")
    X = scaler.transform(X)

    log.info("Generating ML anomaly predictions ...")
    raw_preds = model.predict(X)

    # IsolationForest convention:  -1 = anomaly,  +1 = normal
    # Our convention:               1 = anomaly,   0 = normal
    df["ml_anomaly"] = (raw_preds == -1).astype(int)

    n_ml = int(df["ml_anomaly"].sum())
    log.info(
        "Gate 2 (ML Engine) complete — %d rows flagged (%.2f%% of %d).",
        n_ml,
        n_ml / len(df) * 100,
        len(df),
    )

    # ── Final verdict: OR of both gates ─────────────────────────────────────
    # Cast to bool before OR to guarantee correct logical union, then back to
    # int so the output column is 0/1 (not True/False).
    df["is_anomaly"] = (
        df["rule_anomaly"].astype(bool) | df["ml_anomaly"].astype(bool)
    ).astype(int)

    # ── Compute summary statistics ───────────────────────────────────────────
    n_total         = len(df)
    n_rule          = int(df["rule_anomaly"].sum())
    n_ml_flagged    = int(df["ml_anomaly"].sum())
    n_total_anomaly = int(df["is_anomaly"].sum())
    n_overlap       = int((df["rule_anomaly"].astype(bool) & df["ml_anomaly"].astype(bool)).sum())
    pct_total       = n_total_anomaly / n_total * 100

    # ── Save output ─────────────────────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Predictions saved -> %s", output_path)

    elapsed = time.perf_counter() - t0

    # ── Console summary ─────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("   EV Charging Anomaly Detector — Two-Gate Results")
    print("=" * 58)
    print(f"  Input file                  : {input_path}")
    print(f"  Output file                 : {output_path}")
    print(f"  Rows processed              : {n_total:>10,}")
    print(f"  -- Gate 1 (Rules Engine)    : {n_rule:>10,}  anomalies")
    print(f"  -- Gate 2 (ML Engine)       : {n_ml_flagged:>10,}  anomalies")
    print(f"  -- Overlap (both gates)     : {n_overlap:>10,}  rows")
    print(f"  Total unique anomalies      : {n_total_anomaly:>10,}  ({pct_total:.2f}%)")
    print(f"  Elapsed                     : {elapsed:.2f}s")
    print("=" * 58)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the hybrid inference pipeline.

    Handles :class:`FileNotFoundError` and unexpected exceptions, printing
    a clean error message and exiting with a non-zero status code so the
    error is visible in shell scripts and CI pipelines.
    """
    parser = _build_parser()
    args = parser.parse_args()

    try:
        predict(input_path=args.input, output_path=args.output)
    except FileNotFoundError as exc:
        log.error("File not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        log.error("Missing feature column: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error during prediction: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
