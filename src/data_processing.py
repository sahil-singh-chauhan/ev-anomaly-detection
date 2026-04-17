"""
data_processing.py
==================
Core data pipeline for the EV Charging Station Anomaly Detection project.

Pipeline stages
---------------
1.  load_data                    – Read raw CSV logs from disk.
2.  clean_data                   – Deduplicate simultaneous sensor dumps per session.
3.  engineer_physics_features    – Compute power_mismatch from electrical identity.
4.  engineer_rolling_features    – Compute rolling statistics grouped by session,
                                   with bfill/ffill to preserve early-session rows.
5.  engineer_time_features       – Extract hour, day_of_week, and cyclical hour
                                   encodings (hour_sin, hour_cos).
6.  engineer_station_baselines   – Compute (or apply) per-station mean temperature
                                   baseline (station_avg_temp).  Accepts a pre-computed
                                   baselines dict so inference never recomputes from
                                   the new batch (train/inference consistency).
7.  engineer_deviation_features  – Compute voltage_deviation and temp_deviation as
                                   the signed difference from each rolling mean.
8.  engineer_session_aggregates  – Compute cumulative energy per session
                                   (session_energy_total).
9.  engineer_categorical_flags   – Derive binary flag is_error from error_code.
10. run_pipeline                 – Convenience wrapper that executes all stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stage 1 – Load
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw EV charging session logs from a CSV file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with a parsed ``timestamp`` column (datetime64).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not point to an existing file.
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Stage 2 – Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate sensor readings logged at the same second.

    EV chargers occasionally emit multiple log lines with an identical
    (session_id, timestamp) pair within the same second (sensor dump
    artefact).  This function collapses those duplicates by:

    * Taking the **mean** of every numeric column so no energy information
      is lost.
    * Taking the **first** value of every non-numeric (categorical) column
      to preserve metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame sorted by ``session_id`` and ``timestamp``,
        with the original index reset.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in df.columns if c not in numeric_cols + ["session_id", "timestamp"]
    ]

    agg_dict: dict = {col: "mean" for col in numeric_cols}
    agg_dict.update({col: "first" for col in categorical_cols})

    cleaned = (
        df.groupby(["session_id", "timestamp"], sort=True)
        .agg(agg_dict)
        .reset_index()
    )
    return cleaned


# ---------------------------------------------------------------------------
# Stage 3 – Physics features
# ---------------------------------------------------------------------------

def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a physics-based anomaly indicator: ``power_mismatch``.

    According to the electrical identity P = V × I, the product
    ``voltage × current / 1000`` should equal ``power_kw``.  A large
    discrepancy is a strong signal of a metering fault or a dangerous
    operating condition.

    .. math::

        \\text{power\\_mismatch} = |\\text{power\\_kw} - (V \\times I) / 1000|

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame as returned by :func:`clean_data`.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``power_mismatch`` column
        (float, unit: kW).
    """
    df = df.copy()
    df["power_mismatch"] = (df["power_kw"] - (df["voltage"] * df["current"]) / 1000).abs()
    return df


# ---------------------------------------------------------------------------
# Stage 4 – Rolling features
# ---------------------------------------------------------------------------

def engineer_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling-mean features for voltage and temperature per session.

    Computing the rolling mean within each charging session smooths
    short-term noise and exposes gradual drift that could indicate
    degradation or an impending fault.

    New columns added:

    * ``voltage_rolling_mean``       – Rolling mean of ``voltage``.
    * ``temperature_c_rolling_mean`` – Rolling mean of ``temperature_c``.

    NaN handling
    ------------
    A standard rolling window of size *window* leaves the first
    ``window - 1`` rows of every session as ``NaN``.  Those early rows
    represent the critical "handshake" phase of a session and must not be
    discarded.  After computing the rolling values, the function therefore
    applies **back-fill** (``bfill``) followed by **forward-fill** (``ffill``)
    within each session group, so no row is left as ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after physics feature engineering.
    window : int, optional
        Number of observations used for the rolling calculation.
        Defaults to ``5``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional rolling-mean columns, fully
        filled within each session (no ``NaN`` values remain).
    """
    df = df.copy()
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    for col, new_col in [
        ("voltage", "voltage_rolling_mean"),
        ("temperature_c", "temperature_c_rolling_mean"),
    ]:
        df[new_col] = (
            df.groupby("session_id")[col]
            .transform(lambda s: s.rolling(window=window).mean())
        )

    rolling_cols = ["voltage_rolling_mean", "temperature_c_rolling_mean"]
    df[rolling_cols] = (
        df.groupby("session_id")[rolling_cols]
        .transform(lambda g: g.bfill().ffill())
    )
    return df


# ---------------------------------------------------------------------------
# Stage 5 – Time features
# ---------------------------------------------------------------------------

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the ``timestamp`` column.

    Raw timestamps carry implicit periodicity that linear models cannot
    exploit directly.  This function extracts:

    * ``hour``        – Hour of day (0–23, int).
    * ``day_of_week`` – Day of week (0 = Monday … 6 = Sunday, int).
    * ``hour_sin``    – Sine encoding of ``hour`` on a 24-hour cycle.
    * ``hour_cos``    – Cosine encoding of ``hour`` on a 24-hour cycle.

    Cyclical encoding explained
    ---------------------------
    If ``hour`` is treated as a plain integer, a model would perceive
    23:00 and 00:00 as maximally distant (difference = 23).  Mapping hours
    onto the unit circle via

    .. math::

        \\text{hour\\_sin} = \\sin\\!\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right), \\quad
        \\text{hour\\_cos} = \\cos\\!\\left(\\frac{2\\pi \\cdot \\text{hour}}{24}\\right)

    ensures that 23:00 and 00:00 are represented as geometrically close
    points on the circle, so any model that uses Euclidean distance
    (k-NN, SVM, neural networks) handles midnight wrap-around correctly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame at any stage after :func:`load_data` (``timestamp``
        column must exist and be of dtype datetime64).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four additional columns:
        ``hour``, ``day_of_week``, ``hour_sin``, ``hour_cos``.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


# ---------------------------------------------------------------------------
# Stage 6 – Station baselines
# ---------------------------------------------------------------------------

def engineer_station_baselines(
    df: pd.DataFrame,
    baselines: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add a per-station mean temperature baseline: ``station_avg_temp``.

    Different charging stations may be installed in environments with
    different ambient temperatures (e.g., indoor vs. outdoor, sunny vs.
    shaded).  A reading that is normal for one station may be anomalous for
    another.

    Train-vs-inference consistency
    --------------------------------
    If *baselines* is ``None`` (training mode), the mean is computed from
    the rows in *df* itself and returned alongside the DataFrame so the
    caller can persist it.

    If *baselines* is provided (inference mode), the pre-computed training
    means are mapped onto *df* via a simple dictionary lookup.  This prevents
    a data-leakage / distribution-shift bug where a small inference batch
    would produce very different baseline values than the scaler was trained on.

    For stations that appear in the inference batch but were **not** seen
    during training (unseen station IDs), the global mean of all training
    baselines is used as a fallback.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame at any stage after :func:`clean_data`.
    baselines : dict[str, float] or None, optional
        Mapping of ``station_id → mean_temperature_c`` computed on the
        training set.  Pass ``None`` during training; pass the saved dict
        during inference.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one additional column ``station_avg_temp``
        (float, unit: °C).
    """
    df = df.copy()
    if baselines is None:
        # Training mode — compute from the current DataFrame
        station_mean = df.groupby("station_id")["temperature_c"].transform("mean")
        df["station_avg_temp"] = station_mean
    else:
        # Inference mode — map from the frozen training baselines
        global_fallback = float(pd.Series(baselines).mean())
        df["station_avg_temp"] = df["station_id"].map(baselines).fillna(global_fallback)
    return df


# ---------------------------------------------------------------------------
# Stage 7 – Deviation features
# ---------------------------------------------------------------------------

def engineer_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed deviation features that capture per-reading anomalous spikes.

    Raw sensor readings (``voltage``, ``temperature_c``) contain two distinct
    signals mixed together:

    1. **Level** — the smoothed baseline captured by the rolling mean.
    2. **Spike** — the instantaneous deviation from that baseline.

    Using both the raw reading *and* its rolling mean as separate model
    features effectively double-weights the level signal while the spike
    signal is implicit and diluted.  Replacing the raw reading with the
    *signed deviation* cleanly separates the two signals:

    * ``voltage_rolling_mean``  → level/trend of voltage within the session.
    * ``voltage_deviation``     → instantaneous spike above/below that trend.

    A positive ``voltage_deviation`` means the voltage suddenly jumped above
    its recent average — a potential over-voltage transient.  A negative
    value means a sudden dip — possible under-voltage or load fault.

    .. math::

        \\text{voltage\\_deviation}  = \\text{voltage}  - \\text{voltage\\_rolling\\_mean}

        \\text{temp\\_deviation}     = \\text{temperature\\_c} - \\text{temperature\\_c\\_rolling\\_mean}

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after :func:`engineer_rolling_features` (both rolling-mean
        columns must already exist).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns:
        ``voltage_deviation`` and ``temp_deviation`` (both float).
    """
    df = df.copy()
    df["voltage_deviation"] = df["voltage"] - df["voltage_rolling_mean"]
    df["temp_deviation"]    = df["temperature_c"] - df["temperature_c_rolling_mean"]
    return df


# ---------------------------------------------------------------------------
# Stage 8 – Session aggregates
# ---------------------------------------------------------------------------

def engineer_session_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-session cumulative energy counter: ``session_energy_total``.

    ``session_energy_total`` is the cumulative sum of ``energy_kwh`` within
    each charging session, computed in chronological order.  It answers the
    question: *"How much energy has been delivered so far in this session?"*

    This feature is useful because:

    * Anomalies (e.g., sudden energy drops or spikes) show up as
      discontinuities in the cumulative curve.
    * It encodes *progress through the session*, giving the model temporal
      context without requiring it to track absolute timestamps.

    The cumulative sum is computed with
    ``groupby('session_id')['energy_kwh'].cumsum()``, so each row receives
    the running total up to and including that row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by ``session_id`` and ``timestamp`` (as produced
        by :func:`engineer_rolling_features`).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one additional column ``session_energy_total``
        (float, unit: kWh).
    """
    df = df.copy()
    df["session_energy_total"] = df.groupby("session_id")["energy_kwh"].cumsum()
    return df


# ---------------------------------------------------------------------------
# Stage 9 – Categorical flags
# ---------------------------------------------------------------------------

def engineer_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary fault flag: ``is_error``.

    The raw ``error_code`` column contains integer codes where ``0`` means
    *no fault* and any non-zero value means *a fault condition was active*.
    Many ML algorithms (and all anomaly detectors) benefit from having an
    explicit binary label for this:

    * ``is_error = 1``  →  ``error_code != 0``  (fault present)
    * ``is_error = 0``  →  ``error_code == 0``   (normal operation)

    Using a binary feature (rather than the raw code) avoids implying any
    ordinal relationship between fault codes (e.g., error 4 is not "twice as
    bad" as error 2).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame at any stage after :func:`clean_data`.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with one additional column ``is_error`` (int: 0 or 1).
    """
    df = df.copy()
    df["is_error"] = (df["error_code"] != 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_pipeline(
    filepath: str,
    rolling_window: int = 5,
    station_baselines: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Execute the full data-processing pipeline.

    Calls all pipeline stages in order:

    1.  :func:`load_data`
    2.  :func:`clean_data`
    3.  :func:`engineer_physics_features`
    4.  :func:`engineer_rolling_features`    (includes bfill/ffill)
    5.  :func:`engineer_time_features`
    6.  :func:`engineer_station_baselines`   (training: compute & freeze;
                                              inference: apply frozen values)
    7.  :func:`engineer_deviation_features`  (voltage_deviation, temp_deviation)
    8.  :func:`engineer_session_aggregates`
    9.  :func:`engineer_categorical_flags`

    Parameters
    ----------
    filepath : str
        Path to the raw ``charging_logs.csv`` file.
    rolling_window : int, optional
        Window size forwarded to :func:`engineer_rolling_features`.
        Defaults to ``5``.
    station_baselines : dict[str, float] or None, optional
        Pre-computed ``station_id → mean_temperature_c`` mapping from the
        training run.  Pass ``None`` during training (baselines will be
        computed from *filepath*'s data).  Pass the saved dict during
        inference to prevent distribution shift.

    Returns
    -------
    pd.DataFrame
        Fully processed DataFrame ready for EDA and model training.
        New columns added by this pipeline (beyond the raw schema):

        ============================  ==========================================
        Column                        Description
        ============================  ==========================================
        ``power_mismatch``            |P_reported − V×I/1000| in kW
        ``voltage_rolling_mean``      Rolling-mean voltage per session
        ``temperature_c_rolling_mean``Rolling-mean temperature per session
        ``voltage_deviation``         voltage − voltage_rolling_mean (spike)
        ``temp_deviation``            temperature_c − temp_c_rolling_mean (spike)
        ``hour``                      Hour of day (0–23)
        ``day_of_week``               Day of week (0=Mon … 6=Sun)
        ``hour_sin``                  Sine cyclical encoding of hour
        ``hour_cos``                  Cosine cyclical encoding of hour
        ``station_avg_temp``          Frozen training mean temperature (°C)
        ``session_energy_total``      Cumulative energy within session (kWh)
        ``is_error``                  Binary fault flag (1 if error_code != 0)
        ============================  ==========================================
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_physics_features(df)
    df = engineer_rolling_features(df, window=rolling_window)
    df = engineer_time_features(df)
    df = engineer_station_baselines(df, baselines=station_baselines)
    df = engineer_deviation_features(df)
    df = engineer_session_aggregates(df)
    df = engineer_categorical_flags(df)
    return df
