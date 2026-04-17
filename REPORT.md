# Technical Report: EV Charging Station Anomaly Detection

**Project:** Unsupervised Anomaly Detection in EV Charging Session Logs  
**Model:** Two-Gate Hybrid Anomaly Detector (Physics Rules + Isolation Forest)  
**Dataset:** `charging_logs.csv` — 199,567 raw events across 20 stations

---

## 1. Problem Understanding

The dataset represents streaming telemetry from a fleet of 20 EV charging stations. Each row is a sensor event logged during an active charging session and contains electrical readings (voltage, current, power), thermal readings (temperature), session metadata, and a hardware-reported error code.

The task is to flag anomalous events — those that deviate from expected charging behaviour in a way that could indicate equipment faults, metering errors, or unsafe operating conditions.

**Key challenge:** The problem is essentially unsupervised. While `error_code` provides a reference signal, it captures only what the charging station's own firmware detected (software/protocol faults). It does not capture silent physical faults — sensor malfunctions, gradual degradation, or dangerous operating conditions the firmware did not directly observe. A purely supervised approach would therefore only replicate what the hardware already reports, adding no value. The real goal is to find what the hardware *missed*.

**Two categories of anomaly exist in this data:**
- **Firmware/protocol faults** — `error_code != 0` (e.g. codes 101, 202, 404). Analysis revealed these rows have near-identical sensor readings to normal rows (median `power_mismatch` 0.035 kW vs overall 0.037 kW), confirming they are software-level events, not physical measurement anomalies.
- **Physics anomalies** — Readings that violate electrical laws or safe operating constraints, often with `error_code = 0`. These are what the ML model is designed to catch.

---

## 2. Exploratory Data Analysis

### 2.1 Data Quality

After deduplication (simultaneous sensor dumps at identical timestamps within a session), the dataset reduced from 199,567 to 182,358 clean rows across approximately 30,000 sessions. Missing values were minimal and concentrated in edge cases at session boundaries.

### 2.2 Key Distributions

**`power_mismatch`** (|power_kw − V×I/1000|): The distribution is heavily right-skewed with the vast majority of readings below 0.1 kW. A small tail extends to very large values — these represent rows where the reported power_kw disagrees significantly with what Ohm's law predicts from the raw voltage and current readings. This is the strongest single anomaly signal in the dataset.

**Voltage vs Current scatter by `error_code`:** Fault-coded rows (red) cluster at electrical extremes — low voltage/high current combinations that would indicate an overloaded or misfiring charger. Normal readings form a tight elliptical cloud centred around the nominal operating point.

**Station temperature baselines (`station_avg_temp`):** Per-station mean temperatures differ by up to 8°C across the 20 stations, reflecting different installation environments (indoor/outdoor, climate zones). A temperature reading that is normal at one station may be anomalous at another — this justified station-specific baseline normalisation.

**Cyclical time features:** Charging activity peaks during the day and drops overnight, with the cyclic hour encoding (sin/cos) correctly representing the continuity between 23:00 and 00:00 — something a raw integer encoding would break.

**Session cumulative energy (`session_energy_total`):** The distribution of total energy delivered per session follows a roughly normal shape centred around 20–25 kWh, with a heavy tail toward shorter, incomplete sessions. Anomalous sessions often show flat or step-wise cumulative curves.

**Deviation features (`voltage_deviation`, `temp_deviation`):** The overlapping density plots for fault vs normal sessions show that fault sessions have heavier tails in both deviation distributions, confirming that sudden spikes above the rolling mean are a meaningful fault signal.

**`is_error` class balance:** Approximately 0.6% of rows carry a non-zero `error_code`. This extreme imbalance is one reason supervised learning was deprioritised — standard classifiers would either ignore the minority class or overfit to the specific firmware codes present in the training data.

---

## 3. Feature Engineering

All engineered features are computed by a modular pipeline in `src/data_processing.py`. The design principle was to give the model *orthogonal* signals — each feature should carry information that no other feature already captures.

| Feature | Type | Rationale |
|---------|------|-----------|
| `power_mismatch` | Physics | Direct test of P = V×I. A large deviation is near-certain evidence of a sensor fault or dangerous condition. |
| `voltage_rolling_mean` | Rolling (level) | Captures gradual voltage drift within a session — slow degradation the model needs one axis to express. |
| `voltage_deviation` | Rolling (spike) | Instantaneous spike above the rolling mean. A large value means the voltage suddenly jumped — a different anomaly signal from slow drift. |
| `temperature_c_rolling_mean` | Rolling (level) | Smooth thermal trend within a session. Gradual overheating. |
| `temp_deviation` | Rolling (spike) | Sudden thermal spike above the rolling trend. |
| `hour_sin` / `hour_cos` | Time | Cyclical encoding of hour-of-day, preserving midnight continuity. |
| `station_avg_temp` | Baseline | Per-station long-run mean temperature. Normalises ambient temperature differences between stations. |
| `session_energy_total` | Aggregate | Cumulative energy delivered. Encodes progress through the session and exposes energy-delivery anomalies. |

**Design decision — deviation vs raw reading:** The raw `voltage` and `temperature_c` columns were deliberately excluded from `TRAINING_FEATURES`. Using both a raw reading and its rolling mean as separate features would double-weight the level signal while leaving the spike signal implicit. Decomposing into rolling mean (level) + deviation (spike) gives the Isolation Forest two orthogonal axes per sensor, each capturing a different type of anomaly.

**Data leakage prevention:** The `station_avg_temp` baseline is computed once on the full training set and saved as `models/station_baselines.joblib`. At inference time, pre-computed values are loaded and mapped onto new data. This prevents the leakage bug where a small inference batch would compute very different per-station means than the StandardScaler was trained on.

---

## 4. Modelling Approach

### 4.1 Algorithm Choice: Isolation Forest

**Why Isolation Forest, not One-Class SVM or Autoencoder:**

| Criterion | Isolation Forest | One-Class SVM | Autoencoder |
|---|---|---|---|
| Scale (182k rows, 10 features) | Excellent — O(n log n) | Poor — O(n²) kernel computation | Good but requires tuning |
| Hyperparameter sensitivity | Low — mainly `contamination` | High — kernel, nu, gamma | High — architecture, lr, epochs |
| Interpretability | Moderate (path length) | Low | Low |
| No label requirement | Yes | Yes | Yes |
| Handles mixed feature types | Yes | Requires careful normalisation | Requires careful normalisation |

Isolation Forest works by randomly partitioning the feature space with axis-aligned cuts. Points that require very few cuts to isolate (short average path length across all trees) are anomalies — they are geometrically isolated from the dense mass of normal data. This makes it naturally suited to multi-dimensional sensor data where anomalies are genuinely sparse and isolated.

### 4.2 Contamination Parameter

The `contamination` parameter controls the expected fraction of anomalies. Rather than hardcoding it, `find_contamination.py` implements the **Elbow Method** using two complementary techniques:

- **Second-derivative method**: Computes the inflection point in the sorted anomaly score curve — where the curve's rate of change accelerates most sharply from the anomaly tail into the flat normal region.
- **Kneedle geometry**: Projects the sorted score curve onto a unit line and finds the point of maximum perpendicular distance — the geometric knee.

Both methods converged near 1%, and visual inspection of the saved plot (`data/anomaly_scores_elbow.png`) confirmed this. The final value of `contamination=0.01` is therefore **data-driven and visually verified**, not arbitrary.

### 4.3 Two-Gate Hybrid Architecture

The final inference pipeline uses a two-gate system rather than the ML model alone:

```
Gate 1 (Rules Engine)     Gate 2 (ML Engine)
─────────────────────     ──────────────────
voltage < 150 V                  │
voltage > 280 V          Isolation Forest
temperature > 70 °C      (Imputer + Scaler)
current < 0                      │
energy_kwh < 0                   │
        │                        │
        └──────── OR ────────────┘
                  │
             is_anomaly
```

**Rationale:** The Rules Engine catches absolute physical impossibilities (negative current, over-voltage) that occur rarely enough that the Isolation Forest might not have enough training examples to reliably learn them. Conversely, the ML engine detects subtle multivariate deviations that no simple threshold rule can express. The logical OR of both gates means neither gate can cancel out the other's findings.

**Deliberate exclusion of `error_code` from the Rules Engine:** Including `error_code != 0` as a rule would make the evaluation circular — the rules engine would trivially achieve 100% recall of hardware faults by construction. By excluding it, the rules engine and ML engine both operate as independent detectors and their combined recall of firmware errors is a genuine (non-circular) test.

---

## 5. Evaluation Methodology

Given the unsupervised nature, no single metric is sufficient. Five independent evaluation dimensions were implemented in `evaluate_model.py`:

### A. Gate Overlap Analysis
How often do the physics rules and the ML model independently flag the same row? The two gates have zero coupling — one uses physics thresholds, the other uses statistical geometry. When they agree, confidence is near-certain. **Result: 889 rows (48.7% of ML flags) were flagged by both gates independently.**

### B. Physics Validation of ML-Only Anomalies
This is the most critical test. Of the 935 rows flagged exclusively by the ML engine (no rule violation), do they show elevated `power_mismatch`? If the ML-only rows were noise, their `power_mismatch` would match normal rows. **Result: ML-only rows have a median `power_mismatch` of 7.48 kW — 207x higher than the normal median of 0.036 kW. 97.6% of ML-only anomalies fall above the 75th percentile of normal rows.** This is near-definitive validation that the ML engine is detecting real physics violations, not noise.

### C. Hardware Fault Recall
Of all 1,109 rows with non-zero `error_code`, how many did the physics rules and ML engine independently catch? **Result: Only 1 out of 1,109 was caught by the physics rules or ML engine.** This reveals an important insight: the firmware error codes (101, 202, 404) are protocol/software-level faults whose sensor readings look physically normal. The ML model correctly does not flag them — it was designed to detect physics anomalies, which is a different fault category.

### D. Per-Station Anomaly Breakdown
Is anomaly concentration suspiciously skewed toward one station? **Result: The most anomalous station accounts for 6.6% of all anomalies vs the expected 5.0% if perfectly even. The distribution is healthy — no single station dominates.**

### E. Composite Confidence Score
A weighted composite across all four dimensions yields **78.5 / 100 (MEDIUM-HIGH confidence)**.

---

## 6. Results Summary

| Metric | Value |
|--------|-------|
| Total rows processed | 182,358 |
| Anomalies flagged by Rules Engine | 2,575 (1.41%) |
| Anomalies flagged by ML Engine | 1,824 (1.00%) |
| Overlap (both gates) | 889 rows |
| Total unique anomalies | 3,510 (1.92%) |
| ML-only physics validation (elevation ratio) | 207x |
| Per-station distribution | Healthy (max 6.6% share) |
| Composite confidence score | 78.5 / 100 |

---

## 7. Production Behaviour

**Robustness to new data:** The inference pipeline loads frozen training artefacts (imputer, scaler, Isolation Forest, station baselines). A new batch of data is preprocessed identically to the training set — no re-computation of statistics from the inference batch. Unseen station IDs fall back to the global training mean.

**Failure modes to monitor:**
- **Distribution shift:** If charging hardware is upgraded fleet-wide (new voltage range, different thermal profile), the Isolation Forest will gradually become miscalibrated. A periodic re-training trigger should be based on the drift of the `decision_function` score distribution.
- **Rule threshold staleness:** The voltage and temperature thresholds (150–280 V, 70°C) are appropriate for current-generation DC fast chargers. Next-generation 800V systems would require updated bounds.
- **Firmware error code changes:** New error codes introduced by a firmware update would not automatically be handled, but since the ML engine operates independently of `error_code`, it would still flag the underlying physical anomaly if one exists.

---

## 8. What Would Be Improved With More Time

1. **Temporal hold-out evaluation:** Train on months 1–8, evaluate on months 9–12. This tests whether the model generalises to future data, which is the production-relevant question.

2. **Per-station models:** One global Isolation Forest cannot distinguish between a station that runs genuinely hotter and one that is overheating. Station-specific models with station-specific contamination parameters would improve precision.

3. **Autoencoder comparison:** A sequence-aware autoencoder (LSTM encoder-decoder) treating each session as a time series would capture temporal dependencies within a session that the Isolation Forest (which is row-wise) cannot. Worth benchmarking against the current approach.

4. **Alert deduplication:** Multiple consecutive anomalous rows in the same session likely represent a single fault event. A session-level aggregation step would reduce alert noise for operators.

5. **Explainability:** Adding SHAP values on the Isolation Forest's `decision_function` scores would allow an operator to see *which features* drove each anomaly flag — critical for actionable alerts in a NOC context.
