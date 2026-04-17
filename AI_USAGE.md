# AI Tool Usage & Documentation

## Tools Used

| Tool | Role |
|------|------|
| **Cursor (Claude — Sonnet)** | Primary coding agent: code generation, refactoring, docstrings, architecture discussion |
| **Gemini Pro 3.1** | Prompt engineering assistant — used to draft optimised prompts before sending to the main agent (chosen for its strong natural language understanding) |

---

## Prompting Strategy

As someone who works with AI systems professionally, I am mindful of token efficiency. Sending a vague, long-winded prompt to a large model wastes context and produces mediocre output.

My workflow was:

1. **Draft intent** — write a rough description of what I want to achieve.
2. **Cheap model pass** — feed the rough description to Gemini Pro 3.1 and ask it to rewrite the description as a precise, well-structured prompt. Gemini was chosen for this step specifically because of its strong natural language understanding — it produces cleaner, more contextually accurate prompt reformulations than smaller models.
3. **Main agent execution** — send the refined prompt to the primary coding agent (Cursor / Claude).

This approach produces sharper outputs on the first attempt, reduces back-and-forth correction rounds, and keeps token spend low. It also forces clarity of thought before any code is written — if you cannot explain the task precisely to a cheap model, you have not thought it through enough yet.

---

## How AI Was Used

### Domain Knowledge Acceleration

I used AI to quickly build up knowledge of EV charging domain specifics:

- The electrical identity **P = V × I** and what a violation of it implies (metering fault, sensor calibration error, dangerous operating condition)
- What `error_code` values like 101, 202, 404 typically represent in charging station firmware (OCPP protocol-level events: session timeouts, handshake failures, authorisation errors — not physical faults)
- Normal operating ranges for DC fast charger voltage and temperature
- What "anomaly" means practically in a NOC context — distinguishing silent physical faults from firmware-reported events

This let me spend time on design decisions rather than background research.

### Code Generation

Approximately **70–80% of the final code** was AI-generated. Specifically:

- All boilerplate: `argparse` CLI, `logging.basicConfig`, `joblib` save/load, `os.makedirs`, `sys.path` management
- The `find_contamination.py` script structure (elbow method scaffolding)
- Docstring formatting (NumPy style)
- The `evaluate_model.py` section structure once I defined the five evaluation dimensions

The remaining **20–30% was written by me directly** — primarily the logic for the deviation feature decomposition, the two-gate architecture wiring, the station baseline inference/training split, and the circular evaluation fix described below. This code was later reviewed and lightly refined by AI for style consistency in a final pass.

### Architecture Discussion

AI was useful as a sounding board for justifying decisions: why Isolation Forest over One-Class SVM, why `contamination=0.01`, the tradeoff between precision and recall when `is_error` is an imperfect label.

---

## Where I Led — AI Followed

These are cases where my own thinking defined the direction and AI implemented it:

### Time Features and Cyclical Encoding

I identified that temporal context matters — charging load patterns and fault rates vary by time of day. I added `hour` and `day_of_week` to the feature list. While thinking about how the model would use `hour` as an integer, I realised a problem: 23:00 and 00:00 are only one hour apart in reality, but as integers they have distance 23. I then researched cyclical encoding and chose `sin(2π × hour / 24)` and `cos(2π × hour / 24)` to map hours onto the unit circle. AI implemented the formula once I described the requirement.

### Station Temperature Baseline

I proposed `station_avg_temp` as a feature. My reasoning was twofold: stations in different environments (outdoor vs indoor) have different ambient temperatures, and newer or more efficient stations run at lower operating temperatures than older hardware. A temperature reading that is normal for an ageing outdoor station could be anomalous for a recently installed unit. AI implemented the groupby aggregation but did not surface this distinction unprompted.

### Two-Gate Hybrid Architecture

AI's default suggestion was a single Isolation Forest. I pushed for a two-gate system because I recognised that absolute physical impossibilities — negative current, out-of-spec voltage — are so rare in training data that a statistical model might assign them low anomaly scores simply due to insufficient examples. A hard-coded rules engine catches these deterministically regardless of training data distribution. AI built both gates once the architecture was decided.

### Deviation Feature Decomposition

I observed that having both raw `voltage` and `voltage_rolling_mean` in `TRAINING_FEATURES` double-weights the level signal while leaving the spike signal implicit. I specified that the model should receive the rolling mean (for gradual drift) and the signed deviation from it (for instantaneous spikes) as two orthogonal axes. AI implemented the arithmetic.

---

## Where AI Struggled

### The `error_code` Circular Evaluation Bug

AI initially included `error_code != 0` as Rule 5 in the Rules Engine. When `evaluate_model.py` ran, the Rules Engine showed 100% recall of hardware faults — a trivially true result since one of its rules was literally re-reading the same column used to define `is_error`. I identified the circularity: a non-trivial evaluation requires both the rules engine and the ML model to be independent of the label being evaluated against. Removing the rule made the recall result genuinely informative — and revealed the insight that firmware error codes have normal-looking sensor readings.

### `station_avg_temp` Data Leakage

AI's initial implementation computed `station_avg_temp` from the inference batch at prediction time. I identified that a small or skewed inference batch would compute very different per-station means than the `StandardScaler` was fitted on, causing distribution shift. The fix required saving the training baselines as `station_baselines.joblib` and loading them at inference — a train/inference consistency pattern that requires understanding the full lifecycle, not just the training script.

### Voltage Feature Redundancy

AI put raw `voltage` and `voltage_rolling_mean` together in `TRAINING_FEATURES` without flagging the overlap. The redundancy only becomes apparent when you think about what information each column actually encodes — a question of feature semantics, not code correctness.

---

## How AI-Generated Code Was Validated

- **Every function was read line by line** before being accepted. No function was merged without being able to explain every line.
- **End-to-end runs** after every significant change, checking row counts, anomaly percentages, absence of warnings.
- **Physics cross-checking** — the `power_mismatch` formula was verified against P = V×I independently before trusting it as a feature.
- **Evaluation sanity checks** — any metric that looked "too good" (100% recall, perfect overlap) was interrogated rather than accepted.
- **The `error_code` tautology** was caught by questioning a suspiciously clean result, not by automated testing.

---

## Summary

AI accelerated implementation by handling repetitive, convention-driven code and domain background research. Every non-trivial design decision — feature decomposition, hybrid architecture, evaluation methodology, data-leakage prevention — required human judgment and was not suggested by AI unprompted. The token-efficient prompting workflow (cheap model → refined prompt → main agent) kept iteration speed high without sacrificing output quality.
