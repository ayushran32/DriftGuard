# DriftGuard: design details

## 1. Problem
A failure-prediction model is trained once, but sensors drift, break or get recalibrated and operating
conditions change. A plain drift alert does not say *why* the data changed, and the right response differs:
a faulty sensor should be repaired (the machine did not change), a real regime shift may need a new model,
a harmless shift needs nothing.

## 2. Pipeline (see docs/DriftGuard_architecture.png)
Sensor window (500 rows) -> Detection -> Triage -> [Online repair | Guarded retraining] -> Prediction -> Dashboard/log.
Delayed failure labels (default lag = 2 windows) feed the label monitor and the retraining guard.

| Module | File | What it does |
|---|---|---|
| Data | `driftguard/data.py` | AI4I-like synthetic generator or real CSV; failure rules (HDF, PWF, OSF, TWF, RNF); features; Random Forest baseline |
| Scenarios | `driftguard/simulate.py` | 7 scenarios. Sensor faults corrupt only the *observation*; regime / concept drift change the true state or failure physics |
| Detection | `driftguard/detect.py` | KS D-statistic + PSI per sensor, k-means occupancy shift + quantisation-error growth; thresholds from a bootstrap of no-drift windows of the same size |
| Triage + repair | `driftguard/triage.py` | Fault signatures, cause classification, regression/bias-correction repair |
| Engine | `driftguard/engine.py` | DriftGuard loop, label monitor, guarded retraining, re-baselining; baselines (Static, Naive, Naive recent-only, Feature-vote) |
| UI | `app.py`, `demo_cli.py` | Streamlit dashboard; terminal demo |
| Benchmark | `run_experiments.py` | All scenarios x all strategies x N seeds |

## 3. Data
Reference (train) / calibration / pool splits. Synthetic mode: 8000 / 3000 / 6000 rows. Real CSV mode (10,000 rows):
50% / 20% / 30%. Labels are re-derived from the documented AI4I failure rules so drift can change them consistently.
Live windows are drawn from the pool (real mode: with replacement) and perturbed by the scenario.

## 4. Algorithms and parameters
**Detection.** Per sensor: KS D-statistic and PSI (10 reference-quantile bins). Multivariate: k-means (k=6) on standardised
sensors; PSI of cluster occupancy and relative growth of mean quantisation error. Each component is divided by its 99.5th
percentile over 400 bootstrap no-drift windows; window score = max component; alert threshold = 1.1 x max(1, 99th
percentile of the bootstrap composite). An alert needs 2 consecutive windows (sensor faults alert immediately).

**Triage rules.** variance ratio < 0.1 -> stuck; > 2.5 -> noise; process-air temperature difference shifts by > 3 reference
standard deviations -> offset (sensor with the larger shift is blamed). No fault and only tool_wear drifts -> FLEET_AGING;
no fault and other drift -> REGIME_SHIFT; inputs normal but label monitor alarms -> CONCEPT_DRIFT.

**Repair.** offset: subtract the estimated bias; stuck/noisy: linear regression from the remaining healthy sensors (fitted on reference data).

**Label monitor.** Pooled F1 over the last 3 labelled windows vs a lower band (LABEL_Q = 0.5th percentile of bootstrap no-drift F1);
two consecutive breaches = alarm.

**Guarded retraining.** Pending retrain starts on a regime-shift alert or label alarm. After >= 4 labelled windows:
(1) if the drift came from inputs and pooled F1 is still inside the band -> log HARMLESS, no retrain;
(2) else train challengers (recent-only; blended with history at weight 0.25), pick the best on the last 2 labelled windows;
(3) deploy only if a paired bootstrap (200 resamples) shows it beats the current model by > 0.01 F1 in >= 90% of resamples,
refit on all labelled windows; otherwise REJECT (max 2 attempts, then 12-window acknowledgement);
(4) after deployment, re-baseline detectors on the new normal and reset the label band.

**Impact estimate (display only).** drift per sensor weighted by permutation importance of that sensor in the model.

## 5. Evaluation
Strategies: Static; Naive drift->retrain (KS p<0.05 on any feature, blind deploy); Naive recent-only; Feature-vote
(drift if >= 3 of 5 features fail KS, blind deploy); DriftGuard. Metrics: mean F1 after onset, F1 over last 10 windows, false-alarm
windows before onset (control: all windows), detection rate and delay, models deployed, triage correctness.

## 6. Limits
Simulated drift on a synthetic benchmark; hand-designed fault rules for 5 sensors; slow creeping sensor drift would be read as a
regime shift; labels needed for retraining and concept-drift detection; small seed count; not benchmarked against NannyML/Evidently.
