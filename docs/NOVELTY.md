# What is adopted and what is new

## Adopted (standard)
KS test, PSI, k-means, Random Forest, bootstrap resampling, champion/challenger deployment pattern, regression imputation.

## Contribution (integration + specific design)
- **N1 Cause-specific triage.** An alert is classified as SENSOR_FAULT, REGIME_SHIFT, FLEET_AGING or CONCEPT_DRIFT using
  variance signatures and a physical consistency check (process temperature tracks air temperature), instead of one undifferentiated "drift".
- **N2 Online repair instead of retraining for sensor faults.** The sensor-vs-process distinction (Lee et al. [5]) is wired into
  a live predictive-maintenance loop: the faulty channel is repaired and the model is left alone.
- **N3 Harm-gated, validated retraining.** Following Dong et al. [6] (not all drift hurts), retraining is attempted only after delayed
  labels show harm, and a challenger is deployed only if a paired bootstrap says it is better; harmless drift is logged and acknowledged.
- **N4 Bootstrap-calibrated thresholds.** Every statistic is normalised by thresholds learned from no-drift windows of the live window size.

## Do not claim
- "No such system exists." The literature reviewed shows the pieces separately; we combine them for predictive maintenance.
- Real-plant performance. Drift scenarios are injected into a synthetic benchmark.
- Superiority over NannyML/Evidently. They were not benchmarked. (Białek et al. [8] label-free estimation is future work.)
- Anything about results beyond what `results_summary.csv` shows (3 seeds).

## Related work (verified entries; see REFERENCES.md)
[2],[3] drift in predictive maintenance; [4] survey; [5] sensor failure vs process anomaly; [6] harmful vs harmless drift;
[7] confidence/uncertainty as a drift signal; [8] label-free performance estimation; [1] dataset.
