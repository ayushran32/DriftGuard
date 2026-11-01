# DriftGuard: drift-cause triage, input self-repair and guarded retraining for predictive maintenance

RVCE ISE, AIML (IS353IA) Lab Part B. Students: Atnakuri Siva Sai Lakshman (1RV24IS019), Ayush Ranjan (1RV24IS022).

## Idea
A failure-prediction model degrades when sensors drift or break. DriftGuard detects drift, diagnoses the CAUSE
(sensor fault / regime shift / fleet ageing / concept drift) and applies the matching action: repair a faulty sensor
without retraining, or retrain only when delayed labels show real harm and a challenger model is validated.
Novelty and what is adopted: docs/NOVELTY.md. Design details: docs/DESIGN.md. Diagram: docs/DriftGuard_architecture.png.

## Setup
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements-dev.txt
    python -m pytest -q                                   # 14 tests, ~20 s
    streamlit run app.py                                  # dashboard
    python demo_cli.py --scenario process_temp_offset     # terminal demo, saves a plot to outputs/
    python run_experiments.py 3                           # full benchmark (~12 min) -> results_summary.csv
    python scripts/download_ai4i.py                       # optional: real AI4I data -> data/ai4i2020.csv

## Scenarios
control, ambient_warming, heavy_load, new_tool_material (concept drift), process_temp_offset, torque_stuck, rpm_noise.

## Example output (one seed, synthetic data, `demo_cli.py --scenario process_temp_offset`)
Mean F1 after onset: Static 0.691, Naive drift->retrain 0.723 (5 models deployed), DriftGuard 0.865 (0 deployed, SENSOR_FAULT diagnosed).
This is a single run. Multi-seed results are not included; run `run_experiments.py` to generate your own.

## Honest limits
Synthetic AI4I-like data unless the real CSV is added; drift is injected; fault rules are hand-designed for 5 sensors;
slow creeping sensor drift is read as a regime shift; not benchmarked against NannyML/Evidently. The real-CSV loader was
tested with a stand-in file in the AI4I format; the real file and the download script were not tested (no UCI access when built).
