# Data
The **real AI4I 2020 Predictive Maintenance Dataset** (`ai4i2020.csv`, 10,000 records from the UCI Machine Learning Repository) is included directly in this directory and automatically detected by `driftguard.data.DataSource`.

If `data/ai4i2020.csv` is absent, DriftGuard automatically falls back to its built-in statistical generator `synth()` that mimics AI4I sensor distributions and physics failure rules.

To re-download the real dataset: `python scripts/download_ai4i.py`.
