# Data
Nothing is required here: with no file present, DriftGuard uses a built-in synthetic generator that
mimics AI4I 2020 (same 5 sensors, product type and documented failure rules).

To use the REAL dataset: `python scripts/download_ai4i.py`  (or download the CSV from
https://archive.ics.uci.edu/dataset/601 and save it as `data/ai4i2020.csv`).
Labels are always re-derived from the documented AI4I failure rules, so drift scenarios can change
the physics consistently; the real file supplies the sensor distributions.
