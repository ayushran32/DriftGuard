"""Data layer.

Uses the real AI4I 2020 CSV if you place it at data/ai4i2020.csv, otherwise a synthetic
generator that mimics AI4I's structure (same 5 sensors + product type, same documented
failure modes). Labels are ALWAYS re-derived from the documented failure rules so that
drift scenarios can change the physics/labels consistently.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

SENSORS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}
COLMAP = {"air_temperature": "air_temp", "process_temperature": "process_temp",
          "rotational_speed": "rpm", "torque": "torque", "tool_wear": "tool_wear", "type": "type"}


def load_raw_csv(path):
    """Read the UCI AI4I 2020 csv; tolerant to column names with/without units like '[K]'."""
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\[.*?\]", "", c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=COLMAP)
    missing = [c for c in ["type"] + SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    return df[["type"] + SENSORS]


def synth(n, rng):
    """Synthetic AI4I-like machine states."""
    typ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = 1168 + rng.gamma(4.0, 95.0, n)
    z = (rpm - 1548) / 190
    torque = np.clip(40 + 10 * (-0.87 * z + 0.49 * rng.normal(size=n)), 3, None)
    wear = np.clip(rng.normal(108, 63, n), 0, 253).round()
    return pd.DataFrame(dict(type=typ, air_temp=air, process_temp=proc, rpm=rpm,
                             torque=torque, tool_wear=wear))


def failure_rules(df, rng, osf_scale=1.0):
    """AI4I documented failure modes (heat dissipation, power, overstrain, tool wear, random)."""
    power = df["torque"].values * df["rpm"].values * 2 * np.pi / 60
    lim = df["type"].map(OSF_LIMIT).values * osf_scale
    hdf = ((df["process_temp"] - df["air_temp"]).values < 8.6) & (df["rpm"].values < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = df["tool_wear"].values * df["torque"].values > lim
    twf = (df["tool_wear"].values >= 200) & (rng.random(len(df)) < 0.12)
    rnf = rng.random(len(df)) < 0.001
    return (hdf | pwf | osf | twf | rnf).astype(int)


def make_features(df):
    X = df[SENSORS].copy()
    X.insert(0, "type_num", df["type"].map({"L": 0, "M": 1, "H": 2}).values)
    X["temp_diff"] = df["process_temp"] - df["air_temp"]
    return X


def train_model(df, y, sample_weight=None, seed=0):
    m = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=-1,
                               class_weight="balanced_subsample", random_state=seed)
    m.fit(make_features(df), y, sample_weight=sample_weight)
    return m


def predict(model, df):
    return model.predict(make_features(df))


class DataSource:
    def __init__(self, csv=None, seed=0):
        rng = np.random.default_rng(seed)
        self.real = False
        if csv is not None and Path(csv).exists():
            raw = load_raw_csv(csv).sample(frac=1, random_state=seed).reset_index(drop=True)
            n = len(raw)
            a, b = int(.5 * n), int(.7 * n)
            self.train, self.cal, self.pool = (raw.iloc[:a].reset_index(drop=True),
                                               raw.iloc[a:b].reset_index(drop=True),
                                               raw.iloc[b:].reset_index(drop=True))
            self.real = True
        else:
            self.train, self.cal, self.pool = synth(8000, rng), synth(3000, rng), synth(6000, rng)
        self.train_y = failure_rules(self.train, rng)
        self.cal_y = failure_rules(self.cal, rng)
        self.pool_y = failure_rules(self.pool, rng)
        self.base_model = train_model(self.train, self.train_y, seed=seed)

    def window(self, n, rng):
        if self.real:
            return self.pool.iloc[rng.integers(0, len(self.pool), n)].reset_index(drop=True)
        return synth(n, rng)
