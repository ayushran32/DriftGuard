import numpy as np, pandas as pd
from driftguard.data import synth, failure_rules, make_features, load_raw_csv, DataSource, SENSORS


def test_failure_rate_is_realistic():
    rng = np.random.default_rng(0); d = synth(20000, rng); y = failure_rules(d, rng)
    assert 0.02 < y.mean() < 0.10          # AI4I is ~3.4% failures; synthetic stand-in must stay in the same ballpark


def test_features_shape():
    d = synth(50, np.random.default_rng(0))
    X = make_features(d)
    assert list(X.columns) == ["type_num"] + SENSORS + ["temp_diff"]


def test_loader_accepts_uci_column_names(tmp_path):
    d = synth(300, np.random.default_rng(0))
    raw = pd.DataFrame({"UDI": range(300), "Type": d.type, "Air temperature [K]": d.air_temp,
                        "Process temperature [K]": d.process_temp, "Rotational speed [rpm]": d.rpm,
                        "Torque [Nm]": d.torque, "Tool wear [min]": d.tool_wear, "Machine failure": 0})
    f = tmp_path / "ai4i.csv"; raw.to_csv(f, index=False)
    assert list(load_raw_csv(f).columns) == ["type"] + SENSORS
    src = DataSource(f, 0)
    assert src.real and len(src.train) + len(src.cal) + len(src.pool) == 300
