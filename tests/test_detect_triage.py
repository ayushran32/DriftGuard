import numpy as np, pytest
from driftguard.data import DataSource, synth
from driftguard.detect import Detector, ks_stat, psi_from_props
from driftguard.triage import Triage


@pytest.fixture(scope="module")
def env():
    src = DataSource(None, 0)
    return src, Detector(src.train, src.cal, 500, seed=0, n_null=150), Triage(src.train)


def win(seed=1):
    return synth(500, np.random.default_rng(seed))


def test_ks_and_psi_basics():
    a = np.sort(np.random.default_rng(0).normal(size=2000))
    assert ks_stat(a, np.random.default_rng(1).normal(size=500)) < 0.1
    assert ks_stat(a, np.random.default_rng(1).normal(2, 1, size=500)) > 0.5
    p = np.array([.25, .25, .25, .25])
    assert psi_from_props(p, p) == pytest.approx(0, abs=1e-9)


def test_detector_calibrated_on_clean_windows(env):
    _, det, _ = env
    over = sum(det.score(win(s))["composite"] > det.comp_thr for s in range(100, 140))
    assert over <= 4                       # <=10% single-window false alarms on 40 fresh clean windows


def test_detector_flags_real_shift(env):
    _, det, _ = env
    w = win(); w["air_temp"] += 6; w["process_temp"] += 6
    assert det.score(w)["composite"] > det.comp_thr


def test_triage_sensor_faults(env):
    _, _, tri = env
    w = win(); w["torque"] = 40.0
    assert ("torque", "stuck") in tri.detect_faults(w)
    w = win(); w["rpm"] += np.random.default_rng(0).normal(0, 300, 500)
    assert ("rpm", "noise") in tri.detect_faults(w)
    w = win(); w["process_temp"] += 6
    assert ("process_temp", "offset") in tri.detect_faults(w)


def test_triage_does_not_blame_sensor_for_regime_shift(env):
    _, det, tri = env
    w = win(); w["air_temp"] += 6; w["process_temp"] += 6      # both move together: physical change, not a fault
    assert tri.detect_faults(w) == []
    s = det.score(w)
    assert tri.classify(s["drifting"], [], s["multivar"]) == "REGIME_SHIFT"


def test_clean_window_has_no_faults(env):
    assert env[2].detect_faults(win()) == []


def test_repair_reduces_error(env):
    _, _, tri = env
    truth = win(); bad = truth.copy(); bad["process_temp"] += 6
    fixed = tri.repair(bad, tri.detect_faults(bad))
    assert abs(fixed["process_temp"] - truth["process_temp"]).mean() < abs(bad["process_temp"] - truth["process_temp"]).mean() / 3
