import pytest
from driftguard.data import DataSource
from driftguard.simulate import build_stream
from driftguard.engine import DriftGuard, Static, run_strategy


@pytest.fixture(scope="module")
def src():
    return DataSource(None, 0)


def run(src, scenario, cls):
    return run_strategy(cls, src, build_stream(src, scenario, 40, 500, 12, 0), 500, 2, 0)


def test_control_has_no_alerts_and_no_deployments(src):
    s, reps, _ = run(src, "control", DriftGuard)
    assert not any(r["alert"] for r in reps) and s.version == 0


def test_sensor_fault_is_triaged_repaired_not_retrained(src):
    s, reps, f = run(src, "process_temp_offset", DriftGuard)
    _, _, f_static = run(src, "process_temp_offset", Static)
    assert any(r["alert"] and r["cause"] == "SENSOR_FAULT" for r in reps[12:])
    assert s.version == 0                                  # no retraining for a sensor fault
    assert f[12:].mean() > f_static[12:].mean() + 0.05     # repair clearly helps


def test_concept_drift_found_through_delayed_labels(src):
    s, reps, f = run(src, "new_tool_material", DriftGuard)
    _, _, f_static = run(src, "new_tool_material", Static)
    assert any(r["cause"] == "CONCEPT_DRIFT" for r in reps[12:])
    assert f[12:].mean() > f_static[12:].mean()


def test_regime_shift_triggers_validated_response(src):
    s, reps, f = run(src, "heavy_load", DriftGuard)
    assert any(r["alert"] and r["cause"] == "REGIME_SHIFT" for r in reps[12:])
    assert {"DEPLOY", "HARMLESS", "REJECT"} & {e[1] for e in s.log}   # a guarded decision was logged
