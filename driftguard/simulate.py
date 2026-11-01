"""Drift / fault scenario injection.

Each scenario separates the TRUE machine state (drives failure labels) from the OBSERVED
sensor readings (what the model sees). Sensor faults corrupt only the observation;
regime / concept drift changes the true state or the failure physics.
"""
from dataclasses import dataclass
import numpy as np
from .data import failure_rules


@dataclass
class Scenario:
    name: str
    description: str
    expected_cause: str   # what a correct triage should say


SCENARIOS = {s.name: s for s in [
    Scenario("control", "No drift (false-alarm test)", "NONE"),
    Scenario("ambient_warming", "Gradual +6K warming of air & process temp (diff preserved)", "REGIME_SHIFT"),
    Scenario("heavy_load", "Sudden heavier machining: torque +12 Nm, rpm -80", "REGIME_SHIFT"),
    Scenario("new_tool_material", "Overstrain limit drops 30% (inputs unchanged: concept drift)", "CONCEPT_DRIFT"),
    Scenario("process_temp_offset", "Process-temp sensor calibration offset of +6K", "SENSOR_FAULT"),
    Scenario("torque_stuck", "Torque sensor stuck at a constant value", "SENSOR_FAULT"),
    Scenario("rpm_noise", "rpm sensor noise burst (electrical interference)", "SENSOR_FAULT"),
]}


def build_stream(source, scenario, n_windows=40, window=500, onset=12, seed=0):
    rng = np.random.default_rng(seed + 1000)
    stream = []
    for t in range(n_windows):
        true = source.window(window, rng)
        obs, osf, on = None, 1.0, t >= onset
        if on:
            if scenario == "ambient_warming":
                p = min(1.0, (t - onset + 1) / 10)
                true["air_temp"] += 6 * p
                true["process_temp"] += 6 * p
            elif scenario == "heavy_load":
                true["torque"] += 12
                true["rpm"] -= 80
            elif scenario == "new_tool_material":
                osf = 0.7
        obs = true.copy()
        if on:
            if scenario == "process_temp_offset":
                obs["process_temp"] += 6
            elif scenario == "torque_stuck":
                obs["torque"] = 40.0
            elif scenario == "rpm_noise":
                obs["rpm"] += rng.normal(0, 300, len(obs))
        y = failure_rules(true, rng, osf)
        stream.append(dict(t=t, obs=obs, y=y, active=on))
    return stream
