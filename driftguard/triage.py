"""Drift-cause triage + input self-repair.

Not all drift is the same. A *sensor fault* (stuck / noisy / mis-calibrated) must NOT trigger
retraining -- the machine did not change, the instrument did -- and can be repaired online by
re-estimating the faulty channel from correlated sensors. A genuine *operating-regime shift*
may need retraining. Tool-wear drift is expected fleet aging (informational).
"""
import numpy as np
from .data import SENSORS

AIR, PROC = "air_temp", "process_temp"


class Triage:
    def __init__(self, ref_df):
        self.ref = ref_df[SENSORS].copy()
        self.mu, self.var = self.ref.mean(), self.ref.var()
        d = self.ref[PROC] - self.ref[AIR]
        self.d_mu, self.d_sd = d.mean(), d.std()
        self._reg = {}

    def signatures(self, obs):
        return {s: dict(shift_z=float((obs[s].mean() - self.mu[s]) / np.sqrt(self.var[s])),
                        var_ratio=float(obs[s].var() / self.var[s])) for s in SENSORS}

    def detect_faults(self, obs):
        faults = []
        for s in SENSORS:
            vr = obs[s].var() / self.var[s]
            if vr < 0.1:
                faults.append((s, "stuck"))
            elif vr > 2.5:
                faults.append((s, "noise"))
        zd = ((obs[PROC] - obs[AIR]).mean() - self.d_mu) / self.d_sd
        if abs(zd) > 3:   # temperatures no longer physically consistent -> one sensor is offset
            bad = PROC if abs(obs[PROC].mean() - self.mu[PROC]) >= abs(obs[AIR].mean() - self.mu[AIR]) else AIR
            faults.append((bad, "offset"))
        return faults

    @staticmethod
    def classify(drifting, faults, multivar):
        if faults:
            return "SENSOR_FAULT"
        if not drifting and not multivar:
            return "NONE"
        if drifting and set(drifting) <= {"tool_wear"}:
            return "FLEET_AGING"
        return "REGIME_SHIFT"

    def _fit(self, target, others):
        key = (target, tuple(others))
        if key not in self._reg:
            A = np.column_stack([self.ref[others].values, np.ones(len(self.ref))])
            self._reg[key] = np.linalg.lstsq(A, self.ref[target].values, rcond=None)[0]
        return self._reg[key]

    def repair(self, obs, faults):
        """Replace faulty channels by a regression estimate from the healthy ones."""
        fixed = obs.copy()
        for s, kind in faults:              # offset: subtract the estimated bias (keeps the sensor's own signal)
            if kind == "offset":
                bias = ((obs[PROC] - obs[AIR]).mean() - self.d_mu) * (1 if s == PROC else -1)
                fixed[s] = obs[s] - bias
        bad = {s for s, k in faults if k != "offset"}
        good = [s for s in SENSORS if s not in bad]
        for s in bad:
            w = self._fit(s, good)
            fixed[s] = np.column_stack([obs[good].values, np.ones(len(obs))]) @ w
        return fixed
