"""Drift detection written from the statistical definitions (no drift library used).

Components per window: KS D-statistic and PSI for every sensor + two multivariate signals
(k-means occupancy shift, k-means quantization-error growth). Every component is normalised
by a threshold learned from a bootstrap of no-drift windows *of the same size as the live
window*, so alert levels are calibrated (no hand-tuned p-values, no window-size bias).
"""
import numpy as np
from .data import SENSORS


def ks_stat(a_sorted, b):
    b = np.sort(b)
    allv = np.concatenate([a_sorted, b])
    ca = np.searchsorted(a_sorted, allv, side="right") / len(a_sorted)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return np.abs(ca - cb).max()


def psi_from_props(p, q, eps=1e-4):
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))


def kmeans_fit(X, k, rng, iters=30):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C


class Detector:
    def __init__(self, ref_df, cal_df, window, seed=0, n_null=400, k=6):
        self.rng = np.random.default_rng(seed)
        self.W, self.k = window, k
        self.ref_sorted = {s: np.sort(ref_df[s].values) for s in SENSORS}
        self.edges = {s: np.unique(np.quantile(ref_df[s].values, np.linspace(0, 1, 11)[1:-1]))
                      for s in SENSORS}
        self.ref_props = {s: self._props(ref_df[s].values, s) for s in SENSORS}
        self.mu = ref_df[SENSORS].mean().values
        self.sd = ref_df[SENSORS].std().values
        Z = (ref_df[SENSORS].values - self.mu) / self.sd
        self.C = kmeans_fit(Z, k, self.rng)
        lab, qe = self._assign(Z)
        self.km_props = np.bincount(lab, minlength=k) / len(lab)
        self.km_qe = qe.mean()
        # bootstrap null of every component using no-drift windows of the live window size
        null = [self.components(cal_df.iloc[self.rng.integers(0, len(cal_df), window)])
                for _ in range(n_null)]
        self.names = list(null[0].keys())
        self.thr = {c: max(np.quantile([r[c] for r in null], 0.995), 1e-6) for c in self.names}
        comp = [self._composite(self.normalise(r)) for r in null]
        self.comp_thr = 1.1 * max(1.0, float(np.quantile(comp, 0.99)))

    def _props(self, v, s):
        return np.bincount(np.searchsorted(self.edges[s], v), minlength=len(self.edges[s]) + 1) / len(v)

    def _assign(self, Z):
        d = ((Z[:, None, :] - self.C[None]) ** 2).sum(-1)
        return d.argmin(1), np.sqrt(d.min(1))

    def components(self, df):
        out = {}
        for s in SENSORS:
            v = df[s].values
            out[f"ks:{s}"] = ks_stat(self.ref_sorted[s], v)
            out[f"psi:{s}"] = psi_from_props(self._props(v, s), self.ref_props[s])
        Z = (df[SENSORS].values - self.mu) / self.sd
        lab, qe = self._assign(Z)
        out["km_occ"] = psi_from_props(np.bincount(lab, minlength=self.k) / len(lab), self.km_props)
        out["km_qe"] = max(0.0, qe.mean() / self.km_qe - 1)
        return out

    def normalise(self, comps):
        return {c: comps[c] / self.thr[c] for c in comps}

    @staticmethod
    def _composite(norm):
        return max(norm.values())

    def score(self, df):
        norm = self.normalise(self.components(df))
        per_sensor = {s: max(norm[f"ks:{s}"], norm[f"psi:{s}"]) for s in SENSORS}
        return dict(norm=norm, per_sensor=per_sensor, composite=self._composite(norm),
                    drifting=[s for s, v in per_sensor.items() if v > 1.0],
                    multivar=max(norm["km_occ"], norm["km_qe"]) > 1.0)
