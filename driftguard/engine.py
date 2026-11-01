"""DriftGuard engine + two baselines (static model, naive drift->retrain)."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .data import SENSORS, train_model, predict
from .detect import Detector
from .triage import Triage


def f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0


def retrain(source, dfs, ys, ref_weight=0.25, seed=0):
    if ref_weight == 0:                      # recent-only recipe
        return train_model(pd.concat(list(dfs), ignore_index=True), np.concatenate(list(ys)), None, seed)
    df = pd.concat([source.train] + list(dfs), ignore_index=True)
    y = np.concatenate([source.train_y] + list(ys))
    w = np.concatenate([np.full(len(source.train), ref_weight)] + [np.ones(len(d)) for d in dfs])
    return train_model(df, y, w, seed)


def paired_better(y, p_old, p_new, rng, n=200, margin=0.01, conf=0.9):
    m, wins = len(y), 0
    for _ in range(n):
        i = rng.integers(0, m, m)
        wins += (f1(y[i], p_new[i]) - f1(y[i], p_old[i])) > margin
    return wins / n >= conf


def cached_detector(source, window, seed):
    cache = source.__dict__.setdefault("_det_cache", {})
    if (window, seed) not in cache:
        cache[(window, seed)] = Detector(source.train, source.cal, window, seed)
    return cache[(window, seed)]


class Static:
    name = "Static model"
    def __init__(self, source, window=500, lag=2, seed=0):
        self.model, self.log = source.base_model, []
    def process(self, t, obs):
        return dict(t=t, pred=predict(self.model, obs), alert=False, cause="NONE")
    def add_labels(self, u, y, now):
        pass


class NaiveRetrainer:
    """Typical pipeline: KS p<0.05 on ANY feature -> retrain on recent data -> deploy blindly."""
    name = "Naive drift->retrain"
    ref_weight = 0.25
    def __init__(self, source, window=500, lag=2, seed=0):
        self.src, self.seed, self.model = source, seed, source.base_model
        self.ref, self.pending, self.buf, self.log, self.version = source.train, None, {}, [], 0
    def _alert(self, obs):
        return any(ks_2samp(self.ref[s], obs[s]).pvalue < 0.05 for s in SENSORS)

    def process(self, t, obs):
        pred = predict(self.model, obs)
        alert = self._alert(obs)
        if alert and self.pending is None:
            self.pending = dict(start=t)
        self.buf[t] = dict(obs=obs, y=None)
        return dict(t=t, pred=pred, alert=alert, cause="DRIFT" if alert else "NONE")
    def add_labels(self, u, y, now):
        self.buf[u]["y"] = y
        if self.pending and u - self.pending["start"] + 1 >= 3:
            idx = range(self.pending["start"], u + 1)
            dfs = [self.buf[v]["obs"] for v in idx]
            self.model = retrain(self.src, dfs, [self.buf[v]["y"] for v in idx], self.ref_weight, self.seed)
            self.ref = pd.concat(dfs, ignore_index=True)
            self.version += 1
            self.log.append((now, "RETRAIN", "retrained & deployed without validation"))
            self.pending = None


class NaiveRecentOnly(NaiveRetrainer):
    """Same naive trigger, but retrains on the latest windows only (sliding-window practice)."""
    name = "Naive (recent-only)"
    ref_weight = 0.0


class VoteRetrainer(NaiveRetrainer):
    """Dataset-level drift vote (common monitoring default): drift if >=50% of features fail a KS test."""
    name = "Feature-vote retrain"
    def _alert(self, obs):
        return sum(ks_2samp(self.ref[s], obs[s]).pvalue < 0.05 for s in SENSORS) >= 3


class DriftGuard:
    name = "DriftGuard"
    RECIPES = {"recent-only": 0.0, "blend-with-history": 0.25}
    LABEL_Q = 0.005    # lower-tail quantile of the no-drift F1 band; smaller = fewer label false alarms

    def __init__(self, source, window=500, lag=2, seed=0):
        self.src, self.W, self.lag, self.seed = source, window, lag, seed
        self.rng = np.random.default_rng(seed + 7)
        self.model = source.base_model
        self.det = cached_detector(source, window, seed)
        self.tri = Triage(source.train)
        self.buf, self.log, self.version = {}, [], 0
        self.pending, self.ack_until, self.cool_until, self.last_cause = None, -1, -1, "NONE"
        self.weights = self._importance()
        self._init_label_monitor()

    # ---------- setup ----------
    def _importance(self):
        pool, y = self.src.pool.iloc[:2500], self.src.pool_y[:2500]
        base, imp = f1(y, predict(self.model, pool)), {}
        for s in SENSORS:
            drops = []
            for _ in range(3):
                d = pool.copy(); d[s] = self.rng.permutation(d[s].values)
                drops.append(base - f1(y, predict(self.model, d)))
            imp[s] = max(float(np.mean(drops)), 0.0)
        tot = sum(imp.values()) or 1.0
        return {s: v / tot for s, v in imp.items()}

    def _init_label_monitor(self):
        pred, y, n = predict(self.model, self.src.pool), self.src.pool_y, len(self.src.pool)
        vals = []
        for _ in range(300):
            i = self.rng.integers(0, n, 3 * self.W)
            vals.append(f1(y[i], pred[i]))
        self.base_f1, self.lab_thr = float(np.mean(vals)), float(np.quantile(vals, self.LABEL_Q))
        self.margin, self.strikes, self.mon_start, self.roll_f1 = self.base_f1 - self.lab_thr, 0, 0, None

    # ---------- per-window (no labels needed) ----------
    def process(self, t, obs):
        det = self.det.score(obs)
        faults = self.tri.detect_faults(obs)
        raw = det["composite"] > self.det.comp_thr or bool(faults)
        prev = self.buf.get(t - 1)
        alert = bool(faults) or (raw and prev is not None and prev["report"]["raw"])
        cause = self.tri.classify(det["drifting"], faults, det["multivar"]) if raw else "NONE"
        if not alert and cause != "SENSOR_FAULT":
            cause = "NONE" if not raw else cause
        if self.strikes >= 2 and cause == "NONE":
            cause, alert = "CONCEPT_DRIFT", True      # inputs look normal but labels say model is wrong
        rep = self.tri.repair(obs, faults) if faults else obs
        pred = predict(self.model, rep)
        impact = float(sum(self.weights[s] * min(det["per_sensor"][s], 3) / 3 for s in SENSORS))
        report = dict(t=t, raw=raw, alert=alert, cause=cause, composite=det["composite"],
                      thr=self.det.comp_thr, per_sensor=det["per_sensor"], faults=faults,
                      impact=impact, pred=pred, version=self.version, roll_f1=self.roll_f1,
                      sig=self.tri.signatures(obs), action=self._action(cause, faults))
        self.buf[t] = dict(rep=rep, pred=pred, y=None, report=report)
        if alert and cause == "REGIME_SHIFT" and self.pending is None \
                and t > self.ack_until and t > self.cool_until:
            self.pending = dict(start=t - 1, attempts=0, reason="covariate")
        if (cause, alert) != (self.last_cause, True) and alert:
            self.log.append((t, "ALERT", f"{cause}: {self._explain(report)}"))
        elif not alert and self.last_cause != "NONE":
            self.log.append((t, "CLEAR", "no active drift"))
        self.last_cause = cause if alert else "NONE"
        return report

    def _explain(self, r):
        if r["cause"] == "SENSOR_FAULT":
            return "; ".join(f"{s} sensor {k}" for s, k in r["faults"]) + \
                " (machine unchanged; input auto-repaired from correlated sensors)"
        top = sorted(r["per_sensor"].items(), key=lambda kv: -kv[1])[:2]
        if r["cause"] == "CONCEPT_DRIFT":
            return "inputs look normal but failure labels disagree with the model (P(y|x) changed)"
        return "drift mainly in " + ", ".join(f"{s} ({v:.1f}x limit)" for s, v in top)

    @staticmethod
    def _action(cause, faults):
        return {"NONE": "No action",
                "SENSOR_FAULT": "Inspect/recalibrate sensor; input repaired online; do NOT retrain",
                "FLEET_AGING": "Informational: tool-wear distribution aging; no retrain",
                "REGIME_SHIFT": "Wait for delayed labels, then train challenger; deploy only if validated better",
                "CONCEPT_DRIFT": "Retrain challenger from delayed labels; deploy only if validated better"}[cause]

    # ---------- delayed labels ----------
    def add_labels(self, u, y, now):
        self.buf[u]["y"] = y
        if u - 2 >= self.mon_start and all(self.buf[v]["y"] is not None for v in (u - 2, u - 1, u)):
            ys = np.concatenate([self.buf[v]["y"] for v in (u - 2, u - 1, u)])
            ps = np.concatenate([self.buf[v]["pred"] for v in (u - 2, u - 1, u)])
            self.roll_f1 = f1(ys, ps)
            self.strikes = self.strikes + 1 if self.roll_f1 < self.lab_thr else 0
            if self.strikes == 2:
                self.log.append((now, "LABEL_ALARM", f"rolling F1 {self.roll_f1:.2f} below band {self.lab_thr:.2f}"))
        if self.strikes >= 2 and self.pending is None and now > self.cool_until:
            self.pending = dict(start=max(u - 2, self.mon_start), attempts=0, reason="labels")
        if self.pending:
            self._try_retrain(u, now)

    def _try_retrain(self, u, now):
        p = self.pending
        idx = list(range(p["start"], u + 1))
        if len(idx) < 4 or any(self.buf[v]["y"] is None for v in idx):
            return
        ys = np.concatenate([self.buf[v]["y"] for v in idx])
        ps = np.concatenate([self.buf[v]["pred"] for v in idx])
        pooled = f1(ys, ps)
        if p["reason"] == "covariate" and self.strikes < 2 and pooled >= self.lab_thr:
            self.log.append((now, "HARMLESS", f"drift confirmed harmless: F1 {pooled:.2f} still in normal band; no retrain"))
            self.pending, self.ack_until = None, now + 12
            return
        tr, va = idx[:-2], idx[-2:]
        Xv = pd.concat([self.buf[v]["rep"] for v in va], ignore_index=True)
        yv = np.concatenate([self.buf[v]["y"] for v in va])
        chals = {k: retrain(self.src, [self.buf[v]["rep"] for v in tr], [self.buf[v]["y"] for v in tr], w, self.seed)
                 for k, w in self.RECIPES.items()}
        best = max(chals, key=lambda k: f1(yv, predict(chals[k], Xv)))
        pc, pn = predict(self.model, Xv), predict(chals[best], Xv)
        if paired_better(yv, pc, pn, self.rng):
            self.model = retrain(self.src, [self.buf[v]["rep"] for v in idx], [self.buf[v]["y"] for v in idx],
                                 self.RECIPES[best], self.seed)      # validated recipe, refit on all labeled data
            self.version += 1
            self.log.append((now, "DEPLOY", f"challenger '{best}' validated (F1 {f1(yv, pc):.2f} -> {f1(yv, pn):.2f}); deployed v{self.version}"))
            self.mon_start, self.strikes, self.cool_until = now + 1, 0, now + 3
            self.base_f1 = f1(yv, pn); self.lab_thr = self.base_f1 - self.margin
            if p["reason"] == "covariate":
                self._rebaseline(idx, now)
            self.pending = None
        else:
            p["attempts"] += 1
            self.log.append((now, "REJECT", f"challenger not better than champion (F1 {f1(yv, pc):.2f} vs {f1(yv, pn):.2f}); kept current model"))
            if p["attempts"] >= 2:
                self.pending, self.ack_until = None, now + 12

    def _rebaseline(self, idx, now):
        ref = pd.concat([self.buf[v]["rep"] for v in idx[0::2]], ignore_index=True)
        cal = pd.concat([self.buf[v]["rep"] for v in idx[1::2]], ignore_index=True)
        self.det = Detector(ref, cal, self.W, self.seed + 1, n_null=300)
        self.tri = Triage(ref)
        self.log.append((now, "REBASELINE", "detectors recalibrated to the new normal"))


STRATEGIES = {"Static model": Static, "Naive drift->retrain": NaiveRetrainer,
              "Naive (recent-only)": NaiveRecentOnly, "Feature-vote retrain": VoteRetrainer,
              "DriftGuard": DriftGuard}


def run_strategy(cls, source, stream, window=500, lag=2, seed=0):
    s = cls(source, window, lag, seed)
    reports, f1s = [], []
    for t, w in enumerate(stream):
        r = s.process(t, w["obs"]); reports.append(r); f1s.append(f1(w["y"], r["pred"]))
        if t - lag >= 0:
            s.add_labels(t - lag, stream[t - lag]["y"], t)
    return s, reports, np.array(f1s)
