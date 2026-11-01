"""Terminal demo (no browser needed):  python demo_cli.py --scenario process_temp_offset
Runs Static vs Naive-retrain vs DriftGuard on one scenario, prints a summary + DriftGuard's incident log,
and saves a plot to outputs/demo_<scenario>.png."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from driftguard.data import DataSource
from driftguard.simulate import SCENARIOS, build_stream
from driftguard.engine import Static, NaiveRetrainer, DriftGuard, run_strategy

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="process_temp_offset", choices=list(SCENARIOS))
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--onset", type=int, default=12)
a = ap.parse_args()

root = Path(__file__).resolve().parent
csv = root / "data" / "ai4i2020.csv"
src = DataSource(csv if csv.exists() else None, a.seed)
print(f"data: {'real AI4I csv' if src.real else 'synthetic AI4I-like'} | scenario: {a.scenario} -> {SCENARIOS[a.scenario].description}")
print(f"ground-truth cause: {SCENARIOS[a.scenario].expected_cause}\n")
stream = build_stream(src, a.scenario, 40, 500, a.onset, a.seed)
res = {}
for cls in (Static, NaiveRetrainer, DriftGuard):
    s, reps, f = run_strategy(cls, src, stream, 500, 2, a.seed)
    res[cls.name] = (s, reps, f)
    print(f"{cls.name:24s} mean F1 after onset = {f[a.onset:].mean():.3f} | alert windows = {sum(r['alert'] for r in reps):2d}"
          f" | models deployed = {getattr(s, 'version', 0)}")
print("\nDriftGuard incident log:")
for w, kind, msg in res["DriftGuard"][0].log:
    print(f"  window {w:2d}  {kind:12s} {msg}")
causes = {r["cause"] for r in res["DriftGuard"][1][a.onset:] if r["alert"]}
print("\ncauses diagnosed after onset:", causes or "none")

out = root / "outputs"; out.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k, (_, reps, f) in res.items(): ax[0].plot(f, label=k, lw=2 if k == "DriftGuard" else 1.2)
ax[0].axvline(a.onset, color="red", ls="--"); ax[0].set_title("F1 per window"); ax[0].legend(fontsize=7)
dg = res["DriftGuard"][1]
ax[1].plot([r["composite"] for r in dg], label="drift score"); ax[1].plot([r["thr"] for r in dg], "k--", label="threshold")
ax[1].axvline(a.onset, color="red", ls="--"); ax[1].set_yscale("log"); ax[1].set_title("DriftGuard drift score"); ax[1].legend(fontsize=7)
fig.tight_layout(); p = out / f"demo_{a.scenario}.png"; fig.savefig(p, dpi=150); print("plot saved:", p)
