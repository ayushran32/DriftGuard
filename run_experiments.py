"""Benchmark: Static vs Naive drift->retrain vs DriftGuard on every scenario."""
import sys, json, numpy as np, pandas as pd
from driftguard.data import DataSource
from driftguard.simulate import SCENARIOS, build_stream
from driftguard.engine import STRATEGIES, run_strategy

ONSET, N_WIN, WINDOW = 12, 40, 500

def main(seeds=5, csv=None):
    rows = []
    for seed in range(seeds):
        src = DataSource(csv, seed)
        for name, sc in SCENARIOS.items():
            stream = build_stream(src, name, N_WIN, WINDOW, ONSET, seed)
            for sname, cls in STRATEGIES.items():
                s, reps, f = run_strategy(cls, src, stream, WINDOW, 2, seed)
                alerts = [r["alert"] for r in reps]
                pre = ONSET if name != "control" else N_WIN
                first = next((t for t in range(ONSET, N_WIN) if alerts[t]), None)
                post = [r for r in reps[ONSET:] if r["alert"]]
                ok = np.mean([r["cause"] == sc.expected_cause for r in post]) if (post and sname == "DriftGuard") else np.nan
                rows.append(dict(seed=seed, scenario=name, strategy=sname,
                                 f1_post=f[ONSET:].mean() if name != "control" else f.mean(),
                                 f1_last10=f[-10:].mean(),
                                 false_alarm_windows=sum(alerts[:pre]),
                                 delay=(first - ONSET) if (first is not None and name != "control") else np.nan,
                                 detected=first is not None and name != "control",
                                 deployed=getattr(s, "version", 0), cause_ok=ok))
        print(f"seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results_raw.csv", index=False)
    summ = df.groupby(["scenario", "strategy"]).agg(
        F1_after_onset=("f1_post", "mean"), F1_last10=("f1_last10", "mean"),
        false_alarm_windows=("false_alarm_windows", "mean"), detect_rate=("detected", "mean"),
        delay_windows=("delay", "mean"), models_deployed=("deployed", "mean"),
        triage_correct=("cause_ok", "mean")).round(3)
    summ.to_csv("results_summary.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print(summ)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, sys.argv[2] if len(sys.argv) > 2 else None)
