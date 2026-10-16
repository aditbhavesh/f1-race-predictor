import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from backtest import *
from features import standings_baseline
f = pd.read_pickle("outputs/feat.pkl")
A, B = 22, 43
def summ(name, df, order=None):
    m = per_race_metrics(df, order)
    print(f"{name:28s} brierW {m.brier_win.mean():.4f} brierP {m.brier_podium.mean():.4f} brierT10 {m.brier_top10.mean():.4f} "
          f"win@1 {m.winner_picked.mean():.2f} pod3 {m.podium_hits.mean():.2f} top10 {m.top10_hits.mean():.2f} rho {m.spearman.mean():.3f}")
    return m
t=time.time()
u = uniform_baseline(f, A, B); u["exp_pos"]=11.0; summ("uniform", u)
g = baseline_walk_forward(f, A, "grid", B); summ("grid-slot freq", g, "order")
f2 = f.copy(); f2["stand"] = standings_baseline(f)
s = baseline_walk_forward(f2, A, "stand", B); summ("standings freq", s, "order")
for stage in ("pre","post"):
    for kind in ("gbm","ridge"):
        for dnf in (True, False):
            d = walk_forward(f, A, stage, kind, dnf_features=dnf, end_idx=B, n_sims=2000)
            summ(f"{stage} {kind} dnf={dnf}", d)
print("secs", round(time.time()-t))
