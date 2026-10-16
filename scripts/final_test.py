"""Final walk-forward test, 2024 onward. Hyperparameters were fixed on 2023 only (see scripts/tune.py)."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, numpy as np
from backtest import *
from data import load_clean
from features import build_features, standings_baseline, POST, PRE
f = pd.read_pickle("outputs/feat.pkl")
f_keep = build_features(load_clean("data/results_2022_2026.csv"), reset_seasons=())  # last-season carry-over kept in 2026 too
A, B = 44, int(f.race_idx.max())
runs = {}


def go(name, **kw):
    ff = kw.pop("feat", f)
    runs[name] = walk_forward(ff, A, kw.pop("stage"), end_idx=B, n_sims=3000, refit_every=1, **kw)
    print(name, flush=True)


go("pre_gbm", stage="pre")
go("post_gbm", stage="post")
go("pre_ridge", stage="pre", kind="ridge")
go("post_ridge", stage="post", kind="ridge")
go("post_gbm_constDNF", stage="post", dnf_features=False)
go("post_gbm_noq1pen", stage="post", feats=[x for x in POST if x not in ("q1_gap", "grid_pen")])
go("post_gbm_gridonly", stage="post", feats=PRE + ["grid_pct"])
go("pre_gbm_noprev", stage="pre", feats=[x for x in PRE if "prev" not in x])
go("pre_gbm_keepcarry2026", stage="pre", feat=f_keep)
go("post_gbm_keepcarry2026", stage="post", feat=f_keep)
f2 = f.copy()
f2["stand"] = standings_baseline(f)
base = {"uniform": uniform_baseline(f, A, B), "grid_freq": baseline_walk_forward(f, A, "grid", B),
        "standings_freq": baseline_walk_forward(f2, A, "stand", B)}
pickle.dump((runs, base), open("outputs/final_runs.pkl", "wb"))
print("done")
