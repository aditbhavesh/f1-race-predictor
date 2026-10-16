"""Scores, paired comparisons with bootstrap ranges, and calibration bins for the dashboard's trust tab."""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from backtest import per_race_metrics, bootstrap_diff, bootstrap_mean

runs, base = pickle.load(open("outputs/final_runs.pkl", "rb"))
M = {k: per_race_metrics(v) for k, v in runs.items()}
M["grid_freq"] = per_race_metrics(base["grid_freq"], "order")
M["standings_freq"] = per_race_metrics(base["standings_freq"], "order")
M["uniform"] = per_race_metrics(base["uniform"].assign(exp_pos=11.0))
cols = ["brier_win", "brier_podium", "brier_top10", "winner_picked", "podium_hits", "top10_hits", "spearman"]
label = {"pre_gbm": "Model before qualifying", "post_gbm": "Model after qualifying", "pre_ridge": "Simple linear model, before", "post_ridge": "Simple linear model, after",
         "standings_freq": "Baseline: championship order", "grid_freq": "Baseline: starting slot", "uniform": "Baseline: everyone equal"}
scores = []
for k in ("pre_gbm", "standings_freq", "post_gbm", "grid_freq", "pre_ridge", "post_ridge", "uniform"):
    row = {"key": k, "label": label[k]}
    for c in cols:
        row[c] = None if k == "uniform" and c in ("winner_picked", "podium_hits", "top10_hits", "spearman") else (None if np.isnan(M[k][c].mean()) else round(float(M[k][c].mean()), 4))
    scores.append(row)

def diff(a, b):
    r = {"a": a, "b": b}
    for c in ("brier_win", "brier_podium", "brier_top10"):
        m, lo, hi = bootstrap_diff(M[a][c], M[b][c], 4000, 1)
        r[c] = [round(m, 4), round(lo, 4), round(hi, 4)]
    return r
pairs = [("pre_gbm", "standings_freq", "Before qualifying: model vs championship order"), ("post_gbm", "grid_freq", "After qualifying: model vs starting slot"),
         ("post_gbm", "post_gbm_gridonly", "Qualifying pace and penalties vs starting slot alone"), ("post_gbm", "post_gbm_constDNF", "Reliability model vs one flat retirement rate"),
         ("post_gbm", "post_ridge", "Gradient boosting vs simple linear model")]
diffs = []
for a, b, t in pairs:
    d = diff(a, b); d["title"] = t; diffs.append(d)

bins = [0, .02, .05, .1, .2, .4, .7, 1.0001]
cal = {}
for k in ("pre_gbm", "post_gbm"):
    df = runs[k]; cal[k] = {}
    for col, tg in (("p_win", "won"), ("p_podium", "podium"), ("p_top10", "top10")):
        g = df.groupby(pd.cut(df[col], bins, include_lowest=True), observed=True).agg(p=(col, "mean"), obs=(tg, "mean"), n=(tg, "size"))
        cal[k][tg] = [[round(float(r.p), 4), round(float(r.obs), 4), int(r.n)] for r in g.itertuples()]
n_races = int(len(M["post_gbm"]))
by_season = {}
for s in (2024, 2025, 2026):
    by_season[s] = {k: {"brier_podium": round(float(M[k][M[k].Season == s].brier_podium.mean()), 4), "top10_hits": round(float(M[k][M[k].Season == s].top10_hits.mean()), 2),
                        "n": int((M[k].Season == s).sum())} for k in ("pre_gbm", "post_gbm", "standings_freq", "grid_freq")}
out = {"n_races": n_races, "n_driver_races": int(len(runs["post_gbm"])), "scores": scores, "diffs": diffs, "calibration": cal, "by_season": by_season,
       "dnf": {"pred": round(float(runs["post_gbm"].p_dnf.mean()), 4), "obs": round(float(runs["post_gbm"].dnf.mean()), 4)}}
json.dump(out, open("outputs/trust.json", "w"), separators=(",", ":"))
print(json.dumps(out["diffs"], indent=0)[:900]); print(n_races, out["n_driver_races"], out["dnf"])
