import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from data import load_clean
from features import build_features
from models import RaceModel
from backtest import *
c = load_clean("data/results_2022_2026.csv"); f = build_features(c)
f.to_pickle("outputs/feat.pkl")
s = f.groupby("Season")["race_idx"].agg(["min","max"]); print(s)
last = f["race_idx"].max()
m = RaceModel("post").fit(f[f.race_idx < last]); t = f[f.race_idx == last]
sc, p = m.predict(t); sim = m.simulate(sc, p, 4000, 1)
print("sums win/pod/top10:", sim["p_win"].sum(), sim["p_podium"].sum(), sim["p_top10"].sum(), "oof_mae", round(m.oof_mae,3))
print(t[["Abbreviation"]].assign(win=sim["p_win"].round(3), pod=sim["p_podium"].round(2), dnf=sim["p_dnf"].round(2)).sort_values("win", ascending=False).head(6))
