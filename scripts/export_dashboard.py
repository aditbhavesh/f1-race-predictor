"""Freeze the forecast for the next race and export everything the dashboard needs as one JSON file.
Dynamically detects the next upcoming race on the F1 calendar and fits on all races to date."""
import sys, json, pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
from data import load_clean
from features import build_features, PRE
from models import RaceModel
from backtest import per_race_metrics
from f1_calendar import get_next_event

c = load_clean(str(ROOT / "data/results_2022_2026.csv"))
last = c[c.race_idx == c.race_idx.max()].copy()
nxt = last.copy()

# Dynamically resolve next event from calendar
next_ev = get_next_event()
if next_ev:
    nxt["Season"] = next_ev["season"]
    nxt["Round"] = next_ev["round"]
    nxt["race_id"] = f"{next_ev['season']}-{next_ev['round']:02d}"
    nxt["EventName"] = next_ev["name"]
    nxt["Location"] = next_ev.get("place", "")
else:
    nxt["Round"] = int(last.Round.iat[0]) + 1
    nxt["race_id"] = f"{int(last.Season.iat[0])}-{nxt.Round.iat[0]:02d}"
    nxt["EventName"] = "United States Grand Prix"
    next_ev = {"name": "United States Grand Prix", "place": "Austin", "date": "2026-10-25", "season": 2026, "round": int(nxt.Round.iat[0])}

nxt["race_idx"] = c.race_idx.max() + 1
print(f"Generating predictions for next event: {nxt['EventName'].iat[0]} (Round {nxt['Round'].iat[0]}, {nxt['Season'].iat[0]})")

f_all = build_features(pd.concat([c, nxt], ignore_index=True))
train = f_all[f_all.race_idx <= c.race_idx.max()]
up = f_all[f_all.race_idx == nxt.race_idx.iat[0]].reset_index(drop=True)
n = len(up)

# ---------- before qualifying
print("Fitting pre-qualifying model & running 20,000 Monte Carlo sims...")
pre = RaceModel("pre").fit(train)
sc, p = pre.predict(up)
sim = pre.simulate(sc, p, 20000, 7)
drivers = []
for i, r in up.iterrows():
    drivers.append({
        "abbr": r.Abbreviation, "name": r.FullName, "team": r.entry, "teamName": r.TeamName,
        "p_win": round(float(sim["p_win"][i]), 4), "p_podium": round(float(sim["p_podium"][i]), 4),
        "p_top10": round(float(sim["p_top10"][i]), 4), "p_dnf": round(float(sim["p_dnf"][i]), 4),
        "exp_pos": round(float(sim["exp_pos"][i]), 2), "pos_dist": [round(float(x), 4) for x in sim["pos_counts"][i]]
    })

# ---------- after qualifying: what the model says for every driver in every grid slot
print("Fitting post-qualifying model for interactive grid editor...")
post = RaceModel("post").fit(train)
tr = train.dropna(subset=["q1_gap"])
q1_by_slot = [float(tr[(tr.grid >= g - 1) & (tr.grid <= g + 1)].q1_gap.median()) for g in range(1, n + 1)]
slot_score = np.zeros((n, n))
slot_dnf = np.zeros((n, n))
for g in range(1, n + 1):
    t = up.copy()
    t["grid_pct"] = (g - 1) / (n - 1)
    t["grid_pen"] = 0.0
    t["q1_gap"] = q1_by_slot[g - 1]
    s, d = post.predict(t)
    slot_score[:, g - 1], slot_dnf[:, g - 1] = s, d

def compress(pool, k=200):
    return [round(float(x), 5) for x in np.quantile(pool, (np.arange(k) + 0.5) / k)]

pools = [compress(x) for x in post.pools]
edges = [round(float(x), 5) for x in post.edges]

# ---------- backtest browser (with seamless fallback)
final_runs_path = ROOT / "outputs/final_runs.pkl"
dashboard_json_path = ROOT / "outputs/dashboard_data.json"
races = []

if final_runs_path.exists():
    runs, base = pickle.load(open(final_runs_path, "rb"))
    pre_b, post_b = runs["pre_gbm"], runs["post_gbm"]
    for rid, g in post_b.groupby("race_idx", sort=True):
        gp = pre_b[pre_b.race_idx == rid].set_index("Abbreviation")
        rows = []
        for _, r in g.sort_values("grid").iterrows():
            q = gp.loc[r.Abbreviation]
            rows.append({
                "abbr": r.Abbreviation, "team": r.TeamName, "grid": int(r.grid),
                "pos": None if not r.finished else int(r.pos),
                "pre": [round(float(q.p_win), 4), round(float(q.p_podium), 4), round(float(q.p_top10), 4)],
                "post": [round(float(r.p_win), 4), round(float(r.p_podium), 4), round(float(r.p_top10), 4), round(float(r.p_dnf), 4)]
            })
        races.append({
            "id": g.race_id.iat[0], "season": int(g.Season.iat[0]), "round": int(g.Round.iat[0]),
            "event": g.EventName.iat[0], "rows": rows
        })
elif dashboard_json_path.exists():
    try:
        prev_data = json.load(open(dashboard_json_path))
        races = prev_data.get("backtest", [])
        print(f"Preserving existing historical backtest ({len(races)} races).")
    except Exception:
        races = []

out = {
    "next_race": {
        "name": next_ev["name"],
        "place": next_ev.get("place", "Circuit"),
        "date": next_ev.get("date", str(pd.Timestamp.now())[:10]),
        "season": int(next_ev["season"]),
        "round": int(next_ev["round"])
    },
    "trained_through": {
        "race": last.EventName.iat[0],
        "date": str(last.EventDate.iat[0])[:10],
        "n_races": int(train.race_id.nunique())
    },
    "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
    "n": n,
    "drivers": drivers,
    "slot_score": np.round(slot_score, 5).tolist(),
    "slot_dnf": np.round(slot_dnf, 5).tolist(),
    "edges": edges,
    "pools": pools,
    "backtest": races
}

json.dump(out, open(dashboard_json_path, "w"), separators=(",", ":"))
pickle.dump({
    "up": up, "slot_score": slot_score, "slot_dnf": slot_dnf,
    "edges": np.array(edges), "pools": [np.array(x) for x in pools]
}, open(ROOT / "outputs/frozen.pkl", "wb"))

print(f"Exported {dashboard_json_path.stat().st_size // 1024} KB -> outputs/dashboard_data.json")
df_summary = pd.DataFrame(drivers)[["abbr", "team", "p_win", "p_podium", "p_top10", "p_dnf", "exp_pos"]].sort_values("p_win", ascending=False)
print("\nTop 5 Forecast:")
print(df_summary.head(5).to_string(index=False))
print(f"Probability sums: Win={sum(d['p_win'] for d in drivers):.3f}, Podium={sum(d['p_podium'] for d in drivers):.3f}, Top10={sum(d['p_top10'] for d in drivers):.3f}")
