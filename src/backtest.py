"""Walk-forward backtest: before every test race the model is refitted on the races before it, and nothing else.

Baselines are deliberately simple, because a model is only worth something if it beats them:
  grid slot    after qualifying: the chance of winning / podium / points given only the starting slot,
               measured on the training races (smoothed over neighbouring slots)
  standings    before qualifying: the same, given only the championship standing before the race
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models import RaceModel


def walk_forward(feat, start_idx, stage, kind="gbm", hp=None, dnf_features=True, feats=None, dnf_feats=None,
                 n_sims=4000, end_idx=None, refit_every=1, seed=1000):
    rows, model = [], None
    race_idxs = sorted(feat["race_idx"].unique())
    for k, r in enumerate([x for x in race_idxs if x >= start_idx and (end_idx is None or x <= end_idx)]):
        if model is None or k % refit_every == 0:
            model = RaceModel(stage, kind, hp, dnf_features, feats, dnf_feats).fit(feat[feat["race_idx"] < r])
        test = feat[feat["race_idx"] == r]
        score, p = model.predict(test)
        sim = model.simulate(score, p, n_sims, seed + r)
        t = test[["race_idx", "race_id", "Season", "Round", "EventName", "Abbreviation", "FullName", "TeamName", "grid",
                  "finished", "pos", "won", "podium", "top10", "dnf"]].copy()
        t["score"], t["p_dnf_model"] = score, p
        for key in ("p_win", "p_podium", "p_top10", "p_dnf", "exp_pos"):
            t[key] = sim[key]
        t["pos_counts"] = list(sim["pos_counts"])
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _frequency_baseline(train, test, key, smooth=1):
    """P(win / podium / top-10) as a function of an integer rank column, from the training races only."""
    tr = train.dropna(subset=[key])
    cap = int(tr[key].quantile(0.99))
    kk = tr[key].clip(upper=cap).astype(int)
    out = {}
    for tgt in ("won", "podium", "top10"):
        tab = {}
        for g in range(1, cap + 1):
            m = (kk >= g - smooth) & (kk <= g + smooth)
            tab[g] = (tr.loc[m, tgt].sum() + 0.1) / (m.sum() + 2.0)
        s = test[key].clip(upper=cap).astype(int).map(tab).to_numpy(float)
        out[tgt] = s
    return out


def baseline_walk_forward(feat, start_idx, key, end_idx=None):
    rows = []
    for r in sorted(feat["race_idx"].unique()):
        if r < start_idx or (end_idx is not None and r > end_idx):
            continue
        train, test = feat[feat["race_idx"] < r], feat[feat["race_idx"] == r]
        f = _frequency_baseline(train, test, key)
        t = test[["race_idx", "race_id", "Season", "Round", "Abbreviation", "finished", "pos", "won", "podium", "top10", "dnf"]].copy()
        n = len(t)
        # make the odds add up to 1 winner, 3 podium places and 10 points places per race
        t["p_win"] = f["won"] / f["won"].sum() * 1.0
        t["p_podium"] = f["podium"] / f["podium"].sum() * 3.0
        t["p_top10"] = np.clip(f["top10"] / f["top10"].sum() * 10.0, 0, 1)
        t["order"] = test[key].to_numpy()
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def uniform_baseline(feat, start_idx, end_idx=None):
    t = feat[(feat["race_idx"] >= start_idx) & ((feat["race_idx"] <= end_idx) if end_idx is not None else True)]
    t = t[["race_idx", "race_id", "Season", "Round", "Abbreviation", "finished", "pos", "won", "podium", "top10", "dnf"]].copy()
    n = t.groupby("race_idx")["Abbreviation"].transform("size")
    t["p_win"], t["p_podium"], t["p_top10"] = 1.0 / n, 3.0 / n, 10.0 / n
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------- metrics
def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def per_race_metrics(df, order_col=None):
    """One row per race. order_col: smaller = predicted to finish earlier (defaults to the model's expected position)."""
    rows = []
    for rid, g in df.groupby("race_idx", sort=True):
        order = g[order_col] if order_col is not None else g["exp_pos"].fillna(99)
        fin = g[g["finished"]]
        row = {"race_idx": rid, "race_id": g["race_id"].iat[0], "Season": g["Season"].iat[0],
               "brier_win": _brier(g["p_win"], g["won"]), "brier_podium": _brier(g["p_podium"].clip(0, 1) if "p_podium" in g else 0, g["podium"]),
               "brier_top10": _brier(g["p_top10"].clip(0, 1), g["top10"]),
               "logp_winner": float(np.log(max(g.loc[g["won"], "p_win"].sum(), 1e-4))) if g["won"].any() else np.nan}
        pick = g.loc[g["p_win"].idxmax()]
        row["winner_picked"] = float(bool(pick["won"]))
        top3 = g.nlargest(3, "p_podium")
        row["podium_hits"] = float(top3["podium"].sum())
        ten = g.nlargest(10, "p_top10")
        row["top10_hits"] = float(ten["top10"].sum())
        if len(fin) >= 5:
            o = (order.loc[fin.index] if hasattr(order, "loc") else fin.index.map(order))
            row["spearman"] = float(spearmanr(o, fin["pos"]).correlation)
            row["mae_pos"] = float(np.abs(fin["pos"] - np.where(np.isnan(o), np.nan, o)).mean()) if order_col is not None else float(np.abs(fin["pos"] - g.loc[fin.index, "exp_pos"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values, n=2000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def bootstrap_diff(a, b, n=2000, seed=0):
    """Mean of (a - b) over paired races with a 90% bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return bootstrap_mean(d, n, seed)
