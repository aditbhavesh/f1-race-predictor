"""Models that turn features into odds.

Three pieces, fitted only on races before the one being predicted:

 1. Score model   predicts where a driver finishes IF he finishes, as percent of the field (0 = win, 1 = last).
                  Gradient boosting with monotonic constraints: better form or a better grid slot can never
                  predict a worse finish. Ridge regression is kept as a simple comparison.
 2. Finish model  logistic regression for the chance of NOT finishing (crash, failure, disqualification,
                  did not start), from team / driver / whole-field reliability.
 3. Noise         the score model is wrong by a typical amount that depends on where the driver is predicted to
                  finish. We measure that error out-of-sample (5-fold, grouped by race) on the training races and
                  keep the list of errors per predicted-position band.

Odds come from a Monte Carlo: each simulated race draws a not-finish flag for every driver, adds a random
error (picked from the matching band) to each finisher's score, and ranks the finishers. Non-finishers are placed
after all finishers. Counting how often each driver wins / podiums / scores gives the probabilities, and the
full table of finishing positions comes for free.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import DNF_POST, DNF_PRE, POST, PRE

# +1: a higher value should never predict a better (smaller) finishing percentage
MONOTONE = {"team_q_ew3": 1, "team_r_ew3": 1, "team_q_prev": 1, "team_r_prev": 1,
            "drv_q_ew": 1, "drv_r_ew": 1, "drv_q_skill": 1, "drv_r_skill": 1,
            "grid_pct": 1, "q1_gap": 1}

DEFAULT_HP = {"max_depth": 3, "lr": 0.05, "iters": 250, "leaf": 20, "l2": 10.0, "alpha": 10.0, "C": 0.5, "bands": 6}


def _score_model(kind, feats, hp):
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_depth=hp["max_depth"], learning_rate=hp["lr"], max_iter=hp["iters"],
            min_samples_leaf=hp["leaf"], l2_regularization=hp["l2"],
            monotonic_cst=[MONOTONE.get(f, 0) for f in feats], random_state=0)
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             Ridge(alpha=hp["alpha"]))
    raise ValueError(kind)


class RaceModel:
    def __init__(self, stage="pre", kind="gbm", hp=None, dnf_features=True, feats=None, dnf_feats=None):
        self.stage, self.kind = stage, kind
        self.hp = {**DEFAULT_HP, **(hp or {})}
        self.feats = list(feats) if feats is not None else list(PRE if stage == "pre" else POST)
        self.dnf_feats = list(dnf_feats) if dnf_feats is not None else list(DNF_PRE if stage == "pre" else DNF_POST)
        self.use_dnf_features = dnf_features

    def fit(self, train: pd.DataFrame):
        fin = train[train["finished"]].reset_index(drop=True)
        # a feature that is empty in all of the training races (e.g. "last season" before the first season) is set
        # to a constant, which the model then ignores
        self.empty = [f for f in self.feats if fin[f].isna().all()]
        X, y = self._x(fin), fin["r_pct"].to_numpy()
        self.score = _score_model(self.kind, self.feats, self.hp).fit(X, y)

        oof = np.full(len(fin), np.nan)
        for tr, te in GroupKFold(5).split(X, y, groups=fin["race_idx"]):
            Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
            for f in Xtr.columns[Xtr.isna().all()]:      # empty in this fold's training part
                Xtr[f], Xte[f] = 0.5, 0.5
            m = _score_model(self.kind, self.feats, self.hp).fit(Xtr, y[tr])
            oof[te] = m.predict(Xte)
        resid = y - oof
        nb = self.hp["bands"]
        self.edges = np.quantile(oof, np.linspace(0, 1, nb + 1)[1:-1])
        band = np.searchsorted(self.edges, oof)
        self.pools = [resid[band == b] for b in range(nb)]
        self.oof_mae = float(np.abs(resid).mean())

        if self.use_dnf_features:
            self.dnf = make_pipeline(StandardScaler(), LogisticRegression(C=self.hp["C"], max_iter=500))
            self.dnf.fit(train[self.dnf_feats], train["dnf"].astype(int))
            self.dnf_const = None
        else:
            self.dnf, self.dnf_const = None, float(train["dnf"].mean())
        return self

    def _x(self, d):
        X = d[self.feats].copy()
        for f in self.empty:
            X[f] = 0.5
        return X

    def predict(self, test: pd.DataFrame):
        score = self.score.predict(self._x(test))
        if self.dnf is not None:
            p = self.dnf.predict_proba(test[self.dnf_feats])[:, 1]
        else:
            p = np.full(len(test), self.dnf_const)
        return score, p

    def simulate(self, score, p_dnf, n_sims=4000, seed=0):
        return simulate(score, p_dnf, self.edges, self.pools, n_sims, seed)


def simulate(score, p_dnf, edges, pools, n_sims=4000, seed=0):
    """Monte Carlo of one race. Returns a dict of per-driver arrays.

    pos_counts[i, k-1] = share of simulated races in which driver i finished k-th.
    """
    rng = np.random.default_rng(seed)
    n = len(score)
    band = np.searchsorted(edges, score)
    noise = np.empty((n_sims, n))
    for i in range(n):
        pool = pools[band[i]]
        noise[:, i] = rng.choice(pool, size=n_sims, replace=True)
    value = score[None, :] + noise
    out = rng.random((n_sims, n)) < p_dnf[None, :]            # True = does not finish
    value = np.where(out, np.inf, value)
    order = np.argsort(value, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    rank[rows, order] = np.arange(1, n + 1)[None, :]
    rank = np.where(out, 0, rank)                               # 0 = did not finish

    pos_counts = np.zeros((n, n))
    for k in range(1, n + 1):
        pos_counts[:, k - 1] = (rank == k).mean(axis=0)
    fin = rank > 0
    exp_pos = np.where(fin.sum(0) > 0, (rank * fin).sum(0) / np.maximum(fin.sum(0), 1), np.nan)
    return {"p_win": (rank == 1).mean(0), "p_podium": ((rank >= 1) & (rank <= 3)).mean(0),
            "p_top10": ((rank >= 1) & (rank <= 10)).mean(0), "p_dnf": out.mean(0),
            "exp_pos": exp_pos, "pos_counts": pos_counts}
