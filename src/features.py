"""Features for every driver in every race, built ONLY from races that happened earlier.

All positions are turned into "percent of the field" so a 22-car grid and a 20-car grid are comparable:
    pct = (position - 1) / (cars - 1)      0 = first, 1 = last.

Pre-qualifying features (known on Friday):
  team_q_ew3 / team_r_ew3   this season's team form: qualifying and race result of the team's cars, recent races
                            count more (half-life 3 races). Empty at round 1.
  team_q_prev / team_r_prev last season's final six races for the same team entry. Left empty when the car rules
                            were reset (2026), because last year's car says little about this year's.
  n_prior_season            races already run this season (lets the model learn how much to trust team form)
  drv_q_ew / drv_r_ew       the driver's own results this season, recent races weighted more
  drv_q_skill / drv_r_skill driver skill with the car taken out: own result minus teammate's, averaged over
                            up to the last ~30 races across seasons (half-life 10). Negative = better than the teammate.
  drv_n_prior               career races in the data (rookie flag)
Reliability features (used for the did-not-finish model):
  drv_dnf_rate, team_dnf_rate  shrunk toward the overall rate, over recent races
  field_dnf_ew              how often the whole field has been failing recently (catches a new-rules season)
Post-qualifying features (known on Saturday night):
  grid_pct, grid_pen, q1_gap  starting slot, penalty size, and gap to the fastest Q1 lap
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from data import REGULATION_RESET_SEASONS

PRE = ["team_q_ew3", "team_r_ew3", "team_q_prev", "team_r_prev", "n_prior_season",
       "drv_q_ew", "drv_r_ew", "drv_q_skill", "drv_r_skill", "drv_n_prior"]
POST = PRE + ["grid_pct", "grid_pen", "q1_gap"]
DNF_PRE = ["drv_dnf_rate", "team_dnf_rate", "field_dnf_ew"]
DNF_POST = DNF_PRE + ["grid_pct"]
ALL_FEATURES = list(dict.fromkeys(POST + DNF_POST))

PREV_SEASON_RACES = 6
DNF_PRIOR_DEFAULT = 0.12


def ew_mean(values, halflife):
    """Exponentially weighted mean of a list ordered oldest -> newest, ignoring NaN. NaN if nothing valid."""
    v = np.asarray(values, float)
    if v.size == 0:
        return np.nan
    age = np.arange(v.size - 1, -1, -1)
    w = 0.5 ** (age / halflife)
    ok = ~np.isnan(v)
    return float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def _nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if a.size and not np.isnan(a).all() else np.nan


def build_features(c: pd.DataFrame, reset_seasons=REGULATION_RESET_SEASONS, carry_prev: bool = True) -> pd.DataFrame:
    out = c.copy()
    for col in ALL_FEATURES:
        out[col] = np.nan

    team_hist = defaultdict(list)      # entry  -> [(season, team_q, team_r, dnf_rate_of_its_cars, n_cars)]
    drv_hist = defaultdict(list)       # driver -> [(season, q_pct, r_pct, q_delta, r_delta, dnf)]
    field_hist = []                    # [(dnf_rate)]
    n_races_season = defaultdict(int)
    all_dnf, all_rows = 0, 0

    for rid, race in c.groupby("race_idx", sort=True):
        season = int(race["Season"].iat[0])
        p0 = (all_dnf / all_rows) if all_rows >= 40 else DNF_PRIOR_DEFAULT
        field_dnf = ew_mean([f for f in field_hist], 3) if field_hist else np.nan

        for ix, row in race.iterrows():
            ent, drv = row["entry"], row["Abbreviation"]
            th = team_hist[ent]
            cur = [t for t in th if t[0] == season]
            out.at[ix, "team_q_ew3"] = ew_mean([t[1] for t in cur], 3)
            out.at[ix, "team_r_ew3"] = ew_mean([t[2] for t in cur], 3)
            prev = [t for t in th if t[0] == season - 1][-PREV_SEASON_RACES:]
            if carry_prev and season not in reset_seasons and prev:
                out.at[ix, "team_q_prev"] = _nanmean([t[1] for t in prev])
                out.at[ix, "team_r_prev"] = _nanmean([t[2] for t in prev])
            out.at[ix, "n_prior_season"] = n_races_season[season]

            dh = drv_hist[drv]
            dcur = [h for h in dh if h[0] == season]
            out.at[ix, "drv_q_ew"] = ew_mean([h[1] for h in dcur], 3)
            out.at[ix, "drv_r_ew"] = ew_mean([h[2] for h in dcur], 3)
            out.at[ix, "drv_q_skill"] = ew_mean([h[3] for h in dh[-30:]], 10)
            out.at[ix, "drv_r_skill"] = ew_mean([h[4] for h in dh[-30:]], 10)
            out.at[ix, "drv_n_prior"] = len(dh)

            last_d = dh[-30:]
            kd = 15.0
            out.at[ix, "drv_dnf_rate"] = (sum(h[5] for h in last_d) + kd * p0) / (len(last_d) + kd)
            last_t = [(t[3], t[4]) for t in th[-30:]]
            kt = 30.0
            out.at[ix, "team_dnf_rate"] = (sum(r * n for r, n in last_t) + kt * p0) / (sum(n for _, n in last_t) + kt)
            out.at[ix, "field_dnf_ew"] = field_dnf if not np.isnan(field_dnf) else p0

            out.at[ix, "grid_pct"] = row["grid_pct"]
            out.at[ix, "grid_pen"] = row["grid_pen"]
            out.at[ix, "q1_gap"] = row["q1_gap"]

        # ---- only now add this race to the history (so it cannot leak into its own features) ----
        for ent, grp in race.groupby("entry"):
            team_hist[ent].append((season, _nanmean(grp["q_pct"]), _nanmean(grp["r_pct"]),
                                   float(grp["dnf"].mean()), len(grp)))
        for ix, row in race.iterrows():
            mates = race[(race["entry"] == row["entry"]) & (race.index != ix)]
            q_delta = row["q_pct"] - _nanmean(mates["q_pct"]) if len(mates) else np.nan
            r_delta = row["r_pct"] - _nanmean(mates["r_pct"]) if len(mates) else np.nan
            drv_hist[row["Abbreviation"]].append((season, row["q_pct"], row["r_pct"], q_delta, r_delta, float(row["dnf"])))
        field_hist.append(float(race["dnf"].mean()))
        n_races_season[season] += 1
        all_dnf += int(race["dnf"].sum())
        all_rows += len(race)
    return out


def standings_baseline(c: pd.DataFrame) -> pd.Series:
    """Championship standing before each race (1 = leader), from points scored in the same season only.
    Round 1 falls back to the previous season's final standing. Used only as a baseline, never as a feature."""
    pts = defaultdict(float)
    final_prev = {}
    rank = pd.Series(np.nan, index=c.index)
    season_prev = None
    for rid, race in c.groupby("race_idx", sort=True):
        season = int(race["Season"].iat[0])
        if season != season_prev:
            if season_prev is not None:
                order = sorted(pts, key=lambda d: -pts[d])
                final_prev = {d: i + 1 for i, d in enumerate(order)}
            pts = defaultdict(float)
            season_prev = season
        if any(v > 0 for v in pts.values()):
            order = sorted(race["Abbreviation"], key=lambda d: -pts.get(d, 0.0))
            r = {d: i + 1 for i, d in enumerate(order)}
        else:
            r = {d: final_prev.get(d, 99) for d in race["Abbreviation"]}
            order = sorted(r, key=lambda d: r[d])
            r = {d: i + 1 for i, d in enumerate(order)}
        for ix, row in race.iterrows():
            rank.at[ix] = r[row["Abbreviation"]]
        for ix, row in race.iterrows():
            pts[row["Abbreviation"]] += float(row["Points"]) if pd.notna(row["Points"]) else 0.0
    return rank
