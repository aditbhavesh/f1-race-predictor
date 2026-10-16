"""Load and clean the 2022-2026 race results.

One row per driver per race. Everything the models need is derived from this table.

Cleaning rules (each is a deliberate choice, listed in the README):
  - Races with no race result at all (2025 Miami in the FastF1 download) are dropped.
  - Rows with no grid slot (driver withdrew before qualifying) are dropped.
  - "Did not start" counts as a non-finish. A forecast made before the race cannot know it.
  - A pit-lane start (grid slot 0 in the feed) is placed at the back of the grid.
  - A driver is a finisher if the feed gives a numeric classified position.
    Retired, disqualified and did-not-start drivers are non-finishers.
  - Team renames are linked into one "entry" so form carries over a rename:
    Alfa Romeo -> Kick Sauber -> Audi, and AlphaTauri -> RB -> Racing Bulls.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ENTRY = {
    "Alfa Romeo": "Sauber", "Kick Sauber": "Sauber", "Audi": "Sauber",
    "AlphaTauri": "RacingBulls", "RB": "RacingBulls", "Racing Bulls": "RacingBulls",
}
# Seasons where the car rules changed so much that last season's car pace says little.
REGULATION_RESET_SEASONS = (2026,)


def load_results(path) -> pd.DataFrame:
    d = pd.read_csv(Path(path))
    d["Abbreviation"] = d["Abbreviation"].astype(str)
    d["race_date"] = pd.to_datetime(d["EventDate"])
    d = d.sort_values(["Season", "Round", "QualiPosition", "Abbreviation"]).reset_index(drop=True)
    return d


def clean_results(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # races with no race result at all
    has_result = d.groupby(["Season", "Round"])["ClassifiedPosition"].transform(lambda s: s.notna().any())
    d = d[has_result]
    # never took part in qualifying or the race
    d = d[d["GridPosition"].notna()].copy()

    d["race_id"] = d["Season"].astype(str) + "-" + d["Round"].astype(int).astype(str).str.zfill(2)
    d["entry"] = d["TeamName"].map(lambda t: ENTRY.get(t, t))
    d["n_entries"] = d.groupby("race_id")["Abbreviation"].transform("size")

    d["finished"] = d["ClassifiedPosition"].astype(str).str.fullmatch(r"\d+") & (d["Status"] != "Did not start")
    d["pos"] = np.where(d["finished"], pd.to_numeric(d["ClassifiedPosition"], errors="coerce"), np.nan)
    d["k_finishers"] = d.groupby("race_id")["finished"].transform("sum")

    n = d["n_entries"].astype(float)
    grid = d["GridPosition"].where(d["GridPosition"] > 0, n)         # pit-lane start -> back
    d["grid"] = grid
    d["grid_pct"] = (grid - 1) / (n - 1)
    d["q_pct"] = ((d["QualiPosition"] - 1) / (n - 1)).clip(0, 1)
    d["r_pct"] = (d["pos"] - 1) / (n - 1)                            # NaN for non-finishers
    d["grid_pen"] = d["grid_pct"] - d["q_pct"]                       # >0 = penalty, 0 = none
    q1_min = d.groupby("race_id")["Q1"].transform("min")
    d["q1_gap"] = d["Q1"] / q1_min - 1                               # NaN if no Q1 time

    d["won"] = d["pos"] == 1
    d["podium"] = d["pos"] <= 3
    d["top10"] = d["pos"] <= 10
    d["dnf"] = ~d["finished"]
    d = d.sort_values(["Season", "Round", "QualiPosition", "Abbreviation"]).reset_index(drop=True)
    order = {r: i for i, r in enumerate(d["race_id"].drop_duplicates())}
    d["race_idx"] = d["race_id"].map(order)
    return d


def load_clean(path) -> pd.DataFrame:
    return clean_results(load_results(path))


if __name__ == "__main__":
    import sys
    c = load_clean(sys.argv[1])
    print(f"{len(c)} driver-races, {c['race_id'].nunique()} races, seasons {sorted(c['Season'].unique())}")
    print("non-finishers:", int(c["dnf"].sum()), f"({c['dnf'].mean():.1%})")
    print("pole sitter won:", int(((c['grid'] == 1) & c['won']).sum()), "of", int((c['grid'] == 1).sum()))
