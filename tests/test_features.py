"""No-leakage tests: a race's features must not depend on that race's result, or on any later race.

    python tests/test_features.py data/results_2022_2026.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import clean_results, load_results
from features import ALL_FEATURES, PRE, build_features

RESULT_COLS = ["Position", "ClassifiedPosition", "Status", "Points", "Laps"]


def _raw(path):
    return load_results(path)


def test_features_ignore_results_of_the_race_itself_and_the_future(path):
    raw = _raw(path)
    full = build_features(clean_results(raw))
    rng = np.random.default_rng(0)
    ids = full["race_id"].drop_duplicates().tolist()
    for rid in [ids[3], ids[30], ids[60], ids[-1]]:
        season, rnd = (int(x) for x in rid.split("-"))
        cut = raw[(raw["Season"] < season) | ((raw["Season"] == season) & (raw["Round"] <= rnd))].copy()
        m = (cut["Season"] == season) & (cut["Round"] == rnd)
        # scramble the race's own results: positions, status, points
        for col in ["Position", "Points", "Laps"]:
            cut.loc[m, col] = rng.permutation(cut.loc[m, col].to_numpy())
        cut.loc[m, "ClassifiedPosition"] = rng.permutation(cut.loc[m, "ClassifiedPosition"].to_numpy())
        cut.loc[m, "Status"] = rng.permutation(cut.loc[m, "Status"].to_numpy())
        part = build_features(clean_results(cut))
        a = full[full["race_id"] == rid].set_index("Abbreviation")[ALL_FEATURES].sort_index()
        b = part[part["race_id"] == rid].set_index("Abbreviation")[ALL_FEATURES].sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9, obj=f"features of {rid}")
    print("features of a race ignore its own result and every later race: OK")


def test_pre_qualifying_features_ignore_grid_and_quali_of_the_race(path):
    raw = _raw(path)
    full = build_features(clean_results(raw))
    rid = full["race_id"].drop_duplicates().tolist()[50]
    season, rnd = (int(x) for x in rid.split("-"))
    cut = raw[(raw["Season"] < season) | ((raw["Season"] == season) & (raw["Round"] <= rnd))].copy()
    m = (cut["Season"] == season) & (cut["Round"] == rnd)
    for col in ["GridPosition", "QualiPosition", "Q1", "Q2", "Q3"]:
        cut.loc[m, col] = np.random.default_rng(1).permutation(cut.loc[m, col].to_numpy())
    part = build_features(clean_results(cut))
    a = full[full["race_id"] == rid].set_index("Abbreviation")[PRE].sort_index()
    b = part[part["race_id"] == rid].set_index("Abbreviation")[PRE].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)
    print("pre-qualifying features ignore the race's own grid and qualifying: OK")


def test_regulation_reset_blanks_last_seasons_team_form(path):
    f = build_features(clean_results(_raw(path)))
    r1_2026 = f[f["race_id"] == "2026-01"]
    assert r1_2026["team_q_prev"].isna().all() and r1_2026["team_r_prev"].isna().all()
    r1_2025 = f[f["race_id"] == "2025-01"]
    assert r1_2025["team_q_prev"].notna().sum() >= 18
    print("2026 starts with no carried-over car form, 2025 does: OK")


if __name__ == "__main__":
    p = sys.argv[1]
    test_features_ignore_results_of_the_race_itself_and_the_future(p)
    test_pre_qualifying_features_ignore_grid_and_quali_of_the_race(p)
    test_regulation_reset_blanks_last_seasons_team_form(p)
