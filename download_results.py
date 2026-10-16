"""Download race + qualifying results for several F1 seasons with FastF1 and save one CSV.

Usage (run in a terminal, in the folder where you want the file):
    python download_results.py                 # seasons 2022 to 2026 (default)
    python download_results.py 2023 2025       # a custom range: first and last season

Output: results_<first>_<last>.csv   (attach this file back to Claude)

Notes:
  - Only results are loaded (no lap or telemetry data), so this is much faster than the lap download:
    expect roughly 10-20 minutes for 5 seasons the first time. A local "cache" folder is created.
  - Races that fail to load are skipped and listed at the end; everything else still saves.
  - Races that have not happened yet (the current season) are skipped automatically.
"""
import os
import sys

import pandas as pd
import fastf1

first = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
last = int(sys.argv[2]) if len(sys.argv) > 2 else 2026

os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")

KEEP = ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "ClassifiedPosition",
        "GridPosition", "Status", "Points", "Laps", "Q1", "Q2", "Q3"]


def to_sec(s):
    """timedelta column -> seconds (NaN stays NaN)."""
    return s.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(s) else s


def load_results(year, rnd, kind):
    s = fastf1.get_session(year, rnd, kind)
    s.load(laps=False, telemetry=False, weather=(kind == "R"), messages=False)
    res = s.results
    cols = [c for c in KEEP if c in res.columns]
    df = res[cols].copy()
    for c in ("Q1", "Q2", "Q3"):
        if c in df.columns:
            df[c] = to_sec(df[c])
    return s, df


frames, failed = [], []
for year in range(first, last + 1):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[pd.to_datetime(schedule["EventDate"]) < pd.Timestamp.now()]   # only races already run
    for _, ev in schedule.iterrows():
        rnd, name = int(ev["RoundNumber"]), ev["EventName"]
        print(f"[{year} R{rnd}] {name} ...", flush=True)
        try:
            race_s, race = load_results(year, rnd, "R")
            try:
                _, quali = load_results(year, rnd, "Q")
                quali = quali[["Abbreviation", "Position", "Q1", "Q2", "Q3"]].rename(columns={"Position": "QualiPosition"})
                race = race.drop(columns=[c for c in ("Q1", "Q2", "Q3") if c in race.columns]).merge(
                    quali, on="Abbreviation", how="left")
            except Exception as e:
                print(f"   no qualifying data: {e}")
            race["Season"], race["Round"], race["EventName"] = year, rnd, name
            race["Location"] = ev.get("Location", "")
            race["Country"] = ev.get("Country", "")
            race["EventDate"] = str(ev["EventDate"])[:10]
            try:
                race["Rain"] = bool(race_s.weather_data["Rainfall"].any())
            except Exception:
                race["Rain"] = pd.NA
            frames.append(race)
        except Exception as e:
            print(f"   skipped: {e}")
            failed.append(f"{year} R{rnd} {name}")

if not frames:
    sys.exit("No races downloaded. Check your internet connection and fastf1 version.")

out = pd.concat(frames, ignore_index=True)
fname = f"results_{first}_{last}.csv"
out.to_csv(fname, index=False)
print(f"\nSaved {len(out)} driver results from {out.groupby(['Season', 'Round']).ngroups} races to {fname}")
if failed:
    print("Skipped:", ", ".join(failed))
