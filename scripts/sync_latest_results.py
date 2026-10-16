"""Sync any newly completed race results into data/results_2022_2026.csv."""
import os
import sys
from pathlib import Path
import pandas as pd
import fastf1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_calendar import get_next_event

CSV_PATH = ROOT / "data/results_2022_2026.csv"
CACHE_DIR = ROOT / "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

KEEP = ["DriverNumber", "Abbreviation", "FullName", "TeamName", "Position", "ClassifiedPosition",
        "GridPosition", "Status", "Points", "Laps", "Q1", "Q2", "Q3"]


def to_sec(s):
    return s.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(s) else s


def load_session_results(year, rnd, kind):
    s = fastf1.get_session(year, rnd, kind)
    s.load(laps=False, telemetry=False, weather=(kind == "R"), messages=False)
    res = s.results
    cols = [c for c in KEEP if c in res.columns]
    df = res[cols].copy()
    for c in ("Q1", "Q2", "Q3"):
        if c in df.columns:
            df[c] = to_sec(df[c])
    return s, df


def sync_new_races():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return False

    existing = pd.read_csv(CSV_PATH)
    latest_season = int(existing["Season"].max())
    latest_round = int(existing[existing["Season"] == latest_season]["Round"].max())

    now = pd.Timestamp.now()
    schedule = fastf1.get_event_schedule(latest_season, include_testing=False)
    schedule["EventDate"] = pd.to_datetime(schedule["EventDate"])

    # Completed events after the latest round in our CSV
    new_events = schedule[(schedule["RoundNumber"] > latest_round) & (schedule["EventDate"] + pd.Timedelta(hours=4) < now)]

    if new_events.empty:
        print(f"No new completed races to sync. Dataset is up-to-date through {latest_season} Round {latest_round}.")
        return False

    new_frames = []
    for _, ev in new_events.iterrows():
        rnd, name = int(ev["RoundNumber"]), ev["EventName"]
        print(f"Syncing new results: [{latest_season} R{rnd}] {name} ...")
        try:
            race_s, race = load_session_results(latest_season, rnd, "R")
            try:
                _, quali = load_session_results(latest_season, rnd, "Q")
                quali = quali[["Abbreviation", "Position", "Q1", "Q2", "Q3"]].rename(columns={"Position": "QualiPosition"})
                race = race.drop(columns=[c for c in ("Q1", "Q2", "Q3") if c in race.columns]).merge(quali, on="Abbreviation", how="left")
            except Exception as q_err:
                print(f"   Note: Qualifying data unavailable: {q_err}")

            race["Season"], race["Round"], race["EventName"] = latest_season, rnd, name
            race["Location"] = ev.get("Location", "")
            race["Country"] = ev.get("Country", "")
            race["EventDate"] = str(ev["EventDate"])[:10]
            try:
                race["Rain"] = bool(race_s.weather_data["Rainfall"].any())
            except Exception:
                race["Rain"] = pd.NA

            new_frames.append(race)
            print(f"   Successfully loaded {len(race)} driver results.")
        except Exception as err:
            print(f"   Failed to load session for {name}: {err}")

    if new_frames:
        updated = pd.concat([existing] + new_frames, ignore_index=True)
        updated.to_csv(CSV_PATH, index=False)
        print(f"Dataset updated! Added {len(new_frames)} race(s). Total rows: {len(updated)}")
        return True

    return False


if __name__ == "__main__":
    sync_new_races()
