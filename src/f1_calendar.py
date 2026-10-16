"""Calendar helper to dynamically detect upcoming F1 events."""
import pandas as pd
import fastf1
import warnings
warnings.filterwarnings("ignore")


def get_next_event(year: int = None):
    """
    Finds the upcoming race event using FastF1's official event schedule.
    Returns a dict with: name, place, country, date, season, round.
    """
    now = pd.Timestamp.now()
    if year is None:
        year = now.year

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False).copy()
    except Exception as e:
        print(f"Warning: Failed to fetch online schedule for {year}: {e}")
        return None

    schedule.loc[:, "EventDate"] = pd.to_datetime(schedule["EventDate"])
    # Consider an event active until the Monday after the race (+1 day after Sunday)
    upcoming = schedule[schedule["EventDate"] + pd.Timedelta(days=1) >= now]

    if upcoming.empty:
        # Season is over, check next season round 1
        try:
            next_sched = fastf1.get_event_schedule(year + 1, include_testing=False).copy()
            if not next_sched.empty:
                ev = next_sched.iloc[0]
                return {
                    "name": str(ev["EventName"]),
                    "place": str(ev.get("Location", "")),
                    "country": str(ev.get("Country", "")),
                    "date": str(ev["EventDate"])[:10],
                    "season": int(year + 1),
                    "round": int(ev["RoundNumber"])
                }
        except Exception:
            pass
        # Fallback to last event of current season
        ev = schedule.iloc[-1]
        return {
            "name": str(ev["EventName"]),
            "place": str(ev.get("Location", "")),
            "country": str(ev.get("Country", "")),
            "date": str(ev["EventDate"])[:10],
            "season": int(year),
            "round": int(ev["RoundNumber"])
        }

    ev = upcoming.iloc[0]
    return {
        "name": str(ev["EventName"]),
        "place": str(ev.get("Location", "")),
        "country": str(ev.get("Country", "")),
        "date": str(ev["EventDate"])[:10],
        "season": int(year),
        "round": int(ev["RoundNumber"])
    }


if __name__ == "__main__":
    ev = get_next_event()
    print("Next event detected:", ev)
