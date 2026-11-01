"""Circuit-specific features computed from historical race data.

Two track characteristics are computed for each circuit, strictly from races
that happened BEFORE the current race (no lookahead):

  overtake_difficulty   How often the pole sitter wins and how little positions
                        change from grid to finish. High values (near 1.0) mean
                        the track layout discourages overtaking (e.g. Monaco,
                        Singapore). Computed as a weighted average of
                        pole-win-rate and grid-position-retention-rate across
                        all prior visits to this circuit.

  sc_likelihood         How often safety cars or red flags disrupt races at this
                        circuit, measured as the historical DNF-and-incident rate
                        per race. High values increase variance and underdog
                        chances. Falls back to the field average when a circuit
                        has never been visited before.

Both features are normalised to [0, 1] and are available before qualifying
(pre-qualifying features).
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def _circuit_key(event_name: str) -> str:
    """Normalise circuit identity across seasons (handle renames)."""
    return event_name.strip()


def add_circuit_features(c: pd.DataFrame) -> pd.DataFrame:
    """Add circuit DNA columns to the cleaned results DataFrame.

    Must be called AFTER data.load_clean() and BEFORE features.build_features().
    Respects the no-lookahead rule: each race sees only prior history.

    Adds columns:
        circuit_overtake_diff   (float, 0–1)
        circuit_sc_likelihood   (float, 0–1)
    """
    out = c.copy()
    out["circuit_overtake_diff"] = np.nan
    out["circuit_sc_likelihood"] = np.nan

    # History accumulators keyed by circuit
    # pole_wins[circuit] = [bool, ...]   (did pole sitter win?)
    # pos_retention[circuit] = [float, ...]  (mean |grid - finish| / n_entries)
    # incident_rate[circuit] = [float, ...]  (DNF fraction per race)
    pole_wins: dict[str, list[float]] = defaultdict(list)
    pos_retention: dict[str, list[float]] = defaultdict(list)
    incident_rate: dict[str, list[float]] = defaultdict(list)

    # Global fallbacks
    all_pole_wins: list[float] = []
    all_retention: list[float] = []
    all_incident: list[float] = []

    for rid, race in c.groupby("race_idx", sort=True):
        circuit = _circuit_key(race["EventName"].iat[0])

        # ---- Compute features from PRIOR history only ----
        pw_hist = pole_wins[circuit]
        pr_hist = pos_retention[circuit]
        ir_hist = incident_rate[circuit]

        if pw_hist:
            # Circuit-specific: use circuit data
            ot_diff = 0.6 * np.mean(pw_hist) + 0.4 * (1.0 - np.mean(pr_hist))
            sc_like = np.mean(ir_hist)
        elif all_pole_wins:
            # No prior visits to this specific circuit: use global average
            ot_diff = 0.6 * np.mean(all_pole_wins) + 0.4 * (1.0 - np.mean(all_retention))
            sc_like = np.mean(all_incident)
        else:
            # Very first race in the dataset: use reasonable priors
            ot_diff = 0.5
            sc_like = 0.15

        # Assign to all drivers in this race
        out.loc[race.index, "circuit_overtake_diff"] = ot_diff
        out.loc[race.index, "circuit_sc_likelihood"] = sc_like

        # ---- Now add THIS race to history (after assignment, so no lookahead) ----
        # Pole sitter win
        pole_rows = race[race["grid"] == 1]
        if len(pole_rows) > 0:
            pole_won = float((pole_rows["pos"] == 1).any())
        else:
            pole_won = 0.0
        pole_wins[circuit].append(pole_won)
        all_pole_wins.append(pole_won)

        # Position retention: how much grid order is preserved
        finishers = race[race["finished"]].copy()
        if len(finishers) >= 2:
            n_entries = len(race)
            # Normalised absolute position change: 0 = no movement, 1 = max shuffle
            pos_change = (finishers["grid"] - finishers["pos"]).abs().mean() / n_entries
            retention = 1.0 - min(pos_change, 1.0)  # high = positions preserved
        else:
            retention = 0.5
        pos_retention[circuit].append(retention)
        all_retention.append(retention)

        # Incident / DNF rate
        dnf_frac = float(race["dnf"].mean())
        incident_rate[circuit].append(dnf_frac)
        all_incident.append(dnf_frac)

    return out

