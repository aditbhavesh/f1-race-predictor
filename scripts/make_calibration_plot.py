"""Generate reliability diagram (calibration curves) for walk-forward predictions across 64 test races.

Computes Expected Calibration Error (ECE) and plots predicted probability bins vs observed
frequency with bin counts for Win, Podium, and Top-10 Points finishes.
Saves figure to outputs/figures/calibration.png.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_walk_forward_data():
    """Load walk-forward predictions from dashboard_data.json."""
    data_path = ROOT / "outputs/dashboard_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    backtest = data.get("backtest", [])
    records = []
    for race in backtest:
        rid = race["id"]
        for r in race["rows"]:
            pos = r["pos"]
            records.append({
                "race_id": rid,
                "abbr": r["abbr"],
                "pos": pos,
                "won": 1 if (pos is not None and pos == 1) else 0,
                "podium": 1 if (pos is not None and pos <= 3) else 0,
                "top10": 1 if (pos is not None and pos <= 10) else 0,
                "p_win": r["post"][0],
                "p_podium": r["post"][1],
                "p_top10": r["post"][2],
                "p_dnf": r["post"][3] if len(r["post"]) > 3 else None,
            })

    return pd.DataFrame(records), len(backtest)


def compute_calibration_bins(df, prob_col, target_col, bins=None):
    """Compute mean predicted probability, observed frequency, and count per bin."""
    if bins is None:
        bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001]

    cut = pd.cut(df[prob_col], bins, include_lowest=True)
    grouped = df.groupby(cut, observed=True).agg(
        p_mean=(prob_col, "mean"),
        obs_mean=(target_col, "mean"),
        count=(target_col, "size")
    ).reset_index()

    # Calculate ECE: sum (n_b / N) * |p_mean - obs_mean|
    total_n = len(df)
    ece = (grouped["count"] / total_n * (grouped["p_mean"] - grouped["obs_mean"]).abs()).sum()

    return grouped, ece


def plot_calibration(df, n_races, output_path):
    """Plot 3-panel reliability diagram with bin counts and ECE."""
    targets = [
        ("p_win", "won", "Win Probability (P_win)", "#e10600"),
        ("p_podium", "podium", "Podium Probability (P_podium)", "#ff8700"),
        ("p_top10", "top10", "Points Probability (P_top10)", "#1e88e5"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    results = {}
    for ax, (prob_col, target_col, title, color) in zip(axes, targets):
        bins_df, ece = compute_calibration_bins(df, prob_col, target_col)
        results[target_col] = {"ece": ece, "bins": bins_df}

        # Reference diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", linewidth=1.5, label="Perfect calibration", zorder=1)

        # Plot calibration points and line
        ax.plot(bins_df["p_mean"], bins_df["obs_mean"], marker="o", markersize=7,
                color=color, linewidth=2, label="Model (post-quali)", zorder=3)

        # Annotate bin counts with clean, non-overlapping placement
        for idx, row in bins_df.iterrows():
            if np.isnan(row["p_mean"]) or np.isnan(row["obs_mean"]):
                continue
            p = row["p_mean"]
            obs = row["obs_mean"]
            n = int(row["count"])

            if idx == 0:
                xytext = (14, -2)
                ha = "left"
            elif idx == 1:
                xytext = (-8, 8)
                ha = "right"
            elif idx == 2:
                xytext = (12, -8)
                ha = "left"
            elif obs > 0.75:
                xytext = (0, -14)
                ha = "center"
            elif obs >= p:
                xytext = (0, 8)
                ha = "center"
            else:
                xytext = (0, -14)
                ha = "center"

            ax.annotate(f"n={n}", (p, obs), textcoords="offset points",
                        xytext=xytext, ha=ha, fontsize=8,
                        color="#212529", weight="normal", zorder=4)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Inset text box with metrics
        text_str = f"ECE = {ece:.3f} ({ece*100:.2f}%)\nN = {len(df):,} drivers\n({n_races} races)"
        ax.text(0.05, 0.93, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9))

        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    axes[0].set_ylabel("Observed Frequency", fontsize=10)
    fig.suptitle("Reliability Diagram: Out-of-Sample Walk-Forward Predictions (64 Grands Prix, 2024–2026)",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to: {output_path}")
    return results


def main():
    df, n_races = load_walk_forward_data()
    out_img = ROOT / "outputs/figures/calibration.png"
    results = plot_calibration(df, n_races, out_img)

    print("\nExpected Calibration Error (ECE) across 64 test races (1,312 driver-races):")
    for k, v in results.items():
        print(f"  - {k:8s}: ECE = {v['ece']:.4f} ({v['ece']*100:.2f}%)")


if __name__ == "__main__":
    main()
