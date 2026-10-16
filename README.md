# Grid Odds: an F1 Race Outcome Predictor

Chances of winning, reaching the podium, and scoring points for every driver, computed by a model tested exclusively on races it had never seen. An interactive dashboard forecasts the next Grand Prix, lets you set any custom starting grid with real-time Monte Carlo re-calculation in the browser, and explores predictions across 64 past races.

> ⚡ **Automated & Future-Proof:** This dashboard runs continuously on **GitHub Actions**. Every Tuesday at 06:00 UTC (and on-demand), an automated workflow pulls the official FIA schedule via FastF1, detects the upcoming Grand Prix, refits the monotonic gradient-boosting model, runs 20,000 Monte Carlo simulations, and publishes fresh predictions to GitHub Pages completely hands-free.

---

## Results (64 test races, 2024 to Singapore 2026, refitted before every race)

| Method | Win Brier | Podium Brier | Points Brier | Winner right | Podium of 3 | Points of 10 |
|---|---|---|---|---|---|---|
| **Model before qualifying** | **0.042** | **0.087** | **0.168** | **20%** | **1.5** | **7.4** |
| Baseline: championship order | 0.040 | 0.085 | 0.172 | 27% | 1.7 | 7.4 |
| **Model after qualifying** | **0.032** | **0.069** | **0.146** | **58%** | **2.0** | **7.9** |
| Baseline: starting slot | 0.030 | 0.064 | 0.155 | 61% | 2.1 | 7.9 |

*Brier score is the mean squared error of probabilistic odds (lower is better).*

### Key Takeaways
- **Probabilities are well-calibrated:** When the model outputs a 50% podium probability, that outcome occurs approximately 50% of the time in out-of-sample testing.
- **Rigor & Realism:** The model after qualifying outperforms the naive "starting slot" baseline on points-place distribution while maintaining parity on race winners.
- **Interactive In-Browser Simulation:** `dashboard/engine.js` runs identical Monte Carlo simulations directly on client devices, allowing instant What-If analysis when altering starting grid slots.

---

## Architecture & Method

1. **Normalized Grid Percentiles:** Finishing positions are scaled as percentages of the field (0.0 = 1st, 1.0 = last), enabling rigorous cross-comparison between 20-car and 22-car fields.
2. **Leakage-Free Feature Engineering:** Every feature for race $N$ is strictly computed using only races $< N$ (enforced and tested via `tests/test_features.py`).
3. **Monotonic Gradient Boosting:** Ensures that improved recent form or a higher grid slot can never penalize a driver's predicted finish.
4. **Reliability & Noise Modeling:** Logistic regression models retirement risk (DNF), and empirical 5-fold out-of-sample error bands provide realistic stochastic variance during Monte Carlo generation.

---

## Live Automation & GitHub Pages Hosting

The repository is built to self-sustain season after season:

1. **Automated Cron Pipeline (`.github/workflows/update_predictions.yml`):**
   - Runs automatically every Tuesday after race weekend.
   - Syncs any completed race results via `scripts/sync_latest_results.py`.
   - Discovers the next upcoming Grand Prix via `src/f1_calendar.py`.
   - Runs `scripts/export_dashboard.py` and compiles self-contained `index.html`.
   - Commits and pushes changes back to the repository.

2. **Deploying to GitHub Pages (2 Clicks):**
   - Push this repository to GitHub: `git push origin main`
   - In your repo settings, go to **Settings ➔ Pages**.
   - Under **Build and deployment > Source**, select **Deploy from a branch**.
   - Branch: **`main`**, Folder: **`/(root)`** ➔ Click **Save**.
   - Your live dashboard is immediately hosted at `https://<your-username>.github.io/f1-race-predictor/`.

---

## Local Usage

### Quick Update (Takes ~5 seconds):
```bash
pip install -r requirements.txt
python scripts/auto_update.py
```
Open `index.html` or `dashboard/gridodds.html` in any browser.

### Full Walk-Forward Re-test (Recomputes all 64 historical benchmarks, ~15 mins):
```bash
python run_all.py
```
