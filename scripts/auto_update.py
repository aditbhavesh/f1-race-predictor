"""Master automated pipeline script.
Runs end-to-end to sync latest F1 results, detect upcoming GP, run simulations, and build the live dashboard.
Can be executed locally or automatically via GitHub Actions."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_step(step_name, command_args):
    print(f"\n==========================================")
    print(f"==> STEP: {step_name}")
    print(f"==========================================")
    res = subprocess.run([sys.executable, *command_args], cwd=str(ROOT))
    if res.returncode != 0:
        print(f"Error in step '{step_name}' (exit code {res.returncode})")
        sys.exit(res.returncode)

def main():
    print("Starting automated F1 race prediction and dashboard update pipeline...")
    
    # Step 1: Check and sync newly completed race results
    run_step("Syncing Latest F1 Race Results", ["scripts/sync_latest_results.py"])
    
    # Step 2: Detect next race, fit models, and export simulation data
    run_step("Generating Next Race Predictions", ["scripts/export_dashboard.py"])
    
    # Step 3: Build self-contained HTML dashboards (gridodds.html & index.html)
    run_step("Compiling Live Dashboard", ["dashboard/build.py"])
    
    print("\n==========================================")
    print("SUCCESS: Live dashboard successfully updated!")
    print("Next event forecast and interactive grid are ready in index.html.")
    print("==========================================\n")

if __name__ == "__main__":
    main()
