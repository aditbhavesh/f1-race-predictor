"""Rebuild everything from data/results_2022_2026.csv (about 15 minutes).
    python run_all.py
"""
import subprocess, sys
steps = ["scripts/quick.py",          # cleaning + leak-free features
         "tests/test_features.py data/results_2022_2026.csv",
         "scripts/final_test.py",     # walk-forward test, 2024 onward
         "scripts/export_dashboard.py", "scripts/export_trust.py",
         "tests/crosscheck.py",       # browser simulation vs Python
         "dashboard/build.py"]
for s in steps:
    print("==>", s, flush=True)
    subprocess.run([sys.executable, "-I", *s.split()], check=True)
print("done: open dashboard/gridodds.html")
