"""Inline the engine, the forecast and the trust numbers into self-contained HTML pages.
Builds both dashboard/gridodds.html and index.html (for root GitHub Pages deployment)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
template_path = root / "dashboard/template.html"
engine_path = root / "dashboard/engine.js"
data_path = root / "outputs/dashboard_data.json"
trust_path = root / "outputs/trust.json"

t = template_path.read_text(encoding="utf-8")
t = t.replace("/*ENGINE*/", engine_path.read_text(encoding="utf-8"))
t = t.replace("/*DATA*/", data_path.read_text(encoding="utf-8"))
t = t.replace("/*TRUST*/", trust_path.read_text(encoding="utf-8"))

# Write to dashboard/gridodds.html
gridodds_path = root / "dashboard/gridodds.html"
gridodds_path.write_text(t, encoding="utf-8")

# Write to root index.html for direct GitHub Pages deployment
index_path = root / "index.html"
index_path.write_text(t, encoding="utf-8")

print(f"Built {len(t) // 1024} KB -> {gridodds_path.relative_to(root)} & {index_path.relative_to(root)}")
