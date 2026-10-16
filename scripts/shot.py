import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
root = Path(__file__).resolve().parents[1]; out = root / "outputs/shots"; out.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vw, scheme in [("desk-light", 1200, "light"), ("phone-dark", 400, "dark")]:
        pg = b.new_page(viewport={"width": vw, "height": 900}, color_scheme=scheme)
        errs = []; pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("file://" + str(root / "dashboard/gridodds.html")); pg.wait_for_timeout(800)
        for t in ["forecast", "grid", "past", "trust"]:
            pg.click(f"#t-{t}"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(out / f"{name}-{t}.png"), full_page=True)
            sw = pg.evaluate("document.documentElement.scrollWidth"); print(name, t, "scrollWidth", sw, "viewport", vw)
        print(name, "errors:", [e for e in errs if "fonts" not in e.lower() and "ERR_" not in e][:5])
    b.close()
