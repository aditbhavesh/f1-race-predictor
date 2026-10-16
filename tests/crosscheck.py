"""The browser simulation (dashboard/engine.js) must agree with the Python one (src/models.py::simulate).
    python3 -I tests/crosscheck.py     (needs node)"""
import json, subprocess, sys, pickle
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from models import simulate

fz = pickle.load(open(root / "outputs/frozen.pkl", "rb"))
n = fz["slot_score"].shape[0]
data = {"slot_score": fz["slot_score"].tolist(), "slot_dnf": fz["slot_dnf"].tolist(), "edges": fz["edges"].tolist(), "pools": [p.tolist() for p in fz["pools"]]}
rng = np.random.default_rng(5)
orders = [list(range(n)), list(rng.permutation(n)), list(rng.permutation(n)), list(rng.permutation(n)[::-1])]
orders = [[int(x) for x in o] for o in orders]
NS = 200_000
code = f"""
const E=require({json.dumps(str(root/'dashboard/engine.js'))}); const d={json.dumps(data)}; const orders={json.dumps(orders)};
const out=orders.map((o,k)=>{{const r=E.whatIf(d,o,{NS},100+k); return {{p_win:r.p_win,p_podium:r.p_podium,p_top10:r.p_top10,p_dnf:r.p_dnf,pc:r.pos_counts}};}});
const sc=[0.3,0.1,0.5,0.2]; const det=E.simulate(sc,[0,0,0,0],[0.5],[[0],[0]],50,1);
console.log(JSON.stringify({{out,det}}));
"""
js = subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True)
res = json.loads(js.stdout)
worst = 0.0
for k, o in enumerate(orders):
    score, p = np.zeros(n), np.zeros(n)
    for s, d in enumerate(o):                      # driver d starts in slot s
        score[d], p[d] = fz["slot_score"][d][s], fz["slot_dnf"][d][s]
    py = simulate(score, p, fz["edges"], fz["pools"], NS, 900 + k)
    j = res["out"][k]
    for key in ("p_win", "p_podium", "p_top10", "p_dnf"):
        worst = max(worst, float(np.abs(np.array(j[key]) - py[key]).max()))
    worst = max(worst, float(np.abs(np.array(j["pc"]) - py["pos_counts"]).max()))
    assert abs(sum(j["p_win"]) - 1) < 2e-3 and abs(sum(j["p_podium"]) - 3) < 2e-3 and abs(sum(j["p_top10"]) - 10) < 2e-3
print(f"largest difference between browser and Python odds, {len(orders)} grids x {n} drivers: {worst:.4f}")
assert worst < 0.008, worst
assert [int(np.argmax(r)) + 1 for r in res["det"]["pos_counts"]] == [3, 1, 4, 2], res["det"]["pos_counts"]
print("deterministic ranking check: OK")
print("browser simulation agrees with the Python simulation: OK")
