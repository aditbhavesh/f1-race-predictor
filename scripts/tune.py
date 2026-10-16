import sys, itertools, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, numpy as np
from backtest import *
f = pd.read_pickle("outputs/feat.pkl")
res=[]
for d,it,lf,l2 in itertools.product((2,3),(100,250),(20,50),(2.0,10.0)):
    hp=dict(max_depth=d,iters=it,leaf=lf,l2=l2)
    tot=[]
    for st in ("pre","post"):
        df=walk_forward(f,22,st,"gbm",hp,end_idx=43,n_sims=1000,refit_every=2)
        m=per_race_metrics(df); tot.append(m.brier_win.mean()+m.brier_podium.mean()+m.brier_top10.mean())
    res.append((hp,tot)); print(hp,[round(x,4) for x in tot],flush=True)
