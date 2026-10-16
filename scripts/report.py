import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from backtest import *
runs, base = pickle.load(open("outputs/final_runs.pkl","rb"))
M = {k: per_race_metrics(v) for k,v in runs.items()}
M["grid_freq"] = per_race_metrics(base["grid_freq"], "order"); M["standings_freq"] = per_race_metrics(base["standings_freq"], "order")
u = base["uniform"].assign(exp_pos=11.0); M["uniform"] = per_race_metrics(u)
print(f"{len(M['post_gbm'])} test races")
cols=["brier_win","brier_podium","brier_top10","winner_picked","podium_hits","top10_hits","spearman"]
print(pd.DataFrame({k:[m[c].mean() for c in cols] for k,m in M.items()}, index=cols).T.round(4).to_string())
def d(a,b,c): 
    m,lo,hi = bootstrap_diff(M[a][c],M[b][c]); return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"
print("\nPAIRED DIFFERENCES (first minus second; negative Brier = first better), 90% CI over races")
for a,b in [("post_gbm","grid_freq"),("pre_gbm","standings_freq"),("post_gbm","post_ridge"),("pre_gbm","pre_ridge"),("post_gbm","post_gbm_constDNF"),
            ("post_gbm","post_gbm_noq1pen"),("post_gbm","post_gbm_gridonly"),("pre_gbm","pre_gbm_noprev"),("pre_gbm","pre_gbm_keepcarry2026")]:
    print(a,"vs",b); 
    for c in ("brier_win","brier_podium","brier_top10"): print("   ",c,d(a,b,c))
print("\nBY SEASON (post_gbm / pre_gbm / grid_freq / standings_freq): brier_win, brier_podium, top10 hits")
for s in (2024,2025,2026):
    for k in ("post_gbm","pre_gbm","grid_freq","standings_freq"):
        m=M[k][M[k].Season==s]; print(s,k,round(m.brier_win.mean(),4),round(m.brier_podium.mean(),4),round(m.top10_hits.mean(),2),"winner",round(m.winner_picked.mean(),2),"n",len(m))
print("\nFIRST 5 RACES OF 2026 (carry-over blanked vs kept), brier_podium:")
for k in ("pre_gbm","pre_gbm_keepcarry2026","post_gbm","post_gbm_keepcarry2026","standings_freq","grid_freq"):
    m=M[k]; m=m[(m.Season==2026)].head(5); print(k, round(m.brier_win.mean(),4), round(m.brier_podium.mean(),4), round(m.top10_hits.mean(),2))
# calibration of post_gbm
for k in ("post_gbm","pre_gbm"):
    df=runs[k]; print("\ncalibration",k)
    for col,tg in (("p_win","won"),("p_podium","podium"),("p_top10","top10")):
        b=pd.cut(df[col],[0,.02,.05,.1,.2,.4,.7,1.0001],include_lowest=True); g=df.groupby(b,observed=True).agg(p=(col,"mean"),obs=(tg,"mean"),n=(tg,"size"))
        print(col); print(g.round(3).to_string())
df=runs["post_gbm"]; print("\nDNF: predicted mean",df.p_dnf.mean().round(3),"observed",df.dnf.mean().round(3))
b=pd.qcut(df.p_dnf,5,duplicates="drop"); print(df.groupby(b,observed=True).agg(p=("p_dnf","mean"),obs=("dnf","mean")).round(3))
