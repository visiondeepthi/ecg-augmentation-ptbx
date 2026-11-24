"""Exact per-seed tests (Welch + Mann-Whitney, Holm within experiment). Usage: python3 19_exact_tests.py ../results"""
import pandas as pd, numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import sys; A=sys.argv[1].rstrip("/")+"/"
def holm(ps):
    o=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*ps[i]); adj[i]=min(1,run)
    return adj
def contrasts(df, pairs, metric, title):
    rows=[]
    for a,b in pairs:
        x=df[df.config==a].sort_values("seed")[metric].values; y=df[df.config==b].sort_values("seed")[metric].values
        assert len(x)==5 and len(y)==5,(a,b,len(x),len(y))
        tw=ttest_ind(x,y,equal_var=False); mw=mannwhitneyu(x,y,alternative="two-sided")
        rows.append(dict(a=a,b=b,mean_a=x.mean(),sd_a=x.std(ddof=1),mean_b=y.mean(),sd_b=y.std(ddof=1),diff=x.mean()-y.mean(),welch_p=tw.pvalue,mw_p=mw.pvalue))
    d=pd.DataFrame(rows); d["welch_holm"]=holm(d.welch_p.values)
    print("\n##",title,metric); print(d.round(4).to_string(index=False)); return d
ab=pd.read_csv(A+"ablation_runs.csv"); sh=pd.read_csv(A+"shift_runs.csv"); ar=pd.read_csv(A+"arch_runs.csv")
print(ab.groupby("config").size().to_dict(), sh.groupby("config").size().to_dict(), ar.groupby("config").size().to_dict())
cfgs=[c for c in ab.config.unique() if c!="none"]
for m in ["test_macro_auc","ext_macro"]:
    contrasts(ab,[(c,"none") for c in cfgs],m,"ablation vs none")
    contrasts(ab,[("only_shift","all"),("no_shift","all"),("no_shift","none")],m,"ablation extra")
for m in ["test_macro_auc","ext_macro"]:
    contrasts(sh,[(c,"none") for c in sh.config.unique() if c!="none"],m,"shift vs none")
    contrasts(sh,[("persample_50","batch_shift50"),("persample_100","batch_shift50")],m,"shift persample vs batch")
for m in ["test_macro_auc","ext_macro"]:
    contrasts(ar,[("gru_only_shift","gru_none"),("gru_all","gru_none"),("gru_all","gru_only_shift")],m,"arch")
