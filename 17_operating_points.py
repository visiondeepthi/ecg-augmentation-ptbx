"""Step 17: clinical operating points from saved test-fold ensemble probabilities (CPU only, seconds).
Usage: python3 17_operating_points.py <ptbxl_out> [n_boot=1000]
Reads: y.npy, folds.npy, probs_aug_constLR.npy, probs_baseline_constLR.npy (test fold 10)
Writes: operating_points.csv, operating_points.md
NOTE: operating points are read from the test-fold ROC curve itself (thresholds are NOT chosen on the validation fold), so they are optimistic.
"""
import sys, numpy as np, pandas as pd
d = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
y = np.load(f"{d}/y.npy"); folds = np.load(f"{d}/folds.npy"); yt = y[folds == 10].astype(int)
rng = np.random.default_rng(0)

def points(yy, p):
    o = np.argsort(-p, kind="stable"); ys = yy[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P, N = ys.sum(), len(ys) - ys.sum()
    sens = np.r_[0, tp / P]; spec = np.r_[1, 1 - fp / N]; tpv = np.r_[0, tp]; fpv = np.r_[0, fp]
    out = {}
    for s in (0.90, 0.95):
        ok = np.where(spec >= s)[0]; i = ok[np.argmax(sens[ok])]            # best sensitivity with specificity >= s
        out[f"sens@spec{int(s*100)}"] = sens[i]
    for s in (0.90, 0.95):
        ok = np.where(sens >= s)[0]; i = ok[np.argmax(spec[ok])]            # best specificity with sensitivity >= s
        out[f"spec@sens{int(s*100)}"] = spec[i]
        if s == 0.90: out["ppv@sens90"] = tpv[i] / max(tpv[i] + fpv[i], 1); out["npv@sens90"] = (N - fpv[i]) / max((N - fpv[i]) + (P - tpv[i]), 1)
    return out

rows = []
for model in ["baseline_constLR", "aug_constLR"]:
    p = np.load(f"{d}/probs_{model}.npy"); assert p.shape == yt.shape, (p.shape, yt.shape)
    for k, n in enumerate(SUPER):
        est = points(yt[:, k], p[:, k]); boots = {m: [] for m in est}
        for _ in range(NB):
            i = rng.integers(0, len(yt), len(yt))
            if yt[i, k].sum() in (0, len(i)): continue
            for m, v in points(yt[i, k], p[i, k]).items(): boots[m].append(v)
        r = dict(model=model, cls=n, prevalence=yt[:, k].mean())
        for m, v in est.items(): r[m] = v; r[m + "_lo"] = np.percentile(boots[m], 2.5); r[m + "_hi"] = np.percentile(boots[m], 97.5)
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"{d}/operating_points.csv", index=False)
f = lambda r, m: f"{r[m]:.2f} ({r[m+'_lo']:.2f}-{r[m+'_hi']:.2f})"
out = ["## Operating points on PTB-XL test fold (ensembles; read from the test ROC, optimistic)\n",
       "| Model | Class | Prevalence | Sens @ spec 90% | Sens @ spec 95% | Spec @ sens 90% | Spec @ sens 95% | PPV @ sens 90% | NPV @ sens 90% |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    out.append(f"| {r.model} | {r.cls} | {r.prevalence:.2f} | {f(r,'sens@spec90')} | {f(r,'sens@spec95')} | {f(r,'spec@sens90')} | {f(r,'spec@sens95')} | {f(r,'ppv@sens90')} | {f(r,'npv@sens90')} |")
open(f"{d}/operating_points.md", "w").write("\n".join(out)); print("\n".join(out))
