"""Step 7: formal tests of subgroup differences (sex, age) + calibration by subgroup. No GPU needed.
Usage: python 07_subgroup_tests.py /path/to/ptbxl_out [final_config=aug_constLR] [n_boot=2000]
Reads : meta.csv, y.npy, folds.npy, probs_<final_config>.npy
Writes: subgroup_tests.csv, subgroup_calibration.csv, subgroup_tests.md
Method: for each class (and macro), the AUROC difference between two subgroups is estimated with an UNPAIRED
bootstrap (records resampled within each subgroup independently); two-sided p = 2*min(P(d<=0), P(d>=0)).
Holm correction is applied across all tests. Pre-specified contrasts: women vs men; age >=80 vs age <80.
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

d = sys.argv[1]; final = sys.argv[2] if len(sys.argv) > 2 else "aug_constLR"; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
S = ["NORM", "MI", "STTC", "CD", "HYP"]; rng = np.random.default_rng(0)
meta = pd.read_csv(f"{d}/meta.csv"); y = np.load(f"{d}/y.npy"); folds = np.load(f"{d}/folds.npy")
te = folds == 10; yt = y[te]; p = np.load(f"{d}/probs_{final}.npy"); m = meta[te].reset_index(drop=True)
assert len(p) == len(yt) == len(m)
age = m.age.where(m.age <= 120, 90)
contrasts = {"Women vs men": (m.sex == 1).values, "Women vs men_ref": (m.sex == 0).values,
             "Age >=80 vs <80": (age >= 80).values, "Age >=80 vs <80_ref": (age < 80).values}
pairs = [("Women vs men", "Women vs men_ref"), ("Age >=80 vs <80", "Age >=80 vs <80_ref")]

from scipy.stats import rankdata
def aucs5(idx):
    """all five class AUROCs for the records in idx, via ranks (fast); nan if a class has no positives/negatives"""
    r = rankdata(p[idx], axis=0); yy = yt[idx]; npos = yy.sum(0); nneg = len(idx) - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ((r * yy).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    out[(npos == 0) | (nneg == 0)] = np.nan
    return out
def ece(pr, yy, bins=10):                          # equal-mass bins
    o = np.argsort(pr); chunks = np.array_split(o, bins)
    return sum(len(c) / len(pr) * abs(pr[c].mean() - yy[c].mean()) for c in chunks)

rows = []
for name, ref in pairs:
    A, B = np.flatnonzero(contrasts[name]), np.flatnonzero(contrasts[ref])
    a0, b0 = aucs5(A), aucs5(B)
    D = np.full((NB, 6), np.nan)
    for j in range(NB):
        va, vb = aucs5(rng.choice(A, len(A))), aucs5(rng.choice(B, len(B)))
        D[j, :5] = va - vb; D[j, 5] = np.nanmean(va) - np.nanmean(vb)
        if j % 500 == 0: print(f"  {name}: bootstrap {j}/{NB}", flush=True)
    for k in range(6):
        diffs = D[~np.isnan(D[:, k]), k]
        g0 = np.nanmean(a0) if k == 5 else a0[k]; r0 = np.nanmean(b0) if k == 5 else b0[k]
        pv = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) + 1 / len(diffs))
        rows.append(dict(contrast=name, cls="macro" if k == 5 else S[k], n_group=len(A), n_ref=len(B), auc_group=g0, auc_ref=r0,
                         diff=g0 - r0, ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5), p=pv))
t = pd.DataFrame(rows)
order = np.argsort(t.p.values); holm = np.empty(len(t)); run = 0                 # Holm step-down over all tests
for rank, i in enumerate(order): run = max(run, min(1.0, t.p.values[i] * (len(t) - rank))); holm[i] = run
t["p_holm"] = holm; t.to_csv(f"{d}/subgroup_tests.csv", index=False)

cal = []
for gname, mask in [("Men", m.sex == 0), ("Women", m.sex == 1), ("Age <80", age < 80), ("Age >=80", age >= 80)]:
    i = np.flatnonzero(mask.values)
    cal.append(dict(group=gname, n=len(i), brier_macro=np.mean([np.mean((p[i, k] - yt[i, k]) ** 2) for k in range(5)]),
                    ece_macro=np.mean([ece(p[i, k], yt[i, k]) for k in range(5)]),
                    **{f"prev_{c}": yt[i, k].mean() for k, c in enumerate(S)}))
cal = pd.DataFrame(cal); cal.to_csv(f"{d}/subgroup_calibration.csv", index=False)

fp = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
md = ["## Formal subgroup tests (test fold, ensemble of " + final + ")\n",
      "| Contrast | Class | AUROC group | AUROC reference | Difference (95% CI) | p | p (Holm) |", "| --- | --- | --- | --- | --- | --- | --- |"]
for _, r in t.iterrows():
    md.append(f"| {r.contrast} | {r.cls} | {r.auc_group:.3f} | {r.auc_ref:.3f} | {r['diff']:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f}) | {fp(r.p)} | {fp(r.p_holm)} |")
md += ["", "## Calibration and class prevalence by subgroup\n", "| Group | n | Brier (macro) | ECE (macro) | " + " | ".join(f"Prev. {c}" for c in S) + " |", "| --- | --- | --- | --- | " + " | ".join(["---"] * 5) + " |"]
for _, r in cal.iterrows(): md.append(f"| {r.group} | {r.n} | {r.brier_macro:.4f} | {r.ece_macro:.4f} | " + " | ".join(f"{r[f'prev_{c}']:.3f}" for c in S) + " |")
open(f"{d}/subgroup_tests.md", "w").write("\n".join(md)); print("\n".join(md)); print("\nsaved subgroup_tests.csv/.md, subgroup_calibration.csv in", d)
