"""Step 4B: significance tests + paper figures. No GPU/torch needed.
Usage: python 05_stats_and_figures.py /path/to/ptbxl_out
Reads: y.npy, folds.npy, probs_<config>.npy, results_runs.csv, results_runs_fair.csv (whichever exist)
Writes: stats_report.txt, ensemble_per_class.csv, fig_roc.png, fig_calibration.png, fig_ablation.png
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

d = sys.argv[1]; SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
yt = np.load(f"{d}/y.npy")[np.load(f"{d}/folds.npy") == 10]
runs = pd.concat([pd.read_csv(f"{d}/{f}") for f in ["results_runs.csv", "results_runs_fair.csv"] if os.path.exists(f"{d}/{f}")])
P = {c: np.load(f"{d}/probs_{c}.npy") for c in runs.config.unique() if os.path.exists(f"{d}/probs_{c}.npy")}
out = []; log = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))

log("== Per-run results (5 seeds): macro-AUROC mean +- SD ==")
g = runs.groupby("config", sort=False).test_macro_auc
for c, v in g: log(f"{c:18s} {v.mean():.4f} +- {v.std():.4f}  (n={len(v)})")

log("\n== Pairwise tests on seed-level macro-AUROC (Welch t, Mann-Whitney U) ==")
cfgs = list(runs.config.unique()); ref = "baseline_constLR" if "baseline_constLR" in cfgs else cfgs[0]
for c in cfgs:
    if c == ref: continue
    a, b = runs[runs.config == c].test_macro_auc.values, runs[runs.config == ref].test_macro_auc.values
    log(f"{c} vs {ref}: diff {a.mean()-b.mean():+.4f}  Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.4g}  MWU p={stats.mannwhitneyu(a, b).pvalue:.4g}")

log("\n== Seed-ensemble on fold 10: macro-AUROC, and paired bootstrap difference vs", ref, "==")
rng = np.random.default_rng(0); idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def macro(y, p): return roc_auc_score(y, p, average="macro")
for c, p in P.items(): log(f"{c:18s} ensemble macro-AUROC {macro(yt, p):.4f}")
if ref in P:
    for c, p in P.items():
        if c == ref: continue
        dif = []
        for i in idx:
            if (yt[i].sum(0) == 0).any() or (yt[i].sum(0) == len(i)).any(): continue
            dif.append(macro(yt[i], p[i]) - macro(yt[i], P[ref][i]))
        lo, hi = np.percentile(dif, [2.5, 97.5]); log(f"{c} - {ref}: {np.mean(dif):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]" + ("  (CI excludes 0)" if lo > 0 or hi < 0 else "  (CI includes 0)"))

best = max(P, key=lambda c: macro(yt, P[c])); pb = P[best]; log("\nBest ensemble:", best)
rows = []
for k, n in enumerate(SUPER):
    B = [roc_auc_score(yt[i][:, k], pb[i][:, k]) for i in idx if 0 < yt[i][:, k].sum() < len(i)]
    rows.append(dict(cls=n, auroc=roc_auc_score(yt[:, k], pb[:, k]), ci_lo=np.percentile(B, 2.5), ci_hi=np.percentile(B, 97.5),
                     brier=np.mean((pb[:, k] - yt[:, k]) ** 2)))
pd.DataFrame(rows).round(4).to_csv(f"{d}/ensemble_per_class.csv", index=False); log(pd.DataFrame(rows).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 5))
for k, n in enumerate(SUPER):
    f, t, _ = roc_curve(yt[:, k], pb[:, k]); ax.plot(f, t, label=f"{n} (AUC {rows[k]['auroc']:.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=f"ROC, PTB-XL fold 10 ({best}, seed ensemble)"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{d}/fig_roc.png", dpi=200)

fig, ax = plt.subplots(figsize=(5.5, 5))
for k, n in enumerate(SUPER):
    fp, mp = calibration_curve(yt[:, k], pb[:, k], n_bins=10, strategy="quantile"); ax.plot(mp, fp, "o-", ms=3, label=n)
ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency", title="Calibration"); ax.legend(); fig.tight_layout(); fig.savefig(f"{d}/fig_calibration.png", dpi=200)

m = runs.groupby("config", sort=False).test_macro_auc.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(6.5, 4)); ax.bar(range(len(m)), m["mean"], yerr=m["std"], capsize=4, color="#4a78b5")
ax.set_xticks(range(len(m))); ax.set_xticklabels(m.index, rotation=20, ha="right"); ax.set_ylim(m["mean"].min() - .01, m["mean"].max() + .01)
ax.set(ylabel="Test macro-AUROC (mean +- SD, 5 seeds)", title="Ablation"); fig.tight_layout(); fig.savefig(f"{d}/fig_ablation.png", dpi=200)
open(f"{d}/stats_report.txt", "w").write("\n".join(out)); print("\nsaved report and 3 figures in", d)
