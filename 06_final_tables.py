"""Step 6: final tables + subgroup / label-quality analysis. No GPU needed.
Usage: python 06_final_tables.py /path/to/ptbxl_out [final_config]
 - final_config: optional; default = config with the highest mean VALIDATION macro-AUROC
Reads : meta.csv, y.npy, folds.npy, results_runs.csv, results_runs_fair.csv, probs_<config>.npy
Writes: final_tables.md (paste into the manuscript) and CSVs: per_class_by_config.csv, subgroup_results.csv
"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

d = sys.argv[1]; SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]
meta = pd.read_csv(f"{d}/meta.csv"); y = np.load(f"{d}/y.npy"); folds = np.load(f"{d}/folds.npy")
assert len(meta) == len(y) == len(folds), "meta/y/folds length mismatch"
runs = pd.concat([pd.read_csv(f"{d}/{f}") for f in ["results_runs.csv", "results_runs_fair.csv"] if os.path.exists(f"{d}/{f}")], ignore_index=True)
md = []; P = lambda s="": (print(s), md.append(s))

# ---- 1. final model chosen on VALIDATION fold ----
val = runs.groupby("config", sort=False).val_auc.agg(["mean", "std"])
final = sys.argv[2] if len(sys.argv) > 2 else val["mean"].idxmax()
P("## Model selection on the validation fold (fold 9)\n")
P("| Configuration | Validation macro-AUROC (mean ± SD) |\n| --- | --- |")
for c, r in val.iterrows(): P(f"| {c}{' (selected)' if c == final else ''} | {r['mean']:.4f} ± {r['std']:.4f} |")

# ---- 2. per-class means over seeds for every configuration ----
P("\n## Per-class test-fold results, mean ± SD over 5 seeds\n")
rows = []
for c, g in runs.groupby("config", sort=False):
    for m in ["auc", "f1", "sens", "spec"]:
        for k in SUPER:
            col = f"{m}_{k}"
            if col in g: rows.append(dict(config=c, metric=m, cls=k, mean=g[col].mean(), sd=g[col].std()))
pc = pd.DataFrame(rows); pc.to_csv(f"{d}/per_class_by_config.csv", index=False)
for m, title in [("auc", "AUROC"), ("f1", "F1 at validation-tuned thresholds"), ("sens", "Sensitivity"), ("spec", "Specificity")]:
    t = pc[pc.metric == m]
    P(f"\n**{title}**\n\n| Configuration | " + " | ".join(SUPER) + " |\n| --- | " + " | ".join(["---"] * 5) + " |")
    for c in runs.config.unique():
        r = t[t.config == c].set_index("cls")
        P(f"| {c} | " + " | ".join(f"{r.loc[k,'mean']:.3f} ± {r.loc[k,'sd']:.3f}" for k in SUPER) + " |")

# ---- 3. cohort description (for Methods 2.1 / 2.2) ----
meta["age_c"] = meta.age.where(meta.age <= 120, 90)       # PTB-XL codes ages >89 as 300
meta["fold"] = folds
P("\n## Cohort description\n")
P("| Fold | Records | Patients | Age, median (IQR) | Female % | Human-validated % |\n| --- | --- | --- | --- | --- | --- |")
for name, mask in [("Train (1-8)", folds <= 8), ("Validation (9)", folds == 9), ("Test (10)", folds == 10), ("All", folds > 0)]:
    s = meta[mask]
    P(f"| {name} | {len(s)} | {s.patient_id.nunique()} | {s.age_c.median():.0f} ({s.age_c.quantile(.25):.0f}-{s.age_c.quantile(.75):.0f}) | {100*(s.sex == 1).mean():.1f} | {100*s.validated_by_human.astype(bool).mean():.1f} |")

# ---- 4. subgroup + label-quality analysis on the test fold, final-model seed ensemble ----
pf = f"{d}/probs_{final}.npy"
if os.path.exists(pf):
    p = np.load(pf); te = folds == 10; yt = y[te]; mt = meta[te].reset_index(drop=True)
    assert len(p) == len(yt), "probs length differs from test fold"
    groups = {"Sex: male": mt.sex == 0, "Sex: female": mt.sex == 1,
              "Age < 40": mt.age_c < 40, "Age 40-59": (mt.age_c >= 40) & (mt.age_c < 60),
              "Age 60-79": (mt.age_c >= 60) & (mt.age_c < 80), "Age >= 80": mt.age_c >= 80,
              "Labels human-validated": mt.validated_by_human.astype(bool), "Labels not human-validated": ~mt.validated_by_human.astype(bool)}
    rng = np.random.default_rng(0); out = []
    for name, m in groups.items():
        m = np.asarray(m); row = dict(group=name, n=int(m.sum()))
        aucs = []
        for k, c in enumerate(SUPER):
            pos = int(yt[m][:, k].sum())
            if pos >= 10 and (m.sum() - pos) >= 10:
                a = roc_auc_score(yt[m][:, k], p[m][:, k]); aucs.append(a); row[c] = a
            else: row[c] = np.nan                    # too few cases for a stable estimate
        row["macro"] = np.mean(aucs) if len(aucs) == 5 else np.nan
        if len(aucs) == 5:                           # bootstrap CI of macro-AUROC
            idx = np.flatnonzero(m); B = []
            for _ in range(500):
                i = rng.choice(idx, len(idx))
                try: B.append(np.mean([roc_auc_score(yt[i][:, k], p[i][:, k]) for k in range(5)]))
                except ValueError: pass
            row["ci_lo"], row["ci_hi"] = np.percentile(B, [2.5, 97.5])
        out.append(row)
    sg = pd.DataFrame(out); sg.to_csv(f"{d}/subgroup_results.csv", index=False)
    P(f"\n## Subgroup and label-quality analysis, test fold, ensemble of {final}\n")
    P("| Subgroup | n | NORM | MI | STTC | CD | HYP | Macro (95% CI) |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    f = lambda v: "n/a (few cases)" if pd.isna(v) else f"{v:.3f}"
    for _, r in sg.iterrows():
        ci = f"{f(r['macro'])} ({r['ci_lo']:.3f}-{r['ci_hi']:.3f})" if "ci_lo" in r and not pd.isna(r.get("ci_lo")) else f(r["macro"])
        P(f"| {r['group']} | {r['n']} | " + " | ".join(f(r[c]) for c in SUPER) + f" | {ci} |")
else:
    P(f"\n(No {pf}; subgroup analysis skipped.)")
open(f"{d}/final_tables.md", "w").write("\n".join(md)); print("\nsaved final_tables.md, per_class_by_config.csv, subgroup_results.csv in", d)
