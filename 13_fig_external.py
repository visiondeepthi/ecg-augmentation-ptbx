import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [("NORM",.956,.953,.959,.960,.957,.962),("STTC",.864,.860,.869,.871,.867,.875),
        ("CD",.844,.838,.851,.841,.834,.849),("HYP",.988,.985,.991,.984,.981,.988)]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
for i,(c,b,bl,bh,a,al,ah) in enumerate(rows):
    y = len(rows)-1-i
    ax.errorbar(b, y+0.14, xerr=[[b-bl],[bh-b]], fmt="o", color="#6b7280", capsize=3, label="Baseline" if i==0 else None)
    ax.errorbar(a, y-0.14, xerr=[[a-al],[ah-a]], fmt="s", color="#1d4ed8", capsize=3, label="Augmented" if i==0 else None)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0.82, 1.0); ax.set_xlabel("External AUROC (Chapman–Shaoxing/Ningbo), 95% CI")
ax.grid(axis="x", alpha=.3); ax.legend(frameon=False, loc="lower left")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_external.png")
