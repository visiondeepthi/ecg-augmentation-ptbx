"""Step 11: EXTERNAL validation of PTB-XL-trained models on Chapman-Shaoxing/Ningbo (no retraining, no recalibration).
Usage: python3 11_external_chapman.py <ptbxl_out> <chapman_out> <chapman_signal_dir> [n_boot=1000]
 - <chapman_signal_dir>: folder where the .mat files were extracted (searched recursively by record name)
Outputs (in chapman_out/): chapman_X100.npy cache, chapman_labels.csv, ext_probs_<config>.npy, external_results.csv, external_results.md
"""
import os, sys, glob, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import rankdata, mannwhitneyu

pt_dir, ch_dir, sig_dir = sys.argv[1], sys.argv[2], sys.argv[3]
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
dev = "cuda" if torch.cuda.is_available() else "cpu"
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]; EVAL = ["NORM", "STTC", "CD", "HYP"]       # MI cannot be mapped
CONFIGS = ["baseline_constLR", "aug_constLR"]

# ---------------- label mapping (see 09_chapman_mapping_counts.py) ----------------
SINUS = {"426783006"}
STRICT = {"CD": {"59118001", "698252002", "164909002", "270492004", "233917008"},
          "STTC": {"164934002", "59931005", "429622005", "428750005", "164930006"},
          "HYP": {"164873001"}}
EXTRA = {"CD": {"713427006", "713426002", "733534002", "445118002", "6374002"},   # UNVERIFIED code meanings:
         "STTC": {"55930002"}, "HYP": {"55827005"}}                                # records carrying them are excluded as negatives

m = pd.read_csv(f"{ch_dir}/chapman_meta.csv", dtype=str).fillna("")
m = m[m["n"].astype(str) == "5000"].reset_index(drop=True)          # 10 s at 500 Hz; drops malformed headers
codes = m["dx"].str.split(";").apply(lambda l: set(c.strip() for c in l if c.strip()))
any_strict = codes.apply(lambda s: any(s & v for v in STRICT.values()))
lab = pd.DataFrame(index=m.index, columns=EVAL, dtype=float)         # 1 pos, 0 neg, NaN excluded
lab["NORM"] = np.where(codes.apply(lambda s: s <= SINUS and len(s) > 0), 1.0, np.where(any_strict, 0.0, np.nan))
for k in ["CD", "STTC", "HYP"]:
    pos = codes.apply(lambda s: bool(s & STRICT[k])); extra = codes.apply(lambda s: bool(s & EXTRA[k]))
    lab[k] = np.where(pos, 1.0, np.where(extra, np.nan, 0.0))
keep = lab.notna().any(axis=1).values
m, lab = m[keep].reset_index(drop=True), lab[keep].reset_index(drop=True)
print("records used:", len(m)); print(lab.apply(lambda c: pd.Series({"pos": int((c == 1).sum()), "neg": int((c == 0).sum())})).T)
lab.assign(record=m["record"]).to_csv(f"{ch_dir}/chapman_labels.csv", index=False)

# ---------------- signals ----------------
cache = f"{ch_dir}/chapman_X100.npy"; names_f = f"{ch_dir}/chapman_X100_records.txt"
if os.path.exists(cache) and open(names_f).read().split() == list(m["record"]):
    X = np.load(cache)
else:
    index = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{sig_dir}/**/*.mat", recursive=True)}
    print("mat files found:", len(index)); X = np.zeros((len(m), 12, 1000), np.float32); bad = []
    for i, r in enumerate(m["record"]):
        try:
            a = loadmat(index[r])["val"].astype(np.float64)
            assert a.shape == (12, 5000)
            X[i] = resample_poly(a / 1000.0, 1, 5, axis=1)          # gain 1000/mV -> mV ; 500 Hz -> 100 Hz
        except Exception as e:
            bad.append(r)
        if i % 2000 == 0: print("loaded", i, flush=True)
    if bad: print("UNREADABLE records (will be skipped):", len(bad), bad[:10])
    np.save(cache, X); open(names_f, "w").write("\n".join(m["record"])); open(f"{ch_dir}/chapman_X100_bad.txt", "w").write("\n".join(bad))
bad_f = f"{ch_dir}/chapman_X100_bad.txt"; bad = set(open(bad_f).read().split()) if os.path.exists(bad_f) else set()
if bad:
    okm = ~m["record"].isin(bad).values; X, m, lab = X[okm], m[okm].reset_index(drop=True), lab[okm].reset_index(drop=True)
    print("records after skipping unreadable:", len(m))
print("Chapman amplitude (mV) median|x| = %.3f, std = %.3f" % (np.median(np.abs(X)), X.std()))

# ---------------- model (identical to 04/10) ----------------
class Block(nn.Module):
    def __init__(s, i, o, stride):
        super().__init__()
        s.c1 = nn.Conv1d(i, o, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(o)
        s.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False);      s.b2 = nn.BatchNorm1d(o)
        s.sk = nn.Sequential(nn.Conv1d(i, o, 1, stride, bias=False), nn.BatchNorm1d(o)) if (i != o or stride != 1) else nn.Identity()
    def forward(s, x):
        h = torch.relu(s.b1(s.c1(x))); h = s.b2(s.c2(h)); return torch.relu(h + s.sk(x))
def make_net(w, drop):
    return nn.Sequential(nn.Conv1d(12, w, 15, 2, 7), nn.BatchNorm1d(w), nn.ReLU(),
        Block(w, 2*w, 2), Block(2*w, 4*w, 2), Block(4*w, 4*w, 2), Block(4*w, 8*w, 2),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(8*w, 5)).to(dev)

def predict(net, Xs):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 512):
            out.append(torch.sigmoid(net(torch.from_numpy(Xs[i:i+512]).to(dev))).cpu().numpy())
    return np.concatenate(out)

P = {}                                                              # config -> (seeds, N, 5)
for c in CONFIGS:
    files = sorted(glob.glob(f"{pt_dir}/models/{c}_seed*.pt")); assert files, f"no saved models for {c}"
    ps = []
    for f in files:
        ck = torch.load(f, map_location=dev)
        net = make_net(ck["cfg"]["w"], ck["cfg"]["drop"]); net.load_state_dict(ck["state"])
        mu, sd = np.array(ck["mu"], np.float32), np.array(ck["sd"], np.float32)       # PTB-XL training statistics
        Xn = ((X.transpose(0, 2, 1) - mu) / sd).transpose(0, 2, 1).astype(np.float32)
        ps.append(predict(net, np.ascontiguousarray(Xn)))
    P[c] = np.stack(ps); np.save(f"{ch_dir}/ext_probs_{c}.npy", P[c].mean(0)); print(c, "models:", len(files))

# ---------------- metrics ----------------
def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(0)
rows, md = [], []
ens = {c: P[c].mean(0) for c in CONFIGS}
for k in EVAL:
    j = SUPER.index(k); ok = lab[k].notna().values; yk = lab.loc[ok, k].values.astype(int)
    pb, pa = ens["baseline_constLR"][ok, j], ens["aug_constLR"][ok, j]
    ab, aa = auc(yk, pb), auc(yk, pa); bd, bb, ba = [], [], []
    for _ in range(NB):
        i = rng.integers(0, len(yk), len(yk)); x, q, w = yk[i], pb[i], pa[i]
        if x.sum() == 0 or x.sum() == len(x): continue
        u, v = auc(x, q), auc(x, w); bb.append(u); ba.append(v); bd.append(v - u)
    ci = lambda a: f"{np.percentile(a,2.5):.3f}-{np.percentile(a,97.5):.3f}"
    seed_b = [auc(yk, P["baseline_constLR"][s][ok, j]) for s in range(len(P["baseline_constLR"]))]
    seed_a = [auc(yk, P["aug_constLR"][s][ok, j]) for s in range(len(P["aug_constLR"]))]
    rows.append(dict(cls=k, n_pos=int(yk.sum()), n_neg=int(len(yk) - yk.sum()), auc_baseline=ab, ci_baseline=ci(bb),
                     auc_aug=aa, ci_aug=ci(ba), diff=aa - ab, diff_ci=f"{np.percentile(bd,2.5):+.3f} to {np.percentile(bd,97.5):+.3f}",
                     seed_baseline=f"{np.mean(seed_b):.3f}+-{np.std(seed_b, ddof=1):.3f}", seed_aug=f"{np.mean(seed_a):.3f}+-{np.std(seed_a, ddof=1):.3f}"))
df = pd.DataFrame(rows); df.to_csv(f"{ch_dir}/external_results.csv", index=False)
mac_b, mac_a = df.auc_baseline.mean(), df.auc_aug.mean()
sb = [np.mean([auc(lab.loc[lab[k].notna(), k].values.astype(int), P["baseline_constLR"][s][lab[k].notna().values, SUPER.index(k)]) for k in EVAL]) for s in range(len(P["baseline_constLR"]))]
sa = [np.mean([auc(lab.loc[lab[k].notna(), k].values.astype(int), P["aug_constLR"][s][lab[k].notna().values, SUPER.index(k)]) for k in EVAL]) for s in range(len(P["aug_constLR"]))]
pu = mannwhitneyu(sa, sb, alternative="two-sided").pvalue
out = ["## External validation on Chapman-Shaoxing/Ningbo (PTB-XL-trained, no retraining)\n",
       "| Class | Pos | Neg | AUROC baseline (95% CI) | AUROC augmented (95% CI) | Diff aug-base (95% CI) | Seed mean+-SD base | Seed mean+-SD aug |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['cls']} | {r['n_pos']} | {r['n_neg']} | {r['auc_baseline']:.3f} ({r['ci_baseline']}) | {r['auc_aug']:.3f} ({r['ci_aug']}) | {r['diff']:+.3f} ({r['diff_ci']}) | {r['seed_baseline']} | {r['seed_aug']} |")
out.append(f"\nMacro-AUROC over {len(EVAL)} mapped classes: baseline {mac_b:.4f} (seeds {np.mean(sb):.4f}+-{np.std(sb,ddof=1):.4f}), augmented {mac_a:.4f} (seeds {np.mean(sa):.4f}+-{np.std(sa,ddof=1):.4f}); seed-level Mann-Whitney p = {pu:.4f} (min attainable 0.0079).")
open(f"{ch_dir}/external_results.md", "w").write("\n".join(out)); print("\n".join(out))
