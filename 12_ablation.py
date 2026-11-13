"""Step 12: per-transformation augmentation ablation (internal PTB-XL test + external Chapman), 5 seeds each.
Usage: python3 12_ablation.py <ptbxl_out> <chapman_out> [seeds=5] [max_epochs=40]
Needs chapman_out/chapman_X100.npy, chapman_X100_records.txt, chapman_X100_bad.txt, chapman_labels.csv from step 11.
Outputs: ptbxl_out/ablation_runs.csv, ablation_summary.csv
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

d = sys.argv[1]; CH = sys.argv[2]; SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAXEP = int(sys.argv[4]) if len(sys.argv) > 4 else 40
dev = "cuda" if torch.cuda.is_available() else "cpu"
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

X = np.load(f"{d}/X100.npy"); y = np.load(f"{d}/y.npy"); folds = np.load(f"{d}/folds.npy")
mu, sd = X[folds <= 8].mean((0, 1)), X[folds <= 8].std((0, 1)) + 1e-6      # train-only statistics
X = ((X - mu) / sd).transpose(0, 2, 1).astype(np.float32)
tr, va, te = folds <= 8, folds == 9, folds == 10
Xtr = torch.from_numpy(X[tr]).to(dev); ytr = torch.from_numpy(y[tr]).to(dev)

# name -> (width, dropout, weight_decay, augment, label_smoothing)
ALL = ("amp", "wander", "noise", "shift")
def C(tf): return dict(w=32, drop=0.3, wd=1e-2, aug=len(tf) > 0, ls=0.0, sched="const", pat=6, tf=tuple(tf))
CONFIGS = {"none": C(()), "all": C(ALL),
           "only_amp": C(("amp",)), "only_wander": C(("wander",)), "only_noise": C(("noise",)), "only_shift": C(("shift",)),
           "no_amp": C([t for t in ALL if t != "amp"]), "no_wander": C([t for t in ALL if t != "wander"]),
           "no_noise": C([t for t in ALL if t != "noise"]), "no_shift": C([t for t in ALL if t != "shift"])}

# ---- external data (cached by step 11) ----
from scipy.stats import rankdata
EXT = ["NORM", "STTC", "CD", "HYP"]
Xe = np.load(f"{CH}/chapman_X100.npy"); recs = open(f"{CH}/chapman_X100_records.txt").read().split()
bad = set(open(f"{CH}/chapman_X100_bad.txt").read().split()) if os.path.exists(f"{CH}/chapman_X100_bad.txt") else set()
labe = pd.read_csv(f"{CH}/chapman_labels.csv"); assert list(labe["record"]) == recs
okm = ~np.isin(recs, list(bad)); Xe, labe = Xe[okm], labe[okm].reset_index(drop=True)
Xe = np.ascontiguousarray(((Xe.transpose(0, 2, 1) - mu) / sd).transpose(0, 2, 1).astype(np.float32))
def fast_auc(yy, p):
    r = rankdata(p); n1 = yy.sum(); n0 = len(yy) - n1
    return np.nan if n1 == 0 or n0 == 0 else (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def augment(x, tf):                                # x: (B,12,1000) on device; tf = enabled transformations
    B = x.shape[0]
    if "amp" in tf: x = x * (1 + 0.1 * torch.randn(B, 12, 1, device=x.device))
    if "wander" in tf:
        t = torch.linspace(0, 1, x.shape[2], device=x.device)
        f = torch.rand(B, 1, 1, device=x.device) * 0.5 + 0.1
        x = x + 0.1 * torch.randn(B, 1, 1, device=x.device) * torch.sin(2 * np.pi * f * t + 6.28 * torch.rand(B, 1, 1, device=x.device))
    if "noise" in tf: x = x + 0.03 * torch.randn_like(x)
    if "shift" in tf: x = torch.roll(x, np.random.randint(-50, 51), dims=2)
    return x

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

def best_thresholds(yv, pv):                       # per-class threshold maximising F1 on VALIDATION fold
    ths = []
    for k in range(5):
        c = np.linspace(0.05, 0.95, 91)
        ths.append(c[np.argmax([f1_score(yv[:, k], pv[:, k] > t, zero_division=0) for t in c])])
    return np.array(ths)

def run(cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net(cfg["w"], cfg["drop"])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=cfg["wd"])
    steps = MAXEP * int(np.ceil(len(Xtr) / 128)) + 1   # +1: OneCycleLR errors if stepped exactly total_steps times
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps) if cfg["sched"] == "onecycle" else None
    lossf = nn.BCEWithLogitsLoss(); best, bad, state = 0, 0, None
    for ep in range(MAXEP):
        net.train(); idx = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(idx), 128):
            b = idx[i:i+128]; xb, yb = Xtr[b], ytr[b]
            if cfg["aug"]: xb = augment(xb, cfg["tf"])
            if cfg["ls"]: yb = yb * (1 - cfg["ls"]) + 0.5 * cfg["ls"]
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
            if sched is not None: sched.step()
        auc = roc_auc_score(y[va], predict(net, X[va]), average="macro")
        if auc > best: best, bad, state = auc, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["pat"]: break
    net.load_state_dict(state)
    pv, pt = predict(net, X[va]), predict(net, X[te]); th = best_thresholds(y[va], pv); yt = y[te]
    r = dict(val_auc=best, test_macro_auc=roc_auc_score(yt, pt, average="macro"), epochs=ep + 1)
    for k, n in enumerate(SUPER):
        pred = pt[:, k] > th[k]; tp = (pred & (yt[:, k] == 1)).sum(); tn = (~pred & (yt[:, k] == 0)).sum()
        r[f"auc_{n}"] = roc_auc_score(yt[:, k], pt[:, k]); r[f"f1_{n}"] = f1_score(yt[:, k], pred)
        r[f"sens_{n}"] = tp / max(yt[:, k].sum(), 1); r[f"spec_{n}"] = tn / max((yt[:, k] == 0).sum(), 1)
    r["macro_f1"] = np.mean([r[f"f1_{n}"] for n in SUPER])
    pe = predict(net, Xe)
    for k in EXT:
        ok = labe[k].notna().values; r[f"ext_{k}"] = fast_auc(labe.loc[ok, k].values.astype(int), pe[ok, SUPER.index(k)])
    r["ext_macro"] = np.mean([r[f"ext_{k}"] for k in EXT])
    return r, pt

rows = []
for name, cfg in CONFIGS.items():
    probs = []
    for s in range(SEEDS):
        t0 = time.time(); r, pt = run(cfg, s); r.update(config=name, seed=s); rows.append(r); probs.append(pt)
        print(f"{name:12s} seed {s}  internal {r['test_macro_auc']:.4f}  external {r['ext_macro']:.4f}  ({r['epochs']} ep, {time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(f"{d}/ablation_runs.csv", index=False)
df = pd.DataFrame(rows); cols = ["test_macro_auc", "ext_macro"] + [f"ext_{k}" for k in EXT]
summ = df.groupby("config", sort=False)[cols].agg(["mean", "std"]).round(4)
summ.to_csv(f"{d}/ablation_summary.csv"); print("\n", summ.to_string())
