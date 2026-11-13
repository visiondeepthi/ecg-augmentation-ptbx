"""Step 2: baseline 1D-ResNet. Folds 1-8 train, 9 val, 10 test (official PTB-XL split).
Usage: python 02_train_baseline.py /path/to/output_dir
Needs: torch, scikit-learn
"""
import sys, os, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

d = sys.argv[1]; dev = "cuda" if torch.cuda.is_available() else "cpu"
X = np.load(f"{d}/X100.npy"); y = np.load(f"{d}/y.npy"); folds = np.load(f"{d}/folds.npy")
mu, sd = X[folds <= 8].mean((0, 1)), X[folds <= 8].std((0, 1)) + 1e-6   # train-only stats
X = ((X - mu) / sd).transpose(0, 2, 1).astype(np.float32)               # (N,12,1000)
tr, va, te = folds <= 8, folds == 9, folds == 10
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

class Block(nn.Module):
    def __init__(s, i, o, stride):
        super().__init__()
        s.c1 = nn.Conv1d(i, o, 7, stride, 3, bias=False); s.b1 = nn.BatchNorm1d(o)
        s.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False);      s.b2 = nn.BatchNorm1d(o)
        s.sk = nn.Sequential(nn.Conv1d(i, o, 1, stride, bias=False), nn.BatchNorm1d(o)) if (i != o or stride != 1) else nn.Identity()
    def forward(s, x):
        h = torch.relu(s.b1(s.c1(x))); h = s.b2(s.c2(h)); return torch.relu(h + s.sk(x))
net = nn.Sequential(nn.Conv1d(12, 32, 15, 2, 7), nn.BatchNorm1d(32), nn.ReLU(),
                    Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2), Block(128, 256, 2),
                    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 5)).to(dev)

def predict(Xs):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 512):
            out.append(torch.sigmoid(net(torch.from_numpy(Xs[i:i+512]).to(dev))).cpu().numpy())
    return np.concatenate(out)

opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
lossf = nn.BCEWithLogitsLoss(); Xtr, ytr = X[tr], y[tr]; best, bad = 0, 0
torch.manual_seed(0); np.random.seed(0)
for ep in range(40):
    net.train(); idx = np.random.permutation(len(Xtr))
    for i in range(0, len(idx), 128):
        b = idx[i:i+128]
        opt.zero_grad(); l = lossf(net(torch.from_numpy(Xtr[b]).to(dev)), torch.from_numpy(ytr[b]).to(dev)); l.backward(); opt.step()
    auc = roc_auc_score(y[va], predict(X[va]), average="macro")
    print(f"epoch {ep+1} val macro-AUROC {auc:.4f}", flush=True)
    if auc > best: best, bad = auc, 0; torch.save(net.state_dict(), f"{d}/best.pt")
    else:
        bad += 1
        if bad >= 6: break

net.load_state_dict(torch.load(f"{d}/best.pt")); p = predict(X[te]); yt = y[te]
rng = np.random.default_rng(0); B = []
for _ in range(1000):                                  # bootstrap CI on test fold
    i = rng.integers(0, len(yt), len(yt))
    if (yt[i].sum(0) == 0).any(): continue
    B.append([roc_auc_score(yt[i][:, k], p[i][:, k]) for k in range(5)])
B = np.array(B)
print("\nTEST (fold 10)  macro-AUROC %.4f" % roc_auc_score(yt, p, average="macro"))
for k, n in enumerate(SUPER):
    print(f"  {n:5s} AUROC {roc_auc_score(yt[:,k], p[:,k]):.3f}  95% CI [{np.percentile(B[:,k],2.5):.3f}, {np.percentile(B[:,k],97.5):.3f}]  F1 {f1_score(yt[:,k], p[:,k]>0.5):.3f}")
np.save(f"{d}/test_probs.npy", p)
