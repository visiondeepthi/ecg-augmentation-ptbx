"""Step 1: build arrays from PTB-XL (100 Hz) with diagnostic superclass labels.
Usage: python 01_prepare_ptbxl.py /path/to/ptb-xl-...-1.0.3 /path/to/output_dir
Needs: pip install wfdb pandas numpy tqdm
"""
import sys, ast, os
import numpy as np, pandas as pd, wfdb
from tqdm import tqdm

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
SUPER = ["NORM", "MI", "STTC", "CD", "HYP"]

df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
df["scp_codes"] = df.scp_codes.apply(ast.literal_eval)
agg = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
agg = agg[agg.diagnostic == 1]                      # diagnostic statements only

def to_super(codes):
    s = {agg.loc[c, "diagnostic_class"] for c in codes if c in agg.index}
    return sorted(s)

df["superclasses"] = df.scp_codes.apply(to_super)
y = np.array([[int(c in s) for c in SUPER] for s in df.superclasses], dtype=np.float32)
keep = y.sum(1) > 0                                  # drop records with no diagnostic label
print("records kept:", keep.sum(), "of", len(df))
print("class prevalence:", dict(zip(SUPER, y[keep].mean(0).round(3))))

X = np.zeros((len(df), 1000, 12), dtype=np.float32)
for i, f in enumerate(tqdm(df.filename_lr.values)):
    sig, _ = wfdb.rdsamp(os.path.join(root, f))
    X[i] = sig
X, y, folds = X[keep], y[keep], df.strat_fold.values[keep]
assert not np.isnan(X).any(), "NaNs in signals"
np.save(os.path.join(out, "X100.npy"), X)
np.save(os.path.join(out, "y.npy"), y)
np.save(os.path.join(out, "folds.npy"), folds)
df[keep][["patient_id", "age", "sex", "strat_fold", "validated_by_human", "report"]].to_csv(os.path.join(out, "meta.csv"))
print("saved", X.shape, y.shape)
