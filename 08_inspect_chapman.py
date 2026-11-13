"""Step 8: inspect the Chapman/Ningbo headers (metadata only, no signals). Fast.
Usage: python 08_inspect_chapman.py /path/to/extracted_root /path/to/output_dir
 extracted_root must contain WFDBRecords/ and ConditionNames_SNOMED-CT.csv (extract only *.hea + the csv first).
Writes: chapman_meta.csv (one row per record), chapman_code_counts.csv (label frequencies), prints a summary.
"""
import sys, os, re, glob, collections, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)

# --- SNOMED code -> name table (column layout detected automatically) ---
cn = pd.read_csv(glob.glob(f"{root}/**/ConditionNames_SNOMED-CT.csv", recursive=True)[0], dtype=str)
code_col = max(cn.columns, key=lambda c: cn[c].astype(str).str.fullmatch(r"\d+").mean())
name_col = max([c for c in cn.columns if c != code_col], key=lambda c: cn[c].astype(str).str.len().mean())
names = dict(zip(cn[code_col].str.strip(), cn[name_col].str.strip())); print("name table:", len(names), "codes; columns used:", code_col, "/", name_col)

rows, cnt, bad = [], collections.Counter(), []
heads = sorted(glob.glob(f"{root}/WFDBRecords/*/*/*.hea")); print("header files found:", len(heads))
num = lambda t: int(re.match(r"\d+", t).group()) if re.match(r"\d+", t) else None     # leading integer of a token, else None
for h in heads:
    txt = open(h, errors="ignore").read().splitlines()
    body = [l for l in txt if l.strip() and not l.startswith("#")]
    try:
        first = body[0].split(); fs, n, nl = num(first[2]), num(first[3]), num(first[1])
        if None in (fs, n, nl): raise ValueError("unparsable first line")
    except Exception:
        bad.append((h, body[:2])); continue
    g = lambda key: next((l.split(":", 1)[1].strip() for l in txt if re.match(rf"#\s*{key}\s*:", l, re.I)), "")
    dx = [c.strip() for c in g("Dx").split(",") if c.strip()]
    gains = [l.split()[2] for l in body[1:13] if len(l.split()) > 2]
    rows.append(dict(record=os.path.basename(h)[:-4], path=os.path.dirname(h), fs=fs, n=n, n_leads=nl,
                     gain=gains[0] if gains else "", age=g("Age"), sex=g("Sex"), dx=";".join(dx)))
    cnt.update(dx)
print("headers skipped (unparsable):", len(bad))
for h, b2 in bad[:3]: print("  e.g.", h, "->", b2)
m = pd.DataFrame(rows); m.to_csv(f"{out}/chapman_meta.csv", index=False)
cc = pd.DataFrame([(c, names.get(c, "?"), n, n / len(m)) for c, n in cnt.most_common()], columns=["snomed", "name", "records", "fraction"])
cc.to_csv(f"{out}/chapman_code_counts.csv", index=False)

print("\nfirst header as text:\n" + open(heads[0], errors="ignore").read()[:600])
print(f"\nrecords {len(m)} | fs values {m.fs.unique()} | samples {m.n.unique()[:5]} | leads {m.n_leads.unique()} | gains {m.gain.unique()[:5]}")
odd = m[m.fs != 500]; print("records with fs != 500:", len(odd)); print(odd.head(3)[["record", "fs", "n", "n_leads"]].to_string(index=False) if len(odd) else "")
print("records with no Dx:", int((m.dx == "").sum()), "| median labels/record:", m.dx.str.count(";").add(1).median())
print("\nTop 40 labels:"); print(cc.head(40).to_string(index=False))
