import sys, pandas as pd
# usage: python3 09_chapman_mapping_counts.py <chapman_out_dir>
d = sys.argv[1]
m = pd.read_csv(f"{d}/chapman_meta.csv", dtype=str).fillna("")
codes = m["dx"].str.split(";").apply(lambda l: set(c.strip() for c in l if c.strip()))

# STRICT mapping: only codes whose names appear in the dataset's own ConditionNames file
STRICT = {
    "NORM": {"426783006"},                      # Sinus rhythm (rhythm only; no morphology check)
    "CD":   {"59118001", "698252002", "164909002", "270492004", "233917008"},  # RBBB, IVCD, LAFB, 1st-deg AVB, AV block
    "STTC": {"164934002", "59931005", "429622005", "428750005", "164930006"},  # T change, T opp, ST drop, ST-T, ST ext
    "HYP":  {"164873001"},                      # LVH
}
# EXTRA codes whose names are '?' in the name file: UNVERIFIED, to be confirmed against the Challenge table
EXTRA = {
    "CD":   {"713427006", "713426002", "733534002", "445118002", "6374002"},
    "STTC": {"55930002"},
    "HYP":  {"55827005"},
}
# Records carrying other rhythm labels are excluded from NORM (a "normal" must be sinus rhythm alone)
for name, mp in [("STRICT", STRICT)]:
    print("==", name)
    for k, v in mp.items():
        n = codes.apply(lambda s: len(s & v) > 0).sum()
        print(f"{k:5s} {n:6d} records")
    only_sinus = codes.apply(lambda s: s == {"426783006"}).sum()
    print("NORM (sinus rhythm as the ONLY label):", only_sinus)
print("== EXTRA (unverified)")
for k, v in EXTRA.items():
    print(k, codes.apply(lambda s: len(s & v) > 0).sum())
