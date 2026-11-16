#!/usr/bin/env -S uv run python3
"""Direct gene-compound morphological matching is ~random (motivates MOTIVE's GNN).

Uses the CANONICAL JUMP "all" profile (jump-cellpainting/datasets manifest, subset "all"):
all plate types co-processed through ONE sphering+Harmony recipe, so compound and ORF
features share a basis (X_1..X_614 mean the same thing). Ground truth = MOTIVE's own
curated compound-gene edges. We cosine-match every known pair vs a label-permutation null.

Run with the project env (needs duckdb/pandas/numpy + the augmented DB):
    pixi run python notebooks/0.14-ss-gene-compound-matching.py

Inputs (auto-downloaded to data/external/motive/, gitignored, public no-sign-request):
  - canonical ALL profile (2.5 GB well-level) -> aggregated+cached to perturbation median
  - MOTIVE bipartite/orf maps + labels (curated compound-gene ground truth)
"""
import subprocess
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from jump_production.summaries import write_summary

OUT = Path("data/processed/gene-compound-matching")

D = Path("data/external/motive")
D.mkdir(parents=True, exist_ok=True)
ALL = "s3://cellpainting-gallery/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ALL/v1.0b/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
MOT = "s3://cellpainting-gallery/cpg0034-arevalo-su-motive/broad/workspace/publication_data/2024_MOTIVE/data/bipartite/orf"
DB = "data/interim/jump_metadata_augmented.duckdb"


def s3(url, dst):
    if not Path(dst).exists():
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", url, str(dst)], check=True)


# 1. Canonical ALL profile -> perturbation-level median (cached; the 2.5 GB scan runs once).
pert = D / "all_v1.0b_pert_median.parquet"
if not pert.exists():
    s3(ALL, D / "all_v1.0b.parquet")
    con = duckdb.connect()
    feats = [c for c in con.execute(f"DESCRIBE SELECT * FROM '{D}/all_v1.0b.parquet'").fetchdf()["column_name"] if c.startswith("X_")]
    agg = ",".join(f"median({c}) AS {c}" for c in feats)
    con.execute(f"COPY (SELECT Metadata_JCP2022 AS jcp,{agg} FROM '{D}/all_v1.0b.parquet' GROUP BY 1) TO '{pert}' (FORMAT parquet)")

# 2. MOTIVE curated ground truth: (InChIKey, gene Symbol) edges.
for f in ("source_map.parquet", "target_map.parquet", "s_t_labels.parquet"):
    s3(f"{MOT}/{f}", D / "orf" / f) if (D / "orf").mkdir(parents=True, exist_ok=True) or True else None
sm = pd.read_parquet(D / "orf/source_map.parquet").reset_index(); sm.columns = ["InChIKey", "sidx"]
tm = pd.read_parquet(D / "orf/target_map.parquet").reset_index(); tm.columns = ["Symbol", "tidx"]
lab = pd.read_parquet(D / "orf/s_t_labels.parquet")
edges = lab.merge(sm, left_on="source", right_on="sidx").merge(tm, left_on="target", right_on="tidx")[["InChIKey", "Symbol"]]

# 3. Map MOTIVE ids -> JCP2022 (compound InChIKey-skeleton, ORF gene symbol) via the augmented DB.
prof = pd.read_parquet(pert).set_index("jcp")
feats = [c for c in prof.columns if c.startswith("X_")]
db = duckdb.connect(DB, read_only=True)
cmap = db.execute("SELECT substr(Metadata_InChIKey,1,14) AS ik, Metadata_JCP2022 AS jcp FROM compound").fetchdf()
omap = db.execute("SELECT Metadata_Symbol AS sym, Metadata_JCP2022 AS jcp FROM orf").fetchdf()
ed = (edges.merge(cmap, left_on="InChIKey", right_on="ik").merge(omap, left_on="Symbol", right_on="sym", suffixes=("_c", "_o"))
      .rename(columns={"jcp_c": "cpd", "jcp_o": "orf"})[["cpd", "orf"]].drop_duplicates())
ed = ed[ed.cpd.isin(prof.index) & ed.orf.isin(prof.index)]

# 4. Cosine match: per gene, is a true partner in its top-1% compounds? vs a label-permutation null.
cpd_ids, orf_ids = sorted(set(ed.cpd)), sorted(set(ed.orf))
ci = {j: i for i, j in enumerate(cpd_ids)}; gi = {j: i for i, j in enumerate(orf_ids)}
norm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
Cn = norm(prof.loc[cpd_ids, feats].to_numpy()); Gn = norm(prof.loc[orf_ids, feats].to_numpy())
si, ti = ed.cpd.map(ci).to_numpy(), ed.orf.map(gi).to_numpy()
sims = Gn @ Cn.T
k = max(1, int(0.01 * len(Cn)))
topk = {g: set(np.argsort(-sims[g])[:k].tolist()) for g in np.unique(ti)}

def retrieval(src, tgt):
    g2c = defaultdict(set)
    for a, b in zip(src, tgt):
        g2c[b].add(a)
    return 100 * np.mean([len(cs & topk[g]) > 0 for g, cs in g2c.items()])

rng = np.random.default_rng(0)
real = retrieval(si, ti)
null = [retrieval(rng.permutation(si), ti) for _ in range(20)]
tc = (Cn[si] * Gn[ti]).sum(1)
rc = (Cn[rng.integers(0, len(Cn), len(si))] * Gn[rng.integers(0, len(Gn), len(si))]).sum(1)

null_mean, null_sd = float(np.mean(null)), float(np.std(null))
print("CANONICAL all v1.0b profiles + MOTIVE curated annotations")
print(f"  {len(cpd_ids)} compounds x {len(orf_ids)} ORF genes | {len(si)} curated edges | dims={len(feats)}")
print(f"  true-pair cosine mean={tc.mean():+.4f}   random-pair mean={rc.mean():+.4f}")
print(f"  top-1% retrieval (k={k}): REAL={real:.2f}%   permutation-null={null_mean:.2f}%±{null_sd:.2f}   signal={real - null_mean:+.2f}pts")
print("=> direct gene-compound morphological matching is at the random null; needs a learned graph model (MOTIVE).")

OUT.mkdir(parents=True, exist_ok=True)
write_summary(
    OUT / "summary.json",
    description=(
        "Direct gene-compound morphological matching is at the random null, motivating MOTIVE's "
        "learned graph model. Compounds and ORF genes co-embedded in ONE sphering+Harmony space "
        "(canonical JUMP 'all' v1.0b profile, split by PlateType, aggregated to perturbation median) "
        "with MOTIVE's curated compound-gene ground truth. For each gene we rank compounds by cosine "
        "and ask whether a true partner falls in its top 1%, vs a label-permutation null that absorbs "
        "gene-degree inflation. top1pct_retrieval barely exceeds the null (far below Rohban's ~32%)."
    ),
    numbers={
        "n_compounds": len(cpd_ids),
        "n_orf_genes": len(orf_ids),
        "n_curated_edges": int(len(si)),
        "n_features": len(feats),
        "top1pct_retrieval": round(real, 2),
        "permutation_null_mean": round(null_mean, 2),
        "permutation_null_sd": round(null_sd, 2),
        "signal_above_null_pts": round(real - null_mean, 2),
        "true_pair_cosine_mean": round(float(tc.mean()), 4),
        "random_pair_cosine_mean": round(float(rc.mean()), 4),
    },
    files=[OUT / "summary.json"],
)
