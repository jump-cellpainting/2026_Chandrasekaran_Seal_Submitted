#!/usr/bin/env -S uv run python3
"""Storage footprint of the JUMP single-cell and well-level profiles on CPG.

Sums the raw profile files in each source's backend/ on the Cell Painting Gallery
(cpg0016-jump, per-source - NOT the assembled product):
  - single-cell  = the per-plate `.sqlite` (all cells, all measurements)
  - well-level   = the per-plate `.csv` that rides alongside each .sqlite (the
                   per-well aggregate of that single-cell data)
Reported at two scopes: ALL released sources, and COMPOUND only (the sources with
cell counts, matching the 1,533,879,336 single-cell count). Source 4 (ORF), 13
(CRISPR), 15 (no cell counts) are the non-compound exclusions.

NOTE on units: the paper's "68 TB" single-cell figure is 67.7 TiB (binary). We emit
exact bytes plus both decimal (TB=1e12) and binary (TiB=2**40) so the paper can be
explicit. Read-only S3 listing (no download), public no-sign-request.
"""
import subprocess
from collections import defaultdict
from pathlib import Path

from jump_production.summaries import write_summary

OUT = Path("data/processed/dataset-storage-sizes")
ALL_SOURCES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15]
COMPOUND = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11}  # sources with cell counts == the 1.53B count scope
BUCKET = "s3://cellpainting-gallery/cpg0016-jump/source_{}/workspace/backend/"

sqlite_by_src, csv_by_src = defaultdict(int), defaultdict(int)
for s in ALL_SOURCES:
    out = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET.format(s)],
        capture_output=True, text=True,
    ).stdout
    for ln in out.splitlines():
        p = ln.split()
        if len(p) < 4:
            continue
        size, key = int(p[2]), p[3]
        if key.endswith(".sqlite"):
            sqlite_by_src[s] += size
        elif key.endswith(".csv") or key.endswith(".csv.gz"):
            csv_by_src[s] += size

sc_all = sum(sqlite_by_src.values())
sc_cpd = sum(v for k, v in sqlite_by_src.items() if k in COMPOUND)
wl_all = sum(csv_by_src.values())
wl_cpd = sum(v for k, v in csv_by_src.items() if k in COMPOUND)

TB, TiB, GB, GiB = 1e12, 2**40, 1e9, 2**30
print(f"single-cell .sqlite  ALL={sc_all/TB:6.1f} TB ({sc_all/TiB:.1f} TiB)   COMPOUND={sc_cpd/TB:.1f} TB ({sc_cpd/TiB:.1f} TiB)")
print(f"well-level  .csv     ALL={wl_all/GB:6.1f} GB ({wl_all/GiB:.1f} GiB)   COMPOUND={wl_cpd/GB:.1f} GB ({wl_cpd/GiB:.1f} GiB)")

OUT.mkdir(parents=True, exist_ok=True)
write_summary(
    OUT / "summary.json",
    description=(
        "Storage footprint of JUMP profiles on the Cell Painting Gallery (cpg0016-jump, per-source "
        "backend/). single_cell_* = per-plate .sqlite (all cells); well_level_* = the per-plate .csv "
        "aggregate beside each .sqlite. Scopes: _all = all 13 released sources; _compound = the 10 "
        "sources with cell counts (matches the 1,533,879,336 single-cell count; excludes ORF source_4, "
        "CRISPR source_13, and source_15). Bytes are exact; *_tb uses 1e12, *_tib uses 2**40 (the "
        "paper's '68 TB' is 67.7 TiB). Replaces the stale '36 GB' well-level figure (~103-132 GB)."
    ),
    numbers={
        "single_cell_sqlite_bytes_all": sc_all,
        "single_cell_sqlite_tb_all": round(sc_all / TB, 1),
        "single_cell_sqlite_tib_all": round(sc_all / TiB, 1),
        "single_cell_sqlite_bytes_compound": sc_cpd,
        "single_cell_sqlite_tb_compound": round(sc_cpd / TB, 1),
        "single_cell_sqlite_tib_compound": round(sc_cpd / TiB, 1),
        "well_level_csv_bytes_all": wl_all,
        "well_level_csv_gb_all": round(wl_all / GB, 1),
        "well_level_csv_gib_all": round(wl_all / GiB, 1),
        "well_level_csv_bytes_compound": wl_cpd,
        "well_level_csv_gb_compound": round(wl_cpd / GB, 1),
        "well_level_csv_gib_compound": round(wl_cpd / GiB, 1),
    },
    files=[OUT / "summary.json"],
)
