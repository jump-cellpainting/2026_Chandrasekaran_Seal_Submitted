"""Compare target consistency results between no_source7 and with_source7 datasets.

Usage:
    pixi run python scripts/compare_source7_consistency.py
"""

import pandas as pd
from pathlib import Path

TARGETS = [
    "repurposing", "uniprot", "chemical_probes", "moa", "disease_area",
    "motive_biokg", "motive_opentargets", "motive_primekg", "toxcast_assay",
]
COMPARISONS = [
    ("CP", "compound_no_source7", "compound_with_source7"),
    ("DL", "compound_DL_CPCNN_no_source7", "compound_DL_CPCNN_with_source7"),
]
BASE = Path("data/processed/target-consistency")


def compare(label, ds_no, ds_with):
    rows = []
    for target in TARGETS:
        for dataset, tag in [(ds_no, "no_s7"), (ds_with, "with_s7")]:
            path = BASE / dataset / target / "cosine" / "results.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            rows.append({
                "target": target,
                "dataset": tag,
                "n_targets": len(df),
                "n_significant": int(df["below_corrected_p"].sum()),
                "pct_significant": round(100 * df["below_corrected_p"].mean(), 1),
                "mean_nmap": round(df["mean_normalized_average_precision"].mean(), 4),
            })

    df = pd.DataFrame(rows)
    no = df[df.dataset == "no_s7"].set_index("target")
    ws = df[df.dataset == "with_s7"].set_index("target")

    comp = pd.DataFrame({
        "no_s7 sig": no["n_significant"].astype(str) + "/" + no["n_targets"].astype(str)
                     + " (" + no["pct_significant"].astype(str) + "%)",
        "with_s7 sig": ws["n_significant"].astype(str) + "/" + ws["n_targets"].astype(str)
                     + " (" + ws["pct_significant"].astype(str) + "%)",
        "Δ targets": ws["n_targets"] - no["n_targets"],
        "Δ significant": ws["n_significant"] - no["n_significant"],
        "no_s7 nMAP": no["mean_nmap"],
        "with_s7 nMAP": ws["mean_nmap"],
    })
    return comp


def main():
    for label, ds_no, ds_with in COMPARISONS:
        print(f"\n### {label} ({ds_no} vs {ds_with})\n")
        comp = compare(label, ds_no, ds_with)
        print(comp.to_markdown())
        print()


if __name__ == "__main__":
    main()
