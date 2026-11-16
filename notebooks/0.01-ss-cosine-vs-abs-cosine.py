#!/usr/bin/env python
"""Compare cosine vs absolute cosine distance metrics for consistency analysis.

Creates scatter plot showing nMAP values for both metrics, colored by significance.
This supports the decision to use cosine (not abs_cosine) as the default metric.

Usage:
    pixi run python notebooks/0.01-ss-cosine-vs-abs-cosine.py
    pixi run python notebooks/0.01-ss-cosine-vs-abs-cosine.py --dataset compound_no_source7

Outputs:
    data/processed/exploration/0.01/{dataset}/cosine_vs_abs_cosine_scatter.png

"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.lines import Line2D

from jump_production.summaries import index_files, write_summary

app = typer.Typer()


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name to analyze",
    ),
) -> None:
    # Connect to database
    con = duckdb.connect("data/processed/copairs_results.duckdb", read_only=True)

    # Query comparison data - filter by dataset using parameterized query
    query = """
    WITH cosine_results AS (
        SELECT
            _group_type,
            _preprocessing,
            _dataset,
            group_value,
            mean_normalized_average_precision as cosine_nmap,
            corrected_p_value as cosine_pval,
            n_perturbations
        FROM consistency_results
        WHERE _distance = 'cosine'
          AND _dataset = ?
          AND _group_type IN ('disease_area', 'repurposing')
          AND _preprocessing NOT LIKE '%_sweep'
    ),
    abs_cosine_results AS (
        SELECT
            _group_type,
            _preprocessing,
            _dataset,
            group_value,
            mean_normalized_average_precision as abs_cosine_nmap,
            corrected_p_value as abs_cosine_pval
        FROM consistency_results
        WHERE _distance = 'abs_cosine'
          AND _dataset = ?
          AND _group_type IN ('disease_area', 'repurposing')
          AND _preprocessing NOT LIKE '%_sweep'
    )
    SELECT
        c._group_type,
        c._preprocessing,
        c.group_value,
        c.n_perturbations as n,
        c.cosine_nmap,
        a.abs_cosine_nmap,
        c.cosine_pval < 0.05 as cosine_sig,
        a.abs_cosine_pval < 0.05 as abs_cosine_sig
    FROM cosine_results c
    JOIN abs_cosine_results a
        ON c._group_type = a._group_type
        AND c._preprocessing = a._preprocessing
        AND c._dataset = a._dataset
        AND c.group_value = a.group_value
    """

    df = con.execute(query, [dataset, dataset]).fetchdf()
    con.close()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    display_names = {"disease_area": "Disease Area", "repurposing": "Target"}
    for ax, group_type in zip(axes, ["disease_area", "repurposing"]):
        subset = df[df["_group_type"] == group_type]

        # Color by significance change
        colors = []
        for _, row in subset.iterrows():
            if row["cosine_sig"] and row["abs_cosine_sig"]:
                colors.append("#2ecc71")  # both significant - green
            elif row["cosine_sig"] and not row["abs_cosine_sig"]:
                colors.append("#e74c3c")  # lost significance - red
            elif not row["cosine_sig"] and row["abs_cosine_sig"]:
                colors.append("#3498db")  # gained significance - blue
            else:
                colors.append("#95a5a6")  # neither significant - gray

        # Size by n_perturbations
        sizes = np.clip(subset["n"] * 5, 20, 200)

        ax.scatter(
            subset["cosine_nmap"],
            subset["abs_cosine_nmap"],
            c=colors,
            s=sizes,
            alpha=0.6,
            edgecolors="white",
            linewidth=0.5,
        )

        # Diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.3, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Cosine nMAP")
        ax.set_ylabel("Abs Cosine nMAP")
        ax.set_title(f"{display_names[group_type]}\n(n={len(subset)} groups)")
        ax.set_aspect("equal")

        # Add correlation
        corr = np.corrcoef(subset["cosine_nmap"], subset["abs_cosine_nmap"])[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="Both sig"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="Lost sig (cosine→abs)"
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="Gained sig"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=10, label="Neither sig"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Cosine vs Absolute Cosine: Consistency nMAP Comparison", y=1.02)
    plt.tight_layout()

    # Save to exploration directory with dataset nesting
    outdir = Path("data/processed/exploration/0.01") / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "cosine_vs_abs_cosine_scatter.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outpath}")

    # Summary stats
    print("\n=== Summary ===")
    per_annotation = {}
    for group_type in ["disease_area", "repurposing"]:
        subset = df[df["_group_type"] == group_type]
        diff = subset["abs_cosine_nmap"] - subset["cosine_nmap"]
        print(f"\n{display_names[group_type]} ({group_type}):")
        print(f"  N groups: {len(subset)}")
        print(f"  Mean diff (abs - cos): {diff.mean():.4f}")
        print(f"  Groups improved: {(diff > 0).sum()} ({100 * (diff > 0).mean():.1f}%)")
        print(f"  Groups worse: {(diff < 0).sum()} ({100 * (diff < 0).mean():.1f}%)")

        # Significance changes
        both_sig = ((subset["cosine_sig"]) & (subset["abs_cosine_sig"])).sum()
        lost_sig = ((subset["cosine_sig"]) & (~subset["abs_cosine_sig"])).sum()
        gained_sig = ((~subset["cosine_sig"]) & (subset["abs_cosine_sig"])).sum()
        neither_sig = ((~subset["cosine_sig"]) & (~subset["abs_cosine_sig"])).sum()
        print(f"  Significance: both={both_sig}, lost={lost_sig}, gained={gained_sig}")

        corr = float(np.corrcoef(subset["cosine_nmap"], subset["abs_cosine_nmap"])[0, 1])
        per_annotation[group_type] = {
            "n_groups": int(len(subset)),
            "nmap_correlation_r": round(corr, 4),
            "mean_nmap_diff_abs_minus_cosine": round(float(diff.mean()), 4),
            "groups_improved": int((diff > 0).sum()),
            "groups_worse": int((diff < 0).sum()),
            "sig_both": int(both_sig),
            "sig_lost_cosine_to_abs": int(lost_sig),
            "sig_gained": int(gained_sig),
            "sig_neither": int(neither_sig),
        }

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    write_summary(
        outdir / "summary.json",
        description=(
            "Cosine vs absolute-cosine consistency-metric comparison (Issue #22). For disease-area "
            "and target (repurposing) annotation groups, reports the correlation between nMAP under "
            "the two distance metrics, the mean nMAP shift (abs - cosine), and how many groups change "
            "significance status. Supports using cosine, not abs_cosine, as the default consistency "
            "metric."
        ),
        numbers={"dataset": dataset, "per_annotation": per_annotation},
        files=index_files(outdir),
    )


if __name__ == "__main__":
    app()
