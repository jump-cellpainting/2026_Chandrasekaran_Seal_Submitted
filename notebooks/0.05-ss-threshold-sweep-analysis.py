#!/usr/bin/env python
"""Analyze activity threshold sensitivity for consistency results.

Sweeps activity p-value thresholds (0.05, 0.10,
0.15, 0.20, 0.25, 0.30) and measuring impact on consistency results.

Sweep results are stored in the unified consistency_results table, differentiated
by the _preprocessing column (names ending in '_sweep' for sweep runs).

Usage:
    pixi run python notebooks/0.05-ss-threshold-sweep-analysis.py
    pixi run python notebooks/0.05-ss-threshold-sweep-analysis.py --dataset compound_no_source7 --outdir data/processed/exploration/0.05/compound_no_source7

Outputs:
    {outdir}/threshold_sweep_summary.png
    {outdir}/threshold_sweep_by_annotation.png

"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from jump_production.config import DEFAULT_DPI
from jump_production.summaries import index_files, write_summary

app = typer.Typer()


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name to analyze (must have sweep runs in copairs_results)",
    ),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Output directory for plots (default: data/processed/exploration/0.05/{dataset})",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    # Set default outdir based on dataset if not provided
    if outdir is None:
        outdir = Path(f"data/processed/exploration/0.05/{dataset}")

    # Connect to database
    con = duckdb.connect("data/processed/copairs_results.duckdb", read_only=True)

    # Check if sweep results exist for this dataset
    sweep_count = con.execute(
        """
        SELECT COUNT(*) FROM consistency_results
        WHERE _preprocessing LIKE '%_sweep'
          AND _dataset = ?
    """,
        [dataset],
    ).fetchone()[0]

    if sweep_count == 0:
        print(f"ERROR: No threshold sweep results found for dataset '{dataset}'.")
        print("Run 'just run' to execute the threshold sweep pipeline first.")
        con.close()
        raise typer.Exit(1)

    print(f"Dataset: {dataset}")
    print(f"Found {sweep_count:,} sweep results in consistency_results table")

    # Query threshold sweep results from unified consistency_results table
    # Compute nMAP for both ALL groups and SIGNIFICANT-only groups
    query = """
    SELECT
        _activity_threshold as threshold,
        _group_type as annotation,
        COUNT(*) as n_groups,
        SUM(below_corrected_p::int) as n_significant,
        ROUND(100.0 * SUM(below_corrected_p::int) / COUNT(*), 1) as pct_significant,
        -- nMAP over ALL groups (includes non-significant noise)
        ROUND(AVG(mean_normalized_average_precision), 4) as avg_nmap_all,
        ROUND(STDDEV(mean_normalized_average_precision), 4) as std_nmap_all,
        -- nMAP over SIGNIFICANT groups only (effect size of validated associations)
        ROUND(AVG(CASE WHEN below_corrected_p THEN mean_normalized_average_precision END), 4) as avg_nmap_significant,
        ROUND(STDDEV(CASE WHEN below_corrected_p THEN mean_normalized_average_precision END), 4) as std_nmap_significant,
        SUM(n_perturbations) as total_perturbations
    FROM consistency_results
    WHERE _preprocessing LIKE '%_sweep'
      AND _dataset = ?
    GROUP BY _activity_threshold, _group_type
    ORDER BY _group_type, _activity_threshold
    """

    df = con.execute(query, [dataset]).fetchdf()

    if df.empty:
        print(f"ERROR: No threshold sweep data found for dataset '{dataset}'.")
        con.close()
        raise typer.Exit(1)

    print("\n=== Threshold Sweep Summary ===")
    print(df.to_string(index=False))

    # Baseline comparison: regular runs use threshold 0.10
    # Verify that sweep results at 0.10 match regular runs with equivalent preprocessing
    print("\n=== Baseline Comparison (threshold = 0.10) ===")
    print("Regular consistency runs use activity threshold 0.10 as the default.")
    print("Comparing equivalent preprocessing configs (e.g., consistency_no_target2 vs consistency_no_target2_sweep).")

    # Compare matching preprocessing configs: strip '_sweep' suffix to find base name
    # Include _distance in grouping since some annotations have multiple distance metrics
    baseline_query = """
    WITH regular AS (
        SELECT
            _group_type as annotation,
            _preprocessing,
            _dataset,
            _distance,
            ROUND(AVG(mean_normalized_average_precision), 4) as avg_nmap_significant,
            COUNT(*) as n_groups
        FROM consistency_results
        WHERE _preprocessing NOT LIKE '%_sweep'
          AND _activity_threshold = 0.10
          AND _dataset = ?
        GROUP BY _group_type, _preprocessing, _dataset, _distance
    ),
    sweep AS (
        SELECT
            _group_type as annotation,
            REPLACE(_preprocessing, '_sweep', '') as base_preprocessing,
            _dataset,
            _distance,
            ROUND(AVG(mean_normalized_average_precision), 4) as avg_nmap_significant,
            COUNT(*) as n_groups
        FROM consistency_results
        WHERE _preprocessing LIKE '%_sweep'
          AND _activity_threshold = 0.10
          AND _dataset = ?
        GROUP BY _group_type, _preprocessing, _dataset, _distance
    )
    SELECT
        COALESCE(r.annotation, s.annotation) as annotation,
        COALESCE(r._preprocessing, s.base_preprocessing) as preprocessing,
        COALESCE(r._distance, s._distance) as distance,
        r.avg_nmap_significant as regular_nmap,
        s.avg_nmap_significant as sweep_nmap,
        r.n_groups as regular_n,
        s.n_groups as sweep_n
    FROM regular r
    FULL OUTER JOIN sweep s
        ON r.annotation = s.annotation
        AND r._preprocessing = s.base_preprocessing
        AND r._dataset = s._dataset
        AND r._distance = s._distance
    WHERE s.avg_nmap_significant IS NOT NULL  -- Only show configs that have sweep runs
    ORDER BY annotation, preprocessing, distance
    """
    baseline_df = con.execute(baseline_query, [dataset, dataset]).fetchdf()

    if not baseline_df.empty:
        print("\nSanity check: sweep at 0.10 should match regular runs with same config")
        print(baseline_df.to_string(index=False))

        # Check if values match (within floating point tolerance)
        all_match = True
        for _, row in baseline_df.iterrows():
            reg_nmap = row["regular_nmap"]
            sweep_nmap = row["sweep_nmap"]
            reg_n = row["regular_n"]
            sweep_n = row["sweep_n"]

            nmap_match = abs(reg_nmap - sweep_nmap) < 0.001 if pd.notna(reg_nmap) and pd.notna(sweep_nmap) else True
            n_match = reg_n == sweep_n if pd.notna(reg_n) and pd.notna(sweep_n) else True

            if not (nmap_match and n_match):
                all_match = False

        if all_match:
            print("\n✓ All sweep results at threshold 0.10 match their regular counterparts.")
        else:
            print("\n⚠ Some sweep results differ from regular runs - investigate if unexpected.")

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Summary plot with 3 panels - the key metrics for decision-making
    # - N significant (absolute discovery count - often the most important)
    # - % significant (rate)
    # - Mean nMAP (effect size)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    annotations = df["annotation"].unique()
    # Clean color palette (ColorBrewer Dark2) - same as tradeoff plot
    summary_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    color_map = dict(zip(annotations, summary_colors))

    # Left: N significant (absolute count) - the primary discovery metric
    ax = axes[0]
    for annotation in annotations:
        subset = df[df["annotation"] == annotation]
        ax.plot(
            subset["threshold"],
            subset["n_significant"],
            "o-",
            color=color_map[annotation],
            label=annotation,
            markersize=4,
            linewidth=1,
        )
    ax.set_ylabel("N Significant", fontsize=8)
    ax.set_title("Discovery Count", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xticks(sorted(df["threshold"].unique()))
    ax.tick_params(labelsize=7)

    # Middle: % significant vs threshold
    ax = axes[1]
    for annotation in annotations:
        subset = df[df["annotation"] == annotation]
        ax.plot(
            subset["threshold"],
            subset["pct_significant"],
            "o-",
            color=color_map[annotation],
            label=annotation,
            markersize=4,
            linewidth=1,
        )
    ax.set_ylabel("% Significant", fontsize=8)
    ax.set_title("Significance Rate", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xticks(sorted(df["threshold"].unique()))
    ax.tick_params(labelsize=7)

    # Right: mean nMAP vs threshold
    ax = axes[2]
    for annotation in annotations:
        subset = df[df["annotation"] == annotation]
        ax.errorbar(
            subset["threshold"],
            subset["avg_nmap_significant"],
            yerr=subset["std_nmap_significant"] / np.sqrt(subset["n_significant"]),  # SEM
            fmt="o-",
            color=color_map[annotation],
            label=annotation,
            markersize=4,
            linewidth=1,
            capsize=0,
            elinewidth=0.5,
            alpha=0.8,
        )
    ax.set_ylabel("Mean nMAP (sig.)", fontsize=8)
    ax.set_title("Effect Size", fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xticks(sorted(df["threshold"].unique()))
    ax.tick_params(labelsize=7)

    # Common x-axis label and legend
    fig.supxlabel("Activity Threshold (p-value)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(annotations), fontsize=6, bbox_to_anchor=(0.5, -0.12))

    plt.suptitle("Threshold Sensitivity Analysis", fontsize=10, y=1.02)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    outpath = outdir / "threshold_sweep_summary.png"
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {outpath}")

    # Save CSV output for inspection
    csv_path = outdir / "threshold_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot 2: 2D trade-off plot - N significant vs Mean nMAP (all annotations on same axes)
    # Each annotation is a trajectory; threshold is the progression along the line
    fig, ax = plt.subplots(figsize=(6, 4))

    sorted(df["threshold"].unique())

    # Clean color palette (ColorBrewer Dark2)
    trajectory_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    trajectory_color_map = dict(zip(annotations, trajectory_colors))

    for annotation in annotations:
        subset = df[df["annotation"] == annotation].sort_values("threshold")
        color = trajectory_color_map[annotation]

        # Plot trajectory line
        ax.plot(
            subset["n_significant"],
            subset["avg_nmap_significant"],
            "-",
            color=color,
            linewidth=1,
            alpha=0.8,
            zorder=2,
            label=annotation,
        )

        # Plot small points
        ax.scatter(
            subset["n_significant"],
            subset["avg_nmap_significant"],
            c=[color],
            s=15,
            edgecolor="white",
            linewidth=0.2,
            zorder=3,
        )

        # Thin error bars
        for _, row in subset.iterrows():
            sem = row["std_nmap_significant"] / np.sqrt(row["n_significant"])
            ax.errorbar(
                row["n_significant"],
                row["avg_nmap_significant"],
                yerr=sem,
                fmt="none",
                color=color,
                alpha=0.3,
                linewidth=0.5,
                capsize=0,
                zorder=1,
            )

    ax.set_xlabel("N Significant Groups", fontsize=8)
    ax.set_ylabel("Mean nMAP (sig. only)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(title="Annotation", loc="upper right", fontsize=8, title_fontsize=9)

    plt.title("Trade-off: Discovery Count vs Effect Size", fontsize=9)

    outpath = outdir / "threshold_sweep_tradeoff.png"
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {outpath}")

    # Print detailed analysis
    print("\n=== Detailed Analysis ===")
    print("(nMAP_sig = mean over significant groups only; nMAP_all = mean over all groups)")
    for annotation in annotations:
        subset = df[df["annotation"] == annotation].sort_values("threshold")
        print(f"\n{annotation}:")
        print(f"  Thresholds:    {subset['threshold'].tolist()}")
        print(f"  N groups:      {subset['n_groups'].tolist()}")
        print(f"  N significant: {subset['n_significant'].tolist()}")
        print(f"  % sig:         {subset['pct_significant'].tolist()}")
        print(f"  Avg nMAP_sig:  {subset['avg_nmap_significant'].tolist()}")
        print(f"  Avg nMAP_all:  {subset['avg_nmap_all'].tolist()}")

    # Trade-off summary: compare 0.05 (most stringent) to 0.30 (most lenient)
    print("\n=== Trade-off Summary (0.05 → 0.30) ===")
    print("Relaxing the activity threshold from 0.05 to 0.30 produces these trade-offs:\n")

    for annotation in annotations:
        subset = df[df["annotation"] == annotation].sort_values("threshold")

        # Get values at threshold extremes
        low_thresh = subset[subset["threshold"] == subset["threshold"].min()]
        high_thresh = subset[subset["threshold"] == subset["threshold"].max()]

        if low_thresh.empty or high_thresh.empty:
            continue

        low_thresh["n_groups"].values[0]
        high_thresh["n_groups"].values[0]
        sig_low = low_thresh["n_significant"].values[0]
        sig_high = high_thresh["n_significant"].values[0]
        nmap_low = low_thresh["avg_nmap_significant"].values[0]
        nmap_high = high_thresh["avg_nmap_significant"].values[0]

        # Calculate changes
        sig_change = sig_high - sig_low
        sig_pct_change = ((sig_high - sig_low) / sig_low * 100) if sig_low > 0 else 0
        nmap_change = ((nmap_high - nmap_low) / nmap_low * 100) if nmap_low > 0 else 0

        sig_sign = "+" if sig_change >= 0 else ""
        nmap_sign = "+" if nmap_change >= 0 else ""

        print(f"  {annotation}:")
        print(f"    N significant: {sig_low} → {sig_high} ({sig_sign}{sig_change}, {sig_sign}{sig_pct_change:.0f}%)")
        print(f"    Avg nMAP:      {nmap_low:.4f} → {nmap_high:.4f} ({nmap_sign}{nmap_change:.0f}%)")

    print("\nInterpretation:")
    print("  - N significant = absolute discovery count (primary metric for finding biology)")
    print("  - Avg nMAP = effect size quality (confidence in the associations)")
    print("  - Choose threshold based on whether you prioritize discovery count or effect strength.")

    con.close()

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    per_annotation = {}
    for annotation in annotations:
        s = df[df["annotation"] == annotation].sort_values("threshold")
        low, high = s.iloc[0], s.iloc[-1]
        per_annotation[str(annotation)] = {
            "thresholds": [float(t) for t in s["threshold"]],
            "n_groups": [int(x) for x in s["n_groups"]],
            "n_significant": [int(x) for x in s["n_significant"]],
            "pct_significant": [float(x) for x in s["pct_significant"]],
            "avg_nmap_significant": [round(float(x), 4) for x in s["avg_nmap_significant"]],
            "tradeoff_low_to_high": {
                "thresholds": [float(low["threshold"]), float(high["threshold"])],
                "n_significant": [int(low["n_significant"]), int(high["n_significant"])],
                "avg_nmap_significant": [
                    round(float(low["avg_nmap_significant"]), 4),
                    round(float(high["avg_nmap_significant"]), 4),
                ],
            },
        }
    write_summary(
        outdir / "summary.json",
        description=(
            "Activity-threshold sensitivity sweep (Issue #22). For each annotation type, sweeps the "
            "activity p-value threshold (0.05-0.30) and reports the discovery count (significant "
            "groups), significance rate, and effect size (mean nMAP over significant groups) at each "
            "threshold, plus the 0.05->0.30 trade-off. Informs the default activity-threshold choice."
        ),
        numbers={"dataset": dataset, "per_annotation": per_annotation},
        files=index_files(outdir),
    )


if __name__ == "__main__":
    app()
