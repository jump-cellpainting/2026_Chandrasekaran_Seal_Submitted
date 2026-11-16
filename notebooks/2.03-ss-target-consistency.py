"""Summarize copairs activity and consistency results.

Produces volcano plots, dot plots, and UMAP visualizations for target consistency.
Metadata is joined at visualization time from DuckDB (not stored in h5ad files).

Usage:
    pixi run python notebooks/2.03-ss-target-consistency.py
    pixi run python notebooks/2.03-ss-target-consistency.py --dataset compound_DL_CPCNN_no_source7
    pixi run python notebooks/2.03-ss-target-consistency.py --group-type uniprot
    pixi run python notebooks/2.03-ss-target-consistency.py --umap-filter active

Outputs: data/processed/target-consistency/{dataset}/
"""

import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from adjustText import adjust_text
from loguru import logger

from jump_production.config import DEFAULT_DPI
from jump_production.io import load_umap
from jump_production.queries import query_activity_results, query_consistency_results
from jump_production.visualization import compute_umap_bounds, plot_dotplot

# Database path (used for direct queries, not through io.py)
COPAIRS_DB = Path("data/processed/copairs_results.duckdb")

app = typer.Typer()


def get_group_type_column(con: duckdb.DuckDBPyConnection, group_type: str) -> str | None:
    """Get h5ad column name for a group type from copairs database.

    The mapping is stored in consistency_results._group_column. We strip the
    Metadata_ prefix since load_profiles() strips it when loading into h5ad.
    """
    result = con.execute(
        "SELECT DISTINCT _group_column FROM consistency_results WHERE _group_type = ? AND _preprocessing NOT LIKE '%_sweep'",
        [group_type],
    ).fetchone()
    if result:
        # Strip Metadata_ prefix (load_profiles strips it)
        return result[0].replace("Metadata_", "")
    return None


def plot_volcano(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Target Consistency",
    p_threshold: float = 0.05,
    label_top_n: int = 20,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create volcano plot of consistency results.

    x-axis: normalized mAP (effect size)
    y-axis: -log10(corrected p-value) (significance)
    Labels: significant targets
    Size: n_perturbations
    """
    if len(df) == 0:
        logger.warning("No data to plot")
        return

    # Compute -log10(p-value)
    df = df.copy()
    # Avoid log(0) by clipping
    df["neg_log10_p"] = -np.log10(df["corrected_p_value"].clip(lower=1e-300))

    # Significance threshold line
    sig_threshold = -np.log10(p_threshold)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate significant vs non-significant
    sig_mask = df["below_corrected_p"]
    nonsig = df[~sig_mask]
    sig = df[sig_mask]

    # Size scaling based on n_perturbations
    size_scale = 20
    min_size = 20
    max_size = 200

    def get_sizes(n_perturb):
        sizes = n_perturb * size_scale
        return sizes.clip(min_size, max_size)

    # Plot non-significant (gray)
    if len(nonsig) > 0:
        ax.scatter(
            nonsig["mean_normalized_average_precision"],
            nonsig["neg_log10_p"],
            s=get_sizes(nonsig["n_perturbations"]),
            c="#bdc3c7",
            alpha=0.6,
            edgecolors="none",
            label=f"Not significant (n={len(nonsig)})",
        )

    # Plot significant (colored)
    if len(sig) > 0:
        ax.scatter(
            sig["mean_normalized_average_precision"],
            sig["neg_log10_p"],
            s=get_sizes(sig["n_perturbations"]),
            c="#3498db",
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"Significant p<{p_threshold} (n={len(sig)})",
        )

    # Add significance threshold line
    ax.axhline(y=sig_threshold, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.98,
        sig_threshold + 0.1,
        f"p={p_threshold}",
        ha="right",
        va="bottom",
        color="red",
        fontsize=9,
    )

    # Label top significant targets
    if len(sig) > 0:
        # Sort by mAP and take top N
        top_targets = sig.nlargest(label_top_n, "mean_normalized_average_precision")

        texts = []
        for _, row in top_targets.iterrows():
            texts.append(
                ax.text(
                    row["mean_normalized_average_precision"],
                    row["neg_log10_p"],
                    row["group_value"],
                    fontsize=8,
                )
            )

        # Adjust text positions to avoid overlap
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            ax=ax,
        )

    ax.set_xlabel("Normalized mAP", fontsize=11)
    ax.set_ylabel("-log₁₀(corrected p-value)", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Add size legend (bottom right)
    size_handles = []
    size_labels = []
    for n in [2, 5, 10]:
        size_handles.append(ax.scatter([], [], s=get_sizes(pd.Series([n]))[0], c="gray", alpha=0.6))
        size_labels.append(f"n={n}")
    size_legend = ax.legend(
        size_handles,
        size_labels,
        title="# compounds",
        loc="lower right",
        framealpha=0.9,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(size_legend)

    # Add significance legend (bottom right, above size legend)
    ax.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved volcano plot to {output_path}")


def write_summary_stats(
    output_path: Path,
    dataset: str,
    group_type: str,
    activity_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    distance: str = "cosine",
) -> None:
    """Write summary statistics to markdown file."""
    with open(output_path, "w") as f:
        f.write(f"# Copairs Summary: {dataset}\n\n")
        f.write(f"- **Group type**: {group_type}\n")
        f.write(f"- **Distance metric**: {distance}\n\n")

        # Activity summary
        f.write("## Activity Results\n\n")
        n_total = len(activity_df)
        n_active = activity_df["below_corrected_p"].sum()
        rate = n_active / n_total if n_total > 0 else 0
        f.write(f"- Total compounds: {n_total:,}\n")
        f.write(f"- Active compounds (p<0.05): {n_active:,} ({rate:.1%})\n")
        if n_total > 0:
            f.write(f"- Mean normalized mAP: {activity_df['mean_normalized_average_precision'].mean():.4f}\n")
            f.write(f"- Median normalized mAP: {activity_df['mean_normalized_average_precision'].median():.4f}\n")
        f.write("\n")

        # Consistency summary
        f.write(f"## Consistency Results ({group_type}, {distance})\n\n")
        n_targets = len(consistency_df)
        n_consistent = consistency_df["below_corrected_p"].sum()
        rate = n_consistent / n_targets if n_targets > 0 else 0
        f.write(f"- Total targets: {n_targets:,}\n")
        f.write(f"- Consistent targets (p<0.05): {n_consistent:,} ({rate:.1%})\n\n")

        # Top consistent targets
        if len(consistency_df) > 0:
            f.write("### Top 10 Consistent Targets\n\n")
            f.write("| Target | Normalized mAP | p-value | # Compounds |\n")
            f.write("|--------|----------------|---------|-------------|\n")
            for _, row in consistency_df.head(10).iterrows():
                f.write(
                    f"| {row['group_value']} | {row['mean_normalized_average_precision']:.3f} | "
                    f"{row['corrected_p_value']:.2e} | {row['n_perturbations']} |\n"
                )

    logger.info(f"Saved summary stats to {output_path}")


def plot_target_umap_grid(
    con: duckdb.DuckDBPyConnection,
    consistency_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    output_path: Path,
    dataset: str,
    group_type: str,
    top_n: int = 12,
    metric: str = "cosine",
    umap_filter: str = "all",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create 3x4 grid of UMAPs highlighting top consistent targets.

    Each panel shows one target's compounds highlighted against gray background.
    - Blue: compounds in consistency analysis (has target + phenotypically active)
    - Purple: compounds with target annotation but NOT phenotypically active

    Args:
        con: DuckDB connection to copairs_results.duckdb
        umap_filter: UMAP variant to use ('all' or 'active'). With 'active',
            only phenotypically active compounds are in the UMAP, so 'Inactive'
            category will be empty.
    """
    # Get column mapping from database
    target_column = get_group_type_column(con, group_type)
    if target_column is None:
        logger.warning(f"No column mapping found for group_type={group_type}, skipping UMAP grid")
        return

    # Load perturbation-level UMAP with metadata
    try:
        adata = load_umap(dataset, level="perturbation", metric=metric, filter_name=umap_filter)
    except FileNotFoundError as e:
        logger.warning(f"{e}, skipping UMAP grid")
        return

    # Check if target column exists
    if target_column not in adata.obs.columns:
        logger.warning(f"Column {target_column} not found in h5ad, skipping UMAP grid")
        return

    # Get set of phenotypically active JCP2022 IDs (these are used in consistency analysis)
    active_ids = set(activity_df.loc[activity_df["below_corrected_p"], "Metadata_JCP2022"].tolist())
    logger.info(f"Found {len(active_ids)} phenotypically active compounds")

    if umap_filter == "active":
        logger.info(
            "Using active-only UMAP: 'Inactive' category will be empty (all compounds are phenotypically active)"
        )

    # Get top N targets from consistency results
    top_targets = consistency_df.nlargest(top_n, "mean_normalized_average_precision")["group_value"].tolist()
    logger.info(f"Plotting top {len(top_targets)} targets on UMAP")

    # Compute bounds once for consistent axes
    # Use iqr_k=1 for 'all' (tighter zoom), iqr_k=3 for 'active' (wider view)
    iqr_k = 1.5 if umap_filter == "all" else 3
    xlim, ylim = compute_umap_bounds(adata, iqr_k=iqr_k)

    # Get UMAP coordinates
    coords = adata.obsm["X_umap"]

    # Create 3x4 grid
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()

    for idx, target in enumerate(top_targets):
        ax = axes[idx]

        # Get mAP and n for subtitle
        target_row = consistency_df[consistency_df["group_value"] == target].iloc[0]
        n_in_analysis = target_row["n_perturbations"]
        nmap = target_row["mean_normalized_average_precision"]

        # Create mask for this target's compounds
        # Target column has pipe-separated values (e.g., "BRD2|BRD3|BRD4")
        # Use regex to match whole word: (?:^|\|)TARGET(?:\||$)
        pattern = rf"(?:^|\|){re.escape(target)}(?:\||$)"
        has_target = adata.obs[target_column].str.contains(pattern, na=False, regex=True)

        # Split into active vs inactive
        # In consistency analysis = has target + phenotypically active
        is_active = adata.obs["JCP2022"].isin(active_ids)
        mask_in_analysis = has_target & is_active
        mask_not_in_analysis = has_target & ~is_active

        n_total = has_target.sum()

        # Create highlight column for this target
        highlight = pd.Categorical(
            ["Other"] * len(adata),
            categories=["Other", "Inactive", "Active"],
        )
        highlight[mask_not_in_analysis] = "Inactive"
        highlight[mask_in_analysis] = "Active"
        adata.obs["_highlight"] = highlight

        # Plot using scanpy with groups parameter
        sc.pl.umap(
            adata,
            color="_highlight",
            groups=["Active", "Inactive"],
            palette={"Active": "#0072B2", "Inactive": "#CC79A7", "Other": "#E5E5E5"},
            na_color="#E5E5E5",
            size=20,
            ax=ax,
            show=False,
            title=f"{target}\n({n_in_analysis}/{n_total} active, mAP={nmap:.2f})",
        )

        # Add labels with adjustText only if <10 compounds for this target
        if n_total < 10:
            texts = []
            for mask, text_color in [(mask_in_analysis, "#0072B2"), (mask_not_in_analysis, "#CC79A7")]:
                if mask.sum() > 0:
                    indices = np.where(mask)[0]
                    for i in indices:
                        jcp_id = adata.obs["JCP2022"].iloc[i]
                        short_id = jcp_id.replace("JCP2022_", "")
                        texts.append(
                            ax.text(
                                coords[i, 0],
                                coords[i, 1],
                                short_id,
                                fontsize=8,
                                fontweight="bold",
                                color=text_color,
                                zorder=11,
                            )
                        )
            if texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        # Apply consistent bounds
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Clean up
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove() if ax.get_legend() else None

    # Remove unused subplots if top_n < 12
    for idx in range(len(top_targets), len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#0072B2",
            markersize=10,
            markeredgecolor="white",
            label="Active",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#CC79A7",
            markersize=10,
            markeredgecolor="white",
            label="Inactive",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(f"Top Consistent Targets: {group_type} ({dataset}, {umap_filter})", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved target UMAP grid to {output_path}")


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name (e.g., compound_no_source7, compound_DL_CPCNN_no_source7)",
    ),
    group_type: str = typer.Option(
        "repurposing",
        "--group-type",
        "-g",
        help="Consistency group type (e.g., repurposing, uniprot, moa)",
    ),
    preprocessing: str = typer.Option(
        "activity_no_target2",
        "--preprocessing",
        "-p",
        help="Activity preprocessing config name",
    ),
    filter_name: str = typer.Option(
        "all_sources",
        "--filter",
        "-f",
        help="Filter config name",
    ),
    label_top_n: int = typer.Option(
        20,
        "--label-top-n",
        help="Number of top significant targets to label on volcano plot",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/processed/target-consistency/{dataset})",
    ),
    umap_filter: str = typer.Option(
        "all",
        "--umap-filter",
        help="UMAP variant to use: 'all' (all compounds) or 'active' (phenotypically active only)",
    ),
    umap_only: bool = typer.Option(
        False,
        "--umap-only",
        help="Only generate UMAP grid (skip volcano, dotplot, summary)",
    ),
    distance: str = typer.Option(
        "cosine",
        "--distance",
        help="Distance metric for consistency (cosine, abs_cosine)",
    ),
    activity_params: str = typer.Option(
        "default",
        "--activity-params",
        help="Activity params config (default, withinsource, crosssource)",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Summarize copairs activity and consistency results."""
    if output_dir is None:
        output_dir = Path("data/processed/target-consistency") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this group type and distance
    group_output_dir = output_dir / group_type / distance
    group_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Group type: {group_type}")
    logger.info(f"Output: {group_output_dir}")

    if not COPAIRS_DB.exists():
        logger.error(f"Copairs database not found: {COPAIRS_DB}")
        raise typer.Exit(1)

    copairs_con = duckdb.connect(str(COPAIRS_DB), read_only=True)

    try:
        # 1. Query activity results
        logger.info("Querying activity results...")
        activity_df = query_activity_results(copairs_con, dataset, preprocessing, filter_name, activity_params)

        if len(activity_df) == 0:
            logger.error(f"No activity results found for dataset={dataset}")
            raise typer.Exit(1)

        logger.info(f"Found {len(activity_df):,} compounds")

        # 2. Query consistency results for specified group type
        logger.info(f"Querying consistency results for {group_type}...")
        consistency_preprocessing = preprocessing.replace("activity_", "consistency_")
        consistency_df = query_consistency_results(
            copairs_con, dataset, consistency_preprocessing, filter_name, group_type, distance
        )

        if len(consistency_df) == 0:
            logger.warning(f"No consistency results found for group_type={group_type}")
        else:
            logger.info(f"Found {len(consistency_df)} targets")
            n_sig = consistency_df["below_corrected_p"].sum()
            logger.info(f"Significant targets: {n_sig} ({n_sig / len(consistency_df):.1%})")

            if not umap_only:
                # Save consistency results
                consistency_df.to_csv(group_output_dir / "results.csv", index=False)

                # Generate plots
                plot_volcano(
                    consistency_df,
                    group_output_dir / "volcano.png",
                    title=f"Target Consistency: {group_type} ({dataset})",
                    label_top_n=label_top_n,
                    dpi=dpi,
                )
                plot_dotplot(
                    consistency_df,
                    value_col="mean_normalized_average_precision",
                    label_col="group_value",
                    count_col="n_perturbations",
                    sig_col="below_corrected_p",
                    title=f"Top Consistent Targets: {group_type} ({dataset})",
                    xlabel="Normalized mAP",
                    output_dir=group_output_dir,
                    filename="dotplot.png",
                    dpi=dpi,
                )

            # Generate UMAP grid for top targets
            plot_target_umap_grid(
                copairs_con,
                consistency_df,
                activity_df,
                group_output_dir / f"umap_grid_{umap_filter}.png",
                dataset=dataset,
                group_type=group_type,
                umap_filter=umap_filter,
                dpi=dpi,
            )

        if not umap_only:
            # 3. Write summary
            write_summary_stats(
                group_output_dir / "summary_stats.md",
                dataset,
                group_type,
                activity_df,
                consistency_df,
                distance,
            )

        logger.success(f"All outputs saved to {group_output_dir}")

    finally:
        copairs_con.close()


if __name__ == "__main__":
    app()
