"""Cross-source reproducibility analysis for JUMP Cell Painting dataset.

Compares within-source vs cross-source activity (normalized mAP) to measure
how reproducible compound phenotypes are across different lab sites.

NOTE: This is NOT "consistency" (shared biology clustering). This measures
REPRODUCIBILITY of the same compound's phenotypic activity across different lab sites.

Definitions:
- Within-source: Replicates of the same compound at the same source (same lab)
- Cross-source: Same compound compared across different sources (different labs)

Methodology:
- Uses normalized mAP (not raw mAP) because within-source has ~25 positive pairs
  while cross-source has ~122. Normalized mAP = (AP - μ₀) / (1 - μ₀) is scale-independent.
- Poscon correlation appears low due to ceiling effect—all cluster near nmAP≈1.0

Results: See issue #24 for findings and source_9 anomaly analysis.

Usage:
    pixi run python notebooks/2.02a-ss-cross-source-reproducibility.py -d compound_no_source7 -p activity_only_target2 -f no_source9
    pixi run python notebooks/2.02a-ss-cross-source-reproducibility.py -d compound_no_source7 -p activity_poscon_only
    snakemake cross_source_reproducibility  # runs all combinations

Outputs: data/processed/cross-source-reproducibility/{dataset}/{preprocessing}[_{filter}]/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from jump_production.config import DEFAULT_DPI
from jump_production.queries import query_cross_source_reproducibility
from jump_production.summaries import index_files, write_summary

# Database path
COPAIRS_DB = Path("data/processed/copairs_results.duckdb")

app = typer.Typer()

# Color palettes for consistent styling
PALETTE_BOX = {"Within-source": "#3498db", "Cross-source": "#e74c3c", "Difference": "#3498db"}
PALETTE_STRIP = {"Within-source": "#1a5276", "Cross-source": "#922b21", "Difference": "#1a5276"}


def _plot_boxstrip(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    palette_box: dict | None = None,
    palette_strip: dict | None = None,
    show_mean: bool = False,
    strip_size: float = 3,
    strip_alpha: float = 0.5,
    color_box: str = "#3498db",
    color_strip: str = "#1a5276",
) -> None:
    """Plot boxplot with stripplot overlay using consistent styling.

    Args:
        ax: Matplotlib axes to plot on
        data: DataFrame with the data
        x: Column name for x-axis categories
        y: Column name for y-axis values
        hue: Optional column for grouping within x categories
        order: Order of x-axis categories
        hue_order: Order of hue categories
        palette_box: Colors for boxplot (keyed by hue or x values). If None and hue
            is provided, uses PALETTE_BOX. Ignored if no hue (uses color_box).
        palette_strip: Colors for stripplot (darker). Same logic as palette_box.
        show_mean: If True, annotate mean values above each group
        strip_size: Size of strip plot points
        strip_alpha: Alpha for strip plot points
        color_box: Single color for boxplot when no hue (default blue)
        color_strip: Single color for stripplot when no hue (default darker blue)
    """
    # Determine color strategy based on whether we have a hue
    if hue:
        # Use palettes for hue-based coloring
        box_palette = palette_box or PALETTE_BOX
        strip_palette = palette_strip or PALETTE_STRIP
        box_color = None
        strip_color = None
    else:
        # Use single colors when no hue
        box_palette = None
        strip_palette = None
        box_color = color_box
        strip_color = color_strip

    # Boxplot
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=box_palette,
        color=box_color,
        linewidth=0.8,
        fliersize=0,
        legend="auto" if hue else False,
    )

    # Stripplot overlay
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        palette=strip_palette,
        color=strip_color,
        size=strip_size,
        alpha=strip_alpha,
        dodge=True if hue else False,
        legend=False,
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate means if requested
    if show_mean:
        if hue:
            # Multiple groups per x position
            for i, x_val in enumerate(order or data[x].unique()):
                for j, h_val in enumerate(hue_order or data[hue].unique()):
                    subset = data[(data[x] == x_val) & (data[hue] == h_val)]
                    if len(subset) > 0:
                        mean_val = subset[y].mean()
                        # Offset for dodge
                        n_hue = len(hue_order or data[hue].unique())
                        offset = (j - (n_hue - 1) / 2) * 0.4
                        ax.annotate(
                            f"μ={mean_val:.2f}",
                            xy=(i + offset, ax.get_ylim()[1] * 0.98),
                            ha="center",
                            va="top",
                            fontsize=8,
                        )
        else:
            # Single group per x position
            for i, x_val in enumerate(order or data[x].unique()):
                mean_val = data[data[x] == x_val][y].mean()
                ax.annotate(
                    f"μ={mean_val:.2f}",
                    xy=(i, ax.get_ylim()[1] * 0.98),
                    ha="center",
                    va="top",
                    fontsize=9,
                )


def plot_within_vs_cross_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Scatter plot of within-source vs cross-source normalized mAP per compound-source.

    Colors points by significance status:
    - Both significant (below_corrected_p): blue
    - Only one significant: orange
    - Neither significant: gray

    Args:
        df: DataFrame with nmAP_within, nmAP_cross, sig_within, sig_cross columns
        output_dir: Output directory
        title_suffix: Suffix for plot title (e.g., " (TARGET2)")
        save_pdf: If True, also save PDF versions
        dpi: DPI for saved figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Color by significance
    ax = axes[0]
    both_sig = df[df["sig_within"] & df["sig_cross"]]
    one_sig = df[(df["sig_within"] ^ df["sig_cross"])]  # XOR
    neither_sig = df[~df["sig_within"] & ~df["sig_cross"]]

    # Plot in order: neither (back), one (middle), both (front)
    for subset, color, alpha, label in [
        (neither_sig, "#a0a0a0", 0.3, f"Neither sig (n={len(neither_sig)})"),
        (one_sig, "#ff7f0e", 0.5, f"One sig (n={len(one_sig)})"),
        (both_sig, "#1f77b4", 0.6, f"Both sig (n={len(both_sig)})"),
    ]:
        if len(subset) > 0:
            ax.scatter(
                subset["nmAP_within"],
                subset["nmAP_cross"],
                c=color,
                alpha=alpha,
                s=15,
                label=label,
                linewidths=0,
                rasterized=True,
            )

    ax.set_xlabel("Within-source normalized mAP")
    ax.set_ylabel("Cross-source normalized mAP")
    all_vals = pd.concat([df["nmAP_within"], df["nmAP_cross"]])
    vmin = min(-0.1, all_vals.min() - 0.05)
    vmax = max(1.02, all_vals.max() + 0.02)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Statistics for significant pairs only
    sig_df = both_sig if len(both_sig) > 10 else df
    corr = sig_df["nmAP_within"].corr(sig_df["nmAP_cross"])
    n_above = (sig_df["nmAP_cross"] > sig_df["nmAP_within"]).sum()
    n_below = (sig_df["nmAP_cross"] < sig_df["nmAP_within"]).sum()
    sig_label = "both sig" if len(both_sig) > 10 else "all"
    ax.annotate(
        f"r = {corr:.2f} ({sig_label})\nAbove: {n_above}, Below: {n_below}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set_title(f"By Significance{title_suffix}")

    # Right plot: Color by source (for significant pairs only)
    ax2 = axes[1]
    plot_df = both_sig if len(both_sig) > 10 else df
    sources = sorted(plot_df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    source_colors = dict(zip(sources, colors))

    for source in sources:
        source_df = plot_df[plot_df["Metadata_Source"] == source]
        ax2.scatter(
            source_df["nmAP_within"],
            source_df["nmAP_cross"],
            c=[source_colors[source]],
            alpha=0.5,
            s=20,
            label=source.replace("_", " ").title(),
            linewidths=0,
            rasterized=True,
        )

    ax2.set_xlabel("Within-source normalized mAP")
    ax2.set_ylabel("Cross-source normalized mAP")
    ax2.set_xlim(vmin, vmax)
    ax2.set_ylim(vmin, vmax)
    ax2.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1, label="Identity")
    ax2.set_aspect("equal")
    ax2.legend(loc="lower right", fontsize=8, frameon=True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    filter_label = "Both Significant" if len(both_sig) > 10 else "All"
    ax2.set_title(f"By Source ({filter_label}){title_suffix}")

    plt.tight_layout()
    filename = "within_vs_cross_scatter"
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter plot to {output_dir / f'{filename}.png'}")


def plot_map_comparison_box(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Boxplot comparing mean within-source vs cross-source mAP.

    Args:
        df: DataFrame with nmAP_within, nmAP_cross columns
        output_dir: Output directory
        title_suffix: Suffix for plot title
        save_pdf: If True, also save PDF versions
        dpi: DPI for saved figures
    """
    # Use gridspec to make left plot narrower (1:3 ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]})

    # Reshape data for plotting: melt within/cross into single column
    df_melted = pd.melt(
        df,
        id_vars=["Metadata_Source", "Metadata_JCP2022"],
        value_vars=["nmAP_within", "nmAP_cross"],
        var_name="comparison",
        value_name="nmAP",
    )
    df_melted["comparison"] = df_melted["comparison"].map(
        {
            "nmAP_within": "Within-source",
            "nmAP_cross": "Cross-source",
        }
    )

    # Left: Overall comparison
    ax1 = axes[0]
    order = ["Within-source", "Cross-source"]
    _plot_boxstrip(ax1, df_melted, x="comparison", y="nmAP", order=order, strip_size=2, strip_alpha=0.4)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Within", "Cross"])
    ax1.set_xlabel("")
    ax1.set_ylabel("Normalized mAP")
    ax1.set_ylim(-0.2, 1.05)
    ax1.set_title(f"Overall{title_suffix}")

    # Annotate with mean values
    for i, comp in enumerate(order):
        mean_val = df_melted[df_melted["comparison"] == comp]["nmAP"].mean()
        ax1.annotate(f"μ={mean_val:.2f}", xy=(i, 1.02), ha="center", va="bottom", fontsize=9)

    # Right: Per-source comparison
    ax2 = axes[1]
    sources = sorted(df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    n_compounds = df.groupby("Metadata_Source")["Metadata_JCP2022"].nunique().reindex(sources)

    _plot_boxstrip(
        ax2,
        df_melted,
        x="Metadata_Source",
        y="nmAP",
        hue="comparison",
        order=sources,
        hue_order=order,
        strip_alpha=0.6,
    )

    # X-axis labels with compound counts
    ax2.set_xticks(range(len(sources)))
    labels = [f"{s.replace('_', ' ').title()}\n(n={n_compounds[s]})" for s in sources]
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_xlabel("")
    ax2.set_ylabel("Normalized mAP")
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(loc="upper right", fontsize=9, title="")
    ax2.set_title(f"Per-Source Comparison{title_suffix}")

    plt.tight_layout()
    filename = "map_comparison_box"
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bar chart to {output_dir / f'{filename}.png'}")


def plot_map_diff_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Distribution of mAP difference (within - cross) per compound-source.

    Positive values = within-source better (expected)
    Negative values = cross-source better (unexpected, suggests batch effects)

    Args:
        df: DataFrame with nmAP_within, nmAP_cross columns
        output_dir: Output directory
        title_suffix: Suffix for plot title
        save_pdf: If True, also save PDF versions
        dpi: DPI for saved figures
    """
    df = df.copy()
    df["nmAP_diff"] = df["nmAP_within"] - df["nmAP_cross"]
    df["_category"] = "Difference"  # Dummy category for boxplot

    # Use gridspec to make left plot narrower (1:3 ratio) - consistent with map_comparison_box
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 3]})

    # Left: Overall difference distribution
    ax1 = axes[0]
    _plot_boxstrip(ax1, df, x="_category", y="nmAP_diff", order=["Difference"], strip_size=2, strip_alpha=0.4)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="No difference")
    ax1.set_xlabel("")
    ax1.set_ylabel("Normalized mAP difference\n(within - cross)")
    ax1.set_title(f"Overall{title_suffix}")

    # Annotation with mean (consistent with map_comparison_box) and counts
    n_positive = (df["nmAP_diff"] > 0).sum()
    n_negative = (df["nmAP_diff"] < 0).sum()
    mean_val = df["nmAP_diff"].mean()
    ax1.annotate(
        f"μ={mean_val:.2f}\n"
        f"Within > Cross: {n_positive} ({100 * n_positive / len(df):.1f}%)\n"
        f"Cross > Within: {n_negative} ({100 * n_negative / len(df):.1f}%)",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
    )

    # Right: Per-source boxplot
    ax2 = axes[1]
    sources = sorted(df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))

    _plot_boxstrip(ax2, df, x="Metadata_Source", y="nmAP_diff", order=sources, strip_alpha=0.5)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels([s.replace("_", " ").title() for s in sources], rotation=45, ha="right")
    ax2.set_xlabel("")
    ax2.set_ylabel("Normalized mAP difference (within - cross)")
    ax2.set_title(f"Per-Source Difference{title_suffix}")

    plt.tight_layout()
    filename = "map_diff_distribution"
    fig.savefig(output_dir / f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved difference distribution to {output_dir / f'{filename}.png'}")


def save_summary_stats(
    df: pd.DataFrame,
    output_dir: Path,
    preprocessing: str,
) -> None:
    """Save summary statistics to CSV.

    Args:
        df: DataFrame with nmAP_within, nmAP_cross, sig_within, sig_cross columns
        output_dir: Output directory
        preprocessing: Preprocessing config name
    """
    df = df.copy()
    df["nmAP_diff"] = df["nmAP_within"] - df["nmAP_cross"]

    # Significance counts
    n_both_sig = (df["sig_within"] & df["sig_cross"]).sum()
    n_within_only = (df["sig_within"] & ~df["sig_cross"]).sum()
    n_cross_only = (~df["sig_within"] & df["sig_cross"]).sum()
    n_neither = (~df["sig_within"] & ~df["sig_cross"]).sum()

    # Filter to both significant for main stats
    both_sig = df[df["sig_within"] & df["sig_cross"]]

    # Overall stats
    overall = {
        "preprocessing": preprocessing,
        "n_compound_source_pairs": len(df),
        "n_compounds": df["Metadata_JCP2022"].nunique(),
        "n_sources": df["Metadata_Source"].nunique(),
        # Significance counts
        "n_both_significant": n_both_sig,
        "n_within_only_significant": n_within_only,
        "n_cross_only_significant": n_cross_only,
        "n_neither_significant": n_neither,
        "pct_both_significant": 100 * n_both_sig / len(df),
        # All pairs stats (normalized mAP)
        "mean_nmAP_within_all": df["nmAP_within"].mean(),
        "mean_nmAP_cross_all": df["nmAP_cross"].mean(),
        # The paper's headline "reproducibility drops X% across sources" number: the relative
        # fall in mean normalized mAP from within-source to cross-source, over all pairs. The
        # abstract cites this for the no_source9 config (Issue #24). Named here so it is a
        # first-class, traceable hub record rather than something recomputed from two records.
        "pct_signal_loss_cross_vs_within_all": (
            100 * (df["nmAP_within"].mean() - df["nmAP_cross"].mean()) / df["nmAP_within"].mean()
        ),
        "median_nmAP_diff_all": df["nmAP_diff"].median(),
        "pct_within_better_all": 100 * (df["nmAP_diff"] > 0).mean(),
        "correlation_all": df["nmAP_within"].corr(df["nmAP_cross"]),
    }

    # Stats for both-significant subset
    if len(both_sig) > 0:
        both_sig_diff = both_sig["nmAP_within"] - both_sig["nmAP_cross"]
        overall.update(
            {
                "mean_nmAP_within_bothsig": both_sig["nmAP_within"].mean(),
                "mean_nmAP_cross_bothsig": both_sig["nmAP_cross"].mean(),
                "median_nmAP_diff_bothsig": both_sig_diff.median(),
                "pct_within_better_bothsig": 100 * (both_sig_diff > 0).mean(),
                "correlation_bothsig": both_sig["nmAP_within"].corr(both_sig["nmAP_cross"]),
            }
        )

    pd.DataFrame([overall]).to_csv(output_dir / "summary_overall.csv", index=False)

    # Per-source stats
    per_source = (
        df.groupby("Metadata_Source")
        .agg(
            n_compounds=("Metadata_JCP2022", "nunique"),
            mean_nmAP_within=("nmAP_within", "mean"),
            mean_nmAP_cross=("nmAP_cross", "mean"),
            median_nmAP_diff=("nmAP_diff", "median"),
            pct_within_better=("nmAP_diff", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
    )
    per_source.to_csv(output_dir / "summary_per_source.csv", index=False)

    # Problematic compound-source pairs (cross > within by significant margin)
    problematic = df[df["nmAP_diff"] < -0.1].sort_values("nmAP_diff")
    if len(problematic) > 0:
        problematic.to_csv(output_dir / "problematic_pairs.csv", index=False)
        logger.warning(f"Found {len(problematic)} compound-source pairs where cross > within by >0.1")

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    def _num(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return round(float(v), 4)
        return v

    numbers = {k: _num(v) for k, v in overall.items()}
    numbers["per_source"] = [{k: _num(v) for k, v in rec.items()} for rec in per_source.to_dict("records")]
    write_summary(
        output_dir / "summary.json",
        description=(
            f"Cross-source reproducibility of compound phenotypic activity (Issue #24), config "
            f"'{preprocessing}'. Within-source vs cross-source normalized mAP: compound/source/pair "
            "counts, the within-vs-cross significance breakdown, mean nMAP within and across sources, "
            "the relative signal loss from within- to cross-source (pct_signal_loss_cross_vs_within_all "
            "- the paper's headline 'reproducibility drops X%' number), their correlation, and the "
            "fraction of compound-source pairs where the within-source "
            "phenotype reproduces better than cross-source; with per-source breakdowns."
        ),
        numbers=numbers,
        files=index_files(output_dir),
    )

    logger.info(f"Saved summary statistics to {output_dir}")


def analyze_preprocessing(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    output_dir: Path,
    title_suffix: str,
    filter_name: str = "all_sources",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame:
    """Run full analysis for a single preprocessing config.

    Args:
        con: DuckDB connection
        dataset: Dataset name (e.g., "compound_no_source7")
        preprocessing: Preprocessing config name
        output_dir: Output directory (will create subdir)
        title_suffix: Suffix for plot titles
        filter_name: Filter config name (e.g., "all_sources", "no_source9")
        save_pdf: If True, also save PDF versions
        dpi: DPI for saved figures

    Returns:
        DataFrame with analysis results
    """
    # Include filter in subdir name if not default
    subdir_name = preprocessing if filter_name == "all_sources" else f"{preprocessing}_{filter_name}"
    subdir = output_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)

    df = query_cross_source_reproducibility(con, dataset, preprocessing, filter_name)
    # Column naming: nmAP = normalized mAP (from mean_normalized_average_precision)

    logger.info(f"Loaded {len(df):,} compound-source pairs for {subdir_name}")
    logger.info(f"  {df['Metadata_JCP2022'].nunique()} unique compounds")
    logger.info(f"  {df['Metadata_Source'].nunique()} sources")

    plot_within_vs_cross_scatter(df, subdir, title_suffix, save_pdf, dpi=dpi)
    plot_map_comparison_box(df, subdir, title_suffix, save_pdf, dpi=dpi)
    plot_map_diff_distribution(df, subdir, title_suffix, save_pdf, dpi=dpi)
    save_summary_stats(df, subdir, subdir_name)

    return df


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name (e.g., compound_no_source7, compound_DL_CPCNN_no_source7)",
    ),
    preprocessing: str = typer.Option(
        ...,
        "--preprocessing",
        "-p",
        help="Preprocessing config (e.g., activity_only_target2, activity_poscon_only)",
    ),
    filter_name: str = typer.Option(
        "all_sources",
        "--filter",
        "-f",
        help="Filter config (e.g., all_sources, no_source9)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/processed/cross-source-reproducibility/{dataset})",
    ),
    save_pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Also save PDF versions (for publication)",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Analyze cross-source reproducibility for JUMP Cell Painting dataset."""
    # Default to hierarchical output path
    if output_dir is None:
        output_dir = Path("data/processed/cross-source-reproducibility") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if not COPAIRS_DB.exists():
        logger.error(f"Copairs database not found: {COPAIRS_DB}")
        raise typer.Exit(1)

    con = duckdb.connect(str(COPAIRS_DB), read_only=True)

    try:
        # Build title suffix from preprocessing name
        if "target2" in preprocessing.lower():
            title_suffix = " (TARGET2)"
            if filter_name == "no_source9":
                title_suffix = " (TARGET2, no source_9)"
        elif "poscon" in preprocessing.lower():
            title_suffix = " (Poscons)"
        else:
            title_suffix = f" ({preprocessing})"

        df = analyze_preprocessing(
            con, dataset, preprocessing, output_dir, title_suffix, filter_name, save_pdf, dpi=dpi
        )

        logger.success(f"Analysis complete: {len(df):,} compound-source pairs")

    finally:
        con.close()


if __name__ == "__main__":
    app()
