"""Cell count quality analyses for JUMP Cell Painting dataset.

Two analyses:
1. Cell count vs phenotypic activity - Is activity driven by toxicity?
2. Edge well effects - Do edge wells show different cell counts?

Usage:
    pixi run python notebooks/2.02b-ss-cell-count-quality.py
    pixi run python notebooks/2.02b-ss-cell-count-quality.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/data-quality/{dataset}/{preprocessing}/
"""

import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import typer
from loguru import logger
from scipy import stats

from jump_production.config import DEFAULT_DPI
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import (
    plot_hexbin_with_marginals,
    plot_scatter_with_marginals,
)


def _csv_row(path: Path) -> dict | None:
    """Read a single-row summary CSV into a JSON-safe dict (or None if absent)."""
    if not path.exists():
        return None
    out = {}
    for k, v in pd.read_csv(path).iloc[0].to_dict().items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 6)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out

# Database paths
COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")

app = typer.Typer()


def query_cell_counts(
    con: duckdb.DuckDBPyConnection,
    plate_format: int = 384,
    plate_type: str = "COMPOUND",
    normalize_by_area: bool = True,
) -> pd.DataFrame:
    """Query cell counts from metadata database.

    Args:
        con: DuckDB connection
        plate_format: Filter to sources using this plate format (384, 1536, or 0 for all).
        plate_type: Filter to this plate type (e.g., "COMPOUND"). Use None for all.
        normalize_by_area: If True, divide cell counts by well area (in mm²).
    """
    # Sources by plate format (based on well area and n_unique_wells)
    # 384-well (~384 unique wells, well area ~2.5-5.3 mm²)
    # 1536-well (~1472-1536 unique wells, well area ~1.67 mm²)
    sources_384 = ["source_2", "source_3", "source_5", "source_6", "source_8", "source_10", "source_11"]
    sources_1536 = ["source_1", "source_9"]

    if plate_format == 384:
        source_filter = sources_384
    elif plate_format == 1536:
        source_filter = sources_1536
    else:
        source_filter = None

    params = []
    where_clauses = []

    if source_filter:
        placeholders = ", ".join(["?" for _ in source_filter])
        where_clauses.append(f"c.Metadata_Source IN ({placeholders})")
        params.extend(source_filter)

    if plate_type:
        where_clauses.append("p.Metadata_PlateType = ?")
        params.append(plate_type)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
        SELECT
            c.Metadata_Source,
            c.Metadata_Plate,
            c.Metadata_Well,
            c.Metadata_Count_Cells,
            m.Metadata_Sites_Per_Well,
            m.Metadata_Well_Area_Microns2,
            p.Metadata_PlateType
        FROM cell_counts c
        LEFT JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
        LEFT JOIN plate p ON c.Metadata_Plate = p.Metadata_Plate
        WHERE {where_sql}
    """
    df = con.execute(query, params).df()

    if normalize_by_area and len(df) > 0:
        df["Metadata_Well_Area_mm2"] = df["Metadata_Well_Area_Microns2"] / 1e6
        df["Metadata_Count_Cells_Per_mm2"] = df["Metadata_Count_Cells"] / df["Metadata_Well_Area_mm2"]
        logger.info(
            f"Normalized cell counts by well area (range: "
            f"{df['Metadata_Well_Area_mm2'].min():.2f}-{df['Metadata_Well_Area_mm2'].max():.2f} mm²)"
        )

    return df


def query_cell_density(
    con: duckdb.DuckDBPyConnection,
    exclude_sources: list[str] | None = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Query cell density (cells/mm²) from the database.

    Cell counts are normalized by well area to account for different imaging
    configurations across sources (well areas range from 1.67 to 5.26 mm²).

    Args:
        con: DuckDB connection
        exclude_sources: List of sources to exclude (e.g., ["source_2"])
        aggregate: If True, return median per compound. If False, return well-level data.
    """
    where_clause = "w.Metadata_JCP2022 IS NOT NULL"
    params = []

    if exclude_sources:
        placeholders = ", ".join(["?" for _ in exclude_sources])
        where_clause += f" AND c.Metadata_Source NOT IN ({placeholders})"
        params.extend(exclude_sources)

    if aggregate:
        query = f"""
            SELECT
                w.Metadata_JCP2022,
                MEDIAN(c.Metadata_Count_Cells / (m.Metadata_Well_Area_Microns2 / 1e6)) as Metadata_median_cell_density
            FROM well w
            JOIN cell_counts c ON w.Metadata_Plate = c.Metadata_Plate AND w.Metadata_Well = c.Metadata_Well
            JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
            WHERE {where_clause}
            GROUP BY w.Metadata_JCP2022
        """
    else:
        query = f"""
            SELECT
                w.Metadata_JCP2022,
                c.Metadata_Source,
                c.Metadata_Plate,
                c.Metadata_Well,
                c.Metadata_Count_Cells / (m.Metadata_Well_Area_Microns2 / 1e6) as cell_density
            FROM well w
            JOIN cell_counts c ON w.Metadata_Plate = c.Metadata_Plate AND w.Metadata_Well = c.Metadata_Well
            JOIN microscope_config_fov m ON c.Metadata_Source = m.Metadata_Source
            WHERE {where_clause}
            -- Source regression roundoff can change rank ties if well order varies.
            ORDER BY c.Metadata_Source, c.Metadata_Plate, c.Metadata_Well, w.Metadata_JCP2022
        """
    return con.execute(query, params).df()


def normalize_cell_density(df: pd.DataFrame, value_col: str = "cell_density") -> pd.DataFrame:
    """Two-step normalization: regress out source, then robust z-score by plate.

    Step 1: Regress out source effects (removes systematic lab/protocol differences)
    Step 2: Robust z-score by plate (centers and scales within each plate)

    Args:
        df: DataFrame with Metadata_Source, Metadata_Plate, and value_col columns
        value_col: Column name containing raw cell density values

    Returns:
        DataFrame with {value_col}_normalized column
    """
    df = df.copy()

    # Step 1: Regress out source effects
    model = smf.ols(f"{value_col} ~ C(Metadata_Source)", data=df).fit()
    source_adj = model.resid
    logger.info(f"Regressed out source effects (R² = {model.rsquared:.3f})")

    # Step 2: Robust z-score by plate
    def _robust_zscore(x: pd.Series) -> pd.Series:
        median = x.median()
        mad = (x - median).abs().median()
        if mad > 0:
            return (x - median) / (mad * 1.4826)
        std = x.std()
        if std > 0:
            return (x - median) / std
        return x - median

    df[f"{value_col}_normalized"] = (
        pd.Series(source_adj, index=df.index).groupby(df["Metadata_Plate"]).transform(_robust_zscore)
    )

    return df


def aggregate_compound_cell_density(
    df: pd.DataFrame,
    value_col: str = "cell_density_normalized",
    output_col: str = "Metadata_median_cell_density",
) -> pd.DataFrame:
    """Aggregate well-level data to compound-level by taking the median.

    Args:
        df: Well-level DataFrame with Metadata_JCP2022 and value_col
        value_col: Column to aggregate (default matches normalize_cell_density output)
        output_col: Name for the output column
    """
    return df.groupby("Metadata_JCP2022")[value_col].median().reset_index(name=output_col)


def parse_well_position(well: str) -> tuple[str, int]:
    """Parse well ID like 'A01' into row letter and column number."""
    match = re.match(r"([A-Z]+)(\d+)", well)
    if match:
        return match.group(1), int(match.group(2))
    return well, 0


def analyze_cell_density_vs_activity(
    activity_df: pd.DataFrame,
    cell_density_df: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool = False,
    suffix: str = "",
    density_label: str = "Median cell density (cells/mm²)",
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame:
    """Analyze relationship between cell density and phenotypic activity.

    Note: Per-source analysis is not possible here because phenotypic activity
    (mAP) is computed at the compound level by aggregating replicates across
    all sources. Cell density is also aggregated as the median across all wells
    for each compound. Therefore, activity cannot be disaggregated by source.

    Args:
        activity_df: Compound-level activity results
        cell_density_df: Compound-level cell density (can be filtered by source)
        output_dir: Output directory
        save_pdf: If True, also save PDF versions
        suffix: Suffix for output filenames (e.g., "_excl_source2")
        density_label: X-axis label for cell density
        dpi: DPI for saved figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = activity_df.merge(cell_density_df, on="Metadata_JCP2022", how="inner")
    logger.info(f"Merged {len(merged):,} compounds with both activity and cell density data")

    # Publication-ready styling
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )

    sig = merged[merged["below_corrected_p"]]
    nonsig = merged[~merged["below_corrected_p"]]

    valid = merged.dropna(subset=["Metadata_median_cell_density", "mean_normalized_average_precision"])
    r, p = stats.spearmanr(valid["Metadata_median_cell_density"], valid["mean_normalized_average_precision"])

    # Mann-Whitney U test: compare cell densities between significant vs non-significant compounds
    mw_stat, mw_p = stats.mannwhitneyu(
        sig["Metadata_median_cell_density"].dropna(),
        nonsig["Metadata_median_cell_density"].dropna(),
        alternative="two-sided",
    )
    logger.info(f"Mann-Whitney U test: statistic={mw_stat:.0f}, p={mw_p:.2e}")

    # Format stats annotation for both plots
    mw_str = f"p < {mw_p:.0e}" if mw_p < 0.001 else f"p = {mw_p:.3f}"
    stats_annotation = f"Spearman ρ = {r:.2f}\nMann-Whitney {mw_str}\nn = {len(valid):,}"

    # Hexbin density plot (shows overall distribution)
    plot_hexbin_with_marginals(
        merged,
        x_col="Metadata_median_cell_density",
        x_label=density_label,
        stats_annotation=stats_annotation,
        output_dir=output_dir,
        filename=f"cell_density_vs_map_hexbin{suffix}.png",
        save_pdf=save_pdf,
        dpi=dpi,
    )
    logger.info(f"Saved hexbin plot to {output_dir / f'cell_density_vs_map_hexbin{suffix}.png'}")

    # Scatter plot with marginals (shows sig vs nonsig comparison)
    plot_scatter_with_marginals(
        merged,
        x_col="Metadata_median_cell_density",
        x_label=density_label,
        output_dir=output_dir,
        filename=f"cell_density_vs_map_scatter{suffix}.png",
        save_pdf=save_pdf,
        dpi=dpi,
    )
    logger.info(f"Saved scatter plot to {output_dir / f'cell_density_vs_map_scatter{suffix}.png'}")

    # Summary statistics
    summary = {
        "n_compounds": len(merged),
        "spearman_r": r,
        "spearman_p": p,
        "n_significant": len(sig),
        "n_not_significant": len(nonsig),
        "median_cell_density_sig": sig["Metadata_median_cell_density"].median(),
        "median_cell_density_nonsig": nonsig["Metadata_median_cell_density"].median(),
        "mannwhitney_statistic": mw_stat,
        "mannwhitney_p": mw_p,
    }
    pd.DataFrame([summary]).to_csv(output_dir / f"summary{suffix}.csv", index=False)

    return merged


def analyze_cell_density_distribution(
    raw_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Compare compound-level cell density distributions: raw vs plate-normalized.

    Shows whether bimodality in raw cell density is due to plate/source batch effects
    or reflects true compound-level differences.

    Args:
        raw_df: Compound-level raw cell density (Metadata_median_cell_density in cells/mm²)
        normalized_df: Compound-level plate-normalized cell density (robust z-score)
        output_dir: Output directory for plots
        save_pdf: If True, also save PDF versions
        dpi: DPI for saved figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Raw cell density distribution
    ax1 = axes[0]
    raw_vals = raw_df["Metadata_median_cell_density"].dropna()
    bins_raw = np.linspace(raw_vals.quantile(0.01), raw_vals.quantile(0.99), 50)
    ax1.hist(raw_vals, bins=bins_raw, color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.axvline(
        raw_vals.median(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {raw_vals.median():.0f}"
    )
    ax1.set_xlabel("Cell density (cells/mm²)")
    ax1.set_ylabel("Count")
    ax1.set_title("Raw cell density")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Plate-normalized distribution
    ax2 = axes[1]
    norm_vals = normalized_df["Metadata_median_cell_density"].dropna()
    bins_norm = np.linspace(norm_vals.quantile(0.01), norm_vals.quantile(0.99), 50)
    ax2.hist(norm_vals, bins=bins_norm, color="#2ecc71", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(
        norm_vals.median(), color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {norm_vals.median():.2f}"
    )
    ax2.axvline(0, color="#7f8c8d", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("Plate-normalized cell density (robust z-score)")
    ax2.set_ylabel("Count")
    ax2.set_title("Plate-normalized cell density")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add sample size annotation
    fig.suptitle(f"Compound-level cell density distribution (n={len(raw_vals):,} compounds)", fontsize=11, y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / "cell_density_distribution_comparison.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / "cell_density_distribution_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Log statistics
    logger.info(f"Raw cell density: median={raw_vals.median():.0f}, std={raw_vals.std():.0f}")
    logger.info(f"Plate-normalized: median={norm_vals.median():.2f}, std={norm_vals.std():.2f}")
    logger.info(f"Saved distribution comparison to {output_dir / 'cell_density_distribution_comparison.png'}")


def _plot_edge_effects(
    viz_df: pd.DataFrame,
    count_col: str,
    count_label: str,
    output_dir: Path,
    suffix: str = "",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot heatmaps, marginals, and violin plots for edge effects analysis.

    Args:
        viz_df: DataFrame with row/col parsed and Metadata_Source, Metadata_Well_Area_mm2
        count_col: Column name for cell count metric
        count_label: Label for axes
        output_dir: Output directory
        suffix: Suffix for output filenames (e.g., "_excl_source2")
        save_pdf: If True, also save PDF versions (for publication)
    """
    sources = sorted(viz_df["Metadata_Source"].unique(), key=lambda x: int(x.split("_")[1]))
    n_sources = len(sources)
    vmin = viz_df[count_col].quantile(0.01)
    vmax = viz_df[count_col].quantile(0.99)

    # --- Heatmaps ---
    fig, axes = plt.subplots(n_sources, 1, figsize=(8, 1.5 * n_sources), gridspec_kw={"hspace": 0.3})
    if n_sources == 1:
        axes = [axes]

    for i, source in enumerate(sources):
        source_df = viz_df[viz_df["Metadata_Source"] == source]
        source_means = source_df.groupby(["row", "col"])[count_col].mean().reset_index()
        source_means.columns = ["row", "col", "mean_cell_count"]
        src_rows = sorted(source_means["row"].unique(), key=lambda x: (len(x), x))
        well_area_mm2 = source_df["Metadata_Well_Area_mm2"].iloc[0]

        pivot = source_means.pivot(index="row", columns="col", values="mean_cell_count")
        pivot = pivot.reindex(src_rows)[sorted(pivot.columns)]

        ax = axes[i]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlGnBu",
            square=True,
            linewidths=0.3,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_ylabel("")
        ax.set_xlabel(f"{source.replace('_', ' ').title()} ({well_area_mm2:.1f} mm²)", fontsize=8)
        ax.tick_params(axis="both", which="both", length=0, labelsize=6)

    fig.subplots_adjust(bottom=0.06)
    cbar_ax = fig.add_axes([0.4, 0.015, 0.2, 0.008])
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").set_label(count_label, fontsize=9)

    fig.savefig(output_dir / f"cell_count_heatmap_by_source{suffix}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"cell_count_heatmap_by_source{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Marginal bar plots ---
    fig, axes = plt.subplots(n_sources, 2, figsize=(8, 1.5 * n_sources), gridspec_kw={"hspace": 0.1, "wspace": 0.15})
    if n_sources == 1:
        axes = axes.reshape(1, -1)

    for i, source in enumerate(sources):
        source_df = viz_df[viz_df["Metadata_Source"] == source]
        src_rows = sorted(source_df["row"].unique(), key=lambda x: (len(x), x))
        well_area_mm2 = source_df["Metadata_Well_Area_mm2"].iloc[0]

        row_means = source_df.groupby("row")[count_col].mean().reindex(src_rows)
        ax1 = axes[i, 0]
        ax1.barh(range(len(row_means)), row_means.values, color="#3498db", alpha=0.8, height=0.8)
        ax1.set_yticks(range(0, len(row_means), max(1, len(row_means) // 8)))
        ax1.set_yticklabels([src_rows[j] for j in range(0, len(row_means), max(1, len(row_means) // 8))], fontsize=6)
        ax1.set_xlim(vmin * 0.9, vmax * 1.1)
        ax1.invert_yaxis()
        ax1.axvline(row_means.mean(), color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax1.set_ylabel(f"{source.replace('_', ' ').title()} ({well_area_mm2:.1f} mm²)", fontsize=7)
        ax1.tick_params(axis="x", labelsize=6)

        col_means = source_df.groupby("col")[count_col].mean().sort_index()
        ax2 = axes[i, 1]
        ax2.bar(col_means.index.values, col_means.values, width=0.8, color="#3498db", alpha=0.8)
        ax2.set_xlim(0, col_means.index.max() + 1)
        ax2.set_ylim(vmin * 0.9, vmax * 1.1)
        ax2.axhline(col_means.mean(), color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax2.tick_params(axis="both", labelsize=6)

        if i < n_sources - 1:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        else:
            ax2.set_xticks(range(0, col_means.index.max() + 1, 4))

    axes[-1, 0].set_xlabel(count_label, fontsize=8)
    axes[-1, 1].set_xlabel("Column", fontsize=8)
    axes[0, 0].set_title("Row marginals", fontsize=9)
    axes[0, 1].set_title("Column marginals", fontsize=9)
    fig.text(0.98, 0.5, count_label, va="center", rotation=-90, fontsize=8)

    fig.savefig(output_dir / f"position_marginals{suffix}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"position_marginals{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Violin plots ---
    fig, axes = plt.subplots(n_sources, 2, figsize=(14, 2 * n_sources), gridspec_kw={"hspace": 0.4, "wspace": 0.2})
    if n_sources == 1:
        axes = axes.reshape(1, -1)

    for i, source in enumerate(sources):
        source_df = viz_df[viz_df["Metadata_Source"] == source]
        src_rows = sorted(source_df["row"].unique(), key=lambda x: (len(x), x))
        src_cols = sorted(source_df["col"].unique())
        well_area_mm2 = source_df["Metadata_Well_Area_mm2"].iloc[0]

        for ax, x_var, order in [
            (axes[i, 0], "row", src_rows),
            (axes[i, 1], "col", src_cols),
        ]:
            sns.violinplot(
                data=source_df,
                x=x_var,
                y=count_col,
                order=order,
                ax=ax,
                color="#3498db",
                linewidth=0.5,
                inner="box",
                density_norm="width",
            )
            ax.axhline(source_df[count_col].mean(), color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.set_ylim(vmin * 0.8, vmax * 1.2)

        axes[i, 0].set_ylabel(f"{source.replace('_', ' ').title()}\n({well_area_mm2:.1f} mm²)", fontsize=8)
        axes[i, 1].set_ylabel("")

        if i < n_sources - 1:
            axes[i, 0].set_xticklabels([])
            axes[i, 1].set_xticklabels([])

    axes[-1, 0].set_xlabel("Row", fontsize=9)
    axes[-1, 1].set_xlabel("Column", fontsize=9)
    axes[0, 0].set_title("Row distribution", fontsize=10)
    axes[0, 1].set_title("Column distribution", fontsize=10)
    fig.text(0.02, 0.5, count_label, va="center", rotation=90, fontsize=9)

    fig.savefig(output_dir / f"position_marginals_violin{suffix}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"position_marginals_violin{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved edge effects plots with suffix '{suffix}' to {output_dir}")


def analyze_edge_effects(
    cell_count_df: pd.DataFrame,
    output_dir: Path,
    plate_format: int = 384,
    use_normalized: bool = True,
    treatment_cols_only: bool = True,
    all_sources_df: pd.DataFrame | None = None,
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame:
    """Analyze edge well effects on cell counts.

    Args:
        cell_count_df: Well-level cell counts for edge statistics (typically 384-well only)
        output_dir: Output directory for plots and summaries
        plate_format: Plate format (384 or 1536) for the main analysis
        use_normalized: If True, use Metadata_Count_Cells_Per_mm2
        treatment_cols_only: If True, exclude control columns (1, 2, 23, 24)
        all_sources_df: If provided, use for per-source visualizations
        save_pdf: If True, also save PDF versions (for publication)
        dpi: DPI for saved figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = cell_count_df.copy()
    positions = df["Metadata_Well"].apply(parse_well_position)
    df["row"] = positions.apply(lambda x: x[0])
    df["col"] = positions.apply(lambda x: x[1])

    if treatment_cols_only and plate_format == 384:
        control_cols = {1, 2, 23, 24}
        n_before = len(df)
        df = df[~df["col"].isin(control_cols)]
        logger.info(f"Filtered to treatment columns (3-22): {n_before:,} -> {len(df):,} wells")

    if use_normalized and "Metadata_Count_Cells_Per_mm2" in df.columns:
        count_col = "Metadata_Count_Cells_Per_mm2"
        count_label = "Cell Density (cells/mm²)"
        logger.info("Using normalized cell counts (cells per mm² of imaged area)")
    else:
        count_col = "Metadata_Count_Cells"
        count_label = "Cell Count"

    unique_rows = sorted(df["row"].unique(), key=lambda x: (len(x), x))
    unique_cols = sorted(df["col"].unique())

    logger.info(f"Found {len(unique_rows)} rows x {len(unique_cols)} columns ({plate_format}-well format)")

    position_means = df.groupby(["row", "col"])[count_col].mean().reset_index()
    position_means.columns = ["row", "col", "mean_cell_count"]

    # Prepare visualization dataframe
    if all_sources_df is not None:
        viz_df = all_sources_df.copy()
        positions = viz_df["Metadata_Well"].apply(parse_well_position)
        viz_df["row"] = positions.apply(lambda x: x[0])
        viz_df["col"] = positions.apply(lambda x: x[1])
        logger.info(f"Using all sources for per-source visualizations ({len(viz_df):,} wells)")
    else:
        viz_df = df

    # Plot all sources
    _plot_edge_effects(viz_df, count_col, count_label, output_dir, suffix="", save_pdf=save_pdf, dpi=dpi)

    # Plot excluding Source 2 (outlier with ~3x higher cell density)
    viz_df_excl = viz_df[viz_df["Metadata_Source"] != "source_2"]
    _plot_edge_effects(
        viz_df_excl, count_col, count_label, output_dir, suffix="_excl_source2", save_pdf=save_pdf, dpi=dpi
    )

    # Edge vs interior comparison
    edge_rows = set([unique_rows[0], unique_rows[-1]])
    edge_cols = set([unique_cols[0], unique_cols[-1]])

    df["is_edge"] = df["row"].isin(edge_rows) | df["col"].isin(edge_cols)

    edge_mean = df[df["is_edge"]][count_col].mean()
    interior_mean = df[~df["is_edge"]][count_col].mean()
    edge_n = df["is_edge"].sum()
    interior_n = (~df["is_edge"]).sum()

    t_stat, t_pval = stats.ttest_ind(
        df[df["is_edge"]][count_col],
        df[~df["is_edge"]][count_col],
    )

    summary = {
        "n_wells": len(df),
        "n_rows": len(unique_rows),
        "n_cols": len(unique_cols),
        "plate_format": plate_format,
        "normalized_by_area": use_normalized and "Metadata_Count_Cells_Per_mm2" in df.columns,
        "edge_mean_cell_density": edge_mean,
        "interior_mean_cell_density": interior_mean,
        "edge_n_wells": edge_n,
        "interior_n_wells": interior_n,
        "edge_effect_pct": (interior_mean - edge_mean) / interior_mean * 100,
        "ttest_statistic": t_stat,
        "ttest_pvalue": t_pval,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    logger.info(f"Edge wells: mean={edge_mean:.1f}, n={edge_n:,}")
    logger.info(f"Interior wells: mean={interior_mean:.1f}, n={interior_n:,}")
    logger.info(f"Edge effect: {summary['edge_effect_pct']:.1f}% fewer cells on edges (p={t_pval:.2e})")

    return position_means


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name",
    ),
    preprocessing: str = typer.Option(
        "activity_no_target2",
        "--preprocessing",
        "-p",
        help="Activity preprocessing config",
    ),
    filter_name: str = typer.Option(
        "all_sources",
        "--filter",
        "-f",
        help="Filter config",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/processed/data-quality/{dataset}/{preprocessing})",
    ),
    save_pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Also save PDF versions (for publication)",
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
    """Analyze cell count quality metrics for JUMP Cell Painting dataset."""
    # Use hierarchical output path: data-quality/{dataset}/{preprocessing}/
    if output_dir is None:
        output_dir = Path("data/processed/data-quality") / dataset / preprocessing
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Preprocessing: {preprocessing}")
    logger.info(f"Filter: {filter_name}")
    logger.info(f"Output: {output_dir}")

    if not COPAIRS_DB.exists():
        logger.error(f"Copairs database not found: {COPAIRS_DB}")
        raise typer.Exit(1)
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    copairs_con = duckdb.connect(str(COPAIRS_DB), read_only=True)
    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Cell density vs activity analysis
        logger.info("=== Cell Density vs Activity Analysis ===")
        activity_df = query_activity_results(copairs_con, dataset, preprocessing, filter_name, activity_params)
        logger.info(f"Loaded {len(activity_df):,} compound activity scores")

        cell_density_output = output_dir / "cell_density_activity"

        # Cell density vs activity analysis progression:
        # 1a. Raw: Simple median cell density per compound (baseline, shows confounding)
        # 1b. Exclude source_2: Remove outlier source to see if it drives the pattern
        # 1c. Normalized: Regress out source + z-score by plate (proper confounder adjustment)
        # This progression helps diagnose whether correlations are real or driven by batch effects.

        # 1a. Raw cell density (baseline - shows confounding from source/plate effects)
        cell_density_df = query_cell_density(metadata_con, aggregate=True)
        logger.info(f"Loaded {len(cell_density_df):,} compound cell densities (all sources)")
        analyze_cell_density_vs_activity(
            activity_df, cell_density_df, cell_density_output, save_pdf=save_pdf, suffix="", dpi=dpi
        )

        # 1b. Excluding source_2 (test if one outlier source drives the pattern)
        cell_density_df_excl = query_cell_density(metadata_con, exclude_sources=["source_2"], aggregate=True)
        logger.info(f"Loaded {len(cell_density_df_excl):,} compound cell densities (excluding source_2)")
        analyze_cell_density_vs_activity(
            activity_df, cell_density_df_excl, cell_density_output, save_pdf=save_pdf, suffix="_excl_source2", dpi=dpi
        )

        # 1c. Normalized: regress out source effects, then z-score by plate
        logger.info("=== Source-Adjusted Plate-Normalized Cell Density Analysis ===")
        well_level_df = query_cell_density(metadata_con, aggregate=False)
        logger.info(f"Loaded {len(well_level_df):,} well-level cell densities")

        well_level_df = normalize_cell_density(well_level_df, value_col="cell_density")
        norm_col = "cell_density_normalized"
        logger.info(
            f"Applied two-step normalization "
            f"(range: {well_level_df[norm_col].min():.2f} to {well_level_df[norm_col].max():.2f})"
        )

        cell_density_normalized = aggregate_compound_cell_density(
            well_level_df, value_col=norm_col, output_col="Metadata_median_cell_density"
        )
        logger.info(f"Aggregated to {len(cell_density_normalized):,} compounds (source-adjusted, plate-normalized)")
        analyze_cell_density_vs_activity(
            activity_df,
            cell_density_normalized,
            cell_density_output,
            save_pdf=save_pdf,
            suffix="_normalized",
            density_label="Normalized cell density (source-adjusted, plate z-score)",
            dpi=dpi,
        )

        # 1d. Cell density distribution comparison (raw vs normalized)
        logger.info("=== Cell Density Distribution Analysis ===")
        analyze_cell_density_distribution(
            cell_density_df, cell_density_normalized, cell_density_output, save_pdf=save_pdf, dpi=dpi
        )

        # 2. Edge effects analysis
        logger.info("=== Edge Effects Analysis ===")

        well_cell_counts_all = query_cell_counts(metadata_con, plate_format=0, plate_type="COMPOUND")
        logger.info(f"Loaded {len(well_cell_counts_all):,} well-level cell counts from all COMPOUND plates")

        well_cell_counts_384 = query_cell_counts(metadata_con, plate_format=384, plate_type="COMPOUND")
        logger.info(f"Loaded {len(well_cell_counts_384):,} well-level cell counts from 384-well COMPOUND plates")

        edge_output = output_dir / "edge_effects"
        analyze_edge_effects(
            well_cell_counts_384,
            edge_output,
            plate_format=384,
            treatment_cols_only=False,  # Include all columns to see true edge effects
            all_sources_df=well_cell_counts_all,
            save_pdf=save_pdf,
            dpi=dpi,
        )

        # Emit the standard summary envelope so the dataset-reference hub discovers this node.
        write_summary(
            output_dir / "summary.json",
            description=(
                "Cell-count quality checks. (1) Cell density vs phenotypic activity (is activity "
                "driven by toxicity?): Spearman correlation and a significant-vs-nonsignificant "
                "cell-density comparison, reported raw, excluding the outlier source_2, and after "
                "source-adjusted plate normalization. (2) Edge-well effects: whether plate-edge wells "
                "differ in cell density from interior wells (edge-vs-interior gap and t-test)."
            ),
            numbers={
                "dataset": dataset,
                "preprocessing": preprocessing,
                "cell_density_vs_activity": {
                    "raw": _csv_row(cell_density_output / "summary.csv"),
                    "excl_source2": _csv_row(cell_density_output / "summary_excl_source2.csv"),
                    "normalized": _csv_row(cell_density_output / "summary_normalized.csv"),
                },
                "edge_effects": _csv_row(edge_output / "summary.csv"),
            },
            files=index_files(cell_density_output) + index_files(edge_output),
        )

        logger.success(f"All outputs saved to {output_dir}")

    finally:
        copairs_con.close()
        metadata_con.close()


if __name__ == "__main__":
    app()
