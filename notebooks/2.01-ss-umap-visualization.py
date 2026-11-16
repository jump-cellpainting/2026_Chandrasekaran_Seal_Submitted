"""UMAP visualization of JUMP compound profiles.

Generates UMAP visualizations from precomputed h5ad files at two levels:
- Well-level: Preserves Source information (replicates span sources)
- Perturbation-level: Aggregated by compound (mean), no Source coloring

Expects minimal h5ad files with UMAP coordinates (X_umap in obsm) and
basic obs columns (JCP2022, Source, Plate, Well). Metadata and activity
data are joined at visualization time from DuckDB databases, not stored
in the h5ad files. This decouples UMAP computation from metadata updates.

Use jump_production.processing.umap to compute UMAP embeddings first.

Usage:
    pixi run python notebooks/2.01-ss-umap-visualization.py compound_no_source7
    pixi run python notebooks/2.01-ss-umap-visualization.py compound_no_source7 --no-clip
    pixi run python notebooks/2.01-ss-umap-visualization.py compound_no_source7 --metric euclidean
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import typer
from loguru import logger

from jump_production.config import DEFAULT_DPI
from jump_production.io import join_activity, load_umap
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

# Set scanpy/anndata settings
sc.settings.verbosity = 1  # reduce verbosity (0=errors, 1=warnings, 2=info, 3=hints)
sc.settings.set_figure_params(dpi=DEFAULT_DPI, frameon=False, figsize=(8, 8))

app = typer.Typer()


def plot_umaps(
    adata: sc.AnnData,
    output_dir: Path,
    dataset_name: str,
    level: str = "well",
    metric: str = "cosine",
    clip_outliers: bool = True,
    filter_type: str = "all",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Generate and save UMAP plots colored by different annotations.

    Args:
        adata: AnnData with UMAP coordinates
        output_dir: Directory for output plots (should be dataset/metric_filter/)
        dataset_name: Dataset name for plot titles
        level: Profile level ('well' or 'perturbation'). Source plot only for well-level.
        metric: Distance metric used for UMAP (for plot titles).
        clip_outliers: Whether to clip extreme outliers using IQR method.
        filter_type: Filter type used for UMAP (for plot titles).
        dpi: DPI for saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file prefix: just level (dataset/metric/filter are in directory path)
    # e.g., well or perturbation
    prefix = level
    level_label = "Perturbation" if level == "perturbation" else "Well"

    # Compute axis bounds for outlier clipping
    xlim, ylim, clip_label = None, None, ""
    if clip_outliers:
        xlim, ylim = compute_umap_bounds(adata)
        coords = adata.obsm["X_umap"]
        n_outside = np.sum(
            (coords[:, 0] < xlim[0]) | (coords[:, 0] > xlim[1]) | (coords[:, 1] < ylim[0]) | (coords[:, 1] > ylim[1])
        )
        pct_outside = 100 * n_outside / len(coords)
        clip_label = f" [{pct_outside:.1f}% clipped]" if n_outside > 0 else ""
        logger.info(f"IQR clipping: {n_outside} points ({pct_outside:.2f}%) outside bounds")

    def apply_bounds(ax):
        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Track color keys for combined plot
    color_keys = []

    # Source coloring only for well-level (perturbation-level aggregates across sources)
    if level == "well" and "Source" in adata.obs.columns:
        logger.info("Plotting UMAP colored by Source")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata, color="Source", ax=ax, show=False, title=f"{dataset_name} ({level_label}) - Source{clip_label}"
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_source.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("Source")

    # Target annotations (available at both levels)
    if "repurposing_target" in adata.obs.columns:
        if "has_repurposing_target" not in adata.obs.columns:
            adata.obs["has_repurposing_target"] = adata.obs["repurposing_target"].notna().astype(str)
        logger.info("Plotting UMAP colored by repurposing target presence")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="has_repurposing_target",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Has Repurposing Target{clip_label}",
            palette={"True": "#e41a1c", "False": "#cccccc"},
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_repurposing_target.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("has_repurposing_target")

    if "Uniprot_target" in adata.obs.columns:
        if "has_uniprot_target" not in adata.obs.columns:
            adata.obs["has_uniprot_target"] = adata.obs["Uniprot_target"].notna().astype(str)
        logger.info("Plotting UMAP colored by Uniprot target presence")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="has_uniprot_target",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Has Uniprot Target{clip_label}",
            palette={"True": "#377eb8", "False": "#cccccc"},
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_uniprot_target.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("has_uniprot_target")

    # Phenotypic activity significance (available at both levels)
    if "below_corrected_p" in adata.obs.columns:
        if "phenotypic_significant" not in adata.obs.columns:
            adata.obs["phenotypic_significant"] = adata.obs["below_corrected_p"].map(
                {"True": "Significant", "False": "Not significant", "NA": "No data"}
            )
        logger.info("Plotting UMAP colored by phenotypic significance")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="phenotypic_significant",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Phenotypic Activity (corrected p < 0.05){clip_label}",
            palette={"Significant": "#e41a1c", "Not significant": "#377eb8", "No data": "#cccccc"},
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_phenotypic_significance.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("phenotypic_significant")

        # Additional plot: only significant compounds (filtered)
        sig_mask = adata.obs["phenotypic_significant"] == "Significant"
        n_sig = sig_mask.sum()
        logger.info(f"Plotting UMAP with only significant compounds ({n_sig:,} / {adata.n_obs:,})")
        fig, ax = plt.subplots(figsize=(10, 8))
        # Create minimal AnnData for subset (X not needed for plotting)
        obs_sig = adata.obs.loc[sig_mask].copy()
        # Remove unused categories to avoid palette KeyError
        obs_sig["phenotypic_significant"] = obs_sig["phenotypic_significant"].cat.remove_unused_categories()
        adata_sig = sc.AnnData(
            obs=obs_sig,
            obsm={"X_umap": adata.obsm["X_umap"][sig_mask.values]},
        )
        sc.pl.umap(
            adata_sig,
            color="phenotypic_significant",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Significant Only ({n_sig:,}){clip_label}",
            palette={"Significant": "#e41a1c"},
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_significant_only.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Create column for combined plot (NaN for non-significant so they don't render)
        if "significant_only" not in adata.obs.columns:
            adata.obs["significant_only"] = adata.obs["phenotypic_significant"].where(sig_mask, other=np.nan)
        color_keys.append("significant_only")

    # Normalized mAP as continuous color (available at both levels)
    if "mean_normalized_average_precision" in adata.obs.columns:
        logger.info("Plotting UMAP colored by normalized mAP")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="mean_normalized_average_precision",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Normalized Mean Average Precision{clip_label}",
            cmap="viridis",
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_nmap.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("mean_normalized_average_precision")

    # Cell count visualization (well-level: per-well count, perturbation-level: median across wells)
    if level == "well" and "Count_Cells" in adata.obs.columns:
        logger.info("Plotting UMAP colored by cell count (per well)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="Count_Cells",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Cell Count per Well{clip_label}",
            cmap="viridis",
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_cell_count.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("Count_Cells")

    if "median_cell_count" in adata.obs.columns:
        logger.info("Plotting UMAP colored by median cell count (per perturbation)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="median_cell_count",
            ax=ax,
            show=False,
            title=f"{dataset_name} ({level_label}) - Median Cell Count per Perturbation{clip_label}",
            cmap="viridis",
        )
        apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_median_cell_count.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        color_keys.append("median_cell_count")

    # Combined plot with multiple panels
    if color_keys:
        logger.info("Plotting combined UMAP panels")
        fig = sc.pl.umap(
            adata, color=color_keys, ncols=min(len(color_keys), 3), show=False, return_fig=True, na_color="none"
        )
        fig.suptitle(f"{dataset_name} ({level_label}){clip_label}", fontsize=14, y=1.02)
        for ax in fig.axes:
            apply_bounds(ax)
        fig.savefig(output_dir / f"{prefix}_umap_combined.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.success(f"Saved {level}-level UMAP plots to {output_dir}")


def plot_level(
    dataset: str,
    level: str,
    metric: str,
    output_base_dir: Path,
    clip_outliers: bool = True,
    activity_preprocessing: str = "activity_no_target2",
    activity_filter: str = "all_sources",
    filter_type: str = "all",
    dpi: int = DEFAULT_DPI,
) -> tuple[int, dict[str, int] | None]:
    """Load precomputed UMAP and generate plots for a single level.

    Uses load_umap() to load h5ad with metadata, then joins activity data.

    Args:
        output_base_dir: Base output directory (plots go to {base}/{dataset}/{metric}_{filter}/)
        filter_type: Filter type used for UMAP ('all' or 'active').
        dpi: DPI for saved figures.
    """
    # Load UMAP with metadata
    adata = load_umap(dataset, level=level, metric=metric, filter_name=filter_type)

    # Verify UMAP coordinates exist
    if "X_umap" not in adata.obsm:
        raise ValueError(f"X_umap not found for {dataset}. Run UMAP computation first.")

    # Join activity data for phenotypic significance visualization
    adata = join_activity(adata, dataset, preprocessing=activity_preprocessing, filter_name=activity_filter)

    # Output directory: {base}/{dataset}/{metric}_{filter}/
    output_dir = output_base_dir / dataset / f"{metric}_{filter_type}"

    # Generate plots
    plot_umaps(
        adata,
        output_dir,
        dataset,
        level=level,
        metric=metric,
        clip_outliers=clip_outliers,
        filter_type=filter_type,
        dpi=dpi,
    )

    # Well-level points carry their data-generating source; expose the per-source split so the
    # count plotted in each partner's color (Figure 2 well-level source UMAP) is citable, not just
    # the total. Perturbation-level profiles aggregate across sources, so there is no split there.
    source_counts = None
    if level == "well" and "Source" in adata.obs.columns:
        counts = adata.obs["Source"].value_counts()
        source_counts = {
            str(source): int(counts[source])
            for source in sorted(counts.index, key=lambda value: int(str(value).split("_")[-1]))
        }

    return int(adata.n_obs), source_counts


@app.command()
def main(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., compound_no_source7)"),
    metric: str = typer.Option("cosine", "--metric", "-m", help="Distance metric used for UMAP (cosine, euclidean)"),
    output_dir: Path = typer.Option(
        Path("data/processed/umap"),
        "--output",
        "-o",
        help="Output directory for figures",
    ),
    activity_preprocessing: str = typer.Option(
        "activity_no_target2",
        "--activity-preprocessing",
        help="Activity preprocessing config name for phenotypic significance data",
    ),
    activity_filter: str = typer.Option(
        "all_sources",
        "--activity-filter",
        help="Activity filter config name for phenotypic significance data",
    ),
    no_clip: bool = typer.Option(False, "--no-clip", help="Disable IQR-based outlier clipping"),
    filter_type: str = typer.Option(
        "all", "--filter", "-f", help="Filter type for input/output files ('all' or 'active')"
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Generate UMAP visualizations from precomputed h5ad files.

    Uses load_umap() to load h5ad files with metadata joined from DuckDB.
    Activity data is also joined at plot time from copairs_results.duckdb.
    """
    # Process both levels
    n_obs_by_level = {}
    n_points_by_source = None
    for level in ["well", "perturbation"]:
        n_obs, source_counts = plot_level(
            dataset=dataset,
            level=level,
            metric=metric,
            output_base_dir=output_dir,
            clip_outliers=not no_clip,
            activity_preprocessing=activity_preprocessing,
            activity_filter=activity_filter,
            filter_type=filter_type,
            dpi=dpi,
        )
        n_obs_by_level[level] = n_obs
        if source_counts is not None:
            n_points_by_source = source_counts

    final_dir = output_dir / dataset / f"{metric}_{filter_type}"
    logger.success(f"Done with both well-level and perturbation-level UMAP plots: {final_dir}")

    # Emit the standard summary envelope so the dataset-reference hub discovers this leaf.
    write_summary(
        final_dir / "summary.json",
        description=(
            f"UMAP visualizations of {dataset} profiles (metric={metric}, filter={filter_type}) at "
            "well and perturbation level, colored by source, target annotations, phenotypic "
            "significance, normalized mAP, and cell count. Primarily a figure node; reports the number "
            "of points plotted at each level, and the well-level split per data-generating source "
            "(the per-partner sample-well counts shown in the Figure 2 well-level source UMAP)."
        ),
        numbers={
            "dataset": dataset,
            "metric": metric,
            "filter_type": filter_type,
            "n_points_by_level": n_obs_by_level,
            "n_points_by_source": n_points_by_source,
        },
        files=index_files(final_dir),
    )


if __name__ == "__main__":
    app()
