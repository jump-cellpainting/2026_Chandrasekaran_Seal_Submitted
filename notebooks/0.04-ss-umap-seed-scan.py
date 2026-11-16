"""UMAP Seed Stability Scan: Cosine vs Euclidean.

Runs multiple UMAP seeds for both cosine and euclidean metrics to verify
whether the "encircling" pattern in cosine UMAP is reproducible or seed-dependent.

Purpose: Address issue #17 - compare UMAP structure stability across random seeds.

Usage:
    pixi run -e rapids python notebooks/0.04-ss-umap-seed-scan.py
    pixi run -e rapids python notebooks/0.04-ss-umap-seed-scan.py --dataset compound_no_source7
    CUDA_VISIBLE_DEVICES="" pixi run python notebooks/0.04-ss-umap-seed-scan.py  # CPU fallback

Outputs:
    data/processed/exploration/0.04/{dataset}/seed_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import typer
from loguru import logger

from jump_production.config import (
    DEFAULT_DPI,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from jump_production.io import join_activity, load_profiles
from jump_production.processing.umap import compute_umap
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

# Output directories - exploration structure with dataset nesting
SEED_SCAN_DATA_DIR = INTERIM_DATA_DIR / "anndata" / "seed_scan"
SEED_SCAN_OUTPUT_DIR = PROCESSED_DATA_DIR / "exploration" / "0.04"

# Scan parameters
SEEDS = [42, 123, 456, 789, 1000]
METRICS = ["cosine", "euclidean"]

# Significance color palette (consistent with visualization.py)
SIG_PALETTE = {"Significant": "#e41a1c", "Not significant": "#377eb8", "No data": "#cccccc"}

app = typer.Typer()


def run_single_umap(
    adata: sc.AnnData,
    metric: str,
    seed: int,
) -> sc.AnnData:
    """Run UMAP with specific metric and seed, return copy with embeddings.

    Copies the input data to avoid modifying the original, then computes
    PCA and UMAP embeddings.
    """
    logger.info(f"Computing UMAP: metric={metric}, seed={seed}")
    adata_copy = adata.copy()
    adata_copy = compute_umap(
        adata_copy,
        metric=metric,
        random_state=seed,
    )
    return adata_copy


def save_minimal_h5ad(
    adata: sc.AnnData,
    output_path: Path,
) -> None:
    """Save minimal h5ad with only obs and obsm (no X matrix)."""
    adata_minimal = sc.AnnData(
        obs=adata.obs.copy(),
        obsm={"X_pca": adata.obsm["X_pca"], "X_umap": adata.obsm["X_umap"]},
    )
    if "pca" in adata.uns:
        adata_minimal.uns["pca"] = adata.uns["pca"]
    if "umap" in adata.uns:
        adata_minimal.uns["umap"] = adata.uns["umap"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_minimal.write_h5ad(output_path)
    logger.info(f"Saved: {output_path}")


def plot_seed_comparison_grid(
    results: dict[tuple[str, int], sc.AnnData],
    output_path: Path,
    dataset: str,
    clip_outliers: bool = True,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create 2×5 grid comparing metrics (rows) × seeds (columns).

    Points are colored by phenotypic significance (below_corrected_p).
    Uses shared axis limits per metric for fair comparison.
    Uses sc.pl.umap() for consistent visualization with existing notebooks.
    """
    n_metrics = len(METRICS)
    n_seeds = len(SEEDS)

    fig, axes = plt.subplots(
        n_metrics,
        n_seeds,
        figsize=(4 * n_seeds, 4 * n_metrics),
        squeeze=False,
    )

    # Compute shared bounds per metric (from first seed)
    metric_bounds = {}
    for metric in METRICS:
        adata = results[(metric, SEEDS[0])]
        if clip_outliers:
            xlim, ylim = compute_umap_bounds(adata)
        else:
            coords = adata.obsm["X_umap"]
            margin = 0.05
            xlim = (coords[:, 0].min() * (1 - margin), coords[:, 0].max() * (1 + margin))
            ylim = (coords[:, 1].min() * (1 - margin), coords[:, 1].max() * (1 + margin))
        metric_bounds[metric] = (xlim, ylim)

    for i, metric in enumerate(METRICS):
        xlim, ylim = metric_bounds[metric]

        for j, seed in enumerate(SEEDS):
            ax = axes[i, j]
            adata = results[(metric, seed)]

            # Prepare significance column as categorical
            if "phenotypic_significant" not in adata.obs.columns:
                if "below_corrected_p" in adata.obs.columns:
                    adata.obs["phenotypic_significant"] = adata.obs["below_corrected_p"].map(
                        {"True": "Significant", "False": "Not significant", "NA": "No data"}
                    )
                else:
                    adata.obs["phenotypic_significant"] = "No data"

            # Use sc.pl.umap() - same approach as 2.01-ss-umap-visualization.py
            sc.pl.umap(
                adata,
                color="phenotypic_significant",
                ax=ax,
                show=False,
                title="",
                palette=SIG_PALETTE,
                legend_loc=None,  # No per-panel legend
            )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Column headers (seeds)
            if i == 0:
                ax.set_title(f"seed={seed}", fontsize=12)

            # Row labels (metrics)
            if j == 0:
                ax.set_ylabel(metric, fontsize=14, fontweight="bold")

    # Add shared legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Significant"],
            markersize=8,
            label="Significant (p<0.05)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["Not significant"],
            markersize=8,
            label="Not significant",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=SIG_PALETTE["No data"],
            markersize=8,
            label="No activity data",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    # Overall title
    fig.suptitle(
        f"UMAP Seed Stability: {dataset}\n(perturbation-level, colored by phenotypic significance)",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved comparison grid: {output_path}")
    plt.close(fig)


def compute_layout_similarity(
    results: dict[tuple[str, int], sc.AnnData],
) -> dict[str, float]:
    """Compute Procrustes similarity between seed pairs within each metric.

    Returns mean Procrustes correlation for each metric, indicating how
    stable the layout is across seeds (1.0 = identical, 0.0 = uncorrelated).
    """
    from itertools import combinations

    from scipy.spatial import procrustes

    similarities = {}

    for metric in METRICS:
        correlations = []
        seed_adatas = [(seed, results[(metric, seed)]) for seed in SEEDS]

        for (seed1, adata1), (seed2, adata2) in combinations(seed_adatas, 2):
            coords1 = adata1.obsm["X_umap"]
            coords2 = adata2.obsm["X_umap"]

            # Procrustes analysis aligns and compares shapes
            _, mtx2_aligned, disparity = procrustes(coords1, coords2)

            # Correlation between aligned coordinates
            corr = np.corrcoef(coords1.flatten(), mtx2_aligned.flatten())[0, 1]
            correlations.append(corr)

        similarities[metric] = {
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "min_correlation": np.min(correlations),
        }

    return similarities


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name (uses default from analysis.yaml)",
    ),
    activity_preprocessing: str = typer.Option(
        "activity_no_target2",
        "--activity-preprocessing",
        help="Activity preprocessing config name",
    ),
    activity_filter: str = typer.Option(
        "all_sources",
        "--activity-filter",
        help="Activity filter config name",
    ),
    skip_compute: bool = typer.Option(
        False,
        "--skip-compute",
        help="Skip UMAP computation, only regenerate plots from existing h5ad files",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Run UMAP seed stability scan: 5 seeds × 2 metrics.

    Computes 10 UMAP embeddings (perturbation-level only) and creates a
    2×5 comparison grid showing layout stability across random seeds.
    """
    results: dict[tuple[str, int], sc.AnnData] = {}

    if not skip_compute:
        # Load profiles once (perturbation-level only per plan)
        logger.info(f"Loading {dataset} perturbation-level profiles...")
        adata_base = load_profiles(dataset, level="perturbation")
        logger.info(f"Loaded {adata_base.n_obs:,} obs x {adata_base.n_vars} vars")

        # Run all UMAP combinations
        total = len(SEEDS) * len(METRICS)
        for idx, (metric, seed) in enumerate([(m, s) for m in METRICS for s in SEEDS], start=1):
            logger.info(f"[{idx}/{total}] Running UMAP: metric={metric}, seed={seed}")

            # Compute UMAP
            adata_result = run_single_umap(adata_base, metric, seed)

            # Save h5ad with seed suffix
            h5ad_path = SEED_SCAN_DATA_DIR / f"{dataset}_perturbation_{metric}_seed{seed}_umap.h5ad"
            save_minimal_h5ad(adata_result, h5ad_path)

            # Store for plotting (with activity data)
            adata_result = join_activity(
                adata_result,
                dataset,
                preprocessing=activity_preprocessing,
                filter_name=activity_filter,
            )
            results[(metric, seed)] = adata_result

        logger.success(f"Completed {total} UMAP computations")

    else:
        # Load existing h5ad files
        logger.info("Loading existing h5ad files (--skip-compute mode)")
        for metric, seed in [(m, s) for m in METRICS for s in SEEDS]:
            h5ad_path = SEED_SCAN_DATA_DIR / f"{dataset}_perturbation_{metric}_seed{seed}_umap.h5ad"
            if not h5ad_path.exists():
                raise FileNotFoundError(f"Missing h5ad: {h5ad_path}. Run without --skip-compute first.")
            adata = sc.read_h5ad(h5ad_path)
            adata = join_activity(
                adata,
                dataset,
                preprocessing=activity_preprocessing,
                filter_name=activity_filter,
            )
            results[(metric, seed)] = adata
        logger.success(f"Loaded {len(results)} existing h5ad files")

    # Generate comparison grid with dataset nesting
    output_path = SEED_SCAN_OUTPUT_DIR / dataset / "seed_comparison.png"
    plot_seed_comparison_grid(results, output_path, dataset, dpi=dpi)

    # Compute and report layout similarity
    logger.info("Computing layout similarity (Procrustes)...")
    similarities = compute_layout_similarity(results)

    logger.info("=" * 60)
    logger.info("SEED STABILITY SUMMARY")
    logger.info("=" * 60)
    for metric, stats in similarities.items():
        logger.info(
            f"{metric}: mean correlation = {stats['mean_correlation']:.3f} "
            f"(std={stats['std_correlation']:.3f}, min={stats['min_correlation']:.3f})"
        )
    logger.info("=" * 60)
    logger.info("Interpretation: correlation close to 1.0 means layouts are stable across seeds")
    logger.info(f"Output: {output_path}")

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    per_metric = {
        metric: {
            "mean_procrustes_correlation": round(float(stats["mean_correlation"]), 4),
            "std_procrustes_correlation": round(float(stats["std_correlation"]), 4),
            "min_procrustes_correlation": round(float(stats["min_correlation"]), 4),
        }
        for metric, stats in similarities.items()
    }
    out_dir = output_path.parent
    write_summary(
        out_dir / "summary.json",
        description=(
            "UMAP seed-stability scan (Issue #17): 5 random seeds x 2 distance metrics (cosine, "
            "euclidean) at the perturbation level. Reports the Procrustes correlation between every "
            "seed pair within each metric (mean/std/min across the 10 pairs); values near 1.0 mean the "
            "layout is reproducible across seeds rather than seed-dependent."
        ),
        numbers={
            "dataset": dataset,
            "n_seeds": len(SEEDS),
            "seeds": SEEDS,
            "per_metric": per_metric,
        },
        files=index_files(out_dir),
    )

    logger.success("Done!")


if __name__ == "__main__":
    app()
