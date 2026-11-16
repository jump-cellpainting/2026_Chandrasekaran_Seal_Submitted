"""Compute UMAP embeddings for JUMP compound profiles.

Computes PCA and UMAP embeddings at two levels:
- Well-level: Preserves Source information (replicates span sources)
- Perturbation-level: Aggregated by compound (mean)

Saves minimal h5ad files containing only base obs columns (JCP2022, Source,
Plate, Well) and obsm embeddings (no X matrix, no metadata). Metadata is
joined at visualization time via join_metadata(), decoupling UMAP computation
from metadata updates.

Uses GPU-accelerated rapids_singlecell when available.

Usage:
    pixi run -e rapids python -m jump_production.processing.umap compound_no_source7
    pixi run -e rapids python -m jump_production.processing.umap compound_no_source7 --subsample 50000
    CUDA_VISIBLE_DEVICES="" pixi run -e rapids python -m jump_production.processing.umap compound_no_source7  # CPU only
"""

import sys
from pathlib import Path

import scanpy as sc
import typer
from loguru import logger

from jump_production.config import INTERIM_DATA_DIR
from jump_production.gpu import has_gpu
from jump_production.io import join_activity, load_profiles

HAS_GPU = has_gpu()
if HAS_GPU:
    try:
        import rapids_singlecell as _rsc  # noqa: F401
    except ImportError:
        logger.info("rapids_singlecell not installed, falling back to CPU")
        HAS_GPU = False

# Set scanpy/anndata settings
sc.settings.verbosity = 1  # reduce verbosity (0=errors, 1=warnings, 2=info, 3=hints)

app = typer.Typer()


def compute_umap(
    adata: sc.AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> sc.AnnData:
    """Compute PCA and UMAP embedding.

    Uses GPU-accelerated rapids_singlecell when available.
    """
    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Transferring data to GPU")
        rsc.get.anndata_to_GPU(adata)

        logger.info(f"Computing PCA with {n_pcs} components (GPU)")
        rsc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        # Approximate neighbors and spectral initialization vary despite fixed seeds.
        logger.info(f"Computing neighbors with n_neighbors={n_neighbors}, metric={metric} (GPU, exact)")
        rsc.pp.neighbors(
            adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric,
            algorithm="brute", random_state=random_state,
        )

        logger.info(f"Computing UMAP with min_dist={min_dist} (GPU)")
        rsc.tl.umap(adata, min_dist=min_dist, random_state=random_state, init_pos="random")

        logger.info("Transferring data back to CPU")
        rsc.get.anndata_to_CPU(adata)
    else:
        logger.info(f"Computing PCA with {n_pcs} components (CPU)")
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1), random_state=random_state)

        logger.info(f"Computing neighbors with n_neighbors={n_neighbors}, metric={metric} (CPU)")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)

        logger.info(f"Computing UMAP with min_dist={min_dist} (CPU)")
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    return adata


def process_level(
    dataset: str,
    level: str,
    output_dir: Path,
    n_neighbors: int,
    n_pcs: int,
    min_dist: float,
    metric: str,
    seed: int,
    subsample: int | None = None,
    filter_active: bool = False,
) -> int:
    """Process a single level (well or perturbation) through UMAP pipeline.

    Saves a minimal h5ad with only obs metadata and obsm embeddings (no X matrix)
    for efficient storage. Activity data is joined at visualization time, not here.

    Args:
        filter_active: If True, filter to only phenotypically active compounds
            (below_corrected_p == True from copairs activity results)

    Returns exit code (0 for success, 1 for failure).
    """
    prefix = f"{dataset}_perturbation" if level == "perturbation" else dataset
    filter_name = "active" if filter_active else "all"
    h5ad_output = output_dir / f"{prefix}_{metric}_{filter_name}_umap.h5ad"

    logger.info(f"Processing {level}-level for dataset: {dataset}")
    adata = load_profiles(dataset, level=level)
    logger.info(f"Loaded {adata.n_obs:,} obs x {adata.n_vars} vars")

    # Filter to phenotypically active compounds if requested
    if filter_active:
        logger.info("Filtering to phenotypically active compounds...")
        adata = join_activity(adata, dataset)
        if "below_corrected_p" not in adata.obs.columns:
            logger.error("Activity data not found - cannot filter. Run activity analysis first.")
            return 1
        mask = adata.obs["below_corrected_p"] == "True"
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(f"Filtered to {adata.n_obs:,} active observations (from {n_before:,})")

        # Drop activity columns - they'll be joined fresh at visualization time
        activity_cols = [
            "mean_average_precision",
            "mean_normalized_average_precision",
            "p_value",
            "corrected_p_value",
            "below_corrected_p",
        ]
        adata.obs = adata.obs.drop(columns=[c for c in activity_cols if c in adata.obs.columns])

    # Subsample for efficiency (only meaningful for well-level with many replicates)
    if subsample and subsample < adata.n_obs:
        logger.info(f"Subsampling to {subsample:,} observations")
        sc.pp.subsample(adata, n_obs=subsample, random_state=seed)

    logger.info(f"Dataset shape after joins: {adata.n_obs:,} obs x {adata.n_vars} vars")

    # Compute UMAP
    adata = compute_umap(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal AnnData for storage (no X matrix, only obs + obsm)
    # This is self-sufficient for scanpy plotting functions
    adata_minimal = sc.AnnData(
        obs=adata.obs,
        obsm={"X_pca": adata.obsm["X_pca"], "X_umap": adata.obsm["X_umap"]},
    )

    # Copy uns metadata (contains UMAP/PCA parameters)
    if "pca" in adata.uns:
        adata_minimal.uns["pca"] = adata.uns["pca"]
    if "umap" in adata.uns:
        adata_minimal.uns["umap"] = adata.uns["umap"]

    logger.info(f"Saving minimal UMAP h5ad to {h5ad_output}")
    logger.info(f"  obs: {adata_minimal.n_obs:,} x {len(adata_minimal.obs.columns)} columns")
    logger.info(f"  obsm: {list(adata_minimal.obsm.keys())}")
    adata_minimal.write_h5ad(h5ad_output)

    return 0


@app.command()
def main(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., compound_no_source7)"),
    n_neighbors: int = typer.Option(15, "--n-neighbors", "-n", help="Number of neighbors for UMAP"),
    n_pcs: int = typer.Option(50, "--n-pcs", help="Number of PCs to use"),
    min_dist: float = typer.Option(0.1, "--min-dist", help="UMAP min_dist parameter"),
    metric: str = typer.Option(
        "cosine", "--metric", "-m", help="Distance metric for neighbors (cosine, euclidean, etc.)"
    ),
    output_dir: Path = typer.Option(
        INTERIM_DATA_DIR / "anndata",
        "--output",
        "-o",
        help="Output directory for h5ad files",
    ),
    subsample: int = typer.Option(None, "--subsample", "-s", help="Subsample to N observations for faster computation"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    filter_active: bool = typer.Option(False, "--filter-active", help="Filter to phenotypically active compounds only"),
) -> None:
    """Compute UMAP embeddings for JUMP compound profiles.

    Saves minimal h5ad files containing only obs metadata and obsm embeddings
    (no X matrix). These files are self-sufficient for visualization.

    Processes both well-level (preserves Source) and perturbation-level
    (aggregated by compound).
    """
    exit_code = 0

    # Process both levels
    for level in ["well", "perturbation"]:
        result = process_level(
            dataset=dataset,
            level=level,
            output_dir=output_dir,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            min_dist=min_dist,
            metric=metric,
            seed=seed,
            subsample=subsample,
            filter_active=filter_active,
        )
        if result != 0:
            exit_code = result

    if exit_code == 0:
        logger.success("Done computing UMAP for both well-level and perturbation-level!")
    else:
        logger.error("UMAP computation failed for one or more levels")

    sys.exit(exit_code)


if __name__ == "__main__":
    app()
