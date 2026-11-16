#!/usr/bin/env python
"""Convert parquet profile files to AnnData format for scanpy integration.

Converts profile parquet files to .h5ad format with:
- adata.X: feature matrix (columns starting with X_)
- adata.obs: metadata columns (Metadata_*)
- adata.var: feature names

Also creates a perturbation-level aggregated version using mean.

Usage:
    pixi run python -m jump_production.processing.profiles_to_anndata INPUT OUTPUT
    pixi run python -m jump_production.processing.profiles_to_anndata data/raw/profiles/compound_no_source7.parquet data/interim/anndata/compound_no_source7.h5ad
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger

from jump_production.gpu import has_gpu

HAS_GPU = has_gpu()
if HAS_GPU:
    try:
        import rapids_singlecell as _rsc  # noqa: F401
    except ImportError:
        logger.info("rapids_singlecell not installed, falling back to CPU")
        HAS_GPU = False

app = typer.Typer()


def aggregate(adata: ad.AnnData, by: str = "JCP2022", func: str = "mean") -> ad.AnnData:
    """Aggregate profiles by perturbation identifier.

    Uses GPU-accelerated rapids_singlecell when available, falls back to scanpy.

    Args:
        adata: AnnData with well-level profiles
        by: Column in obs to group by (default: JCP2022)
        func: Aggregation function ('mean' or 'sum'; 'median' only on CPU)

    Returns:
        Aggregated AnnData with one row per unique perturbation
    """
    logger.info(f"Aggregating {adata.n_obs:,} observations by {by} using {func}")

    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Using GPU-accelerated aggregation (rapids_singlecell)")
        rsc.get.anndata_to_GPU(adata)
        adata_agg = rsc.get.aggregate(adata, by=by, func=func)
    else:
        logger.info("Using CPU aggregation (scanpy)")
        adata_agg = sc.get.aggregate(adata, by=by, func=func)

    # Both scanpy and rapids store result in layers[func], move to X
    layer_data = adata_agg.layers[func]
    del adata_agg.layers[func]

    # Convert to numpy array (handle cupy arrays from GPU)
    if hasattr(layer_data, "get"):
        # cupy array - use .get() to transfer to CPU
        adata_agg.X = layer_data.get().astype(np.float32)
    elif hasattr(layer_data, "toarray"):
        # sparse matrix
        adata_agg.X = layer_data.toarray().astype(np.float32)
    else:
        adata_agg.X = np.asarray(layer_data, dtype=np.float32)

    # Restore grouping column and reset index for h5ad compatibility
    adata_agg.obs[by] = adata_agg.obs.index
    adata_agg.obs.index = [str(i) for i in range(adata_agg.n_obs)]

    logger.info(f"Aggregated to {adata_agg.n_obs:,} unique perturbations")
    return adata_agg


def convert_parquet_to_anndata(input_path: Path, output_path: Path) -> int:
    """Convert a parquet profile file to AnnData format.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output h5ad file

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        logger.info(f"Loading parquet from {input_path}")
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")

        # Separate metadata columns (Metadata_*) from feature columns (everything else)
        metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
        feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

        logger.info(f"Found {len(feature_cols)} feature columns, {len(metadata_cols)} metadata columns")

        # Extract feature matrix
        X = df[feature_cols].values.astype(np.float32)  # noqa: N806

        # Extract observation metadata
        obs = df[metadata_cols].copy()
        obs.columns = [c.replace("Metadata_", "") for c in obs.columns]
        obs.index = obs.index.astype(str)

        # Convert string columns to object dtype for anndata compatibility
        for col in obs.columns:
            if obs[col].dtype == "string" or str(obs[col].dtype).startswith("String"):
                obs[col] = obs[col].astype(str)

        # Create variable annotations (feature names)
        var = pd.DataFrame(index=feature_cols)
        var.index.name = "feature"

        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)
        logger.info(f"Created AnnData: {adata.n_obs:,} obs × {adata.n_vars} vars")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save well-level h5ad
        logger.info(f"Writing well-level to {output_path}")
        adata.write_h5ad(output_path)
        logger.success(f"Saved well-level AnnData to {output_path}")

        # Create and save perturbation-level aggregated version (mean for GPU compatibility)
        adata_agg = aggregate(adata, by="JCP2022", func="mean")
        perturbation_path = output_path.parent / f"{output_path.stem}_perturbation.h5ad"
        logger.info(f"Writing perturbation-level to {perturbation_path}")
        adata_agg.write_h5ad(perturbation_path)
        logger.success(f"Saved perturbation-level AnnData to {perturbation_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to convert: {e}")
        return 1


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Input parquet file path"),
    output_path: Path = typer.Argument(..., help="Output h5ad file path"),
) -> None:
    """Convert parquet profile to AnnData h5ad format."""
    raise typer.Exit(convert_parquet_to_anndata(input_path, output_path))


if __name__ == "__main__":
    app()
