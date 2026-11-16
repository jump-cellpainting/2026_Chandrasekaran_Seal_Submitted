"""Batch-correct profiles using PCA + Harmony via rapids_singlecell.

Reads pre-Harmony profiles (post-featselect, pre-batch-correction), joins
Metadata_Batch from DuckDB, runs PCA then Harmony, and saves corrected
PCA-Harmony components as parquet with _rsc suffix.

These _rsc profiles are separate from the default recipe-Harmony profiles
downloaded from CellPainting Gallery.

Usage:
    pixi run -e rapids python -m jump_production.processing.batch_correct \
        --input data/raw/profiles/compound_no_source7_pre_harmony.parquet \
        --output data/raw/profiles/compound_no_source7_rsc.parquet \
        --db data/interim/jump_metadata_augmented.duckdb \
        --config configs/batch_correction.yaml
"""

from pathlib import Path

import anndata as ad
import duckdb
import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger

from jump_production.gpu import has_gpu

app = typer.Typer()


CANONICAL_METADATA = {"Metadata_JCP2022", "Metadata_Plate", "Metadata_Source", "Metadata_Well"}


def join_batch_metadata(df: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """Standardize metadata columns and join Metadata_Batch from DuckDB.

    Some pre-Harmony parquets (e.g., from the recipe) have extra metadata columns
    (Metadata_PlateType, Metadata_Microscope, etc.) and may already contain
    Metadata_Batch. We drop everything except canonical columns, then always
    join Metadata_Batch fresh from DuckDB for consistency.
    """
    extra_meta = [c for c in df.columns if c.startswith("Metadata_") and c not in CANONICAL_METADATA]
    if extra_meta:
        df = df.drop(columns=extra_meta)
        logger.info(f"Dropped {len(extra_meta)} extra metadata columns: {extra_meta}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        plate_df = con.sql("SELECT Metadata_Source, Metadata_Plate, Metadata_Batch FROM plate").fetchdf()

    n_before = len(df)
    df = df.merge(plate_df, on=["Metadata_Source", "Metadata_Plate"], how="left")

    n_missing = df["Metadata_Batch"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing:,} rows have no Metadata_Batch after join")

    logger.info(f"Joined Metadata_Batch: {n_before:,} rows, {df['Metadata_Batch'].nunique()} unique batches")
    return df


def parquet_to_anndata(df: pd.DataFrame) -> ad.AnnData:
    """Convert a profile DataFrame to AnnData.

    Metadata_* columns go to obs (with prefix stripped),
    everything else becomes the feature matrix X.
    """
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    obs = df[metadata_cols].copy()
    obs.columns = [c.replace("Metadata_", "") for c in obs.columns]
    obs.index = obs.index.astype(str)

    for col in obs.columns:
        if obs[col].dtype == "string" or str(obs[col].dtype).startswith("String"):
            obs[col] = obs[col].astype(str)

    X = df[feature_cols].values.astype(np.float32)
    var = pd.DataFrame(index=feature_cols)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    logger.info(f"AnnData: {adata.n_obs:,} obs x {adata.n_vars} vars")
    return adata


def run_harmony(
    adata: ad.AnnData,
    batch_key: str,
    n_pcs: int,
    n_clusters: int,
    max_iter_harmony: int,
) -> np.ndarray:
    """Run PCA + Harmony via rapids_singlecell (GPU required).

    Args:
        adata: AnnData with features in X and batch key(s) in obs.
            Config batch_key uses Metadata_ prefix (e.g., "Metadata_Batch");
            obs columns have prefix stripped (e.g., "Batch").
        batch_key: Column name with Metadata_ prefix (stripped internally)
        n_pcs: Number of PCA components
        n_clusters: Harmony n_clusters parameter
        max_iter_harmony: Harmony max iterations

    Returns:
        Corrected components as numpy array (n_obs, n_pcs)
    """
    import rapids_singlecell as rsc

    # Config uses Metadata_ prefix; obs has it stripped
    obs_key = batch_key.replace("Metadata_", "") if batch_key.startswith("Metadata_") else batch_key
    if obs_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{obs_key}' not found in obs. Available: {list(adata.obs.columns)}")

    # Handle NaN values in feature matrix
    n_nan = np.isnan(adata.X).sum()
    if n_nan > 0:
        logger.warning(f"Replacing {n_nan:,} NaN values with 0")
        adata.X = np.nan_to_num(adata.X, nan=0.0)

    n_comps = min(n_pcs, adata.n_vars - 1)

    logger.info("Transferring data to GPU")
    rsc.get.anndata_to_GPU(adata)

    logger.info(f"Computing PCA with {n_comps} components (GPU)")
    rsc.pp.pca(adata, n_comps=n_comps)

    logger.info(f"Running Harmony on '{obs_key}' (n_clusters={n_clusters}, max_iter={max_iter_harmony})")
    rsc.pp.harmony_integrate(
        adata,
        key=obs_key,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
    )

    logger.info("Transferring data back to CPU")
    rsc.get.anndata_to_CPU(adata)

    corrected = np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32)
    logger.info(f"Harmony complete: {corrected.shape}")
    return corrected


def build_output_df(adata: ad.AnnData, corrected: np.ndarray) -> pd.DataFrame:
    """Build output DataFrame from corrected components + metadata.

    Restores Metadata_ prefix on obs columns and names features X_1..X_N.
    """
    n_pcs = corrected.shape[1]
    feature_names = [f"X_{i + 1}" for i in range(n_pcs)]

    meta_df = adata.obs.copy()
    meta_df.columns = [f"Metadata_{c}" for c in meta_df.columns]
    meta_df.index = range(len(meta_df))

    # Only keep canonical metadata columns in output
    canonical = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
    meta_df = meta_df[canonical]

    feat_df = pd.DataFrame(corrected, columns=feature_names)
    return pd.concat([meta_df, feat_df], axis=1)


@app.command()
def main(
    input_file: Path = typer.Option(..., "--input", help="Pre-Harmony profile parquet"),
    output: Path = typer.Option(..., "--output", help="Output corrected parquet"),
    db: Path = typer.Option(..., "--db", help="Augmented metadata DuckDB path"),
    config: Path = typer.Option("configs/batch_correction.yaml", "--config", help="Batch correction config YAML"),
) -> None:
    """Batch-correct profiles using PCA + Harmony (GPU required)."""
    if not has_gpu():
        logger.error("GPU required for batch correction. Run with: pixi run -e rapids python -m ...")
        raise typer.Exit(1)
    if not input_file.exists():
        logger.error(f"Input not found: {input_file}")
        raise typer.Exit(1)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        raise typer.Exit(1)

    # Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)
    batch_key = cfg["batch_key"]
    n_pcs = cfg["n_pcs"]
    n_clusters = cfg["n_clusters"]
    max_iter_harmony = cfg["max_iter_harmony"]
    logger.info(f"Config: batch_key={batch_key}, n_pcs={n_pcs}, n_clusters={n_clusters}")

    # Load and prepare data
    logger.info(f"Loading {input_file}")
    df = pd.read_parquet(input_file)
    n_input_features = len([c for c in df.columns if not c.startswith("Metadata_")])
    logger.info(f"Loaded {len(df):,} rows, {n_input_features} features")

    df = join_batch_metadata(df, db)
    adata = parquet_to_anndata(df)

    # Run PCA + Harmony
    corrected = run_harmony(
        adata,
        batch_key=batch_key,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
    )

    # Save output
    out_df = build_output_df(adata, corrected)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
    logger.success(f"Saved corrected profiles to {output}: {len(out_df):,} rows x {corrected.shape[1]} features")


if __name__ == "__main__":
    app()
