"""I/O utilities for loading JUMP profiles as AnnData objects.

Provides functions to load h5ad profile files and join metadata
from DuckDB databases:
- Static metadata from jump_metadata_augmented.duckdb (via join_metadata)
- Copairs results (activity, consistency) from copairs_results.duckdb (via join_activity)

Design: load_profiles() only loads h5ad files. Metadata and activity are
joined separately via join_metadata() and join_activity(). This decouples
UMAP computation (which doesn't need metadata) from visualization (which does).
"""

import anndata as ad
import duckdb
from loguru import logger

from jump_production.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

ANNDATA_DIR = INTERIM_DATA_DIR / "anndata"
METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
COPAIRS_RESULTS_DB = PROCESSED_DATA_DIR / "copairs_results.duckdb"

# Default config components for copairs queries
DEFAULT_PREPROCESSING = "activity_no_target2"
DEFAULT_FILTER = "all_sources"


def load_profiles(name: str, level: str = "well") -> ad.AnnData:
    """Load a profile dataset as AnnData (without metadata).

    This function only loads the h5ad file. Use join_metadata() to add
    compound annotations and join_activity() to add phenotypic activity data.

    Args:
        name: Dataset name (e.g., 'compound_no_source7')
        level: Profile level - 'well' for well-level or 'perturbation' for
               aggregated perturbation-level (mean across wells)

    Returns:
        AnnData object with profiles (no metadata annotations)
    """
    suffix = "_perturbation" if level == "perturbation" else ""
    h5ad_path = ANNDATA_DIR / f"{name}{suffix}.h5ad"

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Profile not found: {h5ad_path}. Run: snakemake data/interim/anndata/{name}.h5ad --cores 1"
        )

    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"Loaded {adata.n_obs:,} obs x {adata.n_vars} vars")

    return adata


def join_metadata(adata: ad.AnnData, modality: str = "compound", level: str = "well") -> ad.AnnData:
    """Join metadata from DuckDB to AnnData obs.

    Queries compound_metadata or gene_metadata view from jump_metadata_augmented.duckdb.
    The Metadata_ prefix is stripped from column names.

    See scripts/augment_metadata_db.sql for available columns in each view.

    Args:
        adata: AnnData object with JCP2022 column in obs
        modality: Perturbation modality - 'compound' or 'gene'
        level: Profile level - 'well' for well-level cell counts or 'perturbation'

    Returns:
        AnnData with metadata columns joined to obs
    """
    if not METADATA_DB.exists():
        logger.warning(f"Metadata DB not found: {METADATA_DB}, skipping metadata join")
        return adata

    con = duckdb.connect(str(METADATA_DB), read_only=True)

    jcp_ids = adata.obs["JCP2022"].unique().tolist()
    logger.info(f"Joining {modality} metadata for {len(jcp_ids):,} unique perturbations")

    # Query the appropriate metadata view
    view_name = "compound_metadata" if modality == "compound" else "gene_metadata"
    metadata_df = con.execute(
        f"SELECT * FROM {view_name} WHERE Metadata_JCP2022 = ANY(?)",
        [jcp_ids],
    ).df()

    metadata_df.columns = [c.replace("Metadata_", "") if c.startswith("Metadata_") else c for c in metadata_df.columns]

    # Merge into obs
    obs = adata.obs.merge(metadata_df, on="JCP2022", how="left")
    obs.index = adata.obs.index
    adata.obs = obs

    n_joined = metadata_df["JCP2022"].nunique()
    logger.info(f"Joined metadata for {n_joined:,} of {len(jcp_ids):,} perturbations")

    # Join well-level cell counts (only for well-level profiles)
    # Note: compound_metadata already includes median_cell_count per perturbation
    if level == "well" and all(c in adata.obs.columns for c in ["Source", "Plate", "Well"]):
        cell_counts_df = con.execute(
            "SELECT Metadata_Source, Metadata_Plate, Metadata_Well, Metadata_Count_Cells FROM cell_counts"
        ).df()
        cell_counts_df.columns = ["Source", "Plate", "Well", "Count_Cells"]

        obs = adata.obs.merge(cell_counts_df, on=["Source", "Plate", "Well"], how="left")
        obs.index = adata.obs.index
        adata.obs = obs

        n_with_counts = obs["Count_Cells"].notna().sum()
        logger.info(f"Cell counts joined: {n_with_counts:,} wells with cell counts")

    con.close()

    return adata


def load_umap(
    dataset: str,
    level: str = "perturbation",
    metric: str = "cosine",
    filter_name: str = "all",
) -> ad.AnnData:
    """Load UMAP h5ad with metadata joined.

    Convenience function that loads a precomputed UMAP h5ad file and joins
    metadata from DuckDB. The UMAP files contain only coordinates (no metadata),
    so this function provides the complete ready-to-visualize AnnData.

    Args:
        dataset: Dataset name (e.g., 'compound_no_source7')
        level: Profile level - 'well' or 'perturbation'
        metric: Distance metric used for UMAP (e.g., 'cosine', 'euclidean')
        filter_name: Filter used for UMAP (e.g., 'all', 'active')

    Returns:
        AnnData with UMAP coordinates (obsm) and metadata (obs)

    Example:
        >>> adata = load_umap("compound_no_source7", level="perturbation")
        >>> sc.pl.umap(adata, color="Uniprot_target")
    """
    prefix = f"{dataset}_perturbation" if level == "perturbation" else dataset
    h5ad_path = ANNDATA_DIR / f"{prefix}_{metric}_{filter_name}_umap.h5ad"

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"UMAP h5ad not found: {h5ad_path}. "
            f"Run: pixi run -e rapids python -m jump_production.processing.umap {dataset} --metric {metric}"
        )

    logger.info(f"Loading UMAP from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"Loaded {adata.n_obs:,} obs")

    # Join metadata from DuckDB (compound datasets only for now)
    adata = join_metadata(adata, modality="compound", level=level)

    return adata


def join_activity(
    adata: ad.AnnData,
    dataset: str,
    preprocessing: str = DEFAULT_PREPROCESSING,
    filter_name: str = DEFAULT_FILTER,
) -> ad.AnnData:
    """Join phenotypic activity results to AnnData obs.

    Queries the copairs_results.duckdb for activity results matching
    the specified dataset and config.

    Adds columns: mean_average_precision, mean_normalized_average_precision,
    p_value, corrected_p_value, below_corrected_p (as string for h5ad compatibility).

    Args:
        adata: AnnData object with JCP2022 column in obs
        dataset: Dataset name (e.g., 'compound_no_source7')
        preprocessing: Preprocessing config name (default: activity_no_target2)
        filter_name: Filter config name (default: all_sources)

    Returns:
        AnnData with activity columns joined to obs

    Example:
        >>> adata = load_profiles("compound_no_source7")
        >>> adata = join_activity(adata, "compound_no_source7")
        >>> adata.obs["below_corrected_p"].value_counts()
    """
    if not COPAIRS_RESULTS_DB.exists():
        logger.warning(f"Copairs results DB not found: {COPAIRS_RESULTS_DB}, skipping activity join")
        return adata

    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)

    # Check if activity_results table exists
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "activity_results" not in tables:
        logger.warning("activity_results table not found in copairs DB, skipping")
        con.close()
        return adata

    jcp_ids = adata.obs["JCP2022"].unique().tolist()
    logger.info(
        f"Querying activity for {len(jcp_ids):,} perturbations (preprocessing={preprocessing}, filter={filter_name})"
    )

    activity_df = con.execute(
        """
        SELECT Metadata_JCP2022, mean_average_precision, mean_normalized_average_precision,
               p_value, corrected_p_value, below_corrected_p
        FROM activity_results
        WHERE _dataset = ? AND _preprocessing = ? AND _filter = ?
          AND Metadata_JCP2022 = ANY(?)
        """,
        [dataset, preprocessing, filter_name, jcp_ids],
    ).df()
    con.close()

    logger.info(f"Retrieved activity for {len(activity_df):,} perturbations")

    if activity_df.empty:
        logger.warning("No matching activity results found")
        return adata

    # Join to obs on JCP2022
    obs = adata.obs.merge(
        activity_df,
        left_on="JCP2022",
        right_on="Metadata_JCP2022",
        how="left",
    ).drop(columns=["Metadata_JCP2022"], errors="ignore")

    # Convert boolean to string because nullable booleans (with NA from left join) don't survive h5ad round-trip
    if "below_corrected_p" in obs.columns:
        obs["below_corrected_p"] = obs["below_corrected_p"].map({True: "True", False: "False"}).fillna("NA")

    obs.index = adata.obs.index
    adata.obs = obs

    n_significant = (obs["below_corrected_p"] == "True").sum()
    n_with_activity = obs["p_value"].notna().sum()
    logger.info(f"Activity joined: {n_with_activity:,} with p-values, {n_significant:,} significant")

    return adata
