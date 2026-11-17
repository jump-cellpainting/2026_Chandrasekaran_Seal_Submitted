"""Structure-morphology relationship analysis for JUMP Cell Painting dataset.

Analyzes the relationship between chemical structure and morphological profiles
for phenotypically active compounds:
- Are structurally similar compounds also morphologically similar?
- Which mechanisms of action show highest structure-morphology coherence?

Supports two structural representations:
- Morgan fingerprints (2048-bit): L2 norm + cosine ≈ Tanimoto (ρ=0.998, Issue #26)
- ChemBERTa embeddings (384-dim): Learned chemical semantics from SMILES

Usage:
    # Morgan fingerprints (default)
    pixi run -e rapids python notebooks/2.05-ss-structure-morphology.py
    pixi run -e rapids python notebooks/2.05-ss-structure-morphology.py --structure morgan

    # ChemBERTa embeddings
    pixi run -e rapids python notebooks/2.05-ss-structure-morphology.py --structure chemberta

    # Other options
    pixi run -e rapids python notebooks/2.05-ss-structure-morphology.py --dataset compound_DL_CPCNN_no_source7
    pixi run python notebooks/2.05-ss-structure-morphology.py  # CPU fallback

Outputs: data/processed/structure-morphology/{dataset}/{preprocessing}/{structure_rep}/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger

from jump_production.config import DEFAULT_DPI
from jump_production.gpu import has_gpu
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import SIG_COLOR, plot_dotplot

HAS_GPU = has_gpu()

logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled (cupy not available)'}")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
COPAIRS_DB = DATA_DIR / "processed/copairs_results.duckdb"
METADATA_DB = DATA_DIR / "interim/jump_metadata_augmented.duckdb"

K_NEIGHBORS = 50

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load compound fingerprints (Morgan FP)."""
    fp_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]  # Shape: (N, 2048) uint8
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds × {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


def load_chemberta_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load ChemBERTa-77M-MLM embeddings (384-dim)."""
    chemberta_file = DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz"
    emb_data = np.load(chemberta_file, allow_pickle=True)
    embeddings = emb_data["embeddings"]  # Shape: (N, 384) float32
    jcp = emb_data["jcp2022"]
    logger.info(f"ChemBERTa embeddings: {embeddings.shape[0]:,} compounds × {embeddings.shape[1]} dims")
    return embeddings, jcp


def get_valid_jcp_intersection() -> set[str]:
    """Get JCP2022 IDs valid in both Morgan FP and ChemBERTa embeddings.

    This ensures apples-to-apples comparison between representations by
    restricting to compounds where both have valid (non-zero) features.

    Returns:
        Set of JCP2022 IDs valid in both representations.
    """
    # Load Morgan FP
    morgan_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    morgan_fp = morgan_data["fingerprints"].astype(np.float32)
    morgan_jcp = morgan_data["jcp2022"]
    morgan_norms = np.linalg.norm(morgan_fp, axis=1)
    morgan_valid = set(morgan_jcp[morgan_norms > 0])

    # Load ChemBERTa
    chemberta_data = np.load(DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz", allow_pickle=True)
    chemberta_emb = chemberta_data["embeddings"]
    chemberta_jcp = chemberta_data["jcp2022"]
    chemberta_norms = np.linalg.norm(chemberta_emb, axis=1)
    chemberta_valid = set(chemberta_jcp[chemberta_norms > 0])

    intersection = morgan_valid & chemberta_valid
    logger.info(
        f"Valid in Morgan: {len(morgan_valid):,}, ChemBERTa: {len(chemberta_valid):,}, both: {len(intersection):,}"
    )
    return intersection


def load_profiles(dataset: str) -> sc.AnnData:
    """Load compound-level aggregated profiles.

    Args:
        dataset: Dataset name (e.g., 'compound_no_source7')
    """
    adata = sc.read_h5ad(DATA_DIR / f"interim/anndata/{dataset}_perturbation.h5ad")
    logger.info(f"Profiles: {adata.shape[0]:,} compounds × {adata.shape[1]} features")
    return adata


def load_activity_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
) -> pd.DataFrame:
    """Load activity status from copairs results.

    Args:
        con: DuckDB connection to copairs_results.duckdb
        dataset: Dataset name (e.g., 'compound_no_source7')
        preprocessing: Preprocessing config name (e.g., 'activity_no_target2')
        filter_name: Filter config name (e.g., 'all_sources')
        activity_params: Activity params (default, withinsource, crosssource)
    """
    df = query_activity_results(
        con,
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
    )
    # Rename for clarity
    df = df.rename(
        columns={
            "Metadata_JCP2022": "JCP2022",
            "mean_average_precision": "mAP",
            "mean_normalized_average_precision": "nmAP",
            "below_corrected_p": "active",
        }
    )
    logger.info(f"Activity: {len(df):,} compounds, {df.active.sum():,} active ({df.active.mean() * 100:.1f}%)")
    return df[["JCP2022", "mAP", "nmAP", "active"]]


def load_moa_annotations(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load MoA annotations from Drug Repurposing Hub."""
    df = con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_repurposing_moa as moa
        FROM compound_metadata
        WHERE Metadata_repurposing_moa IS NOT NULL
    """).df()
    logger.info(f"MoA annotations: {len(df):,} compounds")
    return df


def align_data(
    adata_profiles: sc.AnnData,
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
    activity_df: pd.DataFrame,
    valid_jcp: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fingerprints and profiles by JCP2022 ID, filtering to active compounds.

    Args:
        valid_jcp: Optional set of JCP2022 IDs to restrict to (e.g., from
            get_valid_jcp_intersection() for fair cross-representation comparison).
    """
    # Filter to active compounds first
    active_jcp = set(activity_df[activity_df["active"]]["JCP2022"])
    logger.info(f"Active compounds: {len(active_jcp):,}")

    morph_jcp = adata_profiles.obs["JCP2022"].values
    common_jcp = set(fp_jcp) & set(morph_jcp) & active_jcp

    # Apply additional valid_jcp filter if provided (for cross-representation comparison)
    if valid_jcp is not None:
        common_jcp = common_jcp & valid_jcp
        logger.info(f"After valid_jcp filter: {len(common_jcp):,}")

    common_jcp = sorted(common_jcp)
    logger.info(f"Common active compounds: {len(common_jcp):,}")

    fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    morph_idx = {k: i for i, k in enumerate(morph_jcp)}

    fp_order = [fp_idx[k] for k in common_jcp]
    morph_order = [morph_idx[k] for k in common_jcp]

    X_fp = fp_array[fp_order].astype(np.float32)
    X_morph = adata_profiles.X[morph_order].astype(np.float32)
    jcp_ids = np.array(common_jcp)

    # Filter valid (no NaN, non-zero norm)
    fp_norms = np.linalg.norm(X_fp, axis=1)
    morph_norms = np.linalg.norm(X_morph, axis=1)
    valid_mask = (fp_norms > 0) & (morph_norms > 0) & ~np.isnan(morph_norms)
    logger.info(f"Valid active compounds: {valid_mask.sum():,}")

    return X_fp[valid_mask], X_morph[valid_mask], jcp_ids[valid_mask]


# =============================================================================
# k-NN Computation
# =============================================================================


def compute_knn(
    X_fp: np.ndarray, X_morph: np.ndarray, jcp_ids: np.ndarray, k: int = K_NEIGHBORS
) -> tuple[sc.AnnData, sc.AnnData]:
    """Compute k-NN in structure and morphology spaces."""
    # L2 normalize fingerprints (cosine on L2-normed ≈ Tanimoto, ρ=0.998)
    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)
    X_morph_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)

    # Select neighbors function based on GPU availability
    if HAS_GPU:
        import rapids_singlecell as rsc

        neighbors_fn = rsc.pp.neighbors
        logger.info("Using GPU-accelerated k-NN (rapids_singlecell)")
    else:
        neighbors_fn = sc.pp.neighbors
        logger.info("Using CPU k-NN (scanpy)")

    # Structure k-NN
    logger.info(f"Computing structure k-NN (k={k})...")
    adata_struct = sc.AnnData(X=X_fp_norm)
    adata_struct.obs["JCP2022"] = jcp_ids
    neighbors_fn(adata_struct, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {adata_struct.obsp['connectivities'].nnz:,}")

    # Morphology k-NN
    logger.info(f"Computing morphology k-NN (k={k})...")
    adata_morph = sc.AnnData(X=X_morph_norm)
    adata_morph.obs["JCP2022"] = jcp_ids
    neighbors_fn(adata_morph, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {adata_morph.obsp['connectivities'].nnz:,}")

    return adata_struct, adata_morph


def compute_neighbor_overlap(adata_struct: sc.AnnData, adata_morph: sc.AnnData) -> np.ndarray:
    """Compute Jaccard overlap of k-NN between structure and morphology."""
    logger.info("Computing neighbor overlap...")
    dist1 = adata_struct.obsp["distances"]
    dist2 = adata_morph.obsp["distances"]

    overlaps = []
    for i in range(adata_struct.n_obs):
        n1 = set(dist1[i].nonzero()[1])
        n2 = set(dist2[i].nonzero()[1])
        if n1 | n2:
            overlaps.append(len(n1 & n2) / len(n1 | n2))
        else:
            overlaps.append(0.0)

    return np.array(overlaps)


# =============================================================================
# Visualization
# =============================================================================


def _compute_moa_stats(
    df: pd.DataFrame, moa_df: pd.DataFrame, k: int, min_count: int = 10, top_n: int = 20
) -> pd.DataFrame:
    """Compute MoA statistics for overlap analysis.

    Returns DataFrame with columns: moa, mean_overlap, count, enrichment
    """
    n_compounds = len(df)
    expected_random = k / (2 * n_compounds)

    merged = df.merge(moa_df, on="JCP2022", how="inner")
    moa_stats = merged.groupby("moa").agg({"neighbor_overlap": ["mean", "count"]}).reset_index()
    moa_stats.columns = ["moa", "mean_overlap", "count"]
    moa_stats["enrichment"] = moa_stats["mean_overlap"] / expected_random
    moa_stats = moa_stats[moa_stats["count"] >= min_count].sort_values("enrichment", ascending=True)
    return moa_stats.tail(top_n)


def plot_combined_figure(
    df: pd.DataFrame,
    moa_df: pd.DataFrame,
    output_dir: Path | None = None,
    k: int = K_NEIGHBORS,
    dpi: int = DEFAULT_DPI,
    structure_rep: str = "morgan",
) -> None:
    """Create combined publication figure for active compounds.

    Layout: A (overlap histogram) | B (MoA dotplot) side by side
    """
    rep_label = "Morgan FP" if structure_rep == "morgan" else "ChemBERTa"
    logger.info(f"Creating combined publication figure ({rep_label})...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Compute summary stats for annotations
    overlap = df["neighbor_overlap"].values
    n_compounds = len(df)
    mean_overlap = overlap.mean()
    pct_nonzero = (overlap > 0).mean() * 100
    expected_random = k / (2 * n_compounds)
    ratio_vs_random = mean_overlap / expected_random

    # Panel A: Overlap distribution histogram
    ax_a = axes[0]
    ax_a.hist(overlap, bins=50, color=SIG_COLOR, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax_a.set_xlabel(f"Jaccard Overlap (k={k})", fontsize=10)
    ax_a.set_ylabel("Count", fontsize=10)
    ax_a.set_title(f"A. k-NN Overlap Distribution ({rep_label})", fontsize=11)
    ax_a.axvline(mean_overlap, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_overlap:.3f}")
    ax_a.axvline(expected_random, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {expected_random:.4f}")
    # Add stats text box
    stats_text = f"n = {n_compounds:,}\nNon-zero: {pct_nonzero:.1f}%\n{ratio_vs_random:.1f}× random"
    ax_a.text(
        0.95,
        0.95,
        stats_text,
        transform=ax_a.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax_a.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.78))

    # Panel B: MoA dotplot (enrichment vs random)
    ax_b = axes[1]
    moa_stats = _compute_moa_stats(df, moa_df, k=k)
    plot_dotplot(
        moa_stats,
        value_col="enrichment",
        label_col="moa",
        count_col="count",
        top_n=20,
        title=f"B. Structure-Morphology Coherence ({rep_label})",
        xlabel="Enrichment (× random)",
        size_scale=8.0,  # Lower scale since counts are small (10-21)
        size_range=(40, 200),
        ax=ax_b,
    )

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "structure_morphology_combined.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "structure_morphology_combined.pdf", bbox_inches="tight")
        logger.info(f"Saved: {output_dir / 'structure_morphology_combined.png'}")
        logger.info(f"Saved: {output_dir / 'structure_morphology_combined.pdf'}")
    plt.close(fig)


# =============================================================================
# Summary Statistics
# =============================================================================


def generate_summary(
    df: pd.DataFrame,
    n_compounds: int,
    k: int,
    output_dir: Path,
    structure_rep: str = "morgan",
) -> dict:
    """Generate summary statistics for structure-morphology analysis (active compounds only)."""
    overlap = df["neighbor_overlap"].values

    mean_overlap = overlap.mean()
    median_overlap = np.median(overlap)
    expected_random = k / (2 * n_compounds)
    ratio_vs_random = mean_overlap / expected_random
    pct_zero = (overlap == 0).mean() * 100
    pct_nonzero = 100 - pct_zero

    numbers = {
        "structure_rep": structure_rep,
        "n_compounds": n_compounds,
        "k_neighbors": k,
        "mean_jaccard": float(mean_overlap),
        "median_jaccard": float(median_overlap),
        "expected_random": float(expected_random),
        "ratio_vs_random": float(ratio_vs_random),
        "pct_zero_overlap": float(pct_zero),
        "pct_nonzero_overlap": float(pct_nonzero),
        "use_gpu": HAS_GPU,
    }

    # Log key findings
    logger.info(f"\n=== RESULTS (n={n_compounds:,} active compounds, k={k}) ===")
    logger.info(f"Mean Jaccard:     {mean_overlap:.4f}")
    logger.info(f"Median Jaccard:   {median_overlap:.4f}")
    logger.info(f"Expected random:  {expected_random:.6f}")
    logger.info(f"Ratio vs random:  {ratio_vs_random:.1f}x")
    logger.info(f"Non-zero overlap: {pct_nonzero:.1f}%")

    # Standardized summary envelope: description (intent) + numbers + index of generated files
    rep_label = "Morgan fingerprints" if structure_rep == "morgan" else "ChemBERTa embeddings"
    description = (
        f"Structure-morphology coherence for phenotypically active JUMP compounds using {rep_label} as the "
        "structural representation: the Jaccard overlap between each compound's k-nearest neighbors in structure "
        "space versus morphology space, summarized across compounds and compared to the random-overlap baseline."
    )
    output_file = output_dir / "structure_morphology_summary.json"
    summary = write_summary(output_file, description=description, numbers=numbers, files=index_files(output_dir))
    logger.info(f"Saved: {output_file}")

    return summary


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name (e.g., compound_no_source7, compound_DL_CPCNN_no_source7)",
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
    activity_params: str = typer.Option(
        "default",
        "--activity-params",
        help="Activity params config (default, withinsource, crosssource)",
    ),
    structure_rep: str = typer.Option(
        "morgan",
        "--structure",
        "-s",
        help="Structural representation: 'morgan' (2048-bit fingerprints) or 'chemberta' (384-dim embeddings)",
    ),
    k_neighbors: int = typer.Option(K_NEIGHBORS, "--k", "-k", help="Number of neighbors for k-NN"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
) -> None:
    """Analyze structure-morphology relationship for active compounds in JUMP dataset."""
    # Validate structure representation
    valid_reps = {"morgan", "chemberta"}
    if structure_rep not in valid_reps:
        logger.error(f"Unknown structure representation: {structure_rep}. Valid options: {valid_reps}")
        raise typer.Exit(1)

    # Use hierarchical output path: structure-morphology/{dataset}/{preprocessing}/{structure_rep}/
    if output_dir is None:
        output_dir = DATA_DIR / "processed" / "structure-morphology" / dataset / preprocessing / structure_rep
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Preprocessing: {preprocessing}")
    logger.info(f"Filter: {filter_name}")
    logger.info(f"Activity params: {activity_params}")
    logger.info(f"Structure rep: {structure_rep}")
    logger.info(f"k-neighbors: {k_neighbors}")
    logger.info(f"Output: {output_dir}")

    # Check databases exist
    if not COPAIRS_DB.exists():
        logger.error(f"Copairs database not found: {COPAIRS_DB}")
        raise typer.Exit(1)
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    copairs_con = duckdb.connect(str(COPAIRS_DB), read_only=True)
    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Load data
        logger.info("Loading data...")
        if structure_rep == "morgan":
            fp_array, fp_jcp = load_fingerprints()
        else:  # chemberta
            fp_array, fp_jcp = load_chemberta_embeddings()
        adata_profiles = load_profiles(dataset)
        activity_df = load_activity_data(copairs_con, dataset, preprocessing, filter_name, activity_params)
        moa_df = load_moa_annotations(metadata_con)

        # Get intersection of valid compounds (for fair cross-representation comparison)
        valid_jcp = get_valid_jcp_intersection()

        # 2. Align data (filters to active compounds with valid representations)
        X_fp, X_morph, jcp_ids = align_data(adata_profiles, fp_array, fp_jcp, activity_df, valid_jcp=valid_jcp)
        n_compounds = len(jcp_ids)

        # 3. Compute k-NN
        adata_struct, adata_morph = compute_knn(X_fp, X_morph, jcp_ids, k=k_neighbors)

        # 4. Compute neighbor overlap
        overlap = compute_neighbor_overlap(adata_struct, adata_morph)

        # 5. Create results DataFrame (merge to get nmAP values)
        results_df = pd.DataFrame({"JCP2022": jcp_ids, "neighbor_overlap": overlap})
        results_df = results_df.merge(activity_df[["JCP2022", "mAP", "nmAP"]], on="JCP2022", how="left")

        # 6. Generate combined publication figure
        plot_combined_figure(
            results_df,
            moa_df,
            output_dir,
            k=k_neighbors,
            dpi=dpi,
            structure_rep=structure_rep,
        )

        # 7. Save results
        results_df.to_parquet(output_dir / "neighbor_overlap.parquet", index=False)
        logger.info(f"Saved: {output_dir / 'neighbor_overlap.parquet'}")

        # 8. Generate summary
        generate_summary(results_df, n_compounds, k_neighbors, output_dir, structure_rep=structure_rep)

        logger.success(f"Structure-morphology analysis complete: {output_dir}")

    finally:
        copairs_con.close()
        metadata_con.close()


if __name__ == "__main__":
    app()
