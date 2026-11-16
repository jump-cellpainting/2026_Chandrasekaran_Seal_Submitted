"""Activity cliffs analysis using Cell Painting morphology profiles.

Investigates the relationship between structural similarity and morphological
similarity to identify "activity cliffs" - compound pairs that are:
1. Structurally similar but morphologically different (cliffs)
2. Structurally different but morphologically similar (reverse cliffs)

Research Questions:
- Q1: Do structurally similar compounds have similar morphological profiles?
- Q2: For compounds sharing ChEMBL targets, do similar structures yield similar phenotypes?

Note: ChEMBL data here is binary target annotations (compound→target: yes/no),
not activity potency values. So this analyzes "annotation cliffs" rather than
traditional IC50-based activity cliffs.

Usage:
    # Quick test (10% sample)
    pixi run python notebooks/2.31-seal-activity-cliffs-analysis.py --sample 0.1

    # Full dataset
    pixi run python notebooks/2.31-seal-activity-cliffs-analysis.py --sample 1.0

    # With GPU (if available)
    pixi run -e rapids python notebooks/2.31-seal-activity-cliffs-analysis.py --sample 1.0

    # Different dataset (e.g., deep learning embeddings)
    pixi run python notebooks/2.31-seal-activity-cliffs-analysis.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/activity-cliffs/{dataset}/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.gpu import has_gpu
from jump_production.io import load_profiles
from jump_production.queries import query_activity_results
from jump_production.statistics import (
    permutation_null_cliff_rate,
    permutation_null_correlation,
)
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import plot_hexbin_with_marginals

HAS_GPU = has_gpu()

logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled (cupy not available)'}")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
COPAIRS_DB = PROCESSED_DATA_DIR / "copairs_results.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "activity-cliffs"

# Default thresholds for cliff detection
STRUCT_HIGH_THRESHOLD = 0.7  # "structurally similar" (Tanimoto ≈ cosine on L2-normed FP)
MORPH_LOW_THRESHOLD = 0.3  # "morphologically different" (cosine similarity)
K_NEIGHBORS = 100  # Number of structural nearest neighbors to analyze

RANDOM_STATE = 42

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load Morgan fingerprints (2048-bit)."""
    fp_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds × {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


def load_chembl_targets(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load ChEMBL target annotations from metadata database.

    Returns DataFrame with columns: JCP2022, targets (list of Uniprot IDs)
    """
    df = con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_Uniprot_target as targets_str
        FROM chembl_protein_targets
        WHERE Metadata_Uniprot_target IS NOT NULL
    """).df()

    # Parse pipe-delimited targets into lists
    df["targets"] = df["targets_str"].str.split("|")
    df = df.drop(columns=["targets_str"])

    n_compounds = len(df)
    n_unique_targets = len(set(t for targets in df["targets"] for t in targets))
    logger.info(f"ChEMBL targets: {n_compounds:,} compounds, {n_unique_targets:,} unique targets")

    return df


def load_active_compounds(dataset: str) -> set[str]:
    """Load phenotypically active JCP2022 IDs from copairs results.

    Same activity call used in 2.05 panel C (copairs mAP, FDR-corrected p),
    so an --active-only cliff run is apples-to-apples with the 7-fold overlap.
    """
    con = duckdb.connect(str(COPAIRS_DB), read_only=True)
    try:
        df = query_activity_results(
            con,
            dataset=dataset,
            preprocessing="activity_no_target2",
            filter_name="all_sources",
            activity_params="default",
        )
    finally:
        con.close()
    active = set(df.loc[df["below_corrected_p"], "Metadata_JCP2022"])
    logger.info(f"Active compounds (copairs FDR): {len(active):,} / {len(df):,}")
    return active


def align_data(
    adata_profiles,
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fingerprints and morphology profiles by JCP2022 ID.

    Returns:
        X_fp: Fingerprint matrix (aligned)
        X_morph: Morphology feature matrix (aligned)
        jcp_ids: Array of JCP2022 IDs
    """
    morph_jcp = adata_profiles.obs["JCP2022"].values
    common_jcp = sorted(set(fp_jcp) & set(morph_jcp))
    logger.info(f"Common compounds: {len(common_jcp):,}")

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
    logger.info(f"Valid compounds: {valid_mask.sum():,}")

    return X_fp[valid_mask], X_morph[valid_mask], jcp_ids[valid_mask]


# =============================================================================
# k-NN Based Pairwise Similarity Computation
# =============================================================================


def compute_structural_knn(
    X_fp: np.ndarray,
    jcp_ids: np.ndarray,
    k: int = K_NEIGHBORS,
) -> sc.AnnData:
    """Compute k-NN in structural (fingerprint) space.

    Uses L2-normalized cosine similarity which approximates Tanimoto (ρ=0.998).

    Returns:
        AnnData with k-NN graph in obsp['distances'] and obsp['connectivities']
    """
    # L2 normalize fingerprints
    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)

    logger.info(f"Computing structural k-NN (k={k})...")
    adata = sc.AnnData(X=X_fp_norm)
    adata.obs["JCP2022"] = jcp_ids
    adata.obs_names = jcp_ids.astype(str)

    # Select neighbors function based on GPU availability (deferred import)
    if HAS_GPU:
        import rapids_singlecell as rsc

        neighbors_fn = rsc.pp.neighbors
        logger.info("Using GPU-accelerated k-NN (rapids_singlecell)")
    else:
        neighbors_fn = sc.pp.neighbors
        logger.info("Using CPU k-NN (scanpy)")

    neighbors_fn(adata, n_neighbors=k, use_rep="X", metric="cosine")
    logger.info(f"  Edges: {adata.obsp['connectivities'].nnz:,}")

    return adata


def _paired_dot(X: np.ndarray, rows: np.ndarray, cols: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Dot product of X[rows[k]] and X[cols[k]] for every k, in chunks.

    Indexing all pairs at once copies X[rows] and X[cols] in full: for 5.6M pairs of
    2048-bit fingerprints that is two 46 GB arrays. Chunking gives the same values.
    """
    out = np.empty(len(rows), dtype=X.dtype)
    for start in range(0, len(rows), chunk):
        stop = start + chunk
        out[start:stop] = np.einsum("ij,ij->i", X[rows[start:stop]], X[cols[start:stop]])
    return out


def extract_pairwise_similarities(
    adata_struct: sc.AnnData,
    X_fp: np.ndarray,
    X_morph: np.ndarray,
    jcp_ids: np.ndarray,
) -> pd.DataFrame:
    """Extract pairwise similarities for all k-NN pairs.

    For each pair in the k-NN graph, compute both structural and morphological
    similarity using cosine similarity on L2-normalized vectors.

    Returns:
        DataFrame with columns: i, j, jcp_i, jcp_j, struct_sim, morph_sim
    """
    logger.info("Extracting pairwise similarities for k-NN pairs...")

    # L2 normalize both feature spaces
    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)
    X_morph_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)

    # Get k-NN indices from distance matrix
    dist_matrix = adata_struct.obsp["distances"]

    # Extract all (i, j) pairs from sparse matrix
    rows, cols = dist_matrix.nonzero()

    # Only keep upper triangle (i < j) to avoid duplicates
    upper_mask = rows < cols
    rows = rows[upper_mask]
    cols = cols[upper_mask]

    logger.info(f"  Computing similarities for {len(rows):,} pairs...")

    # Compute structural similarity (cosine = 1 - cosine_distance)
    # For L2-normalized vectors: cosine_sim = dot product
    struct_sim = _paired_dot(X_fp_norm, rows, cols)

    # Compute morphological similarity
    morph_sim = _paired_dot(X_morph_norm, rows, cols)

    pairs_df = pd.DataFrame(
        {
            "i": rows,
            "j": cols,
            "jcp_i": jcp_ids[rows],
            "jcp_j": jcp_ids[cols],
            "struct_sim": struct_sim,
            "morph_sim": morph_sim,
        }
    )

    logger.info(f"  Extracted {len(pairs_df):,} unique pairs")

    return pairs_df


# =============================================================================
# Q1: Structure-Morphology Cliff Analysis
# =============================================================================


def analyze_structure_morphology_relationship(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
) -> dict:
    """Analyze the relationship between structural and morphological similarity.

    Classifies pairs into quadrants:
    - Concordant: high struct, high morph (structure predicts phenotype)
    - Cliffs: high struct, low morph (activity cliffs)
    - Reverse cliffs: low struct, high morph (different structures, same phenotype)
    - Neutral: low struct, low morph (expected divergence)

    Returns:
        Dict with statistics and cliff pair counts
    """
    logger.info("Analyzing structure-morphology relationship...")

    n_pairs = len(pairs_df)
    struct_sim = pairs_df["struct_sim"].values
    morph_sim = pairs_df["morph_sim"].values

    # Compute correlation
    from scipy.stats import spearmanr

    spearman_r, spearman_p = spearmanr(struct_sim, morph_sim)
    logger.info(f"  Spearman correlation: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

    # Define quadrants using thresholds
    morph_high = 1 - morph_low  # Symmetric threshold for "high" morph

    high_struct_mask = struct_sim >= struct_high
    low_struct_mask = struct_sim < struct_high
    high_morph_mask = morph_sim >= morph_high
    low_morph_mask = morph_sim < morph_low

    # Count quadrants
    n_concordant = int((high_struct_mask & high_morph_mask).sum())
    n_cliffs = int((high_struct_mask & low_morph_mask).sum())
    n_reverse = int((low_struct_mask & high_morph_mask).sum())
    n_neutral = int((low_struct_mask & low_morph_mask).sum())
    # Pairs in neither corner (intermediate morphological similarity at any structure)
    n_intermediate = int(n_pairs - n_concordant - n_cliffs - n_reverse - n_neutral)
    # All structurally similar pairs (Tanimoto >= struct_high), regardless of morphology
    n_struct_similar = int(high_struct_mask.sum())

    # Additional stats
    pct_cliffs = n_cliffs / n_pairs * 100 if n_pairs > 0 else 0
    pct_concordant = n_concordant / n_pairs * 100 if n_pairs > 0 else 0

    stats = {
        "n_pairs": n_pairs,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "thresholds": {
            "struct_high": struct_high,
            "morph_low": morph_low,
            "morph_high": float(morph_high),
        },
        "quadrants": {
            "concordant": n_concordant,
            "cliffs": n_cliffs,
            "reverse_cliffs": n_reverse,
            "neutral": n_neutral,
            "intermediate": n_intermediate,
        },
        "n_struct_similar_pairs": n_struct_similar,
        "percentages": {
            "pct_cliffs": pct_cliffs,
            "pct_concordant": pct_concordant,
        },
        "summary_stats": {
            "struct_sim_mean": float(struct_sim.mean()),
            "struct_sim_std": float(struct_sim.std()),
            "morph_sim_mean": float(morph_sim.mean()),
            "morph_sim_std": float(morph_sim.std()),
        },
    }

    logger.info("  Quadrant counts:")
    logger.info(f"    Concordant (high-high): {n_concordant:,} ({pct_concordant:.2f}%)")
    logger.info(f"    Cliffs (high struct, low morph): {n_cliffs:,} ({pct_cliffs:.2f}%)")
    logger.info(f"    Reverse cliffs: {n_reverse:,}")
    logger.info(f"    Neutral: {n_neutral:,}")

    return stats


def identify_cliff_pairs(
    pairs_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
) -> pd.DataFrame:
    """Identify all activity cliff pairs.

    Activity cliffs are pairs with high structural similarity but low
    morphological similarity. Returns all cliffs ranked by cliff score:
    struct_sim - morph_sim.

    Returns:
        DataFrame with all cliff pairs sorted by cliff score
    """
    logger.info("Identifying cliff pairs...")

    # Filter to high structural similarity pairs
    cliff_candidates = pairs_df[pairs_df["struct_sim"] >= struct_high].copy()
    logger.info(f"  Candidates (struct >= {struct_high}): {len(cliff_candidates):,}")

    # Filter to low morphological similarity
    cliff_pairs = cliff_candidates[cliff_candidates["morph_sim"] < morph_low].copy()
    logger.info(f"  Cliffs (morph < {morph_low}): {len(cliff_pairs):,}")

    if len(cliff_pairs) == 0:
        logger.warning("  No cliff pairs found with current thresholds")
        return pd.DataFrame()

    # Compute cliff score (higher = more extreme cliff)
    cliff_pairs["cliff_score"] = cliff_pairs["struct_sim"] - cliff_pairs["morph_sim"]

    cliff_pairs = cliff_pairs.sort_values("cliff_score", ascending=False)

    return cliff_pairs[["jcp_i", "jcp_j", "struct_sim", "morph_sim", "cliff_score"]]


# =============================================================================
# Q2: Target-Informed Cliff Analysis
# =============================================================================


def analyze_target_informed_cliffs(
    pairs_df: pd.DataFrame,
    target_df: pd.DataFrame,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
    min_compounds: int = 5,
) -> dict:
    """Analyze cliffs within compounds sharing the same ChEMBL target.

    For each target with sufficient compound coverage, identify pairs of
    compounds that share the target and compute cliff statistics.

    Args:
        pairs_df: Pairwise similarity DataFrame
        target_df: Target annotations (JCP2022, targets list)
        struct_high: Structural similarity threshold
        morph_low: Morphological similarity threshold
        min_compounds: Minimum compounds per target for analysis

    Returns:
        Dict with per-target cliff statistics
    """
    logger.info("Analyzing target-informed cliffs...")

    # Expand target annotations (one row per compound-target pair)
    expanded = target_df.explode("targets")
    expanded = expanded.rename(columns={"targets": "target"})
    logger.info(f"  Expanded annotations: {len(expanded):,} compound-target pairs")

    # Count compounds per target
    target_counts = expanded.groupby("target").size().reset_index(name="n_compounds")
    targets_with_coverage = target_counts[target_counts["n_compounds"] >= min_compounds]
    logger.info(f"  Targets with >= {min_compounds} compounds: {len(targets_with_coverage):,}")

    # Create JCP2022 to targets mapping
    jcp_to_targets = expanded.groupby("JCP2022")["target"].apply(set).to_dict()

    # For each pair, find shared targets
    logger.info("  Finding shared targets for pairs...")
    shared_targets = []
    for _, row in pairs_df.iterrows():
        jcp_i, jcp_j = row["jcp_i"], row["jcp_j"]
        targets_i = jcp_to_targets.get(jcp_i, set())
        targets_j = jcp_to_targets.get(jcp_j, set())
        shared = targets_i & targets_j
        shared_targets.append(shared if shared else None)

    pairs_df = pairs_df.copy()
    pairs_df["shared_targets"] = shared_targets

    # Filter to pairs with shared targets
    target_pairs = pairs_df[pairs_df["shared_targets"].notna()].copy()
    logger.info(f"  Pairs with shared targets: {len(target_pairs):,}")

    if len(target_pairs) == 0:
        logger.warning("  No pairs with shared targets found")
        return {"n_target_pairs": 0}

    # Analyze within-target statistics
    target_pair_stats = analyze_structure_morphology_relationship(target_pairs, struct_high, morph_low)
    target_pair_stats["n_target_pairs"] = len(target_pairs)

    # Find cliffs within target pairs
    target_cliffs = target_pairs[(target_pairs["struct_sim"] >= struct_high) & (target_pairs["morph_sim"] < morph_low)]
    target_pair_stats["n_target_cliffs"] = len(target_cliffs)

    # Per-target cliff counts
    target_cliff_counts = {}
    for _, row in target_cliffs.iterrows():
        for target in row["shared_targets"]:
            target_cliff_counts[target] = target_cliff_counts.get(target, 0) + 1

    # Top targets with most cliffs
    top_cliff_targets = sorted(target_cliff_counts.items(), key=lambda x: -x[1])[:20]
    target_pair_stats["top_cliff_targets"] = [{"target": t, "n_cliffs": n} for t, n in top_cliff_targets]

    logger.info(f"  Target cliffs found: {len(target_cliffs):,}")
    if top_cliff_targets:
        logger.info(f"  Top cliff target: {top_cliff_targets[0][0]} ({top_cliff_targets[0][1]} cliffs)")

    return target_pair_stats


# =============================================================================
# Visualization
# =============================================================================


def plot_structure_morphology_hexbin(
    pairs_df: pd.DataFrame,
    output_dir: Path,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    morph_low: float = MORPH_LOW_THRESHOLD,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create hexbin plot of structural vs morphological similarity with marginals."""
    logger.info("Creating structure-morphology hexbin plot...")

    # Compute statistics for annotation
    from scipy.stats import spearmanr

    spearman_r, _ = spearmanr(pairs_df["struct_sim"], pairs_df["morph_sim"])
    stats_annotation = f"Spearman ρ = {spearman_r:.3f}\nn = {len(pairs_df):,}"

    # Compute cliff counts for annotation
    n_pairs = len(pairs_df)
    high_struct = pairs_df["struct_sim"] >= struct_high
    low_morph = pairs_df["morph_sim"] < morph_low
    n_cliffs = (high_struct & low_morph).sum()
    stats_annotation += f"\nCliffs: {n_cliffs:,} ({n_cliffs / n_pairs * 100:.2f}%)"

    # Use the visualization module's hexbin function
    plot_hexbin_with_marginals(
        pairs_df,
        x_col="struct_sim",
        y_col="morph_sim",
        x_label="Structural Similarity (Tanimoto ≈ cosine)",
        y_label="Morphological Similarity (cosine)",
        x_lim=(0, 1),
        y_lim=(-0.5, 1),
        title="Structure vs Morphology Similarity",
        stats_annotation=stats_annotation,
        output_dir=output_dir,
        filename="structure_morphology_hexbin.png",
        save_pdf=True,
        dpi=dpi,
    )
    logger.info(f"Saved: {output_dir / 'structure_morphology_hexbin.png'}")


def plot_cliff_distribution(
    pairs_df: pd.DataFrame,
    output_dir: Path,
    struct_high: float = STRUCT_HIGH_THRESHOLD,
    struct_low: float = 0.5,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot morphological similarity distributions for structurally similar vs dissimilar pairs."""
    logger.info("Creating cliff distribution plot...")

    similar_pairs = pairs_df[pairs_df["struct_sim"] >= struct_high]
    dissimilar_pairs = pairs_df[pairs_df["struct_sim"] < struct_low]

    if len(similar_pairs) == 0:
        logger.warning("No structurally similar pairs found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    from scipy.stats import gaussian_kde

    x_grid = np.linspace(-0.5, 1.0, 300)
    for data, color, label in [
        (similar_pairs["morph_sim"], "#1f77b4", f"Structurally similar (>= {struct_high}, n={len(similar_pairs):,})"),
        (dissimilar_pairs["morph_sim"], "#ff7f0e", f"Structurally dissimilar (< {struct_low}, n={len(dissimilar_pairs):,})"),
    ]:
        kde = gaussian_kde(data)
        ax.plot(x_grid, kde(x_grid), linewidth=2, color=color, label=label)

    ax.set_xlabel("Morphological Similarity (cosine)")
    ax.set_ylabel("Density")
    ax.set_title("Morphological Similarity: Structurally Similar vs Dissimilar Pairs")

    mean_similar = similar_pairs["morph_sim"].mean()
    mean_dissimilar = dissimilar_pairs["morph_sim"].mean()
    ax.axvline(mean_similar, color="#1f77b4", linestyle="--", alpha=0.7)
    ax.axvline(mean_dissimilar, color="#ff7f0e", linestyle="--", alpha=0.7)

    stats_text = (
        f"Similar: mean={mean_similar:.3f}, std={similar_pairs['morph_sim'].std():.3f}\n"
        f"Dissimilar: mean={mean_dissimilar:.3f}, std={dissimilar_pairs['morph_sim'].std():.3f}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / "cliff_distribution.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "cliff_distribution.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'cliff_distribution.png'}")
    plt.close(fig)


def plot_target_cliff_analysis(
    target_stats: dict,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot bar chart of targets with most cliffs."""
    logger.info("Creating target cliff analysis plot...")

    top_targets = target_stats.get("top_cliff_targets", [])
    if not top_targets:
        logger.warning("No target cliff data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    targets = [t["target"] for t in top_targets]
    n_cliffs = [t["n_cliffs"] for t in top_targets]

    y_pos = np.arange(len(targets))
    ax.barh(y_pos, n_cliffs, color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Cliff Pairs")
    ax.set_title("Targets with Most Activity Cliff Pairs")

    plt.tight_layout()
    fig.savefig(output_dir / "target_cliff_analysis.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "target_cliff_analysis.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'target_cliff_analysis.png'}")
    plt.close(fig)


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    q1_stats: dict,
    q2_stats: dict,
    cliff_pairs_df: pd.DataFrame,
    n_compounds: int,
    n_pairs: int,
    output_dir: Path,
    null_cliff_results: dict | None = None,
    null_corr_results: dict | None = None,
) -> dict:
    """Generate summary JSON with all results."""
    summary = {
        "task": "Activity cliffs analysis",
        "research_questions": [
            "Q1: Do structurally similar compounds have similar morphological profiles?",
            "Q2: For compounds sharing ChEMBL targets, do similar structures yield similar phenotypes?",
        ],
        "n_compounds": n_compounds,
        "n_pairs_analyzed": n_pairs,
        "use_gpu": HAS_GPU,
        "q1_structure_morphology": {
            "spearman_correlation": q1_stats["spearman_r"],
            "spearman_p_value": q1_stats["spearman_p"],
            "thresholds": q1_stats["thresholds"],
            "quadrants": q1_stats["quadrants"],
            "n_struct_similar_pairs": q1_stats.get("n_struct_similar_pairs"),
            "interpretation": "",
        },
        "q2_target_informed": {
            "n_target_pairs": q2_stats.get("n_target_pairs", 0),
            "n_target_cliffs": q2_stats.get("n_target_cliffs", 0),
            "spearman_correlation": q2_stats.get("spearman_r"),
            "top_cliff_targets": q2_stats.get("top_cliff_targets", [])[:10],
        },
        "cliff_pairs": {
            "n_cliff_pairs": len(cliff_pairs_df) if cliff_pairs_df is not None else 0,
            "cliff_pair_rate": (len(cliff_pairs_df) / n_pairs) if cliff_pairs_df is not None and n_pairs else None,
            "top_cliff_score": float(cliff_pairs_df["cliff_score"].max()) if len(cliff_pairs_df) > 0 else None,
        },
    }

    # Add null model results if available
    if null_cliff_results is not None or null_corr_results is not None:
        summary["null_model"] = {}
        if null_cliff_results is not None:
            summary["null_model"]["n_permutations"] = null_cliff_results.get("n_permutations", 1000)
            summary["null_model"]["cliff_rate"] = {
                "observed": null_cliff_results["observed_rate"],
                "null_mean": null_cliff_results["null_mean"],
                "null_std": null_cliff_results["null_std"],
                "z_score": null_cliff_results["z_score"],
                "p_value": null_cliff_results["p_value"],
            }
        if null_corr_results is not None:
            summary["null_model"]["correlation"] = {
                "observed": null_corr_results["observed_r"],
                "null_mean": null_corr_results["null_mean"],
                "null_std": null_corr_results["null_std"],
                "z_score": null_corr_results["z_score"],
            }

    # Generate interpretation
    r = q1_stats["spearman_r"]
    if null_corr_results is not None:
        # Use null model context for interpretation
        z = null_corr_results["z_score"]
        if abs(z) > 10:
            interpretation = (
                f"Correlation (ρ={r:.3f}) is highly significant vs null (z={z:.1f}): "
                "structure-morphology relationship is real but weak"
            )
        elif abs(z) > 3:
            interpretation = (
                f"Correlation (ρ={r:.3f}) is significant vs null (z={z:.1f}): structure modestly predicts morphology"
            )
        else:
            interpretation = (
                f"Correlation (ρ={r:.3f}) is not significant vs null (z={z:.1f}): "
                "no meaningful structure-morphology relationship"
            )
    else:
        # Fallback to original interpretation
        if r > 0.5:
            interpretation = f"Strong positive correlation (ρ={r:.3f}): structurally similar compounds tend to have similar morphology"
        elif r > 0.3:
            interpretation = f"Moderate positive correlation (ρ={r:.3f}): some structure-morphology relationship"
        elif r > 0.1:
            interpretation = f"Weak positive correlation (ρ={r:.3f}): limited structure-morphology relationship"
        else:
            interpretation = f"Very weak/no correlation (ρ={r:.3f}): structure does not predict morphology"

    summary["q1_structure_morphology"]["interpretation"] = interpretation

    # Log summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Compounds: {n_compounds:,}")
    logger.info(f"Pairs analyzed: {n_pairs:,}")
    logger.info(f"Q1 - Structure-Morphology correlation: ρ = {r:.4f}")
    logger.info(f"Q1 - Cliff pairs: {q1_stats['quadrants']['cliffs']:,}")
    logger.info(f"Q2 - Target pairs: {q2_stats.get('n_target_pairs', 0):,}")
    logger.info(f"Q2 - Target cliffs: {q2_stats.get('n_target_cliffs', 0):,}")
    logger.info(f"Interpretation: {interpretation}")

    # Save standardized summary envelope
    description = (
        "Activity cliffs analysis using Cell Painting morphology profiles: compares structural "
        "(Morgan fingerprint) similarity against morphological similarity across structural k-NN "
        "compound pairs to find cliffs (similar structure, different phenotype) and reverse cliffs, "
        "including a target-informed pass over pairs sharing ChEMBL annotations. Produces pairwise "
        "similarity tables, cliff pair lists, and hexbin/distribution/target-cliff figures."
    )
    summary = write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=summary,
        files=index_files(output_dir),
    )
    logger.info(f"Saved: {output_dir / 'summary.json'}")

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
    sample_frac: float = typer.Option(
        0.1,
        "--sample",
        help="Fraction of data to sample (0.1 = 10%%, 1.0 = full dataset)",
    ),
    k_neighbors: int = typer.Option(
        K_NEIGHBORS,
        "--k-neighbors",
        "-k",
        help="Number of structural nearest neighbors to analyze",
    ),
    struct_threshold: float = typer.Option(
        STRUCT_HIGH_THRESHOLD,
        "--struct-high",
        help="Structural similarity threshold for 'similar'",
    ),
    morph_threshold: float = typer.Option(
        MORPH_LOW_THRESHOLD,
        "--morph-low",
        help="Morphological similarity threshold for 'different'",
    ),
    active_only: bool = typer.Option(
        False, "--active-only", help="Restrict to phenotypically active compounds (recomputes k-NN on that subset)"
    ),
    skip_eda: bool = typer.Option(False, "--skip-eda", help="Skip EDA plots"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
) -> None:
    """Analyze activity cliffs using Cell Painting morphology profiles."""
    # Set output directory (separate subdir so --active-only never clobbers the full run)
    if output_dir is None:
        output_dir = OUTPUT_DIR / (f"{dataset}_active" if active_only else dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"k-neighbors: {k_neighbors}")
    logger.info(f"Struct threshold: {struct_threshold}")
    logger.info(f"Morph threshold: {morph_threshold}")

    # Check database exists
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Load data
        logger.info("Loading data...")
        fp_array, fp_jcp = load_fingerprints()
        adata = load_profiles(dataset, level="perturbation")
        target_df = load_chembl_targets(metadata_con)

        # 2. Align data
        X_fp, X_morph, jcp_ids = align_data(adata, fp_array, fp_jcp)

        # 2b. Restrict to phenotypically active compounds (matches 2.05 panel C)
        if active_only:
            active_set = load_active_compounds(dataset)
            keep = np.array([j in active_set for j in jcp_ids])
            X_fp, X_morph, jcp_ids = X_fp[keep], X_morph[keep], jcp_ids[keep]
            logger.info(f"Active-only: kept {keep.sum():,} / {len(keep):,} compounds")

        # 3. Sample data if requested
        if sample_frac < 1.0:
            n_sample = int(len(X_fp) * sample_frac)
            rng = np.random.default_rng(RANDOM_STATE)
            sample_idx = rng.choice(len(X_fp), size=n_sample, replace=False)
            X_fp = X_fp[sample_idx]
            X_morph = X_morph[sample_idx]
            jcp_ids = jcp_ids[sample_idx]
            logger.info(f"Sampled {sample_frac:.0%} of data: {n_sample:,} compounds")

        n_compounds = len(jcp_ids)
        logger.info(f"Final dataset: {n_compounds:,} compounds")

        # 4. Compute structural k-NN
        adata_struct = compute_structural_knn(X_fp, jcp_ids, k=k_neighbors)

        # 5. Extract pairwise similarities for k-NN pairs
        pairs_df = extract_pairwise_similarities(adata_struct, X_fp, X_morph, jcp_ids)
        n_pairs = len(pairs_df)

        # 6. Q1: Structure-Morphology analysis
        logger.info("\n=== Q1: Structure-Morphology Cliff Analysis ===")
        q1_stats = analyze_structure_morphology_relationship(
            pairs_df, struct_high=struct_threshold, morph_low=morph_threshold
        )

        # 6b. Null model analysis
        logger.info("\n=== Null Model Analysis ===")
        logger.info("Running permutation null model for cliff rate...")
        null_cliff_results = permutation_null_cliff_rate(
            pairs_df["struct_sim"].values,
            pairs_df["morph_sim"].values,
            struct_threshold=struct_threshold,
            morph_threshold=morph_threshold,
            n_permutations=1000,
        )
        logger.info(
            f"  Cliff rate: observed={null_cliff_results['observed_rate']:.4f}, "
            f"null={null_cliff_results['null_mean']:.4f}±{null_cliff_results['null_std']:.4f}, "
            f"z={null_cliff_results['z_score']:.1f}"
        )

        logger.info("Running permutation null model for correlation...")
        null_corr_results = permutation_null_correlation(
            pairs_df["struct_sim"].values,
            pairs_df["morph_sim"].values,
            n_permutations=100,
        )
        logger.info(
            f"  Correlation: observed={null_corr_results['observed_r']:.4f}, "
            f"null={null_corr_results['null_mean']:.4f}±{null_corr_results['null_std']:.4f}, "
            f"z={null_corr_results['z_score']:.1f}"
        )

        # 7. Identify top cliff pairs
        cliff_pairs_df = identify_cliff_pairs(pairs_df, struct_high=struct_threshold, morph_low=morph_threshold)

        # 8. Q2: Target-informed analysis
        logger.info("\n=== Q2: Target-Informed Cliff Analysis ===")
        # Filter target_df to compounds in our dataset
        target_df_filtered = target_df[target_df["JCP2022"].isin(jcp_ids)]
        logger.info(f"Compounds with target annotations: {len(target_df_filtered):,}")

        q2_stats = analyze_target_informed_cliffs(
            pairs_df, target_df_filtered, struct_high=struct_threshold, morph_low=morph_threshold
        )

        # 9. Create visualizations
        if not skip_eda:
            plot_structure_morphology_hexbin(
                pairs_df, output_dir, struct_high=struct_threshold, morph_low=morph_threshold, dpi=dpi
            )
            plot_cliff_distribution(pairs_df, output_dir, struct_high=struct_threshold, dpi=dpi)
            if q2_stats.get("top_cliff_targets"):
                plot_target_cliff_analysis(q2_stats, output_dir, dpi=dpi)

        # 10. Save results
        pairs_df.to_parquet(output_dir / "all_pairs.parquet", index=False)
        logger.info(f"Saved: {output_dir / 'all_pairs.parquet'}")

        if len(cliff_pairs_df) > 0:
            cliff_pairs_df.to_csv(output_dir / "cliff_pairs.csv", index=False)
            cliff_pairs_df.to_parquet(output_dir / "cliff_pairs.parquet", index=False)
            logger.info(f"Saved: {output_dir / 'cliff_pairs.csv'} ({len(cliff_pairs_df):,} pairs)")

        # 11. Generate summary
        generate_summary(
            q1_stats,
            q2_stats,
            cliff_pairs_df,
            n_compounds,
            n_pairs,
            output_dir,
            null_cliff_results=null_cliff_results,
            null_corr_results=null_corr_results,
        )

        logger.success(f"Activity cliffs analysis complete: {output_dir}")

    finally:
        metadata_con.close()


if __name__ == "__main__":
    app()
