"""PhenoSeeker procedure transfer using scaffold-based morphology clustering.

Transfers the released PhenoSeeker clustering and selection procedure to JUMP
Bemis-Murcko scaffold memberships and CellProfiler features. PhenoSeeker recomputed
scaffolds from InChI and used DINOv2-Giant embeddings, so this is a method transfer
rather than a replication.

Research Question: Can we identify scaffold series where structurally similar
compounds (same Bemis-Murcko scaffold) exhibit dramatically different morphological
phenotypes?

Approach, with clustering and selection matched to the released figure code:
1. Group compounds by the JUMP RDKit Bemis-Murcko scaffold annotation
2. Retain scaffold series with more than 6 profiled compounds
3. Perform Ward hierarchical clustering on Euclidean profile distance
4. Retain two-cluster solutions with at least 3 compounds in each cluster
5. Flag series where mean intra-cluster cosine exceeds both mean inter-cluster
   cosine by 0.2 and an absolute cosine of 0.2

Reference: PhenoSeeker found 81 of 532 eligible scaffold series with activity cliffs.

Usage:
    # Default run (publication-code threshold: N>6 compounds)
    pixi run python notebooks/2.32-seal-activity-cliffs-phenoseeker.py

    # More stringent base threshold (primary N>10; sensitivity N>=10)
    pixi run python notebooks/2.32-seal-activity-cliffs-phenoseeker.py --min-compounds 10

    # Different threshold
    pixi run python notebooks/2.32-seal-activity-cliffs-phenoseeker.py --separation-threshold 0.3

    # Different dataset (e.g., deep learning embeddings)
    pixi run python notebooks/2.32-seal-activity-cliffs-phenoseeker.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/activity-cliffs-phenoseeker/{dataset}/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import load_profiles
from jump_production.summaries import write_summary

DATA_DIR = Path("data")

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "activity-cliffs-phenoseeker"

# Default parameters
SCAFFOLD_SIZE_THRESHOLD = 6  # Released code uses len(group) > 6, not >= 6.
MIN_COMPOUNDS = 6  # Paper-text sensitivity includes N=6; primary code match requires N>6.
N_CLUSTERS = 2  # Number of clusters for hierarchical clustering
MIN_CLUSTER_SIZE = 3
SEPARATION_THRESHOLD = 0.2
MIN_INTRA_SIMILARITY = 0.2
PHENOSEEKER_RELEASE_COMMIT = "147c70795e2bc59d7aeec71fe040b14dea933560"
PHENOSEEKER_NOTEBOOK_SHA256 = "d82249d5662928a5b0a238c6ca15375919f6c3945cbad22e240593abd628ced3"
PHENOSEEKER_TABLE_SHA256 = "6c4381fe2c072cf6f5b28cb2efa4efea5a55315b1760baa0358848ca52d24088"

# Try to import RDKit for scaffold visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available - scaffold structure plots will be skipped")

RANDOM_STATE = 42

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_scaffolds_with_counts(
    con: duckdb.DuckDBPyConnection,
    min_compounds: int = MIN_COMPOUNDS,
) -> pd.DataFrame:
    """Load scaffolds with N>=min_compounds.

    Returns DataFrame with columns: scaffold, n_compounds, compound_ids (list)
    """
    df = con.execute(
        """
        SELECT
            Metadata_MurckoScaffold as scaffold,
            COUNT(*) as n_compounds,
            LIST(Metadata_JCP2022 ORDER BY Metadata_JCP2022) as compound_ids
        FROM compound_metadata
        WHERE Metadata_ValidMol = TRUE
          AND Metadata_MurckoScaffold IS NOT NULL
          AND Metadata_MurckoScaffold != ''
        GROUP BY Metadata_MurckoScaffold
        HAVING COUNT(*) >= ?
        ORDER BY n_compounds DESC, scaffold
    """,
        [min_compounds],
    ).df()

    logger.info(f"Loaded {len(df):,} scaffolds with N>={min_compounds} compounds")
    logger.info(f"Total compounds in these scaffolds: {df['n_compounds'].sum():,}")

    return df


def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load Morgan fingerprints (2048-bit)."""
    fp_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds × {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


def global_pairwise_cosine_moments(matrix: np.ndarray) -> dict:
    """Compute exact mean and SD over all off-diagonal profile cosine values.

    The feature-space Gram identity avoids materializing the full compound-by-
    compound similarity matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix[(norms > 0) & np.isfinite(norms)]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    n_profiles = len(matrix)
    denominator = n_profiles * (n_profiles - 1)
    sum_vector = matrix.sum(axis=0)
    mean = (float(sum_vector @ sum_vector) - n_profiles) / denominator
    feature_gram = matrix.T @ matrix
    mean_square = (float(np.square(feature_gram).sum()) - n_profiles) / denominator
    standard_deviation = float(np.sqrt(max(mean_square - mean**2, 0.0)))
    return {
        "n_profiles": n_profiles,
        "mean_pairwise_cosine": mean,
        "sd_pairwise_cosine": standard_deviation,
        "mean_plus_3sd": float(mean + 3 * standard_deviation),
    }


def align_morphology_to_scaffolds(
    adata,
    scaffold_df: pd.DataFrame,
    min_compounds: int = MIN_COMPOUNDS,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Align morphology profiles to scaffold compound lists.

    Returns:
        Dict mapping scaffold SMILES -> (X_morph array, jcp_ids list)
    """
    logger.info("Aligning morphology profiles to scaffolds...")

    # Build JCP2022 -> index mapping for profiles
    profile_jcp = adata.obs["JCP2022"].values
    profile_idx = {k: i for i, k in enumerate(profile_jcp)}
    profile_set = set(profile_jcp)

    scaffold_data = {}
    n_scaffolds_with_data = 0
    n_compounds_aligned = 0

    for _, row in scaffold_df.iterrows():
        scaffold = row["scaffold"]
        compound_ids = row["compound_ids"]

        # Filter to compounds with morphology data
        valid_ids = [jcp for jcp in compound_ids if jcp in profile_set]

        if len(valid_ids) < min_compounds:
            continue

        # Extract morphology features
        indices = [profile_idx[jcp] for jcp in valid_ids]
        X_morph = adata.X[indices].astype(np.float32)

        # Filter out compounds with zero-norm or NaN features
        norms = np.linalg.norm(X_morph, axis=1)
        valid_mask = (norms > 0) & ~np.isnan(norms)

        if valid_mask.sum() < min_compounds:
            continue

        X_morph = X_morph[valid_mask]
        valid_ids = [valid_ids[i] for i in range(len(valid_ids)) if valid_mask[i]]

        scaffold_data[scaffold] = (X_morph, valid_ids)
        n_scaffolds_with_data += 1
        n_compounds_aligned += len(valid_ids)

    logger.info(f"Scaffolds with valid morphology data: {n_scaffolds_with_data:,}")
    logger.info(f"Total compounds aligned: {n_compounds_aligned:,}")

    return scaffold_data


# =============================================================================
# Hierarchical Clustering Analysis
# =============================================================================


def analyze_scaffold_series(
    X_morph: np.ndarray,
    jcp_ids: list[str],
    n_clusters: int = N_CLUSTERS,
) -> dict | None:
    """Apply the released PhenoSeeker clustering and similarity calculations.

    Args:
        X_morph: Morphology feature matrix for scaffold compounds
        jcp_ids: JCP2022 IDs
        n_clusters: Number of clusters for cutting dendrogram

    Returns:
        Analysis dictionary, or None when a cluster has fewer than three members.
    """
    n_compounds = len(jcp_ids)

    # PhenoSeeker clusters the unnormalized profiles using Ward linkage over
    # Euclidean distance, then evaluates the resulting clusters with cosine.
    euclidean_distances = pdist(X_morph, metric="euclidean")
    Z = linkage(euclidean_distances, method="ward")
    labels = fcluster(Z, n_clusters, criterion="maxclust")
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) != n_clusters or int(counts.min()) < MIN_CLUSTER_SIZE:
        return None
    # Canonicalize otherwise arbitrary cluster numbers by the smallest member ID.
    label_order = sorted(
        unique_labels,
        key=lambda label: min(jcp for jcp, value in zip(jcp_ids, labels, strict=True) if value == label),
    )
    label_mapping = {old: new for new, old in enumerate(label_order, start=1)}
    labels = np.asarray([label_mapping[label] for label in labels])
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))

    X_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)
    cosine_matrix = X_norm @ X_norm.T
    intra_similarities = []
    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        off_diagonal = ~np.eye(len(indices), dtype=bool)
        intra_similarities.append(float(cosine_matrix[np.ix_(indices, indices)][off_diagonal].mean()))
    inter_similarities = []
    for position, label_i in enumerate(unique_labels):
        indices_i = np.flatnonzero(labels == label_i)
        for label_j in unique_labels[position + 1 :]:
            indices_j = np.flatnonzero(labels == label_j)
            inter_similarities.append(float(cosine_matrix[np.ix_(indices_i, indices_j)].mean()))

    # The released code gives each cluster (and each cluster pair) equal weight.
    mean_intra_sim = float(np.mean(intra_similarities))
    mean_inter_sim = float(np.mean(inter_similarities))
    separation = mean_intra_sim - mean_inter_sim

    return {
        "n_compounds": n_compounds,
        "labels": labels.tolist(),
        "jcp_ids": jcp_ids,
        "mean_intra_similarity": mean_intra_sim,
        "mean_inter_similarity": mean_inter_sim,
        "separation_score": separation,
        "cluster_sizes": cluster_sizes,
        "linkage_matrix": Z,
        "distance_matrix": 1.0 - cosine_matrix,
    }


def analyze_all_scaffolds(
    scaffold_data: dict[str, tuple[np.ndarray, list[str]]],
    n_clusters: int = N_CLUSTERS,
) -> list[dict]:
    """Analyze all scaffolds and compute separation scores.

    Returns:
        List of dicts with scaffold analysis results
    """
    logger.info(f"Analyzing {len(scaffold_data):,} scaffolds...")

    results = []
    for scaffold, (X_morph, jcp_ids) in scaffold_data.items():
        result = analyze_scaffold_series(X_morph, jcp_ids, n_clusters)
        if result is None:
            continue
        result["scaffold"] = scaffold
        results.append(result)

    # Stable output order for equal separation scores.
    results.sort(key=lambda x: (-x["separation_score"], x["scaffold"]))

    logger.info(f"Analysis complete for {len(results):,} scaffolds")

    return results


def add_structural_similarities(
    scaffold_results: list[dict],
    fp_array: np.ndarray,
    fp_jcp: np.ndarray,
) -> None:
    """Add structural (Tanimoto) intra/inter-cluster similarities to scaffold results.

    Uses the cluster labels from morphology clustering and computes structural
    similarity within and between clusters.

    Modifies scaffold_results in place.
    """
    logger.info("Computing structural similarities for scaffolds...")

    # Build JCP2022 -> fingerprint index mapping
    fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    fp_set = set(fp_jcp)

    for result in scaffold_results:
        jcp_ids = result["jcp_ids"]
        labels = result["labels"]

        # Get fingerprints for this scaffold's compounds
        valid_indices = []
        valid_labels = []
        fp_indices = []
        for i, jcp in enumerate(jcp_ids):
            if jcp in fp_set:
                valid_indices.append(i)
                valid_labels.append(labels[i])
                fp_indices.append(fp_idx[jcp])

        if len(valid_indices) < 2:
            result["struct_mean_intra_similarity"] = None
            result["struct_mean_inter_similarity"] = None
            continue

        # Exact Tanimoto similarity over binary Morgan fingerprints, matching the
        # released PhenoSeeker code rather than using a fingerprint-cosine proxy.
        X_fp = fp_array[fp_indices].astype(np.int32)
        intersections = X_fp @ X_fp.T
        bit_counts = X_fp.sum(axis=1)
        unions = bit_counts[:, None] + bit_counts[None, :] - intersections
        sim_matrix = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions > 0,
        )

        valid_labels = np.asarray(valid_labels)
        unique_labels = np.unique(valid_labels)
        intra_sims = []
        for label in unique_labels:
            indices = np.flatnonzero(valid_labels == label)
            if len(indices) < 2:
                continue
            off_diagonal = ~np.eye(len(indices), dtype=bool)
            intra_sims.append(float(sim_matrix[np.ix_(indices, indices)][off_diagonal].mean()))
        inter_sims = []
        for position, label_i in enumerate(unique_labels):
            indices_i = np.flatnonzero(valid_labels == label_i)
            for label_j in unique_labels[position + 1 :]:
                indices_j = np.flatnonzero(valid_labels == label_j)
                inter_sims.append(float(sim_matrix[np.ix_(indices_i, indices_j)].mean()))

        result["struct_mean_intra_similarity"] = float(np.mean(intra_sims)) if intra_sims else None
        result["struct_mean_inter_similarity"] = float(np.mean(inter_sims)) if inter_sims else None

    # Count how many scaffolds have valid structural data
    n_valid = sum(1 for r in scaffold_results if r["struct_mean_intra_similarity"] is not None)
    logger.info(f"Structural similarities computed for {n_valid:,} scaffolds")


# =============================================================================
# Activity Cliff Detection
# =============================================================================


def detect_cliff_scaffolds(
    scaffold_results: list[dict],
    threshold: float = SEPARATION_THRESHOLD,
    min_intra_similarity: float = MIN_INTRA_SIMILARITY,
    scaffold_size_threshold: int = SCAFFOLD_SIZE_THRESHOLD,
    require_released_code_size_rule: bool = True,
) -> list[dict]:
    """Apply the two phenotype-selection conditions in the released code.

    A candidate must exceed the intra-minus-inter cosine threshold and have
    sufficiently positive absolute within-cluster cosine similarity.
    """
    cliff_scaffolds = [
        result
        for result in scaffold_results
        if result["separation_score"] > threshold
        and result["mean_intra_similarity"] > min_intra_similarity
        and (not require_released_code_size_rule or result["n_compounds"] > scaffold_size_threshold)
    ]

    size_operator = ">" if require_released_code_size_rule else ">="
    size_rule = f"N {size_operator} {scaffold_size_threshold}"
    logger.info(
        f"Candidate cliff scaffolds (separation > {threshold}, "
        f"intra cosine > {min_intra_similarity}, {size_rule}): {len(cliff_scaffolds):,}"
    )

    return cliff_scaffolds


def identify_cliff_pairs(
    scaffold_result: dict,
    top_n: int = 5,
) -> list[dict]:
    """For a cliff scaffold, identify the most extreme cliff pairs.

    A cliff pair: two compounds in different clusters with
    maximum morphological distance.
    """
    labels = scaffold_result["labels"]
    jcp_ids = scaffold_result["jcp_ids"]
    dist_matrix = scaffold_result["distance_matrix"]
    scaffold = scaffold_result["scaffold"]

    cliff_pairs = []
    n_compounds = len(labels)

    for i in range(n_compounds):
        for j in range(i + 1, n_compounds):
            if labels[i] != labels[j]:  # Different clusters
                cliff_pairs.append(
                    {
                        "scaffold": scaffold,
                        "jcp_i": jcp_ids[i],
                        "jcp_j": jcp_ids[j],
                        "morph_distance": float(dist_matrix[i, j]),
                        "cluster_i": int(labels[i]),
                        "cluster_j": int(labels[j]),
                    }
                )

    # Sort by distance and return top pairs
    cliff_pairs.sort(key=lambda x: -x["morph_distance"])
    return cliff_pairs[:top_n]


def collect_all_cliff_pairs(
    cliff_scaffolds: list[dict],
    top_per_scaffold: int = 3,
) -> pd.DataFrame:
    """Collect top cliff pairs from all cliff scaffolds."""
    all_pairs = []
    for result in cliff_scaffolds:
        pairs = identify_cliff_pairs(result, top_n=top_per_scaffold)
        all_pairs.extend(pairs)

    if not all_pairs:
        return pd.DataFrame()

    df = pd.DataFrame(all_pairs)
    df = df.sort_values("morph_distance", ascending=False).reset_index(drop=True)

    logger.info(f"Collected {len(df):,} cliff pairs from {len(cliff_scaffolds):,} scaffolds")

    return df


# =============================================================================
# Visualization
# =============================================================================


def plot_scaffold_size_distribution(
    scaffold_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot histogram of compounds per scaffold."""
    logger.info("Creating scaffold size distribution plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = scaffold_df["n_compounds"].values
    ax.hist(sizes, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Compounds per Scaffold")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(f"Scaffold Size Distribution (N >= {sizes.min()})")

    # Add statistics
    stats_text = f"n = {len(sizes):,}\nMean = {sizes.mean():.1f}\nMedian = {np.median(sizes):.0f}\nMax = {sizes.max()}"
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

    plt.tight_layout()
    fig.savefig(output_dir / "scaffold_size_distribution.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "scaffold_size_distribution.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'scaffold_size_distribution.png'}")
    plt.close(fig)


def plot_separation_score_distribution(
    scaffold_results: list[dict],
    released_code_candidates: list[dict],
    paper_rule_candidates: list[dict],
    scaffold_size_threshold: int,
    threshold: float,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot separation scores while reporting candidates from the complete rule."""
    logger.info("Creating separation score distribution plot...")

    scores = [r["separation_score"] for r in scaffold_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Separation threshold ({threshold}; one criterion)",
    )

    ax.set_xlabel("Separation Score (inter - intra cluster distance)")
    ax.set_ylabel("Count")
    ax.set_title("Separation Score Distribution Across Scaffolds")
    ax.legend()

    released_code_eligible = sum(result["n_compounds"] > scaffold_size_threshold for result in scaffold_results)
    stats_text = (
        f"Eligible: {released_code_eligible:,} (N > {scaffold_size_threshold}); "
        f"{len(scaffold_results):,} (N >= {scaffold_size_threshold})\n"
        "Candidates passing all criteria: "
        f"{len(released_code_candidates):,}; {len(paper_rule_candidates):,}"
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

    plt.tight_layout()
    fig.savefig(output_dir / "separation_score_distribution.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "separation_score_distribution.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'separation_score_distribution.png'}")
    plt.close(fig)


def plot_top_cliff_scaffolds(
    cliff_scaffolds: list[dict],
    top_n: int = 20,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot bar chart of scaffolds with highest separation scores."""
    logger.info("Creating top cliff scaffolds plot...")

    top_scaffolds = cliff_scaffolds[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Truncate scaffold SMILES for display
    labels = [r["scaffold"][:30] + "..." if len(r["scaffold"]) > 30 else r["scaffold"] for r in top_scaffolds]
    scores = [r["separation_score"] for r in top_scaffolds]
    n_compounds = [r["n_compounds"] for r in top_scaffolds]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color="#ff7f0e", edgecolor="black", linewidth=0.5)

    # Add compound counts as annotations
    for i, (bar, n) in enumerate(zip(bars, n_compounds)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={n}", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Separation Score")
    ax.set_title(f"Top {len(top_scaffolds)} Scaffold Series with Activity Cliffs")

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "top_cliff_scaffolds.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "top_cliff_scaffolds.pdf", bbox_inches="tight")
        logger.info(f"Saved: {output_dir / 'top_cliff_scaffolds.png'}")
    plt.close(fig)


def plot_example_dendrograms(
    cliff_scaffolds: list[dict],
    top_n: int = 4,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot dendrograms for top cliff scaffolds."""
    logger.info("Creating example dendrograms...")

    n_plots = min(top_n, len(cliff_scaffolds))
    if n_plots == 0:
        logger.warning("No cliff scaffolds to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, result in enumerate(cliff_scaffolds[:n_plots]):
        ax = axes[i]
        scaffold = result["scaffold"]
        Z = result["linkage_matrix"]
        jcp_ids = result["jcp_ids"]
        result["labels"]

        # Truncate JCP IDs for display
        short_labels = [jcp[-6:] for jcp in jcp_ids]

        # Color by cluster assignment
        dendrogram(
            Z,
            ax=ax,
            labels=short_labels,
            leaf_rotation=90,
            leaf_font_size=8,
        )

        # Truncate scaffold for title
        scaffold_short = scaffold[:40] + "..." if len(scaffold) > 40 else scaffold
        ax.set_title(f"{scaffold_short}\nSep={result['separation_score']:.3f}", fontsize=9)
        ax.set_xlabel("")

    # Hide unused subplots
    for i in range(n_plots, 4):
        axes[i].set_visible(False)

    plt.suptitle("Hierarchical Clustering Dendrograms for Top Cliff Scaffolds", fontsize=11)
    plt.tight_layout()

    if output_dir:
        fig.savefig(output_dir / "example_dendrograms.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "example_dendrograms.pdf", bbox_inches="tight")
        logger.info(f"Saved: {output_dir / 'example_dendrograms.png'}")
    plt.close(fig)


def plot_top_scaffold_structures(
    cliff_scaffolds: list[dict],
    top_n: int = 10,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot molecular structures of top cliff scaffolds using RDKit.

    Requires RDKit to be installed (available in cheminformatics environment).
    """
    if not HAS_RDKIT:
        logger.warning("RDKit not available - skipping scaffold structure plot")
        return

    logger.info(f"Creating top {top_n} scaffold structures plot...")

    n_scaffolds = min(top_n, len(cliff_scaffolds))
    if n_scaffolds == 0:
        logger.warning("No cliff scaffolds to plot")
        return

    # Parse scaffold SMILES and create molecules
    mols = []
    legends = []
    for i, result in enumerate(cliff_scaffolds[:n_scaffolds]):
        scaffold_smiles = result["scaffold"]
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is not None:
            mols.append(mol)
            legends.append(f"#{i + 1}: n={result['n_compounds']}, sep={result['separation_score']:.2f}")
        else:
            logger.warning(f"Could not parse scaffold SMILES: {scaffold_smiles[:50]}...")

    if not mols:
        logger.warning("No valid molecules to draw")
        return

    # Create grid image
    n_cols = min(5, len(mols))
    (len(mols) + n_cols - 1) // n_cols

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(300, 300),
        legends=legends,
        returnPNG=False,
    )

    # Save image
    if output_dir:
        img_path = output_dir / "top_scaffold_structures.png"
        img.save(str(img_path), dpi=(dpi, dpi))
        logger.info(f"Saved: {img_path}")


def plot_intra_inter_phenotypic(
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot intra vs inter-cluster phenotypic similarity.

    Cliff scaffolds (high separation) are highlighted in red.
    """
    logger.info("Creating intra vs inter phenotypic similarity plot...")

    # Get cliff scaffold names for highlighting
    cliff_scaffold_set = {r["scaffold"] for r in cliff_scaffolds}

    # Separate cliff and non-cliff scaffolds
    cliff_intra, cliff_inter = [], []
    noncliff_intra, noncliff_inter = [], []

    for r in scaffold_results:
        intra = r["mean_intra_similarity"]
        inter = r["mean_inter_similarity"]
        if r["scaffold"] in cliff_scaffold_set:
            cliff_intra.append(intra)
            cliff_inter.append(inter)
        else:
            noncliff_intra.append(intra)
            noncliff_inter.append(inter)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot non-cliff scaffolds in gray
    ax.scatter(noncliff_intra, noncliff_inter, c="gray", alpha=0.4, s=20, label="Other scaffolds")

    # Plot cliff scaffolds in red
    ax.scatter(cliff_intra, cliff_inter, c="red", alpha=0.7, s=30, label=f"Activity cliffs (n={len(cliff_scaffolds)})")

    # Add diagonal line (intra = inter)
    all_intra = cliff_intra + noncliff_intra
    all_inter = cliff_inter + noncliff_inter
    min_val = min(min(all_intra), min(all_inter)) - 0.05
    max_val = max(max(all_intra), max(all_inter)) + 0.05
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Intra = Inter")

    ax.set_xlabel("Intra-cluster Phenotypic Similarity", fontsize=11)
    ax.set_ylabel("Inter-cluster Phenotypic Similarity", fontsize=11)
    ax.set_title(
        f"Intra vs Inter-cluster Phenotypic Similarity\n(Activity cliff scaffolds in red, n={len(cliff_intra)})",
        fontsize=12,
    )
    ax.legend(loc="upper left")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "intra_inter_phenotypic.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "intra_inter_phenotypic.pdf", bbox_inches="tight")
        logger.info(f"Saved: {output_dir / 'intra_inter_phenotypic.png'}")
    plt.close(fig)


def plot_intra_inter_structural(
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot intra vs inter-cluster structural similarity.

    Cliff scaffolds (identified from phenotypic analysis) are highlighted in blue.
    """
    logger.info("Creating intra vs inter structural similarity plot...")

    # Get cliff scaffold names for highlighting
    cliff_scaffold_set = {r["scaffold"] for r in cliff_scaffolds}

    # Separate cliff and non-cliff scaffolds (only those with structural data)
    cliff_intra, cliff_inter = [], []
    noncliff_intra, noncliff_inter = [], []

    for r in scaffold_results:
        intra = r.get("struct_mean_intra_similarity")
        inter = r.get("struct_mean_inter_similarity")
        if intra is None or inter is None:
            continue
        if r["scaffold"] in cliff_scaffold_set:
            cliff_intra.append(intra)
            cliff_inter.append(inter)
        else:
            noncliff_intra.append(intra)
            noncliff_inter.append(inter)

    if not cliff_intra and not noncliff_intra:
        logger.warning("No structural similarity data available")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot non-cliff scaffolds in gray
    ax.scatter(noncliff_intra, noncliff_inter, c="gray", alpha=0.4, s=20, label="Other scaffolds")

    # Plot cliff scaffolds in blue
    ax.scatter(cliff_intra, cliff_inter, c="blue", alpha=0.7, s=30, label=f"Activity cliffs (n={len(cliff_intra)})")

    # Add diagonal line (intra = inter)
    lims = [0, 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Intra = Inter")

    ax.set_xlabel("Intra-cluster Structural Similarity (Tanimoto)", fontsize=11)
    ax.set_ylabel("Inter-cluster Structural Similarity (Tanimoto)", fontsize=11)
    n_cliff_total = len(cliff_scaffolds)
    n_cliff_with_struct = len(cliff_intra)
    title = f"Intra vs Inter-cluster Structural Similarity\n(Phenotypic activity cliff scaffolds in blue: {n_cliff_with_struct}/{n_cliff_total})"
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "intra_inter_structural.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / "intra_inter_structural.pdf", bbox_inches="tight")
        logger.info(f"Saved: {output_dir / 'intra_inter_structural.png'}")
    plt.close(fig)


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    scaffold_df: pd.DataFrame,
    scaffold_results: list[dict],
    cliff_scaffolds: list[dict],
    paper_rule_cliff_scaffolds: list[dict],
    cliff_pairs_df: pd.DataFrame,
    threshold: float,
    scaffold_size_threshold: int,
    n_clusters: int,
    profile_similarity_stats: dict,
    output_dir: Path,
) -> dict:
    """Generate the clustering-and-selection procedure-transfer summary."""
    scores = [r["separation_score"] for r in scaffold_results]
    n_cliffs = len(cliff_scaffolds)
    released_code_results = [result for result in scaffold_results if result["n_compounds"] > scaffold_size_threshold]
    matches_released_configuration = (
        scaffold_size_threshold == SCAFFOLD_SIZE_THRESHOLD
        and n_clusters == N_CLUSTERS
        and threshold == SEPARATION_THRESHOLD
    )

    summary = {
        "task": "PhenoSeeker clustering-and-selection procedure transfer",
        "reference": "Sanchez et al. 2026 (PhenoSeeker)",
        "research_question": (
            "Apply the released PhenoSeeker scaffold-series clustering and selection "
            "procedure to JUMP CellProfiler profiles"
        ),
        "method": (
            "Released PhenoSeeker Figure 4 clustering and selection procedure, using "
            "JUMP scaffold memberships, Morgan fingerprints, and CellProfiler profiles"
        ),
        "parameters": {
            "matches_released_configuration": matches_released_configuration,
            "min_compounds_per_scaffold": scaffold_size_threshold + 1,
            "paper_text_min_compounds_per_scaffold": scaffold_size_threshold,
            "configured_primary_scaffold_size_rule": f"N > {scaffold_size_threshold}",
            "released_code_scaffold_size_rule": "N > 6",
            "n_clusters": n_clusters,
            "clustering_distance": "euclidean",
            "clustering_linkage": "ward",
            "min_compounds_per_cluster": MIN_CLUSTER_SIZE,
            "separation_threshold": threshold,
            "min_intra_similarity": MIN_INTRA_SIMILARITY,
            "selection_rule": (
                "mean_intra_cosine - mean_inter_cosine "
                f"> {threshold:g} and mean_intra_cosine > {MIN_INTRA_SIMILARITY:g}"
            ),
            "phenotypic_activity_used_for_selection": False,
        },
        "scaffold_statistics": {
            "n_scaffolds_meeting_size_rule_before_profile_alignment": len(scaffold_df),
            "n_scaffolds_analyzed": len(released_code_results),
            "n_scaffolds_paper_rule_sensitivity": len(scaffold_results),
            "total_compounds": int(sum(r["n_compounds"] for r in released_code_results)),
            "total_compounds_paper_rule_sensitivity": int(sum(r["n_compounds"] for r in scaffold_results)),
            "mean_compounds_per_scaffold": float(np.mean([r["n_compounds"] for r in released_code_results])),
            "max_compounds_per_scaffold": int(max(r["n_compounds"] for r in released_code_results)),
        },
        "separation_scores": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "max": float(np.max(scores)),
            "min": float(np.min(scores)),
        },
        "activity_cliffs": {
            "n_cliff_scaffolds": n_cliffs,
            "pct_cliff_scaffolds": float(n_cliffs / len(released_code_results) * 100),
            "n_cliff_scaffolds_paper_rule_sensitivity": len(paper_rule_cliff_scaffolds),
            "pct_cliff_scaffolds_paper_rule_sensitivity": float(
                len(paper_rule_cliff_scaffolds) / len(scaffold_results) * 100
            ),
            "n_example_cliff_pairs": len(cliff_pairs_df),
            "phenotypic_activity_used_for_selection": False,
            "comparison_to_phenoseeker": {
                "role": "matched_clustering_and_selection_different_upstream_substrate",
                "paper_url": "https://www.nature.com/articles/s42003-025-09500-y",
                "release_commit": PHENOSEEKER_RELEASE_COMMIT,
                "released_notebook_url": (
                    "https://github.com/mxfly14/2025_sanchez_phenoseeker/blob/"
                    f"{PHENOSEEKER_RELEASE_COMMIT}/notebooks/fig_4_sup_11_12.ipynb"
                ),
                "released_table_url": (
                    "https://github.com/mxfly14/2025_sanchez_phenoseeker/blob/"
                    f"{PHENOSEEKER_RELEASE_COMMIT}/data/fig_data/fig_4_data.csv"
                ),
                "released_notebook_sha256": PHENOSEEKER_NOTEBOOK_SHA256,
                "released_table_sha256": PHENOSEEKER_TABLE_SHA256,
                "reference_scaffolds_analyzed_reported": 532,
                "reference_release_table_rows": 533,
                "reference_cliff_scaffolds": 81,
                "reference_min_compounds_per_cluster": 3,
                "reference_separation_threshold": 0.2,
                "reference_min_intra_similarity": 0.2,
                "reference_profile_representation": "DINOv2-Giant",
                "reference_scaffold_source": "Bemis-Murcko scaffolds recomputed from InChI",
                "transfer_scaffold_source": "JUMP Metadata_MurckoScaffold annotation",
                "transfer_fingerprint_source": "JUMP Morgan ECFP4 fingerprints",
            },
        },
        "profile_similarity_distribution": profile_similarity_stats,
        "top_cliff_scaffolds": [
            {
                "scaffold": r["scaffold"],
                "n_compounds": r["n_compounds"],
                "separation_score": r["separation_score"],
            }
            for r in cliff_scaffolds[:10]
        ],
        "interpretation": (
            "Applying the released PhenoSeeker clustering and selection procedure to "
            "source-7-excluded perturbation-level CellProfiler profiles and JUMP scaffold "
            f"memberships flagged {n_cliffs} of {len(released_code_results)} eligible series. "
            f"Following the paper's N >= {scaffold_size_threshold} prose instead gives "
            f"{len(paper_rule_cliff_scaffolds)} of {len(scaffold_results)}. "
            "PhenoSeeker reported 81 of 532 using DINOv2-Giant profiles (its released table "
            "contains 533 rows). Profile representation, scaffold derivation, and eligible "
            "membership differ, so this is a procedure transfer rather than a direct prevalence "
            "benchmark. Phenotypic activity was not used for selection; the flagged series are "
            "morphology-cluster divergence candidates, not validated activity cliffs."
        ),
    }

    # Log summary
    logger.info("\n=== SUMMARY ===")
    logger.info(
        f"Scaffolds analyzed (configured primary N > {scaffold_size_threshold}): {len(released_code_results):,}"
    )
    logger.info(f"Candidate cliff scaffolds: {n_cliffs:,} ({n_cliffs / len(released_code_results) * 100:.1f}%)")
    logger.info(
        f"Paper-rule sensitivity (N >= {scaffold_size_threshold}): "
        f"{len(paper_rule_cliff_scaffolds):,}/{len(scaffold_results):,}"
    )
    logger.info(f"Example cliff pairs retained: {len(cliff_pairs_df):,}")
    logger.info(f"Mean separation score: {np.mean(scores):.4f}")
    logger.info(f"Interpretation: {summary['interpretation']}")

    # Save JSON
    description = (
        "Transfer of the released PhenoSeeker Figure 4 clustering and selection procedure "
        "to JUMP scaffold memberships and CellProfiler profiles. Reports every eligible "
        "series and every candidate passing the released-code cosine rules."
    )
    write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=summary,
        files=[
            output_dir / "scaffold_analysis.parquet",
            output_dir / "cliff_scaffolds.csv",
        ],
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
    min_compounds: int = typer.Option(
        MIN_COMPOUNDS,
        "--min-compounds",
        "-n",
        help="Base size threshold: primary uses N>value; paper-text sensitivity uses N>=value",
    ),
    n_clusters: int = typer.Option(
        N_CLUSTERS,
        "--n-clusters",
        help="Number of clusters for hierarchical clustering",
    ),
    separation_threshold: float = typer.Option(
        SEPARATION_THRESHOLD,
        "--separation-threshold",
        help="Threshold for cliff detection (inter - intra distance)",
    ),
    top_scaffolds: int = typer.Option(
        100,
        "--top-scaffolds",
        help="Number of top cliff scaffolds to report",
    ),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
) -> None:
    """Transfer PhenoSeeker clustering and selection to JUMP scaffold series."""

    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"Min compounds per scaffold: {min_compounds}")
    logger.info(f"N clusters: {n_clusters}")
    logger.info(f"Separation threshold: {separation_threshold}")

    # Check database exists
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Load scaffold data
        logger.info("Loading scaffold data...")
        scaffold_df = load_scaffolds_with_counts(metadata_con, min_compounds)

        # 2. Load morphology profiles
        logger.info("Loading morphology profiles...")
        adata = load_profiles(dataset, level="perturbation")
        profile_similarity_stats = global_pairwise_cosine_moments(adata.X)
        logger.info(
            "Global profile cosine: "
            f"mean={profile_similarity_stats['mean_pairwise_cosine']:.4f}, "
            f"SD={profile_similarity_stats['sd_pairwise_cosine']:.4f}, "
            f"mean+3SD={profile_similarity_stats['mean_plus_3sd']:.4f}"
        )

        # 3. Load fingerprints for structural similarity
        logger.info("Loading fingerprints...")
        fp_array, fp_jcp = load_fingerprints()

        # 4. Align morphology to scaffolds
        scaffold_data = align_morphology_to_scaffolds(adata, scaffold_df, min_compounds)

        # 5. Analyze all scaffolds
        scaffold_results = analyze_all_scaffolds(scaffold_data, n_clusters)

        # 6. Add structural similarities
        add_structural_similarities(scaffold_results, fp_array, fp_jcp)

        # 7. Detect cliff scaffolds
        cliff_scaffolds = detect_cliff_scaffolds(
            scaffold_results,
            separation_threshold,
            scaffold_size_threshold=min_compounds,
        )
        paper_rule_cliff_scaffolds = detect_cliff_scaffolds(
            scaffold_results,
            separation_threshold,
            scaffold_size_threshold=min_compounds,
            require_released_code_size_rule=False,
        )

        # 8. Collect cliff pairs
        cliff_pairs_df = collect_all_cliff_pairs(cliff_scaffolds[:top_scaffolds])

        # 9. Create visualizations
        if not skip_plots:
            plot_scaffold_size_distribution(scaffold_df, output_dir, dpi)
            plot_separation_score_distribution(
                scaffold_results,
                cliff_scaffolds,
                paper_rule_cliff_scaffolds,
                min_compounds,
                separation_threshold,
                output_dir,
                dpi,
            )
            if cliff_scaffolds:
                plot_top_cliff_scaffolds(cliff_scaffolds, top_n=20, output_dir=output_dir, dpi=dpi)
                plot_example_dendrograms(cliff_scaffolds, top_n=4, output_dir=output_dir, dpi=dpi)
                plot_top_scaffold_structures(cliff_scaffolds, top_n=10, output_dir=output_dir, dpi=dpi)
            # Intra vs inter similarity plots
            plot_intra_inter_phenotypic(scaffold_results, cliff_scaffolds, output_dir, dpi)
            plot_intra_inter_structural(scaffold_results, cliff_scaffolds, output_dir, dpi)

        # 10. Save results
        primary_scaffolds = {result["scaffold"] for result in cliff_scaffolds}
        paper_rule_scaffolds = {result["scaffold"] for result in paper_rule_cliff_scaffolds}

        # Complete eligible-series analysis. Member IDs and canonical cluster labels
        # make both the released-code result and paper-text sensitivity auditable.
        scaffold_analysis_df = pd.DataFrame(
            [
                {
                    "scaffold": r["scaffold"],
                    "n_compounds": r["n_compounds"],
                    "mean_intra_similarity": r["mean_intra_similarity"],
                    "mean_inter_similarity": r["mean_inter_similarity"],
                    "separation_score": r["separation_score"],
                    "struct_mean_intra_similarity": r.get("struct_mean_intra_similarity"),
                    "struct_mean_inter_similarity": r.get("struct_mean_inter_similarity"),
                    "cluster_sizes": "|".join(str(value) for value in sorted(r["cluster_sizes"].values())),
                    "min_cluster_size": min(r["cluster_sizes"].values()),
                    "is_released_code_candidate": r["scaffold"] in primary_scaffolds,
                    "is_paper_rule_candidate": r["scaffold"] in paper_rule_scaffolds,
                    "jcp_ids": "|".join(sorted(r["jcp_ids"])),
                    "cluster_labels": "|".join(
                        str(label) for _, label in sorted(zip(r["jcp_ids"], r["labels"], strict=True))
                    ),
                }
                for r in scaffold_results
            ]
        ).sort_values(
            ["separation_score", "scaffold"],
            ascending=[False, True],
            kind="mergesort",
            ignore_index=True,
        )
        scaffold_analysis_df.to_parquet(output_dir / "scaffold_analysis.parquet", index=False)
        logger.info(f"Saved: {output_dir / 'scaffold_analysis.parquet'}")

        # Complete, deterministic cliff-series table. `top_scaffolds` limits only the
        # example pair table and plots; it must never truncate this analysis artifact.
        cliff_scaffolds_df = scaffold_analysis_df.loc[scaffold_analysis_df["is_released_code_candidate"]].reset_index(
            drop=True
        )
        cliff_scaffolds_df.to_csv(output_dir / "cliff_scaffolds.csv", index=False)
        logger.info(f"Saved: {output_dir / 'cliff_scaffolds.csv'}")

        # Cliff pairs CSV
        if len(cliff_pairs_df) > 0:
            cliff_pairs_df.to_csv(output_dir / "cliff_pairs.csv", index=False)
            logger.info(f"Saved: {output_dir / 'cliff_pairs.csv'}")

        # 11. Generate summary
        generate_summary(
            scaffold_df,
            scaffold_results,
            cliff_scaffolds,
            paper_rule_cliff_scaffolds,
            cliff_pairs_df,
            separation_threshold,
            min_compounds,
            n_clusters,
            profile_similarity_stats,
            output_dir,
        )

        logger.success(f"PhenoSeeker method-transfer screen complete: {output_dir}")

    finally:
        metadata_con.close()


if __name__ == "__main__":
    app()
