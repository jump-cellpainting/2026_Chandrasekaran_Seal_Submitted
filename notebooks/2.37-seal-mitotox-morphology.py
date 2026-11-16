#!/usr/bin/env python3
"""MitoTox morphological mechanism analysis.

Clusters mitotoxicants by morphology and tests whether functional submechanisms
(F01-F08) separate in phenotypic space.

Research question:
    Can morphology distinguish mitotoxicity submechanisms, not just toxic vs non-toxic?

Success criteria:
    - Positive separation scores (between-mechanism > within-mechanism distance)
    - Significant p-values from permutation tests
    - Visual clustering in UMAP by mechanism

Usage:
    pixi run -e cheminformatics python notebooks/2.37-seal-mitotox-morphology.py
    pixi run -e cheminformatics python notebooks/2.37-seal-mitotox-morphology.py --sample 0.5
    pixi run -e cheminformatics python notebooks/2.37-seal-mitotox-morphology.py --dataset compound_no_source7
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score

from jump_production.config import (
    DEFAULT_DPI,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from jump_production.io import load_profiles
from jump_production.summaries import index_files, write_summary

# ============================================================================
# Configuration
# ============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
MORPH_UMAP_PATH = INTERIM_DATA_DIR / "anndata" / "compound_no_source7_perturbation_cosine_all_umap.h5ad"
CHEM_UMAP_PATH = PROCESSED_DATA_DIR / "chemical-space" / "all" / "structure_umap.h5ad"

# 8 top-level MitoTox functional mechanisms
MECHANISMS = [
    "Transmembrane potential",
    "Function of mitochondria",
    "Organization of mitochondria",
    "Movement of mitochondria",
    "Oxidative stress",
    "Mitochondrial DNA",
    "Cell death",
    "Signaling",
]

# Short names for plotting
MECHANISM_SHORT = {
    "Transmembrane potential": "F01: Membrane potential",
    "Function of mitochondria": "F02: Mito function",
    "Organization of mitochondria": "F03: Mito organization",
    "Movement of mitochondria": "F04: Mito movement",
    "Oxidative stress": "F05: Oxidative stress",
    "Mitochondrial DNA": "F06: mtDNA",
    "Cell death": "F07: Cell death",
    "Signaling": "F08: Signaling",
}

app = typer.Typer()


# ============================================================================
# Data Loading
# ============================================================================


def load_mitotox_matched(sample_frac: float = 1.0) -> pd.DataFrame:
    """Load MitoTox data from augmented database (already matched to JUMP).

    Args:
        sample_frac: Fraction of data to sample (for testing)

    Returns:
        DataFrame with MitoTox annotations matched to JUMP JCP2022 IDs
    """
    logger.info("Loading MitoTox data from augmented database...")
    con = duckdb.connect(str(METADATA_DB), read_only=True)
    matched_df = con.execute("""
        SELECT
            Metadata_JCP2022,
            Metadata_mitotox_label AS mitotox_label,
            Metadata_mitotox_toxic,
            Metadata_mitotox_mechanisms AS functional_mechanism
        FROM mitotox_annotations
    """).df()
    con.close()
    logger.info(f"Loaded {len(matched_df)} MitoTox compounds from database")

    # Sample if requested
    if sample_frac < 1.0:
        matched_df = matched_df.sample(frac=sample_frac, random_state=42)
        logger.info(f"Sampled to {len(matched_df)} compounds")

    # Parse mechanisms into binary columns
    for mech in MECHANISMS:
        matched_df[f"has_{mech}"] = matched_df["functional_mechanism"].fillna("").str.contains(mech, regex=False)

    # Add toxic flag
    matched_df["is_toxic"] = matched_df["Metadata_mitotox_toxic"] == 1

    return matched_df


# ============================================================================
# UMAP Visualization
# ============================================================================


def compute_umap_limits(coords: np.ndarray, iqr_multiplier: float = 1.5) -> tuple[float, float, float, float]:
    """Compute axis limits using IQR-based outlier clipping.

    Args:
        coords: UMAP coordinates (N, 2)
        iqr_multiplier: IQR multiplier for outlier threshold (default 1.5)

    Returns:
        (xlim_min, xlim_max, ylim_min, ylim_max)
    """
    x, y = coords[:, 0], coords[:, 1]

    # X axis limits
    q1_x, q3_x = np.percentile(x, [25, 75])
    iqr_x = q3_x - q1_x
    xlim = (q1_x - iqr_multiplier * iqr_x, q3_x + iqr_multiplier * iqr_x)

    # Y axis limits
    q1_y, q3_y = np.percentile(y, [25, 75])
    iqr_y = q3_y - q1_y
    ylim = (q1_y - iqr_multiplier * iqr_y, q3_y + iqr_multiplier * iqr_y)

    return xlim[0], xlim[1], ylim[0], ylim[1]


def plot_umap_full_with_mitotox(
    full_coords: np.ndarray,
    mitotox_coords: np.ndarray,
    mitotox_obs: pd.DataFrame,
    title_prefix: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot full UMAP with all compounds as background, MitoTox colored by mechanism.

    Creates a 3x3 grid:
    - First panel: Non-toxic
    - Panels 2-9: Each of the 8 mechanisms

    Args:
        full_coords: UMAP coordinates for ALL compounds (N, 2)
        mitotox_coords: UMAP coordinates for MitoTox subset (M, 2)
        mitotox_obs: DataFrame with mechanism columns for MitoTox subset
        title_prefix: Prefix for plot title
        output_path: Output file path (without extension)
        dpi: DPI for saving
    """
    # Compute axis limits from full data using IQR clipping
    xlim_min, xlim_max, ylim_min, ylim_max = compute_umap_limits(full_coords)

    # Categories to plot: non-toxic + 8 mechanisms
    categories = ["Non-toxic"] + MECHANISMS

    # Colors for each category
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"{title_prefix} - Full UMAP with MitoTox Compounds", fontsize=14, y=1.02)

    for ax, cat, color in zip(axes.flat, categories, colors):
        # Plot ALL compounds as gray background
        ax.scatter(
            full_coords[:, 0],
            full_coords[:, 1],
            c="lightgray",
            s=1,
            alpha=0.3,
            rasterized=True,
        )

        # Determine which MitoTox compounds to highlight
        if cat == "Non-toxic":
            mask = ~mitotox_obs["is_toxic"].values
            title = f"Non-toxic (n={mask.sum()})"
        else:
            mask = mitotox_obs[f"has_{cat}"].values
            title = f"{MECHANISM_SHORT[cat]} (n={mask.sum()})"

        # Plot highlighted MitoTox compounds
        if mask.sum() > 0:
            ax.scatter(
                mitotox_coords[mask, 0],
                mitotox_coords[mask, 1],
                c=[color],
                s=20,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.3,
                rasterized=True,
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def plot_umap_by_mechanisms(
    coords: np.ndarray,
    obs_df: pd.DataFrame,
    title_prefix: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot UMAP colored by each mechanism in a 2x4 grid.

    Args:
        coords: UMAP coordinates (N, 2)
        obs_df: DataFrame with mechanism columns
        title_prefix: Prefix for plot title
        output_path: Output file path (without extension)
        dpi: DPI for saving
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title_prefix} - MitoTox Compounds by Mechanism", fontsize=14, y=1.02)

    for ax, mech in zip(axes.flat, MECHANISMS):
        has_mech = obs_df[f"has_{mech}"].values

        # Plot non-mechanism compounds first (gray)
        ax.scatter(
            coords[~has_mech, 0],
            coords[~has_mech, 1],
            c="lightgray",
            s=10,
            alpha=0.5,
            label="Other",
            rasterized=True,
        )

        # Plot mechanism compounds on top (colored)
        ax.scatter(
            coords[has_mech, 0],
            coords[has_mech, 1],
            c="tab:red",
            s=15,
            alpha=0.8,
            label=f"n={has_mech.sum()}",
            rasterized=True,
        )

        ax.set_title(MECHANISM_SHORT[mech], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def plot_umap_toxic_vs_nontoxic(
    coords: np.ndarray,
    obs_df: pd.DataFrame,
    title_prefix: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot UMAP colored by toxic vs non-toxic."""
    fig, ax = plt.subplots(figsize=(8, 8))

    is_toxic = obs_df["is_toxic"].values

    # Non-toxic (blue)
    ax.scatter(
        coords[~is_toxic, 0],
        coords[~is_toxic, 1],
        c="tab:blue",
        s=15,
        alpha=0.7,
        label=f"Non-toxic (n={(~is_toxic).sum()})",
        rasterized=True,
    )

    # Toxic (red)
    ax.scatter(
        coords[is_toxic, 0],
        coords[is_toxic, 1],
        c="tab:red",
        s=15,
        alpha=0.7,
        label=f"Toxic (n={is_toxic.sum()})",
        rasterized=True,
    )

    ax.set_title(f"{title_prefix} - Toxic vs Non-toxic")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


# ============================================================================
# Hierarchical Clustering
# ============================================================================


def plot_dendrogram(
    dist_matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
    dpi: int = 150,
) -> np.ndarray:
    """Plot hierarchical clustering dendrogram.

    Returns:
        Linkage matrix Z
    """
    # Convert distance matrix to condensed form for linkage
    distances = squareform(dist_matrix, checks=False)

    # Hierarchical clustering (average linkage)
    Z = linkage(distances, method="average")

    # Create dendrogram
    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering of MitoTox Compounds (Morphology)")
    ax.set_ylabel("Cosine Distance")

    plt.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    logger.info(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)

    return Z


# ============================================================================
# Statistical Tests
# ============================================================================


def compute_mechanism_separation(
    dist_matrix: np.ndarray,
    obs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute within vs between mechanism distances.

    For each mechanism, tests whether compounds with that mechanism
    are closer to each other than to compounds without it.

    Returns:
        DataFrame with separation statistics per mechanism
    """
    results = []

    for mech in MECHANISMS:
        mech_mask = obs_df[f"has_{mech}"].values
        n_mech = mech_mask.sum()

        if n_mech < 5:
            logger.warning(f"Skipping {mech}: only {n_mech} compounds")
            continue

        # Within-mechanism distances (upper triangle only)
        within_idx = np.where(mech_mask)[0]
        within_dists = []
        for i, idx_i in enumerate(within_idx):
            for idx_j in within_idx[i + 1 :]:
                within_dists.append(dist_matrix[idx_i, idx_j])
        within_dists = np.array(within_dists)

        # Between-mechanism distances
        between_idx = np.where(~mech_mask)[0]
        between_dists = dist_matrix[np.ix_(within_idx, between_idx)].flatten()

        # Mann-Whitney U test: within < between means separation
        if len(within_dists) > 0 and len(between_dists) > 0:
            stat, pval = mannwhitneyu(within_dists, between_dists, alternative="less")
        else:
            stat, pval = np.nan, np.nan

        results.append(
            {
                "mechanism": mech,
                "n_compounds": n_mech,
                "mean_within_dist": float(within_dists.mean()) if len(within_dists) > 0 else np.nan,
                "std_within_dist": float(within_dists.std()) if len(within_dists) > 0 else np.nan,
                "mean_between_dist": float(between_dists.mean()),
                "std_between_dist": float(between_dists.std()),
                "separation": float(between_dists.mean() - within_dists.mean()) if len(within_dists) > 0 else np.nan,
                "mann_whitney_stat": float(stat),
                "mann_whitney_p": float(pval),
            }
        )

    return pd.DataFrame(results)


def permutation_test_silhouette(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
) -> tuple[float, float, np.ndarray]:
    """Permutation test for silhouette score significance.

    Returns:
        (observed_score, p_value, null_distribution)
    """
    # Filter to compounds with valid labels (not -1 or nan)
    valid_mask = labels >= 0
    if valid_mask.sum() < 10:
        return np.nan, np.nan, np.array([])

    dist_sub = dist_matrix[np.ix_(valid_mask, valid_mask)]
    labels_sub = labels[valid_mask]

    # Check we have at least 2 clusters
    unique_labels = np.unique(labels_sub)
    if len(unique_labels) < 2:
        return np.nan, np.nan, np.array([])

    observed = silhouette_score(dist_sub, labels_sub, metric="precomputed")

    rng = np.random.default_rng(42)
    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels_sub)
        null_scores.append(silhouette_score(dist_sub, shuffled, metric="precomputed"))

    null_scores = np.array(null_scores)
    pval = (np.sum(null_scores >= observed) + 1) / (n_permutations + 1)

    return observed, pval, null_scores


# ============================================================================
# Main
# ============================================================================


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name",
    ),
    sample: float = typer.Option(
        1.0,
        "--sample",
        help="Fraction of MitoTox compounds to sample (for testing)",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for figures",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory",
    ),
    n_permutations: int = typer.Option(
        1000,
        "--n-permutations",
        help="Number of permutations for significance testing",
    ),
) -> None:
    """Analyze MitoTox compounds by morphological mechanism."""
    # Setup output directory
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "mitotox-analysis" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # 1. Load and match MitoTox data
    # ========================================================================
    matched_df = load_mitotox_matched(sample_frac=sample)
    matched_jcp = set(matched_df["Metadata_JCP2022"])

    # Log mechanism counts
    logger.info("Mechanism counts:")
    for mech in MECHANISMS:
        count = matched_df[f"has_{mech}"].sum()
        logger.info(f"  {MECHANISM_SHORT[mech]}: {count}")

    toxic_count = matched_df["is_toxic"].sum()
    logger.info(f"Toxic: {toxic_count}, Non-toxic: {len(matched_df) - toxic_count}")

    # ========================================================================
    # 2. Load precomputed UMAPs (keep full for background, filter for MitoTox)
    # ========================================================================
    logger.info("Loading precomputed morphology UMAP...")
    adata_morph_full = sc.read_h5ad(MORPH_UMAP_PATH)
    logger.info(f"Full morphology UMAP: {adata_morph_full.n_obs} compounds")

    logger.info("Loading precomputed chemistry UMAP...")
    adata_chem_full = sc.read_h5ad(CHEM_UMAP_PATH)
    if "Metadata_JCP2022" in adata_chem_full.obs.columns:
        adata_chem_full.obs["JCP2022"] = adata_chem_full.obs["Metadata_JCP2022"]
    logger.info(f"Full chemistry UMAP: {adata_chem_full.n_obs} compounds")

    # Find intersection of compounds in both UMAPs for fair comparison
    morph_jcp = set(adata_morph_full.obs["JCP2022"])
    chem_jcp = set(adata_chem_full.obs["JCP2022"])
    common_jcp = morph_jcp & chem_jcp
    logger.info(f"Compounds in both UMAPs: {len(common_jcp)}")

    # Filter both full UMAPs to common compounds
    adata_morph_full = adata_morph_full[adata_morph_full.obs["JCP2022"].isin(common_jcp)].copy()
    adata_chem_full = adata_chem_full[adata_chem_full.obs["JCP2022"].isin(common_jcp)].copy()
    logger.info(f"Filtered to common: morph={adata_morph_full.n_obs}, chem={adata_chem_full.n_obs}")

    # Filter MitoTox to compounds in both UMAPs
    matched_jcp_common = matched_jcp & common_jcp

    # Filter to MitoTox compounds
    morph_mask = adata_morph_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_morph = adata_morph_full[morph_mask].copy()
    logger.info(f"MitoTox in morphology UMAP: {adata_morph.n_obs} compounds")

    chem_mask = adata_chem_full.obs["JCP2022"].isin(matched_jcp_common)
    adata_chem = adata_chem_full[chem_mask].copy()
    logger.info(f"MitoTox in chemistry UMAP: {adata_chem.n_obs} compounds")

    # Join mechanism annotations to adata obs
    # Use drop_duplicates to handle multiple MitoTox entries mapping to same JUMP compound
    mech_cols = [f"has_{m}" for m in MECHANISMS] + ["is_toxic", "mitotox_label", "functional_mechanism"]
    matched_dedup = matched_df.drop_duplicates(subset=["Metadata_JCP2022"])[["Metadata_JCP2022"] + mech_cols]
    matched_dedup = matched_dedup.rename(columns={"Metadata_JCP2022": "JCP2022"})

    # Morphology obs - merge on JCP2022
    morph_obs = adata_morph.obs.copy().reset_index()
    morph_obs = morph_obs.merge(
        matched_dedup,
        on="JCP2022",
        how="left",
    )
    morph_obs = morph_obs.set_index(adata_morph.obs.index.name or "index")
    adata_morph.obs = morph_obs

    # Chemistry obs - merge on JCP2022
    chem_obs = adata_chem.obs.copy().reset_index()
    chem_obs = chem_obs.merge(
        matched_dedup,
        on="JCP2022",
        how="left",
    )
    chem_obs = chem_obs.set_index(adata_chem.obs.index.name or "index")
    adata_chem.obs = chem_obs

    # ========================================================================
    # 3. Plot UMAPs (full background + MitoTox overlay)
    # ========================================================================
    logger.info("Plotting morphology UMAPs with full background...")
    plot_umap_full_with_mitotox(
        adata_morph_full.obsm["X_umap"],
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology",
        output_dir / "umap_morphology_full",
        dpi=dpi,
    )

    logger.info("Plotting chemistry UMAPs with full background...")
    plot_umap_full_with_mitotox(
        adata_chem_full.obsm["X_umap"],
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry",
        output_dir / "umap_chemistry_full",
        dpi=dpi,
    )

    # Also keep the original MitoTox-only plots for comparison
    logger.info("Plotting MitoTox-only morphology UMAPs...")
    plot_umap_by_mechanisms(
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology UMAP",
        output_dir / "umap_morphology_by_mechanism",
        dpi=dpi,
    )
    plot_umap_toxic_vs_nontoxic(
        adata_morph.obsm["X_umap"],
        adata_morph.obs,
        "Morphology UMAP",
        output_dir / "umap_morphology_toxic",
        dpi=dpi,
    )

    logger.info("Plotting MitoTox-only chemistry UMAPs...")
    plot_umap_by_mechanisms(
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry UMAP",
        output_dir / "umap_chemistry_by_mechanism",
        dpi=dpi,
    )
    plot_umap_toxic_vs_nontoxic(
        adata_chem.obsm["X_umap"],
        adata_chem.obs,
        "Chemistry UMAP",
        output_dir / "umap_chemistry_toxic",
        dpi=dpi,
    )

    # ========================================================================
    # 4. Load full profiles for clustering and statistical tests
    # ========================================================================
    logger.info("Loading full morphology profiles for clustering...")
    adata_full = load_profiles(dataset, level="perturbation")
    full_mask = adata_full.obs["JCP2022"].isin(matched_jcp)
    adata_full = adata_full[full_mask].copy()
    logger.info(f"Full profiles: {adata_full.n_obs} compounds, {adata_full.n_vars} features")

    # Join mechanism annotations
    full_obs = adata_full.obs.copy().reset_index()
    full_obs = full_obs.merge(
        matched_dedup,
        on="JCP2022",
        how="left",
    )
    full_obs = full_obs.set_index(adata_full.obs.index.name or "index")
    adata_full.obs = full_obs

    # L2 normalize for cosine distance
    X = adata_full.X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.maximum(norms, 1e-10)

    # Compute pairwise cosine distances
    logger.info("Computing pairwise distances...")
    distances = pdist(X_norm, metric="cosine")
    dist_matrix = squareform(distances)

    # ========================================================================
    # 5. Hierarchical clustering dendrogram
    # ========================================================================
    logger.info("Plotting hierarchical clustering dendrogram...")

    # Create labels for dendrogram (primary mechanism or "multiple"/"none")
    def get_primary_mechanism(row):
        mechs = [m for m in MECHANISMS if row.get(f"has_{m}", False)]
        if len(mechs) == 0:
            return "None"
        elif len(mechs) == 1:
            return MECHANISM_SHORT[mechs[0]][:10]
        else:
            return "Multiple"

    dendro_labels = [get_primary_mechanism(row) for _, row in adata_full.obs.iterrows()]
    plot_dendrogram(dist_matrix, dendro_labels, output_dir / "dendrogram_morphology", dpi=dpi)

    # ========================================================================
    # 6. Statistical tests for mechanism separation
    # ========================================================================
    logger.info("Computing mechanism separation statistics...")
    separation_df = compute_mechanism_separation(dist_matrix, adata_full.obs)
    separation_df.to_csv(output_dir / "mechanism_separation.csv", index=False)
    logger.info(f"Saved: {output_dir / 'mechanism_separation.csv'}")

    # Log results
    logger.info("Mechanism separation results:")
    for _, row in separation_df.iterrows():
        sig = "*" if row["mann_whitney_p"] < 0.05 else ""
        logger.info(f"  {row['mechanism']}: separation={row['separation']:.4f}, p={row['mann_whitney_p']:.2e}{sig}")

    # Overall silhouette score using primary mechanism as label
    logger.info("Computing silhouette score with permutation test...")

    # Assign numeric labels for primary mechanism
    def get_mech_label(row):
        for i, m in enumerate(MECHANISMS):
            if row.get(f"has_{m}", False):
                return i
        return -1  # No mechanism

    mech_labels = np.array([get_mech_label(row) for _, row in adata_full.obs.iterrows()])

    silhouette_obs, silhouette_pval, null_dist = permutation_test_silhouette(
        dist_matrix, mech_labels, n_permutations=n_permutations
    )
    logger.info(f"Silhouette score: {silhouette_obs:.4f} (p={silhouette_pval:.4f})")

    # ========================================================================
    # 7. Summary output
    # ========================================================================
    numbers = {
        "task": "MitoTox morphological mechanism analysis",
        "research_question": "Can morphology distinguish mitotoxicity submechanisms?",
        "dataset": dataset,
        "n_matched_to_jump": len(matched_df),
        "n_in_umap_morph": int(adata_morph.n_obs),
        "n_in_umap_chem": int(adata_chem.n_obs),
        "n_toxic": int(matched_df["is_toxic"].sum()),
        "n_nontoxic": int((~matched_df["is_toxic"]).sum()),
        "mechanism_counts": {MECHANISM_SHORT[m]: int(matched_df[f"has_{m}"].sum()) for m in MECHANISMS},
        "silhouette_score": float(silhouette_obs) if not np.isnan(silhouette_obs) else None,
        "silhouette_pvalue": float(silhouette_pval) if not np.isnan(silhouette_pval) else None,
        "mechanism_separation": separation_df.to_dict(orient="records"),
        "n_mechanisms_significant": int((separation_df["mann_whitney_p"] < 0.05).sum()),
        "interpretation": (
            "Positive separation scores indicate mechanisms cluster in morphology space. "
            "Significant p-values (< 0.05) suggest non-random clustering. "
            f"Overall silhouette: {silhouette_obs:.3f} (p={silhouette_pval:.3f})."
        ),
    }

    write_summary(
        output_dir / "summary.json",
        description=(
            "Clusters MitoTox mitotoxicant compounds by Cell Painting morphology and tests whether "
            "functional submechanisms (F01-F08) separate in phenotypic space, going beyond toxic vs "
            "non-toxic. Produces morphology and chemistry UMAP overlays, a hierarchical clustering "
            "dendrogram, a per-mechanism within-vs-between separation table, and a permutation-tested "
            "silhouette score."
        ),
        numbers=numbers,
        files=index_files(output_dir),
    )
    logger.info(f"Saved: {output_dir / 'summary.json'}")

    logger.success(f"Analysis complete: {output_dir}")


if __name__ == "__main__":
    app()
