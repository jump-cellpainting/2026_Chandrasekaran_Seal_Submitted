"""Curate commercially available compounds with distinctive Cell Painting profiles.

Two main objectives:
1. Aggregate Molport batch search results (batches were split for website limitations)
2. Curate multiple compound lists for different plate sizes

Approach:
- Filter to phenotypically active compounds (reproducible profiles via FDR-corrected p-value)
- Filter to Molport-available compounds FIRST
- Cluster profile data using rapids/cuML K-means
- Select N compounds per cluster (closest to centroid)

Output lists (6 total):
- 160 clusters × 1 compound = 160 compounds
- 160 clusters × 2 compounds = 320 compounds
- 320 clusters × 1 compound = 320 compounds
- 320 clusters × 2 compounds = 640 compounds

The 320-compound lists fit a 384-well plate with room for controls.
The 640-compound list allows users to select subsets based on availability/price.

Usage:
    pixi run -e rapids-cheminformatics python notebooks/2.36-seal-commercial-compounds.py

Outputs: data/processed/commercial-compounds/
"""

import zipfile
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib.colors import Normalize
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans

from jump_production.config import DEFAULT_DPI, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.gpu import has_gpu
from jump_production.io import join_activity, join_metadata, load_profiles, load_umap
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

HAS_GPU = has_gpu()

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
ANNDATA_DIR = INTERIM_DATA_DIR / "anndata"
CHEMICAL_SPACE_DIR = PROCESSED_DATA_DIR / "chemical-space" / "all"
OUTPUT_DIR = PROCESSED_DATA_DIR / "commercial-compounds"

# Molport batch search data (zip of batch inputs + Molport quote results)
MOLPORT_ZIP = EXTERNAL_DATA_DIR / "molport_batch_search.zip"

# Price cutoff for Molport compounds (applied early in filtering)
MAX_PRICE_USD = 200

# Clustering configurations: (n_clusters, n_per_cluster) -> output name
# 320 compounds fits a 384-well plate with room for controls
CLUSTER_CONFIGS = [
    (160, 1, "160c_1pc"),  # 160 clusters × 1 = 160 compounds
    (160, 2, "160c_2pc"),  # 160 clusters × 2 = 320 compounds
    (320, 1, "320c_1pc"),  # 320 clusters × 1 = 320 compounds
    (320, 2, "320c_2pc"),  # 320 clusters × 2 = 640 compounds
]

RANDOM_STATE = 42

app = typer.Typer()


# =============================================================================
# New Visualization Functions
# =============================================================================


def plot_price_distribution(
    curated_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_path: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Histogram of Molport prices for curated compounds.

    Args:
        curated_df: DataFrame with 'Price, USD' column
        ax: Matplotlib axes. If None, creates standalone figure.
        output_path: Output path for saving figure (standalone mode only)
        dpi: DPI for saved figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    prices = curated_df["Price, USD"].dropna()

    # Use log-scaled bins for price distribution
    if len(prices) > 0:
        # Clip extreme values for visualization
        prices_clipped = prices.clip(upper=prices.quantile(0.99))

        ax.hist(prices_clipped, bins=30, color="#3498db", edgecolor="white", alpha=0.8)

        # Add median line
        median_price = prices.median()
        ax.axvline(median_price, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: ${median_price:.0f}")

        # Add statistics annotation
        ax.annotate(
            f"n = {len(prices):,}\nMedian: ${median_price:.0f}\nMean: ${prices.mean():.0f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    ax.set_xlabel("Price (USD)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Price Distribution of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_coverage(
    curated_df: pd.DataFrame,
    n_clusters: int,
    ax: plt.Axes | None = None,
    output_path: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Bar chart showing compounds per cluster.

    Args:
        curated_df: DataFrame with 'cluster_id' column
        n_clusters: Total number of clusters
        ax: Matplotlib axes. If None, creates standalone figure.
        output_path: Output path for saving figure (standalone mode only)
        dpi: DPI for saved figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Count compounds per cluster
    cluster_counts = curated_df["cluster_id"].value_counts().sort_index()

    # Fill in missing clusters with 0
    all_clusters = pd.Series(0, index=range(n_clusters))
    all_clusters.update(cluster_counts)

    # Color bars by count
    colors = plt.cm.Blues(Normalize(vmin=0, vmax=all_clusters.max())(all_clusters.values))

    ax.bar(all_clusters.index, all_clusters.values, color=colors, edgecolor="none", width=0.8)

    # Add horizontal line for mean
    mean_count = curated_df.shape[0] / n_clusters
    ax.axhline(mean_count, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_count:.1f}")

    # Add statistics
    n_represented = (all_clusters > 0).sum()
    ax.annotate(
        f"Clusters represented: {n_represented}/{n_clusters}\nTotal compounds: {len(curated_df):,}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    ax.set_xlabel("Cluster ID", fontsize=10)
    ax.set_ylabel("Compounds", fontsize=10)
    ax.set_title("Cluster Coverage of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_activity_distribution(
    curated_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_path: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Histogram of normalized mAP values for curated compounds.

    Args:
        curated_df: DataFrame with 'mean_normalized_average_precision' column
        ax: Matplotlib axes. If None, creates standalone figure.
        output_path: Output path for saving figure (standalone mode only)
        dpi: DPI for saved figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    activity_col = "mean_normalized_average_precision"
    if activity_col not in curated_df.columns:
        logger.warning(f"Column {activity_col} not found in curated_df")
        if standalone:
            plt.close(fig)
        return

    activity = curated_df[activity_col].dropna()

    if len(activity) > 0:
        ax.hist(activity, bins=30, color="#2ca02c", edgecolor="white", alpha=0.8)

        # Add median line
        median_activity = activity.median()
        ax.axvline(
            median_activity, color="#e74c3c", linestyle="--", linewidth=2, label=f"Median: {median_activity:.3f}"
        )

        # Add statistics annotation
        ax.annotate(
            f"n = {len(activity):,}\nMedian: {median_activity:.3f}\nMean: {activity.mean():.3f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    ax.set_xlabel("Normalized mAP", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Phenotypic Activity of Curated Compounds", fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_comparison_distributions(
    all_results: dict[str, dict],
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot comparison of activity and price distributions across all configurations.

    Args:
        all_results: Dict mapping config_name to result dict containing 'curated_df'
        output_path: Output path for saving figure
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette for configurations
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # --- Left panel: Activity distribution ---
    ax = axes[0]
    activity_col = "mean_normalized_average_precision"

    for i, (config_name, result) in enumerate(all_results.items()):
        curated_df = result["curated_df"]
        if activity_col in curated_df.columns:
            activity = curated_df[activity_col].dropna()
            n_compounds = result["n_compounds"]
            label = f"{config_name} (n={n_compounds})"
            ax.hist(activity, bins=25, alpha=0.5, color=colors[i], label=label, edgecolor="white")

    ax.set_xlabel("Normalized mAP", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Phenotypic Activity Distribution", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right panel: Price distribution ---
    ax = axes[1]

    for i, (config_name, result) in enumerate(all_results.items()):
        curated_df = result["curated_df"]
        if "Price, USD" in curated_df.columns:
            prices = curated_df["Price, USD"].dropna()
            # Clip extreme values for visualization
            prices_clipped = prices.clip(upper=prices.quantile(0.99))
            n_compounds = result["n_compounds"]
            label = f"{config_name} (n={n_compounds})"
            ax.hist(prices_clipped, bins=25, alpha=0.5, color=colors[i], label=label, edgecolor="white")

    ax.set_xlabel("Price (USD)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Price Distribution", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_umap(
    adata: ad.AnnData,
    curated_jcp: set[str],
    output_path: Path,
    space_name: str = "Phenotypic",
    color_col: str | None = None,
    color_label: str = "",
    cmap: str = "viridis",
    title_suffix: str = "",
    curated_df: pd.DataFrame | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create a single UMAP figure with curated compounds highlighted and outliers clipped.

    Args:
        adata: AnnData with X_umap in obsm and JCP2022 in obs
        curated_jcp: Set of JCP2022 IDs for curated compounds
        output_path: Output path for saving figure
        space_name: Name of the space (e.g., "Phenotypic", "Chemical")
        color_col: Column name for continuous coloring (from curated_df)
        color_label: Label for colorbar
        cmap: Colormap for continuous coloring
        title_suffix: Suffix for title (e.g., "by Price", "by Activity")
        curated_df: DataFrame with curated compound data (for color_col values)
        dpi: DPI for saved figure
    """
    # Compute IQR-based bounds for outlier clipping
    xlim, ylim = compute_umap_bounds(adata, iqr_k=3.0)

    coords = adata.obsm["X_umap"]
    jcp_ids = adata.obs["JCP2022"].values

    # Log outlier statistics
    outlier_mask = (
        (coords[:, 0] < xlim[0]) | (coords[:, 0] > xlim[1]) | (coords[:, 1] < ylim[0]) | (coords[:, 1] > ylim[1])
    )
    n_outliers = outlier_mask.sum()
    logger.info(f"{space_name} UMAP: {n_outliers:,} outliers ({100 * n_outliers / len(coords):.1f}%) clipped")

    # Create mask for curated compounds
    curated_mask = np.array([jcp in curated_jcp for jcp in jcp_ids])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot background (all compounds in gray)
    ax.scatter(
        coords[~curated_mask, 0],
        coords[~curated_mask, 1],
        s=0.3,
        alpha=0.1,
        c="#cccccc",
        rasterized=True,
    )

    # Plot curated compounds
    curated_coords = coords[curated_mask]
    curated_jcp_arr = jcp_ids[curated_mask]

    if color_col is not None and curated_df is not None:
        # Get color values aligned with curated_coords
        jcp_to_color = dict(zip(curated_df["JCP2022"], curated_df[color_col]))
        color_values = np.array([jcp_to_color.get(jcp, np.nan) for jcp in curated_jcp_arr])

        # Filter out NaN values
        valid_mask = ~np.isnan(color_values)
        if valid_mask.sum() > 0:
            scatter = ax.scatter(
                curated_coords[valid_mask, 0],
                curated_coords[valid_mask, 1],
                s=8,
                alpha=0.8,
                c=color_values[valid_mask],
                cmap=cmap,
                edgecolor="black",
                linewidth=0.3,
                rasterized=True,
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
            cbar.set_label(color_label, fontsize=10)
            cbar.ax.tick_params(labelsize=9)
    else:
        # Simple colored scatter without continuous coloring
        ax.scatter(
            curated_coords[:, 0],
            curated_coords[:, 1],
            s=8,
            alpha=0.8,
            c="#e74c3c",
            edgecolor="black",
            linewidth=0.3,
            rasterized=True,
        )

    # Set axis limits (clips outliers)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)

    # Build title
    if title_suffix:
        title = f"{space_name} Space - {title_suffix}"
    else:
        title = f"{space_name} Space"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add count annotation
    ax.annotate(
        f"Curated: n = {len(curated_jcp):,}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path.name}")


# =============================================================================
# Part 1: Molport Search Validation
# =============================================================================


def load_batch_smiles(zf: zipfile.ZipFile) -> dict[int, list[str]]:
    """Load all batch files from zip and return dict of batch_id -> SMILES list."""
    batches = {}
    batch_files = sorted(n for n in zf.namelist() if "/to_search/batch_" in n and n.endswith(".txt"))
    for name in batch_files:
        batch_id = int(Path(name).stem.split("_")[1])
        smiles_list = [line.strip() for line in zf.read(name).decode().splitlines() if line.strip()]
        batches[batch_id] = smiles_list
        logger.info(f"Batch {batch_id:02d}: {len(smiles_list):,} SMILES")
    return batches


def load_molport_results(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Load all Molport result Excel files from zip."""
    all_results = []
    xls_files = sorted(n for n in zf.namelist() if "/search_results/Quote_" in n and n.endswith(".xls"))

    if not xls_files:
        raise FileNotFoundError("No Molport result files (Quote_*.xls) found in zip")

    for name in xls_files:
        df = pd.read_excel(zf.open(name))
        df["_source_file"] = Path(name).name
        all_results.append(df)
        logger.info(f"Loaded {Path(name).name}: {len(df):,} rows")

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"Total Molport results: {len(combined):,} rows")
    return combined


def validate_molport_search(batches: dict[int, list[str]], results: pd.DataFrame) -> dict:
    """Cross-validate batch inputs against Molport results.

    The batch splitting was due to Molport website limitations - scientifically it's
    one search. This function aggregates all batches and returns overall statistics.

    Returns:
        stats: Dictionary with aggregate statistics (no per-batch breakdown)
    """
    # Get unique SMILES that got hits
    smiles_with_hits = set(results["Search Criteria"].dropna().unique())

    # Aggregate all batches
    all_searched = set()
    for batch_id in sorted(batches.keys()):
        smiles_list = batches[batch_id]
        all_searched.update(smiles_list)

    # Overall stats
    total_searched = len(all_searched)
    total_hits = len(smiles_with_hits)

    # Check for SMILES in results that weren't in batches (shouldn't happen)
    unexpected = smiles_with_hits - all_searched
    if unexpected:
        logger.warning(f"Found {len(unexpected)} SMILES in results not in batches")

    stats = {
        "n_batches": len(batches),
        "n_result_files": results["_source_file"].nunique(),
        "total_searched": total_searched,
        "total_hits": total_hits,
        "overall_hit_rate_pct": total_hits / total_searched * 100 if total_searched > 0 else 0,
    }

    return stats


# =============================================================================
# Part 2: Profile Clustering (using rapids for GPU acceleration)
# =============================================================================


def cluster_profiles(
    adata: ad.AnnData,
    n_clusters: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering of profile data (not UMAP coordinates).

    Uses rapids cuML for GPU acceleration when available.

    Args:
        adata: AnnData with profile data in X
        n_clusters: Number of clusters

    Returns:
        labels: Cluster labels for each sample
        centroids: (n_clusters, n_features) array of cluster centroids
    """
    global HAS_GPU
    n_samples = adata.n_obs

    logger.info(f"Clustering {n_samples:,} profiles into {n_clusters} clusters")

    if HAS_GPU:
        try:
            import cupy as cp
            from cuml.cluster import KMeans as cuKMeans

            logger.info("Using GPU-accelerated clustering (cuML)")
            X = cp.asarray(adata.X)
            kmeans = cuKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
            kmeans.fit(X)
            labels = kmeans.labels_.get()
            centroids = kmeans.cluster_centers_.get()
        except ImportError:
            logger.warning("cuML not available, falling back to CPU")
            HAS_GPU = False

    if not HAS_GPU:
        logger.info("Using CPU clustering (sklearn)")
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(adata.X)
        centroids = kmeans.cluster_centers_

    # Log cluster size distribution
    sizes = np.bincount(labels)
    logger.info(f"Cluster sizes: min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}")

    return labels, centroids


def select_from_clusters(
    adata: ad.AnnData,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_per_cluster: int = 2,
) -> pd.DataFrame:
    """Select compounds closest to each cluster centroid.

    Args:
        adata: AnnData with JCP2022 in obs
        labels: Cluster labels for each sample
        centroids: (n_clusters, n_features) array of cluster centroids
        n_per_cluster: Number of compounds per cluster

    Returns:
        DataFrame with columns: JCP2022, cluster_id, centroid_distance
    """
    jcp_ids = adata.obs["JCP2022"].values
    X = adata.X
    n_clusters = len(centroids)

    candidates = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_X = X[mask]

        # Calculate distance to centroid (cosine distance)
        centroid = centroids[cluster_id]
        # Normalize for cosine distance
        cluster_norms = np.linalg.norm(cluster_X, axis=1, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        cluster_X_norm = cluster_X / np.maximum(cluster_norms, 1e-10)
        centroid_norm_vec = centroid / max(centroid_norm, 1e-10)

        # Cosine similarity -> distance
        cos_sim = cluster_X_norm @ centroid_norm_vec
        distances = 1 - cos_sim

        # Get closest compounds
        n_select = min(n_per_cluster, len(cluster_indices))
        closest_local = np.argsort(distances)[:n_select]

        for local_idx in closest_local:
            global_idx = cluster_indices[local_idx]
            candidates.append(
                {
                    "JCP2022": jcp_ids[global_idx],
                    "cluster_id": cluster_id,
                    "centroid_distance": float(distances[local_idx]),
                }
            )

    df = pd.DataFrame(candidates)
    logger.info(f"Selected {len(df):,} compounds from {n_clusters} clusters ({n_per_cluster} per cluster)")
    return df


# =============================================================================
# Part 3: MaxMin Chemical Diversity Selection
# =============================================================================


def compute_morgan_fingerprints(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprints for a list of SMILES.

    Returns:
        fps: List of fingerprint objects (or None for invalid SMILES)
        valid_mask: Boolean array indicating valid molecules
    """
    fps = []
    valid_mask = []

    # Use new RDKit fingerprint generator API
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = fpgen.GetFingerprint(mol)
            fps.append(fp)
            valid_mask.append(True)
        else:
            fps.append(None)
            valid_mask.append(False)

    return fps, np.array(valid_mask)


def maxmin_diversity_selection(
    fps: list,
    valid_mask: np.ndarray,
    target_count: int,
    seed_idx: int | None = None,
) -> list[int]:
    """MaxMin algorithm for chemical diversity selection.

    Greedily selects compounds that maximize the minimum Tanimoto distance
    to the already-selected set.

    Args:
        fps: List of fingerprints (can include None for invalid)
        valid_mask: Boolean array indicating valid fingerprints
        target_count: Number of compounds to select
        seed_idx: Optional starting compound index

    Returns:
        List of selected indices
    """
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        return []

    if target_count >= n_valid:
        logger.warning(f"Target count {target_count} >= valid compounds {n_valid}, returning all")
        return valid_indices.tolist()

    logger.info(f"Running MaxMin diversity selection: {n_valid:,} candidates -> {target_count} compounds")

    # Start with seed or first valid compound
    if seed_idx is not None and valid_mask[seed_idx]:
        first_idx = seed_idx
    else:
        first_idx = valid_indices[0]

    selected = [first_idx]
    selected_set = {first_idx}

    # Track minimum distance to selected set for each candidate
    min_distances = np.full(len(fps), np.inf)
    for idx in valid_indices:
        if idx != first_idx:
            min_distances[idx] = 1 - DataStructs.TanimotoSimilarity(fps[idx], fps[first_idx])

    # Greedy selection
    for i in range(1, target_count):
        if i % 100 == 0:
            logger.info(f"  Selected {i}/{target_count}")

        # Find candidate with maximum minimum distance
        best_idx = None
        best_dist = -1

        for idx in valid_indices:
            if idx in selected_set:
                continue
            if min_distances[idx] > best_dist:
                best_dist = min_distances[idx]
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)

        # Update minimum distances
        for idx in valid_indices:
            if idx not in selected_set:
                dist = 1 - DataStructs.TanimotoSimilarity(fps[idx], fps[best_idx])
                if dist < min_distances[idx]:
                    min_distances[idx] = dist

    logger.info(f"Selected {len(selected)} compounds with MaxMin diversity")
    return selected


# =============================================================================
# Visualization
# =============================================================================


def plot_umap_clusters(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    selected_mask: np.ndarray | None = None,
    output_path: Path | None = None,
    dpi: int = DEFAULT_DPI,
):
    """Plot UMAP with cluster assignments and optionally highlight selected compounds."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all points colored by cluster
    ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=labels,
        cmap="tab20",
        s=1,
        alpha=0.3,
        rasterized=True,
    )

    # Plot centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=50,
        marker="x",
        linewidths=2,
        label="Centroids",
    )

    # Highlight selected compounds if provided
    if selected_mask is not None:
        ax.scatter(
            umap_coords[selected_mask, 0],
            umap_coords[selected_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=20,
            linewidths=1,
            label=f"Selected ({selected_mask.sum():,})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cell Painting UMAP with K-means Clusters")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")

    plt.close()


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="Plot DPI"),
):
    """Curate commercially available compounds with distinctive profiles.

    Generates 6 compound lists with different cluster configurations:
    - 160 clusters × 1 compound = 160 compounds
    - 160 clusters × 2 compounds = 320 compounds
    - 320 clusters × 1 compound = 320 compounds
    - 320 clusters × 2 compounds = 640 compounds
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Part 1: Molport Search Aggregation
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part 1: Molport Search Aggregation")
    logger.info("=" * 60)

    # Load from zip (following project pattern for external zip data)
    logger.info(f"Loading Molport data from {MOLPORT_ZIP}")
    with zipfile.ZipFile(MOLPORT_ZIP, "r") as zf:
        batches = load_batch_smiles(zf)
        molport_results = load_molport_results(zf)

    # Validate (aggregate stats only - batches were split for website limitations)
    validation_stats = validate_molport_search(batches, molport_results)

    logger.info("Molport search summary:")
    logger.info(f"  Batches submitted: {validation_stats['n_batches']}")
    logger.info(f"  Result files received: {validation_stats['n_result_files']}")
    logger.info(f"  Total compounds searched: {validation_stats['total_searched']:,}")
    logger.info(f"  Compounds with Molport hits: {validation_stats['total_hits']:,}")
    logger.info(f"  Overall hit rate: {validation_stats['overall_hit_rate_pct']:.1f}%")

    # Get all SMILES with hits and their best matches (Perfect > Isomer)
    molport_hits = (
        molport_results.groupby("Search Criteria")
        .agg(
            {
                "Molport ID": "first",
                "Match Type": lambda x: "Perfect" if "Perfect" in x.values else "Isomer",
                "Price, USD": "min",
                "Supplier Name": "first",
            }
        )
        .reset_index()
        .rename(columns={"Search Criteria": "SMILES"})
    )

    # Apply price cutoff early to exclude expensive compounds
    n_before_price = len(molport_hits)
    molport_hits = molport_hits[molport_hits["Price, USD"] <= MAX_PRICE_USD]
    logger.info(f"Price cutoff (≤${MAX_PRICE_USD}): {n_before_price:,} → {len(molport_hits):,} compounds")

    molport_hits.to_csv(output_dir / "molport_hits.csv", index=False)
    logger.info(f"Saved {len(molport_hits):,} unique Molport hits")

    # =========================================================================
    # Part 2: Load Profiles and Filter to Active + Molport Available
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part 2: Load Profiles and Filter to Active + Available")
    logger.info("=" * 60)

    # Load profiles (actual feature data)
    adata_profiles = load_profiles("compound_no_source7", level="perturbation")
    adata_profiles = join_metadata(adata_profiles, modality="compound", level="perturbation")
    logger.info(f"Loaded profiles with {adata_profiles.n_obs:,} compounds, {adata_profiles.n_vars:,} features")

    # Join phenotypic activity data
    adata_profiles = join_activity(adata_profiles, "compound_no_source7", preprocessing="activity_no_target2")
    n_total = adata_profiles.n_obs
    n_significant = (adata_profiles.obs["below_corrected_p"] == "True").sum()
    logger.info(f"Activity data: {n_significant:,} phenotypically active of {n_total:,}")

    # Filter to phenotypically active
    active_mask = adata_profiles.obs["below_corrected_p"] == "True"
    adata_active = adata_profiles[active_mask].copy()
    logger.info(f"Filtered to phenotypically active: {adata_active.n_obs:,}")

    # Filter to Molport available
    molport_smiles_set = set(molport_hits["SMILES"])
    adata_active.obs["molport_available"] = adata_active.obs["SMILES"].isin(molport_smiles_set)
    n_available = adata_active.obs["molport_available"].sum()
    logger.info(f"Molport available among active: {n_available:,} of {adata_active.n_obs:,}")

    # Filter to Molport available FIRST (before clustering)
    available_mask = adata_active.obs["molport_available"]
    adata_available = adata_active[available_mask].copy()
    logger.info(f"Compounds for clustering: {adata_available.n_obs:,} (active + Molport available)")

    if adata_available.n_obs == 0:
        logger.error("No active Molport-available compounds! Cannot proceed.")
        return 1

    # Load UMAPs for visualization (once, reuse for all configurations)
    adata_pheno = load_umap("compound_no_source7", level="perturbation", metric="cosine", filter_name="all")
    adata_pheno = join_metadata(adata_pheno, modality="compound", level="perturbation")

    # Load structure UMAP (Morgan fingerprints from notebook 2.04a)
    structure_umap_path = CHEMICAL_SPACE_DIR / "structure_umap.h5ad"
    if structure_umap_path.exists():
        logger.info(f"Loading structure UMAP from {structure_umap_path}")
        adata_structure = ad.read_h5ad(structure_umap_path)
        if "Metadata_JCP2022" in adata_structure.obs.columns:
            adata_structure.obs["JCP2022"] = adata_structure.obs["Metadata_JCP2022"]
        logger.info(f"Loaded structure UMAP with {adata_structure.n_obs:,} compounds")
    else:
        logger.warning(f"Structure UMAP not found at {structure_umap_path}")
        adata_structure = adata_pheno

    # =========================================================================
    # Part 3: Generate Multiple Compound Lists
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Part 3: Generate Multiple Compound Lists")
    logger.info("=" * 60)

    # Track all results for summary
    all_results = {}

    # Get unique n_clusters values to avoid redundant clustering
    unique_n_clusters = sorted(set(cfg[0] for cfg in CLUSTER_CONFIGS))
    cluster_results = {}

    for n_clusters in unique_n_clusters:
        logger.info(f"\nClustering into {n_clusters} clusters...")
        labels, centroids = cluster_profiles(adata_available, n_clusters=n_clusters)
        cluster_results[n_clusters] = (labels, centroids)

    # Generate each configuration
    for n_clusters, n_per_cluster, config_name in CLUSTER_CONFIGS:
        logger.info("=" * 60)
        logger.info(f"Generating: {config_name} ({n_clusters} clusters × {n_per_cluster} per cluster)")
        logger.info("=" * 60)

        labels, centroids = cluster_results[n_clusters]

        # Select compounds
        curated_df = select_from_clusters(adata_available, labels, centroids, n_per_cluster=n_per_cluster)

        # Add metadata
        curated_df = curated_df.merge(
            adata_available.obs[["JCP2022", "SMILES", "InChIKey"]].drop_duplicates(),
            on="JCP2022",
            how="left",
        )

        # Add Molport info
        curated_df = curated_df.merge(
            molport_hits[["SMILES", "Molport ID", "Match Type", "Price, USD", "Supplier Name"]],
            on="SMILES",
            how="left",
        )

        # Add activity data
        activity_cols = ["mean_normalized_average_precision", "corrected_p_value"]
        activity_data = adata_available.obs[
            ["JCP2022"] + [c for c in activity_cols if c in adata_available.obs.columns]
        ].drop_duplicates()
        curated_df = curated_df.merge(activity_data, on="JCP2022", how="left")

        # Sort by cluster_id then centroid_distance
        curated_df = curated_df.sort_values(["cluster_id", "centroid_distance"]).reset_index(drop=True)

        # Save curated compounds
        curated_df.to_csv(output_dir / f"curated_compounds_{config_name}.csv", index=False)
        logger.info(f"Saved {len(curated_df):,} curated compounds to curated_compounds_{config_name}.csv")

        # Store for summary
        all_results[config_name] = {
            "n_clusters": n_clusters,
            "n_per_cluster": n_per_cluster,
            "n_compounds": len(curated_df),
            "n_clusters_represented": curated_df["cluster_id"].nunique(),
            "curated_df": curated_df,
        }

        # Generate visualizations for this configuration
        if not skip_plots:
            logger.info(f"Generating visualizations for {config_name}...")
            curated_jcp = set(curated_df["JCP2022"])

            # Phenotypic UMAP - overview
            plot_single_umap(
                adata_pheno,
                curated_jcp,
                output_path=output_dir / f"umap_phenotypic_{config_name}.png",
                space_name=f"Phenotypic ({config_name})",
                dpi=dpi,
            )

            # Phenotypic UMAP - by price
            plot_single_umap(
                adata_pheno,
                curated_jcp,
                output_path=output_dir / f"umap_phenotypic_{config_name}_price.png",
                space_name=f"Phenotypic ({config_name})",
                color_col="Price, USD",
                color_label="Price (USD)",
                cmap="YlOrRd",
                title_suffix="by Price",
                curated_df=curated_df,
                dpi=dpi,
            )

            # Phenotypic UMAP - by activity
            plot_single_umap(
                adata_pheno,
                curated_jcp,
                output_path=output_dir / f"umap_phenotypic_{config_name}_activity.png",
                space_name=f"Phenotypic ({config_name})",
                color_col="mean_normalized_average_precision",
                color_label="Normalized mAP",
                cmap="viridis",
                title_suffix="by Activity",
                curated_df=curated_df,
                dpi=dpi,
            )

            # Structure UMAP - overview
            plot_single_umap(
                adata_structure,
                curated_jcp,
                output_path=output_dir / f"umap_structure_{config_name}.png",
                space_name=f"Structure ({config_name})",
                dpi=dpi,
            )

            # Structure UMAP - by price
            plot_single_umap(
                adata_structure,
                curated_jcp,
                output_path=output_dir / f"umap_structure_{config_name}_price.png",
                space_name=f"Structure ({config_name})",
                color_col="Price, USD",
                color_label="Price (USD)",
                cmap="YlOrRd",
                title_suffix="by Price",
                curated_df=curated_df,
                dpi=dpi,
            )

            # Structure UMAP - by activity
            plot_single_umap(
                adata_structure,
                curated_jcp,
                output_path=output_dir / f"umap_structure_{config_name}_activity.png",
                space_name=f"Structure ({config_name})",
                color_col="mean_normalized_average_precision",
                color_label="Normalized mAP",
                cmap="viridis",
                title_suffix="by Activity",
                curated_df=curated_df,
                dpi=dpi,
            )

    # Generate comparison distribution plot across all configurations
    if not skip_plots:
        logger.info("Generating comparison distribution plots...")
        plot_comparison_distributions(
            all_results,
            output_path=output_dir / "comparison_distributions.png",
            dpi=dpi,
        )
        logger.info("Saved comparison_distributions.png")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    # Build summary for each configuration
    configs_summary = {}
    for config_name, result in all_results.items():
        curated_df = result["curated_df"]

        # Activity stats
        activity_col = "mean_normalized_average_precision"
        if activity_col in curated_df.columns:
            activity_stats = {
                "median": round(float(curated_df[activity_col].median()), 4),
                "mean": round(float(curated_df[activity_col].mean()), 4),
                "min": round(float(curated_df[activity_col].min()), 4),
                "max": round(float(curated_df[activity_col].max()), 4),
            }
        else:
            activity_stats = {}

        # Price stats
        if "Price, USD" in curated_df.columns:
            prices = curated_df["Price, USD"].dropna()
            price_stats = {
                "median": round(float(prices.median()), 2),
                "mean": round(float(prices.mean()), 2),
                "min": round(float(prices.min()), 2),
                "max": round(float(prices.max()), 2),
            }
        else:
            price_stats = {}

        configs_summary[config_name] = {
            "n_clusters": result["n_clusters"],
            "n_per_cluster": result["n_per_cluster"],
            "n_compounds": result["n_compounds"],
            "n_clusters_represented": result["n_clusters_represented"],
            "activity_stats": activity_stats,
            "price_stats": price_stats,
        }

        logger.info(f"\n{config_name}:")
        logger.info(f"  - {result['n_compounds']:,} compounds from {result['n_clusters']} clusters")
        logger.info(f"  - Clusters represented: {result['n_clusters_represented']}")
        if activity_stats:
            logger.info(f"  - Median normalized mAP: {activity_stats['median']:.3f}")
        if price_stats:
            logger.info(f"  - Median price: ${price_stats['median']:.0f}")

    numbers = {
        "task": "Commercial compounds curation - multiple configurations",
        "molport_search": {
            "total_searched": int(validation_stats["total_searched"]),
            "total_hits": int(validation_stats["total_hits"]),
            "hit_rate_pct": round(float(validation_stats["overall_hit_rate_pct"]), 2),
            "price_cutoff_usd": MAX_PRICE_USD,
            "n_after_price_cutoff": len(molport_hits),
        },
        "phenotypic_activity_prefilter": {
            "n_total_compounds": int(n_total),
            "n_active_compounds": int(n_significant),
            "active_rate_pct": round(float(n_significant / n_total * 100), 2),
        },
        "filtering": {
            "n_active_molport_available": int(adata_available.n_obs),
        },
        "configurations": configs_summary,
    }

    description = (
        "Curates commercially available (Molport) compounds with distinctive Cell Painting profiles "
        "by filtering to phenotypically active, Molport-available compounds and K-means clustering their "
        "profiles to pick compounds near each cluster centroid. Produces six compound lists across "
        "160/320 clusters at 1 or 2 compounds per cluster, sized to fit 384-well plates."
    )

    write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=numbers,
        files=index_files(output_dir),
    )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Generated compound lists:")
    for config_name, result in all_results.items():
        logger.info(f"  - curated_compounds_{config_name}.csv: {result['n_compounds']:,} compounds")

    return 0


if __name__ == "__main__":
    app()
