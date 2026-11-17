"""Investigate fingerprint transformations for Tanimoto approximation.

This experiment evaluates different transformations of Morgan fingerprints
to find methods where standard metrics (Euclidean, cosine) approximate
Tanimoto/Jaccard similarity.

Motivation:
- Tanimoto/Jaccard is the chemically correct metric for binary fingerprints
- Many ML algorithms require Euclidean or inner product spaces
- We want fast transformations that preserve similarity relationships

Transformations tested:
1. L2 normalization (baseline)
2. Popcount normalization (divide by sqrt of bit count)
3. Concatenated features (add bit count info)
4. MinHash signatures
5. IDF-weighted features
6. Tanimoto Random Features (simplified)

Key finding: Simple L2 normalization achieves ρ=0.998 correlation with Tanimoto
for JUMP compounds due to homogeneous bit densities (CV=20.4%). See Issue #26.

Metrics:
- Transformation speed
- Spearman correlation with Tanimoto
- k-NN recall at k=10,50,100

Usage:
    pixi run python notebooks/2.04b-ss-fingerprint-metrics.py
    pixi run python notebooks/2.04b-ss-fingerprint-metrics.py --sample-size 5000

Outputs: data/processed/fingerprint-metrics/
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from jump_production.config import DEFAULT_DPI
from jump_production.summaries import index_files, write_summary

# Configuration
MORGAN_FP_FILE = Path("data/interim/compound_featurization/morgan_fp.npz")
OUTPUT_DIR = Path("data/processed/fingerprint-metrics")

app = typer.Typer()


# =============================================================================
# Tanimoto/Jaccard Reference Implementation
# =============================================================================


def tanimoto_similarity(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise Tanimoto similarity for binary fingerprints.

    Tanimoto(A,B) = |A∩B| / |A∪B| = (A·B) / (|A| + |B| - A·B)

    Args:
        X: Binary fingerprint matrix (n_samples, n_features)
        Y: Optional second matrix. If None, computes X vs X.

    Returns:
        Similarity matrix (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X

    # Compute dot products (intersection counts for binary vectors)
    intersection = X @ Y.T

    # Compute bit counts (popcount)
    X_bits = X.sum(axis=1, keepdims=True)
    Y_bits = Y.sum(axis=1, keepdims=True)

    # Union = |A| + |B| - |A∩B|
    union = X_bits + Y_bits.T - intersection

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = intersection / union
        similarity = np.nan_to_num(similarity, nan=0.0)

    return similarity


def tanimoto_distance(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise Tanimoto distance (1 - similarity)."""
    return 1.0 - tanimoto_similarity(X, Y)


# =============================================================================
# Transformation Functions
# =============================================================================


def transform_l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2 normalize fingerprints to unit sphere.

    After transformation: inner product = cosine similarity.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return X / norms


def transform_popcount_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize by sqrt of popcount (number of set bits).

    After transformation: inner product = (A·B) / sqrt(|A|·|B|)
    This equals cosine similarity for the original binary vectors.
    """
    popcounts = X.sum(axis=1, keepdims=True)
    sqrt_popcounts = np.sqrt(np.maximum(popcounts, 1.0))
    return X / sqrt_popcounts


def transform_concatenate_popcount(X: np.ndarray) -> np.ndarray:
    """L2 normalize and concatenate popcount features.

    Appends sqrt(popcount) and 1/sqrt(popcount) to capture
    the bit count information that Tanimoto uses.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_norm = X / norms

    popcounts = X.sum(axis=1, keepdims=True)
    sqrt_pop = np.sqrt(np.maximum(popcounts, 1.0))
    inv_sqrt_pop = 1.0 / sqrt_pop

    # Scale these features to be on similar scale as normalized bits
    # Typical popcount is ~47, sqrt(47) ≈ 6.8, so scale by 1/10
    sqrt_pop_scaled = sqrt_pop / 10.0
    inv_sqrt_pop_scaled = inv_sqrt_pop * 10.0

    return np.hstack([X_norm, sqrt_pop_scaled, inv_sqrt_pop_scaled])


def transform_minhash_fast(X: np.ndarray, n_hashes: int = 256, seed: int = 42) -> np.ndarray:
    """Fast vectorized MinHash using random hash values.

    Instead of permutations, assigns random hash values to each bit position
    and takes the minimum hash value among set bits.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    signatures = np.zeros((n_samples, n_hashes), dtype=np.float32)

    for h in range(n_hashes):
        # Random hash values for each feature position
        hash_vals = rng.random(n_features).astype(np.float32)

        # For set bits, get their hash values; for unset bits, use infinity
        masked = np.where(X == 1, hash_vals, np.inf)

        # Minimum hash value among set bits
        signatures[:, h] = masked.min(axis=1)

    # Replace inf with 1.0 for samples with no bits set
    signatures[signatures == np.inf] = 1.0

    return signatures


def transform_idf_weighted(X: np.ndarray, smooth_idf: bool = True) -> np.ndarray:
    """Weight bits by inverse document frequency and L2 normalize.

    Rare bits (present in few compounds) get higher weights.
    Common bits get lower weights.
    """
    n_samples, n_features = X.shape

    # Document frequency: how many samples have each bit set
    df = X.sum(axis=0)

    if smooth_idf:
        # Smooth IDF: log((n + 1) / (df + 1)) + 1
        idf = np.log((n_samples + 1) / (df + 1)) + 1
    else:
        # Standard IDF: log(n / df), with smoothing to avoid log(0)
        idf = np.log(n_samples / np.maximum(df, 1))

    # Weight the bits
    X_weighted = X * idf

    # L2 normalize
    norms = np.linalg.norm(X_weighted, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)

    return X_weighted / norms


def transform_tanimoto_random_features(X: np.ndarray, n_features_out: int = 512, seed: int = 42) -> np.ndarray:
    """Simplified Tanimoto random features based on power series expansion.

    This is a simplified version inspired by Tripp (2023).
    Uses random projections with popcount-based scaling.
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)

    # Random projection matrix (Rademacher: +1 or -1)
    W = rng.choice([-1, 1], size=(n_features, n_features_out)).astype(np.float32)
    W /= np.sqrt(n_features)

    # Project
    Z = X @ W

    # Scale by popcount-based factor to approximate Tanimoto
    popcounts = X.sum(axis=1, keepdims=True)
    # Scale factor: inversely proportional to popcount
    scale = 1.0 / np.sqrt(np.maximum(popcounts, 1.0))

    return Z * scale


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_knn_recall(
    X_transformed: np.ndarray,
    X_original: np.ndarray,
    k_values: list[int],
    metric_transformed: str = "cosine",
) -> dict[int, float]:
    """Compute k-NN recall: fraction of true Tanimoto neighbors found.

    Args:
        X_transformed: Transformed fingerprints
        X_original: Original binary fingerprints
        k_values: List of k values to evaluate
        metric_transformed: Metric to use on transformed space

    Returns:
        Dict mapping k to recall value
    """
    n_samples = X_original.shape[0]

    # Ground truth: k-NN by Tanimoto
    logger.info("Computing ground truth Tanimoto k-NN...")
    tanimoto_dist = tanimoto_distance(X_original)
    np.fill_diagonal(tanimoto_dist, np.inf)  # Exclude self

    max_k = max(k_values)

    # True neighbors by Tanimoto
    true_neighbors = np.argsort(tanimoto_dist, axis=1)[:, :max_k]

    # Predicted neighbors by transformed metric
    logger.info(f"Computing predicted k-NN ({metric_transformed})...")
    if metric_transformed == "cosine":
        # Cosine distance = 1 - cosine similarity
        sim = cosine_similarity(X_transformed)
        np.fill_diagonal(sim, -np.inf)  # Exclude self
        pred_neighbors = np.argsort(-sim, axis=1)[:, :max_k]  # Higher sim = closer
    elif metric_transformed == "euclidean":
        nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
        nn.fit(X_transformed)
        _, indices = nn.kneighbors(X_transformed)
        pred_neighbors = indices[:, 1:]  # Exclude self (first neighbor)
    else:
        raise ValueError(f"Unknown metric: {metric_transformed}")

    # Compute recall for each k
    recalls = {}
    for k in k_values:
        # For each sample, what fraction of top-k Tanimoto neighbors are in predicted top-k?
        true_k = true_neighbors[:, :k]
        pred_k = pred_neighbors[:, :k]

        recall_per_sample = []
        for i in range(n_samples):
            intersection = len(set(true_k[i]) & set(pred_k[i]))
            recall_per_sample.append(intersection / k)

        recalls[k] = np.mean(recall_per_sample)

    return recalls


def evaluate_transformation(
    name: str,
    transform_fn,
    X_sample: np.ndarray,
    X_full: np.ndarray,
    tanimoto_flat: np.ndarray,
    k_values: list[int] = [10, 50, 100],
    metric: str = "cosine",
) -> dict:
    """Evaluate a single transformation method.

    Returns dict with timing and accuracy metrics.
    """
    result = {"name": name, "metric": metric}

    # Time transformation on full dataset
    logger.info(f"Timing {name} on full dataset ({X_full.shape[0]:,} samples)...")
    t0 = time.perf_counter()
    _ = transform_fn(X_full)
    result["transform_time_full_s"] = time.perf_counter() - t0

    # Transform sample
    t0 = time.perf_counter()
    X_transformed = transform_fn(X_sample)
    result["transform_time_sample_s"] = time.perf_counter() - t0
    result["output_dim"] = X_transformed.shape[1]

    # Compute pairwise distances/similarities in transformed space
    logger.info(f"Computing {metric} on transformed space...")
    if metric == "cosine":
        sim_transformed = cosine_similarity(X_transformed)
        # Convert to distance for correlation (lower = more similar like Tanimoto distance)
        dist_transformed_flat = 1.0 - sim_transformed[np.triu_indices(len(X_sample), k=1)]
    elif metric == "euclidean":
        dist_transformed = cdist(X_transformed, X_transformed, metric="euclidean")
        dist_transformed_flat = dist_transformed[np.triu_indices(len(X_sample), k=1)]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Spearman correlation with Tanimoto distance
    # We want transformed distances to correlate with Tanimoto distances
    tanimoto_dist_flat = 1.0 - tanimoto_flat  # Convert similarity to distance
    correlation, p_value = spearmanr(dist_transformed_flat, tanimoto_dist_flat)
    result["spearman_correlation"] = correlation
    result["spearman_p_value"] = p_value

    # k-NN recall
    logger.info(f"Computing k-NN recall for {name}...")
    X_original = X_sample  # Original binary fingerprints
    recalls = compute_knn_recall(X_transformed, X_original, k_values, metric_transformed=metric)
    for k, recall in recalls.items():
        result[f"recall_at_{k}"] = recall

    return result


# =============================================================================
# Visualization
# =============================================================================


def plot_correlation_scatter(
    X_sample: np.ndarray,
    transformations: dict,
    output_dir: Path,
    n_pairs: int = 10000,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create scatter plots of transformed distances vs Tanimoto distances."""
    # Compute Tanimoto similarities
    tanimoto_sim = tanimoto_similarity(X_sample)
    tanimoto_dist = 1.0 - tanimoto_sim

    # Get upper triangle (excluding diagonal)
    n = len(X_sample)
    triu_idx = np.triu_indices(n, k=1)
    tanimoto_flat = tanimoto_dist[triu_idx]

    # Subsample pairs for plotting
    rng = np.random.default_rng(42)
    if len(tanimoto_flat) > n_pairs:
        sample_idx = rng.choice(len(tanimoto_flat), size=n_pairs, replace=False)
    else:
        sample_idx = np.arange(len(tanimoto_flat))

    tanimoto_sample = tanimoto_flat[sample_idx]

    n_transforms = len(transformations)
    n_cols = 3
    n_rows = (n_transforms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_transforms > 1 else [axes]

    for ax, (name, (transform_fn, metric)) in zip(axes, transformations.items()):
        X_transformed = transform_fn(X_sample)

        if metric == "cosine":
            sim = cosine_similarity(X_transformed)
            dist_flat = 1.0 - sim[triu_idx]
        else:
            dist = cdist(X_transformed, X_transformed, metric="euclidean")
            dist_flat = dist[triu_idx]

        dist_sample = dist_flat[sample_idx]

        # Compute correlation
        corr, _ = spearmanr(dist_sample, tanimoto_sample)

        ax.scatter(tanimoto_sample, dist_sample, s=1, alpha=0.1, rasterized=True)
        ax.set_xlabel("Tanimoto Distance", fontsize=9)
        ax.set_ylabel(f"{metric.title()} Distance", fontsize=9)
        ax.set_title(f"{name}\nSpearman ρ = {corr:.3f}", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide empty axes
    for ax in axes[len(transformations) :]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "fingerprint_metrics_correlation.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation scatter plots")


def plot_results_summary(results: list[dict], output_dir: Path, dpi: int = DEFAULT_DPI) -> None:
    """Create summary bar chart of results."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    names = [r["name"] for r in results]
    x = np.arange(len(names))

    # Spearman correlation
    correlations = [r["spearman_correlation"] for r in results]
    axes[0].bar(x, correlations, color="#3498db", edgecolor="none")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Spearman Correlation")
    axes[0].set_title("Correlation with Tanimoto", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Recall@10
    recalls = [r.get("recall_at_10", 0) for r in results]
    axes[1].bar(x, recalls, color="#2ecc71", edgecolor="none")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Recall@10")
    axes[1].set_title("k-NN Recall (k=10)", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Transform time
    times = [r["transform_time_full_s"] for r in results]
    axes[2].bar(x, times, color="#e74c3c", edgecolor="none")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Time (seconds)")
    axes[2].set_title("Transform Time (Full Dataset)", fontweight="bold")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "fingerprint_metrics_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved results summary plot")


def run_umap_comparison(X_sample: np.ndarray, output_dir: Path, seed: int = 42, dpi: int = DEFAULT_DPI) -> None:
    """Compare UMAP embeddings using Jaccard vs Cosine on L2-normalized.

    Creates side-by-side UMAP plots to visually compare the structure.
    """
    import umap

    logger.info("Running UMAP comparison (Jaccard vs Cosine on L2-normalized)...")

    # UMAP with Jaccard (ground truth for binary fingerprints)
    logger.info("  Computing UMAP with Jaccard metric...")
    reducer_jaccard = umap.UMAP(metric="jaccard", n_neighbors=15, min_dist=0.1, random_state=seed)
    embedding_jaccard = reducer_jaccard.fit_transform(X_sample)

    # UMAP with cosine on L2-normalized (our best approximation)
    logger.info("  Computing UMAP with cosine on L2-normalized...")
    X_l2 = transform_l2_normalize(X_sample)
    reducer_cosine = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1, random_state=seed)
    embedding_cosine = reducer_cosine.fit_transform(X_l2)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(embedding_jaccard[:, 0], embedding_jaccard[:, 1], s=1, alpha=0.3, c="#3498db", rasterized=True)
    axes[0].set_title("UMAP with Jaccard (Ground Truth)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_aspect("equal")

    axes[1].scatter(embedding_cosine[:, 0], embedding_cosine[:, 1], s=1, alpha=0.3, c="#e74c3c", rasterized=True)
    axes[1].set_title("UMAP with Cosine (L2 Normalized)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_aspect("equal")

    fig.suptitle(f"UMAP Comparison (n={len(X_sample):,} compounds)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fingerprint_metrics_umap_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UMAP comparison plot")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    sample_size: int = typer.Option(
        3000,
        "--sample-size",
        "-n",
        help="Number of compounds to sample for accuracy evaluation",
    ),
    n_minhash: int = typer.Option(
        256,
        "--n-minhash",
        help="Number of MinHash hash functions",
    ),
    skip_slow: bool = typer.Option(
        False,
        "--skip-slow",
        help="Skip slower methods (MinHash)",
    ),
    run_umap: bool = typer.Option(
        False,
        "--run-umap",
        help="Run UMAP comparison (slower, requires umap-learn)",
    ),
    umap_sample: int = typer.Option(
        5000,
        "--umap-sample",
        help="Number of compounds for UMAP comparison",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Evaluate fingerprint transformations for Tanimoto approximation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Morgan fingerprints
    logger.info(f"Loading Morgan fingerprints from {MORGAN_FP_FILE}...")
    fp_data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    X_full = fp_data["fingerprints"].astype(np.float32)
    valid_mask = fp_data["valid_mask"]

    # Filter to valid fingerprints
    X_full = X_full[valid_mask]
    logger.info(f"Loaded {X_full.shape[0]:,} valid Morgan fingerprints, {X_full.shape[1]} bits")

    # Sample for accuracy evaluation
    rng = np.random.default_rng(seed)
    if sample_size < len(X_full):
        sample_idx = rng.choice(len(X_full), size=sample_size, replace=False)
        X_sample = X_full[sample_idx]
    else:
        sample_idx = np.arange(len(X_full))
        X_sample = X_full
    logger.info(f"Using {len(X_sample):,} samples for accuracy evaluation")

    # Compute ground truth Tanimoto similarities (upper triangle)
    logger.info("Computing ground truth Tanimoto similarities...")
    tanimoto_sim = tanimoto_similarity(X_sample)
    triu_idx = np.triu_indices(len(X_sample), k=1)
    tanimoto_flat = tanimoto_sim[triu_idx]

    # Define transformations to test
    transformations = {
        "L2 Normalize": (transform_l2_normalize, "cosine"),
        "Popcount Normalize": (transform_popcount_normalize, "cosine"),
        "Concat Popcount": (transform_concatenate_popcount, "euclidean"),
        "IDF Weighted": (transform_idf_weighted, "cosine"),
        "Random Features": (
            lambda X: transform_tanimoto_random_features(X, n_features_out=512, seed=seed),
            "cosine",
        ),
    }

    if not skip_slow:
        transformations["MinHash (fast)"] = (
            lambda X: transform_minhash_fast(X, n_hashes=n_minhash, seed=seed),
            "euclidean",
        )

    # Evaluate each transformation
    results = []
    for name, (transform_fn, metric) in transformations.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"{'=' * 60}")

        result = evaluate_transformation(
            name=name,
            transform_fn=transform_fn,
            X_sample=X_sample,
            X_full=X_full,
            tanimoto_flat=tanimoto_flat,
            metric=metric,
        )
        results.append(result)

        logger.info(f"  Spearman correlation: {result['spearman_correlation']:.4f}")
        logger.info(f"  Recall@10: {result.get('recall_at_10', 0):.4f}")
        logger.info(f"  Transform time (full): {result['transform_time_full_s']:.3f}s")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    plot_correlation_scatter(X_sample, transformations, OUTPUT_DIR, dpi=dpi)
    plot_results_summary(results, OUTPUT_DIR, dpi=dpi)

    # UMAP comparison (optional)
    if run_umap:
        # Sample for UMAP (can be different from accuracy sample)
        if umap_sample < len(X_full):
            umap_idx = rng.choice(len(X_full), size=umap_sample, replace=False)
            X_umap = X_full[umap_idx]
        else:
            X_umap = X_full
        run_umap_comparison(X_umap, OUTPUT_DIR, seed=seed, dpi=dpi)

    # Best/fastest picks (also reported in the recommendation block below)
    best_by_corr = max(results, key=lambda x: x["spearman_correlation"])
    best_by_recall = max(results, key=lambda x: x.get("recall_at_10", 0))
    fastest = min(results, key=lambda x: x["transform_time_full_s"])

    # Save results as a standardized summary envelope (description + numbers + files)
    results_file = OUTPUT_DIR / "fingerprint_metrics_summary.json"
    numbers = {
        "n_compounds_full": int(X_full.shape[0]),
        "n_bits": int(X_full.shape[1]),
        "n_samples_evaluated": int(len(X_sample)),
        "best_correlation": {
            "name": best_by_corr["name"],
            "spearman_correlation": best_by_corr["spearman_correlation"],
        },
        "best_recall_at_10": {
            "name": best_by_recall["name"],
            "recall_at_10": best_by_recall.get("recall_at_10", 0),
        },
        "fastest": {
            "name": fastest["name"],
            "transform_time_full_s": fastest["transform_time_full_s"],
        },
        "transformations": results,
    }
    description = (
        "Comparison of fingerprint transformations that let standard metrics (cosine, Euclidean) "
        "approximate Tanimoto similarity on JUMP Morgan fingerprints, reporting each method's Spearman "
        "correlation with Tanimoto distance, k-NN recall at k=10/50/100, and transform time."
    )
    write_summary(results_file, description=description, numbers=numbers, files=index_files(OUTPUT_DIR))
    logger.info(f"Saved results to {results_file}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':<25} {'ρ':>8} {'R@10':>8} {'R@50':>8} {'Time(s)':>10}")
    logger.info("-" * 80)
    for r in sorted(results, key=lambda x: -x["spearman_correlation"]):
        logger.info(
            f"{r['name']:<25} "
            f"{r['spearman_correlation']:>8.4f} "
            f"{r.get('recall_at_10', 0):>8.4f} "
            f"{r.get('recall_at_50', 0):>8.4f} "
            f"{r['transform_time_full_s']:>10.3f}"
        )

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION")
    logger.info("=" * 80)

    logger.info(f"Best correlation: {best_by_corr['name']} (ρ = {best_by_corr['spearman_correlation']:.4f})")
    logger.info(f"Best recall@10:  {best_by_recall['name']} (R@10 = {best_by_recall.get('recall_at_10', 0):.4f})")
    logger.info(f"Fastest:         {fastest['name']} ({fastest['transform_time_full_s']:.3f}s)")

    logger.success("\nFingerprint metrics experiment complete!")


if __name__ == "__main__":
    app()
