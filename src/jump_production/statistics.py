"""Statistical utilities for null model analysis in JUMP production pipeline.

Provides permutation-based null model functions for assessing statistical
significance of activity cliff rates and correlations, plus a Wilson interval.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats

from jump_production.gpu import has_gpu

HAS_GPU = has_gpu()


def permutation_null_cliff_rate(
    struct_sim: np.ndarray,
    morph_sim: np.ndarray,
    struct_threshold: float = 0.7,
    morph_threshold: float = 0.3,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute permutation null for activity cliff rate.

    An activity cliff is defined as a pair with high structural similarity
    (>= struct_threshold) but low morphological similarity (< morph_threshold).

    The null model shuffles morphological similarities to break the
    structure-morphology relationship while preserving marginal distributions.

    Uses CuPy for GPU acceleration when available, falls back to NumPy.

    Args:
        struct_sim: Array of structural similarity values for all pairs
        morph_sim: Array of morphological similarity values for all pairs
        struct_threshold: Threshold for "structurally similar" (default 0.7)
        morph_threshold: Threshold for "morphologically different" (default 0.3)
        n_permutations: Number of permutations for null distribution (default 1000)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Dict with keys:
            - observed_rate: Observed cliff rate
            - null_mean: Mean cliff rate under null
            - null_std: Std of cliff rate under null
            - z_score: (observed - null_mean) / null_std
            - p_value: Two-sided p-value (fraction of null >= |observed - null_mean|)
            - n_permutations: Number of permutations used
    """
    # Try GPU acceleration
    xp = np
    use_cupy = False
    if HAS_GPU:
        import cupy as cp

        xp = cp
        use_cupy = True
        struct_sim = cp.asarray(struct_sim)
        morph_sim = cp.asarray(morph_sim)
        logger.debug("Using CuPy GPU acceleration for permutation test")
    else:
        logger.debug("No GPU available, using NumPy for permutation test")

    # Set seed for reproducibility (legacy API works for both NumPy and CuPy)
    xp.random.seed(random_state)

    # Observed cliff rate
    high_struct_mask = struct_sim >= struct_threshold
    n_high_struct = int(high_struct_mask.sum())

    if n_high_struct == 0:
        return {
            "observed_rate": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "z_score": 0.0,
            "p_value": 1.0,
            "n_permutations": n_permutations,
        }

    # Observed: among high-struct pairs, what fraction has low morph?
    observed_cliffs = int((high_struct_mask & (morph_sim < morph_threshold)).sum())
    observed_rate = observed_cliffs / n_high_struct

    # Null distribution: shuffle morph_sim and recompute
    null_rates = xp.zeros(n_permutations, dtype=xp.float32)
    n_samples = len(morph_sim)

    for i in range(n_permutations):
        # Use legacy permutation API (works for both NumPy and CuPy)
        perm_idx = xp.random.permutation(n_samples)
        morph_sim_shuffled = morph_sim[perm_idx]
        null_cliffs = (high_struct_mask & (morph_sim_shuffled < morph_threshold)).sum()
        null_rates[i] = null_cliffs / n_high_struct

    # Convert back to NumPy if needed
    if use_cupy:
        null_rates = null_rates.get()
        observed_rate = float(observed_rate)

    null_mean = float(np.mean(null_rates))
    null_std = float(np.std(null_rates))

    # Z-score and p-value
    if null_std > 0:
        z_score = (observed_rate - null_mean) / null_std
        # Two-sided p-value
        p_value = float(np.mean(np.abs(null_rates - null_mean) >= np.abs(observed_rate - null_mean)))
    else:
        z_score = 0.0 if observed_rate == null_mean else float("inf") * np.sign(observed_rate - null_mean)
        p_value = 1.0 if observed_rate == null_mean else 0.0

    return {
        "observed_rate": float(observed_rate),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def permutation_null_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 100,
    subsample_size: int | None = 100000,
    random_state: int = 42,
) -> dict:
    """Compute permutation null for Spearman correlation.

    Shuffles y to break the x-y relationship while preserving marginal
    distributions.

    Args:
        x: First variable (e.g., structural similarity)
        y: Second variable (e.g., morphological similarity)
        n_permutations: Number of permutations (default 100)
        subsample_size: If not None, subsample to this size for efficiency
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Dict with keys:
            - observed_r: Observed Spearman correlation
            - null_mean: Mean correlation under null
            - null_std: Std of correlation under null
            - z_score: (observed_r - null_mean) / null_std
            - n_permutations: Number of permutations used
    """
    rng = np.random.default_rng(random_state)

    # Subsample if needed
    if subsample_size is not None and len(x) > subsample_size:
        idx = rng.choice(len(x), size=subsample_size, replace=False)
        x = x[idx]
        y = y[idx]

    # Observed correlation
    observed_r, _ = scipy_stats.spearmanr(x, y)

    # Null distribution
    null_rs = np.zeros(n_permutations)
    y_copy = y.copy()

    for i in range(n_permutations):
        rng.shuffle(y_copy)
        null_rs[i], _ = scipy_stats.spearmanr(x, y_copy)

    null_mean = float(np.mean(null_rs))
    null_std = float(np.std(null_rs))

    # Z-score
    if null_std > 0:
        z_score = (observed_r - null_mean) / null_std
    else:
        z_score = 0.0 if observed_r == null_mean else float("inf") * np.sign(observed_r - null_mean)

    return {
        "observed_r": float(observed_r),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_permutations": n_permutations,
    }


def wilson_bounds(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided Wilson 95% interval for one binomial proportion.

    Notebooks 2.42 and 2.45 and the OpenTargets figure each carry their own copy;
    new callers should import this one so the count stops growing.
    """
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z_value = float(scipy_stats.norm.ppf(0.975))
    rate = successes / total
    denominator = 1 + z_value**2 / total
    center = (rate + z_value**2 / (2 * total)) / denominator
    half_width = z_value * np.sqrt(rate * (1 - rate) / total + z_value**2 / (4 * total**2)) / denominator
    return float(center - half_width), float(center + half_width)
