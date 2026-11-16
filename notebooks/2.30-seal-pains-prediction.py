"""Predict PAINS structural alerts from morphological profiles.

Investigates whether Cell Painting morphological phenotypes can predict
PAINS (Pan-Assay INterference compoundS) structural alerts.

Research Question: Do PAINS compounds produce distinctive morphological signatures?

Uses scaffold splitting for realistic generalization assessment, comparing
against random split to quantify optimistic bias.

Usage:
    # Quick test (10% sample)
    pixi run python notebooks/2.30-seal-pains-prediction.py --sample 0.1

    # Full dataset
    pixi run python notebooks/2.30-seal-pains-prediction.py --sample 1.0

    # Different dataset (e.g., deep learning embeddings)
    pixi run python notebooks/2.30-seal-pains-prediction.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/pains-prediction/{dataset}/
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import typer
from joblib import Parallel, delayed
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import load_profiles
from jump_production.summaries import index_files, write_summary

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "pains-prediction"

# Pre-Harmony parquet retains original CellProfiler feature names in the same order as X_1..X_758
FEATSELECT_URL = "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect.parquet"

N_FOLDS = 5
RANDOM_STATE = 42

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_pains_labels_and_metadata(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load PAINS labels, SMILES, scaffolds, and properties from metadata database."""
    df = con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES,
            Metadata_MurckoScaffold as Scaffold,
            Metadata_HasPAINS as HasPAINS,
            Metadata_MW as MW,
            Metadata_LogP as LogP,
            Metadata_TPSA as TPSA,
            Metadata_median_cell_count as median_cell_count
        FROM compound_metadata
        WHERE Metadata_SMILES IS NOT NULL
          AND Metadata_ValidMol = TRUE
          AND Metadata_HasPAINS IS NOT NULL
    """).df()
    n_pains = df["HasPAINS"].sum()
    n_scaffolds = df["Scaffold"].nunique()
    logger.info(f"Loaded PAINS labels: {len(df):,} compounds, {n_pains:,} PAINS ({n_pains / len(df) * 100:.1f}%)")
    logger.info(f"Unique Murcko scaffolds: {n_scaffolds:,}")
    return df


def align_data(
    adata,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align morphology features with PAINS labels by JCP2022 ID.

    Returns:
        X: Feature matrix (morphology profiles)
        y: Binary labels (HasPAINS)
        scaffolds: Murcko scaffold strings for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline model
        X_cellcount: Median cell count (shape (n, 1)) for cell count baseline
        scaffold_ids: Integer group IDs for scaffold splitting
    """
    # Get JCP2022 IDs from profiles
    profile_jcp = adata.obs["JCP2022"].values

    # Find common JCP2022 IDs
    profile_set = set(profile_jcp)
    metadata_set = set(metadata_df["JCP2022"])
    common_jcp = sorted(profile_set & metadata_set)
    logger.info(f"Common compounds: {len(common_jcp):,}")

    # Create index mappings
    profile_idx = {k: i for i, k in enumerate(profile_jcp)}
    metadata_idx = metadata_df.set_index("JCP2022")

    # Align data
    feature_order = [profile_idx[k] for k in common_jcp]
    X = adata.X[feature_order].astype(np.float32)
    y = metadata_idx.loc[common_jcp, "HasPAINS"].values.astype(np.int32)
    scaffolds = metadata_idx.loc[common_jcp, "Scaffold"].values
    X_props = metadata_idx.loc[common_jcp, ["MW", "LogP", "TPSA"]].values.astype(np.float32)
    X_cellcount = metadata_idx.loc[common_jcp, "median_cell_count"].values.astype(np.float32).reshape(-1, 1)

    # Filter out compounds with zero-norm features or NaN values
    feature_norms = np.linalg.norm(X, axis=1)
    props_valid = ~np.isnan(X_props).any(axis=1)
    cellcount_valid = ~np.isnan(X_cellcount).any(axis=1)
    valid_mask = (feature_norms > 0) & props_valid & cellcount_valid
    logger.info(f"Valid compounds: {valid_mask.sum():,}")

    X = X[valid_mask]
    y = y[valid_mask]
    scaffolds = scaffolds[valid_mask]
    X_props = X_props[valid_mask]
    X_cellcount = X_cellcount[valid_mask]

    # Convert scaffolds to integer group IDs for StratifiedGroupKFold
    unique_scaffolds = sorted(set(scaffolds))  # sorted() -> deterministic group IDs across runs (hash-randomization safe)
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    scaffold_ids = np.array([scaffold_to_id[s] for s in scaffolds])
    logger.info(f"Unique scaffolds in aligned data: {len(unique_scaffolds):,}")

    return X, y, scaffolds, X_props, X_cellcount, scaffold_ids


# =============================================================================
# EDA
# =============================================================================


def run_eda(
    y: np.ndarray,
    X: np.ndarray,
    X_props: np.ndarray,
    scaffolds: np.ndarray,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Run exploratory data analysis on the dataset."""
    logger.info("Running EDA...")

    n_total = len(y)
    n_pains = y.sum()
    n_non_pains = n_total - n_pains
    pct_pains = n_pains / n_total * 100

    eda_stats = {
        "n_total": int(n_total),
        "n_pains": int(n_pains),
        "n_non_pains": int(n_non_pains),
        "pct_pains": float(pct_pains),
    }

    # 1. Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    ax = axes[0]
    bars = ax.bar(["Non-PAINS", "PAINS"], [n_non_pains, n_pains], color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Count")
    ax.set_title("A. Class Distribution")
    for bar, count in zip(bars, [n_non_pains, n_pains]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=9)

    # Pie chart
    ax = axes[1]
    ax.pie(
        [n_non_pains, n_pains],
        labels=["Non-PAINS", "PAINS"],
        autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e"],
    )
    ax.set_title("B. Class Proportion")

    plt.tight_layout()
    fig.savefig(output_dir / "eda_class_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'eda_class_distribution.png'}")

    # 2. Feature distributions: find most discriminative features
    logger.info("Computing discriminative features...")
    pains_mask = y == 1
    non_pains_mask = y == 0

    # Compute effect size (Cohen's d) for each feature
    pains_means = X[pains_mask].mean(axis=0)
    non_pains_means = X[non_pains_mask].mean(axis=0)
    pooled_std = np.sqrt(
        (
            (X[pains_mask].std(axis=0) ** 2) * (pains_mask.sum() - 1)
            + (X[non_pains_mask].std(axis=0) ** 2) * (non_pains_mask.sum() - 1)
        )
        / (n_total - 2)
    )
    # Avoid division by zero
    pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)
    effect_sizes = np.abs(pains_means - non_pains_means) / pooled_std

    # Get top 6 most discriminative features
    top_feature_idx = np.argsort(effect_sizes)[-6:][::-1]
    eda_stats["top_discriminative_features"] = [
        {"index": int(i), "effect_size": float(effect_sizes[i])} for i in top_feature_idx
    ]

    # Plot feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax_idx, feat_idx in enumerate(top_feature_idx):
        ax = axes[ax_idx]
        ax.hist(X[non_pains_mask, feat_idx], bins=50, alpha=0.6, label="Non-PAINS", color="#1f77b4", density=True)
        ax.hist(X[pains_mask, feat_idx], bins=50, alpha=0.6, label="PAINS", color="#ff7f0e", density=True)
        ax.set_xlabel(f"Feature {feat_idx}")
        ax.set_ylabel("Density")
        ax.set_title(f"d = {effect_sizes[feat_idx]:.2f}")
        if ax_idx == 0:
            ax.legend()

    plt.suptitle("Top 6 Discriminative Morphology Features (by effect size)")
    plt.tight_layout()
    fig.savefig(output_dir / "eda_feature_distributions.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'eda_feature_distributions.png'}")

    # 3. Properties vs PAINS
    prop_names = ["MW", "LogP", "TPSA"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax_idx, prop_name in enumerate(prop_names):
        ax = axes[ax_idx]
        ax.hist(X_props[non_pains_mask, ax_idx], bins=50, alpha=0.6, label="Non-PAINS", color="#1f77b4", density=True)
        ax.hist(X_props[pains_mask, ax_idx], bins=50, alpha=0.6, label="PAINS", color="#ff7f0e", density=True)
        ax.set_xlabel(prop_name)
        ax.set_ylabel("Density")
        ax.set_title(f"{prop_name} Distribution")
        if ax_idx == 0:
            ax.legend()

    plt.suptitle("Molecular Properties by PAINS Status")
    plt.tight_layout()
    fig.savefig(output_dir / "eda_properties_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'eda_properties_distribution.png'}")

    # 4. Scaffold analysis (using precomputed Murcko scaffolds)
    logger.info("Computing scaffold statistics...")
    unique_scaffolds = len(set(scaffolds))

    # Scaffolds with PAINS
    pains_scaffolds = set(scaffolds[pains_mask])
    non_pains_scaffolds = set(scaffolds[non_pains_mask])
    shared_scaffolds = len(pains_scaffolds & non_pains_scaffolds)

    eda_stats["scaffold_analysis"] = {
        "unique_scaffolds": int(unique_scaffolds),
        "pains_scaffolds": int(len(pains_scaffolds)),
        "non_pains_scaffolds": int(len(non_pains_scaffolds)),
        "shared_scaffolds": int(shared_scaffolds),
    }
    logger.info(
        f"Unique scaffolds: {unique_scaffolds:,}, PAINS scaffolds: {len(pains_scaffolds):,}, Shared: {shared_scaffolds:,}"
    )

    return eda_stats


# =============================================================================
# Model Training and Evaluation
# =============================================================================


def evaluate_dummy_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate dummy baseline (stratified random prediction)."""
    dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy_clf.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_proba = dummy_clf.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
    }


def train_evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    class_weight: dict | None = None,
) -> tuple[dict, HistGradientBoostingClassifier]:
    """Train HistGradientBoostingClassifier and evaluate with threshold-independent metrics."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
    }, clf


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    X_props: np.ndarray,
    X_cellcount: np.ndarray,
    split_type: str = "scaffold",
    workers: int = 1,
) -> dict:
    """Run cross-validation with specified split type.

    Args:
        X: Feature matrix (morphology profiles)
        y: Binary labels (HasPAINS)
        scaffold_ids: Integer scaffold group IDs for StratifiedGroupKFold
        X_props: Simple properties (MW, LogP, TPSA) for baseline
        X_cellcount: Median cell count (shape (n, 1)) for cell count baseline
        split_type: "scaffold" or "random"
        workers: Number of fits to run side by side, one process each

    Returns:
        Dict with classification metrics across folds
    """
    logger.info(f"Running {N_FOLDS}-fold CV with {split_type} split...")

    # Calculate class imbalance weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    class_weight = "balanced"  # Let sklearn compute weights automatically
    logger.info(f"  Class imbalance: {n_neg:,} neg / {n_pos:,} pos = {n_neg / n_pos:.1f}:1")

    # Choose splitter based on split type (both stratified to handle class imbalance)
    if split_type == "scaffold":
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y, groups=scaffold_ids)
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)

    # Collect metrics across folds
    morphology_metrics = {"roc_auc": [], "avg_precision": []}
    dummy_metrics = {"roc_auc": [], "avg_precision": []}
    cellcount_metrics = {"roc_auc": [], "avg_precision": []}
    props_metrics = {"roc_auc": [], "avg_precision": []}
    combined_metrics = {"roc_auc": [], "avg_precision": []}

    # The fits are independent, so run them side by side, one process each. Every fold fits,
    # in this order: cell count, properties (MW, LogP, TPSA), morphology, morphology + properties.
    folds = list(split_iter)
    feature_sets = [X_cellcount, X_props, X, np.hstack([X, X_props])]
    fits = iter(
        Parallel(n_jobs=workers)(
            delayed(train_evaluate_classifier)(features, y, train_idx, test_idx, class_weight)
            for train_idx, test_idx in folds
            for features in feature_sets
        )
    )

    all_models = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"  Fold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)}, test={len(test_idx)}")

        # 1. Dummy baseline
        dummy_result = evaluate_dummy_baseline(y[train_idx], y[test_idx])
        for k, v in dummy_result.items():
            dummy_metrics[k].append(v)

        # 2. Cell count baseline (median_cell_count only)
        cellcount_result, _ = next(fits)
        for k, v in cellcount_result.items():
            cellcount_metrics[k].append(v)

        # 3. Properties-only model (MW, LogP, TPSA)
        props_result, _ = next(fits)
        for k, v in props_result.items():
            props_metrics[k].append(v)

        # 4. Full model (morphology profiles)
        morphology_result, clf = next(fits)
        for k, v in morphology_result.items():
            morphology_metrics[k].append(v)
        all_models.append(clf)

        # 5. Combined model (morphology + properties)
        combined_result, _ = next(fits)
        for k, v in combined_result.items():
            combined_metrics[k].append(v)

    # Aggregate results
    results = {
        "morphology": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in morphology_metrics.items()},
        "baseline_dummy": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in dummy_metrics.items()},
        "baseline_cellcount": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in cellcount_metrics.items()
        },
        "baseline_properties": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in props_metrics.items()
        },
        "combined": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in combined_metrics.items()},
        "models": all_models,
    }

    # Log results
    logger.info(f"  Dummy ROC-AUC:      {results['baseline_dummy']['roc_auc']['mean']:.3f}")
    logger.info(f"  Cell count ROC-AUC: {results['baseline_cellcount']['roc_auc']['mean']:.3f}")
    logger.info(f"  Props ROC-AUC:      {results['baseline_properties']['roc_auc']['mean']:.3f}")
    logger.info(
        f"  Morphology ROC-AUC: {results['morphology']['roc_auc']['mean']:.3f} +/- {results['morphology']['roc_auc']['std']:.3f}"
    )
    logger.info(
        f"  Combined ROC-AUC:   {results['combined']['roc_auc']['mean']:.3f} +/- {results['combined']['roc_auc']['std']:.3f}"
    )
    logger.info(
        f"  Morphology Avg-PR:  {results['morphology']['avg_precision']['mean']:.3f} +/- {results['morphology']['avg_precision']['std']:.3f}"
    )

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_comparison(
    random_results: dict,
    scaffold_results: dict,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create comparison plot: baselines vs models across split types."""
    logger.info("Creating comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: ROC-AUC comparison across all models (scaffold split)
    ax = axes[0]
    model_keys = ["baseline_dummy", "baseline_cellcount", "baseline_properties", "morphology", "combined"]
    model_labels = ["Dummy", "Cell Count", "Properties", "Morphology", "Morph + Props"]
    colors = ["#999999", "#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]

    scaffold_roc = [scaffold_results[k]["roc_auc"]["mean"] for k in model_keys]
    scaffold_roc_std = [scaffold_results[k]["roc_auc"]["std"] for k in model_keys]

    bars = ax.bar(model_labels, scaffold_roc, yerr=scaffold_roc_std, capsize=3, color=colors)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("A. Model Comparison (Scaffold Split)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, scaffold_roc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9)

    # Panel B: Random vs scaffold split (morphology only)
    ax = axes[1]
    metrics = ["roc_auc", "avg_precision"]
    x = np.arange(len(metrics))
    width = 0.35

    random_means = [random_results["morphology"][m]["mean"] for m in metrics]
    random_stds = [random_results["morphology"][m]["std"] for m in metrics]
    scaffold_means = [scaffold_results["morphology"][m]["mean"] for m in metrics]
    scaffold_stds = [scaffold_results["morphology"][m]["std"] for m in metrics]

    ax.bar(x - width / 2, random_means, width, yerr=random_stds, label="Random Split", capsize=3, color="#1f77b4")
    ax.bar(x + width / 2, scaffold_means, width, yerr=scaffold_stds, label="Scaffold Split", capsize=3, color="#ff7f0e")

    ax.set_ylabel("Score")
    ax.set_title("B. Generalization Gap (Morphology)")
    ax.set_xticks(x)
    ax.set_xticklabels(["ROC-AUC", "Avg Precision"])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    gap = random_results["morphology"]["roc_auc"]["mean"] - scaffold_results["morphology"]["roc_auc"]["mean"]
    ax.annotate(
        f"Gap: {gap:.2f}",
        xy=(0, scaffold_means[0]),
        xytext=(0.5, 0.3),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "comparison_plot.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "comparison_plot.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'comparison_plot.png'}")
    plt.close(fig)


def plot_roc_pr_curves(
    X: np.ndarray,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    clf: HistGradientBoostingClassifier,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot ROC and PR curves for the first scaffold-split fold, using its morphology model `clf`."""
    logger.info("Creating ROC and PR curves...")

    # The first fold of the scaffold cross-validation (80/20 split)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(X, y, groups=scaffold_ids))

    X_test, y_test = X[test_idx], y[test_idx]
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Morphology (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC Curve (Scaffold Split)")
    ax.legend(loc="lower right")

    # PR Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    baseline_precision = y_test.sum() / len(y_test)
    ax.plot(recall, precision, color="#1f77b4", lw=2, label=f"Morphology (AP = {avg_precision:.3f})")
    ax.axhline(y=baseline_precision, color="gray", linestyle="--", label=f"Baseline ({baseline_precision:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("B. Precision-Recall Curve (Scaffold Split)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "roc_pr_curves.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'roc_pr_curves.png'}")
    plt.close(fig)


def get_feature_names() -> list[str]:
    """Fetch CellProfiler feature names from pre-Harmony parquet schema (no data download)."""
    cols = pl.scan_parquet(FEATSELECT_URL).collect_schema().names()
    feat_cols = [c for c in cols if not c.startswith("Metadata_")]
    logger.info(f"Loaded {len(feat_cols)} CellProfiler feature names from parquet schema")
    return feat_cols


def plot_feature_importance(
    models: list[HistGradientBoostingClassifier],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    top_n: int = 20,
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame:
    """Plot and save feature importance from ensemble of models using permutation importance."""
    from sklearn.inspection import permutation_importance

    logger.info("Computing feature importance (permutation-based)...")

    # Map feature indices to names (CellProfiler names if feature count matches, else generic X_N)
    feat_names = get_feature_names()
    if len(feat_names) != X.shape[1]:
        logger.info(f"Feature count mismatch ({len(feat_names)} names vs {X.shape[1]} columns), using generic names")
        feat_names = [f"X_{i + 1}" for i in range(X.shape[1])]

    # Use the first model and a sample of the data for permutation importance
    # This is more reliable than HistGradientBoosting's internal feature importance
    model = models[0]

    # Sample data if too large
    n_sample = min(5000, len(X))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sample, y_sample = X[sample_idx], y[sample_idx]

    # Compute permutation importance
    result = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    mean_importance = result.importances_mean
    std_importance = result.importances_std

    # Create dataframe with CellProfiler feature names
    importance_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(mean_importance)),
            "feature_name": feat_names,
            "importance_mean": mean_importance,
            "importance_std": std_importance,
        }
    )
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Save full CSV
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info(f"Saved: {output_dir / 'feature_importance.csv'}")

    # Plot top N features
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    ax.barh(
        y_pos,
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        align="center",
        color="#1f77b4",
        capsize=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["feature_name"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Morphology Features for PAINS Prediction")

    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "feature_importance.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'feature_importance.png'}")
    plt.close(fig)

    return importance_df


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    random_results: dict,
    scaffold_results: dict,
    eda_stats: dict,
    n_compounds: int,
    n_pains: int,
    n_features: int,
    output_dir: Path,
) -> dict:
    """Generate summary JSON with all results."""
    # Calculate generalization gap
    roc_gap = random_results["morphology"]["roc_auc"]["mean"] - scaffold_results["morphology"]["roc_auc"]["mean"]

    # Calculate overestimation percentage
    if scaffold_results["morphology"]["roc_auc"]["mean"] > 0.5:
        roc_overestimate_pct = (roc_gap / (scaffold_results["morphology"]["roc_auc"]["mean"] - 0.5)) * 100
    else:
        roc_overestimate_pct = 0

    # Get baseline metrics (from scaffold split - more realistic)
    dummy_roc = scaffold_results["baseline_dummy"]["roc_auc"]["mean"]
    cellcount_roc = scaffold_results["baseline_cellcount"]["roc_auc"]["mean"]
    props_roc = scaffold_results["baseline_properties"]["roc_auc"]["mean"]
    morphology_roc = scaffold_results["morphology"]["roc_auc"]["mean"]
    combined_roc = scaffold_results["combined"]["roc_auc"]["mean"]

    # Calculate lift over baselines
    lift_over_dummy = morphology_roc - dummy_roc
    lift_over_cellcount = morphology_roc - cellcount_roc
    lift_over_props = morphology_roc - props_roc
    combined_lift_over_morphology = combined_roc - morphology_roc
    combined_lift_over_props = combined_roc - props_roc

    summary = {
        "task": "PAINS prediction from morphology",
        "research_question": "Can morphological phenotypes predict PAINS structural alerts?",
        "n_compounds": n_compounds,
        "n_pains": n_pains,
        "pct_pains": float(n_pains / n_compounds * 100),
        "n_features": n_features,
        "n_folds": N_FOLDS,
        "model": "HistGradientBoostingClassifier",
        "eda": eda_stats,
        "baselines": {
            "dummy_roc_auc": float(dummy_roc),
            "cellcount_roc_auc": float(cellcount_roc),
            "properties_roc_auc": float(props_roc),
        },
        "results": {
            "random_split": {
                "roc_auc": random_results["morphology"]["roc_auc"]["mean"],
                "roc_auc_std": random_results["morphology"]["roc_auc"]["std"],
                "avg_precision": random_results["morphology"]["avg_precision"]["mean"],
                "avg_precision_std": random_results["morphology"]["avg_precision"]["std"],
            },
            "scaffold_split": {
                "roc_auc": scaffold_results["morphology"]["roc_auc"]["mean"],
                "roc_auc_std": scaffold_results["morphology"]["roc_auc"]["std"],
                "avg_precision": scaffold_results["morphology"]["avg_precision"]["mean"],
                "avg_precision_std": scaffold_results["morphology"]["avg_precision"]["std"],
            },
            "combined_random_split": {
                "roc_auc": random_results["combined"]["roc_auc"]["mean"],
                "roc_auc_std": random_results["combined"]["roc_auc"]["std"],
                "avg_precision": random_results["combined"]["avg_precision"]["mean"],
                "avg_precision_std": random_results["combined"]["avg_precision"]["std"],
            },
            "combined_scaffold_split": {
                "roc_auc": scaffold_results["combined"]["roc_auc"]["mean"],
                "roc_auc_std": scaffold_results["combined"]["roc_auc"]["std"],
                "avg_precision": scaffold_results["combined"]["avg_precision"]["mean"],
                "avg_precision_std": scaffold_results["combined"]["avg_precision"]["std"],
            },
        },
        "generalization_gap": {
            "roc_auc_drop": float(roc_gap),
            "interpretation": f"Random split overestimates by {roc_overestimate_pct:.0f}% of signal",
        },
        "lift_over_baselines": {
            "over_dummy": float(lift_over_dummy),
            "over_cellcount": float(lift_over_cellcount),
            "over_properties": float(lift_over_props),
            "interpretation": f"Morphology adds {lift_over_cellcount:.3f} ROC-AUC over cell count, {lift_over_props:.3f} over properties",
        },
        "combined_lift": {
            "combined_roc_auc": float(combined_roc),
            "over_morphology": float(combined_lift_over_morphology),
            "over_properties": float(combined_lift_over_props),
            "interpretation": f"Combined (morph+props) adds {combined_lift_over_morphology:+.3f} over morphology, {combined_lift_over_props:+.3f} over properties",
        },
        "conclusion": "",
    }

    # Generate conclusion
    if morphology_roc > 0.6:
        summary["conclusion"] = f"Morphology profiles show predictive signal for PAINS (ROC-AUC={morphology_roc:.3f})"
    elif morphology_roc > 0.55:
        summary["conclusion"] = f"Weak predictive signal for PAINS from morphology (ROC-AUC={morphology_roc:.3f})"
    else:
        summary["conclusion"] = f"Limited predictive signal for PAINS from morphology (ROC-AUC={morphology_roc:.3f})"

    # Log key findings
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Compounds: {n_compounds:,} ({n_pains:,} PAINS, {n_pains / n_compounds * 100:.1f}%)")
    logger.info("Classification (ROC-AUC, scaffold split):")
    logger.info(f"  Dummy:       {dummy_roc:.3f}")
    logger.info(f"  Cell count:  {cellcount_roc:.3f}")
    logger.info(f"  Properties:  {props_roc:.3f}")
    logger.info(f"  Morphology:  {morphology_roc:.3f}")
    logger.info(f"  Combined:    {combined_roc:.3f}")
    logger.info(f"  Lift (cell count): +{lift_over_cellcount:.3f}")
    logger.info(f"  Lift (props): +{lift_over_props:.3f}")
    logger.info(f"  Combined lift over morphology: {combined_lift_over_morphology:+.3f}")
    logger.info(f"  Combined lift over properties: {combined_lift_over_props:+.3f}")
    logger.info(f"Generalization gap: {roc_gap:.3f}")
    logger.info(f"Conclusion: {summary['conclusion']}")

    # Save standardized summary envelope
    write_summary(
        output_dir / "summary.json",
        description=(
            "Predicts PAINS (Pan-Assay INterference compoundS) structural alerts from Cell Painting "
            "morphological profiles using a HistGradientBoostingClassifier, testing whether PAINS "
            "compounds produce distinctive morphological signatures. Reports cross-validated ROC-AUC "
            "and average precision under both scaffold and random splits (to quantify optimistic bias), "
            "with dummy, cell-count, and molecular-property baselines plus a combined morphology + "
            "properties model."
        ),
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
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
    skip_eda: bool = typer.Option(False, "--skip-eda", help="Skip EDA plots"),
    workers: int = typer.Option(1, "--workers", help="Model fits to run side by side, one process each"),
) -> None:
    """Predict PAINS structural alerts from Cell Painting morphological profiles."""
    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Check database exists
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Load data
        logger.info("Loading data...")
        adata = load_profiles(dataset, level="perturbation")
        metadata_df = load_pains_labels_and_metadata(metadata_con)

        # 2. Align data
        X, y, scaffolds, X_props, X_cellcount, scaffold_ids = align_data(adata, metadata_df)

        # 3. Sample data if requested
        if sample_frac < 1.0:
            n_sample = int(len(X) * sample_frac)
            rng = np.random.default_rng(RANDOM_STATE)
            sample_idx = rng.choice(len(X), size=n_sample, replace=False)
            X = X[sample_idx]
            y = y[sample_idx]
            scaffolds = scaffolds[sample_idx]
            X_props = X_props[sample_idx]
            X_cellcount = X_cellcount[sample_idx]
            # Recompute scaffold_ids after sampling
            unique_scaffolds = list(set(scaffolds))
            scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
            scaffold_ids = np.array([scaffold_to_id[s] for s in scaffolds])
            logger.info(f"Sampled {sample_frac:.0%} of data: {n_sample:,} compounds")

        n_compounds = len(X)
        n_pains = int(y.sum())

        logger.info(f"Final dataset: {n_compounds:,} compounds, {n_pains:,} PAINS ({n_pains / n_compounds * 100:.1f}%)")

        # 4. Run EDA
        if not skip_eda:
            eda_stats = run_eda(y, X, X_props, scaffolds, output_dir, dpi)
        else:
            eda_stats = {"n_total": n_compounds, "n_pains": n_pains}

        # 5. Run cross-validation with both split types
        random_results = run_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, "random", workers)
        scaffold_results = run_cross_validation(X, y, scaffold_ids, X_props, X_cellcount, "scaffold", workers)

        # 6. Save detailed results (without models)
        random_results_save = {k: v for k, v in random_results.items() if k != "models"}
        scaffold_results_save = {k: v for k, v in scaffold_results.items() if k != "models"}

        with open(output_dir / "random_split_results.json", "w") as f:
            json.dump(random_results_save, f, indent=2)
        with open(output_dir / "scaffold_split_results.json", "w") as f:
            json.dump(scaffold_results_save, f, indent=2)

        # 7. Create plots
        plot_comparison(random_results, scaffold_results, output_dir, dpi)
        plot_roc_pr_curves(X, y, scaffold_ids, scaffold_results["models"][0], output_dir, dpi)

        # 8. Feature importance (use scaffold split models)
        plot_feature_importance(scaffold_results["models"], X, y, output_dir, dpi=dpi)

        # 9. Generate summary
        generate_summary(
            random_results,
            scaffold_results,
            eda_stats,
            n_compounds,
            n_pains,
            X.shape[1],
            output_dir,
        )

        logger.success(f"PAINS prediction complete: {output_dir}")

    finally:
        metadata_con.close()


if __name__ == "__main__":
    app()


# =============================================================================
# Future Refactoring Note
# =============================================================================
