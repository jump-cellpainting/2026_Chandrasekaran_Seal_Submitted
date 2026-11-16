"""Predict DILI, DICT, MitoTox, and PK endpoints from morphology, fingerprints, or both.

Compares three feature modalities per endpoint:
- Morphology: Cell Painting profiles
- Fingerprint: Morgan ECFP4 (2048-bit, radius 2) from precomputed npz
- Combined: Concatenation of morphology + fingerprint features

Toxicity and PK labels come from the augmented metadata database (pre-matched to
JUMP compounds). Same CV fold splits are used across modalities for fair comparison.

Requires:
    data/interim/jump_metadata_augmented.duckdb (from augment_metadata_db)
    data/interim/compound_featurization/morgan_fp.npz (from compound_properties.py)

Usage:
    # Full run
    pixi run -e cheminformatics python notebooks/2.34-seal-toxicity-pk-prediction.py

    # Skip plots for faster iteration
    pixi run -e cheminformatics python notebooks/2.34-seal-toxicity-pk-prediction.py --skip-plots

    # Different dataset (e.g., deep learning embeddings)
    pixi run -e cheminformatics python notebooks/2.34-seal-toxicity-pk-prediction.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/toxicity-pk-prediction/{dataset}/
    - summary.json: All results nested by endpoint and modality
    - classification_results.csv: Fold results with 'modality' column
    - regression_results.csv: Fold results with 'modality' column
    - roc_{task}.png: Three ROC curves per classification task
    - scatter_{target}.png: 1x3 scatter grid per regression target
    - summary_comparison.png: Bar chart with morphology/fingerprint/combined/baseline
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from loguru import logger
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import load_profiles
from jump_production.summaries import index_files, write_summary

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "toxicity-pk-prediction"
MORGAN_FP_FILE = INTERIM_DATA_DIR / "compound_featurization" / "morgan_fp.npz"

# PK regression targets
PK_TARGETS = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]
PK_TARGET_NAMES = {
    "human_VDss_L_kg": "VDss (L/kg)",
    "human_CL_mL_min_kg": "CL (mL/min/kg)",
    "human_fup": "fup",
    "human_mrt": "MRT (h)",
    "human_thalf": "t½ (h)",
}
# Targets to predict in log scale (all except fup which is bounded 0-1)
PK_LOG_SCALE_TARGETS = {"human_VDss_L_kg", "human_CL_mL_min_kg", "human_mrt", "human_thalf"}

N_FOLDS = 5
RANDOM_STATE = 42
MODALITIES = ["morphology", "fingerprint", "combined"]
MODALITY_COLORS = {"morphology": "#1f77b4", "fingerprint": "#2ca02c", "combined": "#d62728"}
MODALITY_LABELS = {"morphology": "Morphology", "fingerprint": "Fingerprint", "combined": "Combined"}

app = typer.Typer()


# =============================================================================
# Data Loading (all from augmented metadata database)
# =============================================================================


def load_dict_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load DICT labels from augmented database (already matched to JUMP).

    Positive = 'less' or 'most' concern, Negative = 'no' concern.
    Ambiguous entries are excluded.
    """
    df = con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               CASE WHEN Metadata_dict_concern IN ('less', 'most') THEN 1 ELSE 0 END AS label
        FROM toxicity_pk_annotations
        WHERE Metadata_dict_concern IN ('less', 'most', 'no')
    """).df()
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"DICT from database: {len(df):,} compounds ({n_pos} positive, {n_neg} negative)")
    return df


def load_dili_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load DILI labels from augmented database (already matched to JUMP).

    Positive = 'less' or 'most' concern, Negative = 'no' concern.
    Ambiguous entries are excluded.
    """
    df = con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               CASE WHEN Metadata_dili_concern IN ('less', 'most') THEN 1 ELSE 0 END AS label
        FROM toxicity_pk_annotations
        WHERE Metadata_dili_concern IN ('less', 'most', 'no')
    """).df()
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"DILI from database: {len(df):,} compounds ({n_pos} positive, {n_neg} negative)")
    return df


def load_mitotox_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load MitoTox labels from augmented database (already matched to JUMP)."""
    df = con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022, Metadata_mitotox_toxic AS label
        FROM mitotox_annotations
    """).df()
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"MitoTox from database: {len(df):,} compounds ({n_pos} toxic, {n_neg} non-toxic)")
    return df


def load_pk_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load PK data from augmented database (already matched to JUMP)."""
    df = con.execute("""
        SELECT Metadata_JCP2022 AS JCP2022,
               Metadata_pk_vdss_l_kg AS human_VDss_L_kg,
               Metadata_pk_cl_ml_min_kg AS human_CL_mL_min_kg,
               Metadata_pk_fup AS human_fup,
               Metadata_pk_mrt_h AS human_mrt,
               Metadata_pk_thalf_h AS human_thalf
        FROM toxicity_pk_annotations
        WHERE Metadata_pk_vdss_l_kg IS NOT NULL
           OR Metadata_pk_cl_ml_min_kg IS NOT NULL
           OR Metadata_pk_fup IS NOT NULL
           OR Metadata_pk_mrt_h IS NOT NULL
           OR Metadata_pk_thalf_h IS NOT NULL
    """).df()
    logger.info(f"PK from database: {len(df):,} compounds with any PK data")
    return df


def load_morgan_fingerprints() -> dict[str, np.ndarray]:
    """Load precomputed Morgan fingerprints (2048-bit, radius 2) from npz.

    Returns:
        Dict mapping JCP2022 ID -> fingerprint array (2048,) for valid molecules.
    """
    data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    jcp_ids = data["jcp2022"]
    fps = data["fingerprints"]
    valid_mask = data["valid_mask"]

    valid_fps = fps[valid_mask].astype(np.float32)
    valid_ids = jcp_ids[valid_mask]

    fp_dict = {jcp: valid_fps[i] for i, jcp in enumerate(valid_ids)}
    logger.info(f"Loaded Morgan FPs: {len(fp_dict):,} valid compounds (of {len(jcp_ids):,} total)")
    return fp_dict


def get_features_multi(
    adata,
    matched_df: pd.DataFrame,
    value_col: str,
    fp_dict: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Extract features for morphology, fingerprint, and combined modalities.

    Uses the intersection of compounds with both morphology profiles and valid
    fingerprints. All three modalities use the SAME compound set and ordering,
    enabling direct fold-to-fold comparison.

    Returns:
        Dict with keys "morphology", "fingerprint", "combined", each mapping to
        (X, y, jcp_ids) tuple. Empty dict if no compounds match.
    """
    profile_jcp = adata.obs["JCP2022"].values
    profile_idx = {k: i for i, k in enumerate(profile_jcp)}

    matched_jcp = list(matched_df["JCP2022"])
    matched_idx = matched_df.set_index("JCP2022")

    # Intersection: must have both profile AND valid fingerprint
    common_jcp = [jcp for jcp in matched_jcp if jcp in profile_idx and jcp in fp_dict]

    n_missing_fp = sum(1 for jcp in matched_jcp if jcp in profile_idx and jcp not in fp_dict)
    if n_missing_fp > 0:
        logger.warning(f"  {n_missing_fp} compounds have profiles but no valid fingerprint — excluded")
    logger.info(f"  Common compound set: {len(common_jcp):,} / {len(matched_jcp):,}")

    if not common_jcp:
        return {}

    # Extract morphology
    morph_indices = [profile_idx[jcp] for jcp in common_jcp]
    X_morph = adata.X[morph_indices].astype(np.float32)

    # Extract fingerprints
    X_fp = np.stack([fp_dict[jcp] for jcp in common_jcp], axis=0)

    # Combined
    X_combined = np.concatenate([X_morph, X_fp], axis=1)

    # Labels/targets
    y = matched_idx.loc[common_jcp, value_col].values
    if value_col == "label":
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)

    # Filter compounds with zero-norm morphology (rare edge case)
    morph_norms = np.linalg.norm(X_morph, axis=1)
    valid_mask = morph_norms > 0
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"  Removed {n_invalid} compounds with zero-norm morphology features")
        X_morph = X_morph[valid_mask]
        X_fp = X_fp[valid_mask]
        X_combined = X_combined[valid_mask]
        y = y[valid_mask]
        common_jcp = [jcp for jcp, v in zip(common_jcp, valid_mask) if v]

    return {
        "morphology": (X_morph, y, common_jcp),
        "fingerprint": (X_fp, y, common_jcp),
        "combined": (X_combined, y, common_jcp),
    }


# =============================================================================
# Classification Model Training
# =============================================================================


def train_classification_fold(
    X: np.ndarray,
    y: np.ndarray,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit and score one CV fold. Returns its metrics, test labels, and model and baseline scores."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Model
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=0,
    )
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Baseline
    baseline = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    y_pred_baseline_proba = baseline.predict_proba(X_test)[:, 1]
    y_pred_baseline = baseline.predict(X_test)

    # Metrics for this fold
    fold_result = {
        "fold": fold_idx + 1,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        # Model metrics
        "model_auc": roc_auc_score(y_test, y_pred_proba),
        "model_ba": balanced_accuracy_score(y_test, y_pred),
        "model_f1": f1_score(y_test, y_pred),
        "model_precision": precision_score(y_test, y_pred, zero_division=0),
        "model_recall": recall_score(y_test, y_pred, zero_division=0),
        # Baseline metrics
        "baseline_auc": roc_auc_score(y_test, y_pred_baseline_proba),
        "baseline_ba": balanced_accuracy_score(y_test, y_pred_baseline),
        "baseline_f1": f1_score(y_test, y_pred_baseline),
    }
    return fold_result, y_test, y_pred_proba, y_pred_baseline_proba


def train_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_FOLDS,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
    workers: int = 1,
) -> dict:
    """Train classification model with stratified K-fold CV.

    Args:
        X: Feature matrix.
        y: Binary labels.
        n_splits: Number of CV folds (ignored if fold_indices provided).
        fold_indices: Pre-computed fold splits for cross-modality comparison.
        workers: Number of folds to fit side by side, one process each.

    Returns dict with per-fold metrics for model and baseline.
    """
    if fold_indices is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_indices = list(skf.split(X, y))

    fold_results = []
    all_y_true = []
    all_y_pred_model = []
    all_y_pred_baseline = []

    folds = Parallel(n_jobs=workers)(
        delayed(train_classification_fold)(X, y, fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(fold_indices)
    )
    for fold_idx, (fold_result, y_test, y_pred_proba, y_pred_baseline_proba) in enumerate(folds):
        fold_results.append(fold_result)

        # Collect for overall ROC curve
        all_y_true.extend(y_test)
        all_y_pred_model.extend(y_pred_proba)
        all_y_pred_baseline.extend(y_pred_baseline_proba)

        logger.info(f"    Fold {fold_idx + 1}: AUC={fold_result['model_auc']:.3f}, BA={fold_result['model_ba']:.3f}")

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    summary = {
        "model_auc_mean": results_df["model_auc"].mean(),
        "model_auc_std": results_df["model_auc"].std(),
        "model_ba_mean": results_df["model_ba"].mean(),
        "model_ba_std": results_df["model_ba"].std(),
        "model_f1_mean": results_df["model_f1"].mean(),
        "model_f1_std": results_df["model_f1"].std(),
        "baseline_auc_mean": results_df["baseline_auc"].mean(),
        "baseline_ba_mean": results_df["baseline_ba"].mean(),
    }

    logger.info(f"    Mean AUC: {summary['model_auc_mean']:.3f} ± {summary['model_auc_std']:.3f}")

    return {
        "fold_results": results_df,
        "summary": summary,
        "y_true": np.array(all_y_true),
        "y_pred_model": np.array(all_y_pred_model),
        "y_pred_baseline": np.array(all_y_pred_baseline),
    }


# =============================================================================
# Regression Model Training
# =============================================================================


def train_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_splits: int = N_FOLDS,
    use_log_scale: bool = False,
    fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Train regression model with K-fold CV.

    Args:
        X: Feature matrix.
        y: Target values (original scale).
        target_name: Name of the target for logging.
        n_splits: Number of CV folds (ignored if fold_indices provided).
        use_log_scale: If True, train on log(y) and evaluate on original scale.
        fold_indices: Pre-computed fold splits for cross-modality comparison.

    Returns dict with per-fold metrics for model and baseline.
    """
    # Apply log transformation if requested
    if use_log_scale:
        # Filter out non-positive values before log transform
        valid_mask = y > 0
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"Removing {n_invalid} non-positive values for log transform")
            X = X[valid_mask]
            y = y[valid_mask]
            if fold_indices is not None:
                # Remap fold indices to filtered array
                old_to_new = np.full(len(valid_mask), -1, dtype=int)
                old_to_new[valid_mask] = np.arange(valid_mask.sum())
                fold_indices = [
                    (
                        old_to_new[ti[ti < len(valid_mask)]][old_to_new[ti[ti < len(valid_mask)]] >= 0],
                        old_to_new[te[te < len(valid_mask)]][old_to_new[te[te < len(valid_mask)]] >= 0],
                    )
                    for ti, te in fold_indices
                ]
        y_transformed = np.log(y)
        logger.info(f"    Using log scale for {target_name}")
    else:
        y_transformed = y

    if fold_indices is None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        fold_indices = list(kf.split(X))

    fold_results = []
    all_y_true = []
    all_y_pred_model = []
    all_y_pred_baseline = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_t, _y_test_t = y_transformed[train_idx], y_transformed[test_idx]
        y_test_orig = y[test_idx]  # Original scale for evaluation

        # Model (trained on transformed scale)
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        model.fit(X_train, y_train_t)
        y_pred_t = model.predict(X_test)

        # Baseline (mean predictor, also on transformed scale)
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train_t)
        y_pred_baseline_t = baseline.predict(X_test)

        # Transform predictions back to original scale for evaluation
        if use_log_scale:
            y_pred = np.exp(y_pred_t)
            y_pred_baseline = np.exp(y_pred_baseline_t)
        else:
            y_pred = y_pred_t
            y_pred_baseline = y_pred_baseline_t

        # Metrics for this fold (evaluated on original scale)
        fold_result = {
            "fold": fold_idx + 1,
            "target": target_name,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "log_scale": use_log_scale,
            # Model metrics
            "model_r2": r2_score(y_test_orig, y_pred),
            "model_rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred)),
            "model_mae": mean_absolute_error(y_test_orig, y_pred),
            "model_spearman": spearmanr(y_test_orig, y_pred).correlation,
            # Baseline metrics
            "baseline_r2": r2_score(y_test_orig, y_pred_baseline),
            "baseline_rmse": np.sqrt(mean_squared_error(y_test_orig, y_pred_baseline)),
            "baseline_mae": mean_absolute_error(y_test_orig, y_pred_baseline),
        }
        fold_results.append(fold_result)

        # Collect for scatter plot (log scale if used, for visualization)
        if use_log_scale:
            all_y_true.extend(np.log(y_test_orig))
            all_y_pred_model.extend(y_pred_t)  # Already in log scale
            all_y_pred_baseline.extend(y_pred_baseline_t)
        else:
            all_y_true.extend(y_test_orig)
            all_y_pred_model.extend(y_pred)
            all_y_pred_baseline.extend(y_pred_baseline)

        logger.info(f"    Fold {fold_idx + 1}: R²={fold_result['model_r2']:.3f}, ρ={fold_result['model_spearman']:.3f}")

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    summary = {
        "model_r2_mean": results_df["model_r2"].mean(),
        "model_r2_std": results_df["model_r2"].std(),
        "model_rmse_mean": results_df["model_rmse"].mean(),
        "model_mae_mean": results_df["model_mae"].mean(),
        "model_spearman_mean": results_df["model_spearman"].mean(),
        "model_spearman_std": results_df["model_spearman"].std(),
        "baseline_r2_mean": results_df["baseline_r2"].mean(),
        "baseline_rmse_mean": results_df["baseline_rmse"].mean(),
    }

    logger.info(f"    Mean R²: {summary['model_r2_mean']:.3f} ± {summary['model_r2_std']:.3f}")
    logger.info(f"    Mean Spearman: {summary['model_spearman_mean']:.3f} ± {summary['model_spearman_std']:.3f}")

    return {
        "fold_results": results_df,
        "summary": summary,
        "y_true": np.array(all_y_true),
        "y_pred_model": np.array(all_y_pred_model),
        "y_pred_baseline": np.array(all_y_pred_baseline),
        "use_log_scale": use_log_scale,
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_classification_roc(
    modality_results: dict[str, dict],
    task_name: str,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot ROC curves for all modalities for a classification task."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for modality in MODALITIES:
        if modality not in modality_results:
            continue
        results = modality_results[modality]
        fpr, tpr, _ = roc_curve(results["y_true"], results["y_pred_model"])
        auc = results["summary"]["model_auc_mean"]
        auc_std = results["summary"]["model_auc_std"]
        ax.plot(
            fpr,
            tpr,
            color=MODALITY_COLORS[modality],
            lw=2,
            label=f"{MODALITY_LABELS[modality]} (AUC = {auc:.3f} ± {auc_std:.3f})",
        )

    # Baseline (same for all modalities)
    first = next(iter(modality_results.values()))
    baseline_auc = first["summary"]["baseline_auc_mean"]
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label=f"Baseline (AUC = {baseline_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{task_name} Prediction — ROC Curves by Modality")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    filename = f"roc_{task_name.lower()}.png"
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / filename}")


def plot_regression_scatter(
    modality_results: dict[str, dict],
    target_name: str,
    target_display: str,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot predicted vs actual scatter for all modalities (1x3 grid)."""
    active_modalities = [m for m in MODALITIES if m in modality_results]
    n = len(active_modalities)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, modality in zip(axes, active_modalities):
        results = modality_results[modality]
        y_true = results["y_true"]
        y_pred = results["y_pred_model"]
        r2 = results["summary"]["model_r2_mean"]
        spearman_val = results["summary"]["model_spearman_mean"]
        use_log_scale = results.get("use_log_scale", False)

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=MODALITY_COLORS[modality])

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nρ = {spearman_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        prefix = "log " if use_log_scale else ""
        ax.set_xlabel(f"Actual {prefix}{target_display}")
        ax.set_ylabel(f"Predicted {prefix}{target_display}")
        ax.set_title(MODALITY_LABELS[modality])

    fig.suptitle(f"{target_display} Prediction by Modality", fontsize=12)
    plt.tight_layout()
    filename = f"scatter_{target_name}.png"
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / filename}")


def plot_summary_bar(
    classification_results: dict,
    regression_results: dict,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create summary bar chart comparing modalities and baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.2

    # Classification summary
    ax = axes[0]
    tasks = list(classification_results.keys())
    x = np.arange(len(tasks))

    for i, modality in enumerate(MODALITIES):
        aucs = [classification_results[t][modality]["summary"]["model_auc_mean"] for t in tasks]
        stds = [classification_results[t][modality]["summary"]["model_auc_std"] for t in tasks]
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            aucs,
            width,
            yerr=stds,
            label=MODALITY_LABELS[modality],
            capsize=3,
            color=MODALITY_COLORS[modality],
        )

    baseline_aucs = [classification_results[t]["morphology"]["summary"]["baseline_auc_mean"] for t in tasks]
    ax.bar(x + 1.5 * width, baseline_aucs, width, label="Baseline", color="gray", alpha=0.7)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Classification Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1])

    # Regression summary
    ax = axes[1]
    targets = list(regression_results.keys())
    x = np.arange(len(targets))

    for i, modality in enumerate(MODALITIES):
        r2s = [regression_results[t][modality]["summary"]["model_r2_mean"] for t in targets]
        stds = [regression_results[t][modality]["summary"]["model_r2_std"] for t in targets]
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            r2s,
            width,
            yerr=stds,
            label=MODALITY_LABELS[modality],
            capsize=3,
            color=MODALITY_COLORS[modality],
        )

    baseline_r2s = [regression_results[t]["morphology"]["summary"]["baseline_r2_mean"] for t in targets]
    ax.bar(x + 1.5 * width, baseline_r2s, width, label="Baseline", color="gray", alpha=0.7)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("R²")
    ax.set_title("Regression Tasks (PK Endpoints)")
    ax.set_xticks(x)
    ax.set_xticklabels([PK_TARGET_NAMES.get(t, t) for t in targets], rotation=45, ha="right")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "summary_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'summary_comparison.png'}")


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    classification_results: dict,
    regression_results: dict,
    matching_stats: dict,
    output_dir: Path,
) -> dict:
    """Generate summary JSON with all results nested by modality."""
    numbers = {
        "task": "DILI/DICT/MitoTox/PK prediction: morphology vs fingerprint vs combined",
        "research_question": "Do morphology, chemical structure, or their combination best predict tox/PK endpoints?",
        "method": "5-fold stratified CV for classification, 5-fold CV for regression",
        "model": "HistGradientBoostingClassifier/Regressor",
        "modalities": {
            "morphology": "Cell Painting profiles",
            "fingerprint": "Morgan ECFP4 (2048-bit, radius 2)",
            "combined": "Morphology + Fingerprint concatenation",
        },
        "compound_matching": {
            "method": "Augmented metadata database (pre-matched via InChIKey)",
            "stats": matching_stats,
        },
        "classification": {},
        "regression": {},
    }

    # Classification results
    for task_name, modality_results in classification_results.items():
        first_modality = next(iter(modality_results.values()))
        n_compounds = int(
            first_modality["fold_results"]["n_train"].iloc[0] + first_modality["fold_results"]["n_test"].iloc[0]
        )

        task_summary = {"n_compounds": n_compounds}

        # Per-modality metrics
        best_auc = -1
        best_mod = None
        for modality, results in modality_results.items():
            s = results["summary"]
            task_summary[modality] = {
                "auc_mean": float(s["model_auc_mean"]),
                "auc_std": float(s["model_auc_std"]),
                "balanced_accuracy_mean": float(s["model_ba_mean"]),
                "f1_mean": float(s["model_f1_mean"]),
            }
            if s["model_auc_mean"] > best_auc:
                best_auc = s["model_auc_mean"]
                best_mod = modality

        # Baseline (same across modalities)
        baseline_s = first_modality["summary"]
        task_summary["baseline"] = {
            "auc_mean": float(baseline_s["baseline_auc_mean"]),
            "balanced_accuracy_mean": float(baseline_s["baseline_ba_mean"]),
        }
        task_summary["best_modality"] = best_mod

        numbers["classification"][task_name] = task_summary

    # Regression results
    for target_name, modality_results in regression_results.items():
        first_modality = next(iter(modality_results.values()))
        n_compounds = int(
            first_modality["fold_results"]["n_train"].iloc[0] + first_modality["fold_results"]["n_test"].iloc[0]
        )

        target_summary = {"n_compounds": n_compounds}

        best_r2 = -np.inf
        best_mod = None
        for modality, results in modality_results.items():
            s = results["summary"]
            target_summary[modality] = {
                "r2_mean": float(s["model_r2_mean"]),
                "r2_std": float(s["model_r2_std"]),
                "rmse_mean": float(s["model_rmse_mean"]),
                "mae_mean": float(s["model_mae_mean"]),
                "spearman_mean": float(s["model_spearman_mean"]),
            }
            if s["model_r2_mean"] > best_r2:
                best_r2 = s["model_r2_mean"]
                best_mod = modality

        baseline_s = first_modality["summary"]
        target_summary["baseline"] = {
            "r2_mean": float(baseline_s["baseline_r2_mean"]),
            "rmse_mean": float(baseline_s["baseline_rmse_mean"]),
        }
        target_summary["best_modality"] = best_mod

        numbers["regression"][target_name] = target_summary

    # Generate conclusions
    conclusions = []

    for task_name, data in numbers["classification"].items():
        best = data["best_modality"]
        parts = []
        for m in MODALITIES:
            if m in data:
                parts.append(f"{m}={data[m]['auc_mean']:.3f}")
        conclusions.append(f"{task_name}: best={best} (AUC={data[best]['auc_mean']:.3f}), {', '.join(parts)}")

    for target_name, data in numbers["regression"].items():
        best = data["best_modality"]
        parts = []
        for m in MODALITIES:
            if m in data:
                parts.append(f"{m}={data[m]['r2_mean']:.3f}")
        conclusions.append(f"{target_name}: best={best} (R²={data[best]['r2_mean']:.3f}), {', '.join(parts)}")

    numbers["conclusions"] = conclusions

    # Save
    description = (
        "Predicts toxicity (DILI, DICT, MitoTox) and pharmacokinetic endpoints from Cell Painting "
        "morphology, Morgan ECFP4 fingerprints, or their combination, comparing all three modalities "
        "per endpoint with 5-fold cross-validation against a dummy baseline."
    )
    summary = write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=numbers,
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
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
    n_splits: int = typer.Option(N_FOLDS, "--n-splits", help="Number of CV folds"),
    workers: int = typer.Option(1, "--workers", help="Classification folds to fit side by side, one process each"),
) -> None:
    """Predict DILI, DICT, MitoTox, and PK endpoints using morphology, fingerprints, or both."""
    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Check required files
    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    if not MORGAN_FP_FILE.exists():
        logger.error(f"Morgan fingerprint file not found: {MORGAN_FP_FILE}")
        raise typer.Exit(1)

    # Connect to metadata DB
    metadata_con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 1. Load Cell Painting profiles
        logger.info("Loading Cell Painting profiles...")
        adata = load_profiles(dataset, level="perturbation")
        logger.info(f"Loaded profiles: {adata.n_obs:,} compounds, {adata.n_vars:,} features")

        # 2. Load Morgan fingerprints
        logger.info("Loading Morgan fingerprints...")
        fp_dict = load_morgan_fingerprints()

        # 3. Load labels from augmented database (already matched to JUMP)
        logger.info("\n=== Loading Labels from Database ===")

        logger.info("\n--- DICT ---")
        dict_matched = load_dict_data(metadata_con)

        logger.info("\n--- DILI ---")
        dili_matched = load_dili_data(metadata_con)

        logger.info("\n--- MitoTox ---")
        mitotox_matched = load_mitotox_data(metadata_con)

        logger.info("\n--- PK ---")
        pk_df = load_pk_data(metadata_con)

        # Track matching stats
        matching_stats = {
            "DICT": {"matched": len(dict_matched)},
            "DILI": {"matched": len(dili_matched)},
            "MitoTox": {"matched": len(mitotox_matched)},
            "PK": {"matched": len(pk_df)},
        }

        # Save matched compounds
        all_matched = []
        for name, df in [("DICT", dict_matched), ("DILI", dili_matched), ("MitoTox", mitotox_matched)]:
            df_save = df[["JCP2022", "label"]].copy()
            df_save["dataset"] = name
            all_matched.append(df_save)

        matched_df = pd.concat(all_matched, ignore_index=True)
        matched_df.to_csv(output_dir / "matched_compounds.csv", index=False)
        logger.info(f"Saved: {output_dir / 'matched_compounds.csv'}")

        # 5. Classification tasks
        logger.info("\n=== Classification Models ===")
        classification_results = {}

        for task_name, task_matched in [("DICT", dict_matched), ("DILI", dili_matched), ("MitoTox", mitotox_matched)]:
            logger.info(f"\n--- {task_name} ---")
            modality_features = get_features_multi(adata, task_matched, "label", fp_dict)
            if not modality_features:
                logger.warning(f"No {task_name} compounds matched, skipping")
                continue

            X_morph, y, _ = modality_features["morphology"]
            logger.info(f"  Training on {len(y):,} compounds (all modalities)")

            # Shared fold indices across modalities
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            fold_indices = list(skf.split(X_morph, y))

            classification_results[task_name] = {}
            for modality in MODALITIES:
                logger.info(f"  Modality: {modality}")
                X, y_mod, _ = modality_features[modality]
                classification_results[task_name][modality] = train_classification_cv(
                    X,
                    y_mod,
                    n_splits,
                    fold_indices=fold_indices,
                    workers=workers,
                )

        # Save classification fold results
        if classification_results:
            all_class_folds = []
            for task_name, modality_results in classification_results.items():
                for modality, results in modality_results.items():
                    df = results["fold_results"].copy()
                    df["task"] = task_name
                    df["modality"] = modality
                    all_class_folds.append(df)
            class_folds_df = pd.concat(all_class_folds, ignore_index=True)
            class_folds_df.to_csv(output_dir / "classification_results.csv", index=False)
            logger.info(f"Saved: {output_dir / 'classification_results.csv'}")

        # 6. Regression tasks (PK)
        logger.info("\n=== Regression Models (PK) ===")
        regression_results = {}

        for target in PK_TARGETS:
            logger.info(f"\n--- {PK_TARGET_NAMES[target]} ---")

            # Filter to compounds with valid values for this target
            pk_target = pk_df.dropna(subset=[target])

            if len(pk_target) == 0:
                logger.warning(f"No PK compounds for {target}, skipping")
                continue

            modality_features = get_features_multi(adata, pk_target, target, fp_dict)
            if not modality_features:
                logger.warning(f"No valid compounds for {target}, skipping")
                continue

            X_morph, y, _ = modality_features["morphology"]

            if len(y) < n_splits:
                logger.warning(f"Not enough samples ({len(y)}) for {n_splits}-fold CV, skipping {target}")
                continue

            # Track matching for this target
            matching_stats["PK"][target] = {"matched": len(pk_target)}
            logger.info(f"  Training on {len(y):,} compounds (all modalities)")

            use_log_scale = target in PK_LOG_SCALE_TARGETS

            # For log-scale targets, filter non-positive values BEFORE generating folds
            if use_log_scale:
                valid = y > 0
                if not valid.all():
                    n_invalid = (~valid).sum()
                    logger.warning(f"  Removing {n_invalid} non-positive values for log transform")
                    modality_features = {
                        k: (X[valid], y_v[valid], [j for j, v in zip(jcp, valid) if v])
                        for k, (X, y_v, jcp) in modality_features.items()
                    }
                    X_morph, y, _ = modality_features["morphology"]

            # Shared fold indices across modalities
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            fold_indices = list(kf.split(X_morph))

            regression_results[target] = {}
            for modality in MODALITIES:
                logger.info(f"  Modality: {modality}")
                X, y_mod, _ = modality_features[modality]
                regression_results[target][modality] = train_regression_cv(
                    X,
                    y_mod,
                    target,
                    n_splits,
                    use_log_scale,
                    fold_indices=fold_indices,
                )

        # Save regression fold results
        if regression_results:
            all_reg_folds = []
            for target_name, modality_results in regression_results.items():
                for modality, results in modality_results.items():
                    df = results["fold_results"].copy()
                    df["modality"] = modality
                    all_reg_folds.append(df)
            reg_folds_df = pd.concat(all_reg_folds, ignore_index=True)
            reg_folds_df.to_csv(output_dir / "regression_results.csv", index=False)
            logger.info(f"Saved: {output_dir / 'regression_results.csv'}")

        # 7. Plots
        if not skip_plots:
            logger.info("\n=== Generating Plots ===")

            # Classification ROC curves
            for task_name, modality_results in classification_results.items():
                plot_classification_roc(modality_results, task_name, output_dir, dpi)

            # Regression scatter plots
            for target_name, modality_results in regression_results.items():
                plot_regression_scatter(
                    modality_results,
                    target_name,
                    PK_TARGET_NAMES.get(target_name, target_name),
                    output_dir,
                    dpi,
                )

            # Summary comparison
            if classification_results and regression_results:
                plot_summary_bar(classification_results, regression_results, output_dir, dpi)

        # 8. Generate summary
        summary = generate_summary(classification_results, regression_results, matching_stats, output_dir)

        # Log conclusions
        logger.info("\n=== Conclusions ===")
        for conclusion in summary["numbers"]["conclusions"]:
            logger.info(f"  {conclusion}")

        logger.success(f"\nToxicity/PK prediction complete: {output_dir}")

    finally:
        metadata_con.close()


if __name__ == "__main__":
    app()


# =============================================================================
# Future Refactoring Note
# =============================================================================
