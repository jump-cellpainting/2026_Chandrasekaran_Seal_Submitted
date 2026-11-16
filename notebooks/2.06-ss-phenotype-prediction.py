"""Phenotype prediction from chemical structure.

Predicts phenotypic outcomes from chemical structure using proper molecular splits:
- Scaffold split using precomputed Murcko scaffolds (from metadata DB, 332x faster than DeepChem)
- Random split with stratification (sklearn's StratifiedKFold) to maintain class balance

Supports two structural representations:
- Morgan fingerprints (2048-bit): L2 norm + cosine ≈ Tanimoto (ρ=0.998, Issue #26)
- ChemBERTa embeddings (384-dim): Learned chemical semantics from SMILES

Uses 5-fold cross-validation with XGBoost. Supports GPU acceleration for fast training.

GPU Configuration:
    By default, uses GPU if available (detected via cupy import).
    Control via CUDA_VISIBLE_DEVICES environment variable.

Usage:
    # Morgan fingerprints (uses GPU if available)
    pixi run -e deepchem python notebooks/2.06-ss-phenotype-prediction.py

    # CPU only (disable GPUs)
    CUDA_VISIBLE_DEVICES="" pixi run -e deepchem python notebooks/2.06-ss-phenotype-prediction.py

    # ChemBERTa embeddings
    pixi run -e deepchem python notebooks/2.06-ss-phenotype-prediction.py --structure chemberta

Outputs: data/processed/phenotype-prediction/{dataset}/{preprocessing}/{structure_rep}/
"""

# ruff: noqa: N803, N806  # Allow uppercase X, X_train, X_test (ML convention)
# ruff: noqa: E402  # Allow imports after environment setup (required for deepchem)
# pyright: reportMissingImports=false, reportOptionalSubscript=false, reportReturnType=false, reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

# Try GPU acceleration with XGBoost
import xgboost as xgb
from loguru import logger
from scipy import stats

from jump_production.config import DEFAULT_DPI
from jump_production.gpu import has_gpu
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary

HAS_GPU = has_gpu()

logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled (cupy not available)'}")

from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
COPAIRS_DB = DATA_DIR / "processed/copairs_results.duckdb"
METADATA_DB = DATA_DIR / "interim/jump_metadata_augmented.duckdb"
OUTPUT_DIR = DATA_DIR / "processed" / "phenotype-prediction"

N_FOLDS = 5
RANDOM_STATE = 42

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_fingerprints() -> tuple[np.ndarray, np.ndarray]:
    """Load compound fingerprints (Morgan FP)."""
    fp_data = np.load(DATA_DIR / "interim/compound_featurization/morgan_fp.npz", allow_pickle=True)
    fp_array = fp_data["fingerprints"]  # Shape: (N, 2048) uint8
    fp_jcp = fp_data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp_array.shape[0]:,} compounds x {fp_array.shape[1]} bits")
    return fp_array, fp_jcp


def load_chemberta_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load ChemBERTa-77M-MLM embeddings (384-dim)."""
    chemberta_file = DATA_DIR / "interim/compound_featurization/chemberta_77m_mlm.npz"
    emb_data = np.load(chemberta_file, allow_pickle=True)
    embeddings = emb_data["embeddings"]  # Shape: (N, 384) float32
    jcp = emb_data["jcp2022"]
    logger.info(f"ChemBERTa embeddings: {embeddings.shape[0]:,} compounds x {embeddings.shape[1]} dims")
    return embeddings, jcp


def load_smiles_and_properties(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load SMILES and simple properties from metadata database."""
    df = con.execute("""
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES,
            Metadata_MW as MW,
            Metadata_LogP as LogP,
            Metadata_TPSA as TPSA
        FROM compound_metadata
        WHERE Metadata_SMILES IS NOT NULL
          AND Metadata_MW IS NOT NULL
    """).df()
    logger.info(f"SMILES + properties: {len(df):,} compounds")
    return df


def load_activity_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
) -> pd.DataFrame:
    """Load activity status from copairs results."""
    df = query_activity_results(
        con,
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
    )
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


def align_data(
    features: np.ndarray,
    feature_jcp: np.ndarray,
    activity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align features, activity labels, SMILES, and properties by JCP2022 ID.

    Returns:
        X: Feature matrix
        y_class: Binary labels (active/inactive)
        y_reg: Continuous labels (nmAP)
        smiles: SMILES strings for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline model
    """
    # Find common JCP2022 IDs
    feature_set = set(feature_jcp)
    activity_set = set(activity_df["JCP2022"])
    metadata_set = set(metadata_df["JCP2022"])

    common_jcp = sorted(feature_set & activity_set & metadata_set)
    logger.info(f"Common compounds: {len(common_jcp):,}")

    # Create index mappings
    feature_idx = {k: i for i, k in enumerate(feature_jcp)}
    activity_idx = activity_df.set_index("JCP2022")
    metadata_idx = metadata_df.set_index("JCP2022")

    # Align data
    feature_order = [feature_idx[k] for k in common_jcp]
    X = features[feature_order].astype(np.float32)
    y_class = activity_idx.loc[common_jcp, "active"].values.astype(np.int32)
    y_reg = activity_idx.loc[common_jcp, "nmAP"].values.astype(np.float32)
    smiles = metadata_idx.loc[common_jcp, "SMILES"].values
    X_props = metadata_idx.loc[common_jcp, ["MW", "LogP", "TPSA"]].values.astype(np.float32)

    # Filter out compounds with zero-norm features or NaN values
    feature_norms = np.linalg.norm(X, axis=1)
    props_valid = ~np.isnan(X_props).any(axis=1)
    valid_mask = (feature_norms > 0) & ~np.isnan(y_reg) & props_valid
    logger.info(f"Valid compounds: {valid_mask.sum():,}")

    return X[valid_mask], y_class[valid_mask], y_reg[valid_mask], smiles[valid_mask], X_props[valid_mask]


# =============================================================================
# Model Training and Evaluation
# =============================================================================


def evaluate_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_reg_train: np.ndarray,
    y_reg_test: np.ndarray,
) -> dict:
    """Evaluate dummy baseline (majority class / mean prediction)."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

    # Dummy classifier - predicts based on class distribution
    dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy_clf.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_proba = dummy_clf.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    # Dummy regressor - predicts mean
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(np.zeros((len(y_reg_train), 1)), y_reg_train)
    y_pred_reg = dummy_reg.predict(np.zeros((len(y_reg_test), 1)))

    return {
        "classification": {
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
        },
        "regression": {
            "r2": float(r2_score(y_reg_test, y_pred_reg)),
        },
    }


def train_evaluate_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Train XGBoost classifier and evaluate with threshold-independent metrics."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        device=device,
        verbosity=0,
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
    }


def train_evaluate_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Train XGBoost regressor and evaluate."""
    from sklearn.metrics import r2_score

    reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        device=device,
        verbosity=0,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # Spearman correlation
    spearman_result = stats.spearmanr(y_test, y_pred)

    return {
        "r2": float(r2_score(y_test, y_pred)),
        "spearman_r": float(spearman_result.statistic),
        "spearman_p": float(spearman_result.pvalue),
    }


def train_evaluate_mlp_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train MLP classifier - better suited for dense embeddings like ChemBERTa."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # Scale features (important for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
    }


def train_evaluate_mlp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train MLP regressor - better suited for dense embeddings like ChemBERTa."""
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    # Scale features (important for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    reg.fit(X_train_scaled, y_train)

    y_pred = reg.predict(X_test_scaled)

    # Spearman correlation
    spearman_result = stats.spearmanr(y_test, y_pred)

    return {
        "r2": float(r2_score(y_test, y_pred)),
        "spearman_r": float(spearman_result.statistic),
        "spearman_p": float(spearman_result.pvalue),
    }


def process_single_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    X_props: np.ndarray,
    use_mlp: bool = False,
) -> dict:
    """Process a single CV fold.

    Args:
        fold_idx: Index of this fold (0-based)
        train_indices: Array of indices for training set
        test_indices: Array of indices for test set
        X: Full feature matrix
        y_class: Full binary labels
        y_reg: Full continuous labels
        X_props: Full properties matrix
        use_mlp: Whether to also train MLP models (for dense embeddings)

    Returns:
        Dict with all metrics for this fold
    """
    # Use GPU 0 if available, otherwise CPU
    device = "cuda" if HAS_GPU else "cpu"
    timings = {}

    # Slice arrays using indices (fast fancy indexing)
    X_train, y_class_train, y_reg_train = X[train_indices], y_class[train_indices], y_reg[train_indices]
    X_test, y_class_test, y_reg_test = X[test_indices], y_class[test_indices], y_reg[test_indices]
    X_props_train, X_props_test = X_props[train_indices], X_props[test_indices]

    # 1. Dummy baseline (CPU only, no GPU needed)
    t0 = time.perf_counter()
    baseline_result = evaluate_baselines(y_class_train, y_class_test, y_reg_train, y_reg_test)
    timings["baseline"] = time.perf_counter() - t0

    # 2. Properties-only model (MW, LogP, TPSA)
    t0 = time.perf_counter()
    props_cls = train_evaluate_classification(X_props_train, y_class_train, X_props_test, y_class_test, device)
    props_reg = train_evaluate_regression(X_props_train, y_reg_train, X_props_test, y_reg_test, device)
    timings["props"] = time.perf_counter() - t0

    # 3. Full model with XGBoost (fingerprints/embeddings)
    t0 = time.perf_counter()
    cls_result = train_evaluate_classification(X_train, y_class_train, X_test, y_class_test, device)
    timings["fp_cls"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg_result = train_evaluate_regression(X_train, y_reg_train, X_test, y_reg_test, device)
    timings["fp_reg"] = time.perf_counter() - t0

    result = {
        "fold_idx": fold_idx,
        "device": device,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "baseline": baseline_result,
        "props_cls": props_cls,
        "props_reg": {k: v for k, v in props_reg.items() if k != "spearman_p"},
        "cls": cls_result,
        "reg": {k: v for k, v in reg_result.items() if k != "spearman_p"},
        "timings": timings,
    }

    # 4. MLP model (optional - better for dense embeddings like ChemBERTa)
    if use_mlp:
        t0 = time.perf_counter()
        mlp_cls = train_evaluate_mlp_classification(X_train, y_class_train, X_test, y_class_test)
        timings["mlp_cls"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mlp_reg = train_evaluate_mlp_regression(X_train, y_reg_train, X_test, y_reg_test)
        timings["mlp_reg"] = time.perf_counter() - t0

        result["mlp_cls"] = mlp_cls
        result["mlp_reg"] = {k: v for k, v in mlp_reg.items() if k != "spearman_p"}

    return result


def run_cross_validation(
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    smiles: np.ndarray,
    X_props: np.ndarray,
    split_type: str = "scaffold",
    use_mlp: bool = False,
) -> dict:
    """Run cross-validation with specified split type.

    Args:
        X: Feature matrix (fingerprints or embeddings)
        y_class: Binary labels
        y_reg: Continuous labels
        smiles: SMILES for scaffold splitting
        X_props: Simple properties (MW, LogP, TPSA) for baseline
        split_type: "scaffold" or "random"
        use_mlp: Whether to also train MLP models (for dense embeddings)

    Returns:
        Dict with classification and regression metrics across folds
    """
    mlp_str = " + MLP" if use_mlp else ""
    logger.info(f"Running {N_FOLDS}-fold CV with {split_type} split{mlp_str}...")

    # Materialize fold datasets to list
    fold_splits = []

    # Build SMILES->index mapping for efficient conversion
    smiles_to_idx = {s: i for i, s in enumerate(smiles)}

    if split_type == "scaffold":
        # Scaffold split using precomputed Murcko scaffolds from metadata DB
        # Note: Stratification would conflict with scaffold grouping, so we don't stratify here
        t0 = time.perf_counter()
        from jump_production.modeling import scaffold_k_fold_split

        smiles_splits = scaffold_k_fold_split(smiles, k=N_FOLDS)
        # Convert SMILES to indices (avoids slow np.isin with object dtype)
        for train_smiles, test_smiles in smiles_splits:
            train_idx = np.array([smiles_to_idx[s] for s in train_smiles if s in smiles_to_idx])
            test_idx = np.array([smiles_to_idx[s] for s in test_smiles if s in smiles_to_idx])
            fold_splits.append((train_idx, test_idx))
        logger.info(f"  Scaffold splitting took {time.perf_counter() - t0:.1f}s")
    else:
        # Random split: use sklearn's StratifiedKFold to maintain class distribution
        # This is important for imbalanced data (15% active vs 85% inactive)
        # StratifiedKFold returns indices directly - no conversion needed
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in skf.split(X, y_class):
            fold_splits.append((train_idx, test_idx))

    # Log class distribution per fold to verify stratification
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        train_pct = y_class[train_idx].mean() * 100
        test_pct = y_class[test_idx].mean() * 100
        logger.debug(f"  Fold {fold_idx + 1}: train active={train_pct:.1f}%, test active={test_pct:.1f}%")

    # Sequential execution (GPU or CPU based on HAS_GPU)
    fold_results: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"  Fold {fold_idx + 1}/{N_FOLDS}")
        result = process_single_fold(fold_idx, train_idx, test_idx, X, y_class, y_reg, X_props, use_mlp)
        fold_results.append(result)

        # Log timing breakdown for this fold
        t = result["timings"]
        timing_parts = [
            f"baseline={t['baseline']:.1f}s",
            f"props={t['props']:.1f}s",
            f"fp_cls={t['fp_cls']:.1f}s",
            f"fp_reg={t['fp_reg']:.1f}s",
        ]
        if use_mlp and "mlp_cls" in t:
            timing_parts.extend([f"mlp_cls={t['mlp_cls']:.1f}s", f"mlp_reg={t['mlp_reg']:.1f}s"])
        logger.info(f"    Timing: {', '.join(timing_parts)}")

    # Log fold details
    for r in fold_results:
        logger.info(f"  Fold {r['fold_idx'] + 1}: train={r['n_train']}, test={r['n_test']} ({r['device']})")

    # Aggregate results from all folds
    class_metrics = {"roc_auc": [], "avg_precision": []}
    reg_metrics = {"r2": [], "spearman_r": []}
    dummy_class_metrics = {"roc_auc": [], "avg_precision": []}
    dummy_reg_metrics = {"r2": []}
    props_class_metrics = {"roc_auc": [], "avg_precision": []}
    props_reg_metrics = {"r2": [], "spearman_r": []}
    mlp_class_metrics = {"roc_auc": [], "avg_precision": []} if use_mlp else None
    mlp_reg_metrics = {"r2": [], "spearman_r": []} if use_mlp else None

    for r in fold_results:
        # Dummy baseline
        for k, v in r["baseline"]["classification"].items():
            dummy_class_metrics[k].append(v)
        dummy_reg_metrics["r2"].append(r["baseline"]["regression"]["r2"])

        # Properties model
        for k, v in r["props_cls"].items():
            props_class_metrics[k].append(v)
        for k, v in r["props_reg"].items():
            props_reg_metrics[k].append(v)

        # Full model (XGBoost)
        for k, v in r["cls"].items():
            class_metrics[k].append(v)
        for k, v in r["reg"].items():
            reg_metrics[k].append(v)

        # MLP model (if enabled)
        if use_mlp and "mlp_cls" in r:
            for k, v in r["mlp_cls"].items():
                mlp_class_metrics[k].append(v)
            for k, v in r["mlp_reg"].items():
                mlp_reg_metrics[k].append(v)

    # Aggregate results
    results = {
        "classification": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in class_metrics.items()},
        "regression": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in reg_metrics.items()},
        "baseline_dummy": {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in dummy_class_metrics.items()
            },
            "regression": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in dummy_reg_metrics.items()
            },
        },
        "baseline_properties": {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in props_class_metrics.items()
            },
            "regression": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in props_reg_metrics.items()
            },
        },
    }

    # Add MLP results if enabled
    if use_mlp and mlp_class_metrics:
        results["mlp"] = {
            "classification": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in mlp_class_metrics.items()
            },
            "regression": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in mlp_reg_metrics.items()},
        }

    # Log results
    logger.info(f"  Dummy ROC-AUC:  {results['baseline_dummy']['classification']['roc_auc']['mean']:.3f}")
    logger.info(f"  Props ROC-AUC:  {results['baseline_properties']['classification']['roc_auc']['mean']:.3f}")
    logger.info(
        f"  XGB ROC-AUC:    {results['classification']['roc_auc']['mean']:.3f} +/- {results['classification']['roc_auc']['std']:.3f}"
    )
    if use_mlp and "mlp" in results:
        logger.info(
            f"  MLP ROC-AUC:    {results['mlp']['classification']['roc_auc']['mean']:.3f} +/- {results['mlp']['classification']['roc_auc']['std']:.3f}"
        )
    logger.info(
        f"  XGB R2:         {results['regression']['r2']['mean']:.3f} +/- {results['regression']['r2']['std']:.3f}"
    )
    if use_mlp and "mlp" in results:
        logger.info(
            f"  MLP R2:         {results['mlp']['regression']['r2']['mean']:.3f} +/- {results['mlp']['regression']['r2']['std']:.3f}"
        )

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_comparison(
    random_results: dict,
    scaffold_results: dict,
    output_dir: Path,
    structure_rep: str,
    n_compounds: int,
    sample_frac: float = 1.0,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Create comparison plot of random vs scaffold split performance."""
    logger.info("Creating comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rep_label = "Morgan FP" if structure_rep == "morgan" else "ChemBERTa"

    # Add sampling note if not using full dataset
    if sample_frac < 1.0:
        fig.suptitle(f"[{sample_frac:.0%} sample, n={n_compounds:,}]", fontsize=10, style="italic", color="gray")

    # Classification metrics (threshold-independent only)
    ax = axes[0]
    metrics = ["roc_auc", "avg_precision"]
    x = np.arange(len(metrics))
    width = 0.35

    random_means = [random_results["classification"][m]["mean"] for m in metrics]
    random_stds = [random_results["classification"][m]["std"] for m in metrics]
    scaffold_means = [scaffold_results["classification"][m]["mean"] for m in metrics]
    scaffold_stds = [scaffold_results["classification"][m]["std"] for m in metrics]

    ax.bar(x - width / 2, random_means, width, yerr=random_stds, label="Random Split", capsize=3, color="#1f77b4")
    ax.bar(x + width / 2, scaffold_means, width, yerr=scaffold_stds, label="Scaffold Split", capsize=3, color="#ff7f0e")

    ax.set_ylabel("Score")
    ax.set_title(f"A. Classification ({rep_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(["ROC-AUC", "Avg Precision"])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    # Add generalization gap annotation
    gap = random_results["classification"]["roc_auc"]["mean"] - scaffold_results["classification"]["roc_auc"]["mean"]
    ax.annotate(
        f"Gap: {gap:.2f}",
        xy=(0, scaffold_means[0]),
        xytext=(0.5, 0.3),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Regression metrics
    ax = axes[1]
    metrics = ["r2", "spearman_r"]
    x = np.arange(len(metrics))

    random_means = [random_results["regression"][m]["mean"] for m in metrics]
    random_stds = [random_results["regression"][m]["std"] for m in metrics]
    scaffold_means = [scaffold_results["regression"][m]["mean"] for m in metrics]
    scaffold_stds = [scaffold_results["regression"][m]["std"] for m in metrics]

    ax.bar(x - width / 2, random_means, width, yerr=random_stds, label="Random Split", capsize=3, color="#1f77b4")
    ax.bar(x + width / 2, scaffold_means, width, yerr=scaffold_stds, label="Scaffold Split", capsize=3, color="#ff7f0e")

    ax.set_ylabel("Score")
    ax.set_title(f"B. Regression ({rep_label})")
    ax.set_xticks(x)
    ax.set_xticklabels(["R²", "Spearman ρ"])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_dir / "comparison_plot.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "comparison_plot.pdf", bbox_inches="tight")
    logger.info(f"Saved: {output_dir / 'comparison_plot.png'}")
    plt.close(fig)


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    random_results: dict,
    scaffold_results: dict,
    n_compounds: int,
    n_active: int,
    structure_rep: str,
    dataset: str,
    output_dir: Path,
    sample_frac: float = 1.0,
) -> dict:
    """Generate summary JSON with all results."""
    # Calculate generalization gaps
    roc_gap = (
        random_results["classification"]["roc_auc"]["mean"] - scaffold_results["classification"]["roc_auc"]["mean"]
    )
    r2_gap = random_results["regression"]["r2"]["mean"] - scaffold_results["regression"]["r2"]["mean"]

    # Calculate overestimation percentage
    if scaffold_results["classification"]["roc_auc"]["mean"] > 0:
        roc_overestimate_pct = (roc_gap / scaffold_results["classification"]["roc_auc"]["mean"]) * 100
    else:
        roc_overestimate_pct = 0

    # Get baseline metrics (from scaffold split - more realistic)
    dummy_roc = scaffold_results["baseline_dummy"]["classification"]["roc_auc"]["mean"]
    props_roc = scaffold_results["baseline_properties"]["classification"]["roc_auc"]["mean"]
    model_roc = scaffold_results["classification"]["roc_auc"]["mean"]

    # Calculate lift over baselines
    lift_over_dummy = model_roc - dummy_roc
    lift_over_props = model_roc - props_roc

    numbers = {
        "dataset": dataset,
        "structure_rep": structure_rep,
        "sample_frac": sample_frac,
        "n_compounds": n_compounds,
        "n_active": n_active,
        "n_folds": N_FOLDS,
        "use_gpu": HAS_GPU,
        "baselines": {
            "dummy_roc_auc": float(dummy_roc),
            "properties_roc_auc": float(props_roc),
        },
        "classification": {
            "random_split": {k: v["mean"] for k, v in random_results["classification"].items()},
            "scaffold_split": {k: v["mean"] for k, v in scaffold_results["classification"].items()},
        },
        "regression": {
            "random_split": {k: v["mean"] for k, v in random_results["regression"].items()},
            "scaffold_split": {k: v["mean"] for k, v in scaffold_results["regression"].items()},
        },
        "generalization_gap": {
            "roc_auc_drop": float(roc_gap),
            "r2_drop": float(r2_gap),
            "interpretation": f"Random split overestimates ROC-AUC by {roc_overestimate_pct:.0f}%",
        },
        "lift_over_baselines": {
            "over_dummy": float(lift_over_dummy),
            "over_properties": float(lift_over_props),
            "interpretation": f"Fingerprints add {lift_over_props:.3f} ROC-AUC over simple properties",
        },
    }

    # Log key findings
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: {dataset}, Structure: {structure_rep}")
    logger.info(f"Compounds: {n_compounds:,} ({n_active:,} active, {n_active / n_compounds * 100:.1f}%)")
    logger.info("Classification (ROC-AUC, scaffold split):")
    logger.info(f"  Dummy:       {dummy_roc:.3f}")
    logger.info(f"  Properties:  {props_roc:.3f}")
    logger.info(f"  Model:       {model_roc:.3f}")
    logger.info(f"  Lift (props): +{lift_over_props:.3f}")
    logger.info("Generalization gap (random - scaffold):")
    logger.info(f"  ROC-AUC:     {roc_gap:.3f} ({roc_overestimate_pct:.0f}% overestimate)")
    logger.info(f"  R²:          {r2_gap:.3f}")

    # Standardized summary envelope: description (intent) + numbers + index of generated files
    description = (
        f"Phenotype prediction from chemical structure ({structure_rep}): XGBoost models predict "
        "compound activity (binary classification) and nmAP (regression) from molecular features, "
        "compared under random vs scaffold cross-validation splits against dummy and "
        "physicochemical-property baselines. Numbers report ROC-AUC, R2, and the "
        f"random-minus-scaffold generalization gap. Population: {dataset}."
    )
    output_file = output_dir / "summary.json"
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
    sample_frac: float = typer.Option(
        0.1,
        "--sample",
        help="Fraction of data to sample (0.1 = 10%, 1.0 = full dataset)",
    ),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
    use_mlp: bool = typer.Option(
        False,
        "--use-mlp",
        help="Also train MLP models (better for dense embeddings like ChemBERTa)",
    ),
) -> None:
    """Predict phenotypic activity from chemical structure using scaffold vs random splits."""
    # Validate structure representation
    valid_reps = {"morgan", "chemberta"}
    if structure_rep not in valid_reps:
        logger.error(f"Unknown structure representation: {structure_rep}. Valid options: {valid_reps}")
        raise typer.Exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset / preprocessing / structure_rep
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Structure rep: {structure_rep}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"MLP: {use_mlp}")

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
            features, feature_jcp = load_fingerprints()
        else:
            features, feature_jcp = load_chemberta_embeddings()

        activity_df = load_activity_data(copairs_con, dataset, preprocessing, filter_name, activity_params)
        metadata_df = load_smiles_and_properties(metadata_con)

        # 2. Align data
        X, y_class, y_reg, smiles, X_props = align_data(features, feature_jcp, activity_df, metadata_df)

        # 3. Sample data if requested
        if sample_frac < 1.0:
            n_sample = int(len(X) * sample_frac)
            rng = np.random.default_rng(RANDOM_STATE)
            sample_idx = rng.choice(len(X), size=n_sample, replace=False)
            X = X[sample_idx]
            y_class = y_class[sample_idx]
            y_reg = y_reg[sample_idx]
            smiles = smiles[sample_idx]
            X_props = X_props[sample_idx]
            logger.info(f"Sampled {sample_frac:.0%} of data: {n_sample:,} compounds")

        n_compounds = len(X)
        n_active = int(y_class.sum())

        logger.info(
            f"Final dataset: {n_compounds:,} compounds, {n_active:,} active ({n_active / n_compounds * 100:.1f}%)"
        )

        # 4. Run cross-validation with both split types
        random_results = run_cross_validation(X, y_class, y_reg, smiles, X_props, split_type="random", use_mlp=use_mlp)
        scaffold_results = run_cross_validation(
            X, y_class, y_reg, smiles, X_props, split_type="scaffold", use_mlp=use_mlp
        )

        # 5. Save detailed results
        with open(output_dir / "random_split_results.json", "w") as f:
            json.dump(random_results, f, indent=2)
        with open(output_dir / "scaffold_split_results.json", "w") as f:
            json.dump(scaffold_results, f, indent=2)

        # 6. Create comparison plot
        plot_comparison(random_results, scaffold_results, output_dir, structure_rep, n_compounds, sample_frac, dpi)

        # 7. Generate summary
        generate_summary(
            random_results,
            scaffold_results,
            n_compounds,
            n_active,
            structure_rep,
            dataset,
            output_dir,
            sample_frac,
        )

        logger.success(f"Phenotype prediction complete: {output_dir}")

    finally:
        copairs_con.close()
        metadata_con.close()


if __name__ == "__main__":
    app()
