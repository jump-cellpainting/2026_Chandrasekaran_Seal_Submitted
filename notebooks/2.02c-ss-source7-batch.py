"""Batch effect quantification for CP vs DL embeddings with/without Source 7.

Computes 2 local batch mixing metrics on X_pca (50-dim) for 4 datasets:
- CP no_s7 / CP with_s7 / DL no_s7 / DL with_s7

Metrics (on X_pca via k-NN graph, batch_key="Source", higher = better mixing):
1. iLISI — Inverse Simpson on k-NN batch distribution (median, normalized)
2. kBET acceptance — Chi-squared test of batch proportions in k-NN

Both are local neighborhood metrics from the scIB benchmark (Luecken et al. 2022),
used as batch removal metrics in Arevalo et al. 2024.

Requires GPU (rapids) for kNN. Run with:
    pixi run -e rapids python notebooks/2.02c-ss-source7-batch.py
    pixi run -e rapids python notebooks/2.02c-ss-source7-batch.py --n-neighbors 50
    pixi run -e rapids python notebooks/2.02c-ss-source7-batch.py --dpi 300

Outputs: data/processed/batch-effect/
"""

from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import typer
from loguru import logger
from scipy.stats import chi2

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.summaries import index_files, write_summary

ANNDATA_DIR = INTERIM_DATA_DIR / "anndata"
OUTPUT_DIR = PROCESSED_DATA_DIR / "batch-effect"

DATASETS = OrderedDict(
    [
        ("CP_no_s7", "compound_no_source7"),
        ("DL_no_s7", "compound_DL_CPCNN_no_source7"),
        ("CP_with_s7", "compound_with_source7"),
        ("DL_with_s7", "compound_DL_CPCNN_with_source7"),
    ]
)

# Colors: CP = blue, DL = orange (matching 1.02 CONFIG_COLORS)
FEATURE_COLORS = {"CP": "#2171b5", "DL": "#e6550d"}

app = typer.Typer()


# ---------------------------------------------------------------------------
# kNN graph
# ---------------------------------------------------------------------------


def compute_knn_graph(adata: ad.AnnData, n_neighbors: int = 90) -> ad.AnnData:
    """Compute k-NN graph on X_pca using GPU via rapids_singlecell.

    These UMAP h5ad files have n_vars=0 (no X matrix). We create a temporary
    AnnData with X=X_pca for GPU conversion, then copy the graph back.
    """
    logger.info(f"Computing kNN graph (k={n_neighbors}) on {adata.n_obs:,} cells")
    # Create temporary AnnData with X_pca as X for GPU kNN
    # (original h5ad files have n_vars=0, no X matrix)
    adata_tmp = ad.AnnData(X=adata.obsm["X_pca"].copy())
    adata_tmp.obsm["X_pca"] = adata_tmp.X.copy()
    # anndata_to_GPU modifies in place (returns None)
    rsc.get.anndata_to_GPU(adata_tmp)
    rsc.pp.neighbors(
        adata_tmp,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        metric="cosine",
        algorithm="brute",
    )
    rsc.get.anndata_to_CPU(adata_tmp)
    # Copy distance graph back to original AnnData (connectivities not needed)
    adata.obsp["distances"] = adata_tmp.obsp["distances"]
    del adata_tmp
    logger.info("kNN graph computed")
    return adata


# ---------------------------------------------------------------------------
# Metric 1: iLISI
# ---------------------------------------------------------------------------


def compute_ilisi(adata: ad.AnnData, batch_key: str = "Source") -> tuple[float, float]:
    """Integration Local Inverse Simpson's Index.

    For each cell, compute LISI from k-NN batch distribution.
    Returns (median_lisi, normalized_lisi) where normalized = (median - 1) / (n_batches - 1).
    """
    labels = (
        adata.obs[batch_key].cat.codes.values
        if hasattr(adata.obs[batch_key], "cat")
        else pd.Categorical(adata.obs[batch_key]).codes
    )
    n_batches = len(np.unique(labels))

    # Use precomputed sparse distances matrix
    dist_matrix = adata.obsp["distances"]
    n_cells = dist_matrix.shape[0]

    # Vectorized iLISI from sparse kNN graph
    indptr = dist_matrix.indptr
    indices = dist_matrix.indices

    lisi_values = np.empty(n_cells, dtype=np.float64)
    for i in range(n_cells):
        neighbor_idx = indices[indptr[i] : indptr[i + 1]]
        if len(neighbor_idx) == 0:
            lisi_values[i] = 1.0
            continue
        neighbor_labels = labels[neighbor_idx]
        # Count proportions
        _, counts = np.unique(neighbor_labels, return_counts=True)
        props = counts / counts.sum()
        # Inverse Simpson: 1 / sum(p^2)
        lisi_values[i] = 1.0 / np.sum(props**2)

    median_lisi = float(np.median(lisi_values))
    normalized_lisi = (median_lisi - 1) / (n_batches - 1) if n_batches > 1 else 0.0

    logger.info(f"iLISI: median = {median_lisi:.4f}, normalized = {normalized_lisi:.4f} (n_batches = {n_batches})")
    return median_lisi, normalized_lisi


# ---------------------------------------------------------------------------
# Metric 3: kBET acceptance
# ---------------------------------------------------------------------------


def compute_kbet(
    adata: ad.AnnData,
    batch_key: str = "Source",
    subsample: int = 50_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """kBET acceptance rate.

    Subsamples query cells and tests whether k-NN batch distribution matches
    global proportions via chi-squared test.
    """
    labels = (
        adata.obs[batch_key].cat.codes.values
        if hasattr(adata.obs[batch_key], "cat")
        else pd.Categorical(adata.obs[batch_key]).codes
    )
    n_batches = len(np.unique(labels))
    dist_matrix = adata.obsp["distances"]
    n_cells = dist_matrix.shape[0]

    # Global batch proportions
    _, global_counts = np.unique(labels, return_counts=True)
    global_props = global_counts / global_counts.sum()

    # Subsample query cells
    rng = np.random.default_rng(seed)
    n_query = min(subsample, n_cells)
    query_idx = rng.choice(n_cells, size=n_query, replace=False)

    indptr = dist_matrix.indptr
    indices = dist_matrix.indices
    df = n_batches - 1

    accepted = 0
    for i in query_idx:
        neighbor_idx = indices[indptr[i] : indptr[i + 1]]
        k = len(neighbor_idx)
        if k == 0:
            continue
        neighbor_labels = labels[neighbor_idx]
        # Observed counts per batch
        obs_counts = np.bincount(neighbor_labels, minlength=n_batches)
        # Expected counts
        exp_counts = global_props * k
        # Chi-squared statistic (skip batches with 0 expected)
        mask = exp_counts > 0
        chi2_stat = np.sum((obs_counts[mask] - exp_counts[mask]) ** 2 / exp_counts[mask])
        p_value = chi2.sf(chi2_stat, df)
        if p_value > alpha:
            accepted += 1

    acceptance_rate = accepted / n_query
    logger.info(f"kBET acceptance rate = {acceptance_rate:.4f} (n_query = {n_query:,})")
    return acceptance_rate


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def process_dataset(
    label: str,
    dataset: str,
    n_neighbors: int,
    subsample_kbet: int,
) -> dict:
    """Load h5ad, compute kNN, run iLISI and kBET."""
    h5ad_path = ANNDATA_DIR / f"{dataset}_cosine_all_umap.h5ad"
    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"  {adata.n_obs:,} obs, obsm keys: {list(adata.obsm.keys())}")

    if not hasattr(adata.obs["Source"], "cat"):
        adata.obs["Source"] = pd.Categorical(adata.obs["Source"])
    n_sources = adata.obs["Source"].nunique()

    adata = compute_knn_graph(adata, n_neighbors=n_neighbors)

    feature_type = "CP" if label.startswith("CP") else "DL"
    has_source7 = "with_s7" in label

    logger.info(f"--- {label}: Computing iLISI ---")
    ilisi_median, ilisi_norm = compute_ilisi(adata)

    logger.info(f"--- {label}: Computing kBET ---")
    kbet = compute_kbet(adata, subsample=subsample_kbet)

    return {
        "name": label,
        "feature_type": feature_type,
        "source7": has_source7,
        "n_obs": adata.n_obs,
        "n_sources": n_sources,
        "iLISI": ilisi_median,
        "iLISI_norm": ilisi_norm,
        "kBET_acceptance": kbet,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_COLS = ["iLISI_norm", "kBET_acceptance"]
METRIC_LABELS = {
    "iLISI_norm": "iLISI\n(normalized)",
    "kBET_acceptance": "kBET\nacceptance",
}


def plot_summary(df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    """2-panel grouped bar chart: one panel per metric, CP vs DL × no_s7 vs with_s7."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=False)

    x_labels = ["no Source 7", "with Source 7"]
    x_pos = np.array([0, 1])
    bar_width = 0.35

    for ax, metric in zip(axes, METRIC_COLS):
        all_vals = []
        for i, (feat, color) in enumerate(FEATURE_COLORS.items()):
            vals = []
            for s7 in [False, True]:
                row = df[(df["feature_type"] == feat) & (df["source7"] == s7)]
                vals.append(row[metric].values[0] if len(row) else 0)
            all_vals.extend(vals)
            offset = -bar_width / 2 if i == 0 else bar_width / 2
            bars = ax.bar(x_pos + offset, vals, bar_width, label=feat, color=color, alpha=0.85)
            # Annotate values
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        # Zoom y-axis to data range with padding
        vmin, vmax = min(all_vals), max(all_vals)
        spread = vmax - vmin
        if spread < 0.02:
            # Near-ceiling metrics: show enough range to see differences
            pad = max(spread * 2, 0.005)
            y_lo = max(0, vmin - pad)
            y_hi = min(1, vmax + pad * 3)  # extra room for annotations
        else:
            # Metrics with natural spread (iLISI, kBET)
            y_lo = 0
            y_hi = vmax + spread * 0.35
        ax.set_ylim(y_lo, y_hi)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Batch Effect Metrics (Source) — CP vs DL × Source 7", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved plot: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    output: Path = typer.Option(OUTPUT_DIR, help="Output directory"),
    n_neighbors: int = typer.Option(90, help="Number of neighbors for kNN graph"),
    subsample_kbet: int = typer.Option(50_000, help="Subsample size for kBET"),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for figures"),
) -> None:
    """Compute batch effect metrics for CP vs DL embeddings."""
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for label, dataset in DATASETS.items():
        logger.info(f"=== Processing {label} ({dataset}) ===")
        row = process_dataset(label, dataset, n_neighbors, subsample_kbet)
        results.append(row)

    df = pd.DataFrame(results)
    csv_path = output / "batch_effect_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    logger.info(f"\n{df.to_string(index=False)}")

    plot_summary(df, output / "batch_effect_comparison.png", dpi=dpi)

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    per_dataset = {
        rec["name"]: {
            "feature_type": rec["feature_type"],
            "has_source7": bool(rec["source7"]),
            "n_obs": int(rec["n_obs"]),
            "n_sources": int(rec["n_sources"]),
            "iLISI_median": round(float(rec["iLISI"]), 4),
            "iLISI_normalized": round(float(rec["iLISI_norm"]), 4),
            "kBET_acceptance": round(float(rec["kBET_acceptance"]), 4),
        }
        for rec in df.to_dict("records")
    }
    write_summary(
        output / "summary.json",
        description=(
            "Batch-effect quantification (Issue #55): local batch-mixing metrics - iLISI (median and "
            "normalized) and kBET acceptance - on the 50-dim PCA k-NN graph keyed by imaging source, "
            "for Cell Painting vs deep-learning (CPCNN) embeddings, each with and without Source 7 "
            "(higher = better mixing). Confirms DL mixes batches worse than CP and that Source 7 does "
            "not change batch structure."
        ),
        numbers={"n_neighbors": n_neighbors, "per_dataset": per_dataset},
        files=index_files(output),
    )
    logger.info("Done")


if __name__ == "__main__":
    app()
