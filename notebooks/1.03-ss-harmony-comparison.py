"""Batch correction comparison: does re-tuning Harmony fix DL's batch problem?

DL CPCNN features show 3.5× worse batch mixing than CellProfiler by iLISI
(#17, #55). Recipe-Harmony was optimized for CP and applied to DL without
re-tuning. This notebook tests whether in-repo Harmony with different batch
keys improves DL's results.

Compares 6 configurations (2 feature types × 3 Harmony variants):
- CP recipe:      compound_no_source7            (recipe-Harmony, default)
- CP _rsc:        compound_no_source7_rsc         (in-repo, Metadata_Batch)
- CP _rsc_source: compound_no_source7_rsc_source  (in-repo, Metadata_Source)
- DL recipe:      compound_DL_CPCNN_no_source7
- DL _rsc:        compound_DL_CPCNN_no_source7_rsc
- DL _rsc_source: compound_DL_CPCNN_no_source7_rsc_source

Usage:
    pixi run python notebooks/1.03-ss-harmony-comparison.py
    pixi run python notebooks/1.03-ss-harmony-comparison.py --dpi 300

Outputs: data/processed/harmony-comparison/
"""

from collections import OrderedDict
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.queries import query_activity_results, query_consistency_results
from jump_production.summaries import index_files, write_summary

DB_PATH = PROCESSED_DATA_DIR / "copairs_results.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "harmony-comparison"

# 6 configurations: feature type × Harmony variant
CONFIGS = OrderedDict(
    [
        ("CP_recipe", "compound_no_source7"),
        ("CP_rsc", "compound_no_source7_rsc"),
        ("CP_rsc_source", "compound_no_source7_rsc_source"),
        ("DL_recipe", "compound_DL_CPCNN_no_source7"),
        ("DL_rsc", "compound_DL_CPCNN_no_source7_rsc"),
        ("DL_rsc_source", "compound_DL_CPCNN_no_source7_rsc_source"),
    ]
)

# Colors: CP = blues, DL = oranges; saturation = Harmony variant
CONFIG_COLORS = {
    "CP_recipe": "#2171b5",
    "CP_rsc": "#6baed6",
    "CP_rsc_source": "#bdd7e7",
    "DL_recipe": "#e6550d",
    "DL_rsc": "#fdae6b",
    "DL_rsc_source": "#fdd0a2",
}

# Markers: circle = recipe, triangle = _rsc (Batch), square = _rsc_source (Source)
CONFIG_MARKERS = {
    "CP_recipe": "o",
    "CP_rsc": "^",
    "CP_rsc_source": "s",
    "DL_recipe": "o",
    "DL_rsc": "^",
    "DL_rsc_source": "s",
}

CONSISTENCY_TARGET = "repurposing"

app = typer.Typer()


def load_activity_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load activity summary for all 6 configurations."""
    rows = []
    for label, dataset in CONFIGS.items():
        df = query_activity_results(con, dataset)
        if df.empty:
            logger.warning(f"No activity results for {dataset}")
            continue
        feature_type = "CP" if label.startswith("CP") else "DL"
        harmony = label.split("_", 1)[1]
        n_sig = (df["corrected_p_value"] < 0.05).sum()
        rows.append(
            {
                "label": label,
                "dataset": dataset,
                "feature_type": feature_type,
                "harmony": harmony,
                "total": len(df),
                "sig": n_sig,
                "pct_sig": 100.0 * n_sig / len(df),
                "mean_map": df["mean_average_precision"].mean(),
                "mean_nmap": df["mean_normalized_average_precision"].mean(),
            }
        )
    return pd.DataFrame(rows)


def load_consistency_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load consistency summary (repurposing) for all 6 configurations."""
    rows = []
    for label, dataset in CONFIGS.items():
        df = query_consistency_results(con, dataset, group_type=CONSISTENCY_TARGET)
        if df.empty:
            logger.warning(f"No consistency results for {dataset}")
            continue
        feature_type = "CP" if label.startswith("CP") else "DL"
        harmony = label.split("_", 1)[1]
        n_sig = (df["corrected_p_value"] < 0.05).sum()
        rows.append(
            {
                "label": label,
                "dataset": dataset,
                "feature_type": feature_type,
                "harmony": harmony,
                "targets": len(df),
                "sig": n_sig,
                "pct_sig": 100.0 * n_sig / len(df),
                "mean_nmap": df["mean_normalized_average_precision"].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_activity_correlations(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute pairwise per-compound mAP correlations across all configs."""
    activity_dfs = {}
    for label, dataset in CONFIGS.items():
        df = query_activity_results(con, dataset)
        if not df.empty:
            activity_dfs[label] = df.set_index("Metadata_JCP2022")["mean_average_precision"]

    labels = list(activity_dfs.keys())
    corr_rows = []
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1 :]:
            merged = pd.concat([activity_dfs[l1], activity_dfs[l2]], axis=1, join="inner")
            merged.columns = ["a", "b"]
            r = merged["a"].corr(merged["b"])
            diff = (merged["a"] - merged["b"]).abs()
            corr_rows.append(
                {
                    "pair": f"{l1} vs {l2}",
                    "r": r,
                    "mean_diff": diff.mean(),
                    "max_diff": diff.max(),
                    "n_compounds": len(merged),
                }
            )
    return pd.DataFrame(corr_rows)


def plot_activity_bar(summary: pd.DataFrame, dpi: int) -> Path:
    """Bar chart of activity rates across all 6 configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [
        (axes[0], "pct_sig", "Significant compounds (%)"),
        (axes[1], "mean_nmap", "Mean nMAP"),
    ]:
        x = np.arange(len(summary))
        colors = [CONFIG_COLORS[label] for label in summary["label"]]
        bars = ax.bar(x, summary[metric], color=colors, edgecolor="white", width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)

        for bar, val in zip(bars, summary[metric]):
            fmt = f"{val:.1f}%" if "%" in ylabel else f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_title("Activity: fraction significant (p < 0.05)")
    axes[1].set_title("Activity: mean normalized mAP")
    fig.suptitle(
        "Does re-tuning Harmony fix DL's activity deficit?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    path = OUTPUT_DIR / "activity_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_consistency_bar(summary: pd.DataFrame, dpi: int) -> Path:
    """Bar chart of consistency results across all 6 configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [
        (axes[0], "pct_sig", "Significant targets (%)"),
        (axes[1], "mean_nmap", "Mean nMAP"),
    ]:
        x = np.arange(len(summary))
        colors = [CONFIG_COLORS[label] for label in summary["label"]]
        bars = ax.bar(x, summary[metric], color=colors, edgecolor="white", width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)

        for bar, val in zip(bars, summary[metric]):
            fmt = f"{val:.1f}%" if "%" in ylabel else f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_title(f"Consistency ({CONSISTENCY_TARGET}): fraction significant")
    axes[1].set_title(f"Consistency ({CONSISTENCY_TARGET}): mean nMAP")
    fig.suptitle(
        "Does re-tuning Harmony improve DL's consistency?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    path = OUTPUT_DIR / "consistency_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


def plot_correlation_heatmap(corr_df: pd.DataFrame, dpi: int) -> Path:
    """Heatmap of pairwise per-compound mAP correlations."""
    labels = list(CONFIGS.keys())
    n = len(labels)
    corr_matrix = pd.DataFrame(np.eye(n), index=labels, columns=labels)

    for _, row in corr_df.iterrows():
        pair = row["pair"]
        l1, l2 = pair.split(" vs ")
        corr_matrix.loc[l1, l2] = row["r"]
        corr_matrix.loc[l2, l1] = row["r"]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        vmin=0.95,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(
        "Per-compound mAP correlation across Harmony variants",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


@app.command()
def main(dpi: int = DEFAULT_DPI) -> None:
    """Compare Harmony batch correction variants for CP and DL features."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Activity summary
    logger.info("Loading activity results for 6 configurations")
    activity_summary = load_activity_summary(con)
    activity_summary.to_csv(OUTPUT_DIR / "activity_summary.csv", index=False)
    logger.info(f"Activity summary:\n{activity_summary.to_string(index=False)}")

    # Consistency summary
    logger.info(f"Loading consistency results ({CONSISTENCY_TARGET})")
    consistency_summary = load_consistency_summary(con)
    consistency_summary.to_csv(OUTPUT_DIR / "consistency_summary.csv", index=False)
    logger.info(f"Consistency summary:\n{consistency_summary.to_string(index=False)}")

    # Per-compound correlations
    logger.info("Computing per-compound mAP correlations")
    corr_df = compute_activity_correlations(con)
    corr_df.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    logger.info(f"Correlations:\n{corr_df.to_string(index=False)}")

    con.close()

    # Plots
    plot_activity_bar(activity_summary, dpi)
    plot_consistency_bar(consistency_summary, dpi)
    plot_correlation_heatmap(corr_df, dpi)

    # Summary envelope
    summary = {
        "question": "Does re-tuning Harmony batch correction fix DL's batch problem?",
        "answer": "No. CP outperforms DL on activity across all 3 Harmony variants.",
        "configs": list(CONFIGS.keys()),
        "activity": activity_summary.to_dict(orient="records"),
        "consistency": consistency_summary.to_dict(orient="records"),
    }
    write_summary(
        OUTPUT_DIR / "summary.json",
        description=(
            "Compares phenotypic activity and target consistency across re-tuned Harmony "
            "batch-correction variants to test whether re-tuning fixes deep-learning profiles' "
            "batch problem. Reports per-variant activity and consistency tables for CellProfiler "
            "vs deep-learning features, with bar charts and a correlation heatmap."
        ),
        numbers=summary,
        files=index_files(OUTPUT_DIR),
    )

    logger.success(f"Harmony comparison complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    app()
