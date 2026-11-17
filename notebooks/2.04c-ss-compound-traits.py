"""Compound traits vs phenotypic activity.

Analyzes which intrinsic compound properties are associated with higher phenotypic activity:
1. Continuous properties → activity (LogP, QED, MW)
2. Categorical properties → activity (Lipinski violations, PAINS alerts)

Note: Consistency analysis is NOT included here because consistency is a property of
(compound, target) pairs, not compounds alone. A compound can have different consistency
scores for different targets. Target-level consistency analysis belongs in 2.03.

Note: Annotation-based analyses (has target, clinical phase, MOA) are separate from
intrinsic compound traits and may be added as a separate notebook.

Usage:
    pixi run python notebooks/2.04c-ss-compound-traits.py
    pixi run python notebooks/2.04c-ss-compound-traits.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/compound-traits/{dataset}/{preprocessing}/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib.patches import Patch

from jump_production.config import DEFAULT_DPI
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import (
    MAP_COL,
    NONSIG_COLOR,
    SIG_COL,
    SIG_COLOR,
    TraitConfig,
    compute_partial_correlation,
    plot_categorical_box,
    plot_scatter_with_marginals,
)

# Configuration
COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")

app = typer.Typer()

# Trait configurations for continuous properties
CONTINUOUS_TRAITS = [
    TraitConfig("Metadata_LogP", "LogP", (-5, 10), "A. LogP"),
    TraitConfig("Metadata_QED", "QED (Drug-likeness)", (0, 1), "B. QED"),
    TraitConfig("Metadata_MW", "Molecular Weight (Da)", (0, 800), "C. Molecular Weight"),
]

# Additional traits for correlation analysis (not plotted in main figure)
EXTRA_CORRELATION_TRAITS = [
    TraitConfig("Metadata_TPSA", "TPSA", (0, 200), ""),
]


# =============================================================================
# Data Loading
# =============================================================================


def load_compound_properties(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load compound properties from metadata database."""
    query = """
    SELECT
        Metadata_JCP2022,
        Metadata_SMILES,
        Metadata_MW,
        Metadata_LogP,
        Metadata_TPSA,
        Metadata_QED,
        Metadata_HBD,
        Metadata_HBA,
        Metadata_RotatableBonds,
        Metadata_NumRings,
        Metadata_Lipinski_Violations,
        Metadata_HasPAINS,
        Metadata_MurckoScaffold,
        -- Annotation flags
        CASE WHEN Metadata_repurposing_target IS NOT NULL THEN 1 ELSE 0 END as has_repurposing_target,
        CASE WHEN Metadata_Uniprot_target IS NOT NULL THEN 1 ELSE 0 END as has_uniprot_target,
        CASE WHEN Metadata_chmprb_target_genes IS NOT NULL THEN 1 ELSE 0 END as has_probe_target
    FROM compound_metadata
    WHERE Metadata_SMILES IS NOT NULL
    """
    return con.execute(query).fetchdf()


def load_activity_with_properties(
    copairs_con: duckdb.DuckDBPyConnection,
    metadata_con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str = "default",
) -> pd.DataFrame:
    """Load activity results joined with compound properties."""
    # Use shared utility for activity query
    activity_df = query_activity_results(copairs_con, dataset, preprocessing, filter_name, activity_params)
    logger.info(f"Loaded {len(activity_df):,} activity results")

    # Load compound properties
    props_df = load_compound_properties(metadata_con)
    logger.info(f"Loaded {len(props_df):,} compounds with properties")

    # Merge
    merged = activity_df.merge(props_df, on="Metadata_JCP2022", how="left")
    logger.info(f"Merged: {len(merged):,} rows with properties")

    return merged


# =============================================================================
# Combined Publication Figure
# =============================================================================


def plot_combined_figure(df: pd.DataFrame, output_dir: Path, save_pdf: bool = False, dpi: int = DEFAULT_DPI) -> None:
    """Create combined publication figure with intrinsic compound traits.

    Layout (2 rows):
        Row 1 (taller): A: LogP scatter+marginals  B: QED scatter+marginals  C: MW scatter+marginals
        Row 2 (shorter): D: Lipinski box           E: PAINS box              (legend)

    The scatter plots include marginal histograms showing the distribution shift
    between significant (blue) and non-significant (gray) compounds.

    Args:
        df: Compound DataFrame with properties and activity
        output_dir: Output directory for figure files
        save_pdf: If True, also save PDF version
        dpi: DPI for saved figure
    """
    logger.info("Creating combined publication figure with marginals...")

    fig = plt.figure(figsize=(14, 10))

    # Main gridspec: 2 rows with different heights
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.25, wspace=0.25)

    # Row 1: Scatter plots with marginals (using trait configs)
    for col_idx, trait in enumerate(CONTINUOUS_TRAITS):
        plot_scatter_with_marginals(
            df,
            x_col=trait.col,
            x_label=trait.label,
            x_lim=trait.xlim,
            panel_label=trait.panel_label,
            fig=fig,
            gs_parent=gs[0, col_idx],
        )

    # Row 2: Box plots using generic function

    # D. Lipinski violations
    ax_d = fig.add_subplot(gs[1, 0])
    valid_lip = df[df["Metadata_Lipinski_Violations"] <= 3].copy()
    plot_categorical_box(
        valid_lip,
        cat_col="Metadata_Lipinski_Violations",
        order=[0, 1, 2, 3],
        xlabel="Lipinski Violations",
        ax=ax_d,
    )
    ax_d.set_title("D. Lipinski Violations", fontsize=10, fontweight="bold", loc="left")

    # E. PAINS alerts
    ax_e = fig.add_subplot(gs[1, 1])
    if "Metadata_HasPAINS" in df.columns:
        df_pains = df.copy()
        df_pains["PAINS"] = df_pains["Metadata_HasPAINS"].map({False: "Clean", True: "PAINS"})
        plot_categorical_box(
            df_pains,
            cat_col="PAINS",
            order=["Clean", "PAINS"],
            colors={"Clean": "#3498db", "PAINS": "#e74c3c"},
            ax=ax_e,
        )
    ax_e.set_title("E. PAINS Alerts", fontsize=10, fontweight="bold", loc="left")

    # Legend panel
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")

    legend_elements = [
        Patch(facecolor=SIG_COLOR, alpha=0.7, label="Significant (p < 0.05, corrected)"),
        Patch(facecolor=NONSIG_COLOR, alpha=0.5, label="Not significant"),
    ]
    ax_legend.legend(handles=legend_elements, loc="center", fontsize=9, frameon=False)
    ax_legend.text(
        0.5,
        0.25,
        f"n = {len(df):,} compounds\n{df[SIG_COL].sum():,} significant ({100 * df[SIG_COL].mean():.1f}%)",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax_legend.transAxes,
    )

    fig.suptitle(
        "Compound Traits vs Phenotypic Activity",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    fig.savefig(output_dir / "compound_traits_combined.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / "compound_traits_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined publication figure with marginals")


# =============================================================================
# Summary Statistics
# =============================================================================


def compute_correlation_stats(
    df: pd.DataFrame,
    property_col: str,
    activity_col: str = MAP_COL,
) -> dict:
    """Compute Spearman correlation between property and activity."""
    from scipy import stats as scipy_stats

    valid = df[[property_col, activity_col]].dropna()
    if len(valid) < 10:
        return {"spearman_r": np.nan, "n": len(valid)}

    r, _ = scipy_stats.spearmanr(valid[property_col], valid[activity_col])
    return {"spearman_r": r, "n": len(valid)}


def compute_confounding_analysis(df: pd.DataFrame) -> dict:
    """Analyze confounding between LogP, MW, and QED.

    Returns inter-predictor correlations and partial correlations
    controlling for confounders.
    """
    from scipy import stats as scipy_stats

    results = {
        "inter_predictor_correlations": {},
        "partial_correlations": {},
        "lipinski_by_properties": {},
    }

    # Inter-predictor correlations
    predictors = [("Metadata_LogP", "Metadata_MW"), ("Metadata_LogP", "Metadata_QED"), ("Metadata_MW", "Metadata_QED")]
    for col1, col2 in predictors:
        valid = df[[col1, col2]].dropna()
        r, _ = scipy_stats.spearmanr(valid[col1], valid[col2])
        key = f"{col1.replace('Metadata_', '').lower()}_vs_{col2.replace('Metadata_', '').lower()}"
        results["inter_predictor_correlations"][key] = round(r, 3)

    # Partial correlations
    # LogP controlling for MW
    pc = compute_partial_correlation(df, "Metadata_LogP", MAP_COL, ["Metadata_MW"])
    results["partial_correlations"]["logp_controlling_mw"] = (
        round(pc["partial_r"], 3) if not np.isnan(pc["partial_r"]) else None
    )

    # MW controlling for LogP
    pc = compute_partial_correlation(df, "Metadata_MW", MAP_COL, ["Metadata_LogP"])
    results["partial_correlations"]["mw_controlling_logp"] = (
        round(pc["partial_r"], 3) if not np.isnan(pc["partial_r"]) else None
    )

    # QED controlling for LogP and MW
    pc = compute_partial_correlation(df, "Metadata_QED", MAP_COL, ["Metadata_LogP", "Metadata_MW"])
    results["partial_correlations"]["qed_controlling_logp_mw"] = (
        round(pc["partial_r"], 3) if not np.isnan(pc["partial_r"]) else None
    )

    # Mean properties by Lipinski violations (explains the 3-violation anomaly)
    for lip in [0, 1, 2, 3]:
        subset = df[df["Metadata_Lipinski_Violations"] == lip]
        if len(subset) > 0:
            results["lipinski_by_properties"][str(lip)] = {
                "n": len(subset),
                "mean_logp": round(subset["Metadata_LogP"].mean(), 2),
                "mean_mw": round(subset["Metadata_MW"].mean(), 0),
                "mean_qed": round(subset["Metadata_QED"].mean(), 2),
            }

    return results


def generate_summary(df: pd.DataFrame, output_dir: Path) -> dict:
    """Generate summary statistics for all trait-activity analyses.

    Includes:
    - Basic dataset statistics
    - Spearman correlations for continuous traits
    - Partial correlations controlling for confounders
    - Categorical statistics (Lipinski, PAINS)
    - Confounding analysis (inter-predictor correlations, Lipinski anomaly explanation)
    """
    n_total = len(df)
    n_active = df[SIG_COL].sum()

    continuous_correlations = {}
    categorical_stats = {}

    # Continuous property correlations
    all_traits = CONTINUOUS_TRAITS + EXTRA_CORRELATION_TRAITS
    for trait in all_traits:
        if trait.col in df.columns:
            stats_dict = compute_correlation_stats(df, trait.col)
            name = trait.col.replace("Metadata_", "").lower()
            continuous_correlations[name] = {
                "spearman_r": round(stats_dict["spearman_r"], 3) if not np.isnan(stats_dict["spearman_r"]) else None,
                "n": stats_dict["n"],
            }

    # Confounding analysis (partial correlations, inter-predictor correlations, Lipinski anomaly)
    logger.info("Computing confounding analysis...")
    confounding_analysis = compute_confounding_analysis(df)

    # Lipinski violations
    if "Metadata_Lipinski_Violations" in df.columns:
        lip_stats = df.groupby("Metadata_Lipinski_Violations")[SIG_COL].agg(["count", "sum", "mean"])
        categorical_stats["lipinski_violations"] = {
            int(v): {
                "n": int(lip_stats.loc[v, "count"]) if v in lip_stats.index else 0,
                "activity_rate": round(lip_stats.loc[v, "mean"] * 100, 2) if v in lip_stats.index else 0,
            }
            for v in range(5)
        }

    # PAINS
    if "Metadata_HasPAINS" in df.columns:
        pains_stats = df.groupby("Metadata_HasPAINS")[SIG_COL].agg(["count", "sum", "mean"])
        categorical_stats["pains"] = {
            "clean": {
                "n": int(pains_stats.loc[False, "count"]) if False in pains_stats.index else 0,
                "activity_rate": round(pains_stats.loc[False, "mean"] * 100, 2) if False in pains_stats.index else 0,
            },
            "pains_flagged": {
                "n": int(pains_stats.loc[True, "count"]) if True in pains_stats.index else 0,
                "activity_rate": round(pains_stats.loc[True, "mean"] * 100, 2) if True in pains_stats.index else 0,
            },
        }

    # Standardized summary envelope: description (intent) + numbers + index of generated files
    numbers = {
        "dataset_stats": {
            "n_compounds": n_total,
            "n_active": int(n_active),
            "activity_rate_pct": round(100 * n_active / n_total, 2),
        },
        "continuous_correlations": continuous_correlations,
        "categorical_stats": categorical_stats,
        "confounding_analysis": confounding_analysis,
        "statistical_notes": {
            "p_values": "With n=112k, all p-values are at machine precision (<1e-100). Effect sizes (rho) are the relevant metric.",
            "partial_correlations": (
                "Adjusted associations computed via residualization; they do not identify causal effects"
            ),
        },
    }
    description = (
        "Tests which intrinsic compound traits are associated with phenotypic activity in the JUMP collection: "
        "Spearman (and confounder-adjusted partial) correlations of LogP, MW, QED, and TPSA with mean average "
        "precision, plus per-category activity rates for Lipinski Ro5 violations and PAINS structural alerts."
    )
    output_file = output_dir / "compound_traits_summary.json"
    summary = write_summary(output_file, description=description, numbers=numbers, files=index_files(output_dir))
    logger.info(f"Saved summary to {output_file}")

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
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/processed/compound-traits/{dataset}/{preprocessing})",
    ),
    save_pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Also save PDF versions (for publication)",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
) -> None:
    """Analyze compound traits vs phenotypic activity."""
    # Use hierarchical output path: compound-traits/{dataset}/{preprocessing}/
    if output_dir is None:
        output_dir = Path("data/processed/compound-traits") / dataset / preprocessing
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Preprocessing: {preprocessing}")
    logger.info(f"Filter: {filter_name}")
    logger.info(f"Output: {output_dir}")

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
        # 1. Load activity with properties
        logger.info("Loading activity results with compound properties...")
        df = load_activity_with_properties(
            copairs_con, metadata_con, dataset, preprocessing, filter_name, activity_params
        )

        n_active = df[SIG_COL].sum()
        logger.info(f"Total compounds: {len(df):,}")
        logger.info(f"Active compounds: {n_active:,} ({100 * n_active / len(df):.1f}%)")

        # 2. Generate individual plots (standalone mode)
        logger.info("Generating individual plots...")

        # Scatter plots with marginals for continuous traits
        for trait in CONTINUOUS_TRAITS:
            col_name = trait.col.replace("Metadata_", "").lower()
            plot_scatter_with_marginals(
                df,
                x_col=trait.col,
                x_label=trait.label,
                x_lim=trait.xlim,
                title=f"{trait.label} vs Phenotypic Activity",
                output_dir=output_dir,
                filename=f"scatter_{col_name}_marginals.png",
                save_pdf=save_pdf,
                dpi=dpi,
            )
            logger.info(f"Saved scatter with marginals for {trait.col}")

        # Box plots for categorical traits using generic function
        valid_lip = df[df["Metadata_Lipinski_Violations"] <= 3].copy()
        plot_categorical_box(
            valid_lip,
            cat_col="Metadata_Lipinski_Violations",
            order=[0, 1, 2, 3],
            xlabel="Lipinski Violations",
            output_dir=output_dir,
            filename="lipinski_box.png",
            save_pdf=save_pdf,
            dpi=dpi,
        )
        logger.info("Saved Lipinski violations box plot")

        if "Metadata_HasPAINS" in df.columns:
            df_pains = df.copy()
            df_pains["PAINS"] = df_pains["Metadata_HasPAINS"].map({False: "Clean", True: "PAINS"})
            plot_categorical_box(
                df_pains,
                cat_col="PAINS",
                order=["Clean", "PAINS"],
                colors={"Clean": "#3498db", "PAINS": "#e74c3c"},
                output_dir=output_dir,
                filename="pains_box.png",
                save_pdf=save_pdf,
                dpi=dpi,
            )
            logger.info("Saved PAINS box plot")

        # 3. Combined publication figure
        plot_combined_figure(df, output_dir, save_pdf=save_pdf, dpi=dpi)

        # 4. Summary statistics
        generate_summary(df, output_dir)

        logger.success(f"Compound traits analysis complete: {output_dir}")

    finally:
        copairs_con.close()
        metadata_con.close()


if __name__ == "__main__":
    app()
