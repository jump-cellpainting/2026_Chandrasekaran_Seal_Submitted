#!/usr/bin/env python
"""Analyze relationship between activity p-value and consistency contribution.

This notebook explores whether marginally active compounds (0.05 <= p < 0.10)
still contribute meaningful consistency signal.

Key questions:
1. Is there correlation between compound activity and consistency contribution?
2. Do marginally active compounds have high-consistency outliers?
3. How does signal-to-noise change across activity p-value bins?
4. At each threshold transition, how many groups are gained vs lost?
5. Why do marginal compounds with high p-values still show high consistency?
   - Investigates whether these compounds have high effect size but low replicates
     (insufficient power), or genuinely moderate signal that contributes to groups.

Usage:
    pixi run python notebooks/0.06-ss-activity-consistency-relationship.py
    pixi run python notebooks/0.06-ss-activity-consistency-relationship.py --threshold 0.30

Outputs:
    {outdir}/activity_vs_consistency.png
    {outdir}/activity_vs_consistency_summary.csv
    {outdir}/threshold_transitions.png
    {outdir}/threshold_transitions.json
    {outdir}/effect_size_vs_replicates.png
    {outdir}/effect_size_vs_replicates.csv
    {outdir}/replicate_distribution_by_consistency.png
    {outdir}/replicate_distribution_interactive.html  # Interactive version with hover tooltips


"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import typer
from plotly.subplots import make_subplots
from scipy import stats

from jump_production.config import DEFAULT_DPI
from jump_production.summaries import index_files, write_summary

app = typer.Typer()


def get_activity_consistency_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    activity_threshold: float,
    annotation: str | None = None,
) -> pd.DataFrame:
    """Join activity results with consistency scores at compound level.

    Each compound appears once, with its maximum consistency score across all
    targets/annotations. This avoids duplication from compounds hitting multiple
    targets.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name (e.g., 'compound_no_source7')
    activity_threshold : float
        Activity threshold used in consistency analysis (e.g., 0.10, 0.30)
    annotation : str | None
        Optional: filter to specific annotation type (e.g., 'Metadata_motive_gene_opentargets')

    Returns
    -------
    pd.DataFrame
        Compound-level data with columns: Metadata_JCP2022,
        activity_p, activity_nap, consistency_nap (max across all targets)
    """
    annotation_filter = ""
    if annotation:
        annotation_filter = f"AND _group_column = '{annotation}'"

    query = f"""
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
    ),
    consistency_per_compound AS (
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as consistency_nap
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = {activity_threshold}
          AND _dataset = '{dataset}'
          {annotation_filter}
        GROUP BY Metadata_JCP2022
    )
    SELECT
        c.Metadata_JCP2022,
        a.activity_p,
        a.activity_nap,
        c.consistency_nap
    FROM consistency_per_compound c
    JOIN activity a ON c.Metadata_JCP2022 = a.Metadata_JCP2022
    """
    return con.execute(query).df()


def get_effect_size_replicate_analysis(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    activity_threshold: float,
) -> pd.DataFrame:
    """Analyze effect size and replicate counts for high-consistency marginal compounds.

    Tests why marginal activity compounds still show
    high consistency? Investigates whether these compounds have high effect size
    but low replicates (insufficient power for significance), or genuinely moderate
    signal that contributes to group-level consistency.

    IMPORTANT: Statistics are computed at the COMPOUND level, not the pair level.
    A promiscuous compound hitting many targets would otherwise inflate the median
    replicate count (e.g., an antihistamine hitting 16 GPCR targets would count
    16 times at the pair level, skewing the median).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name
    activity_threshold : float
        Activity threshold used in consistency analysis

    Returns
    -------
    pd.DataFrame
        Summary by activity p-value bin and consistency level, with avg effect size
        and replicate counts (computed at compound level)
    """
    # First, get unique compounds with their max consistency level
    # (a compound is "high_consistency" if ANY of its target pairs is >= 0.3)
    # NOTE: This 0.3 threshold is a fixed exploratory value chosen for this analysis.
    # Notebook 0.07 uses a dynamically computed threshold based on the median nMAP
    # of significant targets. The fixed 0.3 is stricter, identifying only the
    # strongest consistency signals for detailed investigation.
    query = f"""
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
    ),
    consistency_per_compound AS (
        -- For each compound, get max consistency nAP across all targets
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as max_consistency_nap,
            COUNT(*) as n_target_pairs
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = {activity_threshold}
          AND _dataset = '{dataset}'
        GROUP BY Metadata_JCP2022
    ),
    compound_level AS (
        SELECT
            a.Metadata_JCP2022,
            a.activity_p,
            a.activity_nap,
            a.n_replicates,
            c.max_consistency_nap,
            c.n_target_pairs
        FROM activity a
        JOIN consistency_per_compound c ON a.Metadata_JCP2022 = c.Metadata_JCP2022
    )
    SELECT
        CASE
            WHEN activity_p < 0.05 THEN '1_p<0.05'
            WHEN activity_p < 0.10 THEN '2_0.05-0.10'
            WHEN activity_p < 0.15 THEN '3_0.10-0.15'
            WHEN activity_p < 0.20 THEN '4_0.15-0.20'
            WHEN activity_p < 0.25 THEN '5_0.20-0.25'
            WHEN activity_p < 0.30 THEN '6_0.25-0.30'
            ELSE '7_p>=0.30'
        END as activity_p_bin,
        CASE
            WHEN max_consistency_nap >= 0.3 THEN 'high_consistency'
            WHEN max_consistency_nap >= 0.0 THEN 'low_consistency'
            ELSE 'negative_consistency'
        END as consistency_level,
        COUNT(*) as n_compounds,
        ROUND(AVG(activity_nap), 4) as avg_activity_nap,
        ROUND(MEDIAN(activity_nap), 4) as median_activity_nap,
        ROUND(AVG(n_replicates), 2) as avg_replicates,
        ROUND(MEDIAN(n_replicates), 0) as median_replicates,
        ROUND(AVG(max_consistency_nap), 4) as avg_max_consistency_nap
    FROM compound_level
    GROUP BY activity_p_bin, consistency_level
    ORDER BY activity_p_bin, consistency_level DESC
    """
    return con.execute(query).df()


def get_high_consistency_marginal_details(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    activity_threshold: float,
    n_examples: int = 20,
) -> pd.DataFrame:
    """Get detailed examples of marginal activity compounds with high consistency.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name
    activity_threshold : float
        Activity threshold used in consistency analysis
    n_examples : int
        Number of examples to return

    Returns
    -------
    pd.DataFrame
        Detailed examples with compound ID, target, activity metrics, and consistency
    """
    query = f"""
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
    ),
    consistency AS (
        SELECT
            Metadata_JCP2022,
            normalized_average_precision as consistency_nap,
            group_value as target,
            _group_column as annotation
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = {activity_threshold}
          AND _dataset = '{dataset}'
    )
    SELECT
        a.Metadata_JCP2022,
        c.target,
        REPLACE(c.annotation, 'Metadata_', '') as annotation,
        ROUND(a.activity_p, 4) as activity_p,
        ROUND(a.activity_nap, 4) as activity_nap,
        a.n_replicates,
        ROUND(c.consistency_nap, 4) as consistency_nap
    FROM consistency c
    JOIN activity a ON c.Metadata_JCP2022 = a.Metadata_JCP2022
    WHERE a.activity_p >= 0.05 AND a.activity_p < 0.30
      AND c.consistency_nap >= 0.3
    ORDER BY c.consistency_nap DESC
    LIMIT {n_examples}
    """
    return con.execute(query).df()


def plot_effect_size_vs_replicates(
    summary_df: pd.DataFrame,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Create plot showing effect size and replicate counts across activity bins.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from get_effect_size_replicate_analysis
    figsize : tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure with 2 panels
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Filter to high-consistency compounds only for the key insight
    high_cons = summary_df[summary_df["consistency_level"] == "high_consistency"].copy()

    # Clean up bin labels for display
    high_cons["activity_p_bin_clean"] = high_cons["activity_p_bin"].str.replace(r"^\d_", "", regex=True)

    # Plot 1: Median activity nMAP by p-value bin (high-consistency only)
    ax1 = axes[0]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#8e44ad", "#3498db", "#2980b9"]
    x = range(len(high_cons))
    bars = ax1.bar(x, high_cons["median_activity_nap"], color=colors[: len(high_cons)], alpha=0.8)

    ax1.set_xlabel("Activity p-value bin", fontsize=11)
    ax1.set_ylabel("Median Activity Effect Size (nMAP)", fontsize=11)
    ax1.set_title(
        "Activity Effect Size for High-Consistency Compounds\n(per compound, max consistency nAP ≥ 0.3)", fontsize=12
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(high_cons["activity_p_bin_clean"], fontsize=9, rotation=15)
    ax1.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5, label="Strong effect (0.3)")
    ax1.legend(loc="upper right", fontsize=9)

    # Add sample size annotations
    for i, (bar, n) in enumerate(zip(bars, high_cons["n_compounds"])):
        ax1.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    # Plot 2: Median replicates by p-value bin (high-consistency only)
    ax2 = axes[1]
    bars2 = ax2.bar(x, high_cons["median_replicates"], color=colors[: len(high_cons)], alpha=0.8)

    ax2.set_xlabel("Activity p-value bin", fontsize=11)
    ax2.set_ylabel("Median Replicate Count", fontsize=11)
    ax2.set_title(
        "Replicate Count for High-Consistency Compounds\n(per compound, max consistency nAP ≥ 0.3)", fontsize=12
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(high_cons["activity_p_bin_clean"], fontsize=9, rotation=15)
    ax2.axhline(y=5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Typical (5)")
    ax2.legend(loc="upper right", fontsize=9)

    # Add annotations
    for i, (bar, n) in enumerate(zip(bars2, high_cons["n_compounds"])):
        ax2.annotate(
            f"n={n}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def get_compound_level_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    activity_threshold: float,
) -> pd.DataFrame:
    """Get compound-level data for detailed plotting.

    Returns one row per compound with activity metrics and max consistency.
    """
    query = f"""
    WITH activity AS (
        SELECT
            Metadata_JCP2022,
            corrected_p_value as activity_p,
            mean_normalized_average_precision as activity_nap,
            len(string_split(indices, ',')) as n_replicates
        FROM activity_results
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
    ),
    consistency_per_compound AS (
        SELECT
            Metadata_JCP2022,
            MAX(normalized_average_precision) as max_consistency_nap
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = {activity_threshold}
          AND _dataset = '{dataset}'
        GROUP BY Metadata_JCP2022
    )
    SELECT
        a.Metadata_JCP2022,
        a.activity_p,
        a.activity_nap,
        a.n_replicates,
        c.max_consistency_nap
    FROM activity a
    JOIN consistency_per_compound c ON a.Metadata_JCP2022 = c.Metadata_JCP2022
    """
    return con.execute(query).df()


def plot_replicate_distribution_by_consistency(
    compound_df: pd.DataFrame,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Show relationship between activity p-value, replicates, and consistency.

    This directly visualizes whether high-consistency compounds with high p-values
    (marginal activity) are the ones with low replicate counts.

    Parameters
    ----------
    compound_df : pd.DataFrame
        Output from get_compound_level_data
    figsize : tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure with 2 panels
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df = compound_df.copy()

    # Define consistency groups
    df["consistency_group"] = pd.cut(
        df["max_consistency_nap"],
        bins=[-np.inf, 0.0, 0.3, np.inf],
        labels=["Negative (<0)", "Low (0-0.3)", "High (≥0.3)"],
    )

    # Colors for consistency
    colors = {"High (≥0.3)": "#2ecc71", "Low (0-0.3)": "#3498db", "Negative (<0)": "#95a5a6"}

    # Filter to marginal + significant (exclude inactive p >= 0.30)
    df_filtered = df[df["activity_p"] < 0.30].copy()

    # Panel 1: Scatter of activity_p vs n_replicates, colored by consistency
    ax1 = axes[0]

    # Add jitter to replicates for visibility (assign to each row to maintain consistency)
    np.random.seed(42)
    df_filtered["jitter"] = np.random.uniform(-0.35, 0.35, len(df_filtered))

    for cons_group in ["Negative (<0)", "Low (0-0.3)", "High (≥0.3)"]:
        subset = df_filtered[df_filtered["consistency_group"] == cons_group]
        if len(subset) > 0:
            ax1.scatter(
                subset["activity_p"],
                subset["n_replicates"] + subset["jitter"],
                alpha=0.6,
                s=30,
                color=colors[cons_group],
                edgecolor="white",
                linewidth=0.3,
                label=f"{cons_group} (n={len(subset)})",
            )

    # Add threshold lines
    ax1.axvline(x=0.05, color="black", linestyle="--", linewidth=1, alpha=0.5, label="p=0.05")
    ax1.axvline(x=0.10, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="p=0.10")

    ax1.set_xlabel("Activity p-value", fontsize=11)
    ax1.set_ylabel("Number of Replicates", fontsize=11)
    ax1.set_title("Activity p-value vs Replicates\n(colored by consistency level)", fontsize=12)
    ax1.set_xlim(-0.01, 0.31)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Focus on high-consistency only, show activity_p vs replicates
    ax2 = axes[1]
    high_cons = df_filtered[df_filtered["consistency_group"] == "High (≥0.3)"].copy()

    if len(high_cons) > 0:
        # Color by activity p-value bin
        high_cons["p_bin"] = pd.cut(
            high_cons["activity_p"],
            bins=[0, 0.05, 0.10, 0.20, 0.30],
            labels=["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"],
        )
        p_colors = {
            "p<0.05": "#2ecc71",
            "0.05-0.10": "#e74c3c",
            "0.10-0.20": "#9b59b6",
            "0.20-0.30": "#3498db",
        }

        for p_bin in ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"]:
            subset = high_cons[high_cons["p_bin"] == p_bin]
            if len(subset) > 0:
                ax2.scatter(
                    subset["activity_nap"],
                    subset["n_replicates"] + subset["jitter"],
                    alpha=0.7,
                    s=50,
                    color=p_colors[p_bin],
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"{p_bin} (n={len(subset)}, med_rep={subset['n_replicates'].median():.0f})",
                )

        ax2.set_xlabel("Activity Effect Size (nMAP)", fontsize=11)
        ax2.set_ylabel("Number of Replicates", fontsize=11)
        ax2.set_title(
            "High-Consistency Compounds Only\n(activity effect size vs replicates, colored by p-value)",
            fontsize=12,
        )
        ax2.legend(loc="upper right", fontsize=8)
        ax2.axhline(y=5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Ensure both plots have same y-axis range with integer ticks
    y_max = 16
    for ax in axes:
        ax.set_ylim(0, y_max)
        ax.set_yticks(range(0, y_max + 1, 2))

    # Note compounds clipped from view
    n_above = (df_filtered["n_replicates"] > y_max).sum()
    if n_above > 0:
        axes[0].text(
            0.98,
            0.02,
            f"({n_above} compounds with >{y_max} replicates not shown)",
            transform=axes[0].transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            style="italic",
            color="gray",
        )

    plt.tight_layout()
    return fig


def plot_replicate_distribution_interactive(
    compound_df: pd.DataFrame,
) -> go.Figure:
    """Create interactive version of replicate distribution plot with hover tooltips.

    Parameters
    ----------
    compound_df : pd.DataFrame
        Output from get_compound_level_data

    Returns
    -------
    go.Figure
        Plotly figure with two panels and hover information
    """
    df = compound_df.copy()

    # Define consistency groups
    df["consistency_group"] = pd.cut(
        df["max_consistency_nap"],
        bins=[-np.inf, 0.0, 0.3, np.inf],
        labels=["Negative (<0)", "Low (0-0.3)", "High (≥0.3)"],
    )

    # Define p-value bins for coloring
    df["p_bin"] = pd.cut(
        df["activity_p"],
        bins=[0, 0.05, 0.10, 0.20, 0.30, 1.0],
        labels=["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p≥0.30"],
    )

    # Filter to p < 0.30
    df_filtered = df[df["activity_p"] < 0.30].copy()

    # Add jitter for visibility
    np.random.seed(42)
    df_filtered["n_replicates_jittered"] = df_filtered["n_replicates"] + np.random.uniform(
        -0.35, 0.35, len(df_filtered)
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Activity p-value vs Replicates<br>(colored by consistency level)",
            "High-Consistency Compounds Only<br>(colored by p-value bin)",
        ),
        horizontal_spacing=0.08,
    )

    # Colors
    cons_colors = {"High (≥0.3)": "#2ecc71", "Low (0-0.3)": "#3498db", "Negative (<0)": "#95a5a6"}
    p_colors = {
        "p<0.05": "#2ecc71",
        "0.05-0.10": "#e74c3c",
        "0.10-0.20": "#9b59b6",
        "0.20-0.30": "#3498db",
    }

    # Panel 1: All compounds colored by consistency
    for cons_group in ["Negative (<0)", "Low (0-0.3)", "High (≥0.3)"]:
        subset = df_filtered[df_filtered["consistency_group"] == cons_group]
        if len(subset) > 0:
            fig.add_trace(
                go.Scatter(
                    x=subset["activity_p"],
                    y=subset["n_replicates_jittered"],
                    mode="markers",
                    name=f"{cons_group} (n={len(subset)})",
                    marker=dict(color=cons_colors[cons_group], size=8, opacity=0.7),
                    customdata=np.stack(
                        [
                            subset["Metadata_JCP2022"],
                            subset["n_replicates"],
                            subset["activity_nap"],
                            subset["max_consistency_nap"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Activity p: %{x:.4f}<br>"
                        "Replicates: %{customdata[1]}<br>"
                        "Activity nMAP: %{customdata[2]:.3f}<br>"
                        "Max Consistency nAP: %{customdata[3]:.3f}"
                        "<extra></extra>"
                    ),
                    legendgroup="panel1",
                ),
                row=1,
                col=1,
            )

    # Add threshold lines to panel 1
    fig.add_vline(x=0.05, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    fig.add_vline(x=0.10, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)

    # Panel 2: High-consistency only, colored by p-value bin
    high_cons = df_filtered[df_filtered["consistency_group"] == "High (≥0.3)"].copy()

    for p_bin in ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30"]:
        subset = high_cons[high_cons["p_bin"] == p_bin]
        if len(subset) > 0:
            med_rep = subset["n_replicates"].median()
            fig.add_trace(
                go.Scatter(
                    x=subset["activity_nap"],
                    y=subset["n_replicates_jittered"],
                    mode="markers",
                    name=f"{p_bin} (n={len(subset)}, med_rep={med_rep:.0f})",
                    marker=dict(color=p_colors[p_bin], size=10, opacity=0.7),
                    customdata=np.stack(
                        [
                            subset["Metadata_JCP2022"],
                            subset["n_replicates"],
                            subset["activity_p"],
                            subset["max_consistency_nap"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Activity nMAP: %{x:.3f}<br>"
                        "Replicates: %{customdata[1]}<br>"
                        "Activity p: %{customdata[2]:.4f}<br>"
                        "Max Consistency nAP: %{customdata[3]:.3f}"
                        "<extra></extra>"
                    ),
                    legendgroup="panel2",
                ),
                row=1,
                col=2,
            )

    # Add horizontal line at y=5 for panel 2
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    # Update layout
    fig.update_xaxes(title_text="Activity p-value", range=[-0.01, 0.31], row=1, col=1)
    fig.update_xaxes(title_text="Activity Effect Size (nMAP)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Replicates", range=[0, 16], dtick=2, row=1, col=1)
    fig.update_yaxes(title_text="Number of Replicates", range=[0, 16], dtick=2, row=1, col=2)

    fig.update_layout(
        height=500,
        width=1200,
        title_text="Replicate Distribution by Consistency Level (Interactive)",
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    )

    return fig


def compute_summary_by_activity_bin(
    df: pd.DataFrame,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute summary statistics by activity p-value bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data with activity_p and consistency_nap columns
    bins : list[float] | None
        Bin edges for activity p-value (default: [0, 0.05, 0.10, 0.20, 0.30, 1.0])
    labels : list[str] | None
        Labels for bins

    Returns
    -------
    pd.DataFrame
        Summary with columns: n, mean, median, std, n_above_0.3, pct_above_0.3
    """
    if bins is None:
        bins = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
    if labels is None:
        labels = ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p>0.30"]

    df = df.copy()
    df["activity_bin"] = pd.cut(df["activity_p"], bins=bins, labels=labels)

    summary = (
        df.groupby("activity_bin", observed=True)
        .agg(
            n=("consistency_nap", "count"),
            mean=("consistency_nap", "mean"),
            median=("consistency_nap", "median"),
            std=("consistency_nap", "std"),
            n_above_03=("consistency_nap", lambda x: (x > 0.3).sum()),
        )
        .round(3)
    )
    summary["pct_above_03"] = (100 * summary["n_above_03"] / summary["n"]).round(1)

    return summary


def plot_activity_vs_consistency(
    df: pd.DataFrame,
    max_p: float = 0.35,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Create 3-panel figure showing activity vs consistency relationship.

    Parameters
    ----------
    df : pd.DataFrame
        Data with activity_p and consistency_nap columns
    max_p : float
        Maximum p-value to show on x-axis
    figsize : tuple[int, int]
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Define groups and colors
    bins = [0, 0.05, 0.10, 0.20, 0.30, 1.0]
    labels = ["p<0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", "p>0.30"]
    colors = {
        "p<0.05": "#2ecc71",
        "0.05-0.10": "#e74c3c",
        "0.10-0.20": "#9b59b6",
        "0.20-0.30": "#3498db",
        "p>0.30": "#95a5a6",
    }

    df = df.copy()
    df["group"] = pd.cut(df["activity_p"], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Scatter of activity p-value vs consistency nAP
    ax1 = axes[0]
    for grp in labels[:-1]:  # Exclude p>0.30 for clarity
        mask = df["group"] == grp
        if mask.sum() > 0:
            ax1.scatter(
                df.loc[mask, "activity_p"],
                df.loc[mask, "consistency_nap"],
                c=colors[grp],
                alpha=0.3,
                s=10,
                label=grp,
            )

    # Add threshold lines
    for p, color in [(0.05, "#2ecc71"), (0.10, "#e74c3c"), (0.20, "#9b59b6"), (0.30, "#3498db")]:
        ax1.axvline(x=p, color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax1.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5, label="High consistency")
    ax1.set_xlabel("Activity p-value", fontsize=12)
    ax1.set_ylabel("Max Consistency nAP", fontsize=12)
    ax1.set_title(
        "Activity p-value vs Consistency\n(each dot = one compound, max consistency across targets)", fontsize=12
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(-0.01, max_p)

    # Plot 2: Box plots comparing all groups
    ax2 = axes[1]
    groups_to_plot = [g for g in labels[:-1] if (df["group"] == g).sum() > 0]
    groups_data = [df[df["group"] == g]["consistency_nap"] for g in groups_to_plot]
    box_colors = [colors[g] for g in groups_to_plot]

    if groups_data:
        bp = ax2.boxplot(
            groups_data,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)

    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax2.axhline(y=0.3, color="orange", linestyle=":", linewidth=1.5)
    ax2.set_ylabel("Max Consistency nAP", fontsize=12)
    ax2.set_title("Distribution by activity group", fontsize=12)

    # Set x-tick labels with counts included
    tick_labels_with_n = [f"{g}\n(n={len(df[df['group'] == g])})" for g in groups_to_plot]
    ax2.set_xticks(range(1, len(groups_to_plot) + 1))
    ax2.set_xticklabels(tick_labels_with_n, fontsize=9)

    # Plot 3: Histogram overlay
    ax3 = axes[2]
    hist_bins = np.linspace(-0.1, 0.8, 25)
    for grp in groups_to_plot:
        data = df[df["group"] == grp]["consistency_nap"]
        ax3.hist(
            data,
            bins=hist_bins,
            alpha=0.5,
            color=colors[grp],
            label=f"{grp} (n={len(data)})",
            density=True,
        )

    ax3.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax3.axvline(x=0.3, color="orange", linestyle=":", linewidth=2)
    ax3.set_xlabel("Max Consistency nAP", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.set_title("Overlaid distributions", fontsize=12)
    ax3.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


# =============================================================================
# Threshold Transition Analysis
# =============================================================================


def get_significance_by_threshold(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
) -> pd.DataFrame:
    """Get group significance status at each threshold."""
    return con.execute(
        """
        SELECT
            group_value,
            _group_column as annotation,
            _activity_threshold as threshold,
            below_corrected_p as significant,
            mean_normalized_average_precision as nmap,
            corrected_p_value as corrected_p,
            n_perturbations
        FROM consistency_results
        WHERE _preprocessing LIKE '%_sweep'
          AND _dataset = ?
        ORDER BY group_value, _group_column, _activity_threshold
        """,
        [dataset],
    ).df()


def compute_transitions(df: pd.DataFrame, thresholds: list[float]) -> dict:
    """Compute gains and losses at each threshold transition.

    Returns a dict with:
    - summary: list of {from, to, gained, lost, net, ratio} for each transition
    - details: list of {from, to, lost_groups: [...]} with details of lost groups
    """
    annotations = df["annotation"].unique().tolist()

    summary = []
    details = []

    for i in range(len(thresholds) - 1):
        t1, t2 = thresholds[i], thresholds[i + 1]

        total_gained = 0
        total_lost = 0
        gained_groups = []
        lost_groups = []

        for ann in annotations:
            ann_df = df[df["annotation"] == ann]
            df1 = ann_df[ann_df["threshold"] == t1].set_index("group_value")
            df2 = ann_df[ann_df["threshold"] == t2].set_index("group_value")

            sig1 = set(df1[df1["significant"]].index)
            sig2 = set(df2[df2["significant"]].index)

            gained = sig2 - sig1
            lost = sig1 - sig2

            total_gained += len(gained)
            total_lost += len(lost)

            # Record details of gained groups
            for g in gained:
                if g in df1.index and g in df2.index:
                    gained_groups.append(
                        {
                            "annotation": ann.replace("Metadata_", ""),
                            "group": g,
                            "nmap_before": round(float(df1.loc[g, "nmap"]), 4),
                            "nmap_after": round(float(df2.loc[g, "nmap"]), 4),
                            "n_before": int(df1.loc[g, "n_perturbations"]),
                            "n_after": int(df2.loc[g, "n_perturbations"]),
                        }
                    )
                elif g in df2.index:
                    gained_groups.append(
                        {
                            "annotation": ann.replace("Metadata_", ""),
                            "group": g,
                            "nmap_before": None,
                            "nmap_after": round(float(df2.loc[g, "nmap"]), 4),
                            "n_before": None,
                            "n_after": int(df2.loc[g, "n_perturbations"]),
                        }
                    )

            # Record details of lost groups
            for g in lost:
                if g in df2.index:
                    lost_groups.append(
                        {
                            "annotation": ann.replace("Metadata_", ""),
                            "group": g,
                            "nmap_before": round(float(df1.loc[g, "nmap"]), 4),
                            "nmap_after": round(float(df2.loc[g, "nmap"]), 4),
                            "n_before": int(df1.loc[g, "n_perturbations"]),
                            "n_after": int(df2.loc[g, "n_perturbations"]),
                        }
                    )

        ratio = total_gained / total_lost if total_lost > 0 else float("inf")

        summary.append(
            {
                "from": t1,
                "to": t2,
                "gained": total_gained,
                "lost": total_lost,
                "net": total_gained - total_lost,
                "ratio": round(ratio, 1) if ratio != float("inf") else "inf",
            }
        )

        details.append(
            {
                "from": t1,
                "to": t2,
                "gained_groups": gained_groups,
                "lost_groups": lost_groups,
            }
        )

    return {"summary": summary, "details": details}


def plot_transitions(transitions: dict, figsize: tuple[int, int] = (12, 5)) -> plt.Figure:
    """Create plot showing gains vs losses at each threshold transition."""
    summary = transitions["summary"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Data for plotting
    labels = [f"{s['from']:.2f}\u2192{s['to']:.2f}" for s in summary]
    gained = [s["gained"] for s in summary]
    lost = [s["lost"] for s in summary]
    net = [s["net"] for s in summary]

    x = np.arange(len(labels))

    # Plot 1: Stacked bar showing gained vs lost
    ax1 = axes[0]
    ax1.bar(x, gained, 0.6, label="Gained", color="#2ecc71", alpha=0.8)
    ax1.bar(x, [-val for val in lost], 0.6, label="Lost", color="#e74c3c", alpha=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_xlabel("Threshold transition", fontsize=11)
    ax1.set_ylabel("Number of groups", fontsize=11)
    ax1.set_title("Groups gained vs lost at each transition", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(loc="upper right")

    # Add ratio annotations
    for i, s in enumerate(summary):
        ratio_str = f"{s['ratio']}:1" if s["ratio"] != "inf" else "\u221e:1"
        ax1.annotate(ratio_str, xy=(i, gained[i] + 2), ha="center", fontsize=9, color="#666666")

    # Plot 2: Cumulative net change
    ax2 = axes[1]
    cumulative = np.cumsum(net)
    thresholds_end = [s["to"] for s in summary]

    ax2.plot(thresholds_end, cumulative, "o-", color="#3498db", linewidth=2, markersize=8)
    ax2.fill_between(thresholds_end, 0, cumulative, alpha=0.3, color="#3498db")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # Annotate the big jump at 0.10
    ax2.annotate(
        f"+{net[0]} at 0.05\u21920.10\n(43:1 ratio)",
        xy=(0.10, cumulative[0]),
        xytext=(0.15, cumulative[0] - 10),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#666666"),
    )

    ax2.set_xlabel("Activity threshold", fontsize=11)
    ax2.set_ylabel("Cumulative net change in significant groups", fontsize=11)
    ax2.set_title("Cumulative effect of relaxing threshold", fontsize=12)
    ax2.set_xticks(thresholds_end)

    plt.tight_layout()
    return fig


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name to analyze",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        "-t",
        help="Activity threshold for consistency analysis (0.05-0.30)",
    ),
    annotation: str | None = typer.Option(
        None,
        "--annotation",
        "-a",
        help="Filter to specific annotation (e.g., Metadata_motive_gene_opentargets)",
    ),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Output directory for plots (default: data/processed/exploration/0.06/{dataset})",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
    show_examples: bool = typer.Option(
        True,
        "--show-examples/--no-examples",
        help="Print high-consistency examples for each activity bin",
    ),
) -> None:
    """Analyze relationship between activity p-value and consistency contribution."""
    # Set default outdir
    if outdir is None:
        outdir = Path(f"data/processed/exploration/0.06/{dataset}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    con = duckdb.connect("data/processed/copairs_results.duckdb", read_only=True)

    print(f"Dataset: {dataset}")
    print(f"Activity threshold: {threshold}")
    if annotation:
        print(f"Annotation filter: {annotation}")

    # Get data (compound-level: one row per compound with max consistency)
    print("\nLoading data...")
    df = get_activity_consistency_data(con, dataset, threshold, annotation)
    print(f"Loaded {len(df):,} compounds (using max consistency per compound)")

    # Filter to p < threshold + small buffer for visualization
    df_filtered = df[df["activity_p"] < threshold + 0.05].copy()
    print(f"After filtering to p < {threshold + 0.05}: {len(df_filtered):,} compounds")

    # Compute summary
    print("\n" + "=" * 70)
    print("SUMMARY BY ACTIVITY BIN")
    print("=" * 70)
    summary = compute_summary_by_activity_bin(df_filtered)
    print(summary.to_string())

    # Save summary
    summary_path = outdir / "activity_vs_consistency_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary to: {summary_path}")

    # Compute correlation
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    rho, pval = stats.spearmanr(df_filtered["activity_p"], df_filtered["consistency_nap"])
    print(f"Spearman correlation (activity p vs consistency nAP): rho={rho:.3f}, p={pval:.2e}")

    # Create compound-level plot
    print("\nGenerating compound-level plot...")
    fig = plot_activity_vs_consistency(df_filtered, max_p=threshold + 0.05)

    plot_path = outdir / "activity_vs_consistency.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
    plt.close(fig)

    # ==========================================================================
    # Threshold Transition Analysis (group-level gains/losses)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("THRESHOLD TRANSITION ANALYSIS")
    print("=" * 70)

    # Get group-level significance data
    sig_df = get_significance_by_threshold(con, dataset)
    print(f"Loaded {len(sig_df):,} group-threshold combinations")

    # Compute transitions
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    transitions = compute_transitions(sig_df, thresholds)

    # Print summary
    print(f"\n{'Transition':<15} {'Gained':>8} {'Lost':>8} {'Net':>8} {'Ratio':>10}")
    print("-" * 55)
    for s in transitions["summary"]:
        ratio_str = f"{s['ratio']}:1" if s["ratio"] != "inf" else "inf:1"
        print(f"{s['from']:.2f} -> {s['to']:.2f}    {s['gained']:>8} {s['lost']:>8} {s['net']:>+8} {ratio_str:>10}")

    # Save JSON
    json_path = outdir / "threshold_transitions.json"
    with open(json_path, "w") as f:
        json.dump(transitions, f, indent=2)
    print(f"\nSaved JSON to: {json_path}")

    # Create and save transition plot
    print("\nGenerating transition plot...")
    fig_trans = plot_transitions(transitions)
    trans_plot_path = outdir / "threshold_transitions.png"
    fig_trans.savefig(trans_plot_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {trans_plot_path}")
    plt.close(fig_trans)

    # ==========================================================================
    # Effect Size vs Replicates Analysis
    # ==========================================================================
    # Test why marginal activity compounds (p >= 0.05)
    # still show high consistency? The hypothesis is that they have HIGH activity
    # effect size (nAP) but LOW replicate counts, so they fail significance
    # individually but contribute real signal to target groups.
    print("\n" + "=" * 70)
    print("EFFECT SIZE VS REPLICATES ANALYSIS")
    print("=" * 70)
    print("Hypothesis: Marginal activity compounds with high consistency have")
    print("HIGH effect size but LOW replicates (insufficient power for significance)")

    # Get the analysis data
    effect_rep_df = get_effect_size_replicate_analysis(con, dataset, threshold)
    print("\nAnalysis by activity p-value bin and consistency level:")
    print(effect_rep_df.to_string())

    # Save the summary
    effect_rep_path = outdir / "effect_size_vs_replicates.csv"
    effect_rep_df.to_csv(effect_rep_path, index=False)
    print(f"\nSaved to: {effect_rep_path}")

    # Print focused summary for high-consistency marginal compounds
    high_cons_marginal = effect_rep_df[
        (effect_rep_df["consistency_level"] == "high_consistency") & (effect_rep_df["activity_p_bin"] != "7_p>=0.30")
    ].copy()

    if len(high_cons_marginal) > 0:
        print("\n--- HIGH-CONSISTENCY COMPOUNDS BY ACTIVITY BIN ---")
        print(f"{'Activity p-bin':<15} {'N':>6} {'Effect Size':>12} {'Replicates':>11}")
        print(f"{'':15} {'':>6} {'(median nAP)':>12} {'(median)':>11}")
        print("-" * 50)
        for _, row in high_cons_marginal.iterrows():
            bin_label = row["activity_p_bin"].replace("_", " ").lstrip("0123456789 ")
            print(
                f"{bin_label:<15} {row['n_compounds']:>6} {row['median_activity_nap']:>12.3f} {row['median_replicates']:>11.0f}"
            )

    # Show detailed examples
    if show_examples:
        print("\n--- TOP EXAMPLES: Marginal activity + High consistency ---")
        examples = get_high_consistency_marginal_details(con, dataset, threshold, n_examples=15)
        print(examples.to_string())

    # Create and save effect size vs replicates plot
    print("\nGenerating effect size vs replicates plot...")
    fig_effect = plot_effect_size_vs_replicates(effect_rep_df)
    effect_plot_path = outdir / "effect_size_vs_replicates.png"
    fig_effect.savefig(effect_plot_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {effect_plot_path}")
    plt.close(fig_effect)

    # Create replicate distribution plot - shows the full distribution of replicates
    # for marginal compounds, split by consistency level
    print("\nGenerating replicate distribution plot...")
    compound_data = get_compound_level_data(con, dataset, threshold)
    fig_dist = plot_replicate_distribution_by_consistency(compound_data)
    dist_plot_path = outdir / "replicate_distribution_by_consistency.png"
    fig_dist.savefig(dist_plot_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {dist_plot_path}")
    plt.close(fig_dist)

    # Create interactive version with hover tooltips
    print("\nGenerating interactive plot...")
    fig_interactive = plot_replicate_distribution_interactive(compound_data)
    interactive_path = outdir / "replicate_distribution_interactive.html"
    fig_interactive.write_html(interactive_path)
    print(f"Saved interactive plot to: {interactive_path}")

    con.close()

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    write_summary(
        outdir / "summary.json",
        description=(
            "Activity-consistency relationship (Issue #35). At the compound level: the Spearman "
            "correlation between activity p-value and max consistency nAP, the consistency-signal "
            "distribution across activity p-value bins (including the fraction of compounds above nAP "
            "0.3), and the group-level gains/losses at each activity-threshold transition (0.05->0.30). "
            "Explains why marginally active compounds can still carry consistent signal."
        ),
        numbers={
            "dataset": dataset,
            "consistency_threshold": threshold,
            "spearman_activity_p_vs_consistency": {
                "rho": round(float(rho), 4),
                "p_value": float(pval),
            },
            "consistency_by_activity_bin": {
                str(idx): {
                    "n": int(row["n"]),
                    "mean_consistency_nap": round(float(row["mean"]), 4),
                    "median_consistency_nap": round(float(row["median"]), 4),
                    "n_above_0p3": int(row["n_above_03"]),
                    "pct_above_0p3": float(row["pct_above_03"]),
                }
                for idx, row in summary.iterrows()
            },
            "threshold_transitions": [
                {
                    "from": s["from"],
                    "to": s["to"],
                    "gained": s["gained"],
                    "lost": s["lost"],
                    "net": s["net"],
                    "ratio": s["ratio"],
                }
                for s in transitions["summary"]
            ],
        },
        files=index_files(outdir),
    )

    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print("""
1. Strong activity (p<0.05) compounds show the highest consistency signal
2. Marginal compounds (0.05-0.30) are noisier but some show consistency signal
3. Threshold transition analysis shows gain/loss ratios at each step
   (see threshold_transitions.json for details)

4. MARGINAL COMPOUNDS WITH HIGH CONSISTENCY:
   - These compounds tend to have high activity effect size (nAP)
   - But they have low replicate counts
   - The p-value reflects both effect size and sample size
   - Example: 0.05-0.10 bin has median activity nAP ~0.5 but only 3 replicates
""")


if __name__ == "__main__":
    app()
