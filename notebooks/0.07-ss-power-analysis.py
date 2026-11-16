#!/usr/bin/env python
"""Analyze statistical power for activity significance.

Tests whether marginal activity compounds (p >= 0.10) contribute consistency signal.

Key insight: Activity significance depends on three factors:
1. Effect size (activity nAP)
2. Number of replicates
3. Number of negative controls per replicate

This analysis investigates whether high-consistency marginal compounds have high
effect sizes but insufficient statistical power (due to low replicates), or whether
they represent genuinely moderate signal that contributes to group-level consistency.

Usage:
    pixi run python notebooks/0.07-ss-power-analysis.py
    pixi run python notebooks/0.07-ss-power-analysis.py --dataset compound_no_source7

Outputs:
    {outdir}/power_analysis.png - Main visualization
    {outdir}/power_analysis_summary.csv - Summary statistics

"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch

from jump_production.config import DEFAULT_DPI
from jump_production.summaries import index_files, write_summary

app = typer.Typer()


def compute_consistency_threshold(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    target_significance: float = 0.05,
    percentile: float = 50,
) -> float:
    """Compute empirical consistency threshold from significant targets.

    The threshold is derived from the distribution of target-level nAP for
    targets that passed the consistency significance test. At median, this
    tells us "what does a consistent target's nAP look like?"

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name
    target_system : str
        Target system (e.g., 'repurposing')
    consistency_threshold : float
        Activity threshold used for consistency analysis
    target_significance : float
        P-value threshold for target significance (default: 0.05)
    percentile : float
        Percentile to use for threshold (default: 50 = median)

    Returns
    -------
    float
        Empirical consistency threshold

    Notes
    -----
    This empirical threshold differs from the fixed exploratory threshold of 0.3
    used in notebook 0.06. The fixed 0.3 is stricter, useful for identifying only
    the strongest consistency signals, while this data-driven threshold captures
    the typical consistency level of significant targets.
    """
    query = f"""
    SELECT mean_normalized_average_precision as target_nap
    FROM consistency_results
    WHERE _preprocessing = 'consistency_no_target2_sweep'
      AND _filter = 'all_sources'
      AND _activity_threshold = {consistency_threshold}
      AND _dataset = '{dataset}'
      AND _group_type = '{target_system}'
      AND corrected_p_value < {target_significance}
    """
    df = con.execute(query).df()

    if len(df) == 0:
        # Fallback if no significant targets
        return 0.12

    return float(np.percentile(df["target_nap"], percentile))


def get_compound_power_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    consistency_threshold: float = 0.30,
    max_median_neg_pairs: int = 40,
) -> pd.DataFrame:
    """Get compound-level data with activity, consistency, and power metrics.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name (e.g., 'compound_no_source7')
    consistency_threshold : float
        Activity threshold used for consistency analysis (more lenient than
        activity significance threshold to include marginal compounds)
    max_median_neg_pairs : int
        Filter to compounds with median neg pairs <= this value (removes outliers)

    Returns
    -------
    pd.DataFrame
        Compound-level data with columns:
        - Metadata_JCP2022, activity_p, activity_nap, n_replicates, median_neg_pairs
        - consistency_median, consistency_std, n_targets
    """
    query = f"""
    WITH scores_agg AS (
        SELECT
            Metadata_JCP2022,
            COUNT(*) as n_replicates,
            MEDIAN(n_total_pairs - n_pos_pairs) as median_neg_pairs
        FROM activity_scores
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
        GROUP BY Metadata_JCP2022
    ),
    consistency_agg AS (
        SELECT
            Metadata_JCP2022,
            MEDIAN(normalized_average_precision) as consistency_median,
            STDDEV(normalized_average_precision) as consistency_std,
            MAX(normalized_average_precision) as consistency_max,
            COUNT(*) as n_targets
        FROM consistency_scores
        WHERE _preprocessing = 'consistency_no_target2_sweep'
          AND _filter = 'all_sources'
          AND _activity_threshold = {consistency_threshold}
          AND _dataset = '{dataset}'
        GROUP BY Metadata_JCP2022
    )
    SELECT
        r.Metadata_JCP2022,
        r.corrected_p_value as activity_p,
        r.mean_normalized_average_precision as activity_nap,
        -log10(r.corrected_p_value) as neg_log10_p,
        s.n_replicates,
        s.median_neg_pairs,
        c.consistency_median,
        c.consistency_std,
        c.consistency_max,
        c.n_targets
    FROM activity_results r
    JOIN scores_agg s ON r.Metadata_JCP2022 = s.Metadata_JCP2022
    JOIN consistency_agg c ON r.Metadata_JCP2022 = c.Metadata_JCP2022
    WHERE r._preprocessing = 'activity_no_target2'
      AND r._filter = 'all_sources'
      AND r._activity_params = 'default'
      AND r._dataset = '{dataset}'
      AND s.median_neg_pairs <= {max_median_neg_pairs}
    """
    return con.execute(query).df()


def get_target_consistency_stats(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    target_significance: float = 0.05,
) -> pd.DataFrame:
    """Get target-level consistency statistics (p-value, nMAP, significance).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name
    target_system : str
        Target system from _group_type column
    consistency_threshold : float
        Activity threshold used for consistency analysis
    target_significance : float
        P-value threshold for target significance

    Returns
    -------
    pd.DataFrame
        Target-level stats: target_name, target_p, target_nap, target_significant, n_perturbations
    """
    query = f"""
    SELECT
        group_value as target_name,
        corrected_p_value as target_p,
        mean_normalized_average_precision as target_nap,
        corrected_p_value < {target_significance} as target_significant,
        n_perturbations
    FROM consistency_results
    WHERE _preprocessing = 'consistency_no_target2_sweep'
      AND _filter = 'all_sources'
      AND _activity_threshold = {consistency_threshold}
      AND _dataset = '{dataset}'
      AND _group_type = '{target_system}'
    """
    return con.execute(query).df()


def get_target_level_power_data(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    target_system: str = "repurposing",
    consistency_threshold: float = 0.30,
    max_median_neg_pairs: int = 40,
) -> pd.DataFrame:
    """Get target-level data with one row per compound-target pair.

    Unlike get_compound_power_data which aggregates consistency across targets,
    this function preserves the compound-target pair granularity to enable
    per-target analysis.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Database connection
    dataset : str
        Dataset name (e.g., 'compound_no_source7')
    target_system : str
        Target system from _group_type column (e.g., 'repurposing', 'chembl_target')
    consistency_threshold : float
        Activity threshold used for consistency analysis
    max_median_neg_pairs : int
        Filter to compounds with median neg pairs <= this value (removes outliers)

    Returns
    -------
    pd.DataFrame
        Target-level data with columns:
        - Metadata_JCP2022, target_name, activity_p, activity_nap, n_replicates,
          consistency_nap
    """
    query = f"""
    WITH scores_agg AS (
        SELECT
            Metadata_JCP2022,
            COUNT(*) as n_replicates,
            MEDIAN(n_total_pairs - n_pos_pairs) as median_neg_pairs
        FROM activity_scores
        WHERE _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources'
          AND _activity_params = 'default'
          AND _dataset = '{dataset}'
        GROUP BY Metadata_JCP2022
    )
    SELECT
        c.Metadata_JCP2022,
        c.group_value as target_name,
        r.corrected_p_value as activity_p,
        r.mean_normalized_average_precision as activity_nap,
        s.n_replicates,
        c.normalized_average_precision as consistency_nap
    FROM consistency_scores c
    JOIN activity_results r ON c.Metadata_JCP2022 = r.Metadata_JCP2022
    JOIN scores_agg s ON c.Metadata_JCP2022 = s.Metadata_JCP2022
    WHERE c._preprocessing = 'consistency_no_target2_sweep'
      AND c._filter = 'all_sources'
      AND c._activity_threshold = {consistency_threshold}
      AND c._group_type = '{target_system}'
      AND c._dataset = '{dataset}'
      AND r._preprocessing = 'activity_no_target2'
      AND r._filter = 'all_sources'
      AND r._activity_params = 'default'
      AND r._dataset = '{dataset}'
      AND s.median_neg_pairs <= {max_median_neg_pairs}
    """
    return con.execute(query).df()


def plot_power_analysis(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    activity_threshold: float = 0.10,
    replicate_cap: int | None = 6,
    replicate_floor: int | None = 3,
    figsize: tuple[int, int] = (12, 5),
    show_legend: bool = True,
    consistency_col: str = "consistency_max",
    consistency_high_threshold: float = 0.3,
    title: str | None = None,
    show_counts_in_legend: bool = True,
) -> plt.Figure | plt.Axes:
    """Single focused scatter: replicates vs effect size.

    Creates a visualization showing compounds in four categories:
    - Gray: not significant, low consistency (background)
    - Green: significant, low consistency (background)
    - Blue: significant AND high consistency (foreground)
    - Red: not significant, high consistency (foreground)

    Blue and red are shown with larger markers for direct comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Output from get_compound_power_data or get_target_level_power_data
    ax : plt.Axes | None
        Axes to plot on. If None, creates new figure.
    activity_threshold : float
        Activity p-value threshold for significance
    replicate_cap : int | None
        Cap replicates at this value (compounds with more are binned here). None for no cap.
    replicate_floor : int | None
        Floor replicates at this value (compounds with fewer are binned here). None for no floor.
    figsize : tuple[int, int]
        Figure size (only used when ax is None)
    show_legend : bool
        Whether to show legend (useful to disable in faceted plots)
    consistency_col : str
        Column name for consistency values (e.g., 'consistency_max' or 'consistency_nap')
    consistency_high_threshold : float
        Threshold for high consistency classification
    title : str | None
        Optional title for the axes
    show_counts_in_legend : bool
        Whether to show counts in legend labels (disable for faceted plots)

    Returns
    -------
    plt.Figure | plt.Axes
        Figure if ax was None, otherwise the axes
    """
    df = df.copy()
    df["significant"] = df["activity_p"] < activity_threshold
    df["high_consistency"] = df[consistency_col] >= consistency_high_threshold

    # Four mutually exclusive categories
    df["sig_high_cons"] = df["significant"] & df["high_consistency"]
    df["sig_low_cons"] = df["significant"] & (~df["high_consistency"])
    df["nonsig_high_cons"] = (~df["significant"]) & df["high_consistency"]
    df["nonsig_low_cons"] = (~df["significant"]) & (~df["high_consistency"])

    # Bin replicates: floor (low values) and cap (high values)
    df["n_replicates_plot"] = df["n_replicates"].copy()
    if replicate_floor is not None:
        df["n_replicates_plot"] = df["n_replicates_plot"].clip(lower=replicate_floor)
    if replicate_cap is not None:
        df["n_replicates_plot"] = df["n_replicates_plot"].clip(upper=replicate_cap)

    # Create figure if no axes provided
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Separate groups
    sig_high = df[df["sig_high_cons"]]
    sig_low = df[df["sig_low_cons"]]
    nonsig_high = df[df["nonsig_high_cons"]]
    nonsig_low = df[df["nonsig_low_cons"]]

    # Jitter function for discrete x-values
    rng = np.random.default_rng(42)

    def jitter(x, scale=0.25):
        return x + rng.uniform(-scale, scale, len(x))

    # Build labels with or without counts
    if show_counts_in_legend:
        label_sig_low = f"Significant, low consistency (n={len(sig_low)})"
        label_nonsig_low = f"Not significant, low consistency (n={len(nonsig_low)})"
        label_sig_high = f"Significant, high consistency (n={len(sig_high)})"
        label_nonsig_high = f"Not significant, high consistency (n={len(nonsig_high)})"
    else:
        label_sig_low = "Significant, low consistency"
        label_nonsig_low = "Not significant, low consistency"
        label_sig_high = f"Significant, high consistency (nAP ≥ {consistency_high_threshold:.2f})"
        label_nonsig_high = f"Not significant, high consistency (nAP ≥ {consistency_high_threshold:.2f})"

    # All dots same size for cleaner visualization
    dot_size = 30

    # Background: significant, low-consistency (green)
    ax.scatter(
        jitter(sig_low["n_replicates_plot"]),
        sig_low["activity_nap"],
        c="#2e8b57",
        s=dot_size,
        alpha=0.6,
        edgecolor="none",
        label=label_sig_low,
    )

    # Background: non-significant, low-consistency (gray)
    ax.scatter(
        jitter(nonsig_low["n_replicates_plot"]),
        nonsig_low["activity_nap"],
        c="#888888",
        s=dot_size,
        alpha=0.6,
        edgecolor="none",
        label=label_nonsig_low,
    )

    # Foreground: significant + high consistency (blue)
    ax.scatter(
        jitter(sig_high["n_replicates_plot"]),
        sig_high["activity_nap"],
        c="#3498db",
        s=dot_size,
        alpha=0.8,
        edgecolor="none",
        zorder=10,
        label=label_sig_high,
    )

    # Foreground: not significant + high consistency (red)
    ax.scatter(
        jitter(nonsig_high["n_replicates_plot"]),
        nonsig_high["activity_nap"],
        c="#e74c3c",
        s=dot_size,
        alpha=0.8,
        edgecolor="none",
        zorder=11,
        label=label_nonsig_high,
    )

    ax.set_xlabel("Number of Replicates")
    ax.set_ylabel("Activity Effect Size (nMAP)")

    # Set x-axis limits based on floor and cap
    x_min = (replicate_floor - 0.5) if replicate_floor else 1.5
    x_max = (replicate_cap + 0.5) if replicate_cap else 20
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)

    # Set x-axis ticks with "≤N" for floor and "N+" for cap
    if replicate_floor is not None or replicate_cap is not None:
        floor_val = replicate_floor or 2
        cap_val = replicate_cap or 20
        ticks = list(range(floor_val, cap_val + 1))
        tick_labels = []
        for t in ticks:
            if replicate_floor is not None and t == replicate_floor:
                tick_labels.append(f"≤{t}")
            elif replicate_cap is not None and t == replicate_cap:
                tick_labels.append(f"{t}+")
            else:
                tick_labels.append(str(t))
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Minimal styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title if provided
    if title:
        ax.set_title(title, fontsize=10)

    # Legend at bottom (only for standalone figure)
    if show_legend and created_figure:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,  # 4 items in 2 columns = 2 rows
            frameon=False,
            fontsize=8,
        )

    if created_figure:
        plt.tight_layout()
        return fig
    return ax


def plot_power_analysis_by_target(
    df: pd.DataFrame,
    target_stats: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    max_targets: int | None = None,
    min_compounds: int = 5,
    ncols: int = 8,
    activity_threshold: float = 0.10,
    replicate_cap: int | None = 6,
    replicate_floor: int | None = 3,
    consistency_high_threshold: float = 0.3,
    panel_size: tuple[float, float] = (3.0, 2.5),
) -> plt.Figure:
    """Create faceted power analysis plot with one panel per target.

    Parameters
    ----------
    df : pd.DataFrame
        Output from get_target_level_power_data with target_name column
    target_stats : pd.DataFrame | None
        Output from get_target_consistency_stats with target-level p-value/nMAP.
        If provided, adds significance indicator and nMAP to panel titles.
    targets : list[str] | None
        Specific targets to show. If None, selects by min_compounds/max_targets.
    max_targets : int | None
        Maximum number of targets to show. None means no limit.
    min_compounds : int
        Minimum compounds per target to include (default: 5)
    ncols : int
        Number of columns in the subplot grid
    activity_threshold : float
        Activity p-value threshold for significance
    replicate_cap : int | None
        Cap replicates at this value (compounds with more are binned here)
    replicate_floor : int | None
        Floor replicates at this value (compounds with fewer are binned here)
    consistency_high_threshold : float
        Threshold for high consistency classification
    panel_size : tuple[float, float]
        Size of each panel (width, height) in inches

    Returns
    -------
    plt.Figure
        Matplotlib figure with faceted scatter plots
    """
    # Select targets if not specified
    if targets is None:
        target_counts = df.groupby("target_name")["Metadata_JCP2022"].nunique()
        # Filter by min_compounds
        target_counts = target_counts[target_counts >= min_compounds]
        # Sort by count descending
        target_counts = target_counts.sort_values(ascending=False)
        # Apply max_targets limit if specified
        if max_targets is not None:
            target_counts = target_counts.head(max_targets)
        targets = target_counts.index.tolist()

    # Filter to selected targets
    df_filtered = df[df["target_name"].isin(targets)].copy()

    # Build target stats lookup if provided
    stats_lookup = {}
    if target_stats is not None:
        for _, row in target_stats.iterrows():
            stats_lookup[row["target_name"]] = {
                "p": row["target_p"],
                "nap": row["target_nap"],
                "sig": row["target_significant"],
            }

    # Calculate grid dimensions
    n_targets = len(targets)
    nrows = (n_targets + ncols - 1) // ncols

    # Create figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    # Plot each target
    for i, target in enumerate(targets):
        target_df = df_filtered[df_filtered["target_name"] == target]
        n_compounds = target_df["Metadata_JCP2022"].nunique()

        # Build title with target stats if available
        # Use background color to highlight significant targets
        if target in stats_lookup:
            stats = stats_lookup[target]
            # Format p-value: use scientific notation if very small
            p_val = stats["p"]
            if p_val < 0.001:
                p_str = f"p={p_val:.1e}"
            else:
                p_str = f"p={p_val:.3f}"
            title = f"{target}\nn={n_compounds}, nMAP={stats['nap']:.2f}, {p_str}"
            if stats["sig"]:
                # Light blue background for significant targets
                axes[i].set_facecolor("#e8f4f8")
        else:
            title = f"{target}\nn={n_compounds}"

        plot_power_analysis(
            target_df,
            ax=axes[i],
            activity_threshold=activity_threshold,
            replicate_cap=replicate_cap,
            replicate_floor=replicate_floor,
            show_legend=False,
            consistency_col="consistency_nap",
            consistency_high_threshold=consistency_high_threshold,
            title=title,
            show_counts_in_legend=False,
        )

        # Reduce label clutter for inner panels
        if i % ncols != 0:
            axes[i].set_ylabel("")
        if i < (nrows - 1) * ncols:
            axes[i].set_xlabel("")

    # Hide unused axes
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)

    # Add shared legend at bottom (4 items in 2 columns = 2 rows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Room for legend
    return fig


def plot_target_consistency_scatter(
    target_stats: pd.DataFrame,
    figsize: tuple[float, float] = (12, 7),
    jitter_scale: float = 0.25,
    compound_bins: list[int] | None = None,
    show_boxplots: bool = True,
) -> plt.Figure:
    """Create scatter plot showing target consistency power relationship.

    Shows how target-level consistency significance depends on both:
    1. Number of compounds (statistical power) - X-axis (binned)
    2. nMAP (effect size) - Y-axis

    Key insight: "More compounds → more likely significant, but nMAP also matters"

    Parameters
    ----------
    target_stats : pd.DataFrame
        Output from get_target_consistency_stats with columns:
        target_name, target_p, target_nap, target_significant, n_perturbations
    figsize : tuple
        Figure size in inches
    jitter_scale : float
        Scale of jitter to add to x-axis (as fraction of bin width)
    compound_bins : list[int] | None
        Bin edges for compound counts. Default: [2, 3, 4, 5, 7, 10, 15, 100]
    show_boxplots : bool
        Whether to show box plots behind the scatter points

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    df = target_stats.copy()

    # Default bins: 2, 3, 4, 5-6, 7-9, 10-14, 15+
    if compound_bins is None:
        compound_bins = [2, 3, 4, 5, 7, 10, 15, 100]

    # Create bin labels and assign bins
    bin_labels = []
    for i in range(len(compound_bins) - 1):
        lo, hi = compound_bins[i], compound_bins[i + 1]
        if hi - lo == 1:
            bin_labels.append(str(lo))
        elif hi == 100:
            bin_labels.append(f"{lo}+")
        else:
            bin_labels.append(f"{lo}-{hi - 1}")

    df["compound_bin"] = pd.cut(
        df["n_perturbations"],
        bins=compound_bins,
        labels=bin_labels,
        right=False,
    )

    # Drop any rows that didn't fit in bins (e.g., n_perturbations < 2)
    df = df.dropna(subset=["compound_bin"])

    # Create numeric x positions for bins (with offset for side-by-side box plots)
    bin_positions = {label: i for i, label in enumerate(bin_labels)}
    df["x_pos"] = df["compound_bin"].map(bin_positions).astype(float)

    # Offset for side-by-side: non-significant left (-0.27), all center (0), significant right (+0.27)
    box_offset = 0.27
    df["x_offset"] = df["target_significant"].map({False: -box_offset, True: box_offset})

    # Add jitter within each side
    rng = np.random.default_rng(42)
    jitter_within_box = jitter_scale * 0.5  # Narrower jitter to stay within box
    df["x_jitter"] = (
        df["x_pos"].values + df["x_offset"].values + rng.uniform(-jitter_within_box, jitter_within_box, len(df))
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Separate significant and non-significant
    sig_mask = df["target_significant"]
    nonsig = df[~sig_mask]
    sig = df[sig_mask]

    # Draw side-by-side box plots if requested
    if show_boxplots:
        box_width = 0.24

        # Non-significant box plots (left, gray)
        nonsig_data = []
        nonsig_positions = []
        for i, label in enumerate(bin_labels):
            bin_data = nonsig[nonsig["compound_bin"] == label]["target_nap"].values
            if len(bin_data) > 0:
                nonsig_data.append(bin_data)
                nonsig_positions.append(i - box_offset)

        if nonsig_data:
            bp_nonsig = ax.boxplot(
                nonsig_data,
                positions=nonsig_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for patch in bp_nonsig["boxes"]:
                patch.set_facecolor("#e0e0e0")
                patch.set_edgecolor("#999999")
                patch.set_alpha(0.8)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp_nonsig[element]:
                    line.set_color("#777777")
                    line.set_linewidth(1)

        # All targets box plots (center, light green/teal)
        all_data = []
        all_positions = []
        for i, label in enumerate(bin_labels):
            bin_data = df[df["compound_bin"] == label]["target_nap"].values
            if len(bin_data) > 0:
                all_data.append(bin_data)
                all_positions.append(i)

        if all_data:
            bp_all = ax.boxplot(
                all_data,
                positions=all_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                zorder=0,  # Behind the others
            )
            for patch in bp_all["boxes"]:
                patch.set_facecolor("#e8f5e9")
                patch.set_edgecolor("#66bb6a")
                patch.set_alpha(0.7)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp_all[element]:
                    line.set_color("#43a047")
                    line.set_linewidth(1)

        # Significant box plots (right, blue)
        sig_data = []
        sig_positions = []
        for i, label in enumerate(bin_labels):
            bin_data = sig[sig["compound_bin"] == label]["target_nap"].values
            if len(bin_data) > 0:
                sig_data.append(bin_data)
                sig_positions.append(i + box_offset)

        if sig_data:
            bp_sig = ax.boxplot(
                sig_data,
                positions=sig_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for patch in bp_sig["boxes"]:
                patch.set_facecolor("#c6dbef")
                patch.set_edgecolor("#2166ac")
                patch.set_alpha(0.8)
            for element in ["whiskers", "caps", "medians"]:
                for line in bp_sig[element]:
                    line.set_color("#2166ac")
                    line.set_linewidth(1)

    # Fixed dot size
    dot_size = 35

    # Plot non-significant first (gray, background)
    ax.scatter(
        nonsig["x_jitter"],
        nonsig["target_nap"],
        c="#969696",
        s=dot_size,
        alpha=0.6,
        edgecolors="none",
        label=f"Not significant (n={len(nonsig)})",
        zorder=5,
    )

    # Plot significant on top (blue, foreground)
    ax.scatter(
        sig["x_jitter"],
        sig["target_nap"],
        c="#2166ac",
        s=dot_size,
        alpha=0.7,
        edgecolors="none",
        label=f"Significant (n={len(sig)})",
        zorder=10,
    )

    # Add annotation with significance percentage at bottom center
    significant_ratio = df["target_significant"].mean()
    n_sig = sig_mask.sum()
    ax.text(
        0.5,
        -0.18,
        f"Overall: {100 * significant_ratio:.0f}% significant ({n_sig} of {len(df)} targets)",
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=10,
        color="#525252",
    )

    # Add counts per bin at bottom (below x-axis labels)
    for i, label in enumerate(bin_labels):
        bin_count = (df["compound_bin"] == label).sum()
        bin_sig = ((df["compound_bin"] == label) & df["target_significant"]).sum()
        sig_pct = 100 * bin_sig / bin_count if bin_count > 0 else 0
        ax.text(
            i,
            -0.12,
            f"n={bin_count}, {sig_pct:.0f}% sig",
            ha="center",
            va="top",
            fontsize=8,
            color="#666666",
            transform=ax.get_xaxis_transform(),
        )

    # Styling
    ax.set_xlabel("Number of Compounds per Target", fontsize=12)
    ax.set_ylabel("Effect Size (nMAP)", fontsize=12)
    ax.set_title("Target Consistency: Power vs Effect Size", fontsize=13)

    # Set x-axis ticks
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)

    # Y-axis: accommodate negative values if present
    y_min = min(-0.05, df["target_nap"].min() - 0.05)
    y_max = max(1.05, df["target_nap"].max() + 0.05)
    ax.set_ylim(y_min, y_max)

    # Add horizontal line at nMAP = 0 for reference
    ax.axhline(0, color="#cccccc", linestyle="-", linewidth=1, alpha=0.5)

    # Grid (only horizontal)
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend - upper right
    # Custom legend entries for the three boxplot types
    legend_elements = [
        Patch(facecolor="#e0e0e0", edgecolor="#999999", label="Not significant"),
        Patch(facecolor="#e8f5e9", edgecolor="#66bb6a", label="All"),
        Patch(facecolor="#c6dbef", edgecolor="#2166ac", label="Significant"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Room for per-bin annotations
    return fig


def compute_summary_statistics(
    df: pd.DataFrame,
    activity_threshold: float = 0.10,
    consistency_high_threshold: float = 0.15,
) -> pd.DataFrame:
    """Compute summary statistics by significance and consistency status.

    Parameters
    ----------
    df : pd.DataFrame
        Output from get_compound_power_data
    activity_threshold : float
        Activity p-value threshold
    consistency_high_threshold : float
        Threshold for high consistency classification

    Returns
    -------
    pd.DataFrame
        Summary statistics by group
    """
    df = df.copy()
    df["significant"] = df["activity_p"] < activity_threshold
    df["high_consistency"] = df["consistency_max"] >= consistency_high_threshold

    summary = (
        df.groupby(["significant", "high_consistency"])
        .agg(
            {
                "Metadata_JCP2022": "count",
                "activity_nap": ["median", "mean", "std"],
                "n_replicates": ["median", "mean"],
                "consistency_median": ["median", "mean"],
                "n_targets": ["median", "mean"],
            }
        )
        .round(3)
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={"Metadata_JCP2022_count": "n_compounds"})

    return summary.reset_index()


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset",
        "-d",
        help="Dataset name to analyze",
    ),
    activity_threshold: float = typer.Option(
        0.10,
        "--threshold",
        "-t",
        help="Activity p-value threshold for significance",
    ),
    consistency_threshold: float = typer.Option(
        0.30,
        "--consistency-threshold",
        "-c",
        help="Activity threshold for consistency data (more lenient to include marginal compounds)",
    ),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Output directory for plots (default: data/processed/exploration/0.07/{dataset})",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
    by_target: bool = typer.Option(
        False,
        "--by-target",
        help="Generate faceted plot with one panel per target",
    ),
    target_system: str = typer.Option(
        "repurposing",
        "--target-system",
        help="Target system for by-target analysis (e.g., 'repurposing', 'chembl_target')",
    ),
    max_targets: int | None = typer.Option(
        None,
        "--max-targets",
        help="Maximum number of targets to show (default: all targets meeting min-compounds)",
    ),
    min_compounds: int = typer.Option(
        5,
        "--min-compounds",
        help="Minimum compounds per target to include in faceted plot",
    ),
    ncols: int = typer.Option(
        8,
        "--ncols",
        help="Number of columns in faceted plot grid",
    ),
    replicate_cap: int = typer.Option(
        6,
        "--replicate-cap",
        help="Cap replicates at this value (higher values binned as N+)",
    ),
    replicate_floor: int = typer.Option(
        3,
        "--replicate-floor",
        help="Floor replicates at this value (lower values binned as ≤N)",
    ),
    consistency_high: float | None = typer.Option(
        None,
        "--consistency-high",
        help="Threshold for 'high consistency' (default: computed from median of significant targets)",
    ),
) -> None:
    """Analyze statistical power for activity significance."""
    # Set default outdir
    if outdir is None:
        outdir = Path(f"data/processed/exploration/0.07/{dataset}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    con = duckdb.connect("data/processed/copairs_results.duckdb", read_only=True)

    # Compute consistency threshold if not provided
    if consistency_high is None:
        consistency_high = compute_consistency_threshold(con, dataset, target_system, consistency_threshold)
        threshold_source = "computed (median of significant targets)"
    else:
        threshold_source = "user-specified"

    print(f"Dataset: {dataset}")
    print(f"Activity significance threshold: p < {activity_threshold}")
    print(f"Consistency data threshold: p < {consistency_threshold}")
    print(f"High consistency threshold: nAP >= {consistency_high:.3f} ({threshold_source})")

    # Get data - use lenient consistency threshold to include marginal compounds
    print("\nLoading data...")
    df = get_compound_power_data(con, dataset, consistency_threshold)
    print(f"Loaded {len(df):,} compounds (filtered to median_neg_pairs <= 40)")

    # Classify into 4 categories
    df["significant"] = df["activity_p"] < activity_threshold
    df["high_consistency"] = df["consistency_max"] >= consistency_high
    df["sig_high_cons"] = df["significant"] & df["high_consistency"]
    df["sig_low_cons"] = df["significant"] & (~df["high_consistency"])
    df["nonsig_high_cons"] = (~df["significant"]) & df["high_consistency"]
    df["nonsig_low_cons"] = (~df["significant"]) & (~df["high_consistency"])

    print("\n4-Category Breakdown:")
    print(f"  Significant, high consistency:     {df['sig_high_cons'].sum():>6,}  (blue)")
    print(f"  Significant, low consistency:      {df['sig_low_cons'].sum():>6,}  (green)")
    print(f"  Not significant, high consistency: {df['nonsig_high_cons'].sum():>6,}  (red)")
    print(f"  Not significant, low consistency:  {df['nonsig_low_cons'].sum():>6,}  (gray)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY BY GROUP")
    print("=" * 70)
    summary = compute_summary_statistics(df, activity_threshold, consistency_high)
    print(summary.to_string(index=False))

    # Save summary
    summary_path = outdir / "power_analysis_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # High-consistency marginal details (the red dots)
    print("\n" + "=" * 70)
    print("NOT SIGNIFICANT, HIGH CONSISTENCY COMPOUNDS (red dots)")
    print("=" * 70)
    nonsig_high = df[df["nonsig_high_cons"]].sort_values("activity_p")
    print(
        nonsig_high[
            [
                "Metadata_JCP2022",
                "activity_p",
                "activity_nap",
                "n_replicates",
                "consistency_median",
                "consistency_max",
                "n_targets",
            ]
        ].to_string(index=False)
    )

    # Create main plot
    print("\nGenerating power analysis plot...")
    fig = plot_power_analysis(
        df,
        activity_threshold=activity_threshold,
        replicate_cap=replicate_cap,
        replicate_floor=replicate_floor,
        consistency_high_threshold=consistency_high,
    )
    plot_path = outdir / "power_analysis.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
    plt.close(fig)

    # Create by-target faceted plot if requested
    if by_target:
        target_df = get_target_level_power_data(con, dataset, target_system, consistency_threshold)
        target_stats = get_target_consistency_stats(con, dataset, target_system, consistency_threshold)

        # Create target consistency scatter plot (volcano-style)
        print("\nGenerating target consistency scatter plot...")
        fig_scatter = plot_target_consistency_scatter(target_stats)
        scatter_path = outdir / "target_consistency_scatter.png"
        fig_scatter.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved scatter plot to: {scatter_path}")
        plt.close(fig_scatter)

        # Count targets meeting threshold
        target_counts = target_df.groupby("target_name")["Metadata_JCP2022"].nunique()
        n_eligible = (target_counts >= min_compounds).sum()
        n_to_plot = n_eligible if max_targets is None else min(n_eligible, max_targets)

        # Count significant targets among eligible
        eligible_targets = target_counts[target_counts >= min_compounds].index
        n_sig = target_stats[
            target_stats["target_name"].isin(eligible_targets) & target_stats["target_significant"]
        ].shape[0]

        print(f"\nGenerating by-target faceted plot ({target_system})...")
        print(f"Loaded {len(target_df):,} compound-target pairs")
        print(f"Targets with >= {min_compounds} compounds: {n_eligible}")
        print(f"  - Significant (p < 0.05): {n_sig} (✓)")
        print(f"  - Not significant: {n_eligible - n_sig} (✗)")
        print(f"Plotting {n_to_plot} targets...")

        fig_target = plot_power_analysis_by_target(
            target_df,
            target_stats=target_stats,
            max_targets=max_targets,
            min_compounds=min_compounds,
            ncols=ncols,
            activity_threshold=activity_threshold,
            replicate_cap=replicate_cap,
            replicate_floor=replicate_floor,
            consistency_high_threshold=consistency_high,
        )
        target_plot_path = outdir / "power_analysis_by_target.png"
        fig_target.savefig(target_plot_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved faceted plot to: {target_plot_path}")
        plt.close(fig_target)

    con.close()

    # Compare red vs blue dots
    sig_high = df[df["sig_high_cons"]]
    print("\n" + "=" * 70)
    print("KEY INSIGHT: RED vs BLUE COMPARISON")
    print("=" * 70)
    print(f"""
Not significant, high consistency (RED, n={len(nonsig_high)}) vs
Significant, high consistency (BLUE, n={len(sig_high)}):

                              RED (not sig)    BLUE (sig)
Activity effect size (nMAP):     {nonsig_high["activity_nap"].median():.3f}          {sig_high["activity_nap"].median():.3f}
Replicate count:                 {nonsig_high["n_replicates"].median():.0f}              {sig_high["n_replicates"].median():.0f}
Consistency (median nAP):        {nonsig_high["consistency_median"].median():.3f}          {sig_high["consistency_median"].median():.3f}

The red dots (not significant but high consistency) have:
- Lower activity effect size than blue
- Fewer replicates than blue
- But similar consistency signal

This suggests they may be underpowered detections of real biological signal.
""")

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    write_summary(
        outdir / "summary.json",
        description=(
            "Statistical-power analysis for activity significance (Issue #35). Splits compounds into "
            "four categories by activity significance x consistency, then compares the underpowered "
            "group (not significant, high consistency) against significant high-consistency compounds "
            "on effect size, replicate count, and consistency. Shows the underpowered group carries "
            "real signal but fails significance due to fewer replicates / smaller effect size."
        ),
        numbers={
            "dataset": dataset,
            "activity_threshold": activity_threshold,
            "consistency_data_threshold": consistency_threshold,
            "high_consistency_threshold_nap": round(float(consistency_high), 3),
            "n_compounds": int(len(df)),
            "category_counts": {
                "significant_high_consistency": int(df["sig_high_cons"].sum()),
                "significant_low_consistency": int(df["sig_low_cons"].sum()),
                "nonsignificant_high_consistency": int(df["nonsig_high_cons"].sum()),
                "nonsignificant_low_consistency": int(df["nonsig_low_cons"].sum()),
            },
            "high_consistency_comparison": {
                "nonsignificant": {
                    "n": int(len(nonsig_high)),
                    "median_activity_nap": round(float(nonsig_high["activity_nap"].median()), 3),
                    "median_n_replicates": float(nonsig_high["n_replicates"].median()),
                    "median_consistency_nap": round(float(nonsig_high["consistency_median"].median()), 3),
                },
                "significant": {
                    "n": int(len(sig_high)),
                    "median_activity_nap": round(float(sig_high["activity_nap"].median()), 3),
                    "median_n_replicates": float(sig_high["n_replicates"].median()),
                    "median_consistency_nap": round(float(sig_high["consistency_median"].median()), 3),
                },
            },
        },
        files=index_files(outdir),
    )


if __name__ == "__main__":
    app()
