"""Visualization utilities for JUMP production pipeline.

Provides plotting functions, visualization constants, and statistical helpers
for activity and consistency analysis.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats

from jump_production.config import DEFAULT_DPI

# =============================================================================
# Visualization Constants
# =============================================================================

# Significance coloring (used across notebooks for activity analysis)
SIG_COLOR = "#2171b5"  # Blue for significant compounds
NONSIG_COLOR = "#a0a0a0"  # Gray for non-significant compounds

# Annotation box style
BBOX_STYLE = {
    "boxstyle": "round,pad=0.3",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.8,
}

# Standard column names for activity analysis
MAP_COL = "mean_normalized_average_precision"
SIG_COL = "below_corrected_p"


@dataclass
class TraitConfig:
    """Configuration for a compound trait in scatter plots.

    Attributes:
        col: Column name in DataFrame (e.g., "Metadata_LogP")
        label: Axis label (e.g., "LogP")
        xlim: X-axis limits as (min, max)
        panel_label: Panel label for combined figures (e.g., "A. LogP")
    """

    col: str
    label: str
    xlim: tuple[float, float]
    panel_label: str = ""


# =============================================================================
# Statistical Helpers
# =============================================================================


def split_by_significance(
    df: pd.DataFrame,
    sig_col: str = SIG_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into significant and non-significant subsets.

    Args:
        df: DataFrame with significance column
        sig_col: Column name for significance flag (boolean)

    Returns:
        Tuple of (significant_df, non_significant_df)
    """
    return df[df[sig_col]], df[~df[sig_col]]


def compute_activity_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    sig_col: str = SIG_COL,
) -> dict:
    """Compute Spearman correlation and Mann-Whitney U test for trait vs activity.

    Args:
        df: DataFrame with trait, activity, and significance columns
        x_col: Column name for the trait (x-axis)
        y_col: Column name for activity (y-axis)
        sig_col: Column name for significance flag

    Returns:
        Dict with spearman_r, spearman_p, mw_stat, mw_p, n
    """
    valid = df[[x_col, y_col, sig_col]].dropna()
    sig, nonsig = split_by_significance(valid, sig_col)

    r, p = stats.spearmanr(valid[x_col], valid[y_col])
    mw_stat, mw_p = stats.mannwhitneyu(
        sig[x_col].dropna(),
        nonsig[x_col].dropna(),
        alternative="two-sided",
    )

    return {
        "spearman_r": r,
        "spearman_p": p,
        "mw_stat": mw_stat,
        "mw_p": mw_p,
        "n": len(valid),
        "n_sig": len(sig),
        "n_nonsig": len(nonsig),
    }


def compute_partial_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariate_cols: list[str],
) -> dict:
    """Compute partial Spearman correlation controlling for covariates.

    Uses residualization approach: regress out covariates from both x and y,
    then compute Spearman correlation of residuals.

    Args:
        df: DataFrame with all columns
        x_col: Column for x variable
        y_col: Column for y variable
        covariate_cols: Columns to control for

    Returns:
        Dict with partial_r and n
    """
    from sklearn.linear_model import LinearRegression

    cols_needed = [x_col, y_col] + covariate_cols
    valid = df[cols_needed].dropna()
    if len(valid) < 10:
        return {"partial_r": np.nan, "n": len(valid)}

    x = valid[x_col].values
    y = valid[y_col].values
    covariates = valid[covariate_cols].values

    # Residualize x
    model_x = LinearRegression().fit(covariates, x)
    x_resid = x - model_x.predict(covariates)

    # Residualize y
    model_y = LinearRegression().fit(covariates, y)
    y_resid = y - model_y.predict(covariates)

    # Correlation of residuals
    r, _ = stats.spearmanr(x_resid, y_resid)

    return {"partial_r": r, "n": len(valid)}


def format_p_value(p: float) -> str:
    """A p-value for a plot annotation. One too small for a float is reported as a bound, not as zero."""
    if p == 0:
        return "p < 1e-300"
    return f"p < {p:.0e}" if p < 0.001 else f"p = {p:.3f}"


def format_stat_annotation(
    spearman_r: float,
    mw_p: float,
    n: int | None = None,
    compact: bool = True,
) -> str:
    """Format Spearman correlation and Mann-Whitney test for plot annotation.

    Args:
        spearman_r: Spearman correlation coefficient
        mw_p: Mann-Whitney U test p-value
        n: Sample size (optional)
        compact: If True, use shorter format (rho vs Spearman rho)

    Returns:
        Formatted annotation string
    """
    mw_str = format_p_value(mw_p)

    if compact:
        text = f"\u03c1 = {spearman_r:.2f}\nMW {mw_str}"
    else:
        text = f"Spearman \u03c1 = {spearman_r:.2f}\nMann-Whitney {mw_str}"

    if n is not None:
        text += f"\nn = {n:,}"

    return text


# =============================================================================
# Figure Output
# =============================================================================


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int = DEFAULT_DPI) -> list[Path]:
    """Save one view as matching opaque PNG and PDF assets, then close the figure.

    Both files use the same figure dimensions and an explicit white background, so a panel
    dropped into the manuscript looks the same as the rendered PNG regardless of the viewer's
    theme. Returns the two paths in PNG, PDF order for a summary's `files` manifest.

    Figures needing a different contract (tight cropping, scrubbed PDF metadata) should keep
    their own saver rather than adding options here.
    """
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=dpi, facecolor="white", transparent=False)
    figure.savefig(paths[1], dpi=dpi, facecolor="white", transparent=False)
    plt.close(figure)
    return paths


# =============================================================================
# Plot Functions
# =============================================================================


def plot_categorical_box(
    df: pd.DataFrame,
    cat_col: str,
    y_col: str = MAP_COL,
    order: list | None = None,
    labels: list[str] | None = None,
    colors: list[str] | dict | None = None,
    xlabel: str = "",
    ylabel: str = "Normalized mAP",
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "categorical_box.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Generic box plot of activity by categorical variable.

    Supports both standalone (creates figure, saves to file) and embedded
    (draws on provided axes) modes.

    Args:
        df: DataFrame with categorical and activity columns
        cat_col: Column name for categorical variable
        y_col: Column name for activity (y-axis)
        order: Order of categories (auto-detected if None)
        labels: Display labels for categories (uses values if None)
        colors: Color(s) for boxes (list or dict mapping values to colors)
        xlabel: X-axis label
        ylabel: Y-axis label
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        filename: Output filename (without path)
        save_pdf: If True, also save PDF version
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))

    # Prepare data
    valid = df[[cat_col, y_col]].dropna()

    # Determine order
    if order is None:
        order = sorted(valid[cat_col].unique())

    # Set up colors
    if colors is None:
        palette = "#3498db"
    elif isinstance(colors, dict):
        palette = colors
    else:
        palette = dict(zip(order, colors))

    # Create box plot
    sns.boxplot(
        data=valid,
        x=cat_col,
        y=y_col,
        order=order,
        hue=cat_col if isinstance(palette, dict) else None,
        palette=palette if isinstance(palette, dict) else None,
        color=palette if isinstance(palette, str) else None,
        legend=False,
        width=0.6,
        ax=ax,
    )

    # Set x-tick labels
    if labels is not None:
        ax.set_xticklabels(labels)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Add n counts above each box
    y_max = valid[y_col].max()
    for i, val in enumerate(order):
        n = int((valid[cat_col] == val).sum())
        n_str = f"{n // 1000}k" if n >= 1000 else str(n)
        ax.annotate(n_str, xy=(i, y_max + 0.08), ha="center", fontsize=8, color="gray")
    ax.set_ylim(None, y_max + 0.15)

    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_scatter_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    sig_col: str = SIG_COL,
    x_label: str = "",
    y_label: str = "Normalized mAP",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] = (0, 1),
    title: str = "",
    panel_label: str = "",
    fig: plt.Figure | None = None,
    gs_parent=None,
    output_dir: Path | None = None,
    filename: str = "scatter_marginals.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Scatter plot with marginal histograms, colored by significance.

    Supports both standalone and embedded modes:
    - Standalone (gs_parent=None): creates figure, saves to output_dir
    - Embedded (gs_parent provided): draws into parent GridSpec slot

    Args:
        df: DataFrame with x_col, y_col, and sig_col columns
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values (default: normalized mAP)
        sig_col: Column name for significance flag (boolean)
        x_label: X-axis label
        y_label: Y-axis label
        x_lim: X-axis limits as (min, max). Auto-computed if None.
        y_lim: Y-axis limits as (min, max). Default (0, 1) for normalized mAP.
        title: Plot title (standalone mode only)
        panel_label: Panel label for embedded mode (e.g., "A. LogP")
        fig: Parent figure (required for embedded mode)
        gs_parent: GridSpec slot to embed into. If None, creates standalone figure.
        output_dir: Output directory (standalone mode only)
        filename: Output filename (standalone mode only)
        save_pdf: If True, also save PDF version (standalone mode only)

    Returns:
        Dict with computed statistics: spearman_r, spearman_p, mw_stat, mw_p, n, n_sig, n_nonsig
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    standalone = gs_parent is None

    # Prepare data
    valid = df[[x_col, y_col, sig_col]].dropna()
    sig, nonsig = split_by_significance(valid, sig_col)

    # Compute statistics
    activity_stats = compute_activity_stats(df, x_col, y_col, sig_col)

    # Auto-compute x limits if not provided
    if x_lim is None:
        x_min, x_max = valid[x_col].quantile(0.01), valid[x_col].quantile(0.99)
        x_lim = (x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))

    # Create figure/gridspec based on mode
    if standalone:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    else:
        gs = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_parent, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05
        )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Styling differs between standalone and embedded
    scatter_alpha = (0.3, 0.5) if standalone else (0.2, 0.4)
    scatter_size = (6, 8) if standalone else (4, 6)
    font_size = 9 if standalone else 7
    n_bins = 50 if standalone else 40

    # Main scatter plot
    ax_main.scatter(
        nonsig[x_col],
        nonsig[y_col],
        c=NONSIG_COLOR,
        alpha=scatter_alpha[0],
        s=scatter_size[0],
        linewidths=0,
        rasterized=True,
    )
    ax_main.scatter(
        sig[x_col],
        sig[y_col],
        c=SIG_COLOR,
        alpha=scatter_alpha[1],
        s=scatter_size[1],
        linewidths=0,
        rasterized=True,
    )

    ax_main.set_xlabel(x_label, fontsize=font_size)
    ax_main.set_ylabel(y_label, fontsize=font_size)
    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.tick_params(labelsize=font_size - 2)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Top marginal: x distribution by significance
    bins_x = np.linspace(x_lim[0], x_lim[1], n_bins)
    ax_histx.hist(
        nonsig[x_col],
        bins=bins_x,
        alpha=0.5,
        color=NONSIG_COLOR,
        density=True,
        label="Not significant" if standalone else None,
    )
    ax_histx.hist(
        sig[x_col],
        bins=bins_x,
        alpha=0.7,
        color=SIG_COLOR,
        density=True,
        label="Significant" if standalone else None,
    )
    ax_histx.tick_params(labelbottom=False, labelsize=font_size - 3)
    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

    if standalone:
        ax_histx.set_ylabel("Density")
        # Above the marginal: inside it the legend covered the histogram's peak.
        ax_histx.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=2, fontsize=8, frameon=False)
    else:
        ax_histx.spines["left"].set_visible(False)
        ax_histx.set_yticks([])
        if panel_label:
            ax_histx.set_title(panel_label, fontsize=10, fontweight="bold", loc="left")

    # Right marginal: y distribution by significance
    bins_y = np.linspace(y_lim[0], y_lim[1], n_bins)
    ax_histy.hist(
        nonsig[y_col],
        bins=bins_y,
        alpha=0.5,
        color=NONSIG_COLOR,
        orientation="horizontal",
        density=True,
    )
    ax_histy.hist(
        sig[y_col],
        bins=bins_y,
        alpha=0.7,
        color=SIG_COLOR,
        orientation="horizontal",
        density=True,
    )
    ax_histy.tick_params(labelleft=False, labelsize=font_size - 3)
    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)

    if standalone:
        ax_histy.set_xlabel("Density")
    else:
        ax_histy.spines["bottom"].set_visible(False)
        ax_histy.set_xticks([])

    # Statistics annotation
    annotation = format_stat_annotation(
        activity_stats["spearman_r"],
        activity_stats["mw_p"],
        n=activity_stats["n"] if standalone else None,
        compact=not standalone,
    )
    if standalone:
        # One line above the legend: a box inside the axes covered scatter points.
        ax_histx.annotate(
            annotation.replace("\n", "; "),
            xy=(0.0, 1.3),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=font_size - 1,
        )
    else:
        ax_main.annotate(
            annotation,
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=font_size,
            bbox=BBOX_STYLE,
        )

    # Standalone mode: add title and save
    if standalone:
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

    return activity_stats


def plot_hexbin_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = MAP_COL,
    x_label: str = "",
    y_label: str = "Normalized mAP",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] = (0, 1),
    title: str = "",
    stats_annotation: str | None = None,
    output_dir: Path | None = None,
    filename: str = "hexbin_marginals.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Hexbin density plot with marginal histograms (count-based).

    Unlike plot_scatter_with_marginals which colors by significance,
    this shows overall data density using hexagonal binning.

    Args:
        df: DataFrame with x_col and y_col columns
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values (default: normalized mAP)
        x_label: X-axis label
        y_label: Y-axis label
        x_lim: X-axis limits as (min, max). Auto-computed from 1st-99th percentile if None.
        y_lim: Y-axis limits as (min, max). Default (0, 1) for normalized mAP.
        title: Plot title
        stats_annotation: Pre-formatted stats string to display (e.g., "Spearman rho = 0.33\\nn = 1000")
        output_dir: Output directory (required for saving)
        filename: Output filename
        save_pdf: If True, also save PDF version
    """
    valid = df[[x_col, y_col]].dropna()

    # Auto-compute x limits if not provided
    if x_lim is None:
        x_lim = (valid[x_col].quantile(0.01), valid[x_col].quantile(0.99))

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_cbar = fig.add_subplot(gs[1, 2])

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main hexbin plot
    hb = ax_main.hexbin(
        valid[x_col],
        valid[y_col],
        gridsize=50,
        cmap="Blues",
        mincnt=1,
        bins="log",
        rasterized=True,
    )
    cb = fig.colorbar(hb, cax=ax_cbar)
    cb.set_label("Count (log scale)", fontsize=9)

    ax_main.set_xlabel(x_label, fontsize=10)
    ax_main.set_ylabel(y_label, fontsize=10)
    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Top marginal: x distribution (counts)
    bins_x = np.linspace(x_lim[0], x_lim[1], 50)
    ax_histx.hist(valid[x_col], bins=bins_x, color="#3498db", alpha=0.7)
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(labelbottom=False)
    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

    # Right marginal: y distribution (counts)
    bins_y = np.linspace(y_lim[0], y_lim[1], 50)
    ax_histy.hist(
        valid[y_col],
        bins=bins_y,
        orientation="horizontal",
        color="#3498db",
        alpha=0.7,
    )
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(labelleft=False)
    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)

    # Statistics annotation
    if stats_annotation:
        ax_main.annotate(
            stats_annotation,
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
        )

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    if output_dir:
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_umap_bounds(adata: sc.AnnData, iqr_k: float = 3.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits excluding outliers via IQR method.

    Uses IQR (Interquartile Range) to detect and clip extreme outliers:
    - Bounds = [Q1 - k*IQR, Q3 + k*IQR]
    - k=1.5 for standard outliers, k=3 for extreme outliers

    Args:
        adata: AnnData with X_umap in obsm
        iqr_k: IQR multiplier for outlier detection (default 3.0 for extreme)

    Returns:
        Tuple of (xlim, ylim) where each is (min, max)

    Example:
        >>> xlim, ylim = compute_umap_bounds(adata)
        >>> ax.set_xlim(xlim)
        >>> ax.set_ylim(ylim)
    """
    coords = adata.obsm["X_umap"]

    def iqr_bounds(values: np.ndarray) -> tuple[float, float]:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        return (q1 - iqr_k * iqr, q3 + iqr_k * iqr)

    xlim = iqr_bounds(coords[:, 0])
    ylim = iqr_bounds(coords[:, 1])

    return xlim, ylim


def plot_dotplot(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    count_col: str | None = None,
    sig_col: str | None = None,
    top_n: int = 40,
    title: str = "",
    xlabel: str = "Value",
    p_threshold: float = 0.05,
    size_scale: float = 15.0,
    size_range: tuple[float, float] = (30, 150),
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    filename: str = "dotplot.png",
    save_pdf: bool = False,
    dpi: int = DEFAULT_DPI,
) -> plt.Axes:
    """Create Cleveland-style dot plot with optional significance coloring.

    Points are sized by count and colored by significance (if sig_col provided).
    Horizontal grid lines aid readability. Count labels shown on right.

    Args:
        df: DataFrame with value, label, and optionally count/significance columns
        value_col: Column for x-axis values (e.g., normalized mAP, mean overlap)
        label_col: Column for y-axis labels (e.g., target name, MoA)
        count_col: Column for point sizes. If None, uniform size.
        sig_col: Column for significance (boolean). If None, all points same color.
        top_n: Number of top entries to show (sorted by value_col descending)
        title: Plot title
        xlabel: X-axis label
        p_threshold: Significance threshold for legend label
        size_scale: Multiplier for point sizes
        size_range: (min, max) for point sizes after scaling
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        filename: Output filename
        save_pdf: If True, also save PDF version

    Returns:
        The matplotlib Axes object
    """
    from matplotlib.lines import Line2D

    standalone = ax is None

    # Take top N by value
    plot_df = df.nlargest(top_n, value_col).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    if standalone:
        fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.25)))

    # Compute sizes
    if count_col is not None:
        sizes = (plot_df[count_col] * size_scale).clip(size_range[0], size_range[1])
    else:
        sizes = 60  # uniform size

    # Compute colors
    if sig_col is not None:
        colors = [SIG_COLOR if sig else NONSIG_COLOR for sig in plot_df[sig_col]]
    else:
        colors = SIG_COLOR

    y_pos = range(len(plot_df))
    ax.scatter(
        plot_df[value_col],
        y_pos,
        s=sizes,
        c=colors,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )

    # Add horizontal lines for readability
    for y in y_pos:
        ax.axhline(y=y, color="#ecf0f1", linewidth=0.5, zorder=0)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df[label_col], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)

    # Add counts as text on right
    if count_col is not None:
        x_max = plot_df[value_col].max()
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                x_max * 1.02,
                i,
                f"n={int(row[count_col])}",
                va="center",
                fontsize=8,
                color="#7f8c8d",
            )

    # Add legend for significance
    if sig_col is not None:
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SIG_COLOR, markersize=10, label=f"p<{p_threshold}"),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=NONSIG_COLOR, markersize=10, label="Not significant"
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        if output_dir:
            fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)

    return ax


# One visual system for the compound-property figures, shared by
# figure-compound-activity-associations.py and figure-opentargets-group-consistency.py
# so the two renderers cannot drift apart. Type is sized for the panel cell, not the
# canvas: the caption is transport that rides along and is cropped in the drawing tool,
# so it grows the canvas downward instead of taking height from the panels.
PROPERTY_FIGURE_WIDTH_IN = 9.6
PROPERTY_FIGURE_PANEL_HEIGHT_IN = 7.2
PROPERTY_FIGURE_COLOR = {
    "blue": "#1769A5",
    "light_blue": "#A8CCE2",
    "dark_gray": "#4B5563",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "box_fill": "#D9EBF6",
    "median": "#111827",
    "band": "#E5E7EB",
}
PROPERTY_FIGURE_TYPE = {
    "suptitle": 15.0,
    "title": 11.5,
    "axis_label": 9.5,
    "y_label": 9.5,
    "tick": 8.5,
    "annotation": 8.5,
    "legend": 9.5,
    "caption": 7.0,
}


def captions_disabled() -> bool:
    """True when JUMP_FIGURE_NO_CAPTION=1: renderers emit bare panels for a code compositor.

    The caption string is still computed and still written to `legend.md` and `summary.json`;
    only the in-figure transport band is skipped, so the numbers cannot differ between modes.
    """
    import os

    return os.environ.get("JUMP_FIGURE_NO_CAPTION", "") == "1"


def property_figure_caption_layout(
    caption: str,
    width_in: float = PROPERTY_FIGURE_WIDTH_IN,
    panel_height_in: float = PROPERTY_FIGURE_PANEL_HEIGHT_IN,
    caption_points: float = PROPERTY_FIGURE_TYPE["caption"],
) -> tuple[str, float, float]:
    """Wrap a transport caption to a canvas width and return the height it adds.

    Returns the wrapped text, the total figure height in inches, and the bottom
    margin fraction that keeps the panel block at its full height. `panel_height_in`
    is everything above the caption, including any title margin. A caption is the
    same string as its `legend.md` section and so can run long; it always grows the
    canvas downward instead of taking height from the panels.
    """
    from textwrap import fill

    if captions_disabled():
        # Composition mode: no caption band, panels keep their full height.
        return "", panel_height_in, 0.0

    # 0.75 em per character is close enough for the condensed sans this project renders in.
    columns = int(width_in * 72 / (0.75 * caption_points))
    wrapped = "\n\n".join(fill(paragraph, width=columns) for paragraph in caption.split("\n\n"))
    caption_in = len(wrapped.splitlines()) * caption_points * 1.35 / 72 + 0.22
    total_in = panel_height_in + caption_in
    return wrapped, total_in, caption_in / total_in


def attach_caption(
    figure: plt.Figure,
    caption: str,
    *,
    caption_points: float = PROPERTY_FIGURE_TYPE["caption"],
    color: str = PROPERTY_FIGURE_COLOR["gray"],
) -> plt.Figure:
    """Add a caption band below a finished figure without taking height from its panels.

    For renderers that size their own canvas, including constrained-layout ones: the current
    placement is frozen, the canvas grows downward by the caption's height, and every axis,
    figure text, and figure legend moves up by that much. Call it last, after the figure is
    complete.
    """
    if captions_disabled():
        return figure
    width_in, height_in = figure.get_size_inches()
    wrapped, total_in, fraction = property_figure_caption_layout(
        caption,
        width_in=width_in,
        panel_height_in=height_in,
        caption_points=caption_points,
    )
    # Run any layout engine once so the frozen positions are the ones it would have produced.
    figure.draw_without_rendering()
    figure.set_layout_engine("none")
    figure.set_size_inches(width_in, total_in)
    for axis in figure.axes:
        box = axis.get_position()
        axis.set_position([box.x0, fraction + box.y0 * (1 - fraction), box.width, box.height * (1 - fraction)])
    for text in figure.texts:
        x, y = text.get_position()
        text.set_position((x, fraction + y * (1 - fraction)))
    for legend in figure.legends:
        # All four coordinates: a legend placed by `loc` alone anchors to the whole figure, and
        # keeping only the corner collapsed that rectangle to the origin and threw the legend
        # off the left edge.
        box = legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted())
        legend.set_bbox_to_anchor(
            (box.x0, fraction + box.y0 * (1 - fraction), box.width, box.height * (1 - fraction)),
            transform=figure.transFigure,
        )
    figure.text(
        0.5,
        0.12 / total_in,
        wrapped,
        ha="center",
        va="bottom",
        fontsize=caption_points,
        color=color,
        linespacing=1.35,
    )
    return figure
