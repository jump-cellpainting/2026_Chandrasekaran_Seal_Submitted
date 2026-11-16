"""4-way comparison of source 7 inclusion across CP and DL embeddings.

Compares activity and consistency results across 4 configurations:
- CP no_s7:   compound_no_source7
- CP with_s7: compound_with_source7
- DL no_s7:   compound_DL_CPCNN_no_source7
- DL with_s7: compound_DL_CPCNN_with_source7

Produces aggregate summaries (activity bar, consistency heatmaps, delta bar)
and per-target status change CSVs with 4-way dot plots.

Usage:
    pixi run python notebooks/1.02-ss-source7-4way-comparison.py
    pixi run python notebooks/1.02-ss-source7-4way-comparison.py --top-n 20
    pixi run python notebooks/1.02-ss-source7-4way-comparison.py --dpi 300

Outputs: data/processed/source7-comparison/
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
from matplotlib.lines import Line2D

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.queries import query_activity_results, query_consistency_results

# Activity uses original datasets (per-compound, no shared pool needed)
ACTIVITY_CONFIGS = OrderedDict(
    [
        ("CP_no_s7", "compound_no_source7"),
        ("CP_with_s7", "compound_with_source7"),
        ("DL_no_s7", "compound_DL_CPCNN_no_source7"),
        ("DL_with_s7", "compound_DL_CPCNN_with_source7"),
    ]
)

# Consistency uses active_union datasets (shared pool of actives from CP∪DL, per issue #21)
CONSISTENCY_CONFIGS = OrderedDict(
    [
        ("CP_no_s7", "compound_no_source7_active_union"),
        ("CP_with_s7", "compound_with_source7_active_union"),
        ("DL_no_s7", "compound_DL_CPCNN_no_source7_active_union"),
        ("DL_with_s7", "compound_DL_CPCNN_with_source7_active_union"),
    ]
)

TARGETS = [
    "repurposing",
    "uniprot",
    "chemical_probes",
    "moa",
    "disease_area",
    "motive_biokg",
    "motive_opentargets",
    "motive_primekg",
    "toxcast_assay",
]

# Colors: CP = blue tones, DL = orange tones; lighter = no_s7, darker = with_s7
CONFIG_COLORS = {
    "CP_no_s7": "#6baed6",
    "CP_with_s7": "#2171b5",
    "DL_no_s7": "#fdae6b",
    "DL_with_s7": "#e6550d",
}

# Markers: circle = no_s7, triangle = with_s7
CONFIG_MARKERS = {
    "CP_no_s7": "o",
    "CP_with_s7": "^",
    "DL_no_s7": "o",
    "DL_with_s7": "^",
}

app = typer.Typer()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_activity_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query activity results for all 4 configs and return summary rows."""
    rows = []
    for label, dataset in ACTIVITY_CONFIGS.items():
        df = query_activity_results(
            con,
            dataset,
            preprocessing="activity_no_target2",
            filter_name="all_sources",
            activity_params="default",
        )
        if len(df) == 0:
            logger.warning(f"No activity results for dataset '{dataset}' — check copairs_results.duckdb")
        n_total = len(df)
        n_active = int(df["below_corrected_p"].sum()) if n_total else 0
        pct = 100 * n_active / n_total if n_total else 0
        mean_nmap = df["mean_normalized_average_precision"].mean() if n_total else 0
        rows.append(
            {
                "config": label,
                "dataset": dataset,
                "n_compounds": n_total,
                "n_active": n_active,
                "pct_active": round(pct, 2),
                "mean_nmap": round(mean_nmap, 4),
            }
        )
        logger.info(f"  {label}: {n_total:,} compounds, {n_active:,} active ({pct:.1f}%)")
    return pd.DataFrame(rows)


def get_consistency_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query consistency results for all 4 configs × 9 targets (union-based)."""
    rows = []
    for label, dataset in CONSISTENCY_CONFIGS.items():
        for target in TARGETS:
            df = query_consistency_results(
                con,
                dataset,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=target,
                distance="cosine",
            )
            n_total = len(df)
            n_sig = int(df["below_corrected_p"].sum()) if n_total else 0
            pct = 100 * n_sig / n_total if n_total else 0
            mean_nmap = df["mean_normalized_average_precision"].mean() if n_total else 0
            rows.append(
                {
                    "config": label,
                    "dataset": dataset,
                    "target": target,
                    "n_targets": n_total,
                    "n_significant": n_sig,
                    "pct_significant": round(pct, 1),
                    "mean_nmap": round(mean_nmap, 4),
                }
            )
    return pd.DataFrame(rows)


def get_per_target_results(
    con: duckdb.DuckDBPyConnection,
    target: str,
) -> dict[str, pd.DataFrame]:
    """Query per-target consistency results for all 4 configs (union-based)."""
    results = {}
    for label, dataset in CONSISTENCY_CONFIGS.items():
        df = query_consistency_results(
            con,
            dataset,
            preprocessing="consistency_prefiltered",
            filter_name="all_sources",
            group_type=target,
            distance="cosine",
        )
        results[label] = df
    return results


# ---------------------------------------------------------------------------
# Plotting: aggregate
# ---------------------------------------------------------------------------


def plot_activity_grouped_bar(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Grouped bar chart: % phenotypically active across 4 configs."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # 2 groups (CP, DL) × 2 bars (no_s7, with_s7)
    groups = ["CP", "DL"]
    variants = ["no_s7", "with_s7"]
    x = np.arange(len(groups))
    width = 0.3

    for i, variant in enumerate(variants):
        labels = [f"{g}_{variant}" for g in groups]
        values = [summary.loc[summary.config == lbl, "pct_active"].values[0] for lbl in labels]
        n_vals = [summary.loc[summary.config == lbl, "n_active"].values[0] for lbl in labels]
        colors = [CONFIG_COLORS[lbl] for lbl in labels]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, pct, n in zip(bars, values, n_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%\nn={n:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("% Phenotypically Active (corrected p < 0.05)", fontsize=10)
    ax.set_title("Activity: Source 7 Impact on CP vs DL", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: show both profile colors × variant shades
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_no_s7"], label="CP no_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_with_s7"], label="CP with_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_no_s7"], label="DL no_s7"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_with_s7"], label="DL with_s7"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_consistency_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
    value_col: str = "pct_significant",
    cell_fmt: str = "pct",
    cmap: str = "Blues",
    title: str = "Consistency: % Significant Targets",
    cbar_label: str = "% Significant",
    dpi: int = DEFAULT_DPI,
) -> None:
    """Heatmap with rows=targets, cols=configs."""
    # Sort targets by mean value across configs
    target_order = summary.groupby("target")[value_col].mean().sort_values(ascending=False).index.tolist()
    config_order = list(CONFIG_COLORS.keys())

    pivot = summary.pivot(index="target", columns="config", values=value_col)
    pivot = pivot.reindex(index=target_order, columns=config_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f" if value_col != "mean_nmap" else ".3f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": cbar_label},
    )

    # Add N_sig/N_total annotations for pct heatmap
    if cell_fmt == "pct":
        for i, target in enumerate(target_order):
            for j, config in enumerate(config_order):
                row = summary[(summary.target == target) & (summary.config == config)]
                if len(row) == 1:
                    n_sig = row.iloc[0]["n_significant"]
                    n_total = row.iloc[0]["n_targets"]
                    ax.text(
                        j + 0.5,
                        i + 0.75,
                        f"{n_sig}/{n_total}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="gray",
                    )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_delta_bar(
    summary: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Grouped bar: delta significant targets (with_s7 - no_s7) for CP and DL."""
    fig, ax = plt.subplots(figsize=(10, 5))

    profiles = ["CP", "DL"]
    x = np.arange(len(TARGETS))
    width = 0.35

    profile_colors = {"CP": CONFIG_COLORS["CP_with_s7"], "DL": CONFIG_COLORS["DL_with_s7"]}
    for i, profile in enumerate(profiles):
        no_s7 = f"{profile}_no_s7"
        with_s7 = f"{profile}_with_s7"
        deltas = []
        for target in TARGETS:
            n_no = summary.loc[(summary.config == no_s7) & (summary.target == target), "n_significant"]
            n_with = summary.loc[(summary.config == with_s7) & (summary.target == target), "n_significant"]
            if len(n_no) == 0 or len(n_with) == 0:
                logger.warning(f"Missing data for {profile}/{target} in delta bar — skipping")
                deltas.append(float("nan"))
            else:
                deltas.append(int(n_with.values[0]) - int(n_no.values[0]))

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            deltas,
            width * 0.9,
            color=profile_colors[profile],
            edgecolor="white",
            alpha=0.8,
        )

        # Add delta label above/below each bar
        for bar, delta in zip(bars, deltas):
            if pd.notna(delta) and delta != 0:
                va = "bottom" if delta > 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    delta,
                    f"{int(delta):+d}",
                    ha="center",
                    va=va,
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Δ Significant Targets (with_s7 − no_s7)", fontsize=10)
    ax.set_title("Impact of Source 7 on Target Consistency", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["CP_with_s7"], label="CP"),
        plt.Rectangle((0, 0), 1, 1, fc=CONFIG_COLORS["DL_with_s7"], label="DL"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Per-target analysis
# ---------------------------------------------------------------------------


def classify_status(row: pd.Series, profile: str) -> str:
    """Classify a target's status change between no_s7 and with_s7.

    Categories:
    - stable_sig: significant in both
    - stable_nonsig: not significant in both
    - gained: not sig in no_s7, sig in with_s7
    - lost: sig in no_s7, not sig in with_s7
    - new_sig: only evaluable in with_s7 and significant
    - new_nonsig: only evaluable in with_s7 and not significant
    - dropped: only evaluable in no_s7 (target lost)
    """
    no_s7 = f"{profile}_no_s7"
    with_s7 = f"{profile}_with_s7"
    sig_no = row[f"sig_{no_s7}"]
    sig_with = row[f"sig_{with_s7}"]

    has_no = pd.notna(row[f"nmap_{no_s7}"])
    has_with = pd.notna(row[f"nmap_{with_s7}"])

    # After outer join, sig columns can be NaN — coerce to bool safely
    sig_no = bool(sig_no) if pd.notna(sig_no) else False
    sig_with = bool(sig_with) if pd.notna(sig_with) else False

    if has_no and has_with:
        if sig_no and sig_with:
            return "stable_sig"
        elif not sig_no and not sig_with:
            return "stable_nonsig"
        elif not sig_no and sig_with:
            return "gained"
        else:
            return "lost"
    elif not has_no and has_with:
        return "new_sig" if sig_with else "new_nonsig"
    elif has_no and not has_with:
        return "dropped"
    else:
        return "absent"


def build_status_changes(
    results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """4-way outer join on group_value with status classification."""
    if not results or all(len(df) == 0 for df in results.values()):
        return pd.DataFrame()
    merged = None
    for config, df in results.items():
        subset = df[["group_value", "mean_normalized_average_precision", "below_corrected_p", "n_perturbations"]].copy()
        subset = subset.rename(
            columns={
                "mean_normalized_average_precision": f"nmap_{config}",
                "below_corrected_p": f"sig_{config}",
                "n_perturbations": f"n_{config}",
            }
        )
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on="group_value", how="outer")

    # Classify status for CP and DL
    merged["status_CP"] = merged.apply(classify_status, axis=1, profile="CP")
    merged["status_DL"] = merged.apply(classify_status, axis=1, profile="DL")

    # Sort by max nMAP across configs
    nmap_cols = [c for c in merged.columns if c.startswith("nmap_")]
    merged["_max_nmap"] = merged[nmap_cols].max(axis=1)
    merged = merged.sort_values("_max_nmap", ascending=False).drop(columns=["_max_nmap"])

    return merged


def plot_4way_dotplot(
    merged: pd.DataFrame,
    output_path: Path,
    target_source: str,
    top_n: int = 30,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Cleveland dot plot with 4 markers per target row.

    CP = blue tones, DL = orange tones.
    Circle = no_s7, Triangle = with_s7.
    Filled = significant, Hollow = not significant.
    Thin gray connecting line per target showing spread.
    """
    # Get top N targets by max nMAP
    nmap_cols = [c for c in merged.columns if c.startswith("nmap_")]
    merged = merged.copy()
    merged["_max_nmap"] = merged[nmap_cols].max(axis=1)
    plot_df = merged.nlargest(top_n, "_max_nmap").copy()
    plot_df = plot_df.sort_values("_max_nmap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))

    y_positions = range(len(plot_df))
    config_labels = list(CONFIG_COLORS.keys())

    for y_idx, (_, row) in enumerate(plot_df.iterrows()):
        # Collect x values for connecting line
        x_vals = []

        for config in config_labels:
            nmap_val = row[f"nmap_{config}"]
            sig_val = row[f"sig_{config}"]
            if pd.isna(nmap_val):
                continue

            x_vals.append(nmap_val)
            color = CONFIG_COLORS[config]
            marker = CONFIG_MARKERS[config]

            # Filled if significant, hollow if not (NaN-safe after outer join)
            is_sig = bool(sig_val) if pd.notna(sig_val) else False
            if is_sig:
                ax.scatter(
                    nmap_val,
                    y_idx,
                    marker=marker,
                    s=70,
                    c=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
            else:
                ax.scatter(
                    nmap_val,
                    y_idx,
                    marker=marker,
                    s=70,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )

        # Connecting line
        if len(x_vals) >= 2:
            ax.plot(
                [min(x_vals), max(x_vals)],
                [y_idx, y_idx],
                color="#cccccc",
                linewidth=0.8,
                zorder=1,
            )

    # Horizontal grid
    for y in y_positions:
        ax.axhline(y=y, color="#f0f0f0", linewidth=0.5, zorder=0)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df["group_value"], fontsize=8)
    ax.set_xlabel("Normalized mAP", fontsize=11)
    ax.set_title(f"4-Way Comparison: {target_source}", fontsize=12)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6baed6", markersize=9, label="CP no_s7 (sig)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#6baed6",
            markersize=9,
            markeredgewidth=1.5,
            label="CP no_s7 (n.s.)",
        ),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2171b5", markersize=9, label="CP with_s7 (sig)"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#2171b5",
            markersize=9,
            markeredgewidth=1.5,
            label="CP with_s7 (n.s.)",
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fdae6b", markersize=9, label="DL no_s7 (sig)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#fdae6b",
            markersize=9,
            markeredgewidth=1.5,
            label="DL no_s7 (n.s.)",
        ),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6550d", markersize=9, label="DL with_s7 (sig)"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#e6550d",
            markersize=9,
            markeredgewidth=1.5,
            label="DL with_s7 (n.s.)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Head-to-head CP vs DL (same compound pool)
# ---------------------------------------------------------------------------


def compute_head_to_head(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """CP vs DL on the same union pool, for both no_s7 and with_s7."""
    rows = []
    for s7_label, cp_ds, dl_ds in [
        ("no_s7", "compound_no_source7_active_union", "compound_DL_CPCNN_no_source7_active_union"),
        ("with_s7", "compound_with_source7_active_union", "compound_DL_CPCNN_with_source7_active_union"),
    ]:
        for target in TARGETS:
            cp = query_consistency_results(
                con,
                cp_ds,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=target,
                distance="cosine",
            )
            dl = query_consistency_results(
                con,
                dl_ds,
                preprocessing="consistency_prefiltered",
                filter_name="all_sources",
                group_type=target,
                distance="cosine",
            )
            n = len(cp)
            cp_sig = int(cp["below_corrected_p"].sum())
            dl_sig = int(dl["below_corrected_p"].sum())
            cp_nmap = cp["mean_normalized_average_precision"].mean()
            dl_nmap = dl["mean_normalized_average_precision"].mean()
            winner_sig = "CP" if cp_sig > dl_sig else ("DL" if dl_sig > cp_sig else "tie")
            winner_nmap = "CP" if cp_nmap > dl_nmap else "DL"

            rows.append(
                {
                    "source7": s7_label,
                    "target": target,
                    "n_targets": n,
                    "cp_sig": cp_sig,
                    "dl_sig": dl_sig,
                    "cp_nmap": round(cp_nmap, 4),
                    "dl_nmap": round(dl_nmap, 4),
                    "winner_sig": winner_sig,
                    "winner_nmap": winner_nmap,
                }
            )
    return pd.DataFrame(rows)


def plot_head_to_head(
    h2h: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Paired bar chart: CP vs DL significant targets per annotation source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, s7_label in zip(axes, ["no_s7", "with_s7"]):
        subset = h2h[h2h.source7 == s7_label]
        x = np.arange(len(TARGETS))
        width = 0.35

        cp_vals = [subset.loc[subset.target == t, "cp_sig"].values[0] for t in TARGETS]
        dl_vals = [subset.loc[subset.target == t, "dl_sig"].values[0] for t in TARGETS]

        ax.barh(x - width / 2, cp_vals, width, color=CONFIG_COLORS["CP_with_s7"], label="CP")
        ax.barh(x + width / 2, dl_vals, width, color=CONFIG_COLORS["DL_with_s7"], label="DL")

        ax.set_yticks(x)
        ax.set_yticklabels(TARGETS, fontsize=9)
        ax.set_xlabel("# Significant Targets", fontsize=10)
        ax.set_title(f"CP vs DL ({s7_label})", fontsize=12)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Signal dilution analysis
# ---------------------------------------------------------------------------


def compute_dilution_stats(per_target_dir: Path) -> pd.DataFrame:
    """Compute signal dilution stats from per-target status_changes CSVs.

    For targets evaluable in both no_s7 and with_s7, counts how many
    stayed significant (stable_sig), lost significance (lost/diluted),
    gained significance, or stayed non-significant.

    Also computes mean nMAP change on matched targets.
    """
    rows = []
    for target in TARGETS:
        csv_path = per_target_dir / target / "status_changes.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)

        for profile in ["CP", "DL"]:
            col = f"status_{profile}"
            existing = df[df[col].isin(["stable_sig", "stable_nonsig", "gained", "lost"])]
            new = df[df[col].isin(["new_sig", "new_nonsig"])]

            stable_sig = (existing[col] == "stable_sig").sum()
            stable_nonsig = (existing[col] == "stable_nonsig").sum()
            gained = (existing[col] == "gained").sum()
            lost = (existing[col] == "lost").sum()
            new_sig = (new[col] == "new_sig").sum()
            new_nonsig = (new[col] == "new_nonsig").sum()
            previously_sig = stable_sig + lost

            # Mean nMAP change on matched targets
            no_col = f"nmap_{profile}_no_s7"
            with_col = f"nmap_{profile}_with_s7"
            has_both = df[no_col].notna() & df[with_col].notna()
            nmap_no = df.loc[has_both, no_col].mean() if has_both.any() else float("nan")
            nmap_with = df.loc[has_both, with_col].mean() if has_both.any() else float("nan")

            rows.append(
                {
                    "target": target,
                    "profile": profile,
                    "stable_sig": stable_sig,
                    "stable_nonsig": stable_nonsig,
                    "gained": gained,
                    "lost": lost,
                    "net": gained - lost,
                    "new_sig": new_sig,
                    "new_nonsig": new_nonsig,
                    "previously_sig": previously_sig,
                    "dilution_rate": round(100 * lost / previously_sig, 1) if previously_sig > 0 else 0,
                    "nmap_no_s7": round(nmap_no, 4),
                    "nmap_with_s7": round(nmap_with, 4),
                    "nmap_delta": round(nmap_with - nmap_no, 4) if pd.notna(nmap_no) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def plot_dilution_bar(
    dilution: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Grouped bar: net gained/lost on existing targets per annotation source."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(TARGETS))
    width = 0.35
    profile_colors = {"CP": CONFIG_COLORS["CP_with_s7"], "DL": CONFIG_COLORS["DL_with_s7"]}

    for i, profile in enumerate(["CP", "DL"]):
        pdf = dilution[dilution.profile == profile].set_index("target")
        nets = [pdf.loc[t, "net"] if t in pdf.index else 0 for t in TARGETS]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            nets,
            width * 0.9,
            color=profile_colors[profile],
            edgecolor="white",
            alpha=0.8,
        )
        for bar, net in zip(bars, nets):
            if net != 0:
                va = "bottom" if net > 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    net,
                    f"{int(net):+d}",
                    ha="center",
                    va=va,
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Net Δ on Existing Targets (gained − lost)", fontsize=10)
    ax.set_title("Signal Dilution: Does Source 7 Help or Hurt Existing Targets?", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=profile_colors["CP"], label="CP"),
        plt.Rectangle((0, 0), 1, 1, fc=profile_colors["DL"], label="DL"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------


def write_summary_markdown(
    output_path: Path,
    activity_summary: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    dilution: pd.DataFrame | None = None,
    head_to_head: pd.DataFrame | None = None,
) -> None:
    """Write narrative summary with key findings."""
    with open(output_path, "w") as f:
        f.write("# Source 7 4-Way Comparison\n\n")
        f.write("Comparing inclusion of source 7 (3,191 compounds at 0.625 µM) ")
        f.write("across CellProfiler (CP) and Deep Learning (DL) embeddings.\n\n")
        f.write("Consistency uses union-based evaluation (issue #21): both methods evaluated ")
        f.write("on the shared pool of perturbations active in either CP or DL.\n\n")

        # Activity
        f.write("## Activity Summary\n\n")
        f.write("| Config | Compounds | Active | % Active | Mean nMAP |\n")
        f.write("|--------|-----------|--------|----------|----------|\n")
        for _, row in activity_summary.iterrows():
            f.write(
                f"| {row['config']} | {row['n_compounds']:,} | {row['n_active']:,} | "
                f"{row['pct_active']:.1f}% | {row['mean_nmap']:.4f} |\n"
            )

        # Delta
        f.write("\n### Activity Delta (with_s7 − no_s7)\n\n")
        for profile in ["CP", "DL"]:
            no_df = activity_summary[activity_summary.config == f"{profile}_no_s7"]
            ws_df = activity_summary[activity_summary.config == f"{profile}_with_s7"]
            if len(no_df) == 0 or len(ws_df) == 0:
                f.write(f"- **{profile}**: data unavailable\n")
                continue
            no = no_df.iloc[0]
            ws = ws_df.iloc[0]
            delta_n = ws["n_active"] - no["n_active"]
            delta_pct = ws["pct_active"] - no["pct_active"]
            f.write(f"- **{profile}**: Δ active = {delta_n:+,} ({delta_pct:+.1f} pp)\n")
        f.write("\n")

        # Consistency
        f.write("## Consistency Summary\n\n")
        f.write("| Target | CP no_s7 | CP with_s7 | DL no_s7 | DL with_s7 |\n")
        f.write("|--------|----------|------------|----------|------------|\n")
        for target in TARGETS:
            vals = []
            for config in CONFIG_COLORS:
                row = consistency_summary[
                    (consistency_summary.target == target) & (consistency_summary.config == config)
                ]
                if len(row):
                    r = row.iloc[0]
                    vals.append(f"{r['n_significant']}/{r['n_targets']} ({r['pct_significant']}%)")
                else:
                    vals.append("—")
            f.write(f"| {target} | {' | '.join(vals)} |\n")

        # Delta consistency
        f.write("\n### Consistency Delta (with_s7 − no_s7)\n\n")
        f.write("| Target | Δ CP sig | Δ DL sig |\n")
        f.write("|--------|----------|----------|\n")
        for target in TARGETS:
            deltas = []
            for profile in ["CP", "DL"]:
                no = consistency_summary[
                    (consistency_summary.target == target) & (consistency_summary.config == f"{profile}_no_s7")
                ]
                ws = consistency_summary[
                    (consistency_summary.target == target) & (consistency_summary.config == f"{profile}_with_s7")
                ]
                if len(no) and len(ws):
                    d = int(ws.iloc[0]["n_significant"]) - int(no.iloc[0]["n_significant"])
                    deltas.append(f"{d:+d}")
                else:
                    deltas.append("—")
            f.write(f"| {target} | {' | '.join(deltas)} |\n")
        f.write("\n")

        # Signal dilution analysis
        if dilution is not None and len(dilution) > 0:
            f.write("## Signal Dilution Analysis\n\n")
            f.write("For targets evaluable in **both** no_s7 and with_s7, does adding source 7 ")
            f.write("cause existing significant targets to lose significance (dilution)?\n\n")

            # Per-target table
            f.write("### Per-Annotation Source\n\n")
            f.write("| Target | CP lost | CP gained | CP net | DL lost | DL gained | DL net |\n")
            f.write("|--------|---------|-----------|--------|---------|-----------|--------|\n")
            for target in TARGETS:
                cp = dilution[(dilution.target == target) & (dilution.profile == "CP")]
                dl = dilution[(dilution.target == target) & (dilution.profile == "DL")]
                if len(cp) and len(dl):
                    c, d = cp.iloc[0], dl.iloc[0]
                    f.write(
                        f"| {target} | {c['lost']} | {c['gained']} | {c['net']:+d} "
                        f"| {d['lost']} | {d['gained']} | {d['net']:+d} |\n"
                    )
            f.write("\n")

            # Aggregate
            f.write("### Aggregate\n\n")
            for profile in ["CP", "DL"]:
                pdf = dilution[dilution.profile == profile]
                total_lost = pdf["lost"].sum()
                total_gained = pdf["gained"].sum()
                total_stable_sig = pdf["stable_sig"].sum()
                total_prev_sig = pdf["previously_sig"].sum()
                total_new_sig = pdf["new_sig"].sum()
                dilution_rate = 100 * total_lost / total_prev_sig if total_prev_sig else 0
                f.write(f"**{profile}** across {len(TARGETS)} annotation sources:\n")
                f.write(f"- Previously significant: {total_prev_sig}\n")
                f.write(f"- Stable (still significant): {total_stable_sig}\n")
                f.write(f"- Lost significance (diluted): {total_lost} ({dilution_rate:.1f}%)\n")
                f.write(f"- Gained significance: {total_gained}\n")
                f.write(f"- Net on existing: {total_gained - total_lost:+d}\n")
                f.write(f"- New targets only evaluable with s7: {total_new_sig} significant\n\n")

        # Head-to-head CP vs DL
        if head_to_head is not None and len(head_to_head) > 0:
            f.write("## Head-to-Head: CP vs DL (Same Compound Pool)\n\n")
            f.write("Both methods evaluated on the union of actives (identical compound set per row).\n\n")

            for s7_label in ["no_s7", "with_s7"]:
                subset = head_to_head[head_to_head.source7 == s7_label]
                f.write(f"### {s7_label}\n\n")
                f.write("| Target | N | CP sig | DL sig | CP nMAP | DL nMAP | Winner (sig) |\n")
                f.write("|--------|---|--------|--------|---------|---------|-------------|\n")
                for _, row in subset.iterrows():
                    f.write(
                        f"| {row['target']} | {row['n_targets']} | {row['cp_sig']} | {row['dl_sig']} "
                        f"| {row['cp_nmap']:.4f} | {row['dl_nmap']:.4f} | {row['winner_sig']} |\n"
                    )
                cp_wins = (subset.winner_sig == "CP").sum()
                dl_wins = (subset.winner_sig == "DL").sum()
                ties = (subset.winner_sig == "tie").sum()
                cp_nmap_wins = (subset.winner_nmap == "CP").sum()
                dl_nmap_wins = (subset.winner_nmap == "DL").sum()
                f.write(f"\n**Score**: # sig targets — CP {cp_wins}, DL {dl_wins}, ties {ties}")
                f.write(f" | mean nMAP — CP {cp_nmap_wins}, DL {dl_nmap_wins}\n\n")

    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@app.command()
def main(
    db_path: str = typer.Option(
        "data/processed/copairs_results.duckdb",
        "--db",
        "-d",
        help="Path to copairs results database",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: data/processed/source7-comparison/)",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
    top_n: int = typer.Option(
        30,
        "--top-n",
        help="Max targets in per-source dot plots",
    ),
) -> None:
    """4-way comparison of source 7 inclusion across CP and DL."""
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "source7-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    db = Path(db_path)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        raise typer.Exit(1)

    con = duckdb.connect(str(db), read_only=True)

    try:
        # ── 1. Activity summary ──────────────────────────────────────────
        logger.info("Querying activity results for 4 configs...")
        activity_summary = get_activity_summary(con)
        activity_summary.to_csv(output_dir / "activity_summary.csv", index=False)

        plot_activity_grouped_bar(activity_summary, output_dir / "activity_grouped_bar.png", dpi=dpi)

        # ── 2. Consistency summary ───────────────────────────────────────
        logger.info("Querying consistency results for 4 configs × 9 targets...")
        consistency_summary = get_consistency_summary(con)
        consistency_summary.to_csv(output_dir / "consistency_summary.csv", index=False)

        plot_consistency_heatmap(
            consistency_summary,
            output_dir / "consistency_heatmap.png",
            value_col="pct_significant",
            cell_fmt="pct",
            cmap="Blues",
            title="Consistency: % Significant Targets",
            cbar_label="% Significant",
            dpi=dpi,
        )
        plot_consistency_heatmap(
            consistency_summary,
            output_dir / "consistency_nmap_heatmap.png",
            value_col="mean_nmap",
            cell_fmt="value",
            cmap="YlGnBu",
            title="Consistency: Mean Normalized mAP",
            cbar_label="Mean nMAP",
            dpi=dpi,
        )
        plot_delta_bar(consistency_summary, output_dir / "delta_significant_bar.png", dpi=dpi)

        # ── 3. Per-target analysis ───────────────────────────────────────
        logger.info("Building per-target status changes and dot plots...")
        per_target_dir = output_dir / "per_target"

        for target in TARGETS:
            target_dir = per_target_dir / target
            target_dir.mkdir(parents=True, exist_ok=True)

            results = get_per_target_results(con, target)
            merged = build_status_changes(results)
            merged.to_csv(target_dir / "status_changes.csv", index=False)

            if len(merged) > 0:
                plot_4way_dotplot(merged, target_dir / "4way_dotplot.png", target, top_n=top_n, dpi=dpi)

            n_gained_cp = (merged["status_CP"] == "gained").sum()
            n_lost_cp = (merged["status_CP"] == "lost").sum()
            n_gained_dl = (merged["status_DL"] == "gained").sum()
            n_lost_dl = (merged["status_DL"] == "lost").sum()
            logger.info(
                f"  {target}: CP gained={n_gained_cp} lost={n_lost_cp}, DL gained={n_gained_dl} lost={n_lost_dl}"
            )

        # ── 4. Signal dilution analysis ───────────────────────────────────
        logger.info("Computing signal dilution stats...")
        dilution = compute_dilution_stats(per_target_dir)
        dilution.to_csv(output_dir / "dilution_summary.csv", index=False)
        plot_dilution_bar(dilution, output_dir / "dilution_bar.png", dpi=dpi)

        for profile in ["CP", "DL"]:
            pdf = dilution[dilution.profile == profile]
            total_lost = pdf["lost"].sum()
            total_gained = pdf["gained"].sum()
            total_prev_sig = pdf["previously_sig"].sum()
            rate = 100 * total_lost / total_prev_sig if total_prev_sig else 0
            logger.info(
                f"  {profile}: {total_gained} gained, {total_lost} lost "
                f"({rate:.1f}% dilution), net={total_gained - total_lost:+d}"
            )

        # ── 5. Head-to-head CP vs DL ──────────────────────────────────────
        logger.info("Computing head-to-head CP vs DL on union pools...")
        h2h = compute_head_to_head(con)
        h2h.to_csv(output_dir / "head_to_head.csv", index=False)
        plot_head_to_head(h2h, output_dir / "head_to_head.png", dpi=dpi)

        for s7_label in ["no_s7", "with_s7"]:
            subset = h2h[h2h.source7 == s7_label]
            cp_wins = (subset.winner_sig == "CP").sum()
            dl_wins = (subset.winner_sig == "DL").sum()
            logger.info(f"  {s7_label}: CP wins {cp_wins}, DL wins {dl_wins} of 9 sources")

        # ── 6. Summary markdown ──────────────────────────────────────────
        write_summary_markdown(
            output_dir / "summary_stats.md",
            activity_summary,
            consistency_summary,
            dilution,
            h2h,
        )

        logger.success(f"All outputs saved to {output_dir}")

    finally:
        con.close()


if __name__ == "__main__":
    app()
