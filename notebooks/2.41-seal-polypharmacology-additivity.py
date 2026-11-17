#!/usr/bin/env python
"""Polypharmacology profile additivity in JUMP Cell Painting.

Research question: When a drug hits two targets (A+B), does its morphological
profile look like a "blend" of profiles from single-target compounds?

Finds "trios": compound AB (hits A+B), compound A (hits only A), compound B
(hits only B). Compares their profiles via cosine similarity and blend metrics.

Usage:
    pixi run python notebooks/2.41-seal-polypharmacology-additivity.py
    pixi run python notebooks/2.41-seal-polypharmacology-additivity.py --skip-plots
    pixi run python notebooks/2.41-seal-polypharmacology-additivity.py --min-nmap 0.8

Outputs: data/processed/polypharmacology/{dataset}/
    - all_trios.csv: All candidate trios with activity and selectivity info
    - best_trios.csv: Best trio per target pair
    - similarity_metrics.csv: Pairwise and blend cosine similarities per trio
    - trio_heatmaps.png: 4x4 similarity heatmaps (AB, A, B, blend) for top trios
    - profile_heatmap_{pair}.png: Hart-style feature heatmaps for featured trios
    - trio_categories.png: Classification of trios into 5 polypharmacology scenarios
    - additivity_scatter.png: Blend vs individual similarity across target pairs
    - umap_trio_overlay.png: Top trios on UMAP
    - additivity_summary.png: Summary bar chart of blend similarity per pair
    - summary.json: Counts and key findings
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, load_profiles, load_umap
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

DATASET = "compound_no_source7"
METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "polypharmacology"

app = typer.Typer()


# =============================================================================
# Trio Discovery
# =============================================================================


def find_all_trios(
    meta_con: duckdb.DuckDBPyConnection,
    copairs_con: duckdb.DuckDBPyConnection,
    dataset: str,
    profiled_jcps: set[str],
) -> pd.DataFrame:
    """Find all (AB, A, B) trios from repurposing hub annotations.

    AB has exactly 2 targets, A hits only target_a, B hits only target_b.
    All three must have profiles and activity data.
    """
    # Step 1: Find dual-target and single-target compounds
    trios_df = meta_con.execute("""
        WITH dual_target AS (
            SELECT
                Metadata_JCP2022 AS jcp_ab,
                Metadata_repurposing_name AS name_ab,
                Metadata_repurposing_target AS targets_ab,
                Metadata_repurposing_moa AS moa_ab,
                TRIM(STRING_SPLIT(Metadata_repurposing_target, '|')[1]) AS target_a,
                TRIM(STRING_SPLIT(Metadata_repurposing_target, '|')[2]) AS target_b
            FROM repurposing_hub_annotations
            WHERE Metadata_repurposing_target IS NOT NULL
              AND LEN(STRING_SPLIT(Metadata_repurposing_target, '|')) = 2
        ),
        single_target AS (
            SELECT
                Metadata_JCP2022 AS jcp,
                Metadata_repurposing_name AS name,
                TRIM(Metadata_repurposing_target) AS target
            FROM repurposing_hub_annotations
            WHERE Metadata_repurposing_target IS NOT NULL
              AND LEN(STRING_SPLIT(Metadata_repurposing_target, '|')) = 1
        )
        SELECT
            d.jcp_ab, d.name_ab, d.targets_ab, d.moa_ab,
            d.target_a, d.target_b,
            a.jcp AS jcp_a, a.name AS name_a,
            b.jcp AS jcp_b, b.name AS name_b
        FROM dual_target d
        JOIN single_target a ON a.target = d.target_a
        JOIN single_target b ON b.target = d.target_b
        WHERE a.jcp != b.jcp
    """).df()

    logger.info(f"Raw trios from SQL: {len(trios_df):,}")

    if trios_df.empty:
        return trios_df

    # Step 2: Filter to compounds with profiles
    all_jcps = set(trios_df["jcp_ab"]) | set(trios_df["jcp_a"]) | set(trios_df["jcp_b"])
    has_profile = all_jcps & profiled_jcps
    trios_df = trios_df[
        trios_df["jcp_ab"].isin(has_profile)
        & trios_df["jcp_a"].isin(has_profile)
        & trios_df["jcp_b"].isin(has_profile)
    ]
    logger.info(f"Trios with all three profiled: {len(trios_df):,}")

    # Step 3: Join activity for all three members
    all_jcps_list = list(
        set(trios_df["jcp_ab"]) | set(trios_df["jcp_a"]) | set(trios_df["jcp_b"])
    )
    activity_df = copairs_con.execute(
        """
        SELECT Metadata_JCP2022 AS jcp,
               mean_normalized_average_precision AS nmap,
               below_corrected_p AS active
        FROM activity_results
        WHERE _dataset = ? AND _preprocessing = 'activity_no_target2'
          AND _filter = 'all_sources' AND _activity_params = 'default'
          AND Metadata_JCP2022 = ANY(?)
        """,
        [dataset, all_jcps_list],
    ).df()

    activity_map = dict(zip(activity_df["jcp"], activity_df["nmap"]))
    active_set = set(activity_df[activity_df["active"]]["jcp"])

    trios_df["nmap_ab"] = trios_df["jcp_ab"].map(activity_map)
    trios_df["nmap_a"] = trios_df["jcp_a"].map(activity_map)
    trios_df["nmap_b"] = trios_df["jcp_b"].map(activity_map)
    trios_df["active_ab"] = trios_df["jcp_ab"].isin(active_set)
    trios_df["active_a"] = trios_df["jcp_a"].isin(active_set)
    trios_df["active_b"] = trios_df["jcp_b"].isin(active_set)
    trios_df["all_active"] = trios_df["active_ab"] & trios_df["active_a"] & trios_df["active_b"]
    trios_df["min_nmap"] = trios_df[["nmap_ab", "nmap_a", "nmap_b"]].min(axis=1)

    n_all_active = trios_df["all_active"].sum()
    logger.info(f"Trios where all three are active: {n_all_active:,}")

    # Step 4: Cross-annotation check for single-target compounds
    single_jcps = list(set(trios_df["jcp_a"]) | set(trios_df["jcp_b"]))
    cross_df = meta_con.execute(
        """
        SELECT
            Metadata_JCP2022 AS jcp,
            Metadata_motive_gene_biokg,
            Metadata_motive_gene_opentargets,
            Metadata_chmprb_target_genes
        FROM compound_metadata
        WHERE Metadata_JCP2022 = ANY(?)
        """,
        [single_jcps],
    ).df()

    # Build lookup: jcp -> set of extra genes per annotation source
    cross_map = {}
    for _, row in cross_df.iterrows():
        genes = set()
        for col in ["Metadata_motive_gene_biokg", "Metadata_motive_gene_opentargets", "Metadata_chmprb_target_genes"]:
            val = row[col]
            if pd.notna(val):
                for g in str(val).split("|"):
                    g = g.strip()
                    if g:
                        genes.add(g)
        cross_map[row["jcp"]] = genes

    def selectivity_label(jcp: str, expected_target: str) -> str:
        genes = cross_map.get(jcp, set())
        other_genes = genes - {expected_target}
        if not genes:
            return "single_source"  # only repurposing hub annotates it
        if not other_genes:
            return "confirmed"  # other sources agree on single target
        return f"contested({','.join(sorted(other_genes))})"

    trios_df["selectivity_a"] = [
        selectivity_label(row["jcp_a"], row["target_a"]) for _, row in trios_df.iterrows()
    ]
    trios_df["selectivity_b"] = [
        selectivity_label(row["jcp_b"], row["target_b"]) for _, row in trios_df.iterrows()
    ]

    # Create target pair column (alphabetically sorted)
    trios_df["target_pair"] = trios_df.apply(
        lambda r: "|".join(sorted([r["target_a"], r["target_b"]])), axis=1
    )

    return trios_df.sort_values("min_nmap", ascending=False)


# =============================================================================
# Trio Selection
# =============================================================================


def rank_and_select_trios(
    trios_df: pd.DataFrame,
    max_per_pair: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select best trios per target pair. Returns (all_active_trios, best_trios)."""
    active_trios = trios_df[trios_df["all_active"]].copy()

    if active_trios.empty:
        logger.warning("No all-active trios found")
        return active_trios, active_trios

    # Summary per target pair
    pair_stats = (
        active_trios.groupby("target_pair")
        .agg(
            n_trios=("jcp_ab", "size"),
            n_dual_compounds=("jcp_ab", "nunique"),
            n_single_a=("jcp_a", "nunique"),
            n_single_b=("jcp_b", "nunique"),
            best_min_nmap=("min_nmap", "max"),
        )
        .sort_values("best_min_nmap", ascending=False)
    )

    logger.info(f"\nTarget pairs with all-active trios: {len(pair_stats)}")
    logger.info(f"\n{pair_stats.head(15).to_string()}")

    # Pick best trio(s) per pair
    best = (
        active_trios.sort_values("min_nmap", ascending=False)
        .groupby("target_pair")
        .head(max_per_pair)
    )

    return active_trios, best


# =============================================================================
# Profile Comparison
# =============================================================================


def compute_trio_similarities(
    best_trios: pd.DataFrame,
    profiles: np.ndarray,
    jcp_to_idx: dict[str, int],
) -> pd.DataFrame:
    """Compute pairwise and blend similarities for each trio."""
    rows = []
    for _, trio in best_trios.iterrows():
        jcps = [trio["jcp_ab"], trio["jcp_a"], trio["jcp_b"]]
        if not all(j in jcp_to_idx for j in jcps):
            continue

        idx = [jcp_to_idx[j] for j in jcps]
        X = profiles[idx]  # shape (3, n_features)

        sim = cosine_similarity(X)
        sim_ab_a = sim[0, 1]
        sim_ab_b = sim[0, 2]
        sim_a_b = sim[1, 2]

        # Blend: average of A and B profiles
        blend = (X[1] + X[2]) / 2
        sim_ab_blend = cosine_similarity(X[0:1], blend.reshape(1, -1))[0, 0]

        # Is blend better than either individual?
        blend_beats_individuals = sim_ab_blend > max(sim_ab_a, sim_ab_b)

        # Residual: how much of AB is not captured by the blend
        residual = np.linalg.norm(X[0] - blend) / np.linalg.norm(X[0])

        rows.append({
            "target_pair": trio["target_pair"],
            "jcp_ab": trio["jcp_ab"],
            "name_ab": trio["name_ab"],
            "jcp_a": trio["jcp_a"],
            "name_a": trio["name_a"],
            "jcp_b": trio["jcp_b"],
            "name_b": trio["name_b"],
            "target_a": trio["target_a"],
            "target_b": trio["target_b"],
            "nmap_ab": trio["nmap_ab"],
            "nmap_a": trio["nmap_a"],
            "nmap_b": trio["nmap_b"],
            "sim_AB_A": sim_ab_a,
            "sim_AB_B": sim_ab_b,
            "sim_A_B": sim_a_b,
            "sim_AB_blend": sim_ab_blend,
            "blend_beats_individuals": blend_beats_individuals,
            "residual_fraction": residual,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Visualizations
# =============================================================================


def plot_trio_heatmaps(
    sim_df: pd.DataFrame,
    profiles: np.ndarray,
    jcp_to_idx: dict[str, int],
    output_path: Path,
    n_trios: int = 5,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot 4x4 cosine similarity heatmaps (AB, A, B, blend) for top trios."""
    best_per_pair = (
        sim_df.sort_values("sim_AB_blend", ascending=False)
        .drop_duplicates(subset="target_pair")
        .head(n_trios)
    )

    n = len(best_per_pair)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)

    for i, (_, row) in enumerate(best_per_pair.iterrows()):
        ax = axes[0, i]

        jcps = [row["jcp_ab"], row["jcp_a"], row["jcp_b"]]
        if not all(j in jcp_to_idx for j in jcps):
            continue

        idx = [jcp_to_idx[j] for j in jcps]
        X = profiles[idx]
        blend = ((X[1] + X[2]) / 2).reshape(1, -1)
        X_with_blend = np.vstack([X, blend])  # (4, n_features)
        sim_matrix = cosine_similarity(X_with_blend)

        labels = [
            f"{row['name_ab'] or row['jcp_ab']}\n(AB: {row['target_a']}+{row['target_b']})",
            f"{row['name_a'] or row['jcp_a']}\n(A: {row['target_a']})",
            f"{row['name_b'] or row['jcp_b']}\n(B: {row['target_b']})",
            f"blend(A,B)",
        ]

        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.3, vmax=1.0, aspect="equal")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels, fontsize=7)

        for ii in range(4):
            for jj in range(4):
                ax.text(
                    jj, ii, f"{sim_matrix[ii, jj]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(sim_matrix[ii, jj]) > 0.4 else "black",
                )

        ax.set_title(row["target_pair"], fontsize=10)

    fig.suptitle("Polypharmacology trios: pairwise cosine similarity (incl. blend)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_profile_heatmap(
    trio_row: pd.Series,
    profiles: np.ndarray,
    jcp_to_idx: dict[str, int],
    feature_names: list[str],
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Hart et al. 2021-style heatmap: compounds as rows, all features as columns.

    Rows: AB, A, B, blend(A,B) — labeled with compound name and similarity to AB.
    """
    jcps = [trio_row["jcp_ab"], trio_row["jcp_a"], trio_row["jcp_b"]]
    if not all(j in jcp_to_idx for j in jcps):
        return

    idx = [jcp_to_idx[j] for j in jcps]
    X = profiles[idx]  # (3, n_features): AB, A, B
    blend = ((X[1] + X[2]) / 2).reshape(1, -1)

    # Similarities to AB for labels and ordering
    sim_ab_a = cosine_similarity(X[0:1], X[1:2])[0, 0]
    sim_ab_b = cosine_similarity(X[0:1], X[2:3])[0, 0]
    sim_ab_blend = cosine_similarity(X[0:1], blend)[0, 0]

    # Order: AB, blend, most-similar-to-AB, least-similar-to-AB
    name_a = trio_row["name_a"] or trio_row["jcp_a"]
    name_b = trio_row["name_b"] or trio_row["jcp_b"]
    if sim_ab_a >= sim_ab_b:
        X_all = np.vstack([X[0:1], blend, X[1:2], X[2:3]])
        row_labels = [
            f"{trio_row['name_ab'] or trio_row['jcp_ab']}  (AB)",
            f"blend(A,B)  sim={sim_ab_blend:.2f}",
            f"{name_a}  (A)  sim={sim_ab_a:.2f}",
            f"{name_b}  (B)  sim={sim_ab_b:.2f}",
        ]
    else:
        X_all = np.vstack([X[0:1], blend, X[2:3], X[1:2]])
        row_labels = [
            f"{trio_row['name_ab'] or trio_row['jcp_ab']}  (AB)",
            f"blend(A,B)  sim={sim_ab_blend:.2f}",
            f"{name_b}  (B)  sim={sim_ab_b:.2f}",
            f"{name_a}  (A)  sim={sim_ab_a:.2f}",
        ]

    # Sort features by AB profile values for visual coherence
    sort_order = np.argsort(X[0])
    X_sorted = X_all[:, sort_order]

    vmax = np.percentile(np.abs(X_sorted), 99)

    fig, ax = plt.subplots(figsize=(16, 2.5))
    im = ax.imshow(X_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")

    ax.set_yticks(range(4))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel(f"Cell Painting features (n={len(feature_names)}, sorted by AB profile value)")

    target_label = f"{trio_row['target_a']}+{trio_row['target_b']}"
    ax.set_title(
        f"{trio_row['name_ab'] or trio_row['jcp_ab']} ({target_label}): "
        f"profile comparison with single-target compounds and computational blend",
        fontsize=10,
    )

    plt.colorbar(im, ax=ax, label="Feature value", shrink=0.8, pad=0.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


SIM_THRESHOLD = 0.3  # Cosine similarity threshold for "similar profiles"

CATEGORY_COLORS = {
    "Dominant target": "#D55E00",
    "Additive polypharmacology": "#009E73",
    "Emergent phenotype": "#CC79A7",
    "Convergent targets": "#0072B2",
    "Divergent from shared": "#E69F00",
}


def categorize_trios(sim_df: pd.DataFrame, threshold: float = SIM_THRESHOLD) -> pd.DataFrame:
    """Classify each trio into one of 5 polypharmacology scenarios."""
    df = sim_df.copy()
    categories = []

    for _, row in df.iterrows():
        ab_similar = row["sim_A_B"] >= threshold
        ab_looks_like_a = row["sim_AB_A"] >= threshold
        ab_looks_like_b = row["sim_AB_B"] >= threshold
        blend_wins = row["blend_beats_individuals"]

        if ab_similar and (ab_looks_like_a or ab_looks_like_b):
            categories.append("Convergent targets")
        elif ab_similar and not ab_looks_like_a and not ab_looks_like_b:
            categories.append("Divergent from shared")
        elif not ab_similar and blend_wins:
            categories.append("Additive polypharmacology")
        elif not ab_similar and (ab_looks_like_a or ab_looks_like_b):
            categories.append("Dominant target")
        else:
            categories.append("Emergent phenotype")

    df["category"] = categories
    return df


def plot_trio_categories(
    sim_df: pd.DataFrame,
    output_path: Path,
    threshold: float = SIM_THRESHOLD,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Bar chart showing distribution of trios across 5 polypharmacology scenarios."""
    # Use best trio per target pair
    pair_df = categorize_trios(
        sim_df.sort_values("sim_AB_blend", ascending=False).drop_duplicates(subset="target_pair"),
        threshold=threshold,
    )

    counts = pair_df["category"].value_counts()
    # Ensure all categories present
    for cat in CATEGORY_COLORS:
        if cat not in counts:
            counts[cat] = 0

    ordered_cats = list(CATEGORY_COLORS.keys())
    counts = counts.reindex(ordered_cats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2]})

    # Left: bar chart of counts
    ax = axes[0]
    bars = ax.barh(
        range(len(ordered_cats)), [counts[c] for c in ordered_cats],
        color=[CATEGORY_COLORS[c] for c in ordered_cats],
        edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    ax.set_yticks(range(len(ordered_cats)))
    ax.set_yticklabels(ordered_cats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of target pairs")
    ax.set_title("Trio categories")
    for j, (c, v) in enumerate(zip(ordered_cats, [counts[c] for c in ordered_cats])):
        ax.text(v + 0.3, j, str(v), va="center", fontsize=9)

    # Right: scatter plot colored by category
    ax2 = axes[1]
    for cat in ordered_cats:
        mask = pair_df["category"] == cat
        if not mask.any():
            continue
        subset = pair_df[mask]
        ax2.scatter(
            subset["sim_A_B"], subset["sim_AB_blend"],
            c=CATEGORY_COLORS[cat], label=f"{cat} ({mask.sum()})",
            s=60, alpha=0.7, edgecolors="black", linewidths=0.5,
        )
        for _, row in subset.iterrows():
            ax2.annotate(
                row["target_pair"], (row["sim_A_B"], row["sim_AB_blend"]),
                fontsize=5, alpha=0.6, xytext=(3, 3), textcoords="offset points",
            )

    ax2.axvline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"sim(A,B)={threshold}")
    ax2.set_xlabel("sim(A, B) — target similarity")
    ax2.set_ylabel("sim(AB, blend) — additivity")
    ax2.set_title("Polypharmacology scenarios")
    ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_additivity_scatter(
    sim_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Scatter: blend similarity vs best individual similarity per target pair."""
    if sim_df.empty:
        return

    # Best trio per target pair
    pair_df = (
        sim_df.sort_values("sim_AB_blend", ascending=False)
        .drop_duplicates(subset="target_pair")
    )

    pair_df = pair_df.copy()
    pair_df["max_individual"] = pair_df[["sim_AB_A", "sim_AB_B"]].max(axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        pair_df["sim_AB_blend"],
        pair_df["max_individual"],
        s=60, alpha=0.7, edgecolors="black", linewidths=0.5,
        c=pair_df["blend_beats_individuals"].map({True: "#2ca02c", False: "#d62728"}),
    )

    # Diagonal
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Label points
    for _, row in pair_df.iterrows():
        ax.annotate(
            row["target_pair"],
            (row["sim_AB_blend"], row["max_individual"]),
            fontsize=6, alpha=0.7,
            xytext=(4, 4), textcoords="offset points",
        )

    ax.set_xlabel("sim(AB, blend of A+B)")
    ax.set_ylabel("max(sim(AB, A), sim(AB, B))")
    ax.set_title(
        "Polypharmacology additivity: blend vs individual similarity\n"
        "Green = blend is better predictor; Red = individual is better"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_umap_trio_overlay(
    sim_df: pd.DataFrame,
    output_path: Path,
    dataset: str,
    n_trios: int = 3,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot top trios on UMAP with connecting lines."""
    best_per_pair = (
        sim_df.sort_values("sim_AB_blend", ascending=False)
        .drop_duplicates(subset="target_pair")
        .head(n_trios)
    )

    if best_per_pair.empty:
        return

    adata = load_umap(dataset, level="perturbation", metric="cosine", filter_name="all")
    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for i, (_, row) in enumerate(best_per_pair.iterrows()):
        color = colors[i % len(colors)]
        jcps = [row["jcp_ab"], row["jcp_a"], row["jcp_b"]]
        valid = [j for j in jcps if j in jcp_to_idx]
        if len(valid) < 3:
            continue

        trio_idx = [jcp_to_idx[j] for j in valid]
        trio_coords = coords[trio_idx]

        # Draw connecting lines
        for a_idx in range(3):
            for b_idx in range(a_idx + 1, 3):
                ax.plot(
                    [trio_coords[a_idx, 0], trio_coords[b_idx, 0]],
                    [trio_coords[a_idx, 1], trio_coords[b_idx, 1]],
                    color=color, alpha=0.5, linewidth=1.5, zorder=3,
                )

        # Plot points: circle for AB, triangles for A and B
        markers = ["o", "^", "v"]
        labels_short = [
            f"{row['name_ab'] or row['jcp_ab']} (AB)",
            f"{row['name_a'] or row['jcp_a']} (A)",
            f"{row['name_b'] or row['jcp_b']} (B)",
        ]

        for j, (tc, marker, label) in enumerate(zip(trio_coords, markers, labels_short)):
            ax.scatter(
                tc[0], tc[1],
                c=color, marker=marker, s=120, edgecolors="black", linewidths=0.8,
                zorder=5, label=label if i == 0 or j == 0 else None,
            )
            ax.annotate(
                label.split(" (")[0],  # compound name only
                (tc[0], tc[1]),
                fontsize=6, alpha=0.8,
                xytext=(6, 6), textcoords="offset points",
            )

    ax.set_title("Polypharmacology trios in morphological space")
    ax.legend(loc="upper right", fontsize=7, ncol=1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_additivity_summary(
    sim_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Horizontal bar chart of blend similarity per target pair."""
    if sim_df.empty:
        return

    pair_df = (
        sim_df.sort_values("sim_AB_blend", ascending=False)
        .drop_duplicates(subset="target_pair")
        .sort_values("sim_AB_blend")
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(pair_df) * 0.35)))

    colors = [
        "#2ca02c" if b else "#d62728"
        for b in pair_df["blend_beats_individuals"]
    ]

    ax.barh(range(len(pair_df)), pair_df["sim_AB_blend"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(pair_df)))
    ax.set_yticklabels(pair_df["target_pair"], fontsize=8)
    ax.set_xlabel("sim(AB, blend of A+B)")
    ax.set_title(
        "Profile additivity by target pair\n"
        "Green = blend beats max individual; Red = individual wins"
    )
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    dataset: str = typer.Option(DATASET, "--dataset", "-d", help="Dataset name"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    min_nmap: float = typer.Option(0.5, "--min-nmap", help="Minimum nmAP for all trio members"),
    max_trios_per_pair: int = typer.Option(3, "--max-trios", help="Max trios to analyze per target pair"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
) -> None:
    """Explore polypharmacology profile additivity in JUMP Cell Painting."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load profiles to get the set of profiled compounds
    logger.info("Loading profiles...")
    adata = load_profiles(dataset, level="perturbation")
    profiled_jcps = set(adata.obs["JCP2022"].values)
    profiles = adata.X
    if hasattr(profiles, "toarray"):
        profiles = profiles.toarray()
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}
    feature_names = list(adata.var_names)
    logger.info(f"Profiles loaded: {len(profiled_jcps):,} compounds, {profiles.shape[1]} features")

    # Open databases
    meta_con = duckdb.connect(str(METADATA_DB), read_only=True)
    copairs_con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)

    try:
        # Step 1: Find all trios
        logger.info("Finding polypharmacology trios...")
        all_trios = find_all_trios(meta_con, copairs_con, dataset, profiled_jcps)

        if all_trios.empty:
            logger.warning("No trios found — exiting")
            raise typer.Exit(0)

        # Check Kenpaullone specifically
        kenp = all_trios[all_trios["jcp_ab"] == "JCP2022_075293"]
        if not kenp.empty:
            logger.info(f"Kenpaullone (JCP2022_075293) found in {len(kenp)} trios")
        else:
            # Check if it's in repurposing hub at all
            kenp_check = meta_con.execute(
                "SELECT Metadata_repurposing_target FROM repurposing_hub_annotations WHERE Metadata_JCP2022 = 'JCP2022_075293'"
            ).df()
            if kenp_check.empty:
                logger.info("Kenpaullone (JCP2022_075293) not in repurposing hub")
            else:
                logger.info(f"Kenpaullone targets: {kenp_check.iloc[0, 0]} (no matching single-target compounds found)")

        # Step 2: Rank and select
        logger.info("Ranking trios...")
        active_trios, best_trios = rank_and_select_trios(all_trios, max_per_pair=max_trios_per_pair)

        # Filter by min nmAP
        best_trios = best_trios[best_trios["min_nmap"] >= min_nmap]
        logger.info(f"Best trios after nmAP >= {min_nmap} filter: {len(best_trios)}")

        # Save CSVs
        all_trios.to_csv(output_dir / "all_trios.csv", index=False)
        best_trios.to_csv(output_dir / "best_trios.csv", index=False)
        logger.info(f"Saved: {output_dir / 'all_trios.csv'} ({len(all_trios):,} rows)")
        logger.info(f"Saved: {output_dir / 'best_trios.csv'} ({len(best_trios)} rows)")

        # Step 3: Compute similarities
        logger.info("Computing trio profile similarities...")
        sim_df = compute_trio_similarities(best_trios, profiles, jcp_to_idx)
        sim_df.to_csv(output_dir / "similarity_metrics.csv", index=False)
        logger.info(f"Saved: {output_dir / 'similarity_metrics.csv'} ({len(sim_df)} rows)")

        # Log top results
        if not sim_df.empty:
            top = sim_df.sort_values("sim_AB_blend", ascending=False).head(10)
            logger.info("\n--- Top trios by blend similarity ---")
            for _, row in top.iterrows():
                logger.info(
                    f"  {row['target_pair']}: "
                    f"{row['name_ab']}(AB) vs {row['name_a']}(A) vs {row['name_b']}(B) | "
                    f"sim(AB,blend)={row['sim_AB_blend']:.3f} "
                    f"sim(AB,A)={row['sim_AB_A']:.3f} sim(AB,B)={row['sim_AB_B']:.3f} "
                    f"sim(A,B)={row['sim_A_B']:.3f} "
                    f"{'BLEND WINS' if row['blend_beats_individuals'] else ''}"
                )

        # Step 4: Categorize trios
        if not sim_df.empty:
            cat_df = categorize_trios(
                sim_df.sort_values("sim_AB_blend", ascending=False).drop_duplicates(subset="target_pair"),
            )
            cat_df.to_csv(output_dir / "trio_categories.csv", index=False)
            logger.info(f"\n--- Trio categories (threshold={SIM_THRESHOLD}) ---")
            for cat, count in cat_df["category"].value_counts().items():
                logger.info(f"  {cat}: {count}")

        # Step 5: Visualizations
        if not skip_plots and not sim_df.empty:
            logger.info("Generating plots...")

            plot_trio_heatmaps(sim_df, profiles, jcp_to_idx, output_dir / "trio_heatmaps.png", dpi=dpi)
            plot_trio_categories(sim_df, output_dir / "trio_categories.png", dpi=dpi)
            plot_additivity_scatter(sim_df, output_dir / "additivity_scatter.png", dpi=dpi)
            plot_additivity_summary(sim_df, output_dir / "additivity_summary.png", dpi=dpi)
            plot_umap_trio_overlay(sim_df, output_dir / "umap_trio_overlay.png", dataset=dataset, dpi=dpi)

            # Hart-style profile heatmaps for featured trios
            featured_pairs = ["ALK|MET", "MDM2|TP53"]
            for pair in featured_pairs:
                pair_rows = sim_df[sim_df["target_pair"] == pair].sort_values("sim_AB_blend", ascending=False)
                if pair_rows.empty:
                    continue
                best_row = pair_rows.iloc[0]
                safe_pair = pair.replace("|", "_")
                plot_profile_heatmap(
                    best_row, profiles, jcp_to_idx, feature_names,
                    output_dir / f"profile_heatmap_{safe_pair}.png", dpi=dpi,
                )

        # Step 6: Summary
        n_blend_wins = int(sim_df["blend_beats_individuals"].sum()) if not sim_df.empty else 0
        n_pairs = sim_df["target_pair"].nunique() if not sim_df.empty else 0

        category_counts = {}
        if not sim_df.empty:
            cat_df = categorize_trios(
                sim_df.sort_values("sim_AB_blend", ascending=False).drop_duplicates(subset="target_pair"),
            )
            category_counts = cat_df["category"].value_counts().to_dict()

        numbers = {
            "dataset": dataset,
            "n_raw_trios": len(all_trios),
            "n_all_active_trios": int(all_trios["all_active"].sum()) if not all_trios.empty else 0,
            "n_target_pairs_analyzed": n_pairs,
            "n_trios_analyzed": len(sim_df),
            "n_blend_beats_individuals": n_blend_wins,
            "mean_blend_similarity": float(sim_df["sim_AB_blend"].mean()) if not sim_df.empty else None,
            "category_counts": category_counts,
            "sim_threshold": SIM_THRESHOLD,
            "top_trio": sim_df.sort_values("sim_AB_blend", ascending=False).iloc[0].to_dict() if not sim_df.empty else None,
        }

        description = (
            "Tests whether a dual-target compound's morphological profile looks like a blend of "
            "single-target compounds. Finds trios (compound AB hitting targets A+B, compound A hitting "
            "only A, compound B hitting only B) from Repurposing Hub annotations, then compares their "
            "Cell Painting profiles by cosine similarity and classifies each target pair into "
            "polypharmacology scenarios (additive, dominant, emergent, convergent, divergent)."
        )
        write_summary(
            output_dir / "summary.json",
            description=description,
            numbers=numbers,
            files=index_files(output_dir),
        )
        logger.info(f"Saved: {output_dir / 'summary.json'}")

        logger.success(f"\nPolypharmacology additivity analysis complete: {output_dir}")

    finally:
        meta_con.close()
        copairs_con.close()


if __name__ == "__main__":
    app()
