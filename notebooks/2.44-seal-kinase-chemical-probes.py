"""Kinase / chemical-probe subset visualisation in JUMP.

Visualises the JUMP-overlap subsets of published probe libraries:
  - KCGS (Kinase Chemogenomic Set, SGC-UNC, selective kinase inhibitors)
  - PKIS (Published Kinase Inhibitor Set, GSK / ChEMBL CHEMBL2303647)
  - Chemical Probes (Probes & Drugs Portal; subsumes the SGC chemical-probe set)
  - PKIS2 (Drewry et al., per-compound x per-kinase binding matrix) - optional

Probe-set membership and JUMP matching come from the augmented metadata DB
(`kinase_probes`, `chemical_probes`, and `pkis2_binding` tables), which ingest these
sources through the standard `process_*` pipeline rules keyed by `Metadata_JCP2022`.
No raw probe files are read and no InChIKey re-matching happens here.

PKIS2 differs from the membership sets in that its value is the wide binding matrix,
so `rule process_pkis2` ingests it into the long `pkis2_binding` table (one row per
compound x kinase) rather than a membership flag. If that table is absent (e.g. an
older DB), the PKIS2 panels are skipped and the rest of the notebook runs unchanged.

Usage:
    pixi run python notebooks/2.44-seal-kinase-chemical-probes.py

Outputs: data/processed/kinase-chemical-probes/{dataset}/
    - probe_overlap_summary.csv
    - umap_probe_overlay.{png,pdf}            (small-multiples)
    - kcgs_similarity_heatmap.{png,pdf}       (compound x compound)
    - pkis2_compound_kinase_heatmap.{png,pdf} (compound x kinase binding; if PKIS2 present)
    - summary.json
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import cosine_similarity

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, METADATA_DB, load_profiles, load_umap
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = PROCESSED_DATA_DIR / "kinase-chemical-probes"

ACTIVITY_P_THRESHOLD = 0.10
CONSISTENCY_P_THRESHOLD = 0.05
PKIS2_BINDING_CUTOFF = 80.0  # % inhibition at 1 µM (PKIS2 standard)

# Group_type → metadata column for per-compound consistency (mirrors 2.43)
CONSISTENCY_GROUP_COLS = {
    "repurposing": "Metadata_repurposing_target",
    "uniprot": "Metadata_Uniprot_target",
    "moa": "Metadata_repurposing_moa",
}

# Distinct colours per probe set
PROBE_COLOURS = {
    "KCGS": "#D55E00",
    "PKIS": "#0072B2",
    "PKIS2": "#009E73",
    "Chemical Probes": "#CC79A7",
}


# =============================================================================
# Loaders — probe sets come from the augmented metadata DB
# =============================================================================


def load_kinase_probe_set(meta_con: duckdb.DuckDBPyConnection, probe_set: str) -> pd.DataFrame:
    """Load one kinase probe set (KCGS or PKIS) from the `kinase_probes` table.

    The table only stores probes already matched to JUMP, so every row is an
    in-JUMP hit. Returns columns: JCP2022, probe_id, target_info, probe_set.
    """
    logger.info(f"Loading {probe_set} from kinase_probes table…")
    df = meta_con.execute(
        "SELECT Metadata_JCP2022 AS JCP2022, "
        "Metadata_kinase_probe_name AS probe_id, "
        "Metadata_kinase_target_info AS target_info "
        "FROM kinase_probes WHERE Metadata_kinase_probe_set = ?",
        [probe_set],
    ).df()
    df["probe_set"] = probe_set
    df = df.dropna(subset=["JCP2022"]).drop_duplicates(subset="JCP2022").reset_index(drop=True)
    logger.info(f"  {len(df)} {probe_set} compounds in JUMP")
    return df


def load_chemical_probes(meta_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load chemical probes (Probes & Drugs Portal) from the `chemical_probes` table.

    This set subsumes the SGC chemical-probe collection used in the original 2.44.
    Returns columns: JCP2022, probe_id, target_info, probe_set.
    """
    logger.info("Loading Chemical Probes from chemical_probes table…")
    df = meta_con.execute(
        "SELECT Metadata_JCP2022 AS JCP2022, "
        "Metadata_chmprb_name AS probe_id, "
        "Metadata_chmprb_target_genes AS target_info "
        "FROM chemical_probes WHERE Metadata_chmprb_in_jump = 1"
    ).df()
    df["probe_set"] = "Chemical Probes"
    df = df.dropna(subset=["JCP2022"]).drop_duplicates(subset="JCP2022").reset_index(drop=True)
    logger.info(f"  {len(df)} Chemical Probes compounds in JUMP")
    return df


def load_pkis2(
    meta_con: duckdb.DuckDBPyConnection, binding_cutoff: float
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load the PKIS2 binding matrix from the `pkis2_binding` table in the augmented DB.

    Returns
    -------
    probes : DataFrame[JCP2022, probe_id, target_info, probe_set, n_kinases_bound]
    binding : long DataFrame[JCP2022, kinase, pct_inhibition] (only kinases with ≥1 binder)

    Returns (None, None) when the table is absent or empty (e.g. an older DB built
    before process_pkis2 was wired in), so the caller can skip the PKIS2 panels.
    """
    try:
        long = meta_con.execute(
            "SELECT Metadata_JCP2022 AS JCP2022, "
            "Metadata_pkis2_probe_id AS probe_id, "
            "Metadata_pkis2_kinase AS kinase, "
            "Metadata_pkis2_pct_inhibition AS pct_inhibition "
            "FROM pkis2_binding"
        ).df()
    except duckdb.CatalogException:
        logger.warning(
            "pkis2_binding table not in augmented DB; skipping PKIS2 panels. "
            "Rebuild it with: snakemake process_pkis2 augment_metadata_db -j1."
        )
        return None, None
    if long.empty:
        logger.warning("pkis2_binding table is empty; skipping PKIS2 panels.")
        return None, None

    logger.info("Loading PKIS2 binding matrix from pkis2_binding table…")
    long["pct_inhibition"] = pd.to_numeric(long["pct_inhibition"], errors="coerce")
    long = long.dropna(subset=["pct_inhibition", "JCP2022"])
    n_compounds = long["JCP2022"].nunique()
    logger.info(f"  PKIS2: {n_compounds} JUMP compounds × {long['kinase'].nunique()} kinases")

    # Binders per compound at cutoff
    counts = (
        long.assign(bound=long["pct_inhibition"] >= binding_cutoff)
        .groupby("JCP2022")["bound"]
        .sum()
        .astype(int)
    )
    probe_id = long.drop_duplicates(subset="JCP2022").set_index("JCP2022")["probe_id"]
    probes = pd.DataFrame({"JCP2022": counts.index})
    probes["probe_id"] = probes["JCP2022"].map(probe_id)
    probes["n_kinases_bound"] = probes["JCP2022"].map(counts)
    probes["target_info"] = probes["n_kinases_bound"].apply(
        lambda n: f"{int(n)} kinases @≥{binding_cutoff:.0f}%"
    )
    probes["probe_set"] = "PKIS2"

    # Keep only kinases with ≥1 binder above cutoff for the long binding df
    kinases_with_hits = (
        long.assign(bound=long["pct_inhibition"] >= binding_cutoff)
        .groupby("kinase")["bound"]
        .any()
    )
    keep = set(kinases_with_hits[kinases_with_hits].index)
    long = long[long["kinase"].isin(keep)].reset_index(drop=True)
    return probes[["JCP2022", "probe_id", "target_info", "probe_set", "n_kinases_bound"]], long


# =============================================================================
# Activity / consistency labels (already DB-backed)
# =============================================================================


def load_activity_flags(dataset: str, threshold: float) -> pd.DataFrame:
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        df = query_activity_results(con, dataset)
    finally:
        con.close()
    df = df.rename(columns={"Metadata_JCP2022": "JCP2022"})[["JCP2022", "corrected_p_value", "mean_normalized_average_precision"]]
    df["is_active"] = df["corrected_p_value"] < threshold
    return df


def load_consistency_flags(dataset: str, threshold: float) -> pd.DataFrame:
    """Mirrors 2.43: union flag is_consistent_any across repurposing/uniprot/moa."""
    cop = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    meta = duckdb.connect(str(METADATA_DB), read_only=True)
    consistent: dict[str, set[str]] = {}
    annotated: dict[str, set[str]] = {}
    try:
        for gt, col in CONSISTENCY_GROUP_COLS.items():
            sig = set(cop.execute(
                "SELECT DISTINCT group_value FROM consistency_results "
                "WHERE _dataset=? AND _preprocessing='consistency_no_target2' "
                "AND _filter='all_sources' AND _group_type=? AND _distance='cosine' "
                "AND corrected_p_value < ?",
                [dataset, gt, threshold],
            ).df()["group_value"])
            ann = meta.execute(
                f"SELECT Metadata_JCP2022 AS JCP2022, {col} AS vals "
                f"FROM compound_metadata WHERE {col} IS NOT NULL AND {col} != ''"
            ).df()
            con_set: set[str] = set()
            for jcp, vals in zip(ann["JCP2022"], ann["vals"]):
                pieces = [v.strip() for v in str(vals).split("|") if v.strip()]
                if any(p in sig for p in pieces):
                    con_set.add(jcp)
            consistent[gt] = con_set
            annotated[gt] = set(ann["JCP2022"])
    finally:
        cop.close()
        meta.close()
    all_ann = sorted(set().union(*annotated.values()))
    rows = [{"JCP2022": j, "is_consistent_any": any(j in consistent[gt] for gt in CONSISTENCY_GROUP_COLS)}
            for j in all_ann]
    return pd.DataFrame(rows)


# =============================================================================
# Stats
# =============================================================================


def probe_overlap_stats(
    probes_per_set: dict[str, pd.DataFrame],
    activity: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    """Per-probe-set: counts + activity / consistency rates with Fisher OR vs background."""
    bg_active = activity["is_active"]
    bg_consistent = consistency["is_consistent_any"]
    bg_rate_active = bg_active.mean()
    bg_rate_consistent = bg_consistent.mean()
    n_bg_active = int(bg_active.sum())
    n_bg_consistent = int(bg_consistent.sum())
    n_bg_act_total = int(len(activity))
    n_bg_cons_total = int(len(consistency))

    activity_idx = set(activity["JCP2022"])
    consistency_idx = set(consistency["JCP2022"])
    act_map = activity.set_index("JCP2022")["is_active"]
    con_map = consistency.set_index("JCP2022")["is_consistent_any"]

    rows = []
    for name, df in probes_per_set.items():
        jcp = list({j for j in df["JCP2022"] if isinstance(j, str)})
        in_act = [j for j in jcp if j in activity_idx]
        in_con = [j for j in jcp if j in consistency_idx]
        n_act = int(act_map.loc[in_act].sum()) if in_act else 0
        n_con = int(con_map.loc[in_con].sum()) if in_con else 0
        # Fisher activity
        a = n_act
        b = len(in_act) - n_act
        c = n_bg_active - n_act
        d = (n_bg_act_total - len(in_act)) - (n_bg_active - n_act)
        if min(a + b, c + d) > 0 and len(in_act) >= 5:
            or_act, p_act = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        else:
            or_act, p_act = (np.nan, np.nan)
        # Fisher consistency
        a = n_con
        b = len(in_con) - n_con
        c = n_bg_consistent - n_con
        d = (n_bg_cons_total - len(in_con)) - (n_bg_consistent - n_con)
        if min(a + b, c + d) > 0 and len(in_con) >= 5:
            or_con, p_con = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        else:
            or_con, p_con = (np.nan, np.nan)
        rows.append({
            "probe_set": name,
            "n_in_jump": len(jcp),
            "n_with_activity_label": len(in_act),
            "n_active": n_act,
            "frac_active": n_act / max(len(in_act), 1),
            "activity_OR_vs_bg": or_act,
            "activity_p_value": p_act,
            "n_with_consistency_label": len(in_con),
            "n_consistent": n_con,
            "frac_consistent": n_con / max(len(in_con), 1),
            "consistency_OR_vs_bg": or_con,
            "consistency_p_value": p_con,
            "background_frac_active": bg_rate_active,
            "background_frac_consistent": bg_rate_consistent,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================


def plot_umap_overlay(
    umap_obs: pd.DataFrame,
    probes_per_set: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> None:
    """Small-multiples: background in grey, in-JUMP probes coloured."""
    sets = list(probes_per_set.keys())
    ncol = 2
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 5 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    bg = umap_obs[["umap_x", "umap_y"]].to_numpy()
    for ax, name in zip(axes, sets):
        ax.scatter(bg[:, 0], bg[:, 1], s=1, color="#cccccc", alpha=0.25, rasterized=True)
        jcp = set(probes_per_set[name]["JCP2022"].dropna())
        mask = umap_obs["JCP2022"].isin(jcp)
        ax.scatter(
            umap_obs.loc[mask, "umap_x"],
            umap_obs.loc[mask, "umap_y"],
            s=14, color=PROBE_COLOURS.get(name, "#333"),
            edgecolors="white", linewidths=0.3, alpha=0.9,
        )
        ax.set_title(f"{name} (n={int(mask.sum())} in JUMP)", fontsize=11)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(sets):]:
        ax.set_visible(False)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"umap_probe_overlay{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: umap_probe_overlay.png")


def plot_similarity_heatmap(
    profiles: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    dpi: int,
    figsize: tuple[float, float] = (10, 9),
) -> None:
    """Cosine similarity heatmap with hierarchical clustering on rows/cols."""
    sim = cosine_similarity(profiles)
    # Cluster on 1-sim distance
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(np.clip(dist, 0, None), checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    order = hierarchy.leaves_list(Z)
    sim_ord = sim[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    if len(labels) <= 80:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_path.with_suffix(ext), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}.png")


def plot_pkis2_binding_heatmap(
    binding_long: pd.DataFrame,
    cp_profiles: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
    binding_cutoff: float,
) -> None:
    """Heatmap of PKIS2 in-JUMP compounds (rows) × kinases (cols).

    Row order = hierarchical clustering on Cell Painting cosine similarity.
    Column order = hierarchical clustering on binding-vector similarity.
    Values = % inhibition (clipped to [0, 100]); cells below `binding_cutoff` shown faded.
    """
    # Wide binding matrix
    wide = binding_long.pivot_table(
        index="JCP2022", columns="kinase", values="pct_inhibition", aggfunc="max", fill_value=0.0,
    )
    # Keep only compounds with at least one binder above cutoff
    keep_rows = wide.index[(wide >= binding_cutoff).any(axis=1)]
    wide = wide.loc[keep_rows]
    # Keep only kinases with at least one binder
    keep_cols = wide.columns[(wide >= binding_cutoff).any(axis=0)]
    wide = wide[keep_cols]
    if wide.empty:
        logger.warning("PKIS2 binding wide matrix empty after filtering; skipping heatmap")
        return
    logger.info(f"  PKIS2 heatmap matrix: {wide.shape[0]} compounds × {wide.shape[1]} kinases")
    # Row order — Cell Painting cosine similarity
    common = [j for j in wide.index if j in cp_profiles]
    if len(common) >= 4:
        X = np.vstack([cp_profiles[j] for j in common])
        sim = cosine_similarity(X)
        d = 1.0 - sim
        np.fill_diagonal(d, 0.0)
        Z = hierarchy.linkage(squareform(np.clip(d, 0, None), checks=False), method="average")
        row_order = [common[i] for i in hierarchy.leaves_list(Z)]
        # Append any rows without CP profile at the bottom
        leftover = [j for j in wide.index if j not in cp_profiles]
        wide = wide.loc[row_order + leftover]
    # Column order — binding similarity
    if wide.shape[1] >= 3:
        col_sim = cosine_similarity(wide.values.T)
        cd = 1.0 - col_sim
        np.fill_diagonal(cd, 0.0)
        Zc = hierarchy.linkage(squareform(np.clip(cd, 0, None), checks=False), method="average")
        col_order = [wide.columns[i] for i in hierarchy.leaves_list(Zc)]
        wide = wide[col_order]
    fig, ax = plt.subplots(figsize=(max(8, wide.shape[1] * 0.07), max(6, wide.shape[0] * 0.08)))
    im = ax.imshow(wide.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{wide.shape[1]} kinases (clustered by binding similarity)")
    ax.set_ylabel(f"{wide.shape[0]} PKIS2 compounds (clustered by Cell Painting similarity)")
    ax.set_title(f"PKIS2 binding (% inhibition, cutoff ≥{binding_cutoff:.0f}%)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="% inhibition @ 1 µM")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_path.with_suffix(ext), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}.png")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    dataset: str = typer.Option("compound_no_source7", "--dataset", "-d"),
    activity_threshold: float = typer.Option(ACTIVITY_P_THRESHOLD, "--activity-threshold"),
    consistency_threshold: float = typer.Option(CONSISTENCY_P_THRESHOLD, "--consistency-threshold"),
    pkis2_cutoff: float = typer.Option(PKIS2_BINDING_CUTOFF, "--pkis2-cutoff",
                                        help="% inhibition cutoff for calling a PKIS2 kinase 'bound'"),
    umap_metric: str = typer.Option("cosine", "--umap-metric"),
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi"),
) -> None:
    out = output_dir or (OUTPUT_DIR / dataset)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out}")

    # 1. Load labels (activity + consistency)
    activity = load_activity_flags(dataset, activity_threshold)
    consistency = load_consistency_flags(dataset, consistency_threshold)
    logger.info(f"Activity: {len(activity):,} compounds, {activity['is_active'].sum():,} active "
                f"({activity['is_active'].mean():.1%})")
    logger.info(f"Consistency: {len(consistency):,} annotated, "
                f"{consistency['is_consistent_any'].sum():,} consistent")

    # 2. Load probe sets from the augmented metadata DB
    meta = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        kcgs = load_kinase_probe_set(meta, "KCGS")
        pkis = load_kinase_probe_set(meta, "PKIS")
        chem = load_chemical_probes(meta)
        # PKIS2 is skipped gracefully if the pkis2_binding table is absent/empty
        pkis2, pkis2_binding = load_pkis2(meta, pkis2_cutoff)
    finally:
        meta.close()

    probes_per_set: dict[str, pd.DataFrame] = {"KCGS": kcgs, "PKIS": pkis, "Chemical Probes": chem}
    if pkis2 is not None:
        probes_per_set["PKIS2"] = pkis2
    for name, df in probes_per_set.items():
        logger.info(f"  {name}: {len(df)} compounds in JUMP")

    # 3. Overlap summary
    summary_df = probe_overlap_stats(probes_per_set, activity, consistency)
    summary_df.to_csv(out / "probe_overlap_summary.csv", index=False)
    logger.info(f"Saved: probe_overlap_summary.csv\n{summary_df.to_string(index=False)}")

    # 4. UMAP overlay (perturbation-level cosine UMAP)
    try:
        adata_umap = load_umap(dataset, level="perturbation", metric=umap_metric, filter_name="all")
        umap_obs = pd.DataFrame({
            "JCP2022": adata_umap.obs["JCP2022"].values,
            "umap_x": adata_umap.obsm["X_umap"][:, 0],
            "umap_y": adata_umap.obsm["X_umap"][:, 1],
        })
        plot_umap_overlay(umap_obs, probes_per_set, out, dpi)
    except FileNotFoundError as e:
        logger.warning(f"Skipping UMAP overlay: {e}")
        umap_obs = None

    # 5. Load perturbation-level profiles (for heatmaps)
    adata = load_profiles(dataset, level="perturbation")
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}
    feat_X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    # 6. KCGS similarity heatmap
    kcgs_jcps = [j for j in kcgs["JCP2022"] if j in jcp_to_idx]
    if len(kcgs_jcps) >= 3:
        X = np.vstack([feat_X[jcp_to_idx[j]] for j in kcgs_jcps])
        labels = [j for j in kcgs_jcps]
        plot_similarity_heatmap(
            X, labels,
            title=f"KCGS Cell Painting cosine similarity (n={len(kcgs_jcps)} JUMP-overlap)",
            output_path=out / "kcgs_similarity_heatmap", dpi=dpi,
        )
    else:
        logger.warning("Too few KCGS overlap compounds for heatmap")

    # 7. PKIS2 compound × kinase binding heatmap (only if PKIS2 is available)
    if pkis2 is not None and pkis2_binding is not None:
        cp_profiles = {j: feat_X[jcp_to_idx[j]] for j in pkis2["JCP2022"] if j in jcp_to_idx}
        plot_pkis2_binding_heatmap(
            pkis2_binding, cp_profiles,
            output_path=out / "pkis2_compound_kinase_heatmap", dpi=dpi,
            binding_cutoff=pkis2_cutoff,
        )

    # 8. Summary JSON
    summary = {
        "dataset": dataset,
        "activity_threshold": activity_threshold,
        "consistency_threshold": consistency_threshold,
        "pkis2_binding_cutoff": pkis2_cutoff,
        "pkis2_included": pkis2 is not None,
        "umap_metric": umap_metric,
        "n_in_jump": {k: len(v) for k, v in probes_per_set.items()},
        "overlap_summary": summary_df.to_dict(orient="records"),
    }
    description = (
        "Visualises the JUMP-overlap subsets of published kinase / chemical-probe libraries "
        "(KCGS, PKIS, Chemical Probes, and optionally PKIS2), drawn from the augmented metadata DB, "
        "and reports per-probe-set counts plus phenotypic activity and consistency enrichment versus "
        "background. Produces a probe-overlap table, a small-multiples UMAP overlay, a KCGS Cell "
        "Painting similarity heatmap, and (when PKIS2 is present) a compound-by-kinase binding heatmap."
    )
    write_summary(
        out / "summary.json",
        description=description,
        numbers=summary,
        files=index_files(out),
    )
    logger.info("Saved: summary.json")


if __name__ == "__main__":
    app()
