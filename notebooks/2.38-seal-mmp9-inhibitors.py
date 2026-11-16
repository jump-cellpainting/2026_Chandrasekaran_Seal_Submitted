"""Identify target inhibitors among JUMP compounds with MoA context and UMAP visualization.

Searches all annotation sources in the metadata DB (repurposing hub, chemical probes,
motive/OpenTargets) for compounds targeting a given gene (default: MMP9). Additional
compounds found via external sources (e.g., ChEMBL bioactivity structure matching) can
be included via --extra-jcps.

Also checks phenotypic activity (from copairs results) and computes
pairwise profile similarity among target compounds.

Usage:
    pixi run -e cheminformatics python notebooks/2.38-seal-mmp9-inhibitors.py
    pixi run -e cheminformatics python notebooks/2.38-seal-mmp9-inhibitors.py --target EGFR
    pixi run -e cheminformatics python notebooks/2.38-seal-mmp9-inhibitors.py --skip-plots
    pixi run -e cheminformatics python notebooks/2.38-seal-mmp9-inhibitors.py --extra-jcps JCP2022_000001 --extra-jcps JCP2022_000002

Outputs: data/processed/mmp9-inhibitors/
    - target_inhibitors.csv: JUMP compounds targeting the gene (with activity columns)
    - moa_context.csv: All compounds sharing the same MoA(s)
    - umap_target_vs_background.png: Target compounds on full UMAP
    - umap_moa_context.png: Target + MoA peers colored by group
    - similarity_heatmap.png: Pairwise cosine similarity among target compounds
    - summary.json: Compound counts, MoA breakdown, activity status
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity

from jump_production.config import DEFAULT_DPI, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, load_profiles, load_umap
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

DATASET = "compound_no_source7"
METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "mmp9-inhibitors"
MCE_FILE = EXTERNAL_DATA_DIR / "mmp9_inhibitors.csv"

# Additional MMP9 inhibitors found via ChEMBL bioactivity (CHEMBL321, IC50 < 1uM)
# structure-matched to JUMP by InChIKey. These are not annotated as MMP9 inhibitors
# in any pipeline annotation source (repurposing hub, chemical probes, motive), so
# their Metadata_repurposing_target and Metadata_repurposing_moa columns will be empty.
# Batimastat is in the repurposing hub but annotated as MMP2/8/12/16 (not MMP9).
EXTRA_MMP9_JCPS = [
    "JCP2022_042691",  # tanomastat (CHEMBL261932)
    "JCP2022_103291",  # batimastat (CHEMBL279786)
    "JCP2022_109077",  # prinomastat (CHEMBL75094)
    "JCP2022_111234",  # S-3304 (CHEMBL297792)
]

# Fallback names for compounds not in the Repurposing Hub
CHEMBL_NAMES = {
    "JCP2022_042691": "tanomastat",
    "JCP2022_109077": "prinomastat",
    "JCP2022_111234": "S-3304",
}

app = typer.Typer()


def load_mce_inhibitors(
    con: duckdb.DuckDBPyConnection,
    mce_file: Path,
) -> tuple[list[str], dict[str, str]]:
    """Match MedChemExpress MMP9 inhibitors to JUMP via InChIKey14 skeleton.

    Returns:
        Tuple of (matched JCP2022 IDs, name mapping {jcp: compound_name}).
    """
    mce = pd.read_csv(mce_file)
    mce = mce[mce["INCHIKEY"] != "N/A"].drop_duplicates(subset=["INCHIKEY"]).copy()
    mce["skeleton"] = mce["INCHIKEY"].str[:14]
    logger.info(f"MCE file: {len(mce)} compounds with valid InChIKey (of {len(pd.read_csv(mce_file))} total)")

    jump = con.execute("""
        SELECT Metadata_JCP2022, Metadata_InChIKey
        FROM compound_metadata
        WHERE Metadata_InChIKey IS NOT NULL
    """).df()
    jump["skeleton"] = jump["Metadata_InChIKey"].str[:14]

    merged = mce.merge(jump, on="skeleton").drop_duplicates(subset=["Metadata_JCP2022"])
    logger.info(f"MCE matched to JUMP: {len(merged)} compounds")

    # Build name mapping from INPUT or PREFERRED_NAME
    name_map = {}
    for _, row in merged.iterrows():
        name = row["PREFERRED_NAME"] if pd.notna(row["PREFERRED_NAME"]) and row["PREFERRED_NAME"] != "N/A" else row["INPUT"]
        name_map[row["Metadata_JCP2022"]] = name

    return merged["Metadata_JCP2022"].tolist(), name_map


def get_target_compounds(
    con: duckdb.DuckDBPyConnection,
    target: str,
    extra_jcps: list[str] | None = None,
    mce_jcps: list[str] | None = None,
) -> pd.DataFrame:
    """Query compounds targeting a gene across all annotation sources, plus extras.

    Searches: repurposing hub, chemical probes, motive/OpenTargets.
    Extra JCPs (e.g., from ChEMBL structure matching) and MCE JCPs are included
    even if they lack a target annotation containing the gene symbol.
    """
    pattern = f"%{target}%"
    # Find all JCP2022 IDs annotated with the target in any source
    jcp_df = con.execute(
        """
        SELECT DISTINCT Metadata_JCP2022
        FROM compound_metadata
        WHERE Metadata_repurposing_target LIKE ?
           OR Metadata_chmprb_target_genes LIKE ?
           OR Metadata_motive_gene_opentargets LIKE ?
        """,
        [pattern, pattern, pattern],
    ).df()

    all_jcps = set(jcp_df["Metadata_JCP2022"])
    if extra_jcps:
        all_jcps.update(extra_jcps)
    if mce_jcps:
        all_jcps.update(mce_jcps)

    if not all_jcps:
        return pd.DataFrame()

    # Get metadata + repurposing hub annotations for these compounds
    df = con.execute(
        """
        SELECT
            cm.Metadata_JCP2022,
            r.Metadata_repurposing_name,
            r.Metadata_repurposing_target,
            r.Metadata_repurposing_moa,
            r.Metadata_repurposing_clinical_phase,
            r.Metadata_repurposing_disease_area,
            r.Metadata_repurposing_indication,
            cm.Metadata_SMILES,
            cm.Metadata_InChIKey
        FROM compound_metadata cm
        LEFT JOIN repurposing_hub_annotations r
          ON cm.Metadata_JCP2022 = r.Metadata_JCP2022
        WHERE cm.Metadata_JCP2022 = ANY(?)
        """,
        [list(all_jcps)],
    ).df()

    # Determine which annotation sources flagged this compound for the target
    all_jcp_list = list(df["Metadata_JCP2022"])
    other_sources = con.execute(
        """
        SELECT
            jcps.jcp AS Metadata_JCP2022,
            bool_or(cp.Metadata_JCP2022 IS NOT NULL) AS has_chem_probes,
            bool_or(mt.Metadata_JCP2022 IS NOT NULL) AS has_motive
        FROM (SELECT UNNEST(?) AS jcp) jcps
        LEFT JOIN chemical_probes cp
          ON jcps.jcp = cp.Metadata_JCP2022 AND cp.Metadata_chmprb_target_genes LIKE ?
        LEFT JOIN motive_targets_opentargets mt
          ON jcps.jcp = mt.Metadata_JCP2022 AND mt.Metadata_motive_gene_opentargets LIKE ?
        GROUP BY jcps.jcp
        """,
        [all_jcp_list, pattern, pattern],
    ).df()
    other_map = {
        row["Metadata_JCP2022"]: (row["has_chem_probes"], row["has_motive"]) for _, row in other_sources.iterrows()
    }
    extra_set = set(extra_jcps) if extra_jcps else set()
    mce_set = set(mce_jcps) if mce_jcps else set()
    sources_col = []
    for _, row in df.iterrows():
        sources = []
        if pd.notna(row.get("Metadata_repurposing_target")) and target in str(row["Metadata_repurposing_target"]):
            sources.append("repurposing")
        jcp = row["Metadata_JCP2022"]
        cp_flag, mt_flag = other_map.get(jcp, (False, False))
        if cp_flag:
            sources.append("chem_probes")
        if mt_flag:
            sources.append("motive")
        if jcp in extra_set:
            sources.append("chembl_structure")
        if jcp in mce_set:
            sources.append("mce")
        sources_col.append("|".join(sources) if sources else "extra")
    df["annotation_sources"] = sources_col

    df["target_selective"] = df["Metadata_repurposing_target"].apply(
        lambda x: x.strip() == target if pd.notna(x) else False
    )
    df["n_targets"] = df["Metadata_repurposing_target"].apply(lambda x: len(x.split("|")) if pd.notna(x) else 0)
    df = df.sort_values(["target_selective", "n_targets"], ascending=[False, True])
    return df


def get_moa_context(con: duckdb.DuckDBPyConnection, target_df: pd.DataFrame) -> pd.DataFrame:
    """Find all compounds sharing the same MoA(s) as the target compounds."""
    all_moas = {m.strip() for s in target_df["Metadata_repurposing_moa"].dropna() for m in s.split("|") if m.strip()}

    if not all_moas:
        return pd.DataFrame()

    moa_patterns = [f"%{moa}%" for moa in all_moas]
    conditions = " OR ".join(["r.Metadata_repurposing_moa LIKE ?" for _ in all_moas])
    moa_df = con.execute(
        f"""
        SELECT
            r.Metadata_JCP2022,
            r.Metadata_repurposing_name,
            r.Metadata_repurposing_target,
            r.Metadata_repurposing_moa,
            r.Metadata_repurposing_clinical_phase
        FROM repurposing_hub_annotations r
        WHERE {conditions}
        """,
        moa_patterns,
    ).df()

    target_jcps = set(target_df["Metadata_JCP2022"])
    moa_df["is_target_compound"] = moa_df["Metadata_JCP2022"].isin(target_jcps)
    moa_df = moa_df.sort_values(["is_target_compound", "Metadata_repurposing_moa"], ascending=[False, True])
    return moa_df


# =============================================================================
# UMAP Visualization
# =============================================================================


def plot_target_umap(
    adata: sc.AnnData,
    target_jcps: set[str],
    target: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot UMAP with target compounds highlighted over gray background."""
    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    valid = target_jcps & set(jcp_to_idx)
    if valid:
        idx = np.array([jcp_to_idx[j] for j in valid])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#d62728",
            s=40,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"{target} ({len(idx)})",
            rasterized=True,
        )

    ax.set_title(f"{target} inhibitors in morphological space")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_moa_context_umap(
    adata: sc.AnnData,
    target_jcps: set[str],
    moa_df: pd.DataFrame,
    target: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot UMAP with target compounds + MoA peers colored by group."""
    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    moa_peer_jcps = set(moa_df[~moa_df["is_target_compound"]]["Metadata_JCP2022"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    valid_peers = moa_peer_jcps & set(jcp_to_idx)
    if valid_peers:
        idx = np.array([jcp_to_idx[j] for j in valid_peers])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#1f77b4",
            s=20,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
            label=f"Same MoA peers ({len(idx)})",
            rasterized=True,
        )

    valid_tgt = target_jcps & set(jcp_to_idx)
    if valid_tgt:
        idx = np.array([jcp_to_idx[j] for j in valid_tgt])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c="#d62728",
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"{target} ({len(idx)})",
            rasterized=True,
        )

    ax.set_title(f"{target} inhibitors vs MoA peers in morphological space")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_per_moa_umap(
    adata: sc.AnnData,
    moa_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target: str,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot one UMAP per MoA showing target compounds vs other compounds with that MoA."""
    all_moas = {m.strip() for s in target_df["Metadata_repurposing_moa"].dropna() for m in s.split("|") if m.strip()}
    if not all_moas:
        return

    xlim, ylim = compute_umap_bounds(adata)
    coords = adata.obsm["X_umap"]
    jcp_ids = set(adata.obs["JCP2022"].values)
    jcp_to_idx = {j: i for i, j in enumerate(adata.obs["JCP2022"].values)}

    n_moas = len(all_moas)
    cols = min(n_moas, 3)
    rows = (n_moas + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    for ax_idx, moa in enumerate(sorted(all_moas)):
        ax = axes[ax_idx // cols][ax_idx % cols]
        ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=0.5, alpha=0.2, rasterized=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        peer_jcps = (
            set(
                moa_df[
                    (~moa_df["is_target_compound"])
                    & (moa_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False))
                ]["Metadata_JCP2022"]
            )
            & jcp_ids
        )

        valid_peers = peer_jcps & set(jcp_to_idx)
        if valid_peers:
            idx = np.array([jcp_to_idx[j] for j in valid_peers])
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c="#1f77b4",
                s=20,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.3,
                zorder=4,
                label=f"Peers ({len(idx)})",
                rasterized=True,
            )

        tgt_with_moa = (
            set(
                target_df[target_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False)][
                    "Metadata_JCP2022"
                ]
            )
            & jcp_ids
        )

        valid_tgt = tgt_with_moa & set(jcp_to_idx)
        if valid_tgt:
            idx = np.array([jcp_to_idx[j] for j in valid_tgt])
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                c="#d62728",
                s=60,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
                label=f"{target} ({len(idx)})",
                rasterized=True,
            )

        ax.set_title(moa, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

    for ax_idx in range(n_moas, rows * cols):
        axes[ax_idx // cols][ax_idx % cols].set_visible(False)

    fig.suptitle(f"{target} inhibitors by MoA", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "umap_per_moa.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'umap_per_moa.png'}")


# =============================================================================
# Activity & Similarity
# =============================================================================


def get_activity_status(target_df: pd.DataFrame) -> pd.DataFrame:
    """Join target compounds with phenotypic activity from copairs results."""
    jcp_ids = target_df["Metadata_JCP2022"].tolist()
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        activity = con.execute(
            """
            SELECT Metadata_JCP2022,
                   mean_normalized_average_precision AS nmAP,
                   corrected_p_value AS fdr_p,
                   below_corrected_p AS active
            FROM activity_results
            WHERE _dataset = ? AND _preprocessing = 'activity_no_target2'
              AND _filter = 'all_sources' AND _activity_params = 'default'
              AND Metadata_JCP2022 = ANY(?)
            """,
            [DATASET, jcp_ids],
        ).df()
    finally:
        con.close()
    merged = target_df.merge(activity, on="Metadata_JCP2022", how="left")
    return merged.sort_values("nmAP", ascending=False)


def plot_similarity_heatmap(
    target_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Compute and plot pairwise cosine similarity among target compound profiles."""
    adata = load_profiles(DATASET, level="perturbation")
    jcps = list(target_df["Metadata_JCP2022"])
    names_map = {
        jcp: name if pd.notna(name) else jcp
        for jcp, name in zip(target_df["Metadata_JCP2022"], target_df["Metadata_repurposing_name"])
    }

    mask = adata.obs["JCP2022"].isin(jcps)
    sub = adata[mask].copy()
    if sub.n_obs < 2:
        logger.warning(f"Only {sub.n_obs} compounds have profiles — skipping heatmap")
        return

    logger.info(f"Computing similarity for {sub.n_obs} / {len(jcps)} compounds with profiles")

    X = sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    sim = cosine_similarity(X)

    labels = [names_map.get(j, j) for j in sub.obs["JCP2022"]]

    # Cluster
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist), method="ward")
    order = leaves_list(link)
    sim_ordered = sim[np.ix_(order, order)]
    labels_ordered = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_ordered, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(labels_ordered)))
    ax.set_xticklabels(labels_ordered, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels_ordered)))
    ax.set_yticklabels(labels_ordered, fontsize=10)

    for i in range(len(labels_ordered)):
        for j in range(len(labels_ordered)):
            ax.text(
                j,
                i,
                f"{sim_ordered[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(sim_ordered[i, j]) > 0.5 else "black",
            )

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    ax.set_title("Target inhibitor profile similarity\n(perturbation-level profiles)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    target: str = typer.Option("MMP9", "--target", "-t", help="Gene symbol to search for"),
    extra_jcps: list[str] = typer.Option(EXTRA_MMP9_JCPS, "--extra-jcps", help="Additional JCP2022 IDs to include"),
    mce_file: Path | None = typer.Option(MCE_FILE, "--mce-file", help="MedChemExpress inhibitor CSV (set to empty to skip)"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
) -> None:
    """Identify target inhibitors among JUMP compounds with MoA context and UMAP."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # 0. Load MCE inhibitors if file exists
        mce_jcps = []
        mce_names = {}
        if mce_file and mce_file.exists():
            logger.info(f"Loading MCE inhibitors from {mce_file}...")
            mce_jcps, mce_names = load_mce_inhibitors(con, mce_file)
        else:
            logger.info("No MCE file provided or not found, skipping MCE matching")

        # 1. Get target compounds from all annotation sources + extras + MCE
        logger.info(f"Querying compounds targeting {target}...")
        target_df = get_target_compounds(con, target, extra_jcps=extra_jcps or None, mce_jcps=mce_jcps or None)
        # Fill missing names from ChEMBL lookup and MCE names
        all_fallback_names = {**CHEMBL_NAMES, **mce_names}
        mask = target_df["Metadata_repurposing_name"].isna()
        target_df.loc[mask, "Metadata_repurposing_name"] = target_df.loc[mask, "Metadata_JCP2022"].map(all_fallback_names)
        logger.info(f"Found {len(target_df)} JUMP compounds targeting {target}")

        if len(target_df) == 0:
            logger.warning(f"No compounds found for target {target}")
            raise typer.Exit(0)

        # 1b. Get activity status
        logger.info("Checking phenotypic activity...")
        target_df = get_activity_status(target_df)
        n_active = int(target_df["active"].sum())
        n_profiled = int(target_df["nmAP"].notna().sum())
        logger.info(f"Active: {n_active} / {n_profiled} profiled ({len(target_df) - n_profiled} missing profiles)")

        target_df.to_csv(output_dir / "target_inhibitors.csv", index=False)
        logger.info(f"Saved: {output_dir / 'target_inhibitors.csv'}")

        # 2. Get MoA context
        logger.info("Finding compounds with shared MoA...")
        moa_df = get_moa_context(con, target_df)

        n_same_moa = len(moa_df[~moa_df["is_target_compound"]]) if len(moa_df) > 0 else 0
        logger.info(f"Found {n_same_moa} additional compounds sharing same MoA(s)")

        if len(moa_df) > 0:
            moa_df.to_csv(output_dir / "moa_context.csv", index=False)
            logger.info(f"Saved: {output_dir / 'moa_context.csv'}")

        # 3. MoA breakdown
        moa_counts = {}
        if len(moa_df) > 0:
            for moa_str in target_df["Metadata_repurposing_moa"].dropna():
                for moa in moa_str.split("|"):
                    moa = moa.strip()
                    if moa and moa not in moa_counts:
                        n_total = len(
                            moa_df[moa_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False)]
                        )
                        n_target = len(
                            target_df[target_df["Metadata_repurposing_moa"].fillna("").str.contains(moa, regex=False)]
                        )
                        moa_counts[moa] = {"target_compounds": int(n_target), "total_compounds": int(n_total)}

        # 4. UMAP visualization
        if not skip_plots:
            logger.info("\nLoading UMAP coordinates...")
            adata = load_umap(DATASET, level="perturbation", metric="cosine", filter_name="all")

            target_jcps = set(target_df["Metadata_JCP2022"])

            # Plot 1: Target vs background
            plot_target_umap(adata, target_jcps, target, output_dir / "umap_target_vs_background.png", dpi)

            # Plot 2: Target + MoA peers
            if len(moa_df) > 0:
                plot_moa_context_umap(adata, target_jcps, moa_df, target, output_dir / "umap_moa_context.png", dpi)

                # Plot 3: Per-MoA breakdown
                plot_per_moa_umap(adata, moa_df, target_df, target, output_dir, dpi)

            # Plot 4: Pairwise similarity heatmap
            plot_similarity_heatmap(target_df, output_dir / "similarity_heatmap.png", dpi)

        # 5. Summary
        n_selective = int(target_df["target_selective"].sum())
        compound_cols = [
            "Metadata_JCP2022",
            "Metadata_repurposing_name",
            "Metadata_repurposing_target",
            "Metadata_repurposing_moa",
            "annotation_sources",
            "target_selective",
            "nmAP",
            "fdr_p",
            "active",
        ]
        numbers = {
            "target_gene": target,
            "total_compounds": len(target_df),
            "target_selective": n_selective,
            "multi_target": int(len(target_df) - n_selective),
            "activity": {
                "profiled": n_profiled,
                "active": n_active,
                "missing_profiles": int(len(target_df) - n_profiled),
            },
            "moa_context": {
                "shared_moa_compounds": n_same_moa,
                "moa_breakdown": moa_counts,
            },
            "compounds": target_df[compound_cols].to_dict(orient="records"),
        }

        description = (
            f"Identifies JUMP compounds targeting {target} across all annotation sources "
            "(repurposing hub, chemical probes, motive/OpenTargets) plus ChEMBL/MedChemExpress "
            "structure-matched extras, then reports their phenotypic activity, MoA context, "
            "UMAP position, and pairwise profile similarity."
        )

        write_summary(
            output_dir / "summary.json",
            description=description,
            numbers=numbers,
            files=index_files(output_dir),
        )
        logger.info(f"Saved: {output_dir / 'summary.json'}")

        # 6. Log results
        logger.info(f"\n{'=' * 60}")
        logger.info(f"{target} inhibitors in JUMP: {len(target_df)} compounds")
        logger.info(f"  Selective ({target} only): {n_selective}")
        logger.info(f"  Multi-target: {len(target_df) - n_selective}")
        logger.info(f"  Phenotypically active: {n_active} / {n_profiled} profiled")
        logger.info(f"{'=' * 60}")

        for _, row in target_df.iterrows():
            name = (
                row["Metadata_repurposing_name"]
                if pd.notna(row["Metadata_repurposing_name"])
                else row["Metadata_JCP2022"]
            )
            sel = " [SELECTIVE]" if row["target_selective"] else ""
            if pd.notna(row["nmAP"]):
                act = f" | nmAP={row['nmAP']:.3f} {'ACTIVE' if row['active'] else ''}"
            else:
                act = " | no profile"
            tgt = row["Metadata_repurposing_target"] if pd.notna(row.get("Metadata_repurposing_target")) else "—"
            src = f" [{row['annotation_sources']}]" if pd.notna(row.get("annotation_sources")) else ""
            logger.info(f"  {row['Metadata_JCP2022']} | {name} | {tgt}{sel}{act}{src}")

        if moa_counts:
            logger.info("\n--- MoA Context ---")
            for moa, counts in moa_counts.items():
                logger.info(
                    f"  {moa}: {counts['target_compounds']} {target} compounds, "
                    f"{counts['total_compounds']} total in JUMP"
                )

        logger.success(f"\n{target} inhibitor analysis complete: {output_dir}")

    finally:
        con.close()


if __name__ == "__main__":
    app()
