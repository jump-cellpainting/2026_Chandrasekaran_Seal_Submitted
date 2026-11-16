"""Substructure-Morphology Relationship Analysis.

Bidirectional analysis linking chemical substructure to morphological phenotype:

Analysis 1 (Substructure → Morphology): For each Murcko scaffold, is there a
consistent morphological fingerprint? Measures within-scaffold cosine similarity
vs random baseline.

Analysis 2 (Morphology → Substructure): For each morphological cluster, is any
scaffold or BRICS fragment overrepresented? Uses Leiden clustering + Fisher's
exact test for enrichment.

Requires:
    data/interim/jump_metadata_augmented.duckdb (Murcko scaffolds, SMILES)
    data/interim/anndata/ (perturbation profiles, UMAP)

Usage:
    pixi run -e cheminformatics python notebooks/2.40-seal-substructure-morphology.py

    # Skip plots for faster iteration
    pixi run -e cheminformatics python notebooks/2.40-seal-substructure-morphology.py --skip-plots

Outputs: data/processed/substructure-morphology/{dataset}/
    - scaffold_coherence.csv: scaffold, n_compounds, mean_cosine_sim, p_value
    - cluster_enrichment.csv: cluster, scaffold/fragment, fold_enrichment, p_value
    - cluster_assignments.csv: JCP2022, cluster (compound → cluster mapping)
    - cluster_features.csv: top 20 distinguishing features per cluster
    - browser.html: self-contained HTML browser for scaffold ↔ cluster lookup
    - coherence_distribution.png: within-scaffold similarity vs null
    - top_scaffolds_vignettes.png: top 5 most coherent scaffolds
    - cluster_umap.png: UMAP colored by Leiden cluster
    - summary.json
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from joblib import Parallel, delayed
from loguru import logger
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.io import load_profiles, load_umap
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary
from jump_production.visualization import compute_umap_bounds

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
COPAIRS_DB = PROCESSED_DATA_DIR / "copairs_results.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "substructure-morphology"

MIN_SCAFFOLD_COMPOUNDS = 5
N_PERMUTATIONS = 1000
RANDOM_STATE = 42
N_TOP_SCAFFOLDS = 5

# BRICS fragment size limits (from 2.33)
BRICS_MIN_ATOMS = 2
BRICS_MAX_ATOMS = 18

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_scaffolds(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load Murcko scaffolds from metadata database."""
    df = con.execute("""
        SELECT Metadata_JCP2022 as JCP2022,
               Metadata_MurckoScaffold as scaffold
        FROM compound_properties
        WHERE Metadata_MurckoScaffold IS NOT NULL
          AND Metadata_MurckoScaffold != ''
    """).df()
    logger.info(f"Loaded scaffolds: {len(df):,} compounds, {df['scaffold'].nunique():,} unique scaffolds")
    return df


def load_smiles(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load SMILES from metadata database."""
    df = con.execute("""
        SELECT Metadata_JCP2022 as JCP2022,
               Metadata_SMILES as SMILES
        FROM compound
        WHERE Metadata_SMILES IS NOT NULL
    """).df()
    logger.info(f"Loaded SMILES: {len(df):,} compounds")
    return df


# =============================================================================
# Analysis 1: Substructure → Morphology
# =============================================================================


def compute_within_group_similarity(X: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a group of profiles."""
    if X.shape[0] < 2:
        return np.nan
    sim_matrix = cosine_similarity(X)
    # Upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(sim_matrix.shape[0], k=1)
    return float(np.mean(sim_matrix[triu_idx]))


def compute_null_similarities(X: np.ndarray, random_groups: list[np.ndarray]) -> np.ndarray:
    """Within-group similarity of each random group of row indices."""
    return np.array([compute_within_group_similarity(X[idx]) for idx in random_groups])


def compute_scaffold_coherence(
    adata,
    scaffold_df: pd.DataFrame,
    min_compounds: int,
    n_permutations: int,
    rng: np.random.Generator,
    workers: int = 1,
) -> pd.DataFrame:
    """Compute morphological coherence for each scaffold group.

    For each scaffold with ≥min_compounds members that have profiles,
    computes mean pairwise cosine similarity and compares to random groups.
    The random groups are all drawn here, in scaffold order, so the result does not
    depend on `workers`; only their similarities are computed side by side.
    """
    profile_jcp = set(adata.obs["JCP2022"].values)
    jcp_to_idx = {k: i for i, k in enumerate(adata.obs["JCP2022"].values)}

    # Filter scaffold_df to compounds with profiles
    scaffold_df = scaffold_df[scaffold_df["JCP2022"].isin(profile_jcp)].copy()

    # Filter to scaffolds with ≥min_compounds
    scaffold_counts = scaffold_df.groupby("scaffold").size()
    valid_scaffolds = scaffold_counts[scaffold_counts >= min_compounds].index
    scaffold_df = scaffold_df[scaffold_df["scaffold"].isin(valid_scaffolds)]

    logger.info(
        f"Scaffolds with ≥{min_compounds} profiled compounds: {len(valid_scaffolds):,} "
        f"({scaffold_df['JCP2022'].nunique():,} compounds)"
    )

    X = adata.X
    all_indices = np.arange(X.shape[0])

    scaffolds = []
    for scaffold, group in tqdm(
        scaffold_df.groupby("scaffold"), desc="Scaffold coherence", total=len(valid_scaffolds)
    ):
        member_indices = np.array([jcp_to_idx[jcp] for jcp in group["JCP2022"]])
        X_group = X[member_indices]
        n = len(member_indices)

        obs_sim = compute_within_group_similarity(X_group)
        if np.isnan(obs_sim):
            continue

        # Centroid
        centroid = X_group.mean(axis=0)

        # Permutation test: sample random groups of same size
        random_groups = [rng.choice(all_indices, size=n, replace=False) for _ in range(n_permutations)]
        scaffolds.append((scaffold, n, obs_sim, centroid, random_groups))

    all_null_sims = Parallel(n_jobs=workers)(
        delayed(compute_null_similarities)(X, random_groups) for *_, random_groups in scaffolds
    )

    results = []
    for (scaffold, n, obs_sim, centroid, _), null_sims in zip(scaffolds, all_null_sims):
        # p-value: fraction of null >= observed
        p_value = (np.sum(null_sims >= obs_sim) + 1) / (n_permutations + 1)

        results.append({
            "scaffold": scaffold,
            "n_compounds": n,
            "mean_cosine_sim": obs_sim,
            "null_mean": float(np.mean(null_sims)),
            "null_std": float(np.std(null_sims)),
            "z_score": (obs_sim - np.mean(null_sims)) / max(np.std(null_sims), 1e-10),
            "p_value": p_value,
            "centroid": centroid.tolist(),
        })

    coherence_df = pd.DataFrame(results)
    _, corrected_p, _, _ = multipletests(coherence_df["p_value"], method="fdr_bh")
    coherence_df["corrected_p_value"] = corrected_p
    coherence_df = coherence_df.sort_values(
        ["corrected_p_value", "p_value", "scaffold"]
    ).reset_index(drop=True)

    n_sig_nominal = int((coherence_df["p_value"] < 0.05).sum())
    n_sig_fdr = int((coherence_df["corrected_p_value"] < 0.05).sum())
    logger.info(
        f"Scaffold coherence: {len(coherence_df)} scaffolds, "
        f"{n_sig_nominal} nominally significant "
        f"(p < 0.05, {n_sig_nominal / len(coherence_df) * 100:.1f}%), "
        f"{n_sig_fdr} significant at 5% FDR (BH, {n_sig_fdr / len(coherence_df) * 100:.1f}%)"
    )

    return coherence_df


# =============================================================================
# Analysis 2: Morphology → Substructure
# =============================================================================


def count_heavy_atoms(smiles: str) -> int:
    """Count heavy atoms in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def brics_fragments(smiles: str, min_atoms: int, max_atoms: int) -> list[str]:
    """BRICS fragments of one compound whose heavy-atom count is within the limits."""
    kept: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return kept

    try:
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
        for frag in frags:
            clean_frag = frag
            for i in range(20):
                clean_frag = clean_frag.replace(f"[{i}*]", "[*]")

            n_atoms = count_heavy_atoms(clean_frag)
            if min_atoms <= n_atoms <= max_atoms:
                kept.append(frag)
    except Exception:
        pass
    return kept


def enumerate_brics_fragments(
    jcp_ids: list[str],
    smiles_list: list[str],
    min_atoms: int = BRICS_MIN_ATOMS,
    max_atoms: int = BRICS_MAX_ATOMS,
    workers: int = 1,
) -> dict[str, list[str]]:
    """Enumerate BRICS fragments for compounds.

    Returns:
        Dictionary mapping fragment SMILES to list of JCP2022 IDs
    """
    logger.info("Enumerating BRICS fragments...")
    fragment_to_compounds: dict[str, list[str]] = {}

    compound_fragments = Parallel(n_jobs=workers)(
        delayed(brics_fragments)(smiles, min_atoms, max_atoms)
        for smiles in tqdm(smiles_list, desc="BRICS decomposition")
    )
    for jcp, frags in zip(jcp_ids, compound_fragments):
        for frag in frags:
            fragment_to_compounds.setdefault(frag, []).append(jcp)

    logger.info(f"BRICS fragments: {len(fragment_to_compounds):,} unique fragments")
    return fragment_to_compounds


def cluster_profiles(adata, n_clusters: int = 80) -> np.ndarray:
    """Cluster profiles using KMeans on scanpy PCA components."""
    import scanpy as sc
    from sklearn.cluster import MiniBatchKMeans

    logger.info(f"Computing PCA + KMeans clustering (k={n_clusters})...")

    adata_work = adata.copy()
    sc.pp.pca(adata_work, n_comps=50)
    X_pca = adata_work.obsm["X_pca"]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, batch_size=2048)
    labels = kmeans.fit_predict(X_pca)

    logger.info(f"KMeans clustering: {len(np.unique(labels))} clusters")
    return labels


def compute_enrichment(
    cluster_labels: np.ndarray,
    jcp_ids: np.ndarray,
    substructure_map: dict[str, list[str]],
    substructure_type: str,
    min_occurrences: int = 5,
) -> pd.DataFrame:
    """Compute substructure enrichment per cluster using Fisher's exact test.

    For each cluster, tests whether each substructure is overrepresented
    compared to background frequency.
    """
    n_total = len(jcp_ids)
    jcp_set = set(jcp_ids)
    clusters = np.unique(cluster_labels)

    # Pre-filter: only substructures with ≥min_occurrences compounds in dataset
    filtered_map = {}
    for sub, compounds in substructure_map.items():
        in_dataset = [c for c in compounds if c in jcp_set]
        if len(in_dataset) >= min_occurrences:
            filtered_map[sub] = in_dataset

    logger.info(
        f"Testing {len(filtered_map):,} {substructure_type}s "
        f"(≥{min_occurrences} compounds) across {len(clusters)} clusters"
    )

    results = []
    for cluster_id in tqdm(clusters, desc=f"Enrichment ({substructure_type})"):
        cluster_mask = cluster_labels == cluster_id
        n_cluster = int(cluster_mask.sum())
        cluster_jcps = set(jcp_ids[cluster_mask])

        for sub, compounds_in_dataset in filtered_map.items():
            compounds_set = set(compounds_in_dataset)
            n_sub_total = len(compounds_in_dataset)

            # 2x2 contingency table
            n_sub_in_cluster = len(compounds_set & cluster_jcps)
            n_sub_outside = n_sub_total - n_sub_in_cluster
            n_other_in_cluster = n_cluster - n_sub_in_cluster
            n_other_outside = n_total - n_cluster - n_sub_outside

            if n_sub_in_cluster == 0:
                continue

            table = [[n_sub_in_cluster, n_sub_outside],
                     [n_other_in_cluster, n_other_outside]]

            odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

            # Fold enrichment
            expected = n_sub_total * n_cluster / n_total
            fold_enrichment = n_sub_in_cluster / max(expected, 1e-10)

            results.append({
                "cluster": int(cluster_id),
                "substructure": sub,
                "substructure_type": substructure_type,
                "n_in_cluster": n_sub_in_cluster,
                "n_total": n_sub_total,
                "cluster_size": n_cluster,
                "fold_enrichment": fold_enrichment,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            })

    enrichment_df = pd.DataFrame(results)
    if len(enrichment_df) > 0:
        # Multiple testing correction (Benjamini-Hochberg)
        _, corrected_p, _, _ = multipletests(enrichment_df["p_value"], method="fdr_bh")
        enrichment_df["corrected_p_value"] = corrected_p
        enrichment_df = enrichment_df.sort_values("corrected_p_value").reset_index(drop=True)

        n_sig = (enrichment_df["corrected_p_value"] < 0.05).sum()
        logger.info(f"{substructure_type} enrichment: {n_sig} significant (FDR < 0.05)")

    return enrichment_df


# =============================================================================
# Visualization
# =============================================================================


def plot_coherence_distribution(
    coherence_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot distribution of within-scaffold cosine similarity vs null."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of observed vs null mean
    ax = axes[0]
    ax.hist(coherence_df["mean_cosine_sim"], bins=50, alpha=0.7, label="Observed (scaffolds)", color="steelblue")
    ax.hist(coherence_df["null_mean"], bins=50, alpha=0.5, label="Null (random groups)", color="gray")
    ax.set_xlabel("Mean pairwise cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Scaffold morphological coherence")
    ax.legend()

    # Right: p-value distribution
    ax = axes[1]
    ax.hist(coherence_df["corrected_p_value"], bins=50, alpha=0.7, color="steelblue")
    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.7, label="5% FDR")
    n_sig = int((coherence_df["corrected_p_value"] < 0.05).sum())
    ax.set_xlabel("Benjamini-Hochberg adjusted p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"Scaffolds significant at 5% FDR ({n_sig} / {len(coherence_df)})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "coherence_distribution.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'coherence_distribution.png'}")


def plot_top_scaffold_vignettes(
    coherence_df: pd.DataFrame,
    adata,
    scaffold_df: pd.DataFrame,
    n_top: int,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot vignettes for top N most coherent scaffolds."""
    jcp_to_idx = {k: i for i, k in enumerate(adata.obs["JCP2022"].values)}
    top = coherence_df.head(n_top)

    fig, axes = plt.subplots(n_top, 2, figsize=(14, 4 * n_top))
    if n_top == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(top.iterrows()):
        scaffold_smiles = row["scaffold"]
        members = scaffold_df[scaffold_df["scaffold"] == scaffold_smiles]["JCP2022"].tolist()
        member_idx = [jcp_to_idx[jcp] for jcp in members if jcp in jcp_to_idx]

        # Left: scaffold structure
        ax = axes[i, 0]
        mol = Chem.MolFromSmiles(scaffold_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
        ax.set_title(
            f"Scaffold (n={row['n_compounds']}, sim={row['mean_cosine_sim']:.3f}, "
            f"BH p={row['corrected_p_value']:.4f})",
            fontsize=9,
        )
        ax.axis("off")

        # Right: heatmap of member profiles
        ax = axes[i, 1]
        X_members = adata.X[member_idx]
        # Sort by first PC for visual coherence
        if X_members.shape[0] > 1:
            from sklearn.decomposition import PCA
            pc1 = PCA(n_components=1).fit_transform(X_members).ravel()
            sort_order = np.argsort(pc1)
            X_members = X_members[sort_order]

        ax.imshow(X_members, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_xlabel("Features")
        ax.set_ylabel("Compounds")
        ax.set_title("Morphological profiles", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "top_scaffolds_vignettes.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'top_scaffolds_vignettes.png'}")


def plot_cluster_umap(
    adata_umap,
    cluster_labels: np.ndarray,
    profile_jcp: np.ndarray,
    enrichment_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Plot UMAP colored by Leiden cluster with enriched scaffold annotations."""
    umap_jcp = adata_umap.obs["JCP2022"].values
    jcp_to_cluster = dict(zip(profile_jcp, cluster_labels))

    # Map clusters to UMAP compounds
    cluster_col = [str(jcp_to_cluster.get(jcp, -1)) for jcp in umap_jcp]
    adata_umap.obs["leiden_cluster"] = pd.Categorical(cluster_col)

    xlim, ylim = compute_umap_bounds(adata_umap)
    coords = adata_umap.obsm["X_umap"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all points as gray background
    ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.1, c="lightgray", rasterized=True)

    # Color by cluster (top clusters with enrichments only)
    if len(enrichment_df) > 0:
        sig_enrichments = enrichment_df[enrichment_df["corrected_p_value"] < 0.05]
        top_clusters = sig_enrichments["cluster"].unique()[:15]  # Limit for readability
        cmap = plt.cm.tab20
        for ci, cluster_id in enumerate(top_clusters):
            mask = np.array(cluster_col) == str(cluster_id)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=3, alpha=0.5, c=[cmap(ci % 20)],
                label=f"Cluster {cluster_id}", rasterized=True,
            )

        ax.legend(fontsize=7, markerscale=3, loc="upper right", ncol=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Leiden clusters with enriched substructures")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_dir / "cluster_umap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'cluster_umap.png'}")


# =============================================================================
# Cluster Feature Enrichment
# =============================================================================


def compute_cluster_feature_enrichment(
    adata,
    cluster_labels: np.ndarray,
    n_top_features: int = 20,
) -> pd.DataFrame:
    """Compute top distinguishing features for each cluster.

    For each cluster, finds features with the largest absolute deviation
    from the global mean (effect size = cluster_mean - global_mean).
    """
    X = adata.X
    feature_names = list(adata.var_names)
    global_mean = X.mean(axis=0)
    clusters = np.unique(cluster_labels)

    records = []
    for cluster_id in clusters:
        mask = cluster_labels == cluster_id
        cluster_mean = X[mask].mean(axis=0)
        effect_size = cluster_mean - global_mean

        # Top features by absolute effect size
        top_idx = np.argsort(np.abs(effect_size))[::-1][:n_top_features]
        for rank, idx in enumerate(top_idx):
            records.append({
                "cluster": int(cluster_id),
                "rank": rank + 1,
                "feature": feature_names[idx],
                "cluster_mean": float(cluster_mean[idx]),
                "global_mean": float(global_mean[idx]),
                "effect_size": float(effect_size[idx]),
            })

    df = pd.DataFrame(records)
    logger.info(f"Cluster feature enrichment: {len(clusters)} clusters × {n_top_features} features")
    return df


# =============================================================================
# HTML Browser
# =============================================================================


def generate_html_browser(
    coherence_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    cluster_features_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Generate self-contained HTML browser for scaffold ↔ cluster lookup."""
    # Build scaffold → enriched clusters mapping
    sig_enrichment = enrichment_df[enrichment_df["corrected_p_value"] < 0.05].copy()

    # Scaffold table data: join coherence with enriched clusters
    scaffold_records = []
    for _, row in coherence_df.iterrows():
        scaffold = row["scaffold"]
        matches = sig_enrichment[
            (sig_enrichment["substructure"] == scaffold)
            & (sig_enrichment["substructure_type"] == "scaffold")
        ]
        clusters = sorted(matches["cluster"].tolist())
        scaffold_records.append({
            "scaffold": scaffold,
            "n": int(row["n_compounds"]),
            "sim": round(float(row["mean_cosine_sim"]), 3),
            "z": round(float(row["z_score"]), 1),
            "p": float(row["p_value"]),
            "clusters": clusters,
        })

    # Cluster summary data
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_records = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        cl_enriched = sig_enrichment[sig_enrichment["cluster"] == cluster_id].sort_values(
            "fold_enrichment", ascending=False,
        )
        top_subs = []
        for _, erow in cl_enriched.head(20).iterrows():
            top_subs.append({
                "sub": erow["substructure"],
                "type": erow["substructure_type"],
                "n_in": int(erow["n_in_cluster"]),
                "n_tot": int(erow["n_total"]),
                "fold": round(float(erow["fold_enrichment"]), 1),
                "pval": f"{float(erow['corrected_p_value']):.2e}",
            })

        cl_feats = cluster_features_df[cluster_features_df["cluster"] == cluster_id].sort_values("rank")
        top_feats = []
        for _, frow in cl_feats.head(10).iterrows():
            top_feats.append({
                "feature": frow["feature"],
                "cp_feature": frow.get("cp_feature") if pd.notna(frow.get("cp_feature")) else None,
                "effect": round(float(frow["effect_size"]), 3),
                "cmean": round(float(frow["cluster_mean"]), 3),
            })

        cluster_records.append({
            "id": int(cluster_id),
            "size": int(cluster_sizes.get(cluster_id, 0)),
            "n_enriched": len(cl_enriched),
            "subs": top_subs,
            "feats": top_feats,
        })

    # Convert to JSON for embedding
    import json as json_mod
    scaffold_json = json_mod.dumps(scaffold_records)
    cluster_json = json_mod.dumps(cluster_records)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Substructure ↔ Morphology Browser</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.4em; }}
.tabs {{ display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 2px solid #ddd; padding-bottom: 0; }}
.tab {{ padding: 8px 16px; border: 1px solid #ddd; border-bottom: none; background: #f5f5f5; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 14px; }}
.tab.active {{ background: white; border-bottom: 2px solid white; margin-bottom: -2px; font-weight: 600; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
table.dataTable {{ font-size: 13px; }}
table.dataTable td {{ vertical-align: top; }}
.cluster-tag {{ display: inline-block; background: #e3f2fd; color: #1565c0; padding: 1px 6px; border-radius: 3px; margin: 1px; font-size: 11px; cursor: pointer; }}
.cluster-tag:hover {{ background: #1565c0; color: white; }}
.scaffold-smiles {{ font-family: monospace; font-size: 12px; word-break: break-all; max-width: 400px; }}
.cluster-detail {{ margin-top: 16px; }}
.cluster-detail h3 {{ margin-bottom: 8px; }}
.sub-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
.sub-table th, .sub-table td {{ padding: 4px 8px; border: 1px solid #ddd; text-align: left; }}
.sub-table th {{ background: #f5f5f5; }}
.sub-table .mono {{ font-family: monospace; font-size: 11px; word-break: break-all; }}
.feat-bar {{ display: inline-block; height: 12px; min-width: 2px; }}
.feat-pos {{ background: #c62828; }}
.feat-neg {{ background: #1565c0; }}
.note {{ color: #666; font-size: 13px; margin: 8px 0; }}
select {{ padding: 6px 10px; font-size: 14px; }}
.summary {{ background: #f8f9fa; padding: 12px; border-radius: 4px; margin-bottom: 16px; font-size: 13px; }}
</style>
</head>
<body>
<h1>Substructure &harr; Morphology Browser</h1>
<p class="note">JUMP Cell Painting &mdash; <code>compound_no_source7</code> &mdash; 80 KMeans clusters on 50 PCA components</p>

<div class="tabs">
<button class="tab active" onclick="switchTab('scaffolds')">Scaffold &rarr; Clusters</button>
<button class="tab" onclick="switchTab('clusters')">Cluster &rarr; Chemistry &amp; Features</button>
</div>

<div id="scaffolds" class="tab-content active">
<div class="summary">
Search scaffolds by SMILES substring. <strong>n</strong> = number of compounds with this scaffold.
<strong>sim</strong> = mean within-scaffold cosine similarity. <strong>z</strong> = z-score vs random baseline.
Clusters shown are those where the scaffold is significantly enriched (FDR &lt; 0.05).
<br>"Scaffold size" refers to the number of compounds sharing the scaffold, not the atom count of the scaffold itself.
</div>
<table id="scaffold-table" class="display" style="width:100%">
<thead><tr><th>Scaffold (SMILES)</th><th>n</th><th>sim</th><th>z-score</th><th>p-value</th><th>Enriched in clusters</th></tr></thead>
<tbody></tbody>
</table>
</div>

<div id="clusters" class="tab-content">
<div class="summary">
Select a cluster to see its enriched substructures and top distinguishing morphological features.
Feature <strong>effect size</strong> = cluster mean &minus; global mean. Positive (red) = elevated in cluster; negative (blue) = decreased.
The <strong>Column</strong> shows the X_N name used in the profile parquet; <strong>CellProfiler feature</strong> is the original measurement (mapping from <code>data/interim/feature_name_map.csv</code>).
</div>
<select id="cluster-select" onchange="showCluster(this.value)">
<option value="">Select a cluster...</option>
</select>
<div id="cluster-detail" class="cluster-detail"></div>
</div>

<script>
const scaffoldData = {scaffold_json};
const clusterData = {cluster_json};

// --- Tabs ---
function switchTab(tabId) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
}}

// --- Scaffold table ---
$(document).ready(function() {{
    const rows = scaffoldData.map(d => [
        '<span class="scaffold-smiles">' + d.scaffold + '</span>',
        d.n,
        d.sim,
        d.z,
        d.p < 0.001 ? d.p.toExponential(1) : d.p.toFixed(3),
        d.clusters.length > 0
            ? d.clusters.map(c => '<span class="cluster-tag" onclick="goToCluster(' + c + ')">' + c + '</span>').join(' ')
            : '<span style="color:#999">none</span>',
    ]);
    $('#scaffold-table').DataTable({{
        data: rows,
        pageLength: 25,
        order: [[3, 'desc']],
        columnDefs: [
            {{ targets: [1, 2, 3], className: 'dt-right' }},
            {{ targets: [0], width: '35%' }},
        ],
    }});

    // Populate cluster dropdown
    const sel = document.getElementById('cluster-select');
    clusterData.forEach(c => {{
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = 'Cluster ' + c.id + ' (' + c.size + ' compounds, ' + c.n_enriched + ' enriched substructures)';
        sel.appendChild(opt);
    }});
}});

function goToCluster(id) {{
    switchTab('clusters');
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab')[1].classList.add('active');
    document.getElementById('cluster-select').value = id;
    showCluster(id);
}}

function showCluster(id) {{
    const detail = document.getElementById('cluster-detail');
    if (id === '') {{ detail.innerHTML = ''; return; }}
    const c = clusterData.find(d => d.id == id);
    if (!c) {{ detail.innerHTML = '<p>Cluster not found</p>'; return; }}

    let html = '<h3>Cluster ' + c.id + ' &mdash; ' + c.size + ' compounds</h3>';

    // Features
    if (c.feats.length > 0) {{
        const maxEffect = Math.max(...c.feats.map(f => Math.abs(f.effect)));
        html += '<h4>Top distinguishing features</h4>';
        html += '<table class="sub-table"><thead><tr><th>Column</th><th>CellProfiler feature</th><th>Effect size</th><th style="width:120px"></th><th>Cluster mean</th></tr></thead><tbody>';
        c.feats.forEach(f => {{
            const width = Math.round(Math.abs(f.effect) / maxEffect * 100);
            const cls = f.effect > 0 ? 'feat-pos' : 'feat-neg';
            const cpName = f.cp_feature ? f.cp_feature : '<span style="color:#999">(unmapped)</span>';
            html += '<tr><td class="mono">' + f.feature + '</td>';
            html += '<td class="mono">' + cpName + '</td>';
            html += '<td style="text-align:right">' + f.effect.toFixed(3) + '</td>';
            html += '<td><span class="feat-bar ' + cls + '" style="width:' + width + 'px"></span></td>';
            html += '<td style="text-align:right">' + f.cmean.toFixed(3) + '</td></tr>';
        }});
        html += '</tbody></table>';
    }}

    // Enriched substructures
    if (c.subs.length > 0) {{
        html += '<h4>Enriched substructures (top 20)</h4>';
        html += '<table class="sub-table"><thead><tr><th>Substructure (SMILES)</th><th>Type</th><th>In cluster</th><th>Total</th><th>Fold enrichment</th><th>FDR p-value</th></tr></thead><tbody>';
        c.subs.forEach(s => {{
            html += '<tr><td class="mono">' + s.sub + '</td>';
            html += '<td>' + s.type + '</td>';
            html += '<td>' + s.n_in + '</td>';
            html += '<td>' + s.n_tot + '</td>';
            html += '<td>' + s.fold + '&times;</td>';
            html += '<td>' + s.pval + '</td></tr>';
        }});
        html += '</tbody></table>';
    }} else {{
        html += '<p style="color:#999">No significantly enriched substructures in this cluster.</p>';
    }}

    detail.innerHTML = html;
}}
</script>
</body>
</html>"""

    output_path.write_text(html)
    logger.info(f"Saved HTML browser: {output_path}")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    dataset: str = typer.Option(
        "compound_no_source7",
        "--dataset", "-d",
        help="Dataset name for morphology profiles",
    ),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    active_only: bool = typer.Option(
        False,
        "--active-only",
        help="Restrict to compounds significant for phenotypic activity at 10% FDR (BH)",
    ),
    copairs_db: Path = typer.Option(COPAIRS_DB, "--copairs-db"),
    activity_preprocessing: str = typer.Option("activity_no_target2", "--activity-preprocessing"),
    activity_filter: str = typer.Option("all_sources", "--activity-filter"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="DPI for saved figures"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip generating plots"),
    min_scaffold_compounds: int = typer.Option(
        MIN_SCAFFOLD_COMPOUNDS, "--min-scaffold-compounds",
        help="Minimum compounds per scaffold",
    ),
    n_permutations: int = typer.Option(
        N_PERMUTATIONS, "--n-permutations",
        help="Number of permutations for null distribution",
    ),
    skip_brics: bool = typer.Option(False, "--skip-brics", help="Skip BRICS fragment enrichment"),
    skip_analysis1: bool = typer.Option(False, "--skip-analysis1", help="Skip Analysis 1 (scaffold coherence)"),
    workers: int = typer.Option(1, "--workers", help="Processes for the permutation and BRICS steps"),
) -> None:
    """Bidirectional substructure-morphology relationship analysis."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / f"{dataset}{'_active' if active_only else ''}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    rng = np.random.default_rng(RANDOM_STATE)

    # =========================================================================
    # Load data
    # =========================================================================

    logger.info("Loading perturbation-level profiles...")
    adata = load_profiles(dataset, level="perturbation")
    if active_only:
        copairs_con = duckdb.connect(str(copairs_db), read_only=True)
        try:
            activity = query_activity_results(
                copairs_con,
                dataset=dataset,
                preprocessing=activity_preprocessing,
                filter_name=activity_filter,
            )
        finally:
            copairs_con.close()
        active_jcp = set(activity.loc[activity["below_corrected_p"], "Metadata_JCP2022"])
        adata = adata[adata.obs["JCP2022"].isin(active_jcp)].copy()
        logger.info(
            f"Active-only restriction: {adata.n_obs:,} profiled compounds from "
            f"{len(active_jcp):,} significant activity results"
        )
    profile_jcp = adata.obs["JCP2022"].values
    logger.info(f"Profiles: {adata.n_obs:,} compounds, {adata.n_vars:,} features")

    logger.info("Loading scaffolds and SMILES from metadata DB...")
    con = duckdb.connect(str(METADATA_DB), read_only=True)
    scaffold_df = load_scaffolds(con)
    smiles_df = load_smiles(con)
    con.close()

    # =========================================================================
    # Analysis 1: Substructure → Morphology
    # =========================================================================

    coherence_df = pd.DataFrame()
    if not skip_analysis1:
        logger.info("\n=== Analysis 1: Substructure → Morphology ===")
        coherence_df = compute_scaffold_coherence(
            adata, scaffold_df, min_scaffold_compounds, n_permutations, rng, workers,
        )

        # Save (drop centroid column for CSV - too large)
        coherence_csv = coherence_df.drop(columns=["centroid"])
        coherence_csv.to_csv(output_dir / "scaffold_coherence.csv", index=False)
        logger.info(f"Saved: {output_dir / 'scaffold_coherence.csv'} ({len(coherence_csv)} rows)")
    else:
        logger.info("Skipping Analysis 1 (scaffold coherence)")
        # Load from existing CSV if available (for HTML browser)
        coherence_path = output_dir / "scaffold_coherence.csv"
        if coherence_path.exists():
            coherence_df = pd.read_csv(coherence_path)
            logger.info(f"Loaded existing coherence: {len(coherence_df)} scaffolds")

    # =========================================================================
    # Analysis 2: Morphology → Substructure
    # =========================================================================

    logger.info("\n=== Analysis 2: Morphology → Substructure ===")

    # Cluster profiles
    cluster_labels = cluster_profiles(adata)

    # Build scaffold → compounds mapping
    profiled_scaffolds = scaffold_df[scaffold_df["JCP2022"].isin(set(profile_jcp))]
    scaffold_map: dict[str, list[str]] = {}
    for _, row in profiled_scaffolds.iterrows():
        scaffold_map.setdefault(row["scaffold"], []).append(row["JCP2022"])

    logger.info(f"Scaffold map: {len(scaffold_map):,} scaffolds covering {len(profiled_scaffolds):,} compounds")

    # Scaffold enrichment
    scaffold_enrichment = compute_enrichment(
        cluster_labels, profile_jcp, scaffold_map, "scaffold",
    )

    # BRICS fragment enrichment
    if not skip_brics:
        profiled_smiles = smiles_df[smiles_df["JCP2022"].isin(set(profile_jcp))]
        fragment_map = enumerate_brics_fragments(
            profiled_smiles["JCP2022"].tolist(),
            profiled_smiles["SMILES"].tolist(),
            workers=workers,
        )
        fragment_enrichment = compute_enrichment(
            cluster_labels, profile_jcp, fragment_map, "brics_fragment",
        )
        enrichment_df = pd.concat([scaffold_enrichment, fragment_enrichment], ignore_index=True)
    else:
        enrichment_df = scaffold_enrichment

    enrichment_df.to_csv(output_dir / "cluster_enrichment.csv", index=False)
    logger.info(f"Saved: {output_dir / 'cluster_enrichment.csv'} ({len(enrichment_df)} rows)")

    # Save cluster assignments
    cluster_assignments = pd.DataFrame({
        "JCP2022": profile_jcp,
        "cluster": cluster_labels,
    })
    cluster_assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    logger.info(f"Saved: {output_dir / 'cluster_assignments.csv'} ({len(cluster_assignments)} rows)")

    # =========================================================================
    # Cluster Feature Enrichment
    # =========================================================================

    logger.info("\n=== Cluster Feature Enrichment ===")
    cluster_features_df = compute_cluster_feature_enrichment(adata, cluster_labels)

    feature_map_path = INTERIM_DATA_DIR / "feature_name_map.csv"
    if feature_map_path.exists():
        feature_map = pd.read_csv(feature_map_path)
        cluster_features_df = cluster_features_df.merge(
            feature_map, left_on="feature", right_on="x_name", how="left",
        ).drop(columns=["x_name"])
        cluster_features_df = cluster_features_df[
            ["cluster", "rank", "feature", "cp_feature", "cluster_mean", "global_mean", "effect_size"]
        ]
    else:
        logger.warning(f"Feature name map not found at {feature_map_path}; cp_feature column will be omitted")

    cluster_features_df.to_csv(output_dir / "cluster_features.csv", index=False)
    logger.info(f"Saved: {output_dir / 'cluster_features.csv'} ({len(cluster_features_df)} rows)")

    # =========================================================================
    # HTML Browser
    # =========================================================================

    if len(coherence_df) > 0:
        coherence_for_html = coherence_df.drop(columns=["centroid"], errors="ignore")
        generate_html_browser(
            coherence_for_html, enrichment_df, cluster_features_df,
            cluster_labels, output_dir / "browser.html",
        )

    # =========================================================================
    # Plots
    # =========================================================================

    if not skip_plots:
        logger.info("\nGenerating plots...")

        if len(coherence_df) > 0:
            plot_coherence_distribution(coherence_df, output_dir, dpi)
            plot_top_scaffold_vignettes(
                coherence_df, adata, scaffold_df, N_TOP_SCAFFOLDS, output_dir, dpi,
            )

        # UMAP plot
        try:
            adata_umap = load_umap(dataset, level="perturbation")
            plot_cluster_umap(
                adata_umap, cluster_labels, profile_jcp, enrichment_df, output_dir, dpi,
            )
        except FileNotFoundError:
            logger.warning("UMAP h5ad not found, skipping cluster UMAP plot")

    # =========================================================================
    # Summary
    # =========================================================================

    n_scaffolds = len(coherence_df)
    n_sig_coherent_nominal = int((coherence_df["p_value"] < 0.05).sum()) if n_scaffolds > 0 else 0
    n_sig_coherent_fdr = (
        int((coherence_df["corrected_p_value"] < 0.05).sum()) if n_scaffolds > 0 else 0
    )
    n_clusters = int(len(np.unique(cluster_labels)))

    scaffold_sig = enrichment_df[
        (enrichment_df["substructure_type"] == "scaffold")
        & (enrichment_df["corrected_p_value"] < 0.05)
    ] if len(enrichment_df) > 0 else pd.DataFrame()

    summary = {
        "task": "Bidirectional substructure-morphology relationship analysis",
        "dataset": dataset,
        "active_only": active_only,
        "n_compounds_profiled": int(adata.n_obs),
        "analysis_1_scaffold_to_morphology": {
            "n_scaffolds_tested": n_scaffolds,
            "min_compounds_per_scaffold": min_scaffold_compounds,
            "n_significant_nominal_p005": n_sig_coherent_nominal,
            "pct_significant_nominal": round(
                n_sig_coherent_nominal / max(n_scaffolds, 1) * 100, 1
            ),
            "n_significant_fdr005": n_sig_coherent_fdr,
            "pct_significant_fdr005": round(
                n_sig_coherent_fdr / max(n_scaffolds, 1) * 100, 1
            ),
            "multiple_testing": "Benjamini-Hochberg",
            "n_permutations": n_permutations,
            "median_cosine_sim": float(coherence_df["mean_cosine_sim"].median()) if n_scaffolds > 0 else None,
            "median_null_sim": float(coherence_df["null_mean"].median()) if n_scaffolds > 0 else None,
        },
        "analysis_2_morphology_to_substructure": {
            "n_clusters": n_clusters,
            "n_scaffold_enrichments_fdr005": len(scaffold_sig),
            "n_clusters_with_enriched_scaffold": int(scaffold_sig["cluster"].nunique()) if len(scaffold_sig) > 0 else 0,
            "n_enrichment_tests_total": len(enrichment_df),
        },
    }

    if not skip_brics and len(enrichment_df) > 0:
        frag_sig = enrichment_df[
            (enrichment_df["substructure_type"] == "brics_fragment")
            & (enrichment_df["corrected_p_value"] < 0.05)
        ]
        summary["analysis_2_morphology_to_substructure"]["n_brics_enrichments_fdr005"] = len(frag_sig)

    write_summary(
        output_dir / "summary.json",
        description=(
            "Bidirectional substructure-morphology analysis on JUMP Cell Painting profiles. "
            "Analysis 1 tests whether each Murcko scaffold has a consistent morphological "
            "fingerprint (within-scaffold cosine similarity vs a permuted random baseline, "
            "with Benjamini-Hochberg correction across scaffolds); "
            "Analysis 2 clusters profiles and tests, via Fisher's exact test, whether any "
            "scaffold or BRICS fragment is overrepresented within a morphological cluster. "
            "Produces coherence and enrichment tables, cluster assignments and features, "
            "diagnostic figures, and a self-contained HTML browser for scaffold-cluster lookup."
        ),
        numbers=summary,
        files=index_files(output_dir),
    )
    logger.info(f"Saved: {output_dir / 'summary.json'}")

    # Log conclusions
    logger.info("\n=== Summary ===")
    logger.info(
        f"Analysis 1: {n_sig_coherent_fdr}/{n_scaffolds} scaffolds have significant "
        "morphological coherence at 5% FDR (BH)"
    )
    logger.info(f"Analysis 2: {n_clusters} Leiden clusters, {len(scaffold_sig)} enriched scaffolds (FDR < 0.05)")

    logger.success(f"\nSubstructure-morphology analysis complete: {output_dir}")


if __name__ == "__main__":
    app()
