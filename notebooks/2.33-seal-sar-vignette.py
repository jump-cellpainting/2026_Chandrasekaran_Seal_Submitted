"""SAR Vignette - Structural Alert Discovery using BRICS Decomposition.

Discovers structural alerts (substructure patterns) that predict phenotypic
activity in Cell Painting using RDKit's BRICS fragment enumeration.

Research Question: What molecular substructures are associated with
phenotypically active compounds in Cell Painting?

Algorithm (SARpy-inspired):
1. Fragment enumeration: Generate BRICS fragments for all compounds
2. Evaluate each fragment: Compute PPV (precision) for predicting activity
3. Filter alerts: Keep fragments with PPV > threshold, min occurrences >= 5
4. Greedy selection: Build ruleset by iteratively selecting highest-PPV fragments
5. Post-filter: Remove small fragments and common substructures

Usage:
    # Default run
    pixi run -e cheminformatics python notebooks/2.33-seal-sar-vignette.py

    # Stricter PPV threshold
    pixi run -e cheminformatics python notebooks/2.33-seal-sar-vignette.py --min-ppv 0.7

    # Different p-value threshold for activity
    pixi run -e cheminformatics python notebooks/2.33-seal-sar-vignette.py --p-threshold 0.05

    # Different dataset and preprocessing
    pixi run -e cheminformatics python notebooks/2.33-seal-sar-vignette.py --dataset compound_DL_CPCNN_no_source7

Outputs: data/processed/sar-vignette/{dataset}/{preprocessing}/
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from tqdm import tqdm

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary

# =============================================================================
# Configuration
# =============================================================================

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
COPAIRS_DB = PROCESSED_DATA_DIR / "copairs_results.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "sar-vignette"

# Default thresholds
DEFAULT_P_THRESHOLD = 0.10
DEFAULT_MIN_ATOMS = 2
DEFAULT_MAX_ATOMS = 18
DEFAULT_MIN_HITS = 5
DEFAULT_MIN_PPV = 0.5

# Common substructures to filter out (SMARTS patterns)
COMMON_SUBSTRUCTURES = [
    "c1ccccc1",  # benzene
    "C1CCCCC1",  # cyclohexane
    "[CH3]",  # methyl
    "[OH]",  # hydroxyl
]

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_activity_labels(
    con_copairs: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
    p_threshold: float = DEFAULT_P_THRESHOLD,
) -> pd.DataFrame:
    """Load activity labels from copairs results using shared query utility."""
    df = query_activity_results(
        con_copairs,
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
    )

    # Add is_active label based on p_threshold
    df["is_active"] = (df["corrected_p_value"] < p_threshold).astype(int)

    n_active = df["is_active"].sum()
    pct_active = n_active / len(df) * 100
    logger.info(f"Loaded activity: {len(df):,} compounds, {n_active:,} active ({pct_active:.1f}%)")
    return df


def load_smiles(con_metadata: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load SMILES from metadata database."""
    df = con_metadata.execute(
        """
        SELECT
            Metadata_JCP2022 as JCP2022,
            Metadata_SMILES as SMILES
        FROM compound
        WHERE Metadata_SMILES IS NOT NULL
    """
    ).df()
    logger.info(f"Loaded SMILES: {len(df):,} compounds")
    return df


def prepare_dataset(
    activity_df: pd.DataFrame,
    smiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join activity labels with SMILES and create RDKit molecules."""
    df = activity_df.merge(smiles_df, left_on="Metadata_JCP2022", right_on="JCP2022", how="inner")
    logger.info(f"Joined dataset: {len(df):,} compounds")

    # Create RDKit molecule objects
    logger.info("Creating RDKit molecule objects...")
    mols = []
    valid_mask = []
    for smiles in tqdm(df["SMILES"], desc="Parsing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_mask.append(True)
        else:
            mols.append(None)
            valid_mask.append(False)

    df["mol"] = mols
    df = df[valid_mask].reset_index(drop=True)
    logger.info(f"Valid molecules: {len(df):,}")

    return df


# =============================================================================
# Fragment Enumeration
# =============================================================================


def count_heavy_atoms(smiles: str) -> int:
    """Count heavy atoms in a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def enumerate_brics_fragments(
    dataset: pd.DataFrame,
    min_atoms: int = DEFAULT_MIN_ATOMS,
    max_atoms: int = DEFAULT_MAX_ATOMS,
) -> dict[str, list[str]]:
    """Enumerate BRICS fragments for all compounds.

    Returns:
        Dictionary mapping fragment SMILES to list of JCP2022 IDs
    """
    logger.info("Enumerating BRICS fragments...")
    fragment_to_compounds: dict[str, list[str]] = {}

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="BRICS decomposition"):
        mol = row["mol"]
        jcp = row["Metadata_JCP2022"]

        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
            for frag in frags:
                # Clean up BRICS dummy atoms for counting
                clean_frag = frag
                for i in range(20):  # Remove dummy atom markers [1*] through [19*]
                    clean_frag = clean_frag.replace(f"[{i}*]", "[*]")

                n_atoms = count_heavy_atoms(clean_frag)
                if min_atoms <= n_atoms <= max_atoms:
                    if frag not in fragment_to_compounds:
                        fragment_to_compounds[frag] = []
                    fragment_to_compounds[frag].append(jcp)
        except Exception:
            continue  # Skip problematic molecules

    logger.info(f"Found {len(fragment_to_compounds):,} unique fragments")
    return fragment_to_compounds


# =============================================================================
# Alert Evaluation
# =============================================================================


def evaluate_fragments(
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
    min_hits: int = DEFAULT_MIN_HITS,
) -> pd.DataFrame:
    """Evaluate PPV for each fragment.

    Returns DataFrame with columns: smiles, n_hits, tp, fp, ppv
    """
    logger.info("Evaluating fragment PPV...")

    # Create JCP to activity mapping
    jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    results = []
    for frag_smiles, compound_list in tqdm(fragment_to_compounds.items(), desc="Evaluating fragments"):
        n_hits = len(compound_list)
        if n_hits < min_hits:
            continue

        tp = sum(jcp_to_active.get(jcp, 0) for jcp in compound_list)
        fp = n_hits - tp

        ppv = tp / n_hits if n_hits > 0 else 0

        results.append(
            {
                "smiles": frag_smiles,
                "n_hits": n_hits,
                "tp": tp,
                "fp": fp,
                "ppv": ppv,
            }
        )

    df = pd.DataFrame(results)
    logger.info(f"Evaluated {len(df):,} fragments with >= {min_hits} hits")
    return df


def filter_common_substructures(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Remove common/trivial substructures from alerts."""
    logger.info("Filtering common substructures...")

    # Parse common substructures
    common_patterns = []
    for smarts in COMMON_SUBSTRUCTURES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            common_patterns.append(pattern)

    def is_common(smiles: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True

        # Check exact match with common substructures
        for pattern in common_patterns:
            if mol.HasSubstructMatch(pattern) and mol.GetNumHeavyAtoms() <= pattern.GetNumHeavyAtoms() + 2:
                return True

        return False

    initial_count = len(alerts_df)
    alerts_df = alerts_df[~alerts_df["smiles"].apply(is_common)].copy()
    removed = initial_count - len(alerts_df)
    logger.info(f"Removed {removed} common substructures, {len(alerts_df)} remaining")

    return alerts_df


def filter_small_fragments(alerts_df: pd.DataFrame, min_atoms: int = 5) -> pd.DataFrame:
    """Remove fragments with too few atoms."""
    logger.info(f"Filtering fragments with < {min_atoms} heavy atoms...")

    initial_count = len(alerts_df)
    alerts_df = alerts_df[alerts_df["smiles"].apply(count_heavy_atoms) >= min_atoms].copy()
    removed = initial_count - len(alerts_df)
    logger.info(f"Removed {removed} small fragments, {len(alerts_df)} remaining")

    return alerts_df


# =============================================================================
# Greedy Rule Selection
# =============================================================================


def greedy_select_alerts(
    alerts_df: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
    min_ppv: float = DEFAULT_MIN_PPV,
    max_alerts: int = 100,
) -> pd.DataFrame:
    """Greedily select non-redundant alerts.

    Iteratively selects the highest-PPV alert and removes covered compounds.
    """
    logger.info(f"Greedy selection with min_ppv={min_ppv}...")

    # Filter by PPV threshold
    candidates = alerts_df[alerts_df["ppv"] >= min_ppv].copy()
    logger.info(f"Candidates with PPV >= {min_ppv}: {len(candidates)}")

    if len(candidates) == 0:
        logger.warning("No alerts meet the PPV threshold")
        return pd.DataFrame(columns=["smiles", "n_hits", "tp", "fp", "ppv", "rank"])

    # Sort by PPV descending, then by n_hits descending, then by SMILES for deterministic ties
    candidates = candidates.sort_values(["ppv", "n_hits", "smiles"], ascending=[False, False, True])

    # Greedy selection
    selected = []
    covered_compounds = set()
    jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    for _, row in candidates.iterrows():
        if len(selected) >= max_alerts:
            break

        frag_smiles = row["smiles"]
        compounds = fragment_to_compounds.get(frag_smiles, [])

        # Recompute PPV on uncovered compounds
        uncovered = [c for c in compounds if c not in covered_compounds]
        if len(uncovered) < 3:  # Skip if too few uncovered
            continue

        tp_new = sum(jcp_to_active.get(jcp, 0) for jcp in uncovered)
        ppv_new = tp_new / len(uncovered) if uncovered else 0

        if ppv_new >= min_ppv:
            selected.append(
                {
                    "smiles": frag_smiles,
                    "n_hits": len(uncovered),
                    "tp": tp_new,
                    "fp": len(uncovered) - tp_new,
                    "ppv": ppv_new,
                    "rank": len(selected) + 1,
                }
            )
            covered_compounds.update(uncovered)

    logger.info(f"Selected {len(selected)} structural alerts")
    return pd.DataFrame(selected)


# =============================================================================
# Visualization
# =============================================================================


def plot_ppv_distribution(alerts_df: pd.DataFrame, output_path: Path, dpi: int = DEFAULT_DPI):
    """Plot distribution of PPV values across all fragments."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(alerts_df["ppv"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(DEFAULT_MIN_PPV, color="red", linestyle="--", label=f"Threshold = {DEFAULT_MIN_PPV}")
    ax.set_xlabel("Positive Predictive Value (PPV)")
    ax.set_ylabel("Number of Fragments")
    ax.set_title("PPV Distribution of BRICS Fragments")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_top_alerts(selected_alerts: pd.DataFrame, output_path: Path, top_n: int = 20, dpi: int = DEFAULT_DPI):
    """Plot bar chart of top alerts by PPV."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot")
        return

    top = selected_alerts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top))
    bars = ax.barh(y_pos, top["ppv"], color="steelblue", edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Alert {i + 1}" for i in range(len(top))])
    ax.invert_yaxis()
    ax.set_xlabel("Positive Predictive Value (PPV)")
    ax.set_title(f"Top {len(top)} Structural Alerts by PPV")
    ax.axvline(DEFAULT_MIN_PPV, color="red", linestyle="--", alpha=0.7)

    # Add hit counts as labels
    for bar, hits in zip(bars, top["n_hits"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={hits}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_alert_structures(selected_alerts: pd.DataFrame, output_path: Path, top_n: int = 20, dpi: int = DEFAULT_DPI):
    """Plot grid of top alert structures."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot")
        return

    top = selected_alerts.head(top_n)

    mols = []
    legends = []
    for _, row in top.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            mols.append(mol)
            legends.append(f"PPV={row['ppv']:.2f}\nn={row['n_hits']}")

    if not mols:
        logger.warning("No valid molecules to draw")
        return

    # Calculate grid dimensions
    n_cols = min(5, len(mols))
    (len(mols) + n_cols - 1) // n_cols

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(250, 200),
        legends=legends,
        useSVG=False,
    )

    # Save the image
    img.save(output_path)
    logger.info(f"Saved: {output_path}")


def plot_coverage(
    selected_alerts: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    dataset: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
):
    """Plot cumulative coverage of active compounds by alerts."""
    if len(selected_alerts) == 0:
        logger.warning("No alerts for coverage plot")
        return

    total_active = dataset["is_active"].sum()
    jcp_to_active = dict(zip(dataset["Metadata_JCP2022"], dataset["is_active"]))

    covered = set()
    coverage = []

    for _, row in selected_alerts.iterrows():
        compounds = fragment_to_compounds.get(row["smiles"], [])
        active_compounds = {c for c in compounds if jcp_to_active.get(c, 0) == 1}
        covered.update(active_compounds)
        coverage.append(len(covered) / total_active * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(coverage) + 1), coverage, marker="o", markersize=3)
    ax.set_xlabel("Number of Alerts")
    ax.set_ylabel("Coverage of Active Compounds (%)")
    ax.set_title("Cumulative Coverage of Active Compounds")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_alerts_on_structure_umap(
    selected_alerts: pd.DataFrame,
    fragment_to_compounds: dict[str, list[str]],
    output_path: Path,
    top_n: int = 10,
    dpi: int = DEFAULT_DPI,
):
    """Plot top alerts highlighted on structure UMAP.

    Creates a single UMAP plot with all compounds containing any of the
    top structural alerts highlighted in red. Excludes outliers for better visualization.
    """
    import scanpy as sc

    structure_umap_path = PROCESSED_DATA_DIR / "chemical-space" / "all" / "structure_umap.h5ad"
    outliers_path = PROCESSED_DATA_DIR / "chemical-space" / "all" / "structure_umap_outliers.csv"

    if not structure_umap_path.exists():
        logger.warning(f"Structure UMAP not found at {structure_umap_path}")
        return

    if len(selected_alerts) == 0:
        logger.warning("No alerts to plot on UMAP")
        return

    logger.info("Loading structure UMAP...")
    adata = sc.read_h5ad(structure_umap_path)
    umap_coords = adata.obsm["X_umap"]
    jcp_ids = adata.obs["Metadata_JCP2022"].values

    # Load and exclude outliers for better visualization
    outlier_jcps = set()
    if outliers_path.exists():
        outliers_df = pd.read_csv(outliers_path)
        outlier_jcps = set(outliers_df["Metadata_JCP2022"])
        logger.info(f"Excluding {len(outlier_jcps)} outliers")

    # Filter to non-outliers
    non_outlier_mask = np.array([jcp not in outlier_jcps for jcp in jcp_ids])
    umap_filtered = umap_coords[non_outlier_mask]
    jcp_filtered = jcp_ids[non_outlier_mask]
    jcp_to_idx = {jcp: i for i, jcp in enumerate(jcp_filtered)}

    # Collect all compounds from top alerts (excluding outliers)
    top = selected_alerts.head(top_n)
    all_alert_compounds = set()
    for _, row in top.iterrows():
        compounds = fragment_to_compounds.get(row["smiles"], [])
        all_alert_compounds.update(compounds)
    all_alert_compounds -= outlier_jcps

    highlight_idx = [jcp_to_idx[jcp] for jcp in all_alert_compounds if jcp in jcp_to_idx]
    logger.info(f"Highlighting {len(highlight_idx)} compounds from top {len(top)} alerts")

    # Create single UMAP plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all points in gray
    ax.scatter(
        umap_filtered[:, 0],
        umap_filtered[:, 1],
        c="lightgray",
        s=1,
        alpha=0.5,
        rasterized=True,
        label=f"All compounds (n={len(umap_filtered):,})",
    )

    # Highlight compounds with alerts
    if highlight_idx:
        ax.scatter(
            umap_filtered[highlight_idx, 0],
            umap_filtered[highlight_idx, 1],
            c="red",
            s=20,
            alpha=0.9,
            edgecolors="darkred",
            linewidths=0.5,
            label=f"Alert compounds (n={len(highlight_idx)})",
        )

    # Set axis limits from filtered data with padding
    x_min, x_max = umap_filtered[:, 0].min(), umap_filtered[:, 0].max()
    y_min, y_max = umap_filtered[:, 1].min(), umap_filtered[:, 1].max()
    x_pad = (x_max - x_min) * 0.03
    y_pad = (y_max - y_min) * 0.03
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"Structural Alerts on Structure UMAP\n(Top {len(top)} alerts, {len(highlight_idx)} compounds)", fontsize=14
    )
    ax.legend(loc="upper right", markerscale=1.5, fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


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
    p_threshold: float = typer.Option(DEFAULT_P_THRESHOLD, "--p-threshold", help="P-value threshold for activity"),
    min_atoms: int = typer.Option(DEFAULT_MIN_ATOMS, "--min-atoms", help="Minimum fragment size"),
    max_atoms: int = typer.Option(DEFAULT_MAX_ATOMS, "--max-atoms", help="Maximum fragment size"),
    min_hits: int = typer.Option(DEFAULT_MIN_HITS, "--min-hits", help="Minimum compound hits for a fragment"),
    min_ppv: float = typer.Option(DEFAULT_MIN_PPV, "--min-ppv", help="Minimum PPV for structural alerts"),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip plot generation"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="Plot DPI"),
):
    """Discover structural alerts predicting phenotypic activity."""
    if output_dir is None:
        output_dir = OUTPUT_DIR / dataset / preprocessing
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading data...")
    con_copairs = duckdb.connect(str(COPAIRS_DB), read_only=True)
    con_metadata = duckdb.connect(str(METADATA_DB), read_only=True)

    activity_df = load_activity_labels(
        con_copairs,
        dataset=dataset,
        preprocessing=preprocessing,
        filter_name="all_sources",
        activity_params="default",
        p_threshold=p_threshold,
    )
    smiles_df = load_smiles(con_metadata)
    compound_df = prepare_dataset(activity_df, smiles_df)

    con_copairs.close()
    con_metadata.close()

    # Fragment enumeration
    fragment_to_compounds = enumerate_brics_fragments(compound_df, min_atoms, max_atoms)

    # Evaluate fragments
    alerts_df = evaluate_fragments(fragment_to_compounds, compound_df, min_hits)

    # Filter
    alerts_df = filter_small_fragments(alerts_df, min_atoms=5)
    alerts_df = filter_common_substructures(alerts_df)

    # Greedy selection
    selected_alerts = greedy_select_alerts(alerts_df, fragment_to_compounds, compound_df, min_ppv)

    # Calculate coverage
    if len(selected_alerts) > 0:
        total_active = compound_df["is_active"].sum()
        jcp_to_active = dict(zip(compound_df["Metadata_JCP2022"], compound_df["is_active"]))
        covered = set()
        for _, row in selected_alerts.iterrows():
            compounds = fragment_to_compounds.get(row["smiles"], [])
            active_compounds = {c for c in compounds if jcp_to_active.get(c, 0) == 1}
            covered.update(active_compounds)
        coverage_pct = len(covered) / total_active * 100
    else:
        coverage_pct = 0

    # Save results
    logger.info("Saving results...")
    alerts_df.to_csv(output_dir / "all_fragments.csv", index=False)
    selected_alerts.to_csv(output_dir / "structural_alerts.csv", index=False)

    # Summary numbers
    numbers = {
        "task": "SAR Vignette - Structural Alert Discovery",
        "method": "BRICS decomposition",
        "parameters": {
            "p_threshold": p_threshold,
            "min_atoms": min_atoms,
            "max_atoms": max_atoms,
            "min_hits": min_hits,
            "min_ppv": min_ppv,
        },
        "dataset_statistics": {
            "n_compounds": len(compound_df),
            "n_active": int(compound_df["is_active"].sum()),
            "pct_active": float(compound_df["is_active"].mean() * 100),
        },
        "fragment_statistics": {
            "n_unique_fragments": len(fragment_to_compounds),
            "n_fragments_evaluated": len(alerts_df),
        },
        "results": {
            "n_structural_alerts": len(selected_alerts),
            "coverage_pct": coverage_pct,
        },
    }

    if len(selected_alerts) > 0:
        numbers["top_alerts"] = selected_alerts.head(10).to_dict(orient="records")

    # Visualizations
    if not skip_plots:
        logger.info("Generating visualizations...")
        plot_ppv_distribution(alerts_df, output_dir / "ppv_distribution.png", dpi)
        plot_top_alerts(selected_alerts, output_dir / "top_alerts.png", dpi=dpi)
        plot_alert_structures(selected_alerts, output_dir / "alert_structures.png", dpi=dpi)
        plot_coverage(selected_alerts, fragment_to_compounds, compound_df, output_dir / "coverage.png", dpi)
        plot_alerts_on_structure_umap(
            selected_alerts, fragment_to_compounds, output_dir / "alerts_structure_umap.png", top_n=10, dpi=dpi
        )

    # Write standardized summary envelope (after plots so figures are indexed)
    description = (
        "SAR vignette that discovers structural alerts (BRICS substructure patterns) predicting "
        "phenotypic Cell Painting activity. Enumerates BRICS fragments across compounds, scores each by "
        "positive predictive value for activity, greedily selects a non-redundant ruleset, and writes the "
        "alert tables plus PPV, top-alert, coverage, structure, and UMAP figures."
    )
    write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=numbers,
        files=index_files(output_dir),
    )
    logger.info(f"Saved: {output_dir / 'summary.json'}")

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Compounds: {len(compound_df):,}, {compound_df['is_active'].sum():,} active ({compound_df['is_active'].mean() * 100:.1f}%)"
    )
    logger.info(f"Unique BRICS fragments: {len(fragment_to_compounds):,}")
    logger.info(f"Fragments evaluated (>= {min_hits} hits): {len(alerts_df):,}")
    logger.info(f"Structural alerts (PPV >= {min_ppv}): {len(selected_alerts)}")
    logger.info(f"Coverage of active compounds: {coverage_pct:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(app())
