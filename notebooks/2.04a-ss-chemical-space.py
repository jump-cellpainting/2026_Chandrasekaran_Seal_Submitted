"""Chemical space characterization for JUMP Cell Painting dataset.

Analyzes the chemical properties and diversity of the JUMP compound collection:
1. Annotation coverage (Drug Repurposing Hub, ChEMBL, Chemical Probes, MOTIVE)
2. Physicochemical properties (MW, logP, TPSA, etc.)
3. Structure UMAP (Morgan fingerprints, cosine ≈ Tanimoto)
4. Property UMAP (z-scored descriptors, Euclidean)
5. Scaffold diversity (Murcko scaffolds)
6. PAINS / structural alerts
7. Property space coverage

Properties are precomputed by:
    pixi run -e cheminformatics python -m jump_production.processing.compound_properties

Structure UMAP uses L2 normalization + cosine similarity as a proxy for
Tanimoto/Jaccard (achieves 99.8% rank correlation, see issue #26).
This enables GPU acceleration via rapids_singlecell.

Usage:
    pixi run python notebooks/2.04a-ss-chemical-space.py
    pixi run python notebooks/2.04a-ss-chemical-space.py --skip-umap  # faster, skip structure UMAP
    pixi run -e rapids python notebooks/2.04a-ss-chemical-space.py    # with GPU acceleration

Outputs: data/processed/chemical-space/{all,<dataset>}/
"""

import json
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler

from jump_production.config import DEFAULT_DPI
from jump_production.gpu import has_gpu
from jump_production.summaries import index_files, write_legend, write_summary
from jump_production.visualization import (
    PROPERTY_FIGURE_COLOR,
    attach_caption,
    compute_umap_bounds,
    property_figure_caption_layout,
)

HAS_GPU = has_gpu()

logger.info(f"GPU acceleration: {'enabled' if HAS_GPU else 'disabled (cupy not available)'}")

# Configuration
METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")
COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
MORGAN_FP_FILE = Path("data/interim/compound_featurization/morgan_fp.npz")
OUTPUT_DIR = Path("data/processed/chemical-space")
ANALYSIS_CONFIG = Path("configs/snakemake/analysis.yaml")
DATASET_STRUCTURE_SUMMARY = Path("data/processed/dataset_structure/summary.json")

app = typer.Typer()

FIGURE1_ID = "figure_compound_collection"
SUPPLEMENT_ID = "figure_suppl_chemical_properties"
PUBLICATION_FIGURES = {FIGURE1_ID, SUPPLEMENT_ID}
PUBLICATION_DENSITY_CMAP = "Blues"
PUBLICATION_PROPERTY_HEXBIN_GRIDSIZE = 55
PUBLICATION_UMAP_HEXBIN_GRIDSIZE = 110
PUBLICATION_HISTOGRAM_BINS = 40
PUBLICATION_SCAFFOLD_EXPORT_N = 30
PUBLICATION_SCAFFOLD_DISPLAY_N = 12
PUBLICATION_UMAP_IQR_K = 3.0
PUBLICATION_CLINICAL_PHASE_ORDER = ("Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched")
PUBLICATION_PARTNER_SITE_COUNT = 12
# The schematic canvas is much wider than a plotted figure, so its caption type is scaled to wrap
# to the same line length the plotted figures use rather than running the full width at 7pt.
SCHEMATIC_CAPTION_CHARACTERS = 132
PUBLICATION_CPG0012_COMPOUNDS = 30_616
PUBLICATION_LIPINSKI_MW_CUTOFF_DA = 500.0
PUBLICATION_LIPINSKI_LOGP_CUTOFF = 5.0
PUBLICATION_PROPERTY_SPACE_BOUNDS = {
    "Metadata_MW": (0.0, 800.0),
    "Metadata_LogP": (-5.0, 10.0),
}
PUBLICATION_PROPERTY_DISTRIBUTION_BOUNDS = {
    "Metadata_MW": (100.0, 700.0),
    "Metadata_LogP": (-3.0, 8.0),
    "Metadata_TPSA": (0.0, 200.0),
    "Metadata_QED": (0.0, 1.0),
}


def resolve_analysis_output_dir(output_dir: Path | None, analyzed_dataset: str | None) -> Path:
    """Return a population-scoped default path while preserving explicit output paths."""
    if output_dir is None:
        return OUTPUT_DIR / (analyzed_dataset or "all")
    if analyzed_dataset is not None:
        return output_dir / analyzed_dataset
    return output_dir


# =============================================================================
# Data Loading
# =============================================================================


def get_canonical_key() -> dict:
    """Canonical analyzed slice (dataset + preprocessing) from the Snakemake config."""
    import yaml

    cfg = yaml.safe_load(open(ANALYSIS_CONFIG))["analysis"]
    return {"dataset": cfg["default_dataset"], "preprocessing": cfg["default_preprocessing"]}


def get_analyzed_compound_ids(dataset: str, preprocessing: str) -> list[str]:
    """Get compound IDs for the canonical analyzed slice (dataset AND preprocessing).

    Both filters matter: one dataset can carry several preprocessing runs, so filtering on
    _dataset alone unions them and over-counts (the 132-compound unfiltered-union artifact).
    """
    copairs_con = duckdb.connect(str(COPAIRS_DB), read_only=True)
    try:
        ids = copairs_con.execute(
            "SELECT DISTINCT Metadata_JCP2022 FROM activity_results WHERE _dataset = ? AND _preprocessing = ?",
            [dataset, preprocessing],
        ).fetchall()
        return [r[0] for r in ids]
    finally:
        copairs_con.close()


def load_compounds(
    con: duckdb.DuckDBPyConnection,
    analyzed_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load compound metadata with precomputed properties from DuckDB.

    Returns DataFrame with Metadata_JCP2022, SMILES, annotations, and physicochemical properties.
    Properties are precomputed by compound_properties.py and stored in compound_properties table.
    All columns are now available in the compound_metadata view.

    If analyzed_ids is provided, filters to only those compounds.
    """
    if analyzed_ids is not None:
        con.execute(
            "CREATE OR REPLACE TEMPORARY TABLE _analyzed AS SELECT UNNEST(?) as jcp",
            [analyzed_ids],
        )
        query = """
        SELECT cm.*, tx.has_toxcast AS Metadata_toxcast_annotated
        FROM compound_metadata cm
        JOIN _analyzed a ON cm.Metadata_JCP2022 = a.jcp
        LEFT JOIN (SELECT DISTINCT Metadata_JCP2022, TRUE AS has_toxcast FROM toxcast_annotations) tx
            ON cm.Metadata_JCP2022 = tx.Metadata_JCP2022
        WHERE cm.Metadata_SMILES IS NOT NULL
        """
    else:
        query = """
        SELECT cm.*, tx.has_toxcast AS Metadata_toxcast_annotated
        FROM compound_metadata cm
        LEFT JOIN (SELECT DISTINCT Metadata_JCP2022, TRUE AS has_toxcast FROM toxcast_annotations) tx
            ON cm.Metadata_JCP2022 = tx.Metadata_JCP2022
        WHERE cm.Metadata_SMILES IS NOT NULL
        """
    df = con.execute(query).fetchdf()
    logger.info(f"Loaded {len(df):,} compounds with SMILES and properties")

    # Log property coverage
    n_valid = df["Metadata_ValidMol"].sum() if "Metadata_ValidMol" in df.columns else 0
    logger.info(f"Valid molecules (RDKit): {n_valid:,} ({100 * n_valid / len(df):.1f}%)")

    return df


def validate_and_order_publication_compounds(
    df: pd.DataFrame,
    canonical_ids: list[str],
) -> pd.DataFrame:
    """Validate one metadata row per canonical compound and return canonical ID order."""
    canonical = pd.Index(canonical_ids, dtype="object")
    if canonical.hasnans:
        raise ValueError("Canonical analyzed compound IDs contain null values")
    if canonical.duplicated().any():
        duplicates = canonical[canonical.duplicated()].unique().tolist()
        raise ValueError(f"Canonical analyzed compound IDs are not unique: {duplicates[:5]}")
    if "Metadata_JCP2022" not in df.columns:
        raise ValueError("Compound metadata is missing Metadata_JCP2022")

    observed = pd.Index(df["Metadata_JCP2022"], dtype="object")
    if observed.hasnans:
        raise ValueError("Compound metadata contains null JCP2022 IDs")
    if observed.duplicated().any():
        duplicates = observed[observed.duplicated()].unique().tolist()
        raise ValueError(f"Compound metadata has duplicate JCP2022 IDs: {duplicates[:5]}")

    missing = canonical.difference(observed)
    extra = observed.difference(canonical)
    if len(missing) or len(extra):
        raise ValueError(
            "Compound metadata does not match the canonical analyzed roster: "
            f"{len(missing):,} missing and {len(extra):,} extra IDs"
        )

    ordered = df.set_index("Metadata_JCP2022", drop=False).loc[canonical].reset_index(drop=True)
    if ordered["Metadata_JCP2022"].tolist() != canonical.tolist():
        raise ValueError("Compound metadata could not be reordered to the canonical roster")
    return ordered


def target_moa_coverage(
    con: duckdb.DuckDBPyConnection,
    canonical_ids: list[str],
) -> dict:
    """Compute the four paper-facing target/MOA families and their union."""
    roster = pd.DataFrame({"Metadata_JCP2022": canonical_ids})
    con.register("_publication_analyzed_ids", roster)
    row = con.execute(
        """
        WITH flags AS (
            SELECT
                cm.Metadata_JCP2022,
                NULLIF(trim(cm.Metadata_Uniprot_target), '') IS NOT NULL AS chembl,
                EXISTS (
                    SELECT 1
                    FROM motive_annotations motive
                    WHERE motive.Metadata_JCP2022 = cm.Metadata_JCP2022
                      AND NULLIF(trim(motive.Metadata_motive_gene), '') IS NOT NULL
                ) AS motive,
                (
                    NULLIF(trim(cm.Metadata_repurposing_target), '') IS NOT NULL
                    OR NULLIF(trim(cm.Metadata_repurposing_moa), '') IS NOT NULL
                ) AS repurposing_hub,
                NULLIF(trim(cm.Metadata_chmprb_target_genes), '') IS NOT NULL
                    AS chemical_probes
            FROM compound_metadata cm
            JOIN _publication_analyzed_ids analyzed USING (Metadata_JCP2022)
        )
        SELECT
            count(*) AS analyzed_compounds,
            count(*) FILTER (WHERE chembl) AS chembl,
            count(*) FILTER (WHERE motive) AS motive,
            count(*) FILTER (WHERE repurposing_hub) AS repurposing_hub,
            count(*) FILTER (WHERE chemical_probes) AS chemical_probes,
            count(*) FILTER (
                WHERE chembl OR motive OR repurposing_hub OR chemical_probes
            ) AS target_or_moa_any
        FROM flags
        """
    ).fetchone()
    denominator, chembl, motive, repurposing_hub, chemical_probes, target_or_moa_any = map(int, row)
    if denominator != len(canonical_ids):
        raise ValueError(
            "Target/MOA metadata coverage differs from the canonical analyzed roster: "
            f"{denominator:,} != {len(canonical_ids):,}"
        )

    family_counts = {
        "chembl": chembl,
        "motive": motive,
        "repurposing_hub": repurposing_hub,
        "chemical_probes": chemical_probes,
    }
    labels = {
        "chembl": "ChEMBL/UniProt",
        "motive": "MOTIVE",
        "repurposing_hub": "Repurposing Hub",
        "chemical_probes": "Chemical Probes",
    }
    return {
        "denominator": denominator,
        "denominator_definition": "canonical analyzed compounds",
        "families": [
            {
                "id": family,
                "label": labels[family],
                "count": family_counts[family],
                "percent": 100 * family_counts[family] / denominator,
            }
            for family in ("chembl", "motive", "repurposing_hub", "chemical_probes")
        ],
        "target_or_moa_any": {
            "count": target_or_moa_any,
            "percent": 100 * target_or_moa_any / denominator,
        },
    }


def subset_existing_structure_umap(
    structure_umap_input: Path,
    canonical_ids: list[str],
) -> tuple[sc.AnnData, dict]:
    """Subset a frozen full-collection embedding in canonical analyzed-compound order."""
    adata = sc.read_h5ad(structure_umap_input)
    if "X_umap" not in adata.obsm:
        raise ValueError(f"Structure UMAP input has no X_umap coordinates: {structure_umap_input}")
    coords = np.asarray(adata.obsm["X_umap"])
    if coords.ndim != 2 or coords.shape != (adata.n_obs, 2):
        raise ValueError(f"Structure UMAP coordinates must have shape (n_obs, 2), got {coords.shape}")

    id_column = next(
        (column for column in ("Metadata_JCP2022", "JCP2022") if column in adata.obs.columns),
        None,
    )
    if id_column is None:
        raise ValueError("Structure UMAP input is missing a JCP2022 observation column")
    embedding_ids = pd.Index(adata.obs[id_column], dtype="object")
    if embedding_ids.hasnans:
        raise ValueError("Structure UMAP input contains null JCP2022 IDs")
    if embedding_ids.duplicated().any():
        duplicates = embedding_ids[embedding_ids.duplicated()].unique().tolist()
        raise ValueError(f"Structure UMAP input has duplicate JCP2022 IDs: {duplicates[:5]}")

    embedding_positions = pd.Series(np.arange(len(embedding_ids)), index=embedding_ids)
    present_ids = [compound_id for compound_id in canonical_ids if compound_id in embedding_positions.index]
    missing_ids = [compound_id for compound_id in canonical_ids if compound_id not in embedding_positions.index]
    positions = embedding_positions.loc[present_ids].to_numpy(dtype=int)
    subset = adata[positions].copy()
    observed_subset = subset.obs[id_column].tolist()
    if observed_subset != present_ids:
        raise ValueError("Structure UMAP subset is not in canonical analyzed-compound order")

    return subset, {
        "input_observations": int(adata.n_obs),
        "analyzed_coordinate_count": int(subset.n_obs),
        "analyzed_missing_coordinate_count": int(len(missing_ids)),
        "id_column": id_column,
        "method": "full-collection embedding subset in canonical analyzed-compound order",
    }


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_annotation_coverage(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Bar chart showing annotation coverage by source.

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        dpi: DPI for saved figure.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting annotation coverage...")
        fig, ax = plt.subplots(figsize=(4, 3))

    # Check for annotation columns in the compound metadata
    annotation_cols = {
        "Metadata_repurposing_target": "Rep. Hub\n(Target)",
        "Metadata_repurposing_moa": "Rep. Hub\n(MOA)",
        "Metadata_Uniprot_target": "ChEMBL",
        "Metadata_chmprb_target_genes": "Chem.\nProbes",
        "Metadata_motive_gene_biokg": "MOTIVE",
    }

    coverage = {label: df[col].notna().sum() for col, label in annotation_cols.items() if col in df.columns}

    if not coverage:
        logger.warning("No annotation columns found in compound metadata")
        return

    # Plot
    bars = ax.bar(coverage.keys(), coverage.values(), color="#3498db", edgecolor="none", width=0.7)
    for bar, count in zip(bars, coverage.values()):
        ax.annotate(
            f"{100 * count / len(df):.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel("Compounds", fontsize=9)
    ax.set_title("Annotation Coverage", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "annotations_coverage.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved annotation coverage plot")


def plot_property_distributions(
    df: pd.DataFrame,
    axes: list[plt.Axes] | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Faceted histograms of physicochemical property distributions (2×2 grid).

    Each property gets its own panel with appropriate x-axis scale.

    Args:
        df: Compound DataFrame
        axes: List of 4 matplotlib axes (2×2 layout). If None, creates standalone figure.
        output_dir: Output directory. Only used if axes is None.
        dpi: DPI for saved figure.
    """
    standalone = axes is None
    if standalone:
        logger.info("Plotting property distributions...")
        fig, axes = plt.subplots(2, 2, figsize=(6, 4))
        axes = axes.flatten()

    # Properties with appropriate ranges for each
    props = [
        ("Metadata_MW", "MW (Da)", (100, 700), 40),
        ("Metadata_LogP", "LogP", (-3, 8), 40),
        ("Metadata_TPSA", "TPSA (Ų)", (0, 200), 40),
        ("Metadata_HBD", "HBD", (-0.5, 8.5), 9),  # Integer bins for counts
    ]

    for ax, (col, label, xlim, bins) in zip(axes, props):
        if col not in df.columns:
            continue

        data = df[col].dropna()
        ax.hist(data, bins=bins, range=xlim, color="#3498db", edgecolor="none", alpha=0.8)

        # Add median line and annotation
        median = data.median()
        ax.axvline(median, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(
            f"med={median:.1f}",
            xy=(median, ax.get_ylim()[1] * 0.9),
            fontsize=7,
            color="#e74c3c",
            ha="left" if median < (xlim[0] + xlim[1]) / 2 else "right",
        )

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_xlim(xlim)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if standalone:
        fig.suptitle("Property Distributions", fontsize=10, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(output_dir / "properties_distributions.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved property distributions plot")


def plot_lipinski_violations(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
    style: str = "pie",
) -> None:
    """Pie chart of Lipinski rule violations.

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        dpi: DPI for saved figure.
        style: "pie" (default) or "bar" for chart style.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting Lipinski violations...")
        fig, ax = plt.subplots(figsize=(3, 3))

    if "Metadata_Lipinski_Violations" not in df.columns:
        logger.warning("Metadata_Lipinski_Violations column not found")
        return

    counts = df["Metadata_Lipinski_Violations"].value_counts().sort_index()
    # Colors: green for 0 violations (compliant), yellow/orange/red for increasing violations
    violation_colors = {0: "#2ca02c", 1: "#98df8a", 2: "#ffbb78", 3: "#ff7f0e", 4: "#d62728"}
    pie_colors = [violation_colors.get(int(v), "#999999") for v in counts.index]
    labels = [f"{int(v)} viol." for v in counts.index]

    if style == "pie":
        # Pie chart with legend (avoids label overlap on small slices)
        total = counts.sum()

        wedges, texts = ax.pie(
            counts.values,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )

        # Build legend labels with counts and percentages
        legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / total:.0f}%)" for lbl, cnt in zip(labels, counts.values)]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="center")
    else:
        # Bar chart fallback
        ax.bar(counts.index.astype(int), counts.values, color="#3498db", edgecolor="none", width=0.7)
        for i, count in zip(counts.index, counts.values):
            ax.annotate(f"{100 * count / len(df):.0f}%", xy=(i, count), ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Violations", fontsize=9)
        ax.set_ylabel("Compounds", fontsize=9)
        ax.set_title("Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "lipinski_violations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Lipinski violations plot")


def plot_scaffold_diversity(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    top_n: int = 10,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Horizontal bar chart of most common Murcko scaffolds.

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        top_n: Number of top scaffolds to show.
        dpi: DPI for saved figure.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting scaffold diversity...")
        fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_MurckoScaffold" not in df.columns:
        logger.warning("Metadata_MurckoScaffold column not found")
        return

    # Count scaffolds, separating empty/null (acyclic molecules)
    scaffold_col = df["Metadata_MurckoScaffold"]
    n_acyclic = scaffold_col.isna().sum() + (scaffold_col == "").sum()

    # Filter to non-empty scaffolds for diversity analysis
    valid_scaffolds = scaffold_col[scaffold_col.notna() & (scaffold_col != "")]
    scaffold_counts = valid_scaffolds.value_counts()
    n_unique = len(scaffold_counts)
    n_singletons = (scaffold_counts == 1).sum()

    if standalone:
        logger.info(f"Unique scaffolds: {n_unique:,}")
        logger.info(f"Singleton scaffolds: {n_singletons:,} ({100 * n_singletons / n_unique:.1f}%)")
        logger.info(f"Acyclic molecules (no scaffold): {n_acyclic:,}")

    # Top N scaffolds
    top_scaffolds = scaffold_counts.head(top_n)

    ax.barh(range(len(top_scaffolds)), top_scaffolds.values, color="#3498db", edgecolor="none")
    ax.set_yticks(range(len(top_scaffolds)))
    ax.set_yticklabels(
        [s[:15] + "..." if len(s) > 15 else s for s in top_scaffolds.index],
        fontsize=6,
        family="monospace",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Compounds", fontsize=9)
    ax.set_title(f"Top Scaffolds ({n_unique:,} unique)", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "scaffold_diversity.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Save scaffold stats
        stats = {
            "n_compounds": len(df),
            "n_unique_scaffolds": n_unique,
            "n_singleton_scaffolds": n_singletons,
            "n_acyclic": n_acyclic,
            "scaffold_diversity_ratio": n_unique / len(df),
        }
        pd.DataFrame([stats]).to_csv(output_dir / "scaffold_stats.csv", index=False)
        logger.info("Saved scaffold diversity plot and stats")


def plot_property_space(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Scatter plot of MW vs LogP (property space coverage).

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        dpi: DPI for saved figure.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting property space coverage...")
        fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_MW" not in df.columns or "Metadata_LogP" not in df.columns:
        logger.warning("Metadata_MW or Metadata_LogP columns not found")
        return

    # Filter valid data
    mask = df["Metadata_MW"].notna() & df["Metadata_LogP"].notna()
    x = df.loc[mask, "Metadata_MW"]
    y = df.loc[mask, "Metadata_LogP"]

    # Scatter plot with transparency
    ax.scatter(x, y, s=0.5, alpha=0.1, c="#3498db", rasterized=True)

    # Add Lipinski boundaries
    ax.axvline(500, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(5, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("MW (Da)", fontsize=9)
    ax.set_ylabel("LogP", fontsize=9)
    ax.set_xlim(0, 800)
    ax.set_ylim(-5, 10)
    ax.set_title("Property Space", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "property_space.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved property space plot")


def plot_clinical_phase(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
    style: str = "pie",
) -> None:
    """Pie chart of clinical development phases.

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        dpi: DPI for saved figure.
        style: "pie" (default) or "bar" for chart style.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting clinical phase distribution...")
        fig, ax = plt.subplots(figsize=(4, 3))

    phase_col = "Metadata_repurposing_clinical_phase"
    if phase_col not in df.columns:
        logger.warning(f"{phase_col} column not found")
        return

    # Phase order and labels
    phase_order = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
    phase_labels = ["Pre", "P1", "P2", "P3", "Launch"]
    # Colorbrewer sequential blues with launched in green
    colors = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#2ca02c"]

    phase_counts = df[phase_col].value_counts().reindex(phase_order).fillna(0).astype(int)
    # Filter to non-zero phases for cleaner pie
    nonzero_mask = phase_counts.values > 0
    counts = phase_counts.values[nonzero_mask]
    labels = [phase_labels[i] for i in range(len(phase_labels)) if nonzero_mask[i]]
    pie_colors = [colors[i] for i in range(len(colors)) if nonzero_mask[i]]

    if style == "pie":
        # Pie chart with legend (avoids label overlap on small slices)
        total = counts.sum()

        # Create pie without labels, use legend instead
        wedges, texts = ax.pie(
            counts,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )

        # Build legend labels with counts and percentages
        legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / total:.0f}%)" for lbl, cnt in zip(labels, counts)]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("Clinical Phase", fontsize=10, fontweight="bold", loc="center")
    else:
        # Bar chart fallback
        ax.bar(range(len(phase_counts)), phase_counts.values, color="#3498db", edgecolor="none", width=0.7)
        ax.set_xticks(range(len(phase_counts)))
        ax.set_xticklabels(phase_labels, fontsize=8)
        for i, count in enumerate(phase_counts.values):
            if count > 0:
                ax.annotate(f"{count:,}", xy=(i, count), ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("Compounds", fontsize=9)
        ax.set_title("Clinical Phase", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "clinical_phase.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved clinical phase plot")


def plot_pains_summary(
    df: pd.DataFrame,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
    dpi: int = DEFAULT_DPI,
    style: str = "pie",
) -> None:
    """Pie chart showing PAINS structural alerts.

    Args:
        df: Compound DataFrame
        ax: Matplotlib axes. If None, creates standalone figure.
        output_dir: Output directory. Only used if ax is None.
        dpi: DPI for saved figure.
        style: "pie" (default) or "bar" for chart style.
    """
    standalone = ax is None
    if standalone:
        logger.info("Plotting PAINS summary...")
        fig, ax = plt.subplots(figsize=(4, 3))

    if "Metadata_HasPAINS" not in df.columns:
        logger.warning("Metadata_HasPAINS column not found")
        return

    n_pains = df["Metadata_HasPAINS"].sum()
    n_clean = len(df) - n_pains
    total = len(df)

    if style == "pie":
        # Pie chart with legend (consistent with other pie charts)
        counts = [n_clean, n_pains]
        labels = ["Clean", "PAINS"]
        colors = ["#2ca02c", "#d62728"]

        wedges, texts = ax.pie(
            counts,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            explode=(0, 0.05),  # Slightly explode PAINS slice for emphasis
        )

        # Build legend labels with counts and percentages
        legend_labels = [f"{lbl}: {cnt:,} ({100 * cnt / total:.1f}%)" for lbl, cnt in zip(labels, counts)]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            frameon=False,
        )
        ax.set_title("PAINS Alerts", fontsize=10, fontweight="bold", loc="center")
    else:
        # Stacked horizontal bar fallback
        bar_height = 0.18
        ax.barh([0], [n_clean], height=bar_height, color="#3498db", edgecolor="none", label=f"Clean ({n_clean:,})")
        ax.barh(
            [0],
            [n_pains],
            height=bar_height,
            left=[n_clean],
            color="#e74c3c",
            edgecolor="none",
            label=f"PAINS ({n_pains:,}, {100 * n_pains / len(df):.1f}%)",
        )
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x > 0 else "0"))
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Compounds", fontsize=9)
        ax.set_title("PAINS Structural Alerts", fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if standalone:
        plt.tight_layout()
        fig.savefig(output_dir / "pains_summary.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Save PAINS compound list
        jcp_col = "Metadata_JCP2022" if "Metadata_JCP2022" in df.columns else df.columns[0]
        smiles_col = "Metadata_SMILES" if "Metadata_SMILES" in df.columns else "SMILES"
        pains_cols = [jcp_col, smiles_col, "Metadata_MurckoScaffold"]
        pains_out = df[df["Metadata_HasPAINS"]][[c for c in pains_cols if c in df.columns]].copy()
        if len(pains_out) > 0:
            pains_out.to_csv(output_dir / "pains_compounds.csv", index=False)
            logger.info(f"Saved {len(pains_out):,} PAINS compounds to CSV")

        logger.info("Saved PAINS summary")


def compute_structure_umap(
    df: pd.DataFrame,
    output_dir: Path,
    subsample_frac: float = 1.0,
    seed: int = 42,
    dpi: int = DEFAULT_DPI,
) -> sc.AnnData | None:
    """Compute UMAP from precomputed Morgan fingerprints.

    Uses L2 normalization + cosine similarity as a proxy for Tanimoto/Jaccard.
    This achieves 99.8% rank correlation with Tanimoto (see issue #26) and
    enables GPU acceleration via rapids_singlecell.

    Args:
        df: Compound DataFrame with Metadata_JCP2022
        output_dir: Output directory for h5ad and plots
        subsample_frac: Fraction of compounds to use (0-1). Default 1.0 (all).
        seed: Random seed for reproducibility.
        dpi: DPI for saved figures.

    Returns:
        AnnData with UMAP coordinates, or None if fingerprints not found.
    """
    logger.info("Computing structure UMAP from precomputed fingerprints...")

    # Load precomputed fingerprints
    if not MORGAN_FP_FILE.exists():
        logger.error(f"Fingerprints file not found: {MORGAN_FP_FILE}")
        logger.info("Run: pixi run -e cheminformatics python -m jump_production.processing.compound_properties")
        return None

    fp_data = np.load(MORGAN_FP_FILE, allow_pickle=True)
    fp_array = fp_data["fingerprints"]
    valid_mask = fp_data["valid_mask"]
    jcp2022_fps = fp_data["jcp2022"]

    logger.info(f"Loaded fingerprints: {fp_array.shape}")
    logger.info(f"Valid fingerprints: {valid_mask.sum():,}/{len(valid_mask):,}")

    # Filter to valid fingerprints only
    fp_valid = fp_array[valid_mask]
    jcp_valid = jcp2022_fps[valid_mask]

    # Create mapping from JCP2022 to df index for obs
    jcp_to_idx = {jcp: i for i, jcp in enumerate(df["Metadata_JCP2022"])}

    # Match fingerprints to df rows
    matched_fps = []
    matched_idx = []
    for i, jcp in enumerate(jcp_valid):
        if jcp in jcp_to_idx:
            matched_fps.append(fp_valid[i])
            matched_idx.append(jcp_to_idx[jcp])

    logger.info(f"Matched {len(matched_fps):,} fingerprints to compound metadata")

    # Subsample if requested
    fp_matched = np.array(matched_fps, dtype=np.float32)
    obs_df = df.iloc[matched_idx].reset_index(drop=True)

    if subsample_frac < 1.0:
        n_total = len(fp_matched)
        n_sample = int(n_total * subsample_frac)
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_total, size=n_sample, replace=False)
        sample_idx = np.sort(sample_idx)
        fp_matched = fp_matched[sample_idx]
        obs_df = obs_df.iloc[sample_idx].reset_index(drop=True)
        logger.info(f"Subsampled to {n_sample:,} compounds ({subsample_frac:.0%})")

    # L2 normalize fingerprints for cosine similarity
    # This makes cosine similarity approximate Tanimoto/Jaccard (ρ=0.998, see issue #26)
    # L2 normalization is fast (~0.2s for 115k compounds)
    logger.info("L2 normalizing fingerprints for cosine similarity...")
    norms = np.linalg.norm(fp_matched, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    fp_normalized = fp_matched / norms

    # Suppress anndata ImplicitModificationWarning (triggered by str index transformation)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Transforming to str index.*")
    try:
        from anndata import ImplicitModificationWarning

        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
    except ImportError:
        pass

    # Create AnnData with normalized fingerprints
    adata = sc.AnnData(X=fp_normalized)
    adata.obs = obs_df

    # Compute neighbors + UMAP
    # Using cosine on L2-normalized ≈ Jaccard/Tanimoto for homogeneous bit densities
    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Using GPU acceleration via rapids_singlecell")
        rsc.get.anndata_to_GPU(adata)

    # Exact neighbors and seeded random initialization make repeated layouts agree.
        logger.info("Computing neighbors (cosine on L2-normalized, GPU exact)...")
        rsc.pp.neighbors(adata, n_neighbors=15, use_rep="X", metric="cosine", algorithm="brute", random_state=seed)

        logger.info("Computing UMAP (GPU)...")
        rsc.tl.umap(adata, min_dist=0.1, random_state=seed, init_pos="random")

        logger.info("Transferring data back to CPU")
        rsc.get.anndata_to_CPU(adata)
    else:
        logger.info("Using CPU-only scanpy")
        logger.info("Computing neighbors (cosine on L2-normalized, CPU)...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", metric="cosine", random_state=seed)

        logger.info("Computing UMAP (CPU)...")
        sc.tl.umap(adata, min_dist=0.1, random_state=seed)

    # Save minimal h5ad (obs + UMAP coords only, no X matrix or neighbors graph)
    # This reduces file size from ~1GB to ~10MB while preserving all plotting capability
    adata_minimal = sc.AnnData(
        obs=adata.obs,
        obsm={"X_umap": adata.obsm["X_umap"]},
    )
    adata_minimal.write_h5ad(output_dir / "structure_umap.h5ad")
    logger.info("Saved structure UMAP h5ad (minimal: obs + X_umap only)")

    # Compute IQR-based bounds for outlier clipping (reuse utility function)
    xlim, ylim = compute_umap_bounds(adata, iqr_k=3.0)

    # Identify and save outlier compounds (outside IQR bounds)
    coords = adata.obsm["X_umap"]
    outlier_mask = (
        (coords[:, 0] < xlim[0]) | (coords[:, 0] > xlim[1]) | (coords[:, 1] < ylim[0]) | (coords[:, 1] > ylim[1])
    )
    n_outside = outlier_mask.sum()
    logger.info(f"IQR clipping: {n_outside:,} points ({100 * n_outside / len(coords):.1f}%) outside bounds")

    # Save outlier compounds to CSV
    if n_outside > 0:
        outlier_df = adata.obs[outlier_mask].copy()
        outlier_df["UMAP_1"] = coords[outlier_mask, 0]
        outlier_df["UMAP_2"] = coords[outlier_mask, 1]
        outlier_df["outlier_direction"] = ""
        # Label direction of outlier
        for i, (x, y) in enumerate(coords[outlier_mask]):
            directions = []
            if x < xlim[0]:
                directions.append("left")
            if x > xlim[1]:
                directions.append("right")
            if y < ylim[0]:
                directions.append("bottom")
            if y > ylim[1]:
                directions.append("top")
            outlier_df.iloc[i, outlier_df.columns.get_loc("outlier_direction")] = ",".join(directions)

        # Select key columns for output
        out_cols = ["Metadata_JCP2022", "Metadata_SMILES", "UMAP_1", "UMAP_2", "outlier_direction"]
        if "Metadata_MW" in outlier_df.columns:
            out_cols.insert(2, "Metadata_MW")
        if "Metadata_MurckoScaffold" in outlier_df.columns:
            out_cols.insert(3, "Metadata_MurckoScaffold")
        out_cols = [c for c in out_cols if c in outlier_df.columns]
        outlier_df[out_cols].to_csv(output_dir / "structure_umap_outliers.csv", index=False)
        logger.info(f"Saved {n_outside:,} outlier compounds to structure_umap_outliers.csv")

    # Plot with clipping
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, ax=ax, show=False, title="Chemical Structure Space (Morgan FP UMAP)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.savefig(output_dir / "structure_umap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # If we have activity data, color by it
    if "below_corrected_p" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color="below_corrected_p",
            ax=ax,
            show=False,
            title="Structure UMAP - Phenotypic Activity",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.savefig(output_dir / "structure_umap_activity.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Additional colored UMAPs for annotation and property visualization
    _plot_colored_umaps(adata, output_dir, xlim, ylim, dpi=dpi)

    # Combined UMAP gallery figure (2x3 grid)
    plot_umap_gallery(adata, output_dir, xlim, ylim, dpi=dpi)

    logger.info("Saved structure UMAP plots")
    return adata


def compute_property_umap(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    dpi: int = DEFAULT_DPI,
) -> sc.AnnData | None:
    """Compute UMAP from z-scored physicochemical descriptors.

    Uses Euclidean distance on standardized (z-scored) continuous descriptors
    to visualize property space coverage. This complements the structure UMAP
    which shows structural diversity via fingerprint similarity.

    Args:
        df: Compound DataFrame with descriptor columns
        output_dir: Output directory for h5ad and plots
        seed: Random seed for reproducibility.
        dpi: DPI for saved figures.

    Returns:
        AnnData with UMAP coordinates, or None if descriptors not available.
    """
    logger.info("Computing property UMAP from z-scored descriptors...")

    # Continuous descriptors for property space
    descriptor_cols = [
        "Metadata_MW",
        "Metadata_LogP",
        "Metadata_TPSA",
        "Metadata_HBA",
        "Metadata_HBD",
        "Metadata_RotatableBonds",
        "Metadata_NumRings",
        "Metadata_NumAromaticRings",
        "Metadata_NumHeavyAtoms",
        "Metadata_FractionCSP3",
        "Metadata_QED",
    ]

    # Check which columns exist
    available_cols = [c for c in descriptor_cols if c in df.columns]
    if len(available_cols) < 5:
        logger.warning(f"Only {len(available_cols)} descriptor columns found, skipping property UMAP")
        return None

    logger.info(f"Using {len(available_cols)} descriptors: {available_cols}")

    # Filter to rows with all descriptors present
    desc_df = df[available_cols + ["Metadata_JCP2022"]].dropna()
    logger.info(f"Compounds with all descriptors: {len(desc_df):,}")

    # Z-score standardization
    features = desc_df[available_cols].values.astype(np.float32)
    features_scaled = StandardScaler().fit_transform(features)

    # Suppress anndata warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Transforming to str index.*")

    # Create AnnData
    adata = sc.AnnData(X=features_scaled, obs=desc_df.reset_index(drop=True))

    # Compute neighbors + UMAP with Euclidean distance
    if HAS_GPU:
        import rapids_singlecell as rsc

        logger.info("Computing property UMAP (GPU, Euclidean on z-scored descriptors)...")
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.neighbors(adata, n_neighbors=15, use_rep="X", metric="euclidean", random_state=seed)
        rsc.tl.umap(adata, min_dist=0.1, random_state=seed, init_pos="random")
        rsc.get.anndata_to_CPU(adata)
    else:
        logger.info("Computing property UMAP (CPU, Euclidean on z-scored descriptors)...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X", metric="euclidean", random_state=seed)
        sc.tl.umap(adata, min_dist=0.1, random_state=seed)

    # Save minimal h5ad
    adata_minimal = sc.AnnData(
        obs=adata.obs,
        obsm={"X_umap": adata.obsm["X_umap"]},
    )
    adata_minimal.write_h5ad(output_dir / "property_umap.h5ad")
    logger.info("Saved property UMAP h5ad")

    # Simple single-color plot
    fig, ax = plt.subplots(figsize=(8, 8))
    coords = adata.obsm["X_umap"]
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Property Space UMAP (n={len(adata):,} compounds)", fontweight="bold")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "property_umap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved property UMAP plot")
    return adata


def _plot_colored_umaps(adata: sc.AnnData, output_dir: Path, xlim: tuple, ylim: tuple, dpi: int = DEFAULT_DPI) -> None:
    """Plot additional colored UMAPs for annotations and properties.

    Args:
        adata: AnnData with UMAP coordinates and metadata in obs
        output_dir: Output directory for plots
        xlim: X-axis limits for consistent clipping
        ylim: Y-axis limits for consistent clipping
        dpi: DPI for saved figures.
    """
    # Suppress anndata warnings when plotting subsets
    warnings.filterwarnings("ignore", message="Trying to modify attribute.*of view", category=UserWarning)
    warnings.filterwarnings("ignore", message="Transforming to str index", category=FutureWarning)

    coords = adata.obsm["X_umap"]

    # 1. Clinical phase (categorical)
    phase_col = "Metadata_repurposing_clinical_phase"
    if phase_col in adata.obs.columns:
        # Create ordered categorical for proper legend order
        phase_order = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
        adata.obs[phase_col] = pd.Categorical(adata.obs[phase_col], categories=phase_order, ordered=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot unannotated first (gray background)
        mask_na = adata.obs[phase_col].isna()
        ax.scatter(coords[mask_na, 0], coords[mask_na, 1], s=0.5, alpha=0.1, c="#cccccc", rasterized=True, label=None)
        # Plot annotated with colors
        sc.pl.umap(adata[~mask_na], color=phase_col, ax=ax, show=False, title="Structure UMAP - Clinical Phase")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.savefig(output_dir / "structure_umap_clinical_phase.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved clinical phase UMAP")

    # 2. Has target annotation (binary)
    # Check for any target annotation column
    target_cols = ["Metadata_repurposing_target", "Metadata_Uniprot_target", "Metadata_chmprb_target_genes"]
    has_target = pd.Series(False, index=adata.obs.index)
    for col in target_cols:
        if col in adata.obs.columns:
            has_target = has_target | adata.obs[col].notna()

    if has_target.any():
        adata.obs["has_target_annotation"] = has_target.astype(str)
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot without target first (gray)
        mask_no_target = ~has_target
        ax.scatter(
            coords[mask_no_target, 0],
            coords[mask_no_target, 1],
            s=0.5,
            alpha=0.1,
            c="#cccccc",
            rasterized=True,
            label=f"No target ({mask_no_target.sum():,})",
        )
        # Plot with target (colored)
        ax.scatter(
            coords[has_target, 0],
            coords[has_target, 1],
            s=1,
            alpha=0.5,
            c="#e74c3c",
            rasterized=True,
            label=f"Has target ({has_target.sum():,})",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Structure UMAP - Target Annotation Coverage")
        ax.legend(loc="upper right", markerscale=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(output_dir / "structure_umap_has_target.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved target annotation UMAP")

    # 3. Molecular weight (continuous)
    if "Metadata_MW" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        mw = adata.obs["Metadata_MW"].values
        # Clip MW for colormap (most are 200-600)
        mw_clipped = np.clip(mw, 150, 700)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c=mw_clipped, cmap="viridis", rasterized=True
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Structure UMAP - Molecular Weight")
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.3)
        cbar.set_label("MW (Da)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(output_dir / "structure_umap_mw.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved MW UMAP")

    # 4. LogP (continuous)
    if "Metadata_LogP" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        logp = adata.obs["Metadata_LogP"].values
        # Clip LogP for colormap (most are -2 to 7)
        logp_clipped = np.clip(logp, -2, 8)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c=logp_clipped, cmap="RdYlBu_r", rasterized=True
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Structure UMAP - LogP (Lipophilicity)")
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.3)
        cbar.set_label("LogP")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(output_dir / "structure_umap_logp.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved LogP UMAP")

    # 5. PAINS (binary)
    if "Metadata_HasPAINS" in adata.obs.columns:
        has_pains = adata.obs["Metadata_HasPAINS"].astype(bool)
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot clean first (gray background)
        mask_clean = ~has_pains
        ax.scatter(
            coords[mask_clean, 0],
            coords[mask_clean, 1],
            s=0.5,
            alpha=0.1,
            c="#cccccc",
            rasterized=True,
            label=f"Clean ({mask_clean.sum():,})",
        )
        # Plot PAINS (red)
        ax.scatter(
            coords[has_pains, 0],
            coords[has_pains, 1],
            s=1,
            alpha=0.5,
            c="#e74c3c",
            rasterized=True,
            label=f"PAINS ({has_pains.sum():,})",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Structure UMAP - PAINS Structural Alerts")
        ax.legend(loc="upper right", markerscale=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(output_dir / "structure_umap_pains.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PAINS UMAP")


def plot_umap_gallery(adata: sc.AnnData, output_dir: Path, xlim: tuple, ylim: tuple, dpi: int = DEFAULT_DPI) -> None:
    """Create a 2x3 UMAP gallery figure with all colorings.

    Layout:
        Base (overview)     | Clinical Phase      | Has Target
        Molecular Weight    | LogP                | PAINS

    Args:
        adata: AnnData with UMAP coordinates and metadata in obs
        output_dir: Output directory for plots
        xlim: X-axis limits for consistent clipping
        ylim: Y-axis limits for consistent clipping
        dpi: DPI for saved figure.
    """
    logger.info("Creating UMAP gallery figure...")

    coords = adata.obsm["X_umap"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Common plot settings
    point_size = 0.3
    alpha = 0.3

    # --- Row 1 ---

    # A: Base (overview)
    ax = axes[0, 0]
    ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, c="#3498db", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("A. Chemical Structure Space", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # B: Clinical Phase
    ax = axes[0, 1]
    phase_col = "Metadata_repurposing_clinical_phase"
    if phase_col in adata.obs.columns:
        # Gray background for unannotated
        mask_na = adata.obs[phase_col].isna()
        ax.scatter(coords[mask_na, 0], coords[mask_na, 1], s=point_size, alpha=0.1, c="#cccccc", rasterized=True)
        # Colored for annotated
        phase_order = ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]
        colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#e31a1c"]
        for phase, color in zip(phase_order, colors):
            mask = adata.obs[phase_col] == phase
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1], s=1, alpha=0.6, c=color, rasterized=True, label=phase)
        ax.legend(fontsize=6, loc="upper right", markerscale=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("B. Clinical Phase", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # C: Has Target
    ax = axes[0, 2]
    target_cols = ["Metadata_repurposing_target", "Metadata_Uniprot_target", "Metadata_chmprb_target_genes"]
    has_target = pd.Series(False, index=adata.obs.index)
    for col in target_cols:
        if col in adata.obs.columns:
            has_target = has_target | adata.obs[col].notna()
    mask_no_target = ~has_target
    ax.scatter(
        coords[mask_no_target, 0], coords[mask_no_target, 1], s=point_size, alpha=0.1, c="#cccccc", rasterized=True
    )
    ax.scatter(coords[has_target, 0], coords[has_target, 1], s=0.8, alpha=0.5, c="#e74c3c", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title(f"C. Has Target ({has_target.sum():,})", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Row 2 ---

    # D: Molecular Weight
    ax = axes[1, 0]
    if "Metadata_MW" in adata.obs.columns:
        mw = adata.obs["Metadata_MW"].values
        mw_clipped = np.clip(mw, 150, 700)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, c=mw_clipped, cmap="viridis", rasterized=True
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.3)
        cbar.set_label("MW (Da)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("D. Molecular Weight", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # E: LogP
    ax = axes[1, 1]
    if "Metadata_LogP" in adata.obs.columns:
        logp = adata.obs["Metadata_LogP"].values
        logp_clipped = np.clip(logp, -2, 8)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=point_size, alpha=alpha, c=logp_clipped, cmap="RdYlBu_r", rasterized=True
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.3)
        cbar.set_label("LogP", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("E. LogP (Lipophilicity)", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # F: PAINS
    ax = axes[1, 2]
    if "Metadata_HasPAINS" in adata.obs.columns:
        has_pains = adata.obs["Metadata_HasPAINS"].astype(bool)
        mask_clean = ~has_pains
        ax.scatter(coords[mask_clean, 0], coords[mask_clean, 1], s=point_size, alpha=0.1, c="#cccccc", rasterized=True)
        ax.scatter(coords[has_pains, 0], coords[has_pains, 1], s=0.8, alpha=0.5, c="#e74c3c", rasterized=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    n_pains = has_pains.sum() if "Metadata_HasPAINS" in adata.obs.columns else 0
    ax.set_title(f"F. PAINS Alerts ({n_pains:,})", fontsize=10, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Overall title and save
    fig.suptitle("Structure UMAP Gallery", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(output_dir / "structure_umap_gallery.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "structure_umap_gallery.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved UMAP gallery figure (PNG + PDF)")


# =============================================================================
# Combined Publication Figure
# =============================================================================


def plot_combined_figure(
    df: pd.DataFrame, output_dir: Path, adata_umap: sc.AnnData | None = None, dpi: int = DEFAULT_DPI
) -> None:
    """Create combined publication-ready figure with all chemical space panels.

    Layout (3x3 grid):
        A: Annotation coverage    B: Clinical phase       C: Property space
        D: Property distributions (2×2 faceted, spans 2 cols)   E: Lipinski violations
        F: Scaffold diversity     G: PAINS summary        H: Structure UMAP

    Uses the same plotting functions as individual plots to avoid code duplication.

    Args:
        df: Compound DataFrame with properties
        output_dir: Output directory for figure files
        adata_umap: AnnData with UMAP coordinates (optional, panel H empty if None)
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    logger.info("Creating combined publication figure...")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5)

    # Row 1: A, B, C
    ax_a = fig.add_subplot(gs[0, 0])
    plot_annotation_coverage(df, ax=ax_a)
    ax_a.set_title("A. Annotation Coverage", fontsize=10, fontweight="bold", loc="left")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_clinical_phase(df, ax=ax_b, style="pie")
    ax_b.set_title("B. Clinical Phase", fontsize=10, fontweight="bold", loc="center")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_property_space(df, ax=ax_c)
    ax_c.set_title("C. Property Space", fontsize=10, fontweight="bold", loc="left")

    # Row 2: D (2×2 faceted histograms spanning 2 cols), E
    gs_d = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, :2], hspace=0.55, wspace=0.25)
    axes_d = [fig.add_subplot(gs_d[i, j]) for i in range(2) for j in range(2)]
    plot_property_distributions(df, axes=axes_d)
    # Add panel label to top-left subplot
    axes_d[0].annotate(
        "D. Property Distributions",
        xy=(0, 1.25),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
    )

    ax_e = fig.add_subplot(gs[1, 2])
    plot_lipinski_violations(df, ax=ax_e, style="pie")
    ax_e.set_title("E. Lipinski Rule-of-5", fontsize=10, fontweight="bold", loc="center")

    # Row 3: F, G, H
    ax_f = fig.add_subplot(gs[2, 0])
    plot_scaffold_diversity(df, ax=ax_f)
    ax_f.set_title("F. Top Scaffolds", fontsize=10, fontweight="bold", loc="left")

    ax_g = fig.add_subplot(gs[2, 1])
    plot_pains_summary(df, ax=ax_g, style="pie")
    ax_g.set_title("G. PAINS Alerts", fontsize=10, fontweight="bold", loc="center")

    # Panel H: Structure UMAP (with IQR-based outlier clipping)
    ax_h = fig.add_subplot(gs[2, 2])
    if adata_umap is not None and "X_umap" in adata_umap.obsm:
        coords = adata_umap.obsm["X_umap"]
        xlim, ylim = compute_umap_bounds(adata_umap, iqr_k=3.0)

        ax_h.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
        ax_h.set_xlim(xlim)
        ax_h.set_ylim(ylim)
        ax_h.set_xlabel("UMAP 1", fontsize=8)
        ax_h.set_ylabel("UMAP 2", fontsize=8)
        ax_h.tick_params(labelsize=7)
        ax_h.set_aspect("equal")
    else:
        ax_h.text(0.5, 0.5, "UMAP not computed", ha="center", va="center", fontsize=9, color="gray")
        ax_h.set_xticks([])
        ax_h.set_yticks([])
    ax_h.set_title("H. Structure UMAP", fontsize=10, fontweight="bold", loc="left")
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    # Overall title
    fig.suptitle(f"Chemical Space Characterization (n={len(df):,} compounds)", fontsize=12, fontweight="bold", y=0.98)

    fig.savefig(output_dir / "chemical_space_combined.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "chemical_space_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined publication figure (PNG + PDF)")


def plot_publication_annotation_coverage(ax: plt.Axes, coverage: dict) -> None:
    """Plot the four target/MOA provider families and their paper-facing union."""
    rows = [
        {
            "id": "target_or_moa_any",
            "label": "Any target/MOA",
            **coverage["target_or_moa_any"],
        },
        *coverage["families"],
    ]
    display = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    colors = ["#0072B2"] * len(display)
    colors[-1] = "#023858"
    bars = ax.barh(np.arange(len(display)), display["count"], color=colors, edgecolor="none")
    ax.set_yticks(np.arange(len(display)))
    ax.set_yticklabels(display["label"], fontsize=7)
    maximum = float(display["count"].max())
    for bar, row in zip(bars, display.itertuples(index=False), strict=True):
        ax.text(
            bar.get_width() + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.count):,} ({row.percent:.1f}%)",
            va="center",
            ha="left",
            fontsize=7,
        )
    ax.axhline(len(display) - 1.5, color="#DCE3EA", linewidth=0.8)
    ax.set_xlim(0, maximum * 1.42)
    ax.set_xticks([])
    ax.set_title("Target/MOA annotation coverage", fontsize=10, fontweight="bold", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_chemical_space_publication_panels(
    df: pd.DataFrame,
    output_dir: Path,
    caption: str = "",
    dpi: int = DEFAULT_DPI,
    publication_annotation_coverage: dict | None = None,
) -> None:
    """Chemical-space strip for the compound-collection package.

    Three horizontal panels - clinical phase, Lipinski violations, annotation coverage - over a
    plain-text row of median properties. Panels are keyed by position rather than by a letter,
    because the letters belonged to the paper's layout and drift when it is rearranged. Reuses
    the same panel functions as the combined figure; the median row is computed live from df.
    """
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 3, height_ratios=[6, 1], hspace=0.3, wspace=0.4)

    ax_c = fig.add_subplot(gs[0, 0])
    plot_clinical_phase(df, ax=ax_c, style="pie")
    ax_c.set_title("Clinical phase", fontsize=11, fontweight="bold", loc="center")

    ax_d = fig.add_subplot(gs[0, 1])
    plot_lipinski_violations(df, ax=ax_d, style="pie")
    ax_d.set_title("Lipinski Rule-of-5", fontsize=11, fontweight="bold", loc="center")

    ax_e = fig.add_subplot(gs[0, 2])
    if publication_annotation_coverage is None:
        plot_annotation_coverage(df, ax=ax_e)
        ax_e.set_title("Annotation coverage", fontsize=11, fontweight="bold", loc="left")
    else:
        plot_publication_annotation_coverage(ax_e, publication_annotation_coverage)
        ax_e.set_title("Target/MOA annotation coverage", fontsize=11, fontweight="bold", loc="left")

    # The pie panels default their legend to the right (bbox_to_anchor=(1, 0.5)), which in a
    # 3-up strip collides with the next panel. Re-place each beneath its pie, horizontally.
    for _ax in (ax_c, ax_d):
        _leg = _ax.get_legend()
        if _leg is not None:
            _handles = _leg.legend_handles
            _labels = [t.get_text() for t in _leg.get_texts()]
            _leg.remove()
            _ax.legend(
                _handles,
                _labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
                fontsize=7,
                frameon=False,
                handlelength=1.0,
                columnspacing=1.2,
            )

    med = {
        c: df[c].median()
        for c in ["Metadata_QED", "Metadata_MW", "Metadata_LogP", "Metadata_TPSA", "Metadata_HBD", "Metadata_HBA"]
    }
    f_text = (
        "Median properties:  "
        f"QED = {med['Metadata_QED']:.2f},  "
        f"MW = {med['Metadata_MW']:.0f} Da,  "
        f"LogP = {med['Metadata_LogP']:.2f},  "
        f"TPSA = {med['Metadata_TPSA']:.1f} A$^2$,  "
        f"HBD = {med['Metadata_HBD']:.0f},  "
        f"HBA = {med['Metadata_HBA']:.0f}"
    )
    ax_f = fig.add_subplot(gs[1, :])
    ax_f.axis("off")
    ax_f.text(0.0, 0.6, f_text, ha="left", va="center", fontsize=11, fontweight="bold")

    # The exploratory dump of this strip carries no caption; only the publication package does.
    if caption:
        attach_caption(fig, caption, caption_points=6.5)
    fig.savefig(output_dir / "chemical_space.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "chemical_space.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved the chemical-space strip (PNG + PDF)")


def plot_supp_chemical_properties(df: pd.DataFrame, output_dir: Path, adata_umap=None, dpi: int = DEFAULT_DPI) -> None:
    """Supplementary Figure CHEMICAL PROPERTIES.

    The 'remainder' panels after Clinical Phase / Lipinski / Annotation Coverage move to the
    main Figure 1: A property space, B property distributions (2x2 faceted), C top scaffolds,
    D PAINS alerts, E structure UMAP.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_property_space(df, ax=ax_a)
    ax_a.set_title("A. Property Space", fontsize=10, fontweight="bold", loc="left")

    gs_b = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1:], hspace=0.55, wspace=0.3)
    axes_b = [fig.add_subplot(gs_b[i, j]) for i in range(2) for j in range(2)]
    plot_property_distributions(df, axes=axes_b)
    axes_b[0].annotate(
        "B. Property Distributions", xy=(0, 1.25), xycoords="axes fraction", fontsize=10, fontweight="bold"
    )

    ax_c = fig.add_subplot(gs[1, 0])
    plot_scaffold_diversity(df, ax=ax_c)
    ax_c.set_title("C. Top Scaffolds", fontsize=10, fontweight="bold", loc="left")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_pains_summary(df, ax=ax_d, style="pie")
    ax_d.set_title("D. PAINS Alerts", fontsize=10, fontweight="bold", loc="center")

    ax_e = fig.add_subplot(gs[1, 2])
    if adata_umap is not None and "X_umap" in adata_umap.obsm:
        coords = adata_umap.obsm["X_umap"]
        xlim, ylim = compute_umap_bounds(adata_umap, iqr_k=3.0)
        ax_e.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.3, c="#3498db", rasterized=True)
        ax_e.set_xlim(xlim)
        ax_e.set_ylim(ylim)
        ax_e.set_xlabel("UMAP 1", fontsize=8)
        ax_e.set_ylabel("UMAP 2", fontsize=8)
        ax_e.tick_params(labelsize=7)
        ax_e.set_aspect("equal")
    else:
        ax_e.text(0.5, 0.5, "UMAP not computed", ha="center", va="center", fontsize=9, color="gray")
        ax_e.set_xticks([])
        ax_e.set_yticks([])
    ax_e.set_title("E. Structure UMAP", fontsize=10, fontweight="bold", loc="left")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    fig.suptitle(
        f"Extended chemical characterization (n={len(df):,} compounds)", fontsize=12, fontweight="bold", y=0.98
    )
    fig.savefig(output_dir / "supp_chemical_properties.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "supp_chemical_properties.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Supp Fig CHEMICAL PROPERTIES (PNG + PDF)")


def _publication_viewport(values: pd.Series, bounds: tuple[float, float]) -> dict:
    """Count non-missing values inside and outside one fixed inclusive viewport."""
    complete = values.dropna()
    in_view = complete.between(*bounds)
    return {
        "bounds": [float(bounds[0]), float(bounds[1])],
        "candidate_count": int(len(complete)),
        "included_count": int(in_view.sum()),
        "excluded_count": int((~in_view).sum()),
    }


def publication_scaffold_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rank the top 30 Bemis-Murcko scaffolds with an explicit SMILES tiebreaker."""
    scaffold = df["Metadata_MurckoScaffold"]
    valid = scaffold[scaffold.notna() & scaffold.ne("")]
    counts = (
        valid.groupby(valid, sort=False)
        .size()
        .rename_axis("scaffold")
        .rename("count")
        .reset_index()
        .sort_values(["count", "scaffold"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    counts["count"] = counts["count"].astype(int)
    top = counts.head(PUBLICATION_SCAFFOLD_EXPORT_N).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top.insert(1, "display_label", [f"Scaffold {rank:02d}" for rank in top["rank"]])
    unique = int(len(counts))
    singletons = int((counts["count"] == 1).sum())
    statistics = {
        "unique": unique,
        "singletons": singletons,
        "singleton_percent": 100 * singletons / unique,
        "acyclic": int((scaffold.isna() | scaffold.eq("")).sum()),
        "diversity_ratio": unique / len(df),
        "diversity_ratio_definition": "unique non-empty scaffolds divided by analyzed compounds",
        "interpretation": "high structural diversity",
        "exported": int(len(top)),
        "displayed": min(PUBLICATION_SCAFFOLD_DISPLAY_N, len(top)),
        "ordering": "count descending, then scaffold SMILES ascending",
    }
    return top, statistics


def publication_pains_summary(df: pd.DataFrame) -> dict:
    """Partition the analyzed roster into clean, flagged, and unassessable compounds."""
    valid = df["Metadata_ValidMol"].astype("boolean").fillna(False).astype(bool)
    flagged = df["Metadata_HasPAINS"].astype("boolean").fillna(False).astype(bool)
    if (flagged & ~valid).any():
        raise ValueError("A PAINS alert cannot be assigned to an unassessable structure")
    summary = {
        "total": int(len(df)),
        "assessable": int(valid.sum()),
        "clean": int((valid & ~flagged).sum()),
        "flagged": int((valid & flagged).sum()),
        "unassessable": int((~valid).sum()),
        "prevalence_denominator": "compounds with valid molecular structures",
        "downstream_activity_handling": "flagged but not excluded",
    }
    if summary["clean"] + summary["flagged"] + summary["unassessable"] != summary["total"]:
        raise ValueError("PAINS categories do not partition the canonical analyzed roster")
    summary["flagged_percent_of_assessable"] = 100 * summary["flagged"] / summary["assessable"]
    summary["unassessable_percent_of_total"] = 100 * summary["unassessable"] / summary["total"]
    return summary


def publication_supplement_numbers(
    df: pd.DataFrame,
    adata_umap: sc.AnnData,
    embedding: dict,
) -> tuple[dict, pd.DataFrame]:
    """Compute the corrected supplementary display counts and denominators."""
    property_space_complete = df.dropna(subset=["Metadata_MW", "Metadata_LogP"])
    property_space_in_view = property_space_complete["Metadata_MW"].between(
        *PUBLICATION_PROPERTY_SPACE_BOUNDS["Metadata_MW"]
    ) & property_space_complete["Metadata_LogP"].between(*PUBLICATION_PROPERTY_SPACE_BOUNDS["Metadata_LogP"])
    property_space = {
        "bounds": {
            "mw": list(PUBLICATION_PROPERTY_SPACE_BOUNDS["Metadata_MW"]),
            "logp": list(PUBLICATION_PROPERTY_SPACE_BOUNDS["Metadata_LogP"]),
        },
        "candidate_count": int(len(property_space_complete)),
        "included_count": int(property_space_in_view.sum()),
        "excluded_count": int((~property_space_in_view).sum()),
        "density_scale": "log-count hexbin",
    }
    distributions = {
        column.removeprefix("Metadata_").lower(): _publication_viewport(df[column], bounds)
        for column, bounds in PUBLICATION_PROPERTY_DISTRIBUTION_BOUNDS.items()
    }
    medians = {
        column.removeprefix("Metadata_").lower(): {
            "median": float(df[column].median()),
            "denominator": int(df[column].notna().sum()),
        }
        for column in PUBLICATION_PROPERTY_DISTRIBUTION_BOUNDS
    }

    top_scaffolds, scaffold_statistics = publication_scaffold_table(df)
    pains = publication_pains_summary(df)
    coords = np.asarray(adata_umap.obsm["X_umap"])
    xlim, ylim = compute_umap_bounds(adata_umap, iqr_k=PUBLICATION_UMAP_IQR_K)
    in_view = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1]) & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )
    umap = {
        **embedding,
        "bounds": {"x": [float(xlim[0]), float(xlim[1])], "y": [float(ylim[0]), float(ylim[1])]},
        "bound_method": f"independent-axis {PUBLICATION_UMAP_IQR_K:g}-IQR",
        "included_count": int(in_view.sum()),
        "excluded_count": int((~in_view).sum()),
        "density_scale": "log-count hexbin",
        "input_representation": "Morgan fingerprints",
        "distance": "cosine on L2-normalized fingerprints",
        "tanimoto_proxy": True,
        "interpretation": "density structure does not establish discrete chemical classes",
    }
    return {
        "population": {
            "analyzed_compounds": int(len(df)),
            "valid_molecules": int(df["Metadata_ValidMol"].fillna(False).sum()),
        },
        "property_space": property_space,
        "property_distributions": distributions,
        "property_medians": medians,
        "pains": pains,
        "scaffolds": scaffold_statistics,
        "structure_umap": umap,
    }, top_scaffolds


def validate_publication_embedding_coverage(df: pd.DataFrame, adata_umap: sc.AnnData) -> None:
    """Require the frozen embedding to contain every and only valid analyzed molecule."""
    id_column = next(
        (column for column in ("Metadata_JCP2022", "JCP2022") if column in adata_umap.obs.columns),
        None,
    )
    if id_column is None:
        raise ValueError("Structure UMAP subset is missing a JCP2022 observation column")
    valid = df["Metadata_ValidMol"].fillna(False).astype(bool)
    expected_ids = df.loc[valid, "Metadata_JCP2022"].tolist()
    observed_ids = adata_umap.obs[id_column].tolist()
    if observed_ids != expected_ids:
        missing = len(set(expected_ids) - set(observed_ids))
        extra = len(set(observed_ids) - set(expected_ids))
        raise ValueError(
            "Frozen UMAP coverage differs from valid molecules in the canonical roster: "
            f"{missing:,} missing and {extra:,} extra IDs"
        )


def plot_publication_supplement(
    df: pd.DataFrame,
    adata_umap: sc.AnnData,
    numbers: dict,
    top_scaffolds: pd.DataFrame,
    caption: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Render the corrected five-panel supplementary chemical-properties composite."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(2, 3, hspace=0.48, wspace=0.5)

    property_space = numbers["property_space"]
    complete = df.dropna(subset=["Metadata_MW", "Metadata_LogP"])
    in_view = complete["Metadata_MW"].between(*property_space["bounds"]["mw"]) & complete["Metadata_LogP"].between(
        *property_space["bounds"]["logp"]
    )
    property_data = complete.loc[in_view]
    axis_a = figure.add_subplot(grid[0, 0])
    property_hexbin = axis_a.hexbin(
        property_data["Metadata_MW"],
        property_data["Metadata_LogP"],
        gridsize=PUBLICATION_PROPERTY_HEXBIN_GRIDSIZE,
        cmap=PUBLICATION_DENSITY_CMAP,
        bins="log",
        mincnt=1,
        extent=(*property_space["bounds"]["mw"], *property_space["bounds"]["logp"]),
        rasterized=True,
    )
    axis_a.axvline(PUBLICATION_LIPINSKI_MW_CUTOFF_DA, color="#D55E00", linestyle="--", linewidth=1)
    axis_a.axhline(PUBLICATION_LIPINSKI_LOGP_CUTOFF, color="#D55E00", linestyle="--", linewidth=1)
    axis_a.set_xlim(*property_space["bounds"]["mw"])
    axis_a.set_ylim(*property_space["bounds"]["logp"])
    axis_a.set_xlabel("Molecular weight (Da)", fontsize=8)
    axis_a.set_ylabel("LogP", fontsize=8)
    axis_a.set_title("Property space", fontsize=10, fontweight="bold", loc="left")
    axis_a.text(
        0.03,
        0.97,
        f"n = {property_space['included_count']:,} in view\n{property_space['excluded_count']:,} outside bounds",
        transform=axis_a.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    figure.colorbar(property_hexbin, ax=axis_a, shrink=0.75, label="Density (log count)")

    distribution_grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[0, 1:], hspace=0.6, wspace=0.32)
    distribution_specs = [
        ("mw", "Metadata_MW", "Molecular weight (Da)"),
        ("logp", "Metadata_LogP", "LogP"),
        ("tpsa", "Metadata_TPSA", "TPSA (A^2)"),
        ("qed", "Metadata_QED", "QED"),
    ]
    distribution_axes = []
    for index, (key, column, label) in enumerate(distribution_specs):
        axis = figure.add_subplot(distribution_grid[index // 2, index % 2])
        distribution_axes.append(axis)
        viewport = numbers["property_distributions"][key]
        values = df[column].dropna()
        displayed = values.loc[values.between(*viewport["bounds"])]
        axis.hist(
            displayed,
            bins=PUBLICATION_HISTOGRAM_BINS,
            range=viewport["bounds"],
            color="#0072B2",
            edgecolor="none",
            alpha=0.85,
        )
        median = numbers["property_medians"][key]["median"]
        axis.axvline(median, color="#D55E00", linestyle="--", linewidth=1.2)
        axis.text(0.97, 0.9, f"median = {median:g}", transform=axis.transAxes, ha="right", fontsize=7)
        axis.text(
            0.03,
            0.9,
            f"n = {viewport['included_count']:,}\n{viewport['excluded_count']:,} outside",
            transform=axis.transAxes,
            va="top",
            fontsize=6.5,
        )
        axis.set_xlim(viewport["bounds"])
        axis.set_xlabel(label, fontsize=8)
        axis.set_ylabel("Compounds", fontsize=7)
        axis.tick_params(labelsize=7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    distribution_axes[0].annotate(
        "Physicochemical property distributions",
        xy=(0, 1.28),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
    )

    displayed_scaffolds = top_scaffolds.head(PUBLICATION_SCAFFOLD_DISPLAY_N)
    axis_c = figure.add_subplot(grid[1, 0])
    y = np.arange(len(displayed_scaffolds))
    axis_c.barh(y, displayed_scaffolds["count"], color="#0072B2", edgecolor="none")
    axis_c.set_yticks(y)
    axis_c.set_yticklabels(displayed_scaffolds["display_label"], fontsize=7)
    axis_c.invert_yaxis()
    axis_c.set_xlabel("Compounds", fontsize=8)
    axis_c.set_title("Top Bemis-Murcko scaffolds", fontsize=10, fontweight="bold", loc="left")
    axis_c.set_xlim(0, float(displayed_scaffolds["count"].max()) * 1.16)
    for yi, count in zip(y, displayed_scaffolds["count"], strict=True):
        axis_c.text(count, yi, f" {int(count):,}", va="center", fontsize=6.5)
    axis_c.spines["top"].set_visible(False)
    axis_c.spines["right"].set_visible(False)

    pains = numbers["pains"]
    axis_d = figure.add_subplot(grid[1, 1])
    wedges, _ = axis_d.pie(
        [pains["clean"], pains["flagged"], pains["unassessable"]],
        startangle=90,
        counterclock=False,
        colors=["#0072B2", "#D55E00", "#999999"],
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.0},
    )
    axis_d.text(0, 0.06, f"{pains['total']:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    axis_d.text(0, -0.14, "analyzed", ha="center", va="center", fontsize=8, color="#555555")
    axis_d.legend(
        wedges,
        [
            f"No alert: {pains['clean']:,}",
            f"PAINS: {pains['flagged']:,} ({pains['flagged_percent_of_assessable']:.1f}% of valid)",
            f"Unassessable: {pains['unassessable']:,}",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=7,
    )
    axis_d.set_title("PAINS alerts", fontsize=10, fontweight="bold")
    axis_d.set_aspect("equal")

    umap = numbers["structure_umap"]
    coords = np.asarray(adata_umap.obsm["X_umap"])
    xlim = umap["bounds"]["x"]
    ylim = umap["bounds"]["y"]
    umap_in_view = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1]) & (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )
    axis_e = figure.add_subplot(grid[1, 2])
    umap_hexbin = axis_e.hexbin(
        coords[umap_in_view, 0],
        coords[umap_in_view, 1],
        gridsize=PUBLICATION_UMAP_HEXBIN_GRIDSIZE,
        cmap=PUBLICATION_DENSITY_CMAP,
        bins="log",
        mincnt=1,
        extent=(*xlim, *ylim),
        rasterized=True,
    )
    axis_e.set_xlim(xlim)
    axis_e.set_ylim(ylim)
    axis_e.set_aspect("equal", adjustable="box")
    axis_e.set_xlabel("UMAP 1", fontsize=8)
    axis_e.set_ylabel("UMAP 2", fontsize=8)
    axis_e.set_xticks([])
    axis_e.set_yticks([])
    axis_e.set_title("Structure UMAP", fontsize=10, fontweight="bold", loc="left")
    axis_e.text(
        0.03,
        0.97,
        f"{umap['included_count']:,} in view\n{umap['excluded_count']:,} outside bounds",
        transform=axis_e.transAxes,
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    for spine in axis_e.spines.values():
        spine.set_visible(False)
    figure.colorbar(umap_hexbin, ax=axis_e, shrink=0.75, label="Density (log count)")

    figure.suptitle(
        f"Extended chemical characterization (n={len(df):,} analyzed compounds)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    attach_caption(figure, caption, caption_points=6.5)
    figure.savefig(output_dir / "supp_chemical_properties.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "supp_chemical_properties.pdf", bbox_inches="tight")
    plt.close(figure)


def publication_clinical_phase_summary(df: pd.DataFrame) -> dict:
    """Summarize the five exact displayed stages and all other non-null phase values."""
    phase_values = df["Metadata_repurposing_clinical_phase"]
    non_null = phase_values.dropna()
    phase_counts = non_null.value_counts().reindex(PUBLICATION_CLINICAL_PHASE_ORDER).fillna(0).astype(int)
    displayed_count = int(phase_counts.sum())
    excluded_counts = non_null.loc[~non_null.isin(PUBLICATION_CLINICAL_PHASE_ORDER)].value_counts().sort_index()
    excluded_count = int(excluded_counts.sum())
    if displayed_count + excluded_count != len(non_null):
        raise ValueError("Displayed and excluded clinical-phase values do not partition the non-null values")
    return {
        "denominator": displayed_count,
        "denominator_definition": (
            "analyzed compounds assigned exactly one of the five displayed single-stage categories"
        ),
        "non_null_count": int(len(non_null)),
        "excluded_count": excluded_count,
        "excluded_definition": "non-null mixed-stage or Withdrawn values omitted from the display",
        "excluded_value_counts": {str(value): int(count) for value, count in excluded_counts.items()},
        "counts": {phase: int(phase_counts[phase]) for phase in PUBLICATION_CLINICAL_PHASE_ORDER},
    }


def load_dataset_structure_scope(summary_path: Path = DATASET_STRUCTURE_SUMMARY) -> dict:
    """Load the full collection count and retain the manuscript-schematic scope."""
    summary = json.loads(summary_path.read_text())
    try:
        full_collection_compounds = int(summary["numbers"]["headline"]["compounds_perturbation"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Dataset-structure summary is missing the full collection count: {summary_path}") from error
    return {
        "asset_role": "versioned manuscript schematic composited on white for publication",
        "quantitative_evidence": summary_path.as_posix(),
        "full_collection_compounds": full_collection_compounds,
        "partner_sites": PUBLICATION_PARTNER_SITE_COUNT,
        "cpg0012_reference_compounds": PUBLICATION_CPG0012_COMPOUNDS,
    }


def draw_caption_band(image: Image.Image, caption: str, dpi: int) -> Image.Image:
    """Return the image grown downward, with its caption drawn in the new band.

    The matplotlib figures get this from `attach_caption`. This schematic is a versioned asset
    composited pixel for pixel rather than rendered, so its caption is drawn here instead, on the
    same wrap and band-height rule, and its own pixels are left untouched.
    """
    width_in = image.width / dpi
    caption_points = width_in * 72 / (0.75 * SCHEMATIC_CAPTION_CHARACTERS)
    wrapped, total_in, _fraction = property_figure_caption_layout(
        caption,
        width_in=width_in,
        panel_height_in=image.height / dpi,
        caption_points=caption_points,
    )
    band_px = round(total_in * dpi) - image.height
    font_px = max(1, round(caption_points / 72 * dpi))
    canvas = Image.new("RGB", (image.width, image.height + band_px), "white")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).multiline_text(
        (image.width / 2, image.height + round(0.12 * dpi)),
        wrapped,
        font=ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=font_px),
        fill=PROPERTY_FIGURE_COLOR["gray"],
        anchor="ma",
        align="center",
        spacing=round(0.35 * font_px),
    )
    return canvas


def composite_publication_schematic_on_white(source: Path, target: Path, caption: str, dpi: int) -> dict:
    """Composite a transparent authored schematic onto white without modifying its source."""
    if source.resolve() == target.resolve():
        raise ValueError("Publication schematic output must differ from the authored source asset")
    with Image.open(source) as source_image:
        source_mode = source_image.mode
        rgba = source_image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        opaque = canvas.convert("RGB")
        schematic_width, schematic_height = opaque.size
        captioned = draw_caption_band(opaque, caption, dpi)
        captioned.save(target, format="PNG", dpi=(dpi, dpi))
        width_px, height_px = captioned.size
    return {
        "source_mode": source_mode,
        "publication_mode": "RGB",
        "publication_background": "white",
        "width_px": int(width_px),
        "height_px": int(height_px),
        "schematic_height_px": int(schematic_height),
        "caption_band_px": int(height_px - schematic_height),
        "schematic_width_px": int(schematic_width),
    }


def figure1_publication_numbers(
    df: pd.DataFrame,
    coverage: dict,
    key: dict,
    dataset_structure: dict,
) -> dict:
    """Collect the Figure 1 values with their exact displayed denominators."""
    clinical_phase = publication_clinical_phase_summary(df)
    valid = df["Metadata_ValidMol"].fillna(False).astype(bool)
    violations = df.loc[valid, "Metadata_Lipinski_Violations"]
    if violations.isna().any():
        raise ValueError("A valid molecule is missing its Lipinski violation count")
    unexpected_violations = sorted(set(violations.astype(int)) - set(range(5)))
    if unexpected_violations:
        raise ValueError(f"Unexpected Lipinski violation counts: {unexpected_violations}")
    violation_counts = violations.astype(int).value_counts().reindex(range(5)).fillna(0).astype(int)

    property_columns = {
        "qed": "Metadata_QED",
        "molecular_weight_da": "Metadata_MW",
        "logp": "Metadata_LogP",
        "tpsa_angstrom2": "Metadata_TPSA",
        "hbd": "Metadata_HBD",
        "hba": "Metadata_HBA",
    }
    properties = {
        name: {
            "median": float(df[column].median()),
            "denominator": int(df[column].notna().sum()),
        }
        for name, column in property_columns.items()
    }
    coverage = {
        **coverage,
        "family_definitions": {
            "chembl": "at least one UniProt-mapped ChEMBL target",
            "motive": "at least one target gene from any MOTIVE database",
            "repurposing_hub": "a Repurposing Hub target, MOA, or both",
            "chemical_probes": "at least one Chemical Probes Portal target gene",
        },
        "union_definition": (
            "at least one target or MOA annotation across the four provider families; "
            "disease-area and ToxCast annotations excluded"
        ),
    }
    return {
        "population": {
            "dataset": key["dataset"],
            "preprocessing": key["preprocessing"],
            "analyzed_compounds": int(len(df)),
            "source7_profiles_excluded": True,
        },
        "dataset_structure": {
            **dataset_structure,
            "canonical_analyzed_compounds": int(len(df)),
        },
        "clinical_phase": clinical_phase,
        "lipinski_rule_of_5": {
            "denominator": int(valid.sum()),
            "denominator_definition": "analyzed compounds with a valid molecular structure",
            "violation_counts": {str(value): int(violation_counts[value]) for value in range(5)},
        },
        "target_moa_annotations": coverage,
        "median_properties": properties,
    }


def _publication_file(path: str, caption: str, kind: str = "figure") -> dict:
    """Return one standard summary file record with a package-relative path."""
    return {"path": path, "kind": kind, "caption": caption}


def write_figure1_publication_package(
    df: pd.DataFrame,
    coverage: dict,
    key: dict,
    dataset_structure_asset: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    """Write the complete compound-collection package without a separate figure notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_asset = output_dir / "dataset_structure.png"
    scope = load_dataset_structure_scope()
    roster = (
        "Chemical characterization covers the canonical analyzed roster of "
        f"{len(df):,} compounds; source 7 profiles are excluded. "
    )
    # Built before compositing, because the schematic carries its caption the same way the
    # rendered figures do.
    dataset_structure_caption = (
        "The JUMP Cell Painting collection comprises cpg0000, cpg0001, cpg0002, and cpg0016. "
        "Most chemical perturbations are in cpg0016-jump, generated across "
        f"{scope['partner_sites']} partner sites together with previously published genetic "
        "perturbations; Broad (source 4, S4) generated the smaller pilot datasets. Source numbers "
        "identify sites, except S7 and S13 denote the same site. CRISPR data were generated at S13 "
        "and ORF data at S4. Compound replicates were typically generated at two to four additional "
        "sites. Arrows show compound exchange; circle sizes and overlaps are schematic. The cpg0012 "
        f"overlap denotes {scope['cpg0012_reference_compounds']:,} bioactive compounds generated at "
        "Broad (Bray et al., 2017) and reprocessed with the JUMP pipeline."
    )
    scope["render"] = composite_publication_schematic_on_white(
        dataset_structure_asset,
        target_asset,
        dataset_structure_caption,
        dpi,
    )
    numbers = figure1_publication_numbers(df, coverage, key, scope)
    dataset_structure = numbers["dataset_structure"]
    clinical = numbers["clinical_phase"]
    lipinski = numbers["lipinski_rule_of_5"]
    annotations = numbers["target_moa_annotations"]
    properties = numbers["median_properties"]
    annotation_union = annotations["target_or_moa_any"]
    chemical_space_caption = (
        roster + "Left, clinical phase among the "
        f"{clinical['denominator']:,} compounds assigned one displayed stage; "
        f"{clinical['excluded_count']:,} of {clinical['non_null_count']:,} compounds with a phase "
        "annotation had mixed-stage or Withdrawn values and are not shown. Middle, Lipinski "
        f"Rule-of-5 violations among {lipinski['denominator']:,} compounds with a valid structure; "
        f"{lipinski['violation_counts']['0']:,} had no violations. Right, non-mutually-exclusive "
        "target or MOA coverage from ChEMBL/UniProt, MOTIVE, Repurposing Hub, and Chemical Probes "
        f"Portal; {annotation_union['count']:,} of {annotations['denominator']:,} compounds "
        f"({annotation_union['percent']:.1f}%) had at least one annotation. Bottom, median quantitative estimate "
        f"of drug-likeness (QED; Bickerton et al., 2012) {properties['qed']['median']:.2f}, molecular weight "
        f"{properties['molecular_weight_da']['median']:.0f} Da, "
        f"LogP {properties['logp']['median']:.2f}, topological polar surface area "
        f"{properties['tpsa_angstrom2']['median']:.1f} A^2, hydrogen-bond donors "
        f"{properties['hbd']['median']:.0f}, and hydrogen-bond acceptors "
        f"{properties['hba']['median']:.0f}. Medians use all non-missing analyzed values."
    )

    with plt.rc_context({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42}):
        plot_chemical_space_publication_panels(
            df,
            output_dir,
            caption=chemical_space_caption,
            dpi=dpi,
            publication_annotation_coverage=coverage,
        )
    write_legend(
        output_dir / "legend.md",
        figure_id=FIGURE1_ID,
        title=(
            "The JUMP consortium generated Cell Painting data for "
            f"{dataset_structure['full_collection_compounds']:,} compounds in cpg0016-jump"
        ),
        sections=[
            ("dataset_structure.png", "Dataset structure", dataset_structure_caption),
            ("chemical_space.png", "Clinical phase, chemistry, and annotation coverage", chemical_space_caption),
        ],
    )
    write_summary(
        output_dir / "summary.json",
        description=(
            "Dataset-structure schematic and corrected chemical characterization over "
            "the canonical analyzed compound roster."
        ),
        numbers=numbers,
        files=[
            _publication_file("dataset_structure.png", "versioned dataset-structure schematic"),
            _publication_file("chemical_space.png", "clinical, chemistry, and annotation panels"),
            _publication_file("chemical_space.pdf", "clinical, chemistry, and annotation panels"),
            _publication_file("legend.md", "Figure legend", kind="data"),
        ],
    )


def write_supplement_publication_package(
    df: pd.DataFrame,
    adata_umap: sc.AnnData,
    embedding: dict,
    key: dict,
    structure_umap_input: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    """Write the supplementary chemical-properties package from the frozen embedding."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_publication_embedding_coverage(df, adata_umap)
    numbers, top_scaffolds = publication_supplement_numbers(df, adata_umap, embedding)
    numbers["population"].update({"dataset": key["dataset"], "preprocessing": key["preprocessing"]})
    numbers["structure_umap"]["source"] = structure_umap_input.as_posix()
    top_scaffolds.to_csv(output_dir / "top_scaffolds.csv", index=False)

    property_space = numbers["property_space"]
    distributions = numbers["property_distributions"]
    medians = numbers["property_medians"]
    scaffolds = numbers["scaffolds"]
    pains = numbers["pains"]
    umap = numbers["structure_umap"]
    caption = (
        "Top left, molecular weight (MW) and LogP density for "
        f"{property_space['included_count']:,} of {property_space['candidate_count']:,} compounds; "
        f"{property_space['excluded_count']:,} fall outside the displayed MW "
        f"{property_space['bounds']['mw'][0]:g}-{property_space['bounds']['mw'][1]:g} Da and LogP "
        f"{property_space['bounds']['logp'][0]:g}-{property_space['bounds']['logp'][1]:g} ranges. "
        f"Dashed lines mark MW {PUBLICATION_LIPINSKI_MW_CUTOFF_DA:g} Da and LogP "
        f"{PUBLICATION_LIPINSKI_LOGP_CUTOFF:g}. Top right, fixed-range distributions of MW, LogP, "
        "topological polar surface area (TPSA), and quantitative estimate of drug-likeness "
        "(QED; Bickerton et al., 2012); medians from all non-missing values are "
        f"{medians['mw']['median']:.2f}, {medians['logp']['median']:.2f}, "
        f"{medians['tpsa']['median']:.2f}, and {medians['qed']['median']:.2f}, respectively. "
        "Displayed/outside-range counts are MW "
        f"{distributions['mw']['included_count']:,}/{distributions['mw']['excluded_count']:,}, LogP "
        f"{distributions['logp']['included_count']:,}/{distributions['logp']['excluded_count']:,}, TPSA "
        f"{distributions['tpsa']['included_count']:,}/{distributions['tpsa']['excluded_count']:,}, and QED "
        f"{distributions['qed']['included_count']:,}/{distributions['qed']['excluded_count']:,}. "
        f"Bottom left, the {scaffolds['displayed']} most frequent of {scaffolds['unique']:,} non-empty "
        f"Bemis-Murcko scaffolds; {scaffolds['singletons']:,} ({scaffolds['singleton_percent']:.1f}%) "
        "occur once. Bottom center, structural alerts for pan-assay interference compounds "
        "(PAINS; Baell and Holloway, 2010) among "
        f"{pains['assessable']:,} compounds with valid structures: {pains['flagged']:,} "
        f"({pains['flagged_percent_of_assessable']:.1f}%) flagged and {pains['clean']:,} unflagged; "
        f"{pains['unassessable']:,} of {pains['total']:,} compounds were unassessable. Bottom right, "
        "UMAP of the frozen Morgan-fingerprint embedding colored by log compound density; "
        f"{umap['included_count']:,} compounds are shown and {umap['excluded_count']:,} outside the "
        f"{PUBLICATION_UMAP_IQR_K:g}-IQR viewport are retained in summary counts."
    )
    with plt.rc_context({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42}):
        plot_publication_supplement(df, adata_umap, numbers, top_scaffolds, caption, output_dir, dpi)
    write_legend(
        output_dir / "legend.md",
        figure_id=SUPPLEMENT_ID,
        title=(
            "Extended chemical characterization of the "
            f"{numbers['population']['analyzed_compounds']:,} analyzed JUMP compounds"
        ),
        sections=[("supp_chemical_properties.png", "Extended chemical characterization", caption)],
    )
    write_summary(
        output_dir / "summary.json",
        description=(
            "Supplementary chemical-property characterization over the canonical analyzed roster, "
            "using the existing full-collection structure embedding without recomputing UMAP."
        ),
        numbers=numbers,
        files=[
            _publication_file("supp_chemical_properties.png", "supplementary chemical-properties panels"),
            _publication_file("supp_chemical_properties.pdf", "supplementary chemical-properties panels"),
            _publication_file("top_scaffolds.csv", "top 30 Bemis-Murcko scaffolds", kind="table"),
            _publication_file("legend.md", "Figure legend", kind="data"),
        ],
    )


# =============================================================================
# Summary Statistics
# =============================================================================


def generate_summary(df: pd.DataFrame, output_dir: Path, dataset_filter: str = "all") -> dict:
    """Generate summary statistics for LLM consumption.

    Returns a dict with all key statistics from the chemical space analysis.
    Also saves to JSON file.
    """
    n_total = len(df)

    # Annotation coverage - the 9 canonical target-consistency sources (see configs/snakemake/copairs_runs.yaml)
    annotation_cols = {
        "Metadata_repurposing_target": "repurposing_hub_target",
        "Metadata_repurposing_moa": "repurposing_hub_moa",
        "Metadata_Uniprot_target": "chembl",
        "Metadata_chmprb_target_genes": "chemical_probes",
        "Metadata_repurposing_disease_area": "disease_area",
        "Metadata_motive_gene_biokg": "motive_biokg",
        "Metadata_motive_gene_opentargets": "motive_opentargets",
        "Metadata_motive_gene_primekg": "motive_primekg",
        "Metadata_toxcast_annotated": "toxcast_assay",
    }
    annotations = {}
    for col, name in annotation_cols.items():
        if col in df.columns:
            n = int(df[col].notna().sum())
            annotations[name] = {"count": n, "percent": round(100 * n / n_total, 1)}

    # Union: compounds annotated by ANY of the 9 sources (the canonical "annotated" definition)
    present_cols = [c for c in annotation_cols if c in df.columns]
    n_any = int(df[present_cols].notna().any(axis=1).sum()) if present_cols else 0
    annotated_any = {
        "count": n_any,
        "percent": round(100 * n_any / n_total, 1),
        "sources": list(annotation_cols.values()),
    }

    # Clinical phases
    phase_col = "Metadata_repurposing_clinical_phase"
    clinical_phases = {}
    if phase_col in df.columns:
        for phase in ["Preclinical", "Phase 1", "Phase 2", "Phase 3", "Launched"]:
            n = int((df[phase_col] == phase).sum())
            clinical_phases[phase] = n

    # Property distributions
    properties = {}
    prop_cols = [
        ("Metadata_MW", "molecular_weight_da"),
        ("Metadata_LogP", "logp"),
        ("Metadata_TPSA", "tpsa_angstrom2"),
        ("Metadata_HBD", "hbd"),
        ("Metadata_HBA", "hba"),
        ("Metadata_RotatableBonds", "rotatable_bonds"),
        ("Metadata_NumRings", "num_rings"),
        ("Metadata_QED", "qed_drug_likeness"),
    ]
    for col, name in prop_cols:
        if col in df.columns:
            data = df[col].dropna()
            properties[name] = {
                "median": round(float(data.median()), 2),
                "mean": round(float(data.mean()), 2),
                "std": round(float(data.std()), 2),
                "min": round(float(data.min()), 2),
                "max": round(float(data.max()), 2),
                "q25": round(float(data.quantile(0.25)), 2),
                "q75": round(float(data.quantile(0.75)), 2),
            }

    # Lipinski violations
    lipinski = {}
    if "Metadata_Lipinski_Violations" in df.columns:
        counts = df["Metadata_Lipinski_Violations"].value_counts().sort_index()
        for violations, count in counts.items():
            lipinski[f"{int(violations)}_violations"] = {
                "count": int(count),
                "percent": round(100 * count / n_total, 1),
            }
        lipinski["rule_of_5_compliant"] = {
            "count": int((df["Metadata_Lipinski_Violations"] == 0).sum()),
            "percent": round(100 * (df["Metadata_Lipinski_Violations"] == 0).sum() / n_total, 1),
        }

    # Scaffold diversity
    scaffolds = {}
    if "Metadata_MurckoScaffold" in df.columns:
        scaffold_col = df["Metadata_MurckoScaffold"]
        n_acyclic = int(scaffold_col.isna().sum() + (scaffold_col == "").sum())
        valid_scaffolds = scaffold_col[scaffold_col.notna() & (scaffold_col != "")]
        scaffold_counts = valid_scaffolds.value_counts()
        n_unique = len(scaffold_counts)
        n_singletons = int((scaffold_counts == 1).sum())

        scaffolds = {
            "unique_scaffolds": n_unique,
            "singleton_scaffolds": n_singletons,
            "singleton_percent": round(100 * n_singletons / n_unique, 1) if n_unique > 0 else 0,
            "acyclic_molecules": n_acyclic,
            "diversity_ratio": round(n_unique / n_total, 3),
            "top_5_scaffolds": [{"smiles": str(s), "count": int(c)} for s, c in scaffold_counts.head(5).items()],
        }

    # PAINS
    pains = {}
    if "Metadata_HasPAINS" in df.columns:
        n_pains = int(df["Metadata_HasPAINS"].sum())
        pains = {
            "pains_flagged": n_pains,
            "pains_percent": round(100 * n_pains / n_total, 1),
            "clean": n_total - n_pains,
        }

    # Standardized summary envelope: description (intent) + numbers + index of generated files
    numbers = {
        "dataset_filter": dataset_filter,
        "total_compounds": n_total,
        "valid_molecules": int(df["Metadata_ValidMol"].sum()) if "Metadata_ValidMol" in df.columns else n_total,
        "annotation_coverage": annotations,
        "annotated_any": annotated_any,
        "clinical_development": clinical_phases,
        "physicochemical_properties": properties,
        "lipinski_rule_of_5": lipinski,
        "scaffold_diversity": scaffolds,
        "pains_structural_alerts": pains,
    }
    description = (
        "Chemical-space characterization of the JUMP compound collection: physicochemical property "
        "distributions, Lipinski Ro5 compliance, Bemis-Murcko scaffold diversity, PAINS structural alerts, "
        "clinical-phase breakdown, and annotation coverage across the nine target-consistency sources "
        f"(annotated_any). Population: {dataset_filter}."
    )
    output_file = output_dir / "chemical_space_summary.json"
    summary = write_summary(output_file, description=description, numbers=numbers, files=index_files(output_dir))
    logger.info(f"Saved summary statistics to {output_file}")

    return summary


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (required and used exactly in publication-only mode)",
    ),
    analyzed_only: bool = typer.Option(
        False,
        "--analyzed-only",
        help="Filter to analyzed compounds (compound_no_source7_rsc from copairs results)",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help="Skip structure UMAP computation (faster)",
    ),
    subsample_umap: float = typer.Option(
        1.0,
        "--subsample-umap",
        help="Fraction of compounds for UMAP (0-1). Use 0.1 for 10% subsample.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
    publication_only: bool = typer.Option(
        False,
        "--publication-only",
        help="Render one corrected publication package without recomputing UMAP",
    ),
    publication_figure: str | None = typer.Option(
        None,
        "--publication-figure",
        help="Publication package: figure_compound_collection or figure_suppl_chemical_properties",
    ),
    structure_umap_input: Path | None = typer.Option(
        None,
        "--structure-umap-input",
        exists=True,
        dir_okay=False,
        help="Frozen full-collection structure_umap.h5ad (required for the supplementary package)",
    ),
    dataset_structure_asset: Path | None = typer.Option(
        None,
        "--dataset-structure-asset",
        exists=True,
        dir_okay=False,
        help="Versioned manuscript schematic (required for the Figure 1 package)",
    ),
) -> None:
    """Chemical space characterization for JUMP compounds.

    Properties are precomputed by compound_properties.py and loaded from the DB.
    Run compound_properties first if the compound_properties table is missing.
    """
    if publication_only:
        if publication_figure not in PUBLICATION_FIGURES:
            raise typer.BadParameter(
                "--publication-figure must be figure_compound_collection or figure_suppl_chemical_properties"
            )
        if output_dir is None:
            raise typer.BadParameter("--output is required in publication-only mode")
        if publication_figure == "figure_compound_collection" and dataset_structure_asset is None:
            raise typer.BadParameter("--dataset-structure-asset is required for figure_compound_collection")
        if publication_figure == "figure_suppl_chemical_properties" and structure_umap_input is None:
            raise typer.BadParameter("--structure-umap-input is required for figure_suppl_chemical_properties")
        if not COPAIRS_DB.exists():
            logger.error(f"Copairs database not found: {COPAIRS_DB}")
            raise typer.Exit(1)
        if not METADATA_DB.exists():
            logger.error(f"Metadata database not found: {METADATA_DB}")
            raise typer.Exit(1)

        key = get_canonical_key()
        canonical_ids = get_analyzed_compound_ids(key["dataset"], key["preprocessing"])
        if any(compound_id is None for compound_id in canonical_ids):
            raise ValueError("Canonical analyzed compound IDs contain null values")
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("Canonical analyzed compound IDs are not unique")
        canonical_ids = sorted(canonical_ids)
        logger.info(f"Publication roster: {len(canonical_ids):,} compounds ({key['dataset']} / {key['preprocessing']})")

        connection = duckdb.connect(str(METADATA_DB), read_only=True)
        try:
            compounds = validate_and_order_publication_compounds(
                load_compounds(connection, analyzed_ids=canonical_ids),
                canonical_ids,
            )
            if publication_figure == "figure_compound_collection":
                assert dataset_structure_asset is not None
                coverage = target_moa_coverage(connection, canonical_ids)
                write_figure1_publication_package(
                    compounds,
                    coverage,
                    key,
                    dataset_structure_asset,
                    output_dir,
                    dpi,
                )
            else:
                assert structure_umap_input is not None
                adata_umap, embedding = subset_existing_structure_umap(
                    structure_umap_input,
                    canonical_ids,
                )
                write_supplement_publication_package(
                    compounds,
                    adata_umap,
                    embedding,
                    key,
                    structure_umap_input,
                    output_dir,
                    dpi,
                )
        finally:
            connection.close()
        logger.success(f"Wrote {publication_figure} publication package to {output_dir}")
        return

    if publication_figure is not None:
        raise typer.BadParameter("--publication-figure requires --publication-only")
    analyzed_ids = None
    analyzed_dataset = None
    if analyzed_only:
        if not COPAIRS_DB.exists():
            logger.error(f"Copairs database not found: {COPAIRS_DB}")
            raise typer.Exit(1)
        key = get_canonical_key()
        analyzed_dataset = key["dataset"]
        analyzed_ids = get_analyzed_compound_ids(key["dataset"], key["preprocessing"])
        logger.info(
            f"Filtering to {len(analyzed_ids):,} analyzed compounds ({key['dataset']} / {key['preprocessing']})"
        )
    output_dir = resolve_analysis_output_dir(output_dir, analyzed_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not METADATA_DB.exists():
        logger.error(f"Metadata database not found: {METADATA_DB}")
        raise typer.Exit(1)

    con = duckdb.connect(str(METADATA_DB), read_only=True)

    try:
        # Load data (includes precomputed properties from compound_properties table)
        df = load_compounds(con, analyzed_ids=analyzed_ids)

        # Generate individual plots (standalone mode: ax/axes=None, saves to output_dir)
        plot_annotation_coverage(df, output_dir=output_dir, dpi=dpi)
        plot_clinical_phase(df, output_dir=output_dir, dpi=dpi)
        plot_property_distributions(df, output_dir=output_dir, dpi=dpi)  # axes=None for standalone
        plot_lipinski_violations(df, output_dir=output_dir, dpi=dpi)
        plot_scaffold_diversity(df, output_dir=output_dir, dpi=dpi)
        plot_property_space(df, output_dir=output_dir, dpi=dpi)
        plot_pains_summary(df, output_dir=output_dir, dpi=dpi)

        # Compute structure UMAP (before combined figure so it can be included)
        adata_umap = None
        if not skip_umap:
            adata_umap = compute_structure_umap(df, output_dir, subsample_frac=subsample_umap, seed=seed, dpi=dpi)

            # Property UMAP (complementary view using z-scored descriptors)
            compute_property_umap(df, output_dir, seed=seed, dpi=dpi)

        # Generate combined publication figure (includes UMAP if computed)
        plot_combined_figure(df, output_dir, adata_umap=adata_umap, dpi=dpi)

        # Separate main chemical-space and supplementary property views
        plot_chemical_space_publication_panels(df, output_dir, dpi=dpi)
        plot_supp_chemical_properties(df, output_dir, adata_umap=adata_umap, dpi=dpi)

        # Generate summary JSON for LLM consumption
        dataset_filter = analyzed_dataset if analyzed_only else "all"
        generate_summary(df, output_dir, dataset_filter=dataset_filter)

        logger.success(f"Chemical space analysis complete: {output_dir}")

    finally:
        con.close()


if __name__ == "__main__":
    app()
