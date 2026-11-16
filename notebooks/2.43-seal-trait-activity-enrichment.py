"""Trait → phenotypic activity / consistency enrichment.

For each compound trait family (RDKit named functional groups, BRICS fragments,
Murcko scaffolds, target / MoA classes), tests whether compounds carrying the
trait are over- or under-represented among phenotypically active and
phenotypically consistent compounds (Fisher's exact, BH FDR within family).

Analogous to MorphMap's gene-group enrichment, but for chemistry.

Usage:
    pixi run -e cheminformatics python notebooks/2.43-seal-trait-activity-enrichment.py
    pixi run -e cheminformatics python notebooks/2.43-seal-trait-activity-enrichment.py --activity-threshold 0.05

Outputs: data/processed/trait-activity-enrichment/{dataset}/
    - enrichment_activity.csv, enrichment_consistency.csv
    - structural_class_activity_nmap.csv (plot-ready structural-class contract)
    - structural_class_activity_distributions.parquet (displayed compound-level boxes)
    - structural_classes_all_significant_5pct_fdr.csv (complete figure-screen results)
    - consistency_property_effects.csv (paired contextual-model effects)
    - consistency_group_properties.csv (group-level model audit companion)
    - volcano_activity.{png,pdf}, volcano_consistency.{png,pdf}
    - top_traits_bar_activity.{png,pdf}, top_traits_bar_consistency.{png,pdf}
    - summary.json
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import typer
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Fragments
from scipy.stats import fisher_exact, norm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from tqdm.auto import tqdm

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, METADATA_DB
from jump_production.queries import (
    query_activity_results,
    query_consistency_results,
    query_consistency_scores,
)
from jump_production.summaries import index_files, write_summary

RDLogger.DisableLog("rdApp.*")
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = PROCESSED_DATA_DIR / "trait-activity-enrichment"

ACTIVITY_P_THRESHOLD = 0.10  # paper-canonical FDR cutoff for activity
CONSISTENCY_P_THRESHOLD = 0.05  # FDR cutoff for target-group consistency
CONSISTENCY_EFFECT_FDR_THRESHOLD = 0.05
CONSISTENCY_DISPLAY_SELECTION = (
    "Show both scopes for a source-trait pair when either scope passes global "
    f"{100 * CONSISTENCY_EFFECT_FDR_THRESHOLD:g}% FDR"
)

# BRICS fragment size limits (mirrors 2.40)
BRICS_MIN_ATOMS = 2
BRICS_MAX_ATOMS = 18

# Minimum compounds per trait by family (avoid testing tiny groups)
MIN_N = {
    "rdkit_fg": 10,
    "brics": 20,
    "scaffold": 5,
    "target_class": 5,
}

# The selected structural panel uses the figure's stricter display contract:
# each level and reference must have at least 20 compounds and every cell in
# the 2 x 2 activity table must contain at least five compounds.  Recompute BH
# after applying those eligibility checks so the plot-ready q-values describe
# exactly the family of hypotheses represented by this contract.
STRUCTURAL_MIN_MEMBERS = 20
STRUCTURAL_MIN_CONTINGENCY_CELL = 5
STRUCTURAL_FDR_THRESHOLD = 0.05
STRUCTURAL_FIGURE_MAX_PER_DIRECTION = 5

# group_type → metadata column for the per-compound target join.
# Each compound is marked consistent if ANY of its annotated values has
# corrected_p_value < CONSISTENCY_P_THRESHOLD in consistency_results.
GROUP_TYPE_COLUMNS = {
    "repurposing": "Metadata_repurposing_target",
    "uniprot": "Metadata_Uniprot_target",
    "moa": "Metadata_repurposing_moa",
}

STRUCTURAL_FIGURE_FAMILIES = (
    (
        "rdkit_fg",
        "chem.rdkit_functional_group",
        "RDKit functional group",
        "rdkit_functional_group",
        "RDKit functional groups",
        "named",
    ),
    (
        "brics",
        "chem.brics_fragment",
        "BRICS fragment",
        "brics_fragment",
        "BRICS fragments",
        "structure",
    ),
    (
        "scaffold",
        "chem.murcko_scaffold",
        "Murcko scaffold",
        "intrinsic_chemistry",
        "Murcko scaffolds",
        "structure",
    ),
)

CONSISTENCY_SOURCE_SPECS = (
    ("repurposing", "Repurposing Hub targets", "target", "Metadata_repurposing_target"),
    ("uniprot", "UniProt targets", "target", "Metadata_Uniprot_target"),
    ("chemical_probes", "Chemical Probes targets", "target", "Metadata_chmprb_target_genes"),
    ("moa", "Repurposing Hub MOA", "MOA", "Metadata_repurposing_moa"),
    (
        "disease_area",
        "Repurposing Hub disease area",
        "disease area",
        "Metadata_repurposing_disease_area",
    ),
    ("motive_biokg", "MOTIVE BioKG targets", "target", "Metadata_motive_gene_biokg"),
    (
        "motive_opentargets",
        "MOTIVE OpenTargets targets",
        "target",
        "Metadata_motive_gene_opentargets",
    ),
    ("motive_primekg", "MOTIVE PrimeKG targets", "target", "Metadata_motive_gene_primekg"),
    ("toxcast_assay", "ToxCast active assays", "assay", "Metadata_txcst_active_assays"),
)

CONSISTENCY_NUMERIC_TRAITS = (
    ("chem.logp", "logp", "Calculated LogP", "continuous", ""),
    ("chem.molecular_weight", "molecular_weight", "Molecular weight", "continuous", "Da"),
    ("chem.rotatable_bonds", "rotatable_bonds", "Rotatable bonds", "count", "count"),
    ("chem.qed", "qed", "Quantitative estimate of drug-likeness", "continuous", "score"),
    ("chem.tpsa", "tpsa", "Topological polar surface area", "continuous", "angstrom^2"),
    ("chem.fraction_csp3", "fraction_csp3", "Fraction Csp3", "continuous", "fraction"),
    ("chem.has_pains", "has_pains", "PAINS structural alert", "boolean", ""),
    ("chem.lipinski_violations", "lipinski_violations", "Lipinski violations", "ordinal", "count"),
    (
        "cell_painting.normalized_cell_density",
        "normalized_cell_density",
        "Normalized post-treatment Cell Painting cell density",
        "continuous",
        "source-adjusted plate z-score",
    ),
)

CONSISTENCY_STAGE_LEVELS = {
    "derived.clinical_phase.phase_1": "Phase 1",
    "derived.clinical_phase.phase_2": "Phase 2",
    "derived.clinical_phase.phase_3": "Phase 3",
    "derived.clinical_phase.launched": "Launched",
}
CONSISTENCY_TRAIT_IDS = [
    *(trait_id for trait_id, _, _, _, _ in CONSISTENCY_NUMERIC_TRAITS),
    *CONSISTENCY_STAGE_LEVELS,
]
CONSISTENCY_TRAIT_FIELDS = {
    **{
        trait_id: {"trait_label": label, "value_type": value_type, "unit": unit}
        for trait_id, _, label, value_type, unit in CONSISTENCY_NUMERIC_TRAITS
    },
    **{
        trait_id: {"trait_label": stage, "value_type": "boolean", "unit": "indicator"}
        for trait_id, stage in CONSISTENCY_STAGE_LEVELS.items()
    },
}
MIN_CONSISTENCY_GROUP_MEMBERS = 5
MIN_CONSISTENCY_CLUSTERS = 30
CONSISTENCY_COVARIANCE = "two-way cluster robust by compound and annotation group"

# Pair (name, callable). Names matter because Fragments' callables all expose
# __name__ == "<lambda>", so we must keep the attribute name explicitly.
RDKIT_FRAGMENT_FNS: list[tuple[str, callable]] = [
    (n, getattr(Fragments, n)) for n in dir(Fragments) if n.startswith("fr_")
]


def _haldane_or_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Haldane-Anscombe odds ratio with a Woolf 95% confidence interval."""
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = a_ * d_ / (b_ * c_)
    log_or = np.log(odds_ratio)
    standard_error = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    z_value = float(norm.ppf(0.975))
    return (
        float(odds_ratio),
        float(np.exp(log_or - z_value * standard_error)),
        float(np.exp(log_or + z_value * standard_error)),
    )


# =============================================================================
# Loaders
# =============================================================================


def load_activity_flags(
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
    threshold: float,
) -> pd.DataFrame:
    """Per-compound activity flag from copairs results."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        df = query_activity_results(
            con,
            dataset,
            preprocessing=preprocessing,
            filter_name=filter_name,
            activity_params=activity_params,
            columns=columns,
        )
    finally:
        con.close()
    df = df.rename(columns={"Metadata_JCP2022": "JCP2022"})[
        ["JCP2022", "mean_normalized_average_precision", "corrected_p_value"]
    ]
    df["is_active"] = df["corrected_p_value"] < threshold
    logger.info(
        f"Activity: {len(df):,} compounds, "
        f"{df['is_active'].sum():,} active ({df['is_active'].mean():.1%}) at FDR<{threshold}"
    )
    return df[["JCP2022", "mean_normalized_average_precision", "is_active"]]


def load_consistency_flags(
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    distance: str,
    threshold: float,
) -> pd.DataFrame:
    """Per-compound consistency flag.

    A compound is marked `is_consistent_{group_type}` if at least one of its
    annotated values in `GROUP_TYPE_COLUMNS[group_type]` matches a consistent
    group (corrected_p_value < threshold) in consistency_results. The union
    `is_consistent_any` is True if any group_type marks it.

    Background pool = compounds with at least one annotation in any of the
    tracked group_types (otherwise consistency is undefined).
    """
    cop = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    meta = duckdb.connect(str(METADATA_DB), read_only=True)

    consistent_per_gt: dict[str, set[str]] = {}
    annotated_per_gt: dict[str, set[str]] = {}

    try:
        for group_type, col in GROUP_TYPE_COLUMNS.items():
            sig_groups = (
                cop.execute(
                    """
                SELECT DISTINCT group_value
                FROM consistency_results
                WHERE _dataset = ?
                  AND _columns = ?
                  AND _preprocessing = ?
                  AND _filter = ?
                  AND _group_type = ?
                  AND _distance = ?
                  AND corrected_p_value < ?
                """,
                    [
                        dataset,
                        columns,
                        preprocessing,
                        filter_name,
                        group_type,
                        distance,
                        threshold,
                    ],
                )
                .df()["group_value"]
                .tolist()
            )
            sig_set = set(sig_groups)
            logger.info(f"  consistency[{group_type}]: {len(sig_set):,} significant groups")

            ann = meta.execute(
                f"SELECT Metadata_JCP2022 AS JCP2022, {col} AS group_values "
                "FROM compound_metadata "
                f"WHERE {col} IS NOT NULL AND {col} != ''"
            ).df()
            consistent: set[str] = set()
            annotated: set[str] = set(ann["JCP2022"].tolist())
            for jcp, vals in zip(ann["JCP2022"], ann["group_values"]):
                pieces = [v.strip() for v in str(vals).split("|") if v.strip()]
                if any(p in sig_set for p in pieces):
                    consistent.add(jcp)
            consistent_per_gt[group_type] = consistent
            annotated_per_gt[group_type] = annotated
            logger.info(f"  consistency[{group_type}]: {len(annotated):,} annotated, {len(consistent):,} consistent")
    finally:
        cop.close()
        meta.close()

    # Background: compounds with any annotation in tracked group_types
    all_annotated = sorted(set().union(*annotated_per_gt.values()))
    rows = []
    for jcp in all_annotated:
        row: dict[str, object] = {"JCP2022": jcp}
        any_flag = False
        for gt in GROUP_TYPE_COLUMNS:
            in_annotated = jcp in annotated_per_gt[gt]
            in_consistent = jcp in consistent_per_gt[gt]
            row[f"annotated_{gt}"] = in_annotated
            row[f"is_consistent_{gt}"] = in_consistent
            if in_consistent:
                any_flag = True
        row["is_consistent_any"] = any_flag
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info(
        f"Consistency: {len(df):,} annotated compounds, {df['is_consistent_any'].sum():,} consistent in ≥1 group_type"
    )
    return df


def load_smiles_and_scaffolds(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Return compounds with a valid structure and their Murcko scaffold."""
    df = con.execute(
        """
        SELECT Metadata_JCP2022 AS JCP2022,
               Metadata_SMILES AS SMILES,
               Metadata_MurckoScaffold AS scaffold
        FROM compound_metadata
        WHERE Metadata_ValidMol
          AND Metadata_SMILES IS NOT NULL
        """
    ).df()
    logger.info(
        f"Compounds with a valid structure: {len(df):,} ({df['scaffold'].notna().sum():,} with a cyclic scaffold)"
    )
    return df


# =============================================================================
# Trait map builders (each returns dict[trait_name -> list[JCP2022]])
# =============================================================================


def build_rdkit_fragment_map(smiles_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each of ~85 RDKit named functional groups → list of JCP2022 compounds
    containing at least one occurrence of that group."""
    logger.info(f"Computing RDKit functional groups ({len(RDKIT_FRAGMENT_FNS)} groups)...")
    trait_map: dict[str, list[str]] = defaultdict(list)
    for jcp, smi in tqdm(zip(smiles_df["JCP2022"], smiles_df["SMILES"]), total=len(smiles_df), desc="RDKit fragments"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for name, fn in RDKIT_FRAGMENT_FNS:
            try:
                if fn(mol) > 0:
                    trait_map[name].append(jcp)
            except Exception:
                continue
    return dict(trait_map)


def build_brics_map(smiles_df: pd.DataFrame) -> dict[str, list[str]]:
    """Enumerate BRICS fragments → list of compounds containing them."""
    logger.info("Enumerating BRICS fragments...")
    trait_map: dict[str, list[str]] = defaultdict(list)
    for jcp, smi in tqdm(zip(smiles_df["JCP2022"], smiles_df["SMILES"]), total=len(smiles_df), desc="BRICS"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=False)
        except Exception:
            continue
        for frag in frags:
            # Strip BRICS attachment indices for atom count
            clean = frag
            for i in range(20):
                clean = clean.replace(f"[{i}*]", "[*]")
            cm = Chem.MolFromSmiles(clean)
            if cm is None:
                continue
            n = cm.GetNumHeavyAtoms()
            if BRICS_MIN_ATOMS <= n <= BRICS_MAX_ATOMS:
                trait_map[frag].append(jcp)
    logger.info(f"BRICS fragments: {len(trait_map):,} unique")
    return dict(trait_map)


def build_scaffold_map(smiles_df: pd.DataFrame) -> dict[str, list[str]]:
    """Map Murcko scaffolds to compounds, including valid acyclic molecules."""
    trait_map: dict[str, list[str]] = defaultdict(list)
    for jcp, scaf in zip(smiles_df["JCP2022"], smiles_df["scaffold"]):
        if isinstance(scaf, str) and scaf:
            trait_map[scaf].append(jcp)
        else:
            trait_map["acyclic"].append(jcp)
    logger.info(f"Murcko scaffolds: {len(trait_map):,} unique")
    return dict(trait_map)


def build_target_map(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Pipe-split target / MoA classes → list of compounds. Trait names are
    prefixed by group_type to avoid collisions across columns.

    Each trait carries the complete annotated population for its source field so
    missing annotations are excluded rather than treated as negative evidence.
    """
    trait_map: dict[str, list[str]] = defaultdict(list)
    eligible_by_trait: dict[str, set[str]] = {}
    for group_type, col in GROUP_TYPE_COLUMNS.items():
        df = con.execute(
            f"SELECT Metadata_JCP2022 AS JCP2022, {col} AS vals "
            "FROM compound_metadata "
            f"WHERE {col} IS NOT NULL AND {col} != ''"
        ).df()
        annotated = set(df["JCP2022"])
        for jcp, vals in zip(df["JCP2022"], df["vals"]):
            for v in str(vals).split("|"):
                v = v.strip()
                if v:
                    trait = f"{group_type}:{v}"
                    trait_map[trait].append(jcp)
                    eligible_by_trait[trait] = annotated
    # Deduplicate within each (multi-listed compound shouldn't double-count)
    for k, v in trait_map.items():
        trait_map[k] = sorted(set(v))
    logger.info(f"Target/MoA classes: {len(trait_map):,} unique")
    return dict(trait_map), eligible_by_trait


# =============================================================================
# Generic Fisher enrichment (one row per trait)
# =============================================================================


def compute_trait_enrichment(
    trait_map: dict[str, list[str]],
    flag: pd.Series,
    family: str,
    min_n: int,
    shared_eligible: set[str] | None = None,
    eligible_by_trait: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Two-sided Fisher's exact: P(flag | trait) vs P(flag | ¬trait).

    `flag` is a boolean Series indexed by JCP2022. The eligible background is
    restricted either by one shared family population (valid structures) or a
    per-trait population (compounds annotated by that target source). Traits
    with <min_n members are dropped. Benjamini-Hochberg correction is applied
    across the complete family after all eligible-population tests are built.
    """
    # Dedupe flag in case of duplicate JCP2022 rows
    flag = flag[~flag.index.duplicated(keep="first")]
    activity_background = set(flag.index)
    logger.info(f"[{family}] testing traits with n≥{min_n} in explicit eligible populations")

    background_cache: dict[int | None, tuple[set[str], int, int, float]] = {}

    def background_stats(eligible: set[str] | None) -> tuple[set[str], int, int, float]:
        """Cache a shared or source-specific eligible universe once per sweep."""
        key = None if eligible is None else id(eligible)
        if key not in background_cache:
            background = activity_background if eligible is None else activity_background & eligible
            background_ids = sorted(background)
            n_total = len(background_ids)
            n_flag_total = int(flag.loc[background_ids].sum()) if background_ids else 0
            background_cache[key] = (
                background,
                n_total,
                n_flag_total,
                n_flag_total / n_total if n_total else np.nan,
            )
        return background_cache[key]

    results = []
    for trait, members in trait_map.items():
        eligible = eligible_by_trait[trait] if eligible_by_trait is not None else shared_eligible
        background, n_total, n_flag_total, bg_rate = background_stats(eligible)
        if n_total == 0:
            continue
        in_bg = sorted({m for m in members if m in background})
        n_with = len(in_bg)
        if n_with < min_n:
            continue
        n_flag_with = int(flag.loc[in_bg].sum())
        n_flag_without = n_flag_total - n_flag_with
        n_without = n_total - n_with
        n_noflag_with = n_with - n_flag_with
        n_noflag_without = n_without - n_flag_without

        table = [[n_flag_with, n_noflag_with], [n_flag_without, n_noflag_without]]
        _, p_value = fisher_exact(table, alternative="two-sided")
        odds_ratio, ci_lower, ci_upper = _haldane_or_ci(
            n_flag_with,
            n_noflag_with,
            n_flag_without,
            n_noflag_without,
        )
        # Fold enrichment relative to background rate
        rate = n_flag_with / n_with if n_with else 0.0
        fold = rate / bg_rate if bg_rate > 0 else np.nan

        results.append(
            {
                "family": family,
                "trait": trait,
                "n_eligible": n_total,
                "n_missing": len(activity_background) - n_total,
                "n_with_trait": n_with,
                "n_without_trait": n_without,
                "n_flag_with_trait": n_flag_with,
                "n_flag_without_trait": n_flag_without,
                "n_flag_total": n_flag_total,
                "flag_rate_with": rate,
                "flag_rate_background": bg_rate,
                "fold_enrichment": fold,
                "odds_ratio": odds_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "log2_odds_ratio": np.log2(odds_ratio) if odds_ratio > 0 else np.nan,
                "p_value": p_value,
            }
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df
    _, corrected, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["corrected_p_value"] = corrected
    # tiebreak on trait (unique within a family) so tied p-values get a stable row
    # order across runs - otherwise dict/set iteration under PYTHONHASHSEED shuffles them
    df = df.sort_values(["p_value", "trait"]).reset_index(drop=True)
    n_sig = (df["corrected_p_value"] < 0.05).sum()
    logger.info(f"[{family}] tested {len(df):,} traits; {n_sig:,} FDR<0.05")
    return df


def build_structural_class_activity_nmap(
    activity_enrichment: pd.DataFrame,
    trait_maps: dict[str, dict[str, list[str]]],
    valid_structure: set[str],
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate_07 structural rows from 2.43's canonical tests."""
    required = {
        "family",
        "trait",
        "n_eligible",
        "n_missing",
        "n_with_trait",
        "n_without_trait",
        "n_flag_with_trait",
        "n_flag_without_trait",
        "flag_rate_with",
        "flag_rate_background",
        "odds_ratio",
        "ci_lower",
        "ci_upper",
        "p_value",
        "corrected_p_value",
    }
    missing = sorted(required - set(activity_enrichment.columns))
    if missing:
        raise ValueError(f"Activity enrichment is missing structural figure columns: {missing}")
    activity_required = {"JCP2022", "mean_normalized_average_precision", "is_active"}
    missing = sorted(activity_required - set(activity.columns))
    if missing:
        raise ValueError(f"Activity table is missing structural figure columns: {missing}")
    if activity["JCP2022"].duplicated().any():
        raise ValueError("Activity table must contain one row per compound")
    indexed = activity.set_index("JCP2022")
    valid_ids = set(valid_structure) & set(indexed.index)
    if not valid_ids:
        raise ValueError("Structural figure has no valid-structure activity population")

    selected_families: list[pd.DataFrame] = []
    for family_order, (
        family_key,
        trait_id,
        label,
        family_name,
        family_title,
        display_kind,
    ) in enumerate(STRUCTURAL_FIGURE_FAMILIES):
        if family_key not in trait_maps:
            raise ValueError(f"Structural figure requires trait family {family_key}")
        source = activity_enrichment.loc[activity_enrichment["family"].eq(family_key)].copy()
        source["inactive_with_trait"] = source["n_with_trait"] - source["n_flag_with_trait"]
        source["inactive_without_trait"] = source["n_without_trait"] - source["n_flag_without_trait"]
        source = source.loc[
            source["n_with_trait"].ge(STRUCTURAL_MIN_MEMBERS)
            & source["n_without_trait"].ge(STRUCTURAL_MIN_MEMBERS)
            & source[
                [
                    "n_flag_with_trait",
                    "inactive_with_trait",
                    "n_flag_without_trait",
                    "inactive_without_trait",
                ]
            ]
            .min(axis=1)
            .ge(STRUCTURAL_MIN_CONTINGENCY_CELL)
        ].copy()
        if source.empty:
            raise ValueError(f"Structural figure family {family_key} has no eligible relationships")
        source["figure_fdr_bh"] = multipletests(source["p_value"], method="fdr_bh")[1]
        source = source.loc[source["figure_fdr_bh"].lt(STRUCTURAL_FDR_THRESHOLD)].copy()
        if source.empty:
            raise ValueError(f"Structural figure family {family_key} has no significant relationships")
        source["source_level"] = source["trait"].astype(str)
        if family_key == "rdkit_fg":
            source = source.loc[~source["source_level"].eq("fr_COO2")].copy()
        summaries: list[dict[str, object]] = []
        for record in source.itertuples(index=False):
            member_ids = set(trait_maps[family_key].get(str(record.source_level), [])) & valid_ids
            reference_ids = valid_ids - member_ids
            if len(member_ids) != int(record.n_with_trait) or len(reference_ids) != int(record.n_without_trait):
                raise ValueError(
                    f"Structural counts disagree for {family_key}:{record.source_level}: "
                    f"map={len(member_ids)}/{len(reference_ids)}, "
                    f"enrichment={record.n_with_trait}/{record.n_without_trait}"
                )
            member_nmap = indexed.loc[sorted(member_ids), "mean_normalized_average_precision"]
            reference_nmap = indexed.loc[sorted(reference_ids), "mean_normalized_average_precision"]
            display_level = str(record.source_level)
            if family_key == "rdkit_fg" and display_level == "fr_COO":
                display_level = "fr_COO / COO2"
            if family_key == "scaffold":
                comparator = "Other valid structures without the selected scaffold."
                missing_semantics = "Invalid structures are unknown; valid structures without a ring emit 'acyclic'."
            else:
                comparator = f"Other valid structures without the selected {label.lower()}."
                missing_semantics = (
                    "Invalid structures are unknown; absence within valid structures "
                    "means the selected motif was not found."
                )
            summaries.append(
                {
                    "family_order": family_order,
                    "family_key": trait_id,
                    "trait_id": trait_id,
                    "label": label,
                    "level": display_level,
                    "reference_level": "all other eligible compounds",
                    "family": family_name,
                    "interpretation_class": "mechanistic_candidate",
                    "source_representation": "categorical",
                    "universe_id": "valid_structure",
                    "comparator": comparator,
                    "missing_semantics": missing_semantics,
                    "correction_group": f"{family_name}:categorical",
                    "n_eligible": int(record.n_eligible),
                    "n_missing": int(record.n_missing),
                    "n_exposed": int(record.n_with_trait),
                    "n_reference": int(record.n_without_trait),
                    "active_exposed": int(record.n_flag_with_trait),
                    "inactive_exposed": int(record.n_with_trait - record.n_flag_with_trait),
                    "active_reference": int(record.n_flag_without_trait),
                    "inactive_reference": int(record.n_without_trait - record.n_flag_without_trait),
                    "activity_rate_exposed": float(record.flag_rate_with),
                    "activity_rate_reference": float(record.n_flag_without_trait / record.n_without_trait),
                    "estimate": float(record.odds_ratio),
                    "ci_lower": float(record.ci_lower),
                    "ci_upper": float(record.ci_upper),
                    "p_value": float(record.p_value),
                    "status": "tested",
                    "exclusion_reason": "",
                    "fdr_bh": float(record.figure_fdr_bh),
                    "significant_at_5pct_fdr": True,
                    "source_level": str(record.source_level),
                    "nmap_median": float(member_nmap.median()),
                    "nmap_q25": float(member_nmap.quantile(0.25)),
                    "nmap_q75": float(member_nmap.quantile(0.75)),
                    "reference_nmap_median": float(reference_nmap.median()),
                    "activity_pct": 100 * float(record.flag_rate_with),
                    "delta_pp": 100 * float(record.flag_rate_with - activity["is_active"].mean()),
                    "family_title": family_title,
                    "display_kind": display_kind,
                    "nmap_delta": float(member_nmap.median() - reference_nmap.median()),
                }
            )
        complete = pd.DataFrame(summaries)
        enriched = complete.loc[complete["activity_rate_exposed"].gt(complete["activity_rate_reference"])].nlargest(
            STRUCTURAL_FIGURE_MAX_PER_DIRECTION, "activity_pct"
        )
        depleted = complete.loc[complete["activity_rate_exposed"].lt(complete["activity_rate_reference"])].nsmallest(
            STRUCTURAL_FIGURE_MAX_PER_DIRECTION, "activity_pct"
        )
        enriched = enriched.copy()
        depleted = depleted.copy()
        enriched["rank_label"] = [f"E{index}" for index in range(1, len(enriched) + 1)]
        depleted["rank_label"] = [f"D{index}" for index in range(1, len(depleted) + 1)]
        selected = pd.concat([enriched, depleted], ignore_index=True)
        if selected.empty:
            raise ValueError(f"Structural figure family {family_key} has no directional relationships")
        selected.insert(2, "row_order", np.arange(len(selected)))
        selected_families.append(selected)
    return (
        pd.concat(selected_families, ignore_index=True)
        .sort_values(["family_order", "row_order"], kind="stable")
        .reset_index(drop=True)
    )


def build_structural_classes_all_significant(activity_enrichment: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Return every class passing the exact structural-figure FDR screen."""
    selected_keys = set(zip(selected["family_title"], selected["source_level"].astype(str), strict=True))
    blocks: list[pd.DataFrame] = []
    for family_key, _, _, _, family_title, _ in STRUCTURAL_FIGURE_FAMILIES:
        frame = activity_enrichment.loc[activity_enrichment["family"].eq(family_key)].copy()
        frame["inactive_with_trait"] = frame["n_with_trait"] - frame["n_flag_with_trait"]
        frame["inactive_without_trait"] = frame["n_without_trait"] - frame["n_flag_without_trait"]
        frame = frame.loc[
            frame["n_with_trait"].ge(STRUCTURAL_MIN_MEMBERS)
            & frame["n_without_trait"].ge(STRUCTURAL_MIN_MEMBERS)
            & frame[
                [
                    "n_flag_with_trait",
                    "inactive_with_trait",
                    "n_flag_without_trait",
                    "inactive_without_trait",
                ]
            ]
            .min(axis=1)
            .ge(STRUCTURAL_MIN_CONTINGENCY_CELL)
        ].copy()
        n_tested = len(frame)
        frame["fdr_bh"] = multipletests(frame["p_value"], method="fdr_bh")[1]
        frame = frame.loc[frame["fdr_bh"].lt(STRUCTURAL_FDR_THRESHOLD)].copy()
        frame["family_title"] = family_title
        frame["class"] = frame["trait"].astype(str)
        frame["activity_pct"] = 100 * frame["flag_rate_with"]
        frame["reference_activity_pct"] = 100 * frame["n_flag_without_trait"] / frame["n_without_trait"]
        frame["direction"] = np.where(frame["activity_pct"].gt(frame["reference_activity_pct"]), "Higher", "Lower")
        frame["selected_for_figure"] = [(family_title, class_name) in selected_keys for class_name in frame["class"]]
        frame["n_tested_family"] = n_tested
        ranked: list[pd.DataFrame] = []
        for direction, group in frame.groupby("direction", sort=False):
            group = group.sort_values(
                ["activity_pct", "class"],
                ascending=[direction == "Lower", True],
                kind="stable",
            ).copy()
            group["rank_within_direction"] = np.arange(1, len(group) + 1)
            ranked.append(group)
        blocks.append(pd.concat(ranked, ignore_index=True))
    result = pd.concat(blocks, ignore_index=True)
    return result[
        [
            "family_title",
            "class",
            "direction",
            "rank_within_direction",
            "selected_for_figure",
            "n_tested_family",
            "n_eligible",
            "n_missing",
            "n_with_trait",
            "n_without_trait",
            "n_flag_with_trait",
            "n_flag_without_trait",
            "activity_pct",
            "reference_activity_pct",
            "odds_ratio",
            "ci_lower",
            "ci_upper",
            "p_value",
            "fdr_bh",
        ]
    ].sort_values(
        ["family_title", "direction", "rank_within_direction"],
        ascending=[True, False, True],
        kind="stable",
    )


def build_structural_class_activity_distributions(
    selected: pd.DataFrame,
    trait_maps: dict[str, dict[str, list[str]]],
    valid_structure: set[str],
    activity: pd.DataFrame,
) -> pd.DataFrame:
    """Build compound-level nMAP rows for every selected structural-class box."""
    family_lookup = {trait_id: family_key for family_key, trait_id, *_ in STRUCTURAL_FIGURE_FAMILIES}
    indexed = activity.set_index("JCP2022")
    valid_ids = set(valid_structure) & set(indexed.index)
    blocks: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        family_key = family_lookup[str(row.family_key)]
        member_ids = sorted(set(trait_maps[family_key].get(str(row.source_level), [])) & valid_ids)
        block = indexed.loc[member_ids, ["mean_normalized_average_precision", "is_active"]].reset_index()
        block = block.rename(
            columns={
                "JCP2022": "jcp2022",
                "mean_normalized_average_precision": "nmap",
                "is_active": "active",
            }
        )
        if len(block) != int(row.n_exposed) or int(block["active"].sum()) != int(row.active_exposed):
            raise ValueError(f"Structural distribution disagrees with {row.family_title} {row.rank_label}")
        block.insert(0, "source_level", str(row.source_level))
        block.insert(0, "family_key", str(row.family_key))
        blocks.append(block)
    return (
        pd.concat(blocks, ignore_index=True)
        .sort_values(["family_key", "source_level", "jcp2022"], kind="stable")
        .reset_index(drop=True)
    )


def normalize_cell_density_wells(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce source adjustment and within-plate robust z-scoring."""
    required = {"jcp2022", "source", "plate", "cell_density"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cell-density wells are missing columns: {missing}")
    normalized = frame[list(required)].copy()
    normalized["cell_density"] = pd.to_numeric(normalized["cell_density"], errors="raise")
    if not np.isfinite(normalized["cell_density"]).all():
        raise ValueError("Cell density contains missing or non-finite values")
    source_adjusted = normalized["cell_density"] - normalized.groupby("source", sort=False)["cell_density"].transform(
        "mean"
    )

    def robust_zscore(values: pd.Series) -> pd.Series:
        median = values.median()
        mad = (values - median).abs().median()
        if mad > 0:
            return (values - median) / (mad * 1.4826)
        standard_deviation = values.std()
        if standard_deviation > 0:
            return (values - median) / standard_deviation
        return values - median

    normalized["normalized_cell_density"] = source_adjusted.groupby(
        [normalized["source"], normalized["plate"]], sort=False
    ).transform(robust_zscore)
    return normalized


def aggregate_normalized_cell_density(wells: pd.DataFrame, activity_ids: set[str]) -> pd.DataFrame:
    """Return one median normalized cell-density value per analyzed compound."""
    normalized = normalize_cell_density_wells(wells)
    compound = normalized.groupby("jcp2022", sort=False)["normalized_cell_density"].median().reset_index()
    compound = compound.loc[compound["jcp2022"].astype(str).isin(activity_ids)].copy()
    observed = set(compound["jcp2022"].astype(str))
    if observed != activity_ids:
        raise ValueError(
            "Normalized cell density does not match the activity population: "
            f"missing={sorted(activity_ids - observed)[:10]}, extra={sorted(observed - activity_ids)[:10]}"
        )
    return compound.sort_values("jcp2022", kind="stable").reset_index(drop=True)


def validate_consistency_run_metadata(
    con: duckdb.DuckDBPyConnection,
    *,
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    distance: str,
    activity_threshold: float,
) -> None:
    """Require the fixed nine-source consistency configuration."""
    expected = {(source, group_column) for source, _, _, group_column in CONSISTENCY_SOURCE_SPECS}
    for table in ("consistency_results", "consistency_scores"):
        observed = set(
            map(
                tuple,
                con.execute(
                    f"""
                    SELECT DISTINCT _group_type, _group_column
                    FROM {table}
                    WHERE _dataset = ?
                      AND _columns = ?
                      AND _preprocessing = ?
                      AND _filter = ?
                      AND _distance = ?
                      AND _activity_threshold = ?
                      AND _preprocessing NOT LIKE '%_sweep'
                    """,
                    [dataset, columns, preprocessing, filter_name, distance, activity_threshold],
                ).fetchall(),
            )
        )
        if observed != expected:
            raise ValueError(
                f"{table} has a noncanonical consistency source mapping: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )


def load_consistency_member_data(
    activity: pd.DataFrame,
    *,
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    distance: str,
    activity_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load repeated memberships and their upstream group-level audit fields."""
    activity_ids = set(activity["JCP2022"].astype(str))
    score_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    copairs = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        validate_consistency_run_metadata(
            copairs,
            dataset=dataset,
            columns=columns,
            preprocessing=preprocessing,
            filter_name=filter_name,
            distance=distance,
            activity_threshold=activity_threshold,
        )
        for source_order, (source, source_label, category, _) in enumerate(CONSISTENCY_SOURCE_SPECS):
            groups = query_consistency_results(
                copairs,
                dataset,
                preprocessing=preprocessing,
                filter_name=filter_name,
                group_type=source,
                distance=distance,
                columns=columns,
                activity_threshold=activity_threshold,
            ).rename(columns={"group_value": "annotation_group"})
            if groups.empty:
                raise ValueError(f"Canonical consistency source {source} has no group results")
            groups.insert(0, "annotation_source_order", source_order)
            groups.insert(0, "annotation_category", category)
            groups.insert(0, "annotation_source_label", source_label)
            groups.insert(0, "annotation_source", source)
            group_frames.append(groups)
            scores = query_consistency_scores(
                copairs,
                dataset,
                preprocessing=preprocessing,
                filter_name=filter_name,
                group_type=source,
                distance=distance,
                columns=columns,
                activity_threshold=activity_threshold,
            ).rename(columns={"Metadata_JCP2022": "jcp2022", "group_value": "annotation_group"})
            if scores.empty:
                raise ValueError(f"Canonical consistency source {source} is empty")
            scores.insert(0, "annotation_source_order", source_order)
            scores.insert(0, "annotation_category", category)
            scores.insert(0, "annotation_source_label", source_label)
            scores.insert(0, "annotation_source", source)
            score_frames.append(scores)
    finally:
        copairs.close()
    scores = pd.concat(score_frames, ignore_index=True)
    score_ids = set(scores["jcp2022"].astype(str))
    if outside := score_ids - activity_ids:
        raise ValueError(f"Consistency scores contain compounds outside the activity population: {sorted(outside)[:5]}")
    member_key = ["annotation_source", "annotation_group", "jcp2022"]
    metrics = ["average_precision", "normalized_average_precision", "n_pos_pairs", "n_total_pairs"]
    if not np.isfinite(scores[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Consistency scores contain non-finite metrics")
    variation = scores.groupby(member_key, sort=False)[metrics].nunique(dropna=False)
    if variation.gt(1).any(axis=1).any():
        raise ValueError("Duplicate compound-by-group consistency rows have conflicting scores")
    scores = scores.drop_duplicates(member_key, keep="first").copy()

    metadata = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        activity_ids_frame = pd.DataFrame({"jcp2022": sorted(activity_ids)})
        metadata.register("_activity_ids", activity_ids_frame)
        properties = metadata.execute(
            """
            SELECT
                cm.Metadata_JCP2022 AS jcp2022,
                cm.Metadata_ValidMol AS valid_molecule,
                cm.Metadata_LogP AS logp,
                cm.Metadata_MW AS molecular_weight,
                cm.Metadata_RotatableBonds AS rotatable_bonds,
                cm.Metadata_QED AS qed,
                cm.Metadata_TPSA AS tpsa,
                cm.Metadata_FractionCSP3 AS fraction_csp3,
                cm.Metadata_HasPAINS AS has_pains,
                cm.Metadata_Lipinski_Violations AS lipinski_violations,
                cm.Metadata_repurposing_clinical_phase AS clinical_phase
            FROM compound_metadata cm
            JOIN _activity_ids ids ON cm.Metadata_JCP2022 = ids.jcp2022
            ORDER BY cm.Metadata_JCP2022
            """
        ).df()
        wells = metadata.execute(
            """
            SELECT
                w.Metadata_JCP2022 AS jcp2022,
                counts.Metadata_Source AS source,
                counts.Metadata_Plate AS plate,
                counts.Metadata_Count_Cells
                    / (config.Metadata_Well_Area_Microns2 / 1e6) AS cell_density
            FROM well w
            JOIN cell_counts counts
              ON w.Metadata_Source = counts.Metadata_Source
             AND w.Metadata_Plate = counts.Metadata_Plate
             AND w.Metadata_Well = counts.Metadata_Well
            JOIN microscope_config_fov config
              ON counts.Metadata_Source = config.Metadata_Source
            WHERE w.Metadata_JCP2022 IS NOT NULL
            ORDER BY counts.Metadata_Source, counts.Metadata_Plate, counts.Metadata_Well
            """
        ).df()
    finally:
        metadata.close()
    if set(properties["jcp2022"].astype(str)) != activity_ids or properties["jcp2022"].duplicated().any():
        raise ValueError("Compound properties do not match the activity population one-to-one")
    descriptor_columns = [
        column for _, column, _, _, _ in CONSISTENCY_NUMERIC_TRAITS if column != "normalized_cell_density"
    ]
    valid = properties["valid_molecule"].fillna(False).astype(bool)
    if properties.loc[valid, descriptor_columns].isna().any().any():
        missing = properties.loc[valid, descriptor_columns].isna().sum()
        raise ValueError(f"Valid structures have missing consistency descriptors: {missing[missing > 0].to_dict()}")
    properties.loc[~valid, descriptor_columns] = np.nan
    properties["has_pains"] = properties["has_pains"].astype("boolean").astype(float)
    density = aggregate_normalized_cell_density(wells, activity_ids)
    properties = properties.merge(density, on="jcp2022", how="left", validate="one_to_one")

    clinical_eligible = properties["clinical_phase"].notna()
    clinical_tokens = (
        properties["clinical_phase"]
        .fillna("")
        .map(lambda value: tuple(item.strip() for item in str(value).split("|") if item.strip()))
    )
    for trait_id, stage in CONSISTENCY_STAGE_LEVELS.items():
        properties[trait_id] = np.where(
            clinical_eligible,
            clinical_tokens.map(lambda values, stage=stage: stage in values).astype(float),
            np.nan,
        )
    properties = properties.rename(
        columns={column: trait_id for trait_id, column, _, _, _ in CONSISTENCY_NUMERIC_TRAITS}
    )
    properties = properties[["jcp2022", *CONSISTENCY_TRAIT_IDS]]
    activity_nmap = activity[["JCP2022", "mean_normalized_average_precision"]].rename(
        columns={"JCP2022": "jcp2022", "mean_normalized_average_precision": "activity_nmap"}
    )
    member_data = scores.merge(properties, on="jcp2022", how="left", validate="many_to_one").merge(
        activity_nmap, on="jcp2022", how="left", validate="many_to_one"
    )
    if member_data["activity_nmap"].isna().any():
        raise ValueError("Consistency memberships are missing compound activity nMAP")
    if member_data.duplicated(member_key).any():
        raise ValueError("Consistency analytic memberships are not unique")
    member_data = member_data.sort_values(
        ["annotation_source_order", "annotation_group", "jcp2022"], kind="stable"
    ).reset_index(drop=True)
    group_results = pd.concat(group_frames, ignore_index=True).rename(
        columns={
            "mean_normalized_average_precision": "upstream_mean_normalized_average_precision",
            "corrected_p_value": "upstream_corrected_p_value",
            "below_corrected_p": "upstream_below_corrected_p",
            "n_perturbations": "upstream_score_rows",
        }
    )
    unique_compounds = (
        member_data.groupby(["annotation_source", "annotation_group"], sort=False)["jcp2022"]
        .nunique()
        .rename("upstream_unique_compounds")
        .reset_index()
    )
    group_results = group_results.merge(
        unique_compounds,
        on=["annotation_source", "annotation_group"],
        how="left",
        validate="one_to_one",
    )
    if group_results["upstream_unique_compounds"].isna().any():
        raise ValueError("Consistency group results contain groups without member scores")
    group_results["upstream_unique_compounds"] = group_results["upstream_unique_compounds"].astype(int)
    group_results["duplicate_score_rows"] = (
        group_results["upstream_score_rows"] - group_results["upstream_unique_compounds"]
    )
    if (group_results["duplicate_score_rows"] < 0).any():
        raise ValueError("Consistency group result counts are smaller than unique member counts")
    return member_data, group_results


def build_consistency_group_properties(
    member_data: pd.DataFrame,
    group_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize trait-specific group outcomes and properties for audit."""
    group_key = ["annotation_source", "annotation_group"]
    member_summary = (
        member_data.groupby(group_key, sort=False).agg(n_membership_compounds=("jcp2022", "nunique")).reset_index()
    )
    property_long = member_data.melt(
        id_vars=group_key + ["jcp2022", "normalized_average_precision", "activity_nmap"],
        value_vars=CONSISTENCY_TRAIT_IDS,
        var_name="trait_id",
        value_name="property_value",
    ).dropna(subset=["property_value"])
    property_summary = (
        property_long.groupby(group_key + ["trait_id"], sort=False)
        .agg(
            n_property_compounds=("jcp2022", "nunique"),
            analytic_mean_normalized_ap=("normalized_average_precision", "mean"),
            analytic_median_normalized_ap=("normalized_average_precision", "median"),
            activity_nmap_mean=("activity_nmap", "mean"),
            activity_nmap_median=("activity_nmap", "median"),
            property_mean=("property_value", "mean"),
            property_median=("property_value", "median"),
            property_sd=("property_value", "std"),
            property_unique_values=("property_value", "nunique"),
        )
        .reset_index()
    )
    group_fields = group_results[
        [
            "annotation_source",
            "annotation_source_label",
            "annotation_category",
            "annotation_source_order",
            "annotation_group",
            "upstream_mean_normalized_average_precision",
            "upstream_corrected_p_value",
            "upstream_below_corrected_p",
            "upstream_score_rows",
            "upstream_unique_compounds",
            "duplicate_score_rows",
        ]
    ]
    trait_fields = pd.DataFrame(
        [{"trait_id": trait_id, **CONSISTENCY_TRAIT_FIELDS[trait_id]} for trait_id in CONSISTENCY_TRAIT_IDS]
    )
    trait_grid = group_fields.merge(pd.DataFrame({"trait_id": CONSISTENCY_TRAIT_IDS}), how="cross")
    result = (
        trait_grid.merge(member_summary, on=group_key, how="left", validate="many_to_one")
        .merge(property_summary, on=group_key + ["trait_id"], how="left", validate="one_to_one")
        .merge(trait_fields, on="trait_id", how="left", validate="many_to_one")
    )
    result["n_membership_compounds"] = result["n_membership_compounds"].fillna(0).astype(int)
    result["n_property_compounds"] = result["n_property_compounds"].fillna(0).astype(int)
    result["memberships_dropped_for_trait_eligibility"] = (
        result["upstream_unique_compounds"] - result["n_property_compounds"]
    )
    result["property_eligible_coverage"] = result["n_property_compounds"] / result["upstream_unique_compounds"]
    result["analytic_minus_upstream_normalized_ap"] = (
        result["analytic_mean_normalized_ap"] - result["upstream_mean_normalized_average_precision"]
    )
    result["trait_order"] = result["trait_id"].map(
        {trait_id: order for order, trait_id in enumerate(CONSISTENCY_TRAIT_IDS)}
    )
    result["within_group_property_variation"] = result["property_unique_values"].ge(2)
    result["eligible_group_primary"] = result["n_property_compounds"].ge(MIN_CONSISTENCY_GROUP_MEMBERS)
    result["group_exclusion_reason"] = np.where(
        result["eligible_group_primary"],
        "",
        f"fewer than {MIN_CONSISTENCY_GROUP_MEMBERS} compounds with this trait",
    )
    return result.sort_values(
        ["annotation_source_order", "annotation_group", "trait_order"], kind="stable"
    ).reset_index(drop=True)


def _weighted_center_scale(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    center = float(np.average(values_array, weights=weights_array))
    scale = float(np.sqrt(np.average(np.square(values_array - center), weights=weights_array)))
    return center, scale


def _safe_contextual_point_estimates(
    source_data: pd.DataFrame, trait_id: str, minimum_group_members: int
) -> tuple[float, float]:
    """Return contextual WLS point estimates for a group-size sensitivity."""
    group_key = "annotation_group"
    data = source_data.dropna(subset=[trait_id]).copy()
    group_size = data.groupby(group_key, sort=False)["jcp2022"].transform("nunique")
    data = data.loc[group_size.ge(minimum_group_members)].copy()
    if data.empty:
        return np.nan, np.nan
    data["group_size"] = data.groupby(group_key, sort=False)["jcp2022"].transform("nunique")
    data["property_between"] = data.groupby(group_key, sort=False)[trait_id].transform("mean")
    data["property_within"] = data[trait_id] - data["property_between"]
    data["activity_between"] = data.groupby(group_key, sort=False)["activity_nmap"].transform("mean")
    data["activity_within"] = data["activity_nmap"] - data["activity_between"]
    data["log2_group_size"] = np.log2(data["group_size"].astype(float))
    weights = (1.0 / data["group_size"]).to_numpy(dtype=float)
    standardized: dict[str, np.ndarray] = {}
    for component in (
        "property_between",
        "property_within",
        "activity_between",
        "activity_within",
        "log2_group_size",
    ):
        center, scale = _weighted_center_scale(data[component], weights)
        if not np.isfinite(scale) or scale <= 0:
            return np.nan, np.nan
        standardized[component] = (data[component].to_numpy(dtype=float) - center) / scale
    design = np.column_stack(
        [
            np.ones(len(data), dtype=float),
            standardized["property_between"],
            standardized["property_within"],
            standardized["activity_between"],
            standardized["activity_within"],
            standardized["log2_group_size"],
        ]
    )
    try:
        if np.linalg.matrix_rank(design * np.sqrt(weights)[:, None]) != design.shape[1]:
            return np.nan, np.nan
        fitted = sm.WLS(data["normalized_average_precision"].to_numpy(dtype=float), design, weights=weights).fit()
    except Exception:
        return np.nan, np.nan
    estimates = np.asarray(fitted.params[[1, 2]], dtype=float)
    if not np.isfinite(estimates).all():
        return np.nan, np.nan
    return float(estimates[0]), float(estimates[1])


def _safe_median_between_estimate(data: pd.DataFrame, trait_id: str) -> float:
    """Return an equal-group between estimate based on group medians."""
    if data.empty:
        return np.nan
    group_level = (
        data.groupby("annotation_group", sort=False)
        .agg(
            outcome=("normalized_average_precision", "median"),
            property_value=(trait_id, "median"),
            activity_value=("activity_nmap", "median"),
            group_size=("jcp2022", "nunique"),
        )
        .reset_index()
    )
    standardized = []
    for values in (
        group_level["property_value"].to_numpy(dtype=float),
        group_level["activity_value"].to_numpy(dtype=float),
        np.log2(group_level["group_size"].to_numpy(dtype=float)),
    ):
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            return np.nan
        standardized.append((values - float(np.mean(values))) / scale)
    design = np.column_stack([np.ones(len(group_level), dtype=float), *standardized])
    try:
        if np.linalg.matrix_rank(design) != design.shape[1]:
            return np.nan
        estimate = float(sm.OLS(group_level["outcome"].to_numpy(dtype=float), design).fit().params[1])
    except Exception:
        return np.nan
    return estimate if np.isfinite(estimate) else np.nan


def _fit_consistency_source_trait(source_data: pd.DataFrame, trait_id: str) -> list[dict[str, object]]:
    """Fit one inverse-group-size contextual WLS model and return both scopes."""
    group_key = "annotation_group"
    source_id = str(source_data["annotation_source"].iloc[0])
    source_order = int(source_data["annotation_source_order"].iloc[0])
    source_label = str(source_data["annotation_source_label"].iloc[0])
    annotation_category = str(source_data["annotation_category"].iloc[0])
    data = source_data.dropna(subset=[trait_id]).copy()
    group_size = data.groupby(group_key, sort=False)["jcp2022"].transform("nunique")
    data = data.loc[group_size.ge(MIN_CONSISTENCY_GROUP_MEMBERS)].copy()
    if data.empty:
        raise ValueError(f"Consistency source {source_id} trait {trait_id} has no eligible groups")
    data["group_size"] = data.groupby(group_key, sort=False)["jcp2022"].transform("nunique")
    data["property_between"] = data.groupby(group_key, sort=False)[trait_id].transform("mean")
    data["property_within"] = data[trait_id] - data["property_between"]
    data["activity_between"] = data.groupby(group_key, sort=False)["activity_nmap"].transform("mean")
    data["activity_within"] = data["activity_nmap"] - data["activity_between"]
    data["log2_group_size"] = np.log2(data["group_size"].astype(float))
    data["model_weight"] = 1.0 / data["group_size"]

    group_variation = data.groupby(group_key, sort=False)[trait_id].nunique().ge(2)
    within_varying_groups = set(group_variation[group_variation].index.astype(str))
    within_contributors = data.loc[
        data[group_key].astype(str).isin(within_varying_groups) & data["property_within"].abs().gt(1e-12),
        "jcp2022",
    ].nunique()
    n_groups = int(data[group_key].nunique())
    n_compounds = int(data["jcp2022"].nunique())
    n_within_varying_groups = int(len(within_varying_groups))

    component_names = [
        "property_between",
        "property_within",
        "activity_between",
        "activity_within",
        "log2_group_size",
    ]
    component_scales: dict[str, float] = {}
    standardized: dict[str, np.ndarray] = {}
    invalid_scale_components = []
    for component in component_names:
        center, scale = _weighted_center_scale(data[component], data["model_weight"])
        component_scales[component] = scale
        if not np.isfinite(scale) or scale <= 0:
            invalid_scale_components.append(component)
        else:
            standardized[component] = (data[component].to_numpy(dtype=float) - center) / scale

    design = None
    design_rank = 0
    condition_number = np.nan
    fitted = None
    clustered_covariance = None
    unweighted_fit = None
    model_issue = ""
    if invalid_scale_components:
        model_issue = "non-positive or non-finite scale: " + ", ".join(invalid_scale_components)
    else:
        design = np.column_stack(
            [
                np.ones(len(data), dtype=float),
                standardized["property_between"],
                standardized["property_within"],
                standardized["activity_between"],
                standardized["activity_within"],
                standardized["log2_group_size"],
            ]
        )
        y_values = data["normalized_average_precision"].to_numpy(dtype=float)
        weights = data["model_weight"].to_numpy(dtype=float)
        weighted_design = design * np.sqrt(weights)[:, None]
        try:
            design_rank = int(np.linalg.matrix_rank(weighted_design))
            condition_number = float(np.linalg.cond(weighted_design))
        except Exception as error:
            model_issue = f"design decomposition failed: {type(error).__name__}"
        if not model_issue and design_rank != design.shape[1]:
            model_issue = "rank-deficient model design"
        if not model_issue:
            try:
                fitted = sm.WLS(y_values, design, weights=weights).fit()
                unweighted_fit = sm.OLS(y_values, design).fit()
            except Exception as error:
                model_issue = f"model fit failed: {type(error).__name__}"
        if fitted is not None:
            compound_clusters = pd.factorize(data["jcp2022"], sort=True)[0]
            group_clusters = pd.factorize(data[group_key], sort=True)[0]
            try:
                clustered_covariance = cov_cluster_2groups(
                    fitted, compound_clusters, group_clusters, use_correction=True
                )[0]
            except Exception:
                clustered_covariance = None

    equal_group_property_coefficient = np.nan
    equal_group_audit_issue = ""
    if fitted is not None:
        group_level = (
            data.groupby(group_key, sort=False)
            .agg(
                outcome=("normalized_average_precision", "mean"),
                property_between=("property_between", "first"),
                activity_between=("activity_between", "first"),
                log2_group_size=("log2_group_size", "first"),
            )
            .reset_index()
        )
        weights = data["model_weight"].to_numpy(dtype=float)
        group_design = np.column_stack(
            [
                np.ones(len(group_level), dtype=float),
                (
                    group_level["property_between"].to_numpy(dtype=float)
                    - np.average(data["property_between"], weights=weights)
                )
                / component_scales["property_between"],
                (
                    group_level["activity_between"].to_numpy(dtype=float)
                    - np.average(data["activity_between"], weights=weights)
                )
                / component_scales["activity_between"],
                (
                    group_level["log2_group_size"].to_numpy(dtype=float)
                    - np.average(data["log2_group_size"], weights=weights)
                )
                / component_scales["log2_group_size"],
            ]
        )
        try:
            equal_group_property_coefficient = float(
                sm.OLS(group_level["outcome"].to_numpy(dtype=float), group_design).fit().params[1]
            )
            if not np.isclose(fitted.params[1], equal_group_property_coefficient, rtol=1e-10, atol=1e-10):
                equal_group_audit_issue = "inverse-size WLS failed equal-group coefficient audit"
        except Exception as error:
            equal_group_audit_issue = f"equal-group coefficient audit failed: {type(error).__name__}"

    minimum_10_between, minimum_10_within = _safe_contextual_point_estimates(source_data, trait_id, 10)
    median_between = _safe_median_between_estimate(data, trait_id)
    formula = (
        "member normalized AP ~ z(group-mean property) + z(within-group property deviation) + "
        "z(group-mean activity nMAP) + z(within-group activity nMAP deviation) + "
        "z(log2 analytic group size)"
    )
    rows = []
    for scope, coefficient_index, component, minimum_10_estimate in (
        ("between_group", 1, "property_between", minimum_10_between),
        ("within_group", 2, "property_within", minimum_10_within),
    ):
        reasons = [model_issue] if model_issue else []
        if scope == "between_group" and equal_group_audit_issue:
            reasons.append(equal_group_audit_issue)
        group_cluster_count = n_groups if scope == "between_group" else n_within_varying_groups
        compound_cluster_count = n_compounds if scope == "between_group" else int(within_contributors)
        if group_cluster_count < MIN_CONSISTENCY_CLUSTERS:
            group_label = (
                "eligible annotation groups" if scope == "between_group" else "within-varying annotation groups"
            )
            reasons.append(f"fewer than {MIN_CONSISTENCY_CLUSTERS} {group_label}")
        if compound_cluster_count < MIN_CONSISTENCY_CLUSTERS:
            reasons.append(f"fewer than {MIN_CONSISTENCY_CLUSTERS} contributing compounds")

        coefficient = standard_error = natural_coefficient = unweighted_coefficient = np.nan
        component_scale = float(component_scales.get(component, np.nan))
        if fitted is not None:
            coefficient = float(fitted.params[coefficient_index])
            natural_coefficient = coefficient / component_scale
            unweighted_coefficient = float(unweighted_fit.params[coefficient_index])
            if (
                clustered_covariance is None
                or not np.isfinite(clustered_covariance).all()
                or clustered_covariance[coefficient_index, coefficient_index] <= 0
            ):
                reasons.append("invalid two-way clustered covariance")
            else:
                standard_error = float(np.sqrt(clustered_covariance[coefficient_index, coefficient_index]))
        status = "tested" if not reasons else "ineligible"
        inference_df = int(min(group_cluster_count, compound_cluster_count) - 1)
        if status == "tested":
            critical_value = float(student_t.ppf(0.975, inference_df))
            ci_lower = coefficient - critical_value * standard_error
            ci_upper = coefficient + critical_value * standard_error
            p_value = float(2 * student_t.sf(abs(coefficient / standard_error), inference_df))
        else:
            ci_lower = ci_upper = p_value = np.nan
        median_estimate = median_between if scope == "between_group" else np.nan
        rows.append(
            {
                "source_id": source_id,
                "source_order": source_order,
                "source_label": source_label,
                "annotation_category": annotation_category,
                "trait_id": trait_id,
                "trait_order": CONSISTENCY_TRAIT_IDS.index(trait_id),
                "effect_scope": scope,
                "status": status,
                "exclusion_reason": "; ".join(dict.fromkeys(reasons)),
                "estimate_nap_per_component_sd": coefficient,
                "standard_error_nap_per_component_sd": standard_error,
                "ci_lower_nap_per_component_sd": ci_lower,
                "ci_upper_nap_per_component_sd": ci_upper,
                "component_sd_natural_units": component_scale,
                "estimate_nap_per_natural_unit": natural_coefficient,
                "ci_lower_nap_per_natural_unit": (ci_lower / component_scale if np.isfinite(ci_lower) else np.nan),
                "ci_upper_nap_per_natural_unit": (ci_upper / component_scale if np.isfinite(ci_upper) else np.nan),
                "p_value": p_value,
                "sensitivity_unweighted_member_estimate_nap_per_component_sd": unweighted_coefficient,
                "sensitivity_minimum_10_estimate_nap_per_component_sd": minimum_10_estimate,
                "sensitivity_minimum_10_minus_primary_nap_per_component_sd": (
                    minimum_10_estimate - coefficient
                    if np.isfinite(minimum_10_estimate) and np.isfinite(coefficient)
                    else np.nan
                ),
                "sensitivity_median_group_estimate_nap_per_component_sd": median_estimate,
                "sensitivity_median_group_minus_primary_nap_per_component_sd": (
                    median_estimate - coefficient
                    if np.isfinite(median_estimate) and np.isfinite(coefficient)
                    else np.nan
                ),
                "n_memberships": int(len(data)),
                "n_unique_compounds": n_compounds,
                "n_groups": n_groups,
                "n_within_varying_groups": n_within_varying_groups,
                "min_group_size": int(data["group_size"].min()),
                "compound_clusters": compound_cluster_count,
                "group_clusters": group_cluster_count,
                "inference_df": inference_df,
                "covariance_type": CONSISTENCY_COVARIANCE,
                "group_weighting": "inverse analytic group size; one total weight unit per group",
                "formula": formula,
                "design_rank": design_rank,
                "condition_number": condition_number,
                "between_equal_group_ols_difference": (
                    coefficient - equal_group_property_coefficient
                    if scope == "between_group"
                    and np.isfinite(coefficient)
                    and np.isfinite(equal_group_property_coefficient)
                    else np.nan
                ),
            }
        )
    return rows


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def analyze_consistency_properties(member_data: pd.DataFrame) -> pd.DataFrame:
    """Estimate fixed source-stratified between- and within-group effects."""
    rows: list[dict[str, object]] = []
    for source_id in member_data.sort_values("annotation_source_order", kind="stable")[
        "annotation_source"
    ].drop_duplicates():
        source_data = member_data.loc[member_data["annotation_source"].eq(source_id)]
        for trait_id in CONSISTENCY_TRAIT_IDS:
            rows.extend(_fit_consistency_source_trait(source_data, trait_id))
    relationships = pd.DataFrame(rows)
    for trait_id, fields in CONSISTENCY_TRAIT_FIELDS.items():
        mask = relationships["trait_id"].eq(trait_id)
        for field, value in fields.items():
            relationships.loc[mask, field] = value
    relationships["fdr_family"] = relationships["effect_scope"].map(
        {
            "between_group": "consistency_between_all_sources",
            "within_group": "consistency_within_all_sources",
        }
    )
    relationships["q_value_global_scope"] = np.nan
    relationships["q_value_source_scope"] = np.nan
    tested = relationships["status"].eq("tested") & relationships["p_value"].notna()
    for _, indices in relationships.loc[tested].groupby("effect_scope", sort=True).groups.items():
        relationships.loc[indices, "q_value_global_scope"] = bh_adjust(relationships.loc[indices, "p_value"])
    for _, indices in relationships.loc[tested].groupby(["source_id", "effect_scope"], sort=True).groups.items():
        relationships.loc[indices, "q_value_source_scope"] = bh_adjust(relationships.loc[indices, "p_value"])
    relationships["global_significant_at_5pct_fdr"] = (
        relationships["q_value_global_scope"].lt(CONSISTENCY_EFFECT_FDR_THRESHOLD).fillna(False)
    )
    relationships["source_significant_at_5pct_fdr"] = (
        relationships["q_value_source_scope"].lt(CONSISTENCY_EFFECT_FDR_THRESHOLD).fillna(False)
    )
    expected = 2 * len(CONSISTENCY_TRAIT_IDS) * len(CONSISTENCY_SOURCE_SPECS)
    if len(relationships) != expected:
        raise ValueError(f"Consistency relationship grid has {len(relationships)} rows; expected {expected}")
    if relationships.duplicated(["source_id", "trait_id", "effect_scope"]).any():
        raise ValueError("Consistency relationship grid contains duplicate effects")
    tested_values = relationships.loc[
        relationships["status"].eq("tested"),
        [
            "estimate_nap_per_component_sd",
            "standard_error_nap_per_component_sd",
            "ci_lower_nap_per_component_sd",
            "ci_upper_nap_per_component_sd",
            "p_value",
        ],
    ]
    if not np.isfinite(tested_values.to_numpy(dtype=float)).all():
        raise ValueError("Every tested consistency effect must have finite inference")
    return relationships.sort_values(["effect_scope", "source_order", "trait_order"], kind="stable").reset_index(
        drop=True
    )


# =============================================================================
# Plotting
# =============================================================================


def _color_for(row: pd.Series) -> str:
    if row["corrected_p_value"] < 0.05:
        return "#D55E00" if row["odds_ratio"] > 1 else "#0072B2"
    return "#999999"


def plot_volcano(enrichment: pd.DataFrame, label: str, output_dir: Path, dpi: int) -> None:
    """One subplot per family. log2(OR) on x, -log10(FDR) on y."""
    if enrichment.empty:
        return
    families = list(enrichment["family"].unique())
    n = len(families)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, fam in zip(axes[0], families):
        sub = enrichment[enrichment["family"] == fam].copy()
        sub["neglog10"] = -np.log10(sub["corrected_p_value"].clip(lower=1e-300))
        colors = sub.apply(_color_for, axis=1)
        ax.scatter(
            sub["log2_odds_ratio"], sub["neglog10"], c=colors, s=30, edgecolors="white", linewidths=0.4, alpha=0.85
        )
        # Label top 5 enriched + top 5 depleted by |log2 OR| among significant
        sig = sub[sub["corrected_p_value"] < 0.05]
        if not sig.empty:
            top_enr = sig[sig["odds_ratio"] > 1].nlargest(5, "log2_odds_ratio")
            top_dep = sig[sig["odds_ratio"] < 1].nsmallest(5, "log2_odds_ratio")
            for _, r in pd.concat([top_enr, top_dep]).iterrows():
                ax.annotate(
                    r["trait"][:30],
                    (r["log2_odds_ratio"], r["neglog10"]),
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    xytext=(0, 4),
                    textcoords="offset points",
                )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(-np.log10(0.05), color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("log₂(Odds Ratio)")
        ax.set_ylabel("-log₁₀(FDR)")
        ax.set_title(f"{fam} ({label})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"volcano_{label}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: volcano_{label}.png")


def plot_top_bars(enrichment: pd.DataFrame, label: str, output_dir: Path, dpi: int, top_n: int = 10) -> None:
    """Per family: top-N enriched + top-N depleted significant traits."""
    if enrichment.empty:
        return
    families = list(enrichment["family"].unique())
    n = len(families)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 0.35 * top_n * 2) * n), squeeze=False)
    for ax, fam in zip(axes[:, 0], families):
        sub = enrichment[(enrichment["family"] == fam) & (enrichment["corrected_p_value"] < 0.05)]
        top_enr = sub[sub["odds_ratio"] > 1].nlargest(top_n, "log2_odds_ratio")
        top_dep = sub[sub["odds_ratio"] < 1].nsmallest(top_n, "log2_odds_ratio")
        combined = pd.concat([top_dep[::-1], top_enr])
        if combined.empty:
            ax.text(0.5, 0.5, f"No FDR<0.05 hits in {fam}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{fam} ({label})")
            continue
        colors = ["#D55E00" if o > 1 else "#0072B2" for o in combined["odds_ratio"]]
        y = range(len(combined))
        ax.barh(y, combined["log2_odds_ratio"], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(list(y))
        ax.set_yticklabels([t[:50] for t in combined["trait"]], fontsize=8)
        ax.set_xlabel("log₂(Odds Ratio)")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{fam} ({label}) — top enriched (orange) / depleted (blue), FDR<0.05")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"top_traits_bar_{label}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: top_traits_bar_{label}.png")


# =============================================================================
# Main
# =============================================================================


def _run_sweep(
    flag_df: pd.DataFrame,
    flag_col: str,
    trait_maps: dict[str, dict[str, list[str]]],
    shared_eligible_by_family: dict[str, set[str]],
    eligible_by_trait_by_family: dict[str, dict[str, set[str]]],
    label: str,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    """Run the full family sweep for a single flag (activity or consistency)."""
    flag = flag_df.set_index("JCP2022")[flag_col]
    parts = []
    for family, tmap in trait_maps.items():
        df = compute_trait_enrichment(
            tmap,
            flag,
            family,
            MIN_N.get(family, 5),
            shared_eligible=shared_eligible_by_family.get(family),
            eligible_by_trait=eligible_by_trait_by_family.get(family),
        )
        if not df.empty:
            parts.append(df)
    enrichment = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if enrichment.empty:
        logger.warning(f"No enrichments produced for {label}")
        return enrichment
    enrichment.to_csv(output_dir / f"enrichment_{label}.csv", index=False)
    logger.info(f"Saved: enrichment_{label}.csv ({len(enrichment):,} rows)")
    plot_volcano(enrichment, label, output_dir, dpi)
    plot_top_bars(enrichment, label, output_dir, dpi)
    return enrichment


@app.command()
def main(
    dataset: str = typer.Option("compound_no_source7", "--dataset", "-d"),
    columns: str = typer.Option("feat_all", "--columns"),
    preprocessing: str = typer.Option("activity_no_target2", "--preprocessing"),
    filter_name: str = typer.Option("all_sources", "--filter"),
    activity_params: str = typer.Option("default", "--activity-params"),
    activity_threshold: float = typer.Option(ACTIVITY_P_THRESHOLD, "--activity-threshold"),
    consistency_threshold: float = typer.Option(CONSISTENCY_P_THRESHOLD, "--consistency-threshold"),
    consistency_preprocessing: str = typer.Option("consistency_no_target2", "--consistency-preprocessing"),
    consistency_filter: str = typer.Option("all_sources", "--consistency-filter"),
    consistency_distance: str = typer.Option("cosine", "--consistency-distance"),
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi"),
    skip_brics: bool = typer.Option(False, "--skip-brics"),
) -> None:
    """Trait → activity / consistency enrichment."""
    out = output_dir or (OUTPUT_DIR / dataset)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out}")

    # 1. Labels
    activity = load_activity_flags(
        dataset,
        columns,
        preprocessing,
        filter_name,
        activity_params,
        activity_threshold,
    )
    consistency = load_consistency_flags(
        dataset,
        columns,
        consistency_preprocessing,
        consistency_filter,
        consistency_distance,
        consistency_threshold,
    )

    # 2. Trait maps
    meta = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        smiles_df = load_smiles_and_scaffolds(meta)
        # Restrict to compounds we'll actually test (have activity OR consistency labels)
        analyzed = set(activity["JCP2022"]) | set(consistency["JCP2022"])
        smiles_df = smiles_df[smiles_df["JCP2022"].isin(analyzed)].reset_index(drop=True)
        logger.info(f"SMILES restricted to {len(smiles_df):,} analyzed compounds")

        trait_maps: dict[str, dict[str, list[str]]] = {}
        valid_structure = set(smiles_df["JCP2022"])
        shared_eligible_by_family: dict[str, set[str]] = {}
        trait_maps["rdkit_fg"] = build_rdkit_fragment_map(smiles_df)
        shared_eligible_by_family["rdkit_fg"] = valid_structure
        if not skip_brics:
            trait_maps["brics"] = build_brics_map(smiles_df)
            shared_eligible_by_family["brics"] = valid_structure
        trait_maps["scaffold"] = build_scaffold_map(smiles_df)
        shared_eligible_by_family["scaffold"] = valid_structure
        target_map, target_eligible = build_target_map(meta)
        trait_maps["target_class"] = target_map
        eligible_by_trait_by_family = {"target_class": target_eligible}
    finally:
        meta.close()

    # 3. Run sweeps
    activity_enr = _run_sweep(
        activity,
        "is_active",
        trait_maps,
        shared_eligible_by_family,
        eligible_by_trait_by_family,
        "activity",
        out,
        dpi,
    )
    consistency_enr = _run_sweep(
        consistency,
        "is_consistent_any",
        trait_maps,
        shared_eligible_by_family,
        eligible_by_trait_by_family,
        "consistency",
        out,
        dpi,
    )

    structural_plot_table = None
    structural_distributions = None
    structural_all_significant = None
    if skip_brics:
        logger.warning("Skipping structural_class_activity_nmap.csv because candidate_07 requires the BRICS family")
    else:
        structural_plot_table = build_structural_class_activity_nmap(
            activity_enr,
            trait_maps,
            valid_structure,
            activity,
        )
        structural_plot_path = out / "structural_class_activity_nmap.csv"
        structural_plot_table.to_csv(structural_plot_path, index=False, lineterminator="\n")
        logger.info(f"Saved: {structural_plot_path} ({len(structural_plot_table)} selected relationships)")
        structural_distributions = build_structural_class_activity_distributions(
            structural_plot_table,
            trait_maps,
            valid_structure,
            activity,
        )
        structural_distribution_path = out / "structural_class_activity_distributions.parquet"
        structural_distributions.to_parquet(structural_distribution_path, index=False)
        logger.info(f"Saved: {structural_distribution_path} ({len(structural_distributions):,} displayed memberships)")
        structural_all_significant = build_structural_classes_all_significant(activity_enr, structural_plot_table)
        structural_all_path = out / "structural_classes_all_significant_5pct_fdr.csv"
        structural_all_significant.to_csv(structural_all_path, index=False, lineterminator="\n")
        logger.info(f"Saved: {structural_all_path} ({len(structural_all_significant)} significant relationships)")

    consistency_member_data, consistency_group_results = load_consistency_member_data(
        activity,
        dataset=dataset,
        columns=columns,
        preprocessing=consistency_preprocessing,
        filter_name=consistency_filter,
        distance=consistency_distance,
        activity_threshold=activity_threshold,
    )
    consistency_relationships = analyze_consistency_properties(consistency_member_data)
    consistency_group_properties = build_consistency_group_properties(
        consistency_member_data, consistency_group_results
    )
    consistency_group_path = out / "consistency_group_properties.csv"
    consistency_group_properties.to_csv(consistency_group_path, index=False, lineterminator="\n")
    logger.info(f"Saved: {consistency_group_path} ({len(consistency_group_properties)} group-trait audit rows)")
    consistency_effects_path = out / "consistency_property_effects.csv"
    consistency_relationships.to_csv(consistency_effects_path, index=False, lineterminator="\n")
    logger.info(
        f"Saved: {consistency_effects_path} ({len(consistency_relationships)} source-trait-scope relationships)"
    )

    # 4. Summary
    def _per_family_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
        if df.empty:
            return {}
        result = {}
        for fam, sub in df.groupby("family"):
            sig = sub[sub["corrected_p_value"] < 0.05]
            result[fam] = {
                "n_tested": int(len(sub)),
                "n_eligible_min": int(sub["n_eligible"].min()),
                "n_eligible_max": int(sub["n_eligible"].max()),
                "n_significant": int(len(sig)),
                "n_enriched": int((sig["odds_ratio"] > 1).sum()),
                "n_depleted": int((sig["odds_ratio"] < 1).sum()),
                "top_enriched": sig[sig["odds_ratio"] > 1]
                .nlargest(5, "log2_odds_ratio")[
                    ["trait", "log2_odds_ratio", "corrected_p_value", "n_with_trait", "flag_rate_with"]
                ]
                .to_dict(orient="records"),
                "top_depleted": sig[sig["odds_ratio"] < 1]
                .nsmallest(5, "log2_odds_ratio")[
                    ["trait", "log2_odds_ratio", "corrected_p_value", "n_with_trait", "flag_rate_with"]
                ]
                .to_dict(orient="records"),
            }
        return result

    summary = {
        "dataset": dataset,
        "columns": columns,
        "preprocessing": preprocessing,
        "filter_name": filter_name,
        "activity_params": activity_params,
        "activity_threshold": activity_threshold,
        "consistency_threshold": consistency_threshold,
        "consistency_config": {
            "columns": columns,
            "preprocessing": consistency_preprocessing,
            "filter_name": consistency_filter,
            "distance": consistency_distance,
        },
        "skip_brics": skip_brics,
        "min_compounds_by_family": MIN_N,
        "n_activity_compounds": int(len(activity)),
        "n_active": int(activity["is_active"].sum()),
        "n_consistency_compounds": int(len(consistency)),
        "n_consistent_any": int(consistency["is_consistent_any"].sum()),
        "structural_class_activity_nmap": {
            "design": "candidate_07",
            "available": structural_plot_table is not None,
            "n_rows": int(len(structural_plot_table)) if structural_plot_table is not None else 0,
            "n_families": (
                int(structural_plot_table["family_key"].nunique()) if structural_plot_table is not None else 0
            ),
            "selection": (
                f"Up to {STRUCTURAL_FIGURE_MAX_PER_DIRECTION} significant enriched and "
                f"{STRUCTURAL_FIGURE_MAX_PER_DIRECTION} significant depleted relationships per family"
            ),
            "maximum_rows_per_direction": STRUCTURAL_FIGURE_MAX_PER_DIRECTION,
            "fdr_threshold": STRUCTURAL_FDR_THRESHOLD,
            "minimum_members_per_group": STRUCTURAL_MIN_MEMBERS,
            "minimum_contingency_cell_count": STRUCTURAL_MIN_CONTINGENCY_CELL,
            "test": "Two-sided Fisher exact test",
            "multiple_testing": "Benjamini-Hochberg separately within each structural family",
            "eligible_population": "Compounds with Metadata_ValidMol=true",
            "reference_population": "Other compounds with Metadata_ValidMol=true",
            "unavailable_reason": (
                "The --skip-brics option omits a required candidate_07 family" if structural_plot_table is None else ""
            ),
        },
        "structural_class_activity_distributions": {
            "available": structural_distributions is not None,
            "n_rows": int(len(structural_distributions)) if structural_distributions is not None else 0,
            "grain": "one displayed structural-class membership by compound",
        },
        "structural_classes_all_significant_5pct_fdr": {
            "available": structural_all_significant is not None,
            "n_rows": int(len(structural_all_significant)) if structural_all_significant is not None else 0,
            "n_selected_for_figure": (
                int(structural_all_significant["selected_for_figure"].sum())
                if structural_all_significant is not None
                else 0
            ),
            "families": (
                {
                    title: {
                        "n_tested": int(group["n_tested_family"].iloc[0]),
                        "n_significant": int(len(group)),
                    }
                    for title, group in structural_all_significant.groupby("family_title", sort=False)
                }
                if structural_all_significant is not None
                else {}
            ),
        },
        "consistency_property_effects": {
            "design": "paired_between_within_either_scope_global_fdr",
            "n_rows": int(len(consistency_relationships)),
            "n_tested": int(consistency_relationships["status"].eq("tested").sum()),
            "n_global_significant": int(consistency_relationships["global_significant_at_5pct_fdr"].sum()),
            "fdr_threshold": CONSISTENCY_EFFECT_FDR_THRESHOLD,
            "model": "Inverse-group-size contextual WLS with between- and within-group property components",
            "covariance": CONSISTENCY_COVARIANCE,
            "multiple_testing": "Benjamini-Hochberg separately across all tested effects within each scope",
            "display_selection": CONSISTENCY_DISPLAY_SELECTION,
        },
        "consistency_group_properties": {
            "n_rows": int(len(consistency_group_properties)),
            "n_groups": int(
                consistency_group_properties[["annotation_source", "annotation_group"]].drop_duplicates().shape[0]
            ),
            "n_traits": int(consistency_group_properties["trait_id"].nunique()),
            "grain": "one annotation-source group by property row",
        },
        "eligible_population_contract": {
            "structural_families": "Compounds with Metadata_ValidMol=true",
            "target_class": "Compounds annotated by the target or MOA source prefix",
        },
        "enrichment_method": {
            "test": "Two-sided Fisher exact test within each eligible population",
            "odds_ratio": "Haldane-Anscombe corrected point estimate with Woolf 95% confidence interval",
            "multiple_testing": "Benjamini-Hochberg correction separately within each trait family",
        },
        "activity": _per_family_counts(activity_enr),
        "consistency": _per_family_counts(consistency_enr),
    }
    write_summary(
        out / "summary.json",
        description=(
            "Trait-to-phenotype enrichment: for each compound trait family (RDKit named "
            "functional groups, BRICS fragments, Murcko scaffolds, target/MoA classes), tests "
            "whether compounds carrying the trait are over- or under-represented among "
            "phenotypically active and phenotypically consistent compounds (Fisher's exact, "
            "BH FDR within family). Produces per-family enrichment tables, volcano plots, and "
            "top-trait bar charts."
        ),
        numbers=summary,
        files=index_files(out),
    )
    logger.info("Saved: summary.json")


if __name__ == "__main__":
    app()
