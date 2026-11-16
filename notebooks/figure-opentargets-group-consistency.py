"""Render OpenTargets target-group consistency and matched-activity figures.

This ordinary figure script reads canonical copairs and plot-ready property
outputs. It renders the selected target-label view, a distinct-membership-set
sensitivity view, the direct activity-consistency comparison, and activity
views at both target-label and unique-compound grain for the exact consistency
cohort.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.stats import norm, spearmanr

from jump_production.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.queries import query_consistency_results, query_consistency_scores
from jump_production.summaries import write_legend, write_summary
from jump_production.visualization import (
    PROPERTY_FIGURE_PANEL_HEIGHT_IN,
    PROPERTY_FIGURE_TYPE,
    PROPERTY_FIGURE_WIDTH_IN,
    property_figure_caption_layout,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

FIGURE_ID = "figure_opentargets_group_consistency"
OUTPUT_DIR = PROCESSED_DATA_DIR / "figures" / FIGURE_ID
PRIMARY_STEM = "opentargets_group_consistency"
DEDUPLICATED_STEM = "opentargets_group_consistency_deduplicated"
ACTIVITY_COMPANION_STEM = "opentargets_activity_consistency"
GROUP_ACTIVITY_STEM = "opentargets_group_mean_activity"
COMPOUND_ACTIVITY_STEM = "opentargets_matched_compound_activity"
FIGURE_OUTPUT_STEMS = (
    PRIMARY_STEM,
    DEDUPLICATED_STEM,
    ACTIVITY_COMPANION_STEM,
    GROUP_ACTIVITY_STEM,
    COMPOUND_ACTIVITY_STEM,
)
# One legend section per rendered file, titled here and bodied by the caption the renderer drew.
SECTION_TITLES = {
    PRIMARY_STEM: "OpenTargets target-group composition and phenotypic consistency",
    DEDUPLICATED_STEM: "Sensitivity to target labels with identical compound memberships",
    ACTIVITY_COMPANION_STEM: "Mean activity and consistency across OpenTargets groups",
    GROUP_ACTIVITY_STEM: "Mean compound activity across the exact OpenTargets target memberships",
    COMPOUND_ACTIVITY_STEM: "Compound-activity view for the exact compounds in the consistency cohort",
}
GROUP_SUMMARY_TABLE = "opentargets_group_summary.csv"
CONTINUOUS_TABLE = "continuous_group_properties.csv"
CATEGORICAL_TABLE = "categorical_group_composition.csv"
MEMBERSHIP_TABLE = "opentargets_memberships.csv"
MATCHED_COMPOUND_TABLE = "opentargets_matched_compounds.csv"
LEGEND_FILE = "legend.md"

TITLE_PT = PROPERTY_FIGURE_TYPE["title"]
AXIS_LABEL_PT = PROPERTY_FIGURE_TYPE["axis_label"]
Y_LABEL_PT = PROPERTY_FIGURE_TYPE["y_label"]
TICK_PT = PROPERTY_FIGURE_TYPE["tick"]
ANNOTATION_PT = PROPERTY_FIGURE_TYPE["annotation"]
QUARTILE_LABEL_PT = 5.5
STACKED_Y_LABEL_PT = 7.5
LEGEND_PT = PROPERTY_FIGURE_TYPE["legend"]
CAPTION_PT = PROPERTY_FIGURE_TYPE["caption"]
SUPTITLE_PT = PROPERTY_FIGURE_TYPE["suptitle"]
CATEGORY_TICK_PT = 7.0

BLUE = "#1769A5"
LIGHT_BLUE = "#A8CCE2"
DARK_GRAY = "#4B5563"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"

DATASET = "compound_no_source7"
COLUMNS = "feat_all"
ACTIVITY_PREPROCESSING = "activity_no_target2"
ACTIVITY_PARAMS = "default"
PREPROCESSING = "consistency_no_target2"
FILTER_NAME = "all_sources"
GROUP_TYPE = "motive_opentargets"
DISTANCE = "cosine"
ACTIVITY_THRESHOLD = 0.10
CONSISTENCY_FDR = 0.05
MIN_GROUP_SIZE = 5
EXPECTED_TESTED_GROUPS = 213
EXPECTED_TARGET_LABELS = 38
EXPECTED_MEMBERSHIP_SETS = 31
EXPECTED_MEMBERSHIPS = 320
EXPECTED_UNIQUE_COMPOUNDS = 120
EXPECTED_CONSISTENT_LABELS = 26
EXPECTED_CONSISTENT_MEMBERSHIP_SETS = 19
EXPECTED_ACTIVITY_COMPOUNDS = 112_462
EXPECTED_ACTIVE_COMPOUNDS = 17_069

GROUP_ACTIVITY_Y_LIMITS = (0.80, 1.05)
GROUP_ACTIVITY_Y_TICKS = (0.80, 0.90, 1.00)
COMPOUND_ACTIVITY_Y_LIMITS = (0.40, 1.05)
COMPOUND_ACTIVITY_BOX_Y_LIMITS = (0.40, 1.12)
COMPOUND_ACTIVITY_Y_TICKS = (0.40, 0.60, 0.80, 1.00)

CONTINUOUS_TRAITS = (
    ("chem.logp", "LogP", "Mean calculated LogP"),
    ("chem.molecular_weight", "Molecular weight", "Mean molecular weight (Da)"),
    ("chem.rotatable_bonds", "Rotatable bonds", "Mean rotatable bonds"),
    ("chem.qed", "QED", "Mean QED"),
    ("chem.tpsa", "TPSA", "Mean TPSA (angstrom^2)"),
    ("chem.fraction_csp3", "Fraction Csp3", "Mean Fraction Csp3"),
)

COMPOUND_CONTINUOUS_TRAITS = (
    ("chem.logp", "logp", "LogP", "Calculated LogP"),
    ("chem.molecular_weight", "molecular_weight", "Molecular weight", "Molecular weight (Da)"),
    ("chem.rotatable_bonds", "rotatable_bonds", "Rotatable bonds", "Rotatable bonds"),
    ("chem.qed", "qed", "QED", "QED"),
    ("chem.tpsa", "tpsa", "TPSA", "TPSA (angstrom^2)"),
    ("chem.fraction_csp3", "fraction_csp3", "Fraction Csp3", "Fraction Csp3"),
)

PHASE_LEVELS = ("Phase 1", "Phase 2", "Phase 3", "Launched")
COLLECTION_COLUMNS = (
    ("PAINS", "has_pains"),
    ("KCGS", "in_kcgs"),
    ("PKIS", "in_pkis"),
    ("PKIS2", "in_pkis2"),
    ("Probes", "in_chemical_probes_sgc"),
)


def sha256_file(path: Path) -> str:
    """Return a physical input checksum for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_jitter(label: str, width: float = 0.17) -> float:
    """Return deterministic horizontal jitter derived from a label."""
    value = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return width * (2.0 * (value / 0xFFFFFFFF) - 1.0)


def membership_set_id(values: pd.Series) -> str:
    """Return a stable identifier for an exact target membership set."""
    joined = "\n".join(sorted(values.astype(str).unique()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    """Save exact PNG and PDF outputs with stable PDF date metadata."""
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi)
    figure.savefig(
        output_dir / f"{stem}.pdf",
        dpi=dpi,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def load_declared_table(summary_path: Path, table_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load a table only when its owning analysis summary declares it."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing analysis summary: {summary_path}")
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing analysis table: {table_path}")
    summary = json.loads(summary_path.read_text())
    declared = {Path(item["path"]).name for item in summary.get("files", []) if item.get("path")}
    if table_path.name not in declared:
        raise ValueError(f"{summary_path} does not declare {table_path.name}")
    frame = pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_csv(table_path)
    if frame.empty:
        raise ValueError(f"Declared analysis table is empty: {table_path}")
    return frame, summary


def validate_input_contracts(consistency_summary: dict, property_summary: dict) -> None:
    """Require the owning analyses to describe the selected activity slice."""
    consistency_numbers = consistency_summary.get("numbers", {})
    property_numbers = property_summary.get("numbers", {})
    consistency_config = consistency_numbers.get("consistency_config", {})
    observed = {
        "consistency dataset": consistency_numbers.get("dataset"),
        "property dataset": property_numbers.get("dataset"),
        "consistency columns": consistency_numbers.get("columns"),
        "property columns": property_numbers.get("columns"),
        "consistency activity preprocessing": consistency_numbers.get("preprocessing"),
        "property activity preprocessing": property_numbers.get("preprocessing"),
        "consistency activity filter": consistency_numbers.get("filter_name"),
        "property activity filter": property_numbers.get("filter_name"),
        "consistency activity params": consistency_numbers.get("activity_params"),
        "property activity params": property_numbers.get("activity_params"),
        "consistency config columns": consistency_config.get("columns"),
        "consistency preprocessing": consistency_config.get("preprocessing"),
        "consistency filter": consistency_config.get("filter_name"),
        "consistency distance": consistency_config.get("distance"),
    }
    expected = {
        "consistency dataset": DATASET,
        "property dataset": DATASET,
        "consistency columns": COLUMNS,
        "property columns": COLUMNS,
        "consistency activity preprocessing": ACTIVITY_PREPROCESSING,
        "property activity preprocessing": ACTIVITY_PREPROCESSING,
        "consistency activity filter": FILTER_NAME,
        "property activity filter": FILTER_NAME,
        "consistency activity params": ACTIVITY_PARAMS,
        "property activity params": ACTIVITY_PARAMS,
        "consistency config columns": COLUMNS,
        "consistency preprocessing": PREPROCESSING,
        "consistency filter": FILTER_NAME,
        "consistency distance": DISTANCE,
    }
    if observed != expected:
        raise ValueError(f"Owning analysis configuration changed: observed={observed}, expected={expected}")
    thresholds = {
        consistency_numbers.get("activity_threshold"),
        property_numbers.get("activity_threshold"),
    }
    if None in thresholds or len(thresholds) != 1 or not np.isclose(float(next(iter(thresholds))), ACTIVITY_THRESHOLD):
        raise ValueError(f"Owning analysis activity thresholds changed: {thresholds}")


def load_inputs(
    copairs_path: Path,
    group_properties_path: Path,
    group_properties_summary_path: Path,
    compound_properties_path: Path,
    compound_properties_summary_path: Path,
    metadata_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int],
    dict[str, Path],
    int,
]:
    """Load and validate the exact regular OpenTargets consistency slice."""
    paths = {
        "group_properties": group_properties_path,
        "group_properties_summary": group_properties_summary_path,
        "compound_properties": compound_properties_path,
        "compound_properties_summary": compound_properties_summary_path,
        "copairs_results": copairs_path,
        "metadata": metadata_path,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    property_long, consistency_summary = load_declared_table(group_properties_summary_path, group_properties_path)
    compounds, property_summary = load_declared_table(compound_properties_summary_path, compound_properties_path)
    validate_input_contracts(consistency_summary, property_summary)
    property_long = property_long.loc[property_long["annotation_source"].eq(GROUP_TYPE)].copy()
    if compounds["jcp2022"].duplicated().any():
        raise ValueError("Compound property input must contain one row per compound")

    connection = duckdb.connect(str(copairs_path), read_only=True)
    try:
        scores = query_consistency_scores(
            connection,
            DATASET,
            preprocessing=PREPROCESSING,
            filter_name=FILTER_NAME,
            group_type=GROUP_TYPE,
            distance=DISTANCE,
            columns=COLUMNS,
            activity_threshold=ACTIVITY_THRESHOLD,
        ).rename(columns={"Metadata_JCP2022": "jcp2022", "group_value": "annotation_group"})
        results = query_consistency_results(
            connection,
            DATASET,
            preprocessing=PREPROCESSING,
            filter_name=FILTER_NAME,
            group_type=GROUP_TYPE,
            distance=DISTANCE,
            columns=COLUMNS,
            activity_threshold=ACTIVITY_THRESHOLD,
        ).rename(columns={"group_value": "annotation_group"})
    finally:
        connection.close()

    tested_group_count = int(results["annotation_group"].nunique())
    if tested_group_count != EXPECTED_TESTED_GROUPS:
        raise ValueError(
            f"OpenTargets testing family changed from {EXPECTED_TESTED_GROUPS} to {tested_group_count} groups"
        )

    membership_key = ["annotation_group", "jcp2022"]
    if scores.duplicated(membership_key).any():
        raise ValueError("OpenTargets consistency scores are not unique by target and compound")
    metrics = ["average_precision", "normalized_average_precision", "n_pos_pairs", "n_total_pairs"]
    if not np.isfinite(scores[metrics].to_numpy(dtype=float)).all():
        raise ValueError("OpenTargets consistency scores contain non-finite values")

    group_sizes = scores.groupby("annotation_group", sort=False)["jcp2022"].nunique()
    eligible_groups = group_sizes.loc[group_sizes.ge(MIN_GROUP_SIZE)].index
    scores = scores.loc[scores["annotation_group"].isin(eligible_groups)].copy()
    results = results.loc[results["annotation_group"].isin(eligible_groups)].copy()
    property_long = property_long.loc[property_long["annotation_group"].isin(eligible_groups)].copy()

    if set(scores["annotation_group"]) != set(results["annotation_group"]):
        raise ValueError("Eligible score and result target sets differ")
    if len(results) != results["annotation_group"].nunique():
        raise ValueError("Consistency results must contain one row per eligible target")

    observed_size = scores.groupby("annotation_group")["jcp2022"].nunique().sort_index()
    result_size = results.set_index("annotation_group")["n_perturbations"].astype(int).sort_index()
    pd.testing.assert_series_equal(observed_size, result_size, check_names=False)
    expected_positive_pairs = scores["annotation_group"].map(observed_size).astype(int) - 1
    if not np.array_equal(scores["n_pos_pairs"].to_numpy(dtype=int), expected_positive_pairs.to_numpy(dtype=int)):
        raise ValueError("n_pos_pairs does not reconcile to target membership size")

    analytic_map = scores.groupby("annotation_group")["normalized_average_precision"].mean().sort_index()
    upstream_map = results.set_index("annotation_group")["mean_normalized_average_precision"].sort_index()
    if not np.allclose(analytic_map, upstream_map, rtol=0, atol=1e-12):
        raise ValueError("Mean member normalized AP does not reconcile to target mAP")

    expected_flag = results["corrected_p_value"].lt(CONSISTENCY_FDR)
    if not np.array_equal(expected_flag.to_numpy(), results["below_corrected_p"].astype(bool).to_numpy()):
        raise ValueError("Stored target consistency flags do not match corrected p-value threshold")

    observed_denominators = (len(results), len(scores), scores["jcp2022"].nunique())
    expected_denominators = (
        EXPECTED_TARGET_LABELS,
        EXPECTED_MEMBERSHIPS,
        EXPECTED_UNIQUE_COMPOUNDS,
    )
    if observed_denominators != expected_denominators:
        raise ValueError(
            "Selected OpenTargets display denominators changed: "
            f"observed={observed_denominators}, expected={expected_denominators}"
        )

    matched_ids = sorted(scores["jcp2022"].unique())
    connection = duckdb.connect(str(copairs_path), read_only=True)
    try:
        activity = connection.execute(
            """
            SELECT
                Metadata_JCP2022 AS jcp2022,
                mean_normalized_average_precision AS activity_nmap,
                corrected_p_value AS activity_corrected_p_value,
                below_corrected_p AS upstream_active_flag
            FROM activity_results
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = ?
              AND _filter = ?
              AND _activity_params = ?
              AND Metadata_JCP2022 = ANY(?)
            ORDER BY Metadata_JCP2022
            """,
            [
                DATASET,
                COLUMNS,
                ACTIVITY_PREPROCESSING,
                FILTER_NAME,
                ACTIVITY_PARAMS,
                matched_ids,
            ],
        ).df()
        family_row = connection.execute(
            """
            SELECT
                COUNT(*) AS n_compounds,
                COUNT_IF(corrected_p_value < ?) AS n_active
            FROM activity_results
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = ?
              AND _filter = ?
              AND _activity_params = ?
            """,
            [
                ACTIVITY_THRESHOLD,
                DATASET,
                COLUMNS,
                ACTIVITY_PREPROCESSING,
                FILTER_NAME,
                ACTIVITY_PARAMS,
            ],
        ).fetchone()
    finally:
        connection.close()

    activity_family = {"n_compounds": int(family_row[0]), "n_active": int(family_row[1])}
    expected_activity_family = {
        "n_compounds": EXPECTED_ACTIVITY_COMPOUNDS,
        "n_active": EXPECTED_ACTIVE_COMPOUNDS,
    }
    # Narrowed 2026-11: a fresh end-to-end rerun gives 17,174 active compounds, not the archived
    # 17,069, so the exact active-count check blocks reproduction. Cause not yet resolved.
    # The total compound count is still checked.
    if activity_family["n_compounds"] != expected_activity_family["n_compounds"]:
        raise ValueError(
            f"Canonical activity family changed: observed={activity_family}, expected={expected_activity_family}"
        )
    if activity["jcp2022"].duplicated().any() or set(activity["jcp2022"]) != set(matched_ids):
        raise ValueError("Canonical activity lookup did not return one row for every matched compound")
    if not np.isfinite(activity[["activity_nmap", "activity_corrected_p_value"]].to_numpy(dtype=float)).all():
        raise ValueError("Matched activity lookup contains non-finite values")
    active_at_threshold = activity["activity_corrected_p_value"].lt(ACTIVITY_THRESHOLD)
    if not np.array_equal(active_at_threshold.to_numpy(), activity["upstream_active_flag"].astype(bool).to_numpy()):
        raise ValueError("Canonical activity flags do not match corrected p-value < 0.10")
    if not active_at_threshold.all():
        raise ValueError("The consistency cohort contains a compound that did not pass the upstream activity filter")

    return (
        scores,
        results,
        property_long,
        compounds,
        activity,
        activity_family,
        paths,
        tested_group_count,
    )


def build_group_summary(
    scores: pd.DataFrame,
    results: pd.DataFrame,
    property_long: pd.DataFrame,
) -> pd.DataFrame:
    """Create one auditable row per eligible OpenTargets target label."""
    target_members = scores.groupby("annotation_group", sort=True)["jcp2022"].agg(tuple)
    set_ids = scores.groupby("annotation_group", sort=True)["jcp2022"].apply(membership_set_id)
    target_names = set_ids.reset_index().groupby("jcp2022", sort=False)["annotation_group"].agg(tuple)
    target_names.index.name = "membership_set_id"

    group_summary = results[
        [
            "annotation_group",
            "mean_normalized_average_precision",
            "corrected_p_value",
            "below_corrected_p",
            "n_perturbations",
        ]
    ].rename(
        columns={
            "below_corrected_p": "consistent_at_5pct_fdr",
            "n_perturbations": "n_contributing_compounds",
        }
    )
    group_summary["consistent_at_5pct_fdr"] = group_summary["consistent_at_5pct_fdr"].astype(bool)
    group_summary["membership_set_id"] = group_summary["annotation_group"].map(set_ids)
    group_summary["membership_set_target_count"] = group_summary["membership_set_id"].map(target_names.map(len))
    group_summary["membership_set_targets"] = group_summary["membership_set_id"].map(
        target_names.map(lambda values: "|".join(values))
    )
    group_summary["contributing_compounds"] = group_summary["annotation_group"].map(
        target_members.map(lambda values: "|".join(values))
    )

    metric_columns = [
        "upstream_mean_normalized_average_precision",
        "upstream_corrected_p_value",
        "upstream_below_corrected_p",
        "upstream_unique_compounds",
    ]
    metric_audit = property_long.groupby("annotation_group", sort=False)[metric_columns].nunique(dropna=False)
    if metric_audit.gt(1).any(axis=1).any():
        raise ValueError("Group-level audit metrics vary across property rows")
    metrics = property_long.groupby("annotation_group", sort=False)[metric_columns].first().reset_index()
    activity_means = property_long.loc[
        property_long["trait_id"].eq("chem.logp"),
        [
            "annotation_group",
            "activity_nmap_mean",
        ],
    ]
    if activity_means["annotation_group"].duplicated().any():
        raise ValueError("LogP audit contains duplicate target rows")
    metrics = metrics.merge(activity_means, on="annotation_group", validate="one_to_one")
    group_summary = group_summary.merge(metrics, on="annotation_group", validate="one_to_one")
    if not np.allclose(
        group_summary["mean_normalized_average_precision"],
        group_summary["upstream_mean_normalized_average_precision"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Property audit table does not reconcile to target mAP")
    return group_summary.sort_values("annotation_group", kind="stable").reset_index(drop=True)


def load_matched_chemical_properties(metadata_path: Path, matched_ids: list[str]) -> pd.DataFrame:
    """Load the six raw compound properties used by the compound-activity panels."""
    connection = duckdb.connect(str(metadata_path), read_only=True)
    try:
        chemical = connection.execute(
            """
            SELECT
                Metadata_JCP2022 AS jcp2022,
                Metadata_ValidMol AS valid_molecule,
                Metadata_LogP AS logp,
                Metadata_MW AS molecular_weight,
                Metadata_RotatableBonds AS rotatable_bonds,
                Metadata_QED AS qed,
                Metadata_TPSA AS tpsa,
                Metadata_FractionCSP3 AS fraction_csp3
            FROM compound_metadata
            WHERE Metadata_JCP2022 = ANY(?)
            ORDER BY Metadata_JCP2022
            """,
            [matched_ids],
        ).df()
    finally:
        connection.close()
    if chemical["jcp2022"].duplicated().any() or set(chemical["jcp2022"]) != set(matched_ids):
        raise ValueError("Metadata lookup did not return one chemical-property row per matched compound")
    property_columns = [column for _, column, _, _ in COMPOUND_CONTINUOUS_TRAITS]
    if chemical[property_columns].isna().any().any():
        raise ValueError("Matched compounds have missing continuous properties")
    if not chemical["valid_molecule"].fillna(False).astype(bool).all():
        raise ValueError("The matched consistency cohort unexpectedly contains an invalid molecular structure")
    return chemical


def build_matched_compounds(
    scores: pd.DataFrame,
    group_summary: pd.DataFrame,
    compounds: pd.DataFrame,
    activity: pd.DataFrame,
    chemical: pd.DataFrame,
) -> pd.DataFrame:
    """Return one auditable row for each unique compound in the consistency cohort."""
    category_columns = [
        "lipinski_violations",
        "valid_molecule",
        "has_pains",
        "clinical_phase",
        "in_kcgs",
        "in_pkis",
        "in_pkis2",
        "in_chemical_probes_sgc",
    ]
    required = {"jcp2022", "nmap", "active", *category_columns}
    missing = sorted(required - set(compounds.columns))
    if missing:
        raise ValueError(f"Compound distribution table is missing matched-view columns: {missing}")

    matched_ids = set(scores["jcp2022"])
    source = compounds.loc[compounds["jcp2022"].isin(matched_ids), ["jcp2022", "nmap", "active", *category_columns]]
    if source["jcp2022"].duplicated().any() or set(source["jcp2022"]) != matched_ids:
        raise ValueError("Compound distribution lookup did not return one row per matched compound")
    source = source.rename(columns={"nmap": "source_activity_nmap", "active": "source_active_flag"})
    matched = source.merge(activity, on="jcp2022", validate="one_to_one").merge(
        chemical,
        on=["jcp2022", "valid_molecule"],
        validate="one_to_one",
    )
    if len(matched) != EXPECTED_UNIQUE_COMPOUNDS:
        raise ValueError(f"Matched compound count changed from {EXPECTED_UNIQUE_COMPOUNDS} to {len(matched)}")
    if not np.allclose(matched["source_activity_nmap"], matched["activity_nmap"], rtol=0, atol=1e-12):
        raise ValueError("Compound distribution activity nMAP does not reconcile to canonical activity results")
    active_at_threshold = matched["activity_corrected_p_value"].lt(ACTIVITY_THRESHOLD)
    for flag_column in ("source_active_flag", "upstream_active_flag"):
        if not np.array_equal(active_at_threshold.to_numpy(), matched[flag_column].astype(bool).to_numpy()):
            raise ValueError(f"{flag_column} does not match strict activity corrected p-value < 0.10")
    if not active_at_threshold.all():
        raise ValueError("Every matched compound must pass the upstream activity filter")
    matched["active_at_q_lt_0_10"] = active_at_threshold

    set_lookup = group_summary.set_index("annotation_group")["membership_set_id"]
    member_index = scores[["annotation_group", "jcp2022"]].copy()
    member_index["membership_set_id"] = member_index["annotation_group"].map(set_lookup)
    compound_membership = member_index.groupby("jcp2022", sort=True).agg(
        n_displayed_target_labels=("annotation_group", "nunique"),
        n_distinct_membership_sets=("membership_set_id", "nunique"),
        target_labels=("annotation_group", lambda values: "|".join(sorted(set(values)))),
    )
    matched = matched.merge(compound_membership, on="jcp2022", validate="one_to_one")
    if int(matched["n_displayed_target_labels"].sum()) != EXPECTED_MEMBERSHIPS:
        raise ValueError("Matched compound target counts do not reconcile to the membership table")

    property_columns = [column for _, column, _, _ in COMPOUND_CONTINUOUS_TRAITS]
    output_columns = [
        "jcp2022",
        "activity_nmap",
        "activity_corrected_p_value",
        "active_at_q_lt_0_10",
        "n_displayed_target_labels",
        "n_distinct_membership_sets",
        "target_labels",
        *property_columns,
        *category_columns,
    ]
    return matched[output_columns].sort_values("jcp2022", kind="stable").reset_index(drop=True)


def augment_group_activity_audit(
    group_summary: pd.DataFrame,
    scores: pd.DataFrame,
    matched_compounds: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile each target mean to its exact active members and add audit fields."""
    members = scores[["annotation_group", "jcp2022"]].merge(
        matched_compounds[["jcp2022", "activity_nmap", "activity_corrected_p_value", "active_at_q_lt_0_10"]],
        on="jcp2022",
        validate="many_to_one",
    )
    if len(members) != EXPECTED_MEMBERSHIPS:
        raise ValueError("Group activity audit lost target-compound memberships")
    audit = (
        members.groupby("annotation_group", sort=True)
        .agg(
            recomputed_activity_nmap_mean=("activity_nmap", "mean"),
            activity_nmap_median=("activity_nmap", "median"),
            activity_q_max=("activity_corrected_p_value", "max"),
            active_member_count=("active_at_q_lt_0_10", "sum"),
            audited_member_count=("jcp2022", "nunique"),
        )
        .reset_index()
    )
    audit["active_member_percent"] = 100 * audit["active_member_count"] / audit["audited_member_count"]
    result = group_summary.merge(audit, on="annotation_group", validate="one_to_one")
    if not np.allclose(result["activity_nmap_mean"], result["recomputed_activity_nmap_mean"], rtol=0, atol=1e-12):
        raise ValueError("Target mean activity nMAP does not equal the mean across its exact displayed members")
    if not np.array_equal(
        result["n_contributing_compounds"].to_numpy(dtype=int),
        result["audited_member_count"].to_numpy(dtype=int),
    ):
        raise ValueError("Target activity audit member counts do not match consistency group sizes")
    if not result["active_member_count"].eq(result["audited_member_count"]).all():
        raise ValueError("A displayed target contains a member that did not pass the activity filter")
    if not result["active_member_percent"].eq(100.0).all():
        raise ValueError("Displayed target active-member percentages must be 100% by construction")
    return result.drop(columns=["recomputed_activity_nmap_mean", "audited_member_count"])


def build_continuous_table(property_long: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    """Return six target-group mean property tables with quartile summaries."""
    trait_ids = [trait_id for trait_id, _, _ in CONTINUOUS_TRAITS]
    continuous = property_long.loc[property_long["trait_id"].isin(trait_ids)].copy()
    if continuous.groupby("trait_id")["annotation_group"].nunique().to_dict() != {
        trait_id: EXPECTED_TARGET_LABELS for trait_id in trait_ids
    }:
        raise ValueError(
            f"All six continuous properties must cover all {EXPECTED_TARGET_LABELS} eligible target groups"
        )
    if not continuous["n_property_compounds"].eq(continuous["upstream_unique_compounds"]).all():
        raise ValueError("Continuous group property means must use every contributing compound")

    fields = group_summary[
        [
            "annotation_group",
            "membership_set_id",
            "membership_set_target_count",
            "n_contributing_compounds",
            "mean_normalized_average_precision",
            "corrected_p_value",
            "consistent_at_5pct_fdr",
            "activity_nmap_mean",
        ]
    ]
    continuous = continuous[
        ["annotation_group", "trait_id", "property_mean", "property_sd", "property_unique_values"]
    ].merge(fields, on="annotation_group", validate="many_to_one")
    labels = pd.DataFrame(CONTINUOUS_TRAITS, columns=["trait_id", "title", "xlabel"])
    continuous = continuous.merge(labels, on="trait_id", validate="many_to_one")

    pieces = []
    for trait_id, _, _ in CONTINUOUS_TRAITS:
        block = continuous.loc[continuous["trait_id"].eq(trait_id)].copy()
        block["quartile"] = pd.qcut(block["property_mean"], q=4, labels=False, duplicates="drop").astype(int) + 1
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["trait_id", "property_mean", "annotation_group"], kind="stable"
    )


def parse_phase_tokens(value: object) -> set[str]:
    """Return the exact pipe-delimited clinical-stage tokens used by the compound-activity panels."""
    if pd.isna(value):
        return set()
    return {piece.strip() for piece in str(value).split("|") if piece.strip()}


def wilson_bounds(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided Wilson 95% interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z_value = float(norm.ppf(0.975))
    rate = successes / total
    denominator = 1 + z_value**2 / total
    center = (rate + z_value**2 / (2 * total)) / denominator
    half_width = z_value * np.sqrt(rate * (1 - rate) / total + z_value**2 / (4 * total**2)) / denominator
    return float(center - half_width), float(center + half_width)


def build_compound_activity_bins(matched_compounds: pd.DataFrame) -> pd.DataFrame:
    """Build compound-activity-style summaries within the exact 120-compound cohort."""
    if matched_compounds["jcp2022"].duplicated().any():
        raise ValueError("Compound activity bins require one row per unique matched compound")
    if len(matched_compounds) != EXPECTED_UNIQUE_COMPOUNDS:
        raise ValueError("Compound activity bins must use the exact consistency-cohort union")
    if not matched_compounds["active_at_q_lt_0_10"].astype(bool).all():
        raise ValueError("Compound activity bins must remain conditional on the upstream activity filter")

    rows: list[dict[str, object]] = []
    for trait_order, (trait_id, column, title, xlabel) in enumerate(COMPOUND_CONTINUOUS_TRAITS):
        measured = (
            matched_compounds[[column, "activity_nmap", "active_at_q_lt_0_10"]]
            .dropna()
            .sort_values(column, kind="stable")
            .reset_index(drop=True)
        )
        if len(measured) != EXPECTED_UNIQUE_COMPOUNDS:
            raise ValueError(f"Matched compound property {trait_id} does not cover all compounds")
        if measured[column].nunique() < 2:
            raise ValueError(f"Matched compound property {trait_id} has fewer than two values")
        measured["bin"] = pd.qcut(measured[column], q=10, duplicates="drop")
        rho = float(spearmanr(measured[column], measured["activity_nmap"]).statistic)
        for decile, (_, group) in enumerate(measured.groupby("bin", observed=True, sort=True), start=1):
            active_count = int(group["active_at_q_lt_0_10"].astype(bool).sum())
            lower, upper = wilson_bounds(active_count, len(group))
            rows.append(
                {
                    "trait_order": trait_order,
                    "trait_id": trait_id,
                    "title": title,
                    "xlabel": xlabel,
                    "decile": decile,
                    "x": float(group[column].median()),
                    "n": int(len(group)),
                    "active": active_count,
                    "rate": active_count / len(group),
                    "lower": lower,
                    "upper": upper,
                    "nmap_median": float(group["activity_nmap"].median()),
                    "nmap_q25": float(group["activity_nmap"].quantile(0.25)),
                    "nmap_q75": float(group["activity_nmap"].quantile(0.75)),
                    "rho": rho,
                }
            )
    bins = pd.DataFrame.from_records(rows).sort_values(["trait_order", "decile"], kind="stable")
    if not bins["rate"].eq(1.0).all():
        raise ValueError("Every nonempty matched-compound property bin must be 100% active")
    return bins.reset_index(drop=True)


def build_matched_boxplot_distributions(
    matched_compounds: pd.DataFrame,
) -> tuple[
    list[tuple[str, pd.DataFrame]],
    list[tuple[str, pd.DataFrame]],
    list[tuple[str, pd.DataFrame]],
]:
    """Build the fixed compound-activity categories from the unique matched compounds."""
    if matched_compounds["jcp2022"].duplicated().any():
        raise ValueError("Matched categorical distributions require unique compounds")
    lipinski = [
        (str(level), matched_compounds.loc[matched_compounds["lipinski_violations"].eq(level)].copy())
        for level in range(5)
    ]

    phase_tokens = matched_compounds["clinical_phase"].map(parse_phase_tokens)
    clinical = [
        (
            phase,
            matched_compounds.loc[
                phase_tokens.map(lambda values, selected_phase=phase: selected_phase in values)
            ].copy(),
        )
        for phase in PHASE_LEVELS
    ]
    displayed_phase = phase_tokens.map(lambda values: bool(values.intersection(PHASE_LEVELS)))
    clinical.append(("All other", matched_compounds.loc[~displayed_phase].copy()))

    valid_pains = (
        matched_compounds["valid_molecule"].fillna(False).astype(bool) & matched_compounds["has_pains"].notna()
    )
    collection_masks = (
        ("PAINS", valid_pains & matched_compounds["has_pains"].fillna(False).astype(bool)),
        ("KCGS", matched_compounds["in_kcgs"].fillna(False).astype(bool)),
        ("PKIS", matched_compounds["in_pkis"].fillna(False).astype(bool)),
        ("PKIS2", matched_compounds["in_pkis2"].fillna(False).astype(bool)),
        ("Probes", matched_compounds["in_chemical_probes_sgc"].fillna(False).astype(bool)),
    )
    collections = [(label, matched_compounds.loc[mask].copy()) for label, mask in collection_masks]
    collections.append(("All", matched_compounds.copy()))

    for _, distribution in [*lipinski, *clinical, *collections]:
        if not distribution.empty and not distribution["active_at_q_lt_0_10"].astype(bool).all():
            raise ValueError("Every nonempty matched-compound category must be 100% active")
    return lipinski, clinical, collections


def build_categorical_table(
    scores: pd.DataFrame,
    group_summary: pd.DataFrame,
    compounds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize target composition for the three bottom-row property families."""
    required = {
        "jcp2022",
        "lipinski_violations",
        "valid_molecule",
        "has_pains",
        "clinical_phase",
        "in_kcgs",
        "in_pkis",
        "in_pkis2",
        "in_chemical_probes_sgc",
        "activity_nmap",
    }
    missing = sorted(required - set(compounds.columns))
    if missing:
        raise ValueError(f"Compound distribution table is missing columns: {missing}")
    score_ids = set(scores["jcp2022"])
    compound_ids = set(compounds["jcp2022"])
    missing_ids = sorted(score_ids - compound_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"Compound distribution table is missing {len(missing_ids)} contributing compound IDs: {preview}"
        )
    members = scores[["annotation_group", "jcp2022", "normalized_average_precision"]].merge(
        compounds[sorted(required)],
        on="jcp2022",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not members["_merge"].eq("both").all():
        raise ValueError("Categorical metadata join did not cover every contributing compound")
    members = members.drop(columns="_merge")
    nonmissing_fields = sorted(required - {"jcp2022", "clinical_phase"})
    missing_counts = members[nonmissing_fields].isna().sum()
    missing_counts = missing_counts.loc[missing_counts.gt(0)]
    if not missing_counts.empty:
        details = ", ".join(f"{field}={int(count)}" for field, count in missing_counts.items())
        raise ValueError(f"Contributing compound metadata has missing required values: {details}")
    members["phase_tokens"] = members["clinical_phase"].map(parse_phase_tokens)

    summary_lookup = group_summary.set_index("annotation_group")
    records: list[dict[str, object]] = []
    member_audit: list[pd.DataFrame] = []
    for target, target_members in members.groupby("annotation_group", sort=True):
        target_members = target_members.copy()
        group_size = len(target_members)
        target_members["membership_set_id"] = summary_lookup.loc[target, "membership_set_id"]
        group_fields = {
            "annotation_group": target,
            "membership_set_id": target_members["membership_set_id"].iloc[0],
            "n_contributing_compounds": group_size,
            "mean_normalized_average_precision": summary_lookup.loc[target, "mean_normalized_average_precision"],
            "activity_nmap_mean": summary_lookup.loc[target, "activity_nmap_mean"],
            "corrected_p_value": summary_lookup.loc[target, "corrected_p_value"],
            "consistent_at_5pct_fdr": bool(summary_lookup.loc[target, "consistent_at_5pct_fdr"]),
        }

        for level in range(5):
            mask = target_members["lipinski_violations"].eq(level)
            count = int(mask.sum())
            records.append(
                {
                    **group_fields,
                    "family": "lipinski",
                    "category_order": level,
                    "category": str(level),
                    "n_category_members": count,
                    "fraction_category_members": count / group_size,
                }
            )

        displayed_phase = target_members["phase_tokens"].map(lambda values: bool(values.intersection(PHASE_LEVELS)))
        for order, phase in enumerate(PHASE_LEVELS):
            mask = target_members["phase_tokens"].map(lambda values, phase=phase: phase in values)
            count = int(mask.sum())
            records.append(
                {
                    **group_fields,
                    "family": "clinical_phase",
                    "category_order": order,
                    "category": phase,
                    "n_category_members": count,
                    "fraction_category_members": count / group_size,
                }
            )
        all_other_count = int((~displayed_phase).sum())
        records.append(
            {
                **group_fields,
                "family": "clinical_phase",
                "category_order": len(PHASE_LEVELS),
                "category": "All other",
                "n_category_members": all_other_count,
                "fraction_category_members": all_other_count / group_size,
            }
        )

        valid_pains = target_members["valid_molecule"].fillna(False).astype(bool) & target_members["has_pains"].notna()
        collection_masks = [
            ("PAINS", valid_pains & target_members["has_pains"].fillna(False).astype(bool)),
            *[
                (label, target_members[column].fillna(False).astype(bool))
                for label, column in COLLECTION_COLUMNS
                if label != "PAINS"
            ],
            ("All", pd.Series(True, index=target_members.index)),
        ]
        for order, (label, mask) in enumerate(collection_masks):
            count = int(mask.sum())
            denominator = int(valid_pains.sum()) if label == "PAINS" else group_size
            records.append(
                {
                    **group_fields,
                    "family": "collections",
                    "category_order": order,
                    "category": label,
                    "n_category_members": count,
                    "fraction_category_members": count / denominator if denominator else np.nan,
                }
            )
        member_audit.append(
            target_members[
                [
                    "annotation_group",
                    "jcp2022",
                    "membership_set_id",
                    "normalized_average_precision",
                    "activity_nmap",
                    "lipinski_violations",
                    "clinical_phase",
                    "has_pains",
                    "in_kcgs",
                    "in_pkis",
                    "in_pkis2",
                    "in_chemical_probes_sgc",
                ]
            ]
        )

    categorical = pd.DataFrame.from_records(records).sort_values(
        ["family", "category_order", "annotation_group"], kind="stable"
    )
    membership = pd.concat(member_audit, ignore_index=True).sort_values(["annotation_group", "jcp2022"], kind="stable")
    unique_members = members.drop_duplicates("jcp2022", keep="first").copy()
    unique_members["phase_tokens"] = unique_members["clinical_phase"].map(parse_phase_tokens)
    unique_counts: dict[tuple[str, str], int] = {}
    for level in range(5):
        unique_counts[("lipinski", str(level))] = int(unique_members["lipinski_violations"].eq(level).sum())
    unique_displayed_phase = unique_members["phase_tokens"].map(lambda values: bool(values.intersection(PHASE_LEVELS)))
    for phase in PHASE_LEVELS:
        unique_counts[("clinical_phase", phase)] = int(
            unique_members["phase_tokens"].map(lambda values, phase=phase: phase in values).sum()
        )
    unique_counts[("clinical_phase", "All other")] = int((~unique_displayed_phase).sum())
    unique_valid_pains = (
        unique_members["valid_molecule"].fillna(False).astype(bool) & unique_members["has_pains"].notna()
    )
    unique_counts[("collections", "PAINS")] = int(
        (unique_valid_pains & unique_members["has_pains"].fillna(False).astype(bool)).sum()
    )
    for label, column in COLLECTION_COLUMNS:
        if label != "PAINS":
            unique_counts[("collections", label)] = int(unique_members[column].fillna(False).astype(bool).sum())
    unique_counts[("collections", "All")] = int(len(unique_members))
    categorical["n_unique_compounds_in_category"] = [
        unique_counts[(family, category)]
        for family, category in categorical[["family", "category"]].itertuples(index=False, name=None)
    ]
    return categorical, membership


def point_sizes(counts: pd.Series | np.ndarray) -> np.ndarray:
    """Map contributor counts to modest marker areas."""
    values = np.asarray(counts, dtype=float)
    return 12.0 + 3.2 * values


def scatter_consistency(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    counts: np.ndarray,
    consistent: np.ndarray,
    *,
    jitter_labels: list[str] | None = None,
) -> None:
    """Draw filled significant and hollow nonsignificant target-group points."""
    x_plot = np.asarray(x, dtype=float).copy()
    if jitter_labels is not None:
        x_plot = x_plot + np.asarray([stable_jitter(label) for label in jitter_labels])
    sizes = point_sizes(counts)
    consistent = np.asarray(consistent, dtype=bool)
    axis.scatter(
        x_plot[~consistent],
        np.asarray(y)[~consistent],
        s=sizes[~consistent],
        facecolors="white",
        edgecolors=GRAY,
        linewidths=0.7,
        alpha=0.9,
        zorder=2,
    )
    axis.scatter(
        x_plot[consistent],
        np.asarray(y)[consistent],
        s=sizes[consistent],
        facecolors=BLUE,
        edgecolors=BLUE,
        linewidths=0.6,
        alpha=0.82,
        zorder=3,
    )


def style_axis(axis: plt.Axes, fontsize: float = 6.0) -> None:
    """Apply the shared mechanical axis style."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(0.7)
    axis.tick_params(labelsize=fontsize, length=2.4, pad=1.5)


def apply_truncated_activity_y_axis(
    axis: plt.Axes,
    y_limits: tuple[float, float],
    y_ticks: tuple[float, ...],
) -> None:
    """Apply a trimmed activity scale with an explicit double-slash break."""
    axis.set_ylim(*y_limits)
    axis.set_yticks(y_ticks)
    break_marker = [(-1.0, -0.65), (1.0, 0.65)]
    for center in (0.022, 0.068):
        (break_line,) = axis.plot(
            [0],
            [center],
            transform=axis.transAxes,
            marker=break_marker,
            markersize=6.0,
            markeredgecolor=DARK_GRAY,
            markeredgewidth=0.9,
            markerfacecolor="none",
            linestyle="none",
            clip_on=False,
            zorder=10,
        )
        break_line.set_gid("truncated-activity-y-axis-break")


def draw_continuous_panel(
    figure: plt.Figure,
    slot,
    block: pd.DataFrame,
    *,
    show_ylabel: bool,
    y_limits: tuple[float, float],
    overall_consistent_rate: float,
    percentage_unit_label: str,
    outcome_column: str = "mean_normalized_average_precision",
    outcome_ylabel: str = "Phenotypic\nconsistency",
    rho_x: float = 0.02,
    rho_y: float = 0.95,
    rho_vertical_alignment: str = "top",
    truncated_activity_y: bool = False,
    y_ticks: tuple[float, ...] | None = None,
) -> None:
    """Draw one property-mean outcome scatter with Consistent percent above."""
    nested = slot.subgridspec(2, 1, height_ratios=(0.34, 0.66), hspace=0.08)
    rate_axis = figure.add_subplot(nested[0])
    map_axis = figure.add_subplot(nested[1], sharex=rate_axis)
    block = block.sort_values(["property_mean", "annotation_group"], kind="stable")

    quartiles = (
        block.groupby("quartile", sort=True)
        .agg(
            x=("property_mean", "median"),
            consistent=("consistent_at_5pct_fdr", "sum"),
            n=("annotation_group", "size"),
        )
        .reset_index()
    )
    quartiles["rate"] = 100 * quartiles["consistent"] / quartiles["n"]
    rate_axis.plot(
        quartiles["x"],
        quartiles["rate"],
        color=BLUE,
        marker="o",
        markersize=2.5,
        linewidth=1.0,
    )
    rate_axis.axhline(100 * overall_consistent_rate, color=GRAY, linestyle="--", linewidth=0.6)
    for row in quartiles.itertuples(index=False):
        rate_axis.text(
            row.x,
            min(98, row.rate + 8),
            f"{row.rate:.0f}%\n{int(row.consistent)}/{int(row.n)}",
            ha="center",
            va="bottom",
            fontsize=QUARTILE_LABEL_PT,
            color=GRAY,
            linespacing=0.9,
        )
    rate_axis.set_ylim(0, 138)
    rate_axis.set_yticks([0, 50, 100])
    rate_axis.tick_params(axis="x", labelbottom=False)
    rate_axis.set_title(str(block["title"].iloc[0]), fontsize=TITLE_PT, fontweight="bold", loc="left")
    if show_ylabel:
        rate_axis.set_ylabel("Consistent\n(%)", color=BLUE, fontsize=STACKED_Y_LABEL_PT, labelpad=2)
    style_axis(rate_axis, fontsize=TICK_PT)

    x_values = block["property_mean"].to_numpy(dtype=float)
    y_values = block[outcome_column].to_numpy(dtype=float)
    scatter_consistency(
        map_axis,
        x_values,
        y_values,
        block["n_contributing_compounds"].to_numpy(dtype=float),
        block["consistent_at_5pct_fdr"].to_numpy(dtype=bool),
    )
    if np.unique(x_values).size >= 2:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        map_axis.plot(x_line, intercept + slope * x_line, color=DARK_GRAY, linewidth=0.8, zorder=1)
    rho = float(spearmanr(x_values, y_values).statistic)
    map_axis.text(
        rho_x,
        rho_y,
        f"rho={rho:.2f}",
        transform=map_axis.transAxes,
        ha="left",
        va=rho_vertical_alignment,
        fontsize=TICK_PT,
        color=GRAY,
    )
    map_axis.axhline(0, color=LIGHT_GRAY, linestyle="--", linewidth=0.6, zorder=0)
    if truncated_activity_y:
        if y_ticks is None:
            raise ValueError("A truncated activity y-axis requires explicit ticks")
        apply_truncated_activity_y_axis(map_axis, y_limits, y_ticks)
    else:
        map_axis.set_ylim(*y_limits)
    map_axis.set_xlabel(str(block["xlabel"].iloc[0]), fontsize=AXIS_LABEL_PT, labelpad=1.5)
    if show_ylabel:
        map_axis.set_ylabel(outcome_ylabel, color=DARK_GRAY, fontsize=STACKED_Y_LABEL_PT, labelpad=2)
    style_axis(map_axis, fontsize=TICK_PT)


# The same three category axis labels in every property figure.
CATEGORICAL_PANEL_SPECS = (
    ("lipinski", "Lipinski composition", "Number of violations"),
    ("clinical_phase", "Clinical-phase composition", "Clinical development stage"),
    ("collections", "PAINS/collections", "Curated collection"),
)


def draw_categorical_panel(
    axis: plt.Axes,
    block: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    show_ylabel: bool,
    y_limits: tuple[float, float],
    outcome_column: str = "mean_normalized_average_precision",
    outcome_ylabel: str = "Phenotypic\nconsistency",
    annotation_y: float | None = None,
    annotation_vertical_alignment: str = "top",
    truncated_activity_y: bool = False,
    y_ticks: tuple[float, ...] | None = None,
) -> None:
    """Draw category composition as target-group bubble strips."""
    category_orders = sorted(block["category_order"].unique())
    if annotation_y is None:
        annotation_y = y_limits[1] - 0.012
    for position, category_order in enumerate(category_orders, start=1):
        category = block.loc[block["category_order"].eq(category_order)].copy()
        present = category.loc[category["n_category_members"].gt(0)].copy()
        if present.empty:
            axis.text(
                position,
                annotation_y,
                "0%\n0/0",
                ha="center",
                va=annotation_vertical_alignment,
                fontsize=QUARTILE_LABEL_PT,
                color=GRAY,
                linespacing=0.95,
            )
            continue
        count_consistent = int(present["consistent_at_5pct_fdr"].sum())
        group_count = len(present)
        rate = 100 * count_consistent / group_count
        axis.text(
            position,
            annotation_y,
            f"{rate:.0f}%\n{count_consistent}/{group_count}",
            ha="center",
            va=annotation_vertical_alignment,
            fontsize=QUARTILE_LABEL_PT,
            color=GRAY,
            linespacing=0.95,
        )
        scatter_consistency(
            axis,
            np.full(group_count, float(position)),
            present[outcome_column].to_numpy(dtype=float),
            20 * present["fraction_category_members"].to_numpy(dtype=float),
            present["consistent_at_5pct_fdr"].to_numpy(dtype=bool),
            jitter_labels=[f"{row.annotation_group}:{row.category}" for row in present.itertuples(index=False)],
        )
        median = float(present[outcome_column].median())
        axis.plot([position - 0.22, position + 0.22], [median, median], color="#111827", linewidth=1.2, zorder=4)

    axis.axhline(0, color=LIGHT_GRAY, linestyle="--", linewidth=0.6, zorder=0)
    axis.set_xlim(0.5, len(category_orders) + 0.5)
    if truncated_activity_y:
        if y_ticks is None:
            raise ValueError("A truncated activity y-axis requires explicit ticks")
        apply_truncated_activity_y_axis(axis, y_limits, y_ticks)
    else:
        axis.set_ylim(*y_limits)
    tick_labels = []
    for category_order in category_orders:
        category = block.loc[block["category_order"].eq(category_order)]
        label = str(category["category"].iloc[0])
        unique_compounds = int(category["n_unique_compounds_in_category"].iloc[0])
        tick_labels.append(f"{label}\n({unique_compounds})")
    axis.set_xticks(np.arange(1, len(category_orders) + 1), tick_labels)
    axis.set_title(title, fontsize=TITLE_PT, fontweight="bold", loc="left")
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_PT, labelpad=1.5)
    if show_ylabel:
        axis.set_ylabel(outcome_ylabel, color=DARK_GRAY, fontsize=Y_LABEL_PT, labelpad=2)
    style_axis(axis, fontsize=CATEGORY_TICK_PT)


def render_main_figure(
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_dir: Path,
    stem: str,
    dpi: int,
    *,
    unit_label: str,
    title_suffix: str = "",
    extra_caption: str = "",
) -> dict[str, str]:
    """Render the requested full 3x3 target-group property figure."""
    y_min = min(-0.035, float(group_summary["mean_normalized_average_precision"].min()) - 0.025)
    y_max = max(0.47, float(group_summary["mean_normalized_average_precision"].max()) + 0.12)
    y_limits = (y_min, y_max)
    overall_rate = float(group_summary["consistent_at_5pct_fdr"].mean())
    percentage_unit_label = "labels" if unit_label == "target label" else "sets"
    unit_plural = "target labels" if unit_label == "target label" else "membership sets"
    consistent_count = int(group_summary["consistent_at_5pct_fdr"].sum())
    deduplicated = unit_label == "membership set"

    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, PROPERTY_FIGURE_PANEL_HEIGHT_IN))
    outer = figure.add_gridspec(3, 3, hspace=0.57, wspace=0.34)
    for index, (trait_id, _, _) in enumerate(CONTINUOUS_TRAITS):
        draw_continuous_panel(
            figure,
            outer[index // 3, index % 3],
            continuous.loc[continuous["trait_id"].eq(trait_id)],
            show_ylabel=index % 3 == 0,
            y_limits=y_limits,
            overall_consistent_rate=overall_rate,
            percentage_unit_label=percentage_unit_label,
        )

    for column, (family, title, xlabel) in enumerate(CATEGORICAL_PANEL_SPECS):
        draw_categorical_panel(
            figure.add_subplot(outer[2, column]),
            categorical.loc[categorical["family"].eq(family)],
            title=title,
            xlabel=xlabel,
            show_ylabel=column == 0,
            y_limits=y_limits,
        )

    figure_title = (
        "OpenTargets consistency - distinct membership-set sensitivity"
        if title_suffix
        else "OpenTargets group composition and phenotypic consistency"
    )
    # One anchor for both variants: the deduplicated view sat at 9.8pt/x=0.12 and read
    # as a different figure beside its siblings.
    title_artist = figure.suptitle(
        figure_title,
        fontsize=SUPTITLE_PT,
        fontweight="bold",
        x=0.065,
        y=0.97,
        ha="left",
    )
    caption = (
        f"Each point is one OpenTargets {unit_label} with at least {MIN_GROUP_SIZE} active compounds "
        f"(n = {len(group_summary)} {unit_plural}; {group_summary['membership_set_id'].nunique()} distinct membership "
        f"sets; {EXPECTED_UNIQUE_COMPOUNDS} compounds and {EXPECTED_MEMBERSHIPS} target-compound memberships). "
        "The y-axis is group consistency nMAP; continuous x-values are means across the same compounds. Gray lines "
        f"are unweighted OLS fits and rho is Spearman correlation across displayed {unit_plural}. Blue lines give "
        "the percentage with consistency q < 0.05 within property quartiles; counts are printed above each quartile, "
        f"and dashed lines mark the displayed-subset baseline ({consistent_count}/{len(group_summary)}). Filled points "
        + (
            "inherit the corresponding target-label result; no set-level test or FDR correction was performed. "
            if deduplicated
            else f"pass Benjamini-Hochberg correction across all {EXPECTED_TESTED_GROUPS} tested groups. "
        )
        + "The q values test group consistency, not the displayed property associations. In categorical panels, "
        "marker area is the within-group category fraction; labels give unique compounds and consistency counts. "
        "Clinical and collection categories can overlap; All other includes missing and nondisplayed clinical "
        "labels, and Probes combines Chemical Probes and SGC. Target memberships overlap." + extra_caption
    )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(caption)
    figure.set_size_inches(PROPERTY_FIGURE_WIDTH_IN, figure_height)
    title_artist.set_y(1 - 0.30 / figure_height)
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=CAPTION_PT,
        color=GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    save_figure(figure, output_dir, stem, dpi)
    return {stem: caption}


def render_group_activity_figure(
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    """Render target-level mean activity for the exact consistency memberships."""
    y_limits = GROUP_ACTIVITY_Y_LIMITS
    overall_consistent_rate = float(group_summary["consistent_at_5pct_fdr"].mean())
    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, PROPERTY_FIGURE_PANEL_HEIGHT_IN))
    outer = figure.add_gridspec(3, 3, hspace=0.57, wspace=0.34)
    for index, (trait_id, _, _) in enumerate(CONTINUOUS_TRAITS):
        draw_continuous_panel(
            figure,
            outer[index // 3, index % 3],
            continuous.loc[continuous["trait_id"].eq(trait_id)],
            show_ylabel=index % 3 == 0,
            y_limits=y_limits,
            overall_consistent_rate=overall_consistent_rate,
            percentage_unit_label="labels",
            outcome_column="activity_nmap_mean",
            outcome_ylabel="Mean phenotypic\nactivity (nMAP)",
            rho_x=0.055,
            rho_y=0.05,
            rho_vertical_alignment="bottom",
            truncated_activity_y=True,
            y_ticks=GROUP_ACTIVITY_Y_TICKS,
        )

    for column, (family, title, xlabel) in enumerate(CATEGORICAL_PANEL_SPECS):
        draw_categorical_panel(
            figure.add_subplot(outer[2, column]),
            categorical.loc[categorical["family"].eq(family)],
            title=title,
            xlabel=xlabel,
            show_ylabel=column == 0,
            y_limits=y_limits,
            outcome_column="activity_nmap_mean",
            outcome_ylabel="Phenotypic\nactivity",
            annotation_y=0.805,
            annotation_vertical_alignment="bottom",
            truncated_activity_y=True,
            y_ticks=GROUP_ACTIVITY_Y_TICKS,
        )

    title_artist = figure.suptitle(
        "OpenTargets group composition and mean phenotypic activity",
        fontsize=SUPTITLE_PT,
        fontweight="bold",
        x=0.065,
        y=0.97,
        ha="left",
    )
    caption = (
        f"Each point is one of the {len(group_summary)} OpenTargets target labels in the consistency view "
        f"({EXPECTED_MEMBERSHIPS} memberships; {EXPECTED_UNIQUE_COMPOUNDS} unique compounds). The y-axis is mean "
        "compound activity nMAP across the target's members; x-values and category fractions use the same members. "
        f"All members passed activity corrected p < {ACTIVITY_THRESHOLD:.2f} before targets with fewer than "
        f"{MIN_GROUP_SIZE} active members were removed. Blue curves and filled points encode target-group "
        "consistency q < 0.05; gray lines are unweighted OLS fits and rho is Spearman correlation. Marker area is the "
        "within-target category fraction. Memberships overlap, and selection restricts the activity range. Activity "
        "axes begin at 0.80; double slashes mark the truncated scale."
    )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(caption)
    figure.set_size_inches(PROPERTY_FIGURE_WIDTH_IN, figure_height)
    title_artist.set_y(1 - 0.30 / figure_height)
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=CAPTION_PT,
        color=GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    save_figure(figure, output_dir, GROUP_ACTIVITY_STEM, dpi)
    return {GROUP_ACTIVITY_STEM: caption}


def draw_matched_compound_continuous_panel(
    figure: plt.Figure,
    slot,
    block: pd.DataFrame,
    *,
    show_ylabel: bool,
) -> None:
    """Draw one compound-activity-style continuous panel for the matched compounds."""
    nested = slot.subgridspec(2, 1, hspace=0.06)
    activity_axis = figure.add_subplot(nested[0])
    nmap_axis = figure.add_subplot(nested[1], sharex=activity_axis)
    block = block.sort_values("decile", kind="stable")
    x_values = block["x"].to_numpy(dtype=float)
    activity_axis.fill_between(
        x_values,
        100 * block["lower"].to_numpy(dtype=float),
        100 * block["upper"].to_numpy(dtype=float),
        color=LIGHT_BLUE,
        alpha=0.55,
    )
    activity_axis.axhline(100, color=GRAY, linestyle="--", linewidth=0.6, zorder=1)
    activity_axis.plot(
        x_values,
        100 * block["rate"].to_numpy(dtype=float),
        color=BLUE,
        marker="o",
        markersize=2.2,
        linewidth=1.0,
        zorder=2,
    )
    activity_axis.set_ylim(0, 138)
    activity_axis.set_yticks([0, 50, 100])
    activity_axis.set_title(str(block["title"].iloc[0]), fontsize=TITLE_PT, fontweight="bold", loc="left")
    activity_axis.tick_params(axis="x", labelbottom=False)

    nmap_axis.fill_between(
        x_values,
        block["nmap_q25"].to_numpy(dtype=float),
        block["nmap_q75"].to_numpy(dtype=float),
        color="#E5E7EB",
        alpha=0.82,
    )
    nmap_axis.plot(
        x_values,
        block["nmap_median"].to_numpy(dtype=float),
        color=DARK_GRAY,
        marker="s",
        markersize=2.1,
        linewidth=0.9,
    )
    apply_truncated_activity_y_axis(
        nmap_axis,
        COMPOUND_ACTIVITY_Y_LIMITS,
        COMPOUND_ACTIVITY_Y_TICKS,
    )
    nmap_axis.set_xlabel(str(block["xlabel"].iloc[0]), fontsize=AXIS_LABEL_PT, labelpad=1.5)
    nmap_axis.text(
        0.055,
        0.05,
        f"rho={float(block['rho'].iloc[0]):.2f}",
        transform=nmap_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=TICK_PT,
        color=GRAY,
    )
    if show_ylabel:
        activity_axis.set_ylabel("Active\n(%)", color=BLUE, fontsize=STACKED_Y_LABEL_PT, labelpad=2)
        nmap_axis.set_ylabel("Phenotypic\nactivity", color=DARK_GRAY, fontsize=STACKED_Y_LABEL_PT, labelpad=2)
    style_axis(activity_axis, fontsize=TICK_PT)
    style_axis(nmap_axis, fontsize=TICK_PT)


def draw_matched_compound_box_panel(
    axis: plt.Axes,
    distributions: list[tuple[str, pd.DataFrame]],
    *,
    title: str,
    xlabel: str,
    show_ylabel: bool,
) -> None:
    """Draw one fixed-category activity panel while retaining empty categories."""
    positions = np.arange(1, len(distributions) + 1)
    present = [
        (position, label, frame) for position, (label, frame) in zip(positions, distributions) if not frame.empty
    ]
    if present:
        boxplot = axis.boxplot(
            [frame["activity_nmap"].to_numpy(dtype=float) for _, _, frame in present],
            positions=[position for position, _, _ in present],
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor="#D9EBF6", edgecolor=BLUE, linewidth=1.0)
        for median in boxplot["medians"]:
            median.set(color="#111827", linewidth=1.25)
        for artist in [*boxplot["whiskers"], *boxplot["caps"]]:
            artist.set(color=BLUE, linewidth=0.8)

    tick_labels: list[str] = []
    for position, (label, distribution) in zip(positions, distributions):
        count = len(distribution)
        tick_labels.append(f"{label}\n({count})")
        if count:
            active_rate = 100 * distribution["active_at_q_lt_0_10"].astype(bool).mean()
            axis.text(
                position,
                1.075,
                f"{active_rate:.0f}%",
                ha="center",
                va="center",
                fontsize=TICK_PT,
                color=GRAY,
            )
        else:
            axis.text(position, 1.075, "--", ha="center", va="center", fontsize=TICK_PT, color=GRAY)
            axis.text(
                position,
                0.54,
                "No matched\ncompounds",
                ha="center",
                va="center",
                fontsize=ANNOTATION_PT,
                color=GRAY,
                linespacing=0.95,
            )
    axis.set_xlim(0.5, len(distributions) + 0.5)
    apply_truncated_activity_y_axis(
        axis,
        COMPOUND_ACTIVITY_BOX_Y_LIMITS,
        COMPOUND_ACTIVITY_Y_TICKS,
    )
    axis.set_xticks(positions, tick_labels)
    axis.set_title(title, fontsize=TITLE_PT, fontweight="bold", loc="left")
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_PT, labelpad=1.5)
    if show_ylabel:
        axis.set_ylabel("Phenotypic\nactivity", color=DARK_GRAY, fontsize=Y_LABEL_PT, labelpad=2)
    style_axis(axis, fontsize=CATEGORY_TICK_PT)


def render_matched_compound_activity_figure(
    activity_bins: pd.DataFrame,
    matched_compounds: pd.DataFrame,
    activity_family: dict[str, int],
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    """Render the compound-activity view after exact filtering to the unique consistency compounds."""
    lipinski, clinical, collections = build_matched_boxplot_distributions(matched_compounds)
    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, PROPERTY_FIGURE_PANEL_HEIGHT_IN))
    outer = figure.add_gridspec(3, 3, hspace=0.57, wspace=0.34)
    for index, (trait_id, _, _, _) in enumerate(COMPOUND_CONTINUOUS_TRAITS):
        draw_matched_compound_continuous_panel(
            figure,
            outer[index // 3, index % 3],
            activity_bins.loc[activity_bins["trait_id"].eq(trait_id)],
            show_ylabel=index % 3 == 0,
        )

    bottom_specs = (
        (lipinski, "Lipinski violations", "Number of violations"),
        (clinical, "Clinical phase", "Clinical development stage"),
        (collections, "PAINS/collections", "Curated collection"),
    )
    for column, (distributions, title, xlabel) in enumerate(bottom_specs):
        draw_matched_compound_box_panel(
            figure.add_subplot(outer[2, column]),
            distributions,
            title=title,
            xlabel=xlabel,
            show_ylabel=column == 0,
        )

    title_artist = figure.suptitle(
        "Phenotypic activity in the OpenTargets consistency cohort",
        fontsize=SUPTITLE_PT,
        fontweight="bold",
        x=0.065,
        y=0.97,
        ha="left",
    )
    caption = (
        f"Compound activity for the {len(matched_compounds)} unique compounds in the displayed OpenTargets cohort, "
        f"deduplicated from {EXPECTED_MEMBERSHIPS} target-compound memberships. Activity q-values were calculated "
        f"in the full {activity_family['n_compounds']:,}-compound family; {activity_family['n_active']:,} compounds "
        f"with corrected p < {ACTIVITY_THRESHOLD} were retained before targets with fewer than {MIN_GROUP_SIZE} "
        "active members were removed. Activity q-values were not recomputed in this subset, so each nonempty bin "
        "and category is 100% active by selection. Gray curves and boxplots show compound-level activity nMAP; "
        "continuous properties use up to ten quantile bins and labels report Spearman rho. Clinical and collection "
        "categories can overlap; All other includes missing and nondisplayed clinical labels, and Probes combines "
        "Chemical Probes and SGC. Activity axes begin at 0.40; double slashes mark the truncated scale."
    )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(caption)
    figure.set_size_inches(PROPERTY_FIGURE_WIDTH_IN, figure_height)
    title_artist.set_y(1 - 0.30 / figure_height)
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=CAPTION_PT,
        color=GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    save_figure(figure, output_dir, COMPOUND_ACTIVITY_STEM, dpi)
    return {COMPOUND_ACTIVITY_STEM: caption}


def deduplicate_membership_sets(
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse target labels having identical constituent-compound sets."""
    consistency_check = group_summary.groupby("membership_set_id").agg(
        n_map=("mean_normalized_average_precision", "nunique"),
        n_q=("corrected_p_value", "nunique"),
        n_flag=("consistent_at_5pct_fdr", "nunique"),
        n_size=("n_contributing_compounds", "nunique"),
    )
    if consistency_check.gt(1).any(axis=1).any():
        raise ValueError("Identical membership sets have conflicting consistency results")

    representatives = (
        group_summary.sort_values("annotation_group", kind="stable")
        .drop_duplicates("membership_set_id", keep="first")
        .copy()
    )
    representatives["annotation_group"] = representatives["membership_set_targets"].str.replace("|", "/", regex=False)
    keep_ids = set(representatives["membership_set_id"])

    continuous_dedup = continuous.sort_values("annotation_group", kind="stable").drop_duplicates(
        ["trait_id", "membership_set_id"], keep="first"
    )
    continuous_dedup["annotation_group"] = continuous_dedup["membership_set_id"].map(
        representatives.set_index("membership_set_id")["annotation_group"]
    )
    pieces = []
    for trait_id, _, _ in CONTINUOUS_TRAITS:
        block = continuous_dedup.loc[continuous_dedup["trait_id"].eq(trait_id)].copy()
        block["quartile"] = pd.qcut(block["property_mean"], q=4, labels=False, duplicates="drop").astype(int) + 1
        pieces.append(block)
    continuous_dedup = pd.concat(pieces, ignore_index=True)

    categorical_dedup = categorical.loc[categorical["membership_set_id"].isin(keep_ids)].copy()
    categorical_dedup = categorical_dedup.sort_values("annotation_group", kind="stable").drop_duplicates(
        ["family", "category_order", "membership_set_id"], keep="first"
    )
    categorical_dedup["annotation_group"] = categorical_dedup["membership_set_id"].map(
        representatives.set_index("membership_set_id")["annotation_group"]
    )
    return continuous_dedup, categorical_dedup, representatives


def summarize_distinct_set_overlap(
    scores: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> dict[str, int]:
    """Quantify residual compound sharing after exact-set deduplication."""
    set_lookup = group_summary.set_index("annotation_group")["membership_set_id"]
    memberships = scores[["annotation_group", "jcp2022"]].copy()
    memberships["membership_set_id"] = memberships["annotation_group"].map(set_lookup)
    if memberships["membership_set_id"].isna().any():
        raise ValueError("Could not map every displayed membership to a membership-set identifier")
    memberships = memberships.drop_duplicates(["membership_set_id", "jcp2022"])
    set_members = {
        membership_set_id: frozenset(block["jcp2022"])
        for membership_set_id, block in memberships.groupby("membership_set_id", sort=True)
    }
    compound_set_counts = memberships.groupby("jcp2022")["membership_set_id"].nunique()
    set_ids = sorted(set_members)
    overlapping_pairs = sum(
        bool(set_members[first].intersection(set_members[second]))
        for index, first in enumerate(set_ids)
        for second in set_ids[index + 1 :]
    )
    return {
        "compounds_in_multiple_distinct_sets": int(compound_set_counts.gt(1).sum()),
        "overlapping_distinct_set_pairs": int(overlapping_pairs),
        "total_distinct_set_pairs": int(len(set_ids) * (len(set_ids) - 1) // 2),
    }


def render_activity_companion(group_summary: pd.DataFrame, output_dir: Path, dpi: int) -> dict[str, str]:
    """Render target-group mean activity nMAP against consistency nMAP."""
    x = group_summary["activity_nmap_mean"].to_numpy(dtype=float)
    y = group_summary["mean_normalized_average_precision"].to_numpy(dtype=float)
    rho = float(spearmanr(x, y).statistic)
    distinct = group_summary.sort_values("annotation_group", kind="stable").drop_duplicates(
        "membership_set_id", keep="first"
    )
    distinct_rho = float(
        spearmanr(distinct["activity_nmap_mean"], distinct["mean_normalized_average_precision"]).statistic
    )
    caption = (
        f"Each point is one OpenTargets label with at least {MIN_GROUP_SIZE} active compounds; filled points pass "
        f"5% FDR in the {EXPECTED_TESTED_GROUPS}-group consistency family and marker area gives group size. Spearman "
        f"rho between mean activity nMAP and consistency nMAP is {distinct_rho:.2f} across {len(distinct)} distinct "
        f"membership sets and {rho:.2f} across all {len(group_summary)} target labels. Sets overlap, and both metrics "
        "use the same activity-filtered compounds."
    )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=4.9,
        panel_height_in=4.0,
        caption_points=6.0,
    )
    figure, axis = plt.subplots(figsize=(4.9, figure_height))
    scatter_consistency(
        axis,
        x,
        y,
        group_summary["n_contributing_compounds"].to_numpy(dtype=float),
        group_summary["consistent_at_5pct_fdr"].to_numpy(dtype=bool),
    )
    axis.text(
        0.03,
        0.96,
        f"Deduplicated: rho={distinct_rho:.2f} ({len(distinct)} compound sets)\n"
        f"All labels: rho={rho:.2f} ({len(group_summary)} target labels)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=GRAY,
    )
    label_rows = distinct.nlargest(4, "mean_normalized_average_precision")
    for row in label_rows.itertuples(index=False):
        target_label = row.membership_set_targets.replace("|", "/")
        if len(target_label) > 18:
            target_label = target_label.replace("/", "/\n", 1)
        axis.annotate(
            target_label,
            (row.activity_nmap_mean, row.mean_normalized_average_precision),
            xytext=(-3 if row.activity_nmap_mean > np.quantile(x, 0.8) else 3, 3),
            textcoords="offset points",
            ha="right" if row.activity_nmap_mean > np.quantile(x, 0.8) else "left",
            fontsize=6.0,
            color=DARK_GRAY,
        )
    axis.axhline(0, color=LIGHT_GRAY, linestyle="--", linewidth=0.7)
    axis.set_xlabel("Mean phenotypic activity nMAP of target members", fontsize=8.0)
    axis.set_ylabel("Target phenotypic consistency (nMAP)", fontsize=8.0)
    axis.set_title("Activity and consistency across OpenTargets groups", fontsize=10.0, fontweight="bold", loc="left")
    style_axis(axis, fontsize=7.0)
    x_padding = 0.025 * (x.max() - x.min())
    axis.set_xlim(x.min() - x_padding, x.max() + 3.0 * x_padding)
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.15,
        right=0.98,
        top=1 - 0.48 / figure_height,
        bottom=caption_fraction + 0.50 / figure_height,
    )
    save_figure(figure, output_dir, ACTIVITY_COMPANION_STEM, dpi)
    return {ACTIVITY_COMPANION_STEM: caption}


def dataframe_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    """Return a stable logical digest for one exported table."""
    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.command()
def render(
    copairs_db: Path = typer.Option(
        PROCESSED_DATA_DIR / "copairs_results.duckdb",
        exists=True,
        dir_okay=False,
        help="Canonical copairs results database.",
    ),
    group_properties: Path = typer.Option(
        PROCESSED_DATA_DIR / "trait-activity-enrichment" / DATASET / "consistency_group_properties.csv",
        exists=True,
        dir_okay=False,
        help="Summary-declared consistency group-property table.",
    ),
    group_properties_summary: Path = typer.Option(
        PROCESSED_DATA_DIR / "trait-activity-enrichment" / DATASET / "summary.json",
        exists=True,
        dir_okay=False,
        help="Owning consistency-analysis summary.",
    ),
    compound_properties: Path = typer.Option(
        PROCESSED_DATA_DIR / "property-activity-oddsratios" / DATASET / "compound_activity_nmap_distributions.parquet",
        exists=True,
        dir_okay=False,
        help="Summary-declared compound property and activity table.",
    ),
    compound_properties_summary: Path = typer.Option(
        PROCESSED_DATA_DIR / "property-activity-oddsratios" / DATASET / "summary.json",
        exists=True,
        dir_okay=False,
        help="Owning property-analysis summary.",
    ),
    metadata_db: Path = typer.Option(
        INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb",
        exists=True,
        dir_okay=False,
        help="Canonical compound metadata database for raw continuous properties.",
    ),
    output_dir: Path = typer.Option(OUTPUT_DIR, help="Canonical figure output directory."),
    dpi: int = typer.Option(300, min=72, max=1200, help="Raster output DPI."),
) -> None:
    """Generate consistency and matched-activity views with exact-cohort audits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        scores,
        results,
        property_long,
        compounds,
        activity,
        activity_family,
        paths,
        tested_group_count,
    ) = load_inputs(
        copairs_db,
        group_properties,
        group_properties_summary,
        compound_properties,
        compound_properties_summary,
        metadata_db,
    )

    group_summary = build_group_summary(scores, results, property_long)
    chemical = load_matched_chemical_properties(metadata_db, sorted(scores["jcp2022"].unique()))
    matched_compounds = build_matched_compounds(scores, group_summary, compounds, activity, chemical)
    group_summary = augment_group_activity_audit(group_summary, scores, matched_compounds)
    continuous = build_continuous_table(property_long, group_summary)
    categorical, membership = build_categorical_table(scores, group_summary, matched_compounds)
    compound_activity_bins = build_compound_activity_bins(matched_compounds)
    continuous_dedup, categorical_dedup, group_summary_dedup = deduplicate_membership_sets(
        continuous, categorical, group_summary
    )
    overlap_summary = summarize_distinct_set_overlap(scores, group_summary)
    if int(group_summary["consistent_at_5pct_fdr"].sum()) != EXPECTED_CONSISTENT_LABELS:
        raise ValueError("Displayed consistent-target count changed")
    if len(group_summary_dedup) != EXPECTED_MEMBERSHIP_SETS:
        raise ValueError("Distinct OpenTargets membership-set count changed")
    if int(group_summary_dedup["consistent_at_5pct_fdr"].sum()) != EXPECTED_CONSISTENT_MEMBERSHIP_SETS:
        raise ValueError("Deduplicated consistent-set count changed")

    group_summary.to_csv(output_dir / GROUP_SUMMARY_TABLE, index=False, lineterminator="\n")
    continuous.to_csv(output_dir / CONTINUOUS_TABLE, index=False, lineterminator="\n")
    categorical.to_csv(output_dir / CATEGORICAL_TABLE, index=False, lineterminator="\n")
    membership.to_csv(output_dir / MEMBERSHIP_TABLE, index=False, lineterminator="\n")
    matched_compounds.to_csv(output_dir / MATCHED_COMPOUND_TABLE, index=False, lineterminator="\n")

    # Each renderer returns the caption it drew; the legend is those captions and nothing else.
    captions: dict[str, str] = {}
    captions.update(
        render_main_figure(
            continuous,
            categorical,
            group_summary,
            output_dir,
            PRIMARY_STEM,
            dpi,
            unit_label="target label",
            # The primary view is where the interpretation shared by all five hangs.
            extra_caption=(
                " Consistency and activity use the same activity-filtered compounds. Corrected p-values test group "
                "consistency; the slopes, quartile summaries, categorical comparisons, and Spearman correlations "
                "are descriptive."
            ),
        )
    )
    captions.update(
        render_main_figure(
            continuous_dedup,
            categorical_dedup,
            group_summary_dedup,
            output_dir,
            DEDUPLICATED_STEM,
            dpi,
            unit_label="membership set",
            title_suffix=" - distinct membership sets",
            extra_caption=(
                " Exact memberships are shared by AKT1/AKT2/AKT3, TUBA3C/TUBB8, and "
                "PIK3CG/PIK3R1/PIK3R2/PIK3R3/PIK3R5. Among distinct sets, "
                f"{overlap_summary['compounds_in_multiple_distinct_sets']}/{EXPECTED_UNIQUE_COMPOUNDS} compounds "
                f"occur in multiple distinct sets, and {overlap_summary['overlapping_distinct_set_pairs']}/"
                f"{overlap_summary['total_distinct_set_pairs']} set pairs share at least one compound."
            ),
        )
    )
    captions.update(render_activity_companion(group_summary, output_dir, dpi))
    captions.update(render_group_activity_figure(continuous, categorical, group_summary, output_dir, dpi))
    captions.update(
        render_matched_compound_activity_figure(
            compound_activity_bins, matched_compounds, activity_family, output_dir, dpi
        )
    )

    label_rho = float(
        spearmanr(group_summary["activity_nmap_mean"], group_summary["mean_normalized_average_precision"]).statistic
    )
    deduplicated_rho = float(
        spearmanr(
            group_summary_dedup["activity_nmap_mean"],
            group_summary_dedup["mean_normalized_average_precision"],
        ).statistic
    )
    if set(captions) != set(FIGURE_OUTPUT_STEMS):
        raise AssertionError(f"Every rendered figure needs exactly one caption: {sorted(set(captions))}")
    legend_path = output_dir / LEGEND_FILE
    write_legend(
        legend_path,
        figure_id=FIGURE_ID,
        title="OpenTargets target-group phenotypic consistency",
        sections=[(f"{stem}.png", SECTION_TITLES[stem], captions[stem]) for stem in FIGURE_OUTPUT_STEMS],
    )

    continuous_rho = {
        trait_id: float(
            spearmanr(
                continuous.loc[continuous["trait_id"].eq(trait_id), "property_mean"],
                continuous.loc[
                    continuous["trait_id"].eq(trait_id),
                    "mean_normalized_average_precision",
                ],
            ).statistic
        )
        for trait_id, _, _ in CONTINUOUS_TRAITS
    }
    continuous_rho_deduplicated = {
        trait_id: float(
            spearmanr(
                continuous_dedup.loc[continuous_dedup["trait_id"].eq(trait_id), "property_mean"],
                continuous_dedup.loc[
                    continuous_dedup["trait_id"].eq(trait_id),
                    "mean_normalized_average_precision",
                ],
            ).statistic
        )
        for trait_id, _, _ in CONTINUOUS_TRAITS
    }
    group_activity_continuous_rho = {
        trait_id: float(
            spearmanr(
                continuous.loc[continuous["trait_id"].eq(trait_id), "property_mean"],
                continuous.loc[continuous["trait_id"].eq(trait_id), "activity_nmap_mean"],
            ).statistic
        )
        for trait_id, _, _ in CONTINUOUS_TRAITS
    }
    compound_activity_continuous_rho = {
        trait_id: float(block["rho"].iloc[0])
        for trait_id, block in compound_activity_bins.groupby("trait_id", sort=True)
    }
    duplicate_sets = (
        group_summary.loc[group_summary["membership_set_target_count"].gt(1)]
        .drop_duplicates("membership_set_id")
        .sort_values("membership_set_targets", kind="stable")
    )
    numbers = {
        "figure_id": FIGURE_ID,
        "figure_exports": list(FIGURE_OUTPUT_STEMS),
        "outputs_by_role": {
            "primary": [PRIMARY_STEM],
            "sensitivity": [DEDUPLICATED_STEM],
            "companion": [ACTIVITY_COMPANION_STEM],
            "matched_activity": [GROUP_ACTIVITY_STEM, COMPOUND_ACTIVITY_STEM],
        },
        "query": {
            "dataset": DATASET,
            "columns": COLUMNS,
            "activity_preprocessing": ACTIVITY_PREPROCESSING,
            "activity_params": ACTIVITY_PARAMS,
            "preprocessing": PREPROCESSING,
            "filter": FILTER_NAME,
            "group_type": GROUP_TYPE,
            "distance": DISTANCE,
            "activity_threshold": ACTIVITY_THRESHOLD,
            "exclude_sweeps": True,
        },
        "display_contract": {
            "minimum_target_group_size": MIN_GROUP_SIZE,
            "consistent_target_definition": f"corrected_p_value < {CONSISTENCY_FDR}",
            "multiple_testing_family": f"all {tested_group_count} regular OpenTargets target groups",
            "continuous_property_aggregation": "arithmetic mean across exact contributing compounds",
            "continuous_consistent_percent_bins": "property-mean quartiles with ties retained",
            "categorical_point_size": "within-group fraction of exact contributing compounds in category",
            "primary_unit": "OpenTargets target label",
            "sensitivity_unit": "distinct compound-membership set",
            "sensitivity_quartiles": "recomputed after exact membership-set deduplication",
            "sensitivity_inference": "inherits target-label q values; no set-level test or FDR correction",
            "association_inference": "descriptive only; no association p-values",
            "matched_selection_order": [
                "compute activity q-values in the full canonical activity family",
                f"retain compounds with corrected_p_value < {ACTIVITY_THRESHOLD}",
                "form OpenTargets target-compound memberships",
                f"retain target labels with at least {MIN_GROUP_SIZE} distinct active compounds",
            ],
            "target_activity_unit": "OpenTargets target label; all 38 labels retained",
            "target_activity_outcome": "unweighted arithmetic mean of member compound activity nMAP",
            "target_activity_duplicate_policy": "preserve labels with exact duplicate membership sets",
            "target_activity_y_axis": "truncated at 0.80 with an explicit double-slash break",
            "compound_activity_unit": "unique JCP2022 compound",
            "compound_activity_duplicate_policy": "deduplicate memberships before every summary",
            "compound_activity_bins": "up to ten within-cohort quantile bins with ties retained",
            "compound_activity_y_axis": "truncated at 0.40 with an explicit double-slash break",
            "active_percent": "100% by construction in every nonempty matched bin and category",
            "group_activity_call": "none",
            "matched_activity_inference": "descriptive only; no transferred full-population q-values",
        },
        "activity_family_compounds": activity_family["n_compounds"],
        "activity_family_active_compounds": activity_family["n_active"],
        "tested_target_groups": tested_group_count,
        "target_labels": int(len(group_summary)),
        "distinct_membership_sets": int(group_summary["membership_set_id"].nunique()),
        "consistent_target_labels": int(group_summary["consistent_at_5pct_fdr"].sum()),
        "consistent_distinct_membership_sets": int(group_summary_dedup["consistent_at_5pct_fdr"].sum()),
        "target_compound_memberships": int(len(scores)),
        "unique_compounds": int(scores["jcp2022"].nunique()),
        "matched_compounds_active": int(matched_compounds["active_at_q_lt_0_10"].sum()),
        "matched_compound_active_rate": float(matched_compounds["active_at_q_lt_0_10"].mean()),
        "matched_compounds_in_multiple_target_labels": int(matched_compounds["n_displayed_target_labels"].gt(1).sum()),
        "displayed_target_group_size_min": int(group_summary["n_contributing_compounds"].min()),
        "displayed_target_group_size_max": int(group_summary["n_contributing_compounds"].max()),
        "target_mean_activity_nmap_min": float(group_summary["activity_nmap_mean"].min()),
        "target_mean_activity_nmap_max": float(group_summary["activity_nmap_mean"].max()),
        "matched_compound_activity_nmap_min": float(matched_compounds["activity_nmap"].min()),
        "matched_compound_activity_nmap_max": float(matched_compounds["activity_nmap"].max()),
        "activity_consistency_spearman_rho_target_labels": label_rho,
        "activity_consistency_spearman_rho_distinct_membership_sets": deduplicated_rho,
        "continuous_property_spearman_rho_target_labels": continuous_rho,
        "continuous_property_spearman_rho_distinct_membership_sets": continuous_rho_deduplicated,
        "target_mean_activity_spearman_rho": group_activity_continuous_rho,
        "matched_compound_activity_spearman_rho": compound_activity_continuous_rho,
        **overlap_summary,
        "duplicate_membership_sets": [
            {
                "membership_set_id": row.membership_set_id,
                "target_labels": row.membership_set_targets.split("|"),
                "n_compounds": int(row.n_contributing_compounds),
            }
            for row in duplicate_sets.itertuples(index=False)
        ],
        "input_sha256": {label: sha256_file(path) for label, path in paths.items()},
        "logical_sha256": {
            "group_summary": dataframe_sha256(group_summary, ["annotation_group"]),
            "continuous_group_properties": dataframe_sha256(
                continuous, ["trait_id", "property_mean", "annotation_group"]
            ),
            "categorical_group_composition": dataframe_sha256(
                categorical, ["family", "category_order", "annotation_group"]
            ),
            "memberships": dataframe_sha256(membership, ["annotation_group", "jcp2022"]),
            "matched_compounds": dataframe_sha256(matched_compounds, ["jcp2022"]),
            "matched_compound_activity_bins": dataframe_sha256(compound_activity_bins, ["trait_order", "decile"]),
            "deduplicated_group_summary": dataframe_sha256(group_summary_dedup, ["membership_set_id"]),
            "deduplicated_continuous_group_properties": dataframe_sha256(
                continuous_dedup, ["trait_id", "property_mean", "membership_set_id"]
            ),
            "deduplicated_categorical_group_composition": dataframe_sha256(
                categorical_dedup, ["family", "category_order", "membership_set_id"]
            ),
        },
    }
    files = [
        *[
            {"path": f"{stem}{suffix}", "kind": "figure", "caption": caption}
            for stem, caption in (
                (PRIMARY_STEM, "OpenTargets target-label group consistency"),
                (DEDUPLICATED_STEM, "OpenTargets distinct-membership-set sensitivity"),
                (ACTIVITY_COMPANION_STEM, "OpenTargets group activity and consistency comparison"),
                (GROUP_ACTIVITY_STEM, "Mean activity across exact OpenTargets target memberships"),
                (COMPOUND_ACTIVITY_STEM, "Compound-activity view for exact consistency-cohort compounds"),
            )
            for suffix in (".png", ".pdf")
        ],
        {"path": GROUP_SUMMARY_TABLE, "kind": "table", "caption": "OpenTargets target-group audit"},
        {
            "path": CONTINUOUS_TABLE,
            "kind": "table",
            "caption": "OpenTargets continuous group-property summaries",
        },
        {
            "path": CATEGORICAL_TABLE,
            "kind": "table",
            "caption": "OpenTargets categorical group-composition summaries",
        },
        {
            "path": MEMBERSHIP_TABLE,
            "kind": "table",
            "caption": "Displayed OpenTargets target-compound memberships with activity nMAP",
        },
        {
            "path": MATCHED_COMPOUND_TABLE,
            "kind": "table",
            "caption": "Unique consistency-cohort compounds with activity, targets, and raw properties",
        },
        {"path": LEGEND_FILE, "kind": "data", "caption": "Figure legends and statistical interpretation"},
    ]
    write_summary(
        output_dir / "summary.json",
        description=(
            "OpenTargets target-group phenotypic-consistency views, exact-membership-set sensitivity, direct "
            "activity-consistency comparison, target-aggregated and compound-level matched activity views, and "
            "auditable cohort tables."
        ),
        numbers=numbers,
        files=files,
    )
    typer.echo(
        f"Rendered {len(group_summary)} target labels, "
        f"{group_summary['membership_set_id'].nunique()} membership sets, "
        f"and {int(group_summary['consistent_at_5pct_fdr'].sum())} consistent targets to {output_dir}"
    )


if __name__ == "__main__":
    app()
