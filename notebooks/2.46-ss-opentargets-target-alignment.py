"""Relate phenotypic activity to asymmetric OpenTargets target alignment.

This analysis owns the estimand and the tables behind the target-alignment
panels; ``figure-compound-activity-associations.py`` renders them.
It keeps one row per compound-target membership.  The query cohort is
defined from all OpenTargets-annotated compounds with a canonical activity result;
activity significance selects reference anchors but never selects queries.  Target
alignment is computed directly with ``copairs.map.multilabel.average_precision``
using separate query and active-reference roles.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import anndata as ad
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import typer
from copairs.map.multilabel import average_precision as multilabel_average_precision
from loguru import logger
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster
from threadpoolctl import threadpool_limits

from jump_production.statistics import wilson_bounds
from jump_production.summaries import index_files, write_summary


def query_well_metadata(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read plate/control maps and exact well cell counts."""
    connection = duckdb.connect(str(path), read_only=True)
    try:
        plates = connection.execute(
            "SELECT Metadata_Plate AS Plate, Metadata_PlateType AS PlateType FROM plate ORDER BY Metadata_Plate"
        ).df()
        controls = connection.execute(
            """
            SELECT Metadata_JCP2022 AS JCP2022, Metadata_pert_type AS pert_type
            FROM perturbation_control ORDER BY Metadata_JCP2022
            """
        ).df()
        counts = connection.execute(
            """
            SELECT Metadata_Source AS Source, Metadata_Plate AS Plate,
                   Metadata_Well AS Well, Metadata_Count_Cells AS Count_Cells
            FROM cell_counts ORDER BY Metadata_Source, Metadata_Plate, Metadata_Well
            """
        ).df()
    finally:
        connection.close()
    return plates, controls, counts


ANALYSIS_ID = "opentargets_target_alignment"
DATASET = "compound_no_source7"
COLUMNS = "feat_all"
ACTIVITY_PREPROCESSING = "activity_no_target2"
FILTER_NAME = "all_sources"
ACTIVITY_PARAMS = "default"
ACTIVITY_Q_THRESHOLD = 0.10
ANNOTATION_TABLE = "motive_targets_opentargets"
ANNOTATION_COLUMN = "Metadata_motive_gene_opentargets"
MIN_ANALYZED_MEMBERS = 5
PRIMARY_MIN_ACTIVE_ANCHORS = 4
ANCHOR_THRESHOLDS = (3, 4, 5)
SOURCE_HOLDOUT = "source_2"
RANDOM_SEED = 20_260_805
DEFAULT_SHUFFLES = 100

DEFAULT_WELL_PROFILES = Path("data/interim/anndata/compound_no_source7.h5ad")
DEFAULT_METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")
DEFAULT_COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
DEFAULT_OUTPUT_DIR = Path(f"data/processed/opentargets-target-alignment/{DATASET}")

MEMBERSHIP_TABLE = "opentargets_target_alignment_memberships.csv"
TARGET_AUDIT_TABLE = "opentargets_target_reference_audit.csv"
MODEL_TABLE = "opentargets_target_alignment_models.csv"
SENSITIVITY_TABLE = "opentargets_target_alignment_sensitivities.csv"
SHUFFLE_TABLE = "opentargets_shuffled_label_calibration.csv"
MATCHED_TABLE = "opentargets_matched_nonmembers.csv"
QUADRANT_TABLE = "opentargets_activity_consistency_quadrants.csv"
TARGET_SUMMARY_TABLE = "opentargets_target_alignment_target_summary.csv"
CURVE_TABLE = "opentargets_target_alignment_fitted_curve.csv"
COMPOUND_SUMMARY_TABLE = "opentargets_target_alignment_compound_summary.csv"
PROPERTY_BIN_TABLE = "opentargets_target_alignment_property_bins.csv"
GRAIN_AUDIT_TABLE = "opentargets_property_grain_audit.csv"
GRAIN_AUDIT_PERMUTATIONS = 2_000
PROPERTY_BINS = 5

# The structure-derived properties the compound-level figure plots, named as in 2.45.
PROPERTY_COLUMNS = ("mw", "logp", "tpsa", "rotatable", "fraction_csp3", "qed")

# Display order and labels match the compound-activity-association panels.
PROPERTY_TRAITS = (
    ("logp", "LogP", "Calculated LogP"),
    ("mw", "Molecular weight", "Molecular weight (Da)"),
    ("rotatable", "Rotatable bonds", "Rotatable bonds"),
    ("qed", "QED", "QED"),
    ("tpsa", "TPSA", "TPSA (angstrom^2)"),
    ("fraction_csp3", "Fraction Csp3", "Fraction Csp3"),
)

app = typer.Typer(add_completion=False)
_THREADPOOL_LIMITER = threadpool_limits(limits=1)
logging.getLogger("copairs").setLevel(logging.WARNING)


def exact_set_id(values: Iterable[str]) -> str:
    """Return an order-invariant identifier for an exact nonempty string set."""
    members = tuple(sorted({str(value) for value in values if str(value)}))
    if not members:
        return "empty"
    digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
    return f"set_{len(members)}_{digest}"


def _split_labels(value: object) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    labels = tuple(sorted({item.strip() for item in str(value).split("|") if item.strip()}))
    return labels


def load_analysis_inputs(
    metadata_db: Path,
    copairs_db: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]], pd.DataFrame, dict[str, int]]:
    """Load the exact OpenTargets memberships and five-dimensional activity slice."""
    metadata = duckdb.connect(str(metadata_db), read_only=True)
    try:
        annotations = metadata.execute(
            f"""
            SELECT Metadata_JCP2022 AS jcp2022,
                   {ANNOTATION_COLUMN} AS target_labels,
                   Metadata_n_genes_opentargets AS declared_target_count
            FROM {ANNOTATION_TABLE}
            WHERE {ANNOTATION_COLUMN} IS NOT NULL
            ORDER BY Metadata_JCP2022
            """
        ).df()
        compound_metadata = metadata.execute(
            """
            SELECT Metadata_JCP2022 AS jcp2022,
                   Metadata_repurposing_name AS compound_name,
                   Metadata_MurckoScaffold AS murcko_scaffold,
                   Metadata_ValidMol AS valid_molecule,
                   Metadata_SMILES AS smiles,
                   Metadata_MW AS mw,
                   Metadata_LogP AS logp,
                   Metadata_TPSA AS tpsa,
                   Metadata_RotatableBonds AS rotatable,
                   Metadata_FractionCSP3 AS fraction_csp3,
                   Metadata_QED AS qed
            FROM compound_metadata
            WHERE Metadata_JCP2022 = ANY(?)
            ORDER BY Metadata_JCP2022
            """,
            [annotations["jcp2022"].astype(str).tolist()],
        ).df()
    finally:
        metadata.close()

    annotations["jcp2022"] = annotations["jcp2022"].astype(str)
    annotations["target_labels"] = annotations["target_labels"].map(_split_labels)
    observed_counts = annotations["target_labels"].map(len)
    if not np.array_equal(observed_counts.to_numpy(), annotations["declared_target_count"].to_numpy()):
        raise ValueError("OpenTargets declared target counts disagree with parsed labels")
    if annotations["jcp2022"].duplicated().any() or annotations["target_labels"].map(len).eq(0).any():
        raise ValueError("OpenTargets annotations must be one nonempty label set per compound")

    annotated_ids = annotations["jcp2022"].tolist()
    results = duckdb.connect(str(copairs_db), read_only=True)
    try:
        activity = results.execute(
            """
            SELECT Metadata_JCP2022 AS jcp2022,
                   mean_normalized_average_precision AS activity_nmap,
                   corrected_p_value AS activity_corrected_p_value,
                   below_corrected_p AS stored_active_flag
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
                annotated_ids,
            ],
        ).df()
        family = results.execute(
            """
            SELECT COUNT(*) AS analyzed_compounds,
                   COUNT_IF(corrected_p_value < ?) AS active_compounds
            FROM activity_results
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = ?
              AND _filter = ?
              AND _activity_params = ?
            """,
            [
                ACTIVITY_Q_THRESHOLD,
                DATASET,
                COLUMNS,
                ACTIVITY_PREPROCESSING,
                FILTER_NAME,
                ACTIVITY_PARAMS,
            ],
        ).fetchone()
        canonical_consistency = results.execute(
            """
            SELECT COUNT(DISTINCT group_value) AS tested_targets,
                   COUNT(DISTINCT group_value) FILTER (WHERE n_perturbations >= 5) AS targets_ge5
            FROM consistency_results
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = 'consistency_no_target2'
              AND _filter = ?
              AND _group_type = 'motive_opentargets'
              AND _distance = 'cosine'
              AND _activity_threshold = ?
              AND _preprocessing NOT LIKE '%_sweep'
            """,
            [DATASET, COLUMNS, FILTER_NAME, ACTIVITY_Q_THRESHOLD],
        ).fetchone()
        canonical_scores = results.execute(
            """
            SELECT COUNT(*) AS memberships,
                   COUNT(DISTINCT Metadata_JCP2022) AS compounds
            FROM consistency_scores
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = 'consistency_no_target2'
              AND _filter = ?
              AND _group_type = 'motive_opentargets'
              AND _distance = 'cosine'
              AND _activity_threshold = ?
              AND group_value IN (
                    SELECT group_value
                    FROM consistency_results
                    WHERE _dataset = ? AND _columns = ?
                      AND _preprocessing = 'consistency_no_target2'
                      AND _filter = ? AND _group_type = 'motive_opentargets'
                      AND _distance = 'cosine' AND _activity_threshold = ?
                      AND n_perturbations >= 5
                      AND _preprocessing NOT LIKE '%_sweep'
              )
            """,
            [
                DATASET,
                COLUMNS,
                FILTER_NAME,
                ACTIVITY_Q_THRESHOLD,
                DATASET,
                COLUMNS,
                FILTER_NAME,
                ACTIVITY_Q_THRESHOLD,
            ],
        ).fetchone()
        # Every tested group, not only the 38 with at least five active members: this
        # analysis keeps targets on its own eligibility rule and needs a flag for each.
        consistent_targets = results.execute(
            """
            SELECT group_value AS target_label,
                   below_corrected_p AS target_consistent
            FROM consistency_results
            WHERE _dataset = ?
              AND _columns = ?
              AND _preprocessing = 'consistency_no_target2'
              AND _filter = ?
              AND _group_type = 'motive_opentargets'
              AND _distance = 'cosine'
              AND _activity_threshold = ?
              AND _preprocessing NOT LIKE '%_sweep'
            """,
            [DATASET, COLUMNS, FILTER_NAME, ACTIVITY_Q_THRESHOLD],
        ).df()
    finally:
        results.close()

    consistent_targets["target_label"] = consistent_targets["target_label"].astype(str)
    consistent_targets["target_consistent"] = consistent_targets["target_consistent"].astype(bool)
    activity["jcp2022"] = activity["jcp2022"].astype(str)
    if activity["jcp2022"].duplicated().any():
        raise ValueError("Canonical activity slice is not unique by JCP2022")
    activity["is_active"] = activity["activity_corrected_p_value"].lt(ACTIVITY_Q_THRESHOLD)
    if not np.array_equal(activity["is_active"].to_numpy(), activity["stored_active_flag"].astype(bool).to_numpy()):
        raise ValueError("Stored activity flags disagree with strict corrected_p_value < 0.10")
    if not np.isfinite(activity[["activity_nmap", "activity_corrected_p_value"]].to_numpy()).all():
        raise ValueError("Canonical activity contains nonfinite values")

    analyzed_annotations = annotations.merge(
        activity.drop(columns="stored_active_flag"), on="jcp2022", how="inner", validate="one_to_one"
    )
    labels_by_id = dict(zip(analyzed_annotations["jcp2022"], analyzed_annotations["target_labels"], strict=True))
    membership_records = [
        {
            "jcp2022": record.jcp2022,
            "target_label": target,
            "activity_nmap": float(record.activity_nmap),
            "activity_corrected_p_value": float(record.activity_corrected_p_value),
            "is_active": bool(record.is_active),
            "full_annotation_degree": len(record.target_labels),
        }
        for record in analyzed_annotations.itertuples(index=False)
        for target in record.target_labels
    ]
    memberships = pd.DataFrame(membership_records).sort_values(["target_label", "jcp2022"]).reset_index(drop=True)
    if memberships.duplicated(["target_label", "jcp2022"]).any():
        raise ValueError("OpenTargets membership rows are not unique")
    compound_metadata["jcp2022"] = compound_metadata["jcp2022"].astype(str)
    compounds = analyzed_annotations.drop(columns=["declared_target_count"]).merge(
        compound_metadata, on="jcp2022", how="left", validate="one_to_one"
    )
    context = {
        "canonical_activity_family": int(family[0]),
        "canonical_active_family": int(family[1]),
        "canonical_consistency_tested_targets": int(canonical_consistency[0]),
        "canonical_consistency_targets_ge5": int(canonical_consistency[1]),
        "canonical_consistency_memberships_ge5": int(canonical_scores[0]),
        "canonical_consistency_compounds_ge5": int(canonical_scores[1]),
        "canonical_consistency_targets_below_corrected_p": int(consistent_targets["target_consistent"].sum()),
    }
    return memberships, compounds, labels_by_id, consistent_targets, context


def build_target_audit(memberships: pd.DataFrame, min_analyzed: int = MIN_ANALYZED_MEMBERS) -> pd.DataFrame:
    """Audit target eligibility before applying active-anchor thresholds."""
    required = {"jcp2022", "target_label", "is_active"}
    missing = required.difference(memberships.columns)
    if missing:
        raise ValueError(f"Membership table is missing {sorted(missing)}")
    if memberships.duplicated(["target_label", "jcp2022"]).any():
        raise ValueError("Memberships must be unique by target and compound")
    records = []
    for target, frame in memberships.groupby("target_label", sort=True):
        member_ids = tuple(sorted(frame["jcp2022"].astype(str)))
        anchor_ids = tuple(sorted(frame.loc[frame["is_active"].astype(bool), "jcp2022"].astype(str)))
        record: dict[str, object] = {
            "target_label": str(target),
            "analyzed_members": len(member_ids),
            "active_anchors": len(anchor_ids),
            "inactive_members": len(member_ids) - len(anchor_ids),
            "eligible_analyzed_ge5": len(member_ids) >= min_analyzed,
            "membership_set_id": exact_set_id(member_ids),
            "anchor_set_id": exact_set_id(anchor_ids),
            "member_ids": "|".join(member_ids),
            "anchor_ids": "|".join(anchor_ids),
        }
        for threshold in ANCHOR_THRESHOLDS:
            record[f"eligible_anchor_{threshold}"] = bool(
                len(member_ids) >= min_analyzed and len(anchor_ids) >= threshold
            )
        records.append(record)
    audit = pd.DataFrame(records)
    audit["joint_set_id"] = [
        exact_set_id((membership_id, anchor_id))
        for membership_id, anchor_id in zip(audit["membership_set_id"], audit["anchor_set_id"], strict=True)
    ]
    audit["targets_with_membership_set"] = audit.groupby("membership_set_id")["target_label"].transform("size")
    audit["targets_with_anchor_set"] = audit.groupby("anchor_set_id")["target_label"].transform("size")
    audit["targets_with_joint_set"] = audit.groupby("joint_set_id")["target_label"].transform("size")
    return audit.sort_values(["analyzed_members", "target_label"], ascending=[False, True]).reset_index(drop=True)


def validate_profile_contract(profiles: pd.DataFrame, expected_ids: Iterable[str]) -> None:
    """Reject incomplete, nonfinite, duplicate, or zero-norm compound profiles."""
    expected = {str(value) for value in expected_ids}
    observed = set(profiles.index.astype(str))
    if profiles.index.has_duplicates:
        raise ValueError("Profile index contains duplicate JCP2022 identities")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Profile coverage changed: missing={len(missing)}, extra={len(extra)}")
    values = profiles.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Profiles contain nonfinite values")
    zero_ids = profiles.index[np.linalg.norm(values, axis=1) <= 0].astype(str).tolist()
    if zero_ids:
        raise ValueError(f"Profiles contain {len(zero_ids)} zero-norm rows")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or np.any(norms <= 0):
        raise ValueError("Cannot normalize nonfinite or zero-norm profiles")
    return array / norms


def load_exact_well_profiles(
    well_path: Path,
    metadata_db: Path,
    selected_ids: set[str],
    effect_query_ids: set[str],
    source_crossfit_ids: set[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, int | float],
]:
    """Median-aggregate selected no-TARGET2/no-poscon wells exactly as consistency.

    The function deliberately loads features only for the selected OpenTargets
    compounds and the plate-matched negative controls needed by the separate
    phenotype-magnitude sensitivity.
    """
    plates, controls, counts = query_well_metadata(metadata_db)
    adata = ad.read_h5ad(well_path, backed="r")
    raw_wells = int(adata.n_obs)
    try:
        obs = adata.obs.reset_index(drop=True).copy()
        obs = obs.merge(plates, on="Plate", how="left", sort=False, validate="many_to_one")
        obs = obs.merge(controls, on="JCP2022", how="left", sort=False, validate="many_to_one")
        obs = obs.merge(
            counts,
            on=["Source", "Plate", "Well"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        obs["jcp2022"] = obs["JCP2022"].astype(str)
        non_target2 = obs["PlateType"].ne("TARGET2")
        eligible = non_target2 & obs["pert_type"].ne("poscon")
        selected_mask = eligible & obs["jcp2022"].isin(selected_ids)
        selected_indices = np.flatnonzero(selected_mask.to_numpy())
        selected_obs = obs.loc[selected_mask].reset_index(drop=True)
        if not len(selected_indices):
            raise ValueError("No selected wells survive the consistency exclusions")
        logger.info(
            "Loading {:,} OpenTargets wells x {:,} features",
            len(selected_indices),
            adata.n_vars,
        )
        selected_values = np.asarray(adata.X[selected_indices], dtype=np.float32)
        selected_features = pd.DataFrame(selected_values, columns=adata.var_names.astype(str))

        full_profiles = selected_features.groupby(selected_obs["jcp2022"].to_numpy(), sort=True).median()
        full_profiles.index.name = "jcp2022"
        source2_mask = selected_obs["Source"].astype(str).eq(SOURCE_HOLDOUT).to_numpy()
        other_mask = ~source2_mask
        source2_profiles = (
            selected_features.loc[source2_mask]
            .groupby(selected_obs.loc[source2_mask, "jcp2022"].to_numpy(), sort=True)
            .median()
        )
        other_profiles = (
            selected_features.loc[other_mask]
            .groupby(selected_obs.loc[other_mask, "jcp2022"].to_numpy(), sort=True)
            .median()
        )
        source2_profiles.index.name = other_profiles.index.name = "jcp2022"

        well_summary = (
            selected_obs.groupby("jcp2022", sort=True)
            .agg(
                eligible_wells=("Well", "size"),
                source_count=("Source", "nunique"),
                source_signature=("Source", lambda values: "|".join(sorted(set(map(str, values))))),
                median_cell_count=("Count_Cells", "median"),
            )
            .reset_index()
        )
        well_summary["cell_count_percentile"] = well_summary["median_cell_count"].rank(method="average", pct=True)

        effect_wells = selected_obs.loc[selected_obs["jcp2022"].isin(effect_query_ids)].copy()
        effect_strata = set(zip(effect_wells["Source"].astype(str), effect_wells["Plate"].astype(str), strict=True))
        stratum_keys = list(zip(obs["Source"].astype(str), obs["Plate"].astype(str), strict=True))
        control_mask = (
            non_target2
            & obs["pert_type"].eq("negcon")
            & pd.Series([key in effect_strata for key in stratum_keys], index=obs.index)
        )
        control_indices = np.flatnonzero(control_mask.to_numpy())
        control_obs = obs.loc[control_mask].reset_index(drop=True)
        logger.info(
            "Loading {:,} negative-control wells across {:,} source/plate strata",
            len(control_indices),
            len(effect_strata),
        )
        control_values = np.asarray(adata.X[control_indices], dtype=np.float32)
    finally:
        adata.file.close()

    validate_profile_contract(full_profiles, selected_ids)
    if not np.isfinite(selected_values).all() or not np.isfinite(control_values).all():
        raise ValueError("Well-level profile selection contains nonfinite values")
    missing_source2 = source_crossfit_ids - set(source2_profiles.index.astype(str))
    missing_other = source_crossfit_ids - set(other_profiles.index.astype(str))
    if missing_source2 or missing_other:
        raise ValueError(
            "Source cross-fit profile coverage changed: "
            f"missing source_2={len(missing_source2)}, missing other={len(missing_other)}"
        )
    source2_profiles = source2_profiles.reindex(sorted(source_crossfit_ids))
    other_profiles = other_profiles.reindex(sorted(source_crossfit_ids))
    validate_profile_contract(source2_profiles, source_crossfit_ids)
    validate_profile_contract(other_profiles, source_crossfit_ids)

    if not len(control_values):
        raise ValueError("No plate-matched negative controls were found")
    control_frame = pd.DataFrame(control_values, columns=full_profiles.columns)
    control_medians = control_frame.groupby(
        [control_obs["Source"].astype(str).to_numpy(), control_obs["Plate"].astype(str).to_numpy()],
        sort=True,
    ).median()
    control_medians.index.names = ["Source", "Plate"]
    control_norm = _normalize_rows(control_medians.to_numpy())
    control_lookup = {key: control_norm[index] for index, key in enumerate(control_medians.index)}
    selected_norm = _normalize_rows(selected_values)
    selected_positions = selected_obs.index[selected_obs["jcp2022"].isin(effect_query_ids)].to_numpy()
    effect_records = []
    for position in selected_positions:
        record = selected_obs.iloc[position]
        key = (str(record["Source"]), str(record["Plate"]))
        if key not in control_lookup:
            raise ValueError(f"Missing plate-matched negative controls for {key}")
        effect_records.append(
            {
                "jcp2022": str(record["jcp2022"]),
                "source": key[0],
                "plate": key[1],
                "well": str(record["Well"]),
                "plate_matched_negcon_cosine_distance": float(1.0 - selected_norm[position] @ control_lookup[key]),
            }
        )
    effect_well_table = pd.DataFrame(effect_records)
    plate_effect = (
        effect_well_table.groupby("jcp2022", sort=True)
        .agg(
            plate_matched_negcon_cosine_distance=("plate_matched_negcon_cosine_distance", "median"),
            effect_size_wells=("well", "size"),
            effect_size_strata=("plate", "nunique"),
        )
        .reset_index()
    )
    control_counts = control_obs.groupby(["Source", "Plate"], sort=True).size()
    audit: dict[str, int | float] = {
        "raw_wells": raw_wells,
        "selected_eligible_wells": int(len(selected_indices)),
        "selected_compound_profiles": int(len(full_profiles)),
        "features": int(full_profiles.shape[1]),
        "zero_norm_profiles": 0,
        "minimum_profile_norm": float(np.linalg.norm(full_profiles.to_numpy(), axis=1).min()),
        "maximum_profile_norm": float(np.linalg.norm(full_profiles.to_numpy(), axis=1).max()),
        "effect_query_wells": int(len(effect_well_table)),
        "plate_control_strata": int(len(control_counts)),
        "minimum_controls_per_stratum": int(control_counts.min()),
        "median_controls_per_stratum": float(control_counts.median()),
        "maximum_controls_per_stratum": int(control_counts.max()),
    }
    return full_profiles, source2_profiles, other_profiles, well_summary, plate_effect, audit


def _validate_alignment_denominators(
    scores: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    reference_ids: Sequence[str],
) -> pd.DataFrame:
    """Independently enumerate the positive and disjoint-negative reference sets."""
    reference_ids = tuple(map(str, reference_ids))
    records = []
    for row in scores.itertuples(index=False):
        query_id = str(row.jcp2022)
        query_labels = set(labels_by_id[query_id])
        target = str(row.target_label)
        positives = {
            reference_id
            for reference_id in reference_ids
            if reference_id != query_id and target in labels_by_id[reference_id]
        }
        negatives = {
            reference_id
            for reference_id in reference_ids
            if reference_id != query_id and query_labels.isdisjoint(labels_by_id[reference_id])
        }
        expected_total = len(positives) + len(negatives)
        if int(row.n_pos_pairs) != len(positives) or int(row.n_total_pairs) != expected_total:
            raise ValueError(
                f"Pair denominators disagree for {query_id}/{target}: "
                f"copairs={row.n_pos_pairs}/{row.n_total_pairs}, independent={len(positives)}/{expected_total}"
            )
        records.append(
            {
                "jcp2022": query_id,
                "target_label": target,
                "expected_positive_anchors": len(positives),
                "expected_disjoint_negatives": len(negatives),
            }
        )
    return scores.merge(pd.DataFrame(records), on=["jcp2022", "target_label"], validate="one_to_one")


def score_target_alignment(
    query_profiles: pd.DataFrame,
    reference_profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    query_ids: Sequence[str],
    reference_ids: Sequence[str],
) -> pd.DataFrame:
    """Compute asymmetric target-alignment AP/nAP with public multilabel copairs."""
    query_ids = tuple(sorted({str(value) for value in query_ids}))
    reference_ids = tuple(sorted({str(value) for value in reference_ids}))
    if not query_ids or not reference_ids:
        raise ValueError("Both query and reference roles must be nonempty")
    missing_labels = (set(query_ids) | set(reference_ids)) - set(labels_by_id)
    if missing_labels:
        raise ValueError(f"Missing label sets for {len(missing_labels)} scoring identities")
    missing_query = set(query_ids) - set(query_profiles.index.astype(str))
    missing_reference = set(reference_ids) - set(reference_profiles.index.astype(str))
    if missing_query or missing_reference:
        raise ValueError(f"Missing scoring profiles: query={len(missing_query)}, reference={len(missing_reference)}")
    query_values = query_profiles.reindex(query_ids).to_numpy(dtype=np.float32)
    reference_values = reference_profiles.reindex(reference_ids).to_numpy(dtype=np.float32)
    _normalize_rows(query_values)
    _normalize_rows(reference_values)
    meta = pd.DataFrame(
        {
            "jcp2022": [*query_ids, *reference_ids],
            "role": [*["query"] * len(query_ids), *["reference"] * len(reference_ids)],
            "target_labels": [
                *[list(labels_by_id[jcp2022]) for jcp2022 in query_ids],
                *[list(labels_by_id[jcp2022]) for jcp2022 in reference_ids],
            ],
        }
    )
    features = np.vstack([query_values, reference_values])
    result = multilabel_average_precision(
        meta,
        features,
        pos_sameby=["target_labels"],
        pos_diffby=["role", "jcp2022"],
        neg_sameby=[],
        neg_diffby=["target_labels", "role", "jcp2022"],
        multilabel_col="target_labels",
        batch_size=20_000,
        distance="cosine",
        progress_bar=False,
    )
    result = (
        result.loc[result["role"].eq("query")].rename(columns={"target_labels": "target_label"}).drop(columns="role")
    )
    result["jcp2022"] = result["jcp2022"].astype(str)
    result["target_label"] = result["target_label"].astype(str)
    if result.duplicated(["jcp2022", "target_label"]).any():
        raise ValueError("copairs returned duplicate query-target scores")
    metrics = ["average_precision", "normalized_average_precision", "n_pos_pairs", "n_total_pairs"]
    if not np.isfinite(result[metrics].to_numpy(dtype=float)).all():
        raise ValueError("copairs returned nonfinite target-alignment scores")
    return _validate_alignment_denominators(
        result.sort_values(["target_label", "jcp2022"]).reset_index(drop=True),
        labels_by_id,
        reference_ids,
    )


def scaffold_holdout_reference_ids(
    query_id: str,
    target: str,
    active_ids: Iterable[str],
    labels_by_id: dict[str, tuple[str, ...]],
    scaffolds: dict[str, str],
) -> tuple[str, ...]:
    """Return active target anchors after identity and exact-scaffold exclusion."""
    query_scaffold = str(scaffolds.get(str(query_id), "") or "").strip()
    return tuple(
        sorted(
            reference_id
            for reference_id in map(str, active_ids)
            if reference_id != str(query_id)
            and target in labels_by_id[reference_id]
            and (not query_scaffold or str(scaffolds.get(reference_id, "") or "").strip() != query_scaffold)
        )
    )


def add_centroid_calibration(
    memberships: pd.DataFrame,
    profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    active_ids: set[str],
    target_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Add leave-one-compound-out spherical centroids and wrong-target calibration."""
    normalized = pd.DataFrame(
        _normalize_rows(profiles.to_numpy()), index=profiles.index.astype(str), columns=profiles.columns
    )
    primary_targets = target_audit.loc[target_audit["eligible_anchor_4"], "target_label"].astype(str).tolist()
    anchor_sets = target_audit.set_index("target_label")["anchor_set_id"].to_dict()
    target_anchors = {
        target: tuple(sorted(jcp for jcp in active_ids if target in labels_by_id[jcp])) for target in primary_targets
    }

    def centroid(target: str, excluded_id: str) -> np.ndarray:
        ids = [jcp for jcp in target_anchors[target] if jcp != excluded_id]
        if not ids:
            raise ValueError(f"Target {target} has no anchors after excluding {excluded_id}")
        mean = normalized.reindex(ids).to_numpy().mean(axis=0, keepdims=True)
        return _normalize_rows(mean)[0]

    records = []
    for row in memberships.itertuples(index=False):
        query_id = str(row.jcp2022)
        target = str(row.target_label)
        query_profile = normalized.loc[query_id].to_numpy()
        correct = float(query_profile @ centroid(target, query_id))
        query_labels = set(labels_by_id[query_id])
        wrong_by_set: dict[str, float] = {}
        for wrong_target in primary_targets:
            if wrong_target in query_labels or anchor_sets[wrong_target] == anchor_sets[target]:
                continue
            wrong_by_set.setdefault(
                anchor_sets[wrong_target],
                float(query_profile @ centroid(wrong_target, query_id)),
            )
        wrong = np.asarray(list(wrong_by_set.values()), dtype=float)
        if not len(wrong):
            raise ValueError(f"No wrong-target prototype is available for {query_id}/{target}")
        percentile = 100.0 * float(np.mean(wrong < correct) + 0.5 * np.mean(wrong == correct))
        records.append(
            {
                "jcp2022": query_id,
                "target_label": target,
                "centroid_cosine": correct,
                "centroid_margin_vs_best_wrong": correct - float(np.max(wrong)),
                "centroid_percentile_vs_wrong": percentile,
                "wrong_target_prototypes": int(len(wrong)),
            }
        )
    return memberships.merge(pd.DataFrame(records), on=["jcp2022", "target_label"], validate="one_to_one")


def summarize_targets(
    memberships: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """Collapse membership rows to one equally weighted row per target label."""
    required = {
        "jcp2022",
        "target_label",
        "activity_nmap",
        "full_annotation_degree",
        "active_anchors",
        "joint_set_id",
        outcome,
    }
    missing = required.difference(memberships.columns)
    if missing:
        raise ValueError(f"Across-target summary is missing {sorted(missing)}")
    data = memberships.dropna(subset=[outcome]).copy()
    if data.duplicated(["jcp2022", "target_label"]).any():
        raise ValueError("Across-target summary has duplicate memberships")
    invariants = data.groupby("target_label").agg(
        joint_sets=("joint_set_id", "nunique"),
        active_anchor_counts=("active_anchors", "nunique"),
    )
    if invariants["joint_sets"].ne(1).any() or invariants["active_anchor_counts"].ne(1).any():
        raise ValueError("Target rows disagree on their joint set or active-anchor count")
    data["log_annotation_degree"] = np.log1p(data["full_annotation_degree"].astype(float))
    summary = (
        data.groupby("target_label", sort=True)
        .agg(
            mean_activity_nmap=("activity_nmap", "mean"),
            mean_consistency_nap=(outcome, "mean"),
            mean_log_annotation_degree=("log_annotation_degree", "mean"),
            active_anchors=("active_anchors", "first"),
            query_count=("jcp2022", "nunique"),
            joint_set_id=("joint_set_id", "first"),
        )
        .reset_index()
    )
    summary["log_active_anchors"] = np.log2(summary["active_anchors"].astype(float))
    return summary


def _prepare_across_target_design(
    memberships: pd.DataFrame,
    outcome: str,
) -> tuple[pd.DataFrame, list[str], dict[str, float], dict[str, float], np.ndarray]:
    """Build a nonlinear design with one equally weighted observation per target."""
    data = summarize_targets(memberships, outcome)
    activity_center = float(data["mean_activity_nmap"].mean())
    data["mean_activity_linear"] = data["mean_activity_nmap"] - activity_center
    data["mean_activity_quadratic"] = data["mean_activity_linear"].pow(2)
    components = [
        "mean_activity_linear",
        "mean_activity_quadratic",
        "mean_log_annotation_degree",
        "log_active_anchors",
    ]
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    standardized = []
    for component in components:
        center = float(data[component].mean())
        scale = float(data[component].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Across-target design has invalid scale for {component}")
        centers[component] = center
        scales[component] = scale
        standardized.append((data[component].to_numpy(dtype=float) - center) / scale)
    design = np.column_stack([np.ones(len(data)), *standardized])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Across-target design is rank deficient")
    centers["mean_activity_nmap"] = activity_center
    return data, components, centers, scales, design


def fit_across_target_relationship(
    memberships: pd.DataFrame,
    *,
    outcome: str,
    model_id: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit an equal-target nonlinear model clustered by exact target set."""
    data, components, centers, scales, design = _prepare_across_target_design(memberships, outcome)
    fitted = sm.OLS(data["mean_consistency_nap"].to_numpy(dtype=float), design).fit()
    distinct_set_clusters = pd.factorize(data["joint_set_id"], sort=True)[0]
    covariance = cov_cluster(
        fitted,
        distinct_set_clusters,
        use_correction=True,
    )
    if not np.isfinite(covariance).all():
        raise ValueError(f"Across-target model {model_id} has nonfinite clustered covariance")
    n_set_clusters = int(data["joint_set_id"].nunique())
    inference_df = n_set_clusters - 1
    critical = float(student_t.ppf(0.975, inference_df))
    rows = []
    for index, component in enumerate(components, start=1):
        estimate = float(fitted.params[index])
        variance = float(covariance[index, index])
        standard_error = float(np.sqrt(variance)) if variance > 0 else float("nan")
        if not np.isfinite(standard_error):
            raise ValueError(f"Relationship model {model_id} has invalid variance for {component}")
        p_value = float(2 * student_t.sf(abs(estimate / standard_error), inference_df))
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "component": component,
                "estimate_per_component_sd": estimate,
                "standard_error": standard_error,
                "ci_lower": estimate - critical * standard_error,
                "ci_upper": estimate + critical * standard_error,
                "p_value": p_value,
                "component_center": centers[component],
                "component_sd": scales[component],
                "estimate_per_natural_unit": estimate / scales[component],
                "n_memberships": int(len(memberships)),
                "n_compounds": int(memberships["jcp2022"].nunique()),
                "n_targets": int(data["target_label"].nunique()),
                "n_distinct_sets": n_set_clusters,
                "inference_df": inference_df,
                "weighting": "one equally weighted row per target label",
                "covariance": "cluster robust by exact membership/anchor set",
            }
        )
    details: dict[str, object] = {
        "data": data,
        "components": components,
        "centers": centers,
        "scales": scales,
        "fitted": fitted,
        "covariance": covariance,
        "critical": critical,
    }
    return pd.DataFrame(rows), details


def across_target_spearman(
    memberships: pd.DataFrame,
    outcome: str,
    *,
    deduplicate_exact_sets: bool = False,
) -> float:
    """Return the descriptive association between target-level mean activity and consistency."""
    if "joint_set_id" in memberships:
        targets = summarize_targets(memberships, outcome)
    else:
        if deduplicate_exact_sets:
            raise ValueError("Exact-set deduplication requires joint_set_id")
        targets = (
            memberships.groupby("target_label", sort=True)
            .agg(
                mean_activity_nmap=("activity_nmap", "mean"),
                mean_consistency_nap=(outcome, "mean"),
            )
            .reset_index()
        )
    if deduplicate_exact_sets:
        targets = targets.sort_values(["joint_set_id", "target_label"], kind="mergesort").drop_duplicates(
            "joint_set_id", keep="first"
        )
    return float(spearmanr(targets["mean_activity_nmap"], targets["mean_consistency_nap"]).statistic)


def across_target_aggregation_null(
    memberships: pd.DataFrame,
    observed: float,
    iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Calibrate the across-target correlation against aggregation alone.

    Averaging four to seventeen members into one target point removes within-target
    scatter from both axes, which raises a correlation whether or not anything is
    there.  Permuting members across targets keeps the group sizes and destroys the
    real membership, so the resulting distribution is what aggregation buys on its
    own.  Without it the across-target value invites comparison with the
    per-compound value, which is not a comparison the two statistics support.
    """
    columns = ["activity_nmap", "target_alignment_nap"]
    labels = memberships["target_label"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        shuffled = memberships.assign(target_label=rng.permutation(labels))
        grouped = shuffled.groupby("target_label", sort=True)[columns].mean()
        draws.append(spearmanr(grouped["activity_nmap"], grouped["target_alignment_nap"]).statistic)
    null = np.abs(np.asarray(draws))
    return {
        "statistic": "Spearman rho between target mean activity nMAP and target mean consistency nAP",
        "null": "members permuted across targets, group sizes held fixed",
        "iterations": int(iterations),
        "observed_rho": float(observed),
        "null_mean_abs_rho": float(null.mean()),
        "null_p95_abs_rho": float(np.percentile(null, 95)),
        "empirical_p_value": float((null >= abs(observed)).mean()),
    }


def deduplicate_exact_target_sets(memberships: pd.DataFrame) -> pd.DataFrame:
    """Keep one lexicographic target representative per exact member/anchor set."""
    representatives = (
        memberships[["target_label", "joint_set_id"]]
        .drop_duplicates()
        .sort_values(["joint_set_id", "target_label"])
        .drop_duplicates("joint_set_id", keep="first")
    )
    return memberships.loc[memberships["target_label"].isin(representatives["target_label"])].copy()


def add_source_crossfit(
    primary: pd.DataFrame,
    source2_profiles: pd.DataFrame,
    other_profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    active_ids: set[str],
) -> pd.DataFrame:
    """Score source_2 queries against other-source anchors and the reverse."""
    query_ids = sorted(primary["jcp2022"].unique())
    source2_query = score_target_alignment(
        source2_profiles,
        other_profiles,
        labels_by_id,
        query_ids,
        sorted(active_ids),
    )[["jcp2022", "target_label", "normalized_average_precision"]].rename(
        columns={"normalized_average_precision": "source2_query_target_alignment_nap"}
    )
    other_query = score_target_alignment(
        other_profiles,
        source2_profiles,
        labels_by_id,
        query_ids,
        sorted(active_ids),
    )[["jcp2022", "target_label", "normalized_average_precision"]].rename(
        columns={"normalized_average_precision": "other_source_query_target_alignment_nap"}
    )
    result = primary.merge(source2_query, on=["jcp2022", "target_label"], how="left", validate="one_to_one")
    result = result.merge(other_query, on=["jcp2022", "target_label"], how="left", validate="one_to_one")
    crossfit_columns = ["source2_query_target_alignment_nap", "other_source_query_target_alignment_nap"]
    if result[crossfit_columns].isna().any().any():
        raise ValueError("Source cross-fit did not cover every primary membership")
    result["source_crossfit_mean_target_alignment_nap"] = result[crossfit_columns].mean(axis=1)
    return result


def add_leave_scaffold_out(
    primary: pd.DataFrame,
    profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    active_ids: set[str],
    scaffolds: dict[str, str],
) -> pd.DataFrame:
    """Recompute target-alignment AP after excluding the query's entire scaffold."""
    query_scaffolds = {
        query_id: str(scaffolds.get(query_id, "") or "").strip() for query_id in primary["jcp2022"].astype(str).unique()
    }
    records = []
    for scaffold in sorted({value for value in query_scaffolds.values() if value}):
        query_ids = sorted(query_id for query_id, value in query_scaffolds.items() if value == scaffold)
        reference_ids = sorted(
            reference_id
            for reference_id in active_ids
            if str(scaffolds.get(reference_id, "") or "").strip() != scaffold
        )
        scores = score_target_alignment(profiles, profiles, labels_by_id, query_ids, reference_ids)
        records.append(scores)
    if not records:
        raise ValueError("No nonempty query scaffolds are available for leave-scaffold-out scoring")
    scores = pd.concat(records, ignore_index=True)[
        ["jcp2022", "target_label", "normalized_average_precision", "n_pos_pairs", "n_total_pairs"]
    ].rename(
        columns={
            "normalized_average_precision": "leave_scaffold_out_target_alignment_nap",
            "n_pos_pairs": "leave_scaffold_out_n_pos_pairs",
            "n_total_pairs": "leave_scaffold_out_n_total_pairs",
        }
    )
    result = primary.merge(scores, on=["jcp2022", "target_label"], how="left", validate="one_to_one")
    result["leave_scaffold_out_status"] = np.where(
        result["murcko_scaffold"].fillna("").astype(str).str.strip().eq(""),
        "blocked_missing_scaffold",
        np.where(
            result["leave_scaffold_out_target_alignment_nap"].isna(),
            "blocked_no_score",
            np.where(
                result["leave_scaffold_out_n_pos_pairs"].ge(3),
                "scored_at_least_3_positive_anchors",
                "scored_below_3_positive_anchors",
            ),
        ),
    )
    return result


def build_matched_nonmembers(
    primary: pd.DataFrame,
    compounds: pd.DataFrame,
    profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    active_ids: set[str],
) -> pd.DataFrame:
    """Match each target member to a source/cell-count/activity-near nonmember."""
    pool = compounds.set_index("jcp2022").copy()
    normalized = pd.DataFrame(
        _normalize_rows(profiles.to_numpy()), index=profiles.index.astype(str), columns=profiles.columns
    )
    activity_scale = float(pool["activity_nmap"].std(ddof=0))
    log_counts = np.log1p(pool["median_cell_count"].astype(float))
    count_scale = float(log_counts.std(ddof=0))
    used_by_target: dict[str, set[str]] = {}
    records = []
    for row in primary.sort_values(["target_label", "jcp2022"]).itertuples(index=False):
        query_id = str(row.jcp2022)
        target = str(row.target_label)
        query = pool.loc[query_id]
        query_labels = set(labels_by_id[query_id])
        candidate_ids = [
            candidate
            for candidate in pool.index.astype(str)
            if candidate != query_id and query_labels.isdisjoint(labels_by_id[candidate])
        ]
        if not candidate_ids:
            raise ValueError(f"No multilabel-disjoint nonmember is available for {query_id}/{target}")
        candidates = pool.reindex(candidate_ids).copy()
        candidates = candidates.loc[candidates["is_active"].astype(bool).eq(bool(query["is_active"]))]
        if candidates.empty:
            candidates = pool.reindex(candidate_ids).copy()
            match_tier = "activity_status_relaxed"
        else:
            exact_source = candidates["source_signature"].astype(str).eq(str(query["source_signature"]))
            if exact_source.any():
                candidates = candidates.loc[exact_source]
                match_tier = "exact_activity_status_and_source_signature"
            else:
                exact_count = candidates["source_count"].eq(query["source_count"])
                if exact_count.any():
                    candidates = candidates.loc[exact_count]
                    match_tier = "exact_activity_status_and_source_count"
                else:
                    match_tier = "exact_activity_status_only"
        available = candidates.loc[~candidates.index.astype(str).isin(used_by_target.setdefault(target, set()))]
        reused = available.empty
        if not available.empty:
            candidates = available
        candidates["activity_distance_sd"] = (
            candidates["activity_nmap"].astype(float) - float(query["activity_nmap"])
        ).abs() / activity_scale
        candidates["cell_count_distance_sd"] = (
            np.log1p(candidates["median_cell_count"].astype(float)) - np.log1p(float(query["median_cell_count"]))
        ).abs() / count_scale
        candidates["match_distance"] = candidates["activity_distance_sd"] + candidates["cell_count_distance_sd"]
        candidates = candidates.sort_values(["match_distance", "activity_nmap"], kind="mergesort")
        control_id = str(candidates.index[0])
        used_by_target[target].add(control_id)
        anchors = sorted(
            reference_id
            for reference_id in active_ids
            if reference_id != query_id and target in labels_by_id[reference_id]
        )
        centroid = _normalize_rows(normalized.reindex(anchors).to_numpy().mean(axis=0, keepdims=True))[0]
        member_cosine = float(normalized.loc[query_id].to_numpy() @ centroid)
        control_cosine = float(normalized.loc[control_id].to_numpy() @ centroid)
        selected = candidates.loc[control_id]
        control_labels = set(labels_by_id[control_id])
        shared_labels = sorted(query_labels & control_labels)
        if shared_labels:
            raise ValueError(f"Matched control unexpectedly shares labels for {query_id}/{target}")
        records.append(
            {
                "jcp2022": query_id,
                "target_label": target,
                "matched_nonmember_jcp2022": control_id,
                "match_tier": match_tier,
                "control_reused_within_target": reused,
                "activity_nmap_member": float(query["activity_nmap"]),
                "activity_nmap_nonmember": float(selected["activity_nmap"]),
                "activity_nmap_absolute_difference": abs(
                    float(query["activity_nmap"]) - float(selected["activity_nmap"])
                ),
                "cell_count_percentile_member": float(query["cell_count_percentile"]),
                "cell_count_percentile_nonmember": float(selected["cell_count_percentile"]),
                "source_signature_member": str(query["source_signature"]),
                "source_signature_nonmember": str(selected["source_signature"]),
                "member_labels": "|".join(sorted(query_labels)),
                "matched_nonmember_labels": "|".join(sorted(control_labels)),
                "shared_label_count": len(shared_labels),
                "label_sets_disjoint": not shared_labels,
                "member_centroid_cosine": member_cosine,
                "matched_nonmember_centroid_cosine": control_cosine,
                "member_minus_nonmember_centroid_cosine": member_cosine - control_cosine,
            }
        )
    return pd.DataFrame(records)


def _target_balanced_mean(frame: pd.DataFrame, value: str) -> float:
    return float(frame.groupby("target_label", sort=True)[value].mean().mean())


def shuffled_label_calibration(
    profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    compounds: pd.DataFrame,
    active_ids: set[str],
    primary_targets: set[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Permute complete label vectors within activity strata and rescore with copairs."""
    rng = np.random.default_rng(seed)
    all_ids = sorted(labels_by_id)
    activity_lookup = compounds.set_index("jcp2022")["is_active"].astype(bool).to_dict()
    activity_nmap = compounds.set_index("jcp2022")["activity_nmap"].to_dict()
    records = []
    for iteration in range(iterations):
        shuffled: dict[str, tuple[str, ...]] = {}
        for activity_state in (False, True):
            ids = [jcp2022 for jcp2022 in all_ids if activity_lookup[jcp2022] == activity_state]
            label_vectors = [labels_by_id[jcp2022] for jcp2022 in ids]
            permutation = rng.permutation(len(ids))
            shuffled.update(
                {jcp2022: label_vectors[int(position)] for jcp2022, position in zip(ids, permutation, strict=True)}
            )
        scores = score_target_alignment(profiles, profiles, shuffled, all_ids, sorted(active_ids))
        scores = scores.loc[scores["target_label"].isin(primary_targets)].copy()
        expected_rows = sum(1 for jcp2022 in all_ids for target in shuffled[jcp2022] if target in primary_targets)
        if len(scores) != expected_rows:
            raise ValueError(
                f"Shuffled iteration {iteration} lost eligible memberships: {len(scores)} != {expected_rows}"
            )
        scores["activity_nmap"] = scores["jcp2022"].map(activity_nmap)
        scores["full_annotation_degree"] = scores["jcp2022"].map(lambda value: len(shuffled[value]))
        active_anchor_counts: dict[str, int] = {}
        for reference_id in active_ids:
            for target in shuffled[reference_id]:
                active_anchor_counts[target] = active_anchor_counts.get(target, 0) + 1
        scores["active_anchors"] = scores["target_label"].map(active_anchor_counts)
        if scores["active_anchors"].isna().any():
            raise ValueError(f"Shuffled iteration {iteration} lost an active-anchor count")
        records.append(
            {
                "iteration": iteration,
                "target_balanced_mean_nap": _target_balanced_mean(scores, "normalized_average_precision"),
                "memberships": int(len(scores)),
                "queries": int(scores["jcp2022"].nunique()),
                "targets": int(scores["target_label"].nunique()),
            }
        )
    return pd.DataFrame(records)


def add_pair_tie_audit(
    memberships: pd.DataFrame,
    profiles: pd.DataFrame,
    labels_by_id: dict[str, tuple[str, ...]],
    active_ids: set[str],
) -> pd.DataFrame:
    """Count exact positive-negative cosine ties for every retained rank list."""
    normalized = pd.DataFrame(
        _normalize_rows(profiles.to_numpy()), index=profiles.index.astype(str), columns=profiles.columns
    )
    records = []
    for row in memberships.itertuples(index=False):
        query_id = str(row.jcp2022)
        target = str(row.target_label)
        query_labels = set(labels_by_id[query_id])
        positives = sorted(
            reference_id
            for reference_id in active_ids
            if reference_id != query_id and target in labels_by_id[reference_id]
        )
        negatives = sorted(
            reference_id
            for reference_id in active_ids
            if reference_id != query_id and query_labels.isdisjoint(labels_by_id[reference_id])
        )
        query = normalized.loc[query_id].to_numpy()
        positive_scores = normalized.reindex(positives).to_numpy() @ query
        negative_scores = normalized.reindex(negatives).to_numpy() @ query
        ties = int(np.isclose(positive_scores[:, None], negative_scores[None, :], rtol=0, atol=1e-12).sum())
        records.append(
            {
                "jcp2022": query_id,
                "target_label": target,
                "positive_negative_similarity_ties": ties,
            }
        )
    return memberships.merge(pd.DataFrame(records), on=["jcp2022", "target_label"], validate="one_to_one")


def assign_scientific_quadrants(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Classify quartile groups and rank only interpretable individual extremes."""
    unique_activity = primary[["jcp2022", "activity_nmap"]].drop_duplicates("jcp2022")["activity_nmap"]
    activity_low, activity_high = unique_activity.quantile([0.25, 0.75]).to_numpy(dtype=float)
    alignment_low, alignment_high = primary["target_alignment_nap"].quantile([0.25, 0.75]).to_numpy(dtype=float)
    result = primary.copy()
    result["scientific_quadrant"] = "other"
    result.loc[
        result["activity_nmap"].ge(activity_high) & result["target_alignment_nap"].ge(alignment_high),
        "scientific_quadrant",
    ] = "high activity / high alignment"
    result.loc[
        result["activity_nmap"].ge(activity_high) & result["target_alignment_nap"].le(alignment_low),
        "scientific_quadrant",
    ] = "high activity / low alignment"
    result.loc[
        result["activity_nmap"].le(activity_low) & result["target_alignment_nap"].ge(alignment_high),
        "scientific_quadrant",
    ] = "low activity / high alignment"
    selected = []
    used_compounds: set[str] = set()
    used_joint_sets: set[str] = set()
    rank_orders = {
        "high activity / high alignment": (
            ["target_alignment_nap", "activity_nmap", "jcp2022", "target_label"],
            [False, False, True, True],
        ),
        "low activity / high alignment": (
            ["target_alignment_nap", "activity_nmap", "jcp2022", "target_label"],
            [False, True, True, True],
        ),
    }
    for quadrant, (columns, ascending) in rank_orders.items():
        candidates = result.loc[result["scientific_quadrant"].eq(quadrant)].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            columns,
            ascending=ascending,
            kind="mergesort",
        )
        selected_indices = []
        for index, row in candidates.iterrows():
            query_id = str(row["jcp2022"])
            joint_set_id = str(row["joint_set_id"])
            if query_id in used_compounds or joint_set_id in used_joint_sets:
                continue
            selected_indices.append(index)
            used_compounds.add(query_id)
            used_joint_sets.add(joint_set_id)
            if len(selected_indices) == 3:
                break
        if len(selected_indices) != 3:
            raise ValueError(f"Quadrant {quadrant} lacks three distinct compound/set exceptions")
        selected.append(candidates.loc[selected_indices])
    exceptions = pd.concat(selected, ignore_index=True) if selected else result.iloc[0:0].copy()
    if exceptions["jcp2022"].duplicated().any() or exceptions["joint_set_id"].duplicated().any():
        raise ValueError("Exception shortlist contains duplicate compounds or exact target sets")
    exceptions["exception_number"] = np.arange(1, len(exceptions) + 1)
    result = result.merge(
        exceptions[["jcp2022", "target_label", "exception_number"]],
        on=["jcp2022", "target_label"],
        how="left",
        validate="one_to_one",
    )
    thresholds = {
        "activity_low_quartile": float(activity_low),
        "activity_high_quartile": float(activity_high),
        "alignment_low_quartile": float(alignment_low),
        "alignment_high_quartile": float(alignment_high),
    }
    return result, exceptions, thresholds


def build_quadrant_export(primary: pd.DataFrame) -> pd.DataFrame:
    """Export every scientifically interesting category row without plot selection."""
    exported = primary.loc[~primary["scientific_quadrant"].eq("other")].copy()
    exported["scientific_category"] = exported["scientific_quadrant"].str.replace(
        "alignment",
        "consistency",
        regex=False,
    )
    exported["consistency_rank_within_category"] = (
        exported.groupby("scientific_category")["target_alignment_nap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    exported["activity_rank_within_category"] = (
        exported.groupby("scientific_category")["activity_nmap"].rank(method="first", ascending=False).astype(int)
    )
    exported["activity_at_ceiling"] = np.isclose(
        exported["activity_nmap"].to_numpy(dtype=float),
        1.0,
        rtol=0,
        atol=1e-12,
    )
    exported = exported.rename(
        columns={
            "active_anchors": "active_reference_compounds",
            "anchor_set_id": "reference_set_id",
        }
    )
    columns = [
        "scientific_category",
        "consistency_rank_within_category",
        "activity_rank_within_category",
        "jcp2022",
        "compound_name",
        "target_label",
        "activity_nmap",
        "target_alignment_nap",
        "activity_at_ceiling",
        "is_active",
        "active_reference_compounds",
        "analyzed_members",
        "membership_set_id",
        "reference_set_id",
        "joint_set_id",
    ]
    return exported[columns].sort_values(
        ["scientific_category", "consistency_rank_within_category", "jcp2022", "target_label"],
        kind="mergesort",
    )


def _across_target_curve(details: dict[str, object]) -> pd.DataFrame:
    data = details["data"]
    assert isinstance(data, pd.DataFrame)
    components = details["components"]
    centers = details["centers"]
    scales = details["scales"]
    fitted = details["fitted"]
    covariance = np.asarray(details["covariance"])
    critical = float(details["critical"])
    assert isinstance(components, list) and isinstance(centers, dict) and isinstance(scales, dict)
    lower, upper = data["mean_activity_nmap"].quantile([0.01, 0.99]).to_numpy(dtype=float)
    grid = np.linspace(lower, upper, 160)
    linear_index = 1 + components.index("mean_activity_linear")
    quadratic_index = 1 + components.index("mean_activity_quadratic")
    activity_center = float(centers["mean_activity_nmap"])
    rows = []
    for value in grid:
        contrast = np.zeros(len(fitted.params), dtype=float)
        contrast[0] = 1.0
        linear = value - activity_center
        quadratic = linear**2
        contrast[linear_index] = (linear - float(centers["mean_activity_linear"])) / float(
            scales["mean_activity_linear"]
        )
        contrast[quadratic_index] = (quadratic - float(centers["mean_activity_quadratic"])) / float(
            scales["mean_activity_quadratic"]
        )
        estimate = float(contrast @ fitted.params)
        variance = float(contrast @ covariance @ contrast)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        rows.append(
            {
                "mean_activity_nmap": value,
                "mean_consistency_fit": estimate,
                "ci_lower": estimate - critical * standard_error,
                "ci_upper": estimate + critical * standard_error,
            }
        )
    return pd.DataFrame(rows)


def summarize_compounds(primary: pd.DataFrame, consistent_targets: pd.DataFrame) -> pd.DataFrame:
    """Reduce memberships to one row per compound at its best-matching target.

    A compound annotated to several targets need not act through all of them, so the
    maximum over its targets asks whether the compound is consistent with any annotated
    target rather than with all of them.  The maximum is upward biased in the number of
    retained targets, so ``n_retained_targets`` is exported alongside it; the bias does
    not track any exported property, but it does make the level incomparable to a
    group-level mAP.
    """
    required = {"jcp2022", "target_label", "target_alignment_nap", "activity_nmap", *PROPERTY_COLUMNS}
    missing = required.difference(primary.columns)
    if missing:
        raise ValueError(f"Compound summary is missing {sorted(missing)}")
    best = primary.loc[primary.groupby("jcp2022")["target_alignment_nap"].idxmax()].copy()
    best = best.rename(
        columns={
            "target_label": "best_target_label",
            "target_alignment_nap": "best_target_alignment_nap",
            "target_alignment_ap": "best_target_alignment_ap",
        }
    )
    counts = primary.groupby("jcp2022")["target_label"].nunique().rename("n_retained_targets")
    best = best.merge(counts, on="jcp2022", how="left", validate="one_to_one")
    best = best.merge(
        consistent_targets.rename(
            columns={"target_label": "best_target_label", "target_consistent": "best_target_consistent"}
        ),
        on="best_target_label",
        how="left",
        validate="many_to_one",
    )
    if best["best_target_consistent"].isna().any():
        raise ValueError("A best-matching target has no canonical consistency result")
    if len(best) != primary["jcp2022"].nunique():
        raise ValueError("Compound summary lost or duplicated a compound")
    if best[list(PROPERTY_COLUMNS)].isna().any().any():
        raise ValueError("Compound summary has compounds without structure-derived properties")
    columns = [
        "jcp2022",
        "compound_name",
        "best_target_label",
        "n_retained_targets",
        "best_target_alignment_nap",
        "best_target_alignment_ap",
        "best_target_consistent",
        "activity_nmap",
        "activity_corrected_p_value",
        "full_annotation_degree",
        *PROPERTY_COLUMNS,
    ]
    return best[columns]


def build_property_bins(compound_summary: pd.DataFrame) -> pd.DataFrame:
    """Bin compounds by each property and score the share with a consistent best target.

    Equal-count bins, five rather than the ten the activity family uses, because this
    cohort is two orders of magnitude smaller.  The share is a binomial proportion with
    a Wilson interval.  No figure draws that share any more; the columns stay on the
    table because it is the analysis contract, and dropping them is a separate decision
    from changing what the figures show.
    """
    # One Benjamini-Hochberg family for the whole property block: six properties scored
    # on both outcomes, twelve tests, because that is the set of correlations this
    # analysis computes and every figure that shows one of them shows it from here. The
    # properties are strongly collinear in this cohort (QED against molecular weight is
    # -0.75), so twelve tests are far from twelve independent questions and the
    # correction is conservative rather than lenient.
    correlations = {
        (column, outcome): spearmanr(compound_summary[column], compound_summary[outcome])
        for column, _, _ in PROPERTY_TRAITS
        for outcome in ("best_target_alignment_nap", "activity_nmap")
    }
    keys = list(correlations)
    q_values = dict(
        zip(keys, multipletests([correlations[key].pvalue for key in keys], method="fdr_bh")[1], strict=True)
    )
    # The linear reading of the same relationships, so a renderer never fits anything.
    fits = {key: np.polyfit(compound_summary[key[0]], compound_summary[key[1]], 1) for key in correlations}
    rows = []
    for trait_order, (column, title, xlabel) in enumerate(PROPERTY_TRAITS):
        assignments = pd.qcut(compound_summary[column].rank(method="first"), PROPERTY_BINS, labels=False)
        consistency_fit = fits[(column, "best_target_alignment_nap")]
        activity_fit = fits[(column, "activity_nmap")]
        consistency_test = correlations[(column, "best_target_alignment_nap")]
        activity_test = correlations[(column, "activity_nmap")]
        rho = float(consistency_test.statistic)
        rho_activity = float(activity_test.statistic)
        for index, block in compound_summary.groupby(assignments, sort=True):
            successes = int(block["best_target_consistent"].sum())
            lower, upper = wilson_bounds(successes, len(block))
            # The same bins scored on activity, so one table drives both outcome figures.
            active = int(block["activity_corrected_p_value"].lt(ACTIVITY_Q_THRESHOLD).sum())
            active_lower, active_upper = wilson_bounds(active, len(block))
            rows.append(
                {
                    "trait_order": trait_order,
                    "trait_id": column,
                    "title": title,
                    "xlabel": xlabel,
                    "bin": int(index) + 1,
                    "x": float(block[column].median()),
                    "bin_lower": float(block[column].min()),
                    "bin_upper": float(block[column].max()),
                    "n": int(len(block)),
                    "n_consistent_best_target": successes,
                    "rate": successes / len(block),
                    "lower": lower,
                    "upper": upper,
                    "nap_median": float(block["best_target_alignment_nap"].median()),
                    # The mean, not the median, carries the trend lines: 79% of this
                    # cohort sits at the activity ceiling, so a binned median is 1.0
                    # whenever half a bin is pinned there and steps down when it is not.
                    "nap_mean": float(block["best_target_alignment_nap"].mean()),
                    "nap_mean_se": float(block["best_target_alignment_nap"].std(ddof=1) / np.sqrt(len(block))),
                    "rho": rho,
                    "fit_slope": float(consistency_fit[0]),
                    "fit_intercept": float(consistency_fit[1]),
                    "rho_p_value": float(consistency_test.pvalue),
                    "rho_q_value": float(q_values[(column, "best_target_alignment_nap")]),
                    "n_active": active,
                    "rate_active": active / len(block),
                    "lower_active": active_lower,
                    "upper_active": active_upper,
                    "activity_nmap_median": float(block["activity_nmap"].median()),
                    "activity_nmap_mean": float(block["activity_nmap"].mean()),
                    "activity_nmap_mean_se": float(block["activity_nmap"].std(ddof=1) / np.sqrt(len(block))),
                    "rho_activity": rho_activity,
                    "fit_activity_slope": float(activity_fit[0]),
                    "fit_activity_intercept": float(activity_fit[1]),
                    "rho_activity_p_value": float(activity_test.pvalue),
                    "rho_activity_q_value": float(q_values[(column, "activity_nmap")]),
                }
            )
    bins = pd.DataFrame(rows)
    if len(bins) != len(PROPERTY_TRAITS) * PROPERTY_BINS:
        raise ValueError("Property binning did not produce equal-count bins for every trait")
    if not bins.groupby("trait_id")["n"].sum().eq(len(compound_summary)).all():
        raise ValueError("Property bins lost or duplicated compounds")
    return bins


def build_property_grain_audit(
    compound_summary: pd.DataFrame,
    primary: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Quantify why a property relates oppositely to activity and to consistency.

    Three things are conflated when the activity panels are read against the
    consistency panels: the two outcomes are different, the grain is different
    (one compound per row versus one target per row), and averaging members into
    targets inflates a correlation whether or not anything is there.  This
    separates them for each property.

    Reported per property:
      * the compound-level correlation with activity and with consistency;
      * what pure mediation through activity would predict, and the partial
        correlation that removes it;
      * the same consistency correlation after aggregating to targets;
      * a permutation null for that target-level value that keeps the group sizes
        but assigns members at random, so the inflation from averaging alone is
        visible.
    """
    activity_consistency = float(
        spearmanr(compound_summary["activity_nmap"], compound_summary["best_target_alignment_nap"]).statistic
    )
    # primary already carries the property columns, so a merge would only collide them.
    members = primary[["jcp2022", "target_label", "target_alignment_nap", *PROPERTY_COLUMNS]].copy()
    labels = members["target_label"].to_numpy()
    rng = np.random.default_rng(seed)
    null_draws: dict[str, list[float]] = {column: [] for column in PROPERTY_COLUMNS}
    for _ in range(iterations):
        shuffled = members.assign(target_label=rng.permutation(labels))
        grouped = shuffled.groupby("target_label", sort=True).mean(numeric_only=True)
        for column in PROPERTY_COLUMNS:
            null_draws[column].append(float(spearmanr(grouped[column], grouped["target_alignment_nap"]).statistic))

    observed_groups = members.groupby("target_label", sort=True).mean(numeric_only=True)
    rows = []
    for trait_order, (column, title, _) in enumerate(PROPERTY_TRAITS):
        with_activity = float(spearmanr(compound_summary[column], compound_summary["activity_nmap"]).statistic)
        with_consistency = float(
            spearmanr(compound_summary[column], compound_summary["best_target_alignment_nap"]).statistic
        )
        mediated = with_activity * activity_consistency
        partial = (with_consistency - mediated) / np.sqrt((1 - with_activity**2) * (1 - activity_consistency**2))
        target_level = float(spearmanr(observed_groups[column], observed_groups["target_alignment_nap"]).statistic)
        null = np.abs(np.asarray(null_draws[column]))
        rows.append(
            {
                "trait_order": trait_order,
                "trait_id": column,
                "title": title,
                "n_compounds": int(len(compound_summary)),
                "n_targets": int(len(observed_groups)),
                "rho_property_activity_compound": with_activity,
                "rho_property_consistency_compound": with_consistency,
                "rho_activity_consistency_compound": activity_consistency,
                "rho_property_consistency_mediated_prediction": mediated,
                "partial_property_consistency_given_activity": float(partial),
                "rho_property_consistency_target": target_level,
                "aggregation_inflation": abs(target_level) / abs(with_consistency)
                if with_consistency
                else float("nan"),
                "null_mean_abs_rho_target": float(null.mean()),
                "null_p95_abs_rho_target": float(np.percentile(null, 95)),
                "empirical_p_target": float((null >= abs(target_level)).mean()),
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != len(PROPERTY_TRAITS):
        raise ValueError("Property grain audit lost a trait")
    return audit


def _write_table(path: Path, frame: pd.DataFrame, sort_by: Sequence[str]) -> None:
    ordered = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    ordered.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


@app.command()
def main(
    canonical_well_profiles: Path = typer.Option(DEFAULT_WELL_PROFILES, "--canonical-well-profiles"),
    metadata_db: Path = typer.Option(DEFAULT_METADATA_DB, "--metadata-db"),
    copairs_db: Path = typer.Option(DEFAULT_COPAIRS_DB, "--copairs-db"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    shuffle_iterations: int = typer.Option(DEFAULT_SHUFFLES, "--shuffle-iterations", min=1),
    seed: int = typer.Option(RANDOM_SEED, "--seed"),
) -> None:
    """Run the complete OpenTargets target-alignment analysis."""
    for label, path in (
        ("canonical well profiles", canonical_well_profiles),
        ("metadata database", metadata_db),
        ("copairs database", copairs_db),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    memberships, compounds, labels_by_id, consistent_targets, existing_endpoint_context = load_analysis_inputs(
        metadata_db, copairs_db
    )
    target_audit = build_target_audit(memberships)
    active_ids = set(compounds.loc[compounds["is_active"], "jcp2022"].astype(str))
    primary_targets = set(target_audit.loc[target_audit["eligible_anchor_4"], "target_label"].astype(str))
    primary_query_ids = set(memberships.loc[memberships["target_label"].isin(primary_targets), "jcp2022"].astype(str))
    selected_ids = set(labels_by_id)
    logger.info(
        "OpenTargets activity cohort: {:,} compounds, {:,} active references; primary: {:,} targets, {:,} queries",
        len(selected_ids),
        len(active_ids),
        len(primary_targets),
        len(primary_query_ids),
    )

    (
        full_profiles,
        source2_profiles,
        other_profiles,
        well_summary,
        plate_effect,
        profile_audit,
    ) = load_exact_well_profiles(
        canonical_well_profiles,
        metadata_db,
        selected_ids,
        primary_query_ids,
        primary_query_ids | active_ids,
    )
    compounds = (
        compounds.merge(well_summary, on="jcp2022", how="left", validate="one_to_one")
        .merge(plate_effect, on="jcp2022", how="left", validate="one_to_one")
        .sort_values("jcp2022")
        .reset_index(drop=True)
    )
    if compounds[["eligible_wells", "source_count", "median_cell_count"]].isna().any().any():
        raise ValueError("Exact well-profile metadata does not cover every analyzed OpenTargets compound")

    all_scores = score_target_alignment(
        full_profiles,
        full_profiles,
        labels_by_id,
        sorted(selected_ids),
        sorted(active_ids),
    ).rename(
        columns={
            "average_precision": "target_alignment_ap",
            "normalized_average_precision": "target_alignment_nap",
            "n_pos_pairs": "target_alignment_n_pos_pairs",
            "n_total_pairs": "target_alignment_n_total_pairs",
            "expected_positive_anchors": "independent_positive_anchors",
            "expected_disjoint_negatives": "independent_disjoint_negatives",
        }
    )
    analytic = memberships.merge(
        all_scores,
        on=["jcp2022", "target_label"],
        how="inner",
        validate="one_to_one",
    ).merge(
        target_audit.drop(columns=["member_ids", "anchor_ids"]),
        on="target_label",
        how="left",
        validate="many_to_one",
    )
    compound_fields = [
        "jcp2022",
        "compound_name",
        "murcko_scaffold",
        "valid_molecule",
        "smiles",
        "eligible_wells",
        "source_count",
        "source_signature",
        "median_cell_count",
        "cell_count_percentile",
        "plate_matched_negcon_cosine_distance",
        "effect_size_wells",
        "effect_size_strata",
        *PROPERTY_COLUMNS,
    ]
    analytic = analytic.merge(compounds[compound_fields], on="jcp2022", how="left", validate="many_to_one")
    primary = analytic.loc[analytic["eligible_anchor_4"]].copy()
    observed_primary = (
        int(primary["target_label"].nunique()),
        int(len(primary)),
        int(primary["jcp2022"].nunique()),
    )
    if primary.duplicated(["jcp2022", "target_label"]).any():
        raise ValueError("Primary target-alignment memberships are not unique")
    if primary[["target_alignment_nap", "target_alignment_ap"]].isna().any().any():
        raise ValueError("Primary target-alignment cohort contains missing scores")

    primary = add_centroid_calibration(primary, full_profiles, labels_by_id, active_ids, target_audit)
    primary = add_source_crossfit(primary, source2_profiles, other_profiles, labels_by_id, active_ids)
    scaffolds = compounds.set_index("jcp2022")["murcko_scaffold"].fillna("").astype(str).to_dict()
    primary = add_leave_scaffold_out(primary, full_profiles, labels_by_id, active_ids, scaffolds)
    primary = add_pair_tie_audit(primary, full_profiles, labels_by_id, active_ids)
    primary, exceptions, quadrant_thresholds = assign_scientific_quadrants(primary)

    matched_nonmembers = build_matched_nonmembers(
        primary,
        compounds,
        full_profiles,
        labels_by_id,
        active_ids,
    )
    if len(matched_nonmembers) != len(primary):
        raise ValueError("Matched-nonmember calibration did not retain every primary membership")

    models = []
    model_details: dict[str, dict[str, object]] = {}
    for threshold in ANCHOR_THRESHOLDS:
        cohort = analytic.loc[analytic[f"eligible_anchor_{threshold}"]].copy()
        model_id = "min_refs_4_primary" if threshold == PRIMARY_MIN_ACTIVE_ANCHORS else f"min_refs_{threshold}"
        table, details = fit_across_target_relationship(
            cohort,
            outcome="target_alignment_nap",
            model_id=model_id,
        )
        models.append(table)
        model_details[model_id] = details
    distinct_primary = deduplicate_exact_target_sets(primary)
    table, details = fit_across_target_relationship(
        distinct_primary,
        outcome="target_alignment_nap",
        model_id="min_refs_4_distinct_exact_sets",
    )
    models.append(table)
    model_details["min_refs_4_distinct_exact_sets"] = details
    for model_id, outcome in (
        ("source2_query_other_reference", "source2_query_target_alignment_nap"),
        ("other_query_source2_reference", "other_source_query_target_alignment_nap"),
        ("source_crossfit_mean", "source_crossfit_mean_target_alignment_nap"),
        ("leave_scaffold_out", "leave_scaffold_out_target_alignment_nap"),
        ("centroid_wrong_target_percentile", "centroid_percentile_vs_wrong"),
    ):
        table, details = fit_across_target_relationship(primary, outcome=outcome, model_id=model_id)
        models.append(table)
        model_details[model_id] = details
    model_table = pd.concat(models, ignore_index=True)

    shuffle_table = shuffled_label_calibration(
        full_profiles,
        labels_by_id,
        compounds,
        active_ids,
        primary_targets,
        shuffle_iterations,
        seed,
    )

    sensitivity_records: list[dict[str, object]] = []
    for threshold in ANCHOR_THRESHOLDS:
        cohort = analytic.loc[analytic[f"eligible_anchor_{threshold}"]].copy()
        correlation = across_target_spearman(cohort, "target_alignment_nap")
        distinct_correlation = across_target_spearman(
            cohort,
            "target_alignment_nap",
            deduplicate_exact_sets=True,
        )
        sensitivity_records.append(
            {
                "analysis": "minimum_active_reference_threshold",
                "status": "computed",
                "minimum_active_references": threshold,
                "targets": int(cohort["target_label"].nunique()),
                "memberships": int(len(cohort)),
                "queries": int(cohort["jcp2022"].nunique()),
                "active_queries": int(cohort.loc[cohort["is_active"], "jcp2022"].nunique()),
                "inactive_queries": int(cohort.loc[~cohort["is_active"], "jcp2022"].nunique()),
                "estimate": _target_balanced_mean(cohort, "target_alignment_nap"),
                "metric": "equal-target mean target-alignment nAP",
                "descriptive_spearman": correlation,
                "distinct_exact_set_spearman": distinct_correlation,
                "notes": (
                    "one equally weighted row per target; eligibility was defined at >=5 analyzed members "
                    "before active-reference filtering"
                ),
            }
        )
    observed_mean = _target_balanced_mean(primary, "target_alignment_nap")
    observed_across_target_rho = across_target_spearman(primary, "target_alignment_nap")
    observed_distinct_set_rho = across_target_spearman(
        primary,
        "target_alignment_nap",
        deduplicate_exact_sets=True,
    )
    shuffle_mean_exceedances = int(shuffle_table["target_balanced_mean_nap"].ge(observed_mean).sum())
    shuffle_resolution = 1.0 / (len(shuffle_table) + 1)
    shuffle_mean_p = (1 + shuffle_mean_exceedances) / (len(shuffle_table) + 1)
    sensitivity_records.extend(
        [
            {
                "analysis": "shuffled_label_calibration",
                "status": "computed",
                "estimate": observed_mean,
                "reference_estimate": float(shuffle_table["target_balanced_mean_nap"].mean()),
                "empirical_p_value": shuffle_mean_p,
                "metric": "equal-target mean target-alignment nAP",
                "iterations": int(len(shuffle_table)),
                "notes": (
                    "complete label vectors permuted within active/inactive strata; "
                    f"{shuffle_mean_exceedances}/{len(shuffle_table)} null replicates were >= observed; "
                    f"Monte Carlo resolution {shuffle_resolution:.6g}"
                ),
            },
            {
                "analysis": "wrong_target_centroid_calibration",
                "status": "computed",
                "estimate": float(primary["centroid_percentile_vs_wrong"].mean()),
                "reference_estimate": 50.0,
                "metric": "mean correct-target percentile among distinct wrong-anchor prototypes",
                "memberships": int(len(primary)),
                "notes": "query identity excluded from every correct and wrong prototype",
            },
            {
                "analysis": "matched_nonmember_centroid_calibration",
                "status": "computed",
                "estimate": _target_balanced_mean(
                    matched_nonmembers.rename(columns={"member_minus_nonmember_centroid_cosine": "matched_delta"}),
                    "matched_delta",
                ),
                "reference_estimate": 0.0,
                "metric": "equal-target member minus matched-nonmember centroid cosine",
                "memberships": int(len(matched_nonmembers)),
                "notes": "matched on activity status and source signature when available, then nearest activity/cell count",
            },
            {
                "analysis": "source_crossfit",
                "status": "computed",
                "estimate": float(
                    spearmanr(
                        primary["target_alignment_nap"], primary["source_crossfit_mean_target_alignment_nap"]
                    ).statistic
                ),
                "metric": "Spearman primary vs bidirectional source-crossfit target-alignment nAP",
                "memberships": int(len(primary)),
                "notes": "source_2 queries use non-source_2 references and vice versa",
            },
            {
                "analysis": "leave_scaffold_out",
                "status": "computed"
                if primary["leave_scaffold_out_target_alignment_nap"].notna().all()
                else "partially_blocked",
                "estimate": float(
                    spearmanr(
                        primary["target_alignment_nap"],
                        primary["leave_scaffold_out_target_alignment_nap"],
                        nan_policy="omit",
                    ).statistic
                ),
                "metric": "Spearman primary vs leave-scaffold-out target-alignment nAP",
                "memberships": int(primary["leave_scaffold_out_target_alignment_nap"].notna().sum()),
                "blocked_memberships": int(primary["leave_scaffold_out_target_alignment_nap"].isna().sum()),
                "notes": "all references with the query's exact nonempty Murcko scaffold excluded",
            },
            {
                "analysis": "plate_matched_effect_size",
                "status": "computed",
                "estimate": float(
                    spearmanr(
                        primary[["jcp2022", "activity_nmap", "plate_matched_negcon_cosine_distance"]].drop_duplicates(
                            "jcp2022"
                        )["activity_nmap"],
                        primary[["jcp2022", "activity_nmap", "plate_matched_negcon_cosine_distance"]].drop_duplicates(
                            "jcp2022"
                        )["plate_matched_negcon_cosine_distance"],
                    ).statistic
                ),
                "metric": "compound-level Spearman activity nMAP vs plate-matched negcon cosine distance",
                "queries": int(primary["jcp2022"].nunique()),
                "notes": "separate literal phenotype-magnitude sensitivity; activity nMAP is not relabeled as magnitude",
            },
        ]
    )
    primary_well_counts = compounds.set_index("jcp2022").reindex(sorted(primary_query_ids))["eligible_wells"]
    blocked_twofold = int(primary_well_counts.lt(4).sum())
    sensitivity_records.append(
        {
            "analysis": "fully_disjoint_activity_alignment_crossfit",
            "status": "blocked",
            "queries": int(len(primary_well_counts)),
            "blocked_queries": blocked_twofold,
            "metric": "two-fold canonical activity nMAP on one fold and target alignment on the other",
            "notes": (
                f"{blocked_twofold} queries have fewer than four eligible wells, so at least one fold lacks two "
                "positive-replicate wells; canonical activity also pools all sources. The complete bidirectional "
                "source-crossfit target-alignment sensitivity is reported instead."
            ),
        }
    )
    sensitivity_table = pd.DataFrame(sensitivity_records)

    pair_summary = (
        primary.groupby("target_label", sort=True)
        .agg(
            scored_memberships=("jcp2022", "size"),
            scored_queries=("jcp2022", "nunique"),
            minimum_positive_pairs=("target_alignment_n_pos_pairs", "min"),
            maximum_positive_pairs=("target_alignment_n_pos_pairs", "max"),
            minimum_total_pairs=("target_alignment_n_total_pairs", "min"),
            maximum_total_pairs=("target_alignment_n_total_pairs", "max"),
            minimum_disjoint_negatives=("independent_disjoint_negatives", "min"),
            maximum_disjoint_negatives=("independent_disjoint_negatives", "max"),
            positive_negative_similarity_ties=("positive_negative_similarity_ties", "sum"),
        )
        .reset_index()
    )
    target_audit = target_audit.merge(pair_summary, on="target_label", how="left", validate="one_to_one")
    target_audit["profile_members_covered"] = target_audit["member_ids"].map(
        lambda value: sum(member in full_profiles.index for member in str(value).split("|") if member)
    )
    target_audit["profile_coverage_complete"] = target_audit["profile_members_covered"].eq(
        target_audit["analyzed_members"]
    )

    primary_only_columns = [column for column in primary.columns if column not in analytic.columns]
    membership_export = analytic.loc[analytic["eligible_anchor_3"]].merge(
        primary[["jcp2022", "target_label", *primary_only_columns]],
        on=["jcp2022", "target_label"],
        how="left",
        validate="one_to_one",
    )
    if len(membership_export) != int(analytic["eligible_anchor_3"].sum()):
        raise ValueError("The >=3-anchor membership export lost scored rows")

    membership_columns = [
        "jcp2022",
        "compound_name",
        "target_label",
        "activity_nmap",
        "activity_corrected_p_value",
        "is_active",
        "target_alignment_ap",
        "target_alignment_nap",
        "target_alignment_n_pos_pairs",
        "target_alignment_n_total_pairs",
        "independent_positive_anchors",
        "independent_disjoint_negatives",
        "positive_negative_similarity_ties",
        "analyzed_members",
        "active_anchors",
        "eligible_analyzed_ge5",
        "eligible_anchor_3",
        "eligible_anchor_4",
        "eligible_anchor_5",
        "full_annotation_degree",
        "membership_set_id",
        "anchor_set_id",
        "joint_set_id",
        "targets_with_joint_set",
        "centroid_cosine",
        "centroid_margin_vs_best_wrong",
        "centroid_percentile_vs_wrong",
        "wrong_target_prototypes",
        "source2_query_target_alignment_nap",
        "other_source_query_target_alignment_nap",
        "source_crossfit_mean_target_alignment_nap",
        "leave_scaffold_out_target_alignment_nap",
        "leave_scaffold_out_n_pos_pairs",
        "leave_scaffold_out_n_total_pairs",
        "leave_scaffold_out_status",
        "murcko_scaffold",
        "eligible_wells",
        "source_count",
        "source_signature",
        "median_cell_count",
        "cell_count_percentile",
        "plate_matched_negcon_cosine_distance",
        "effect_size_wells",
        "effect_size_strata",
        "scientific_quadrant",
        "exception_number",
    ]
    _write_table(
        output_dir / MEMBERSHIP_TABLE,
        membership_export[membership_columns],
        ["target_label", "jcp2022"],
    )
    _write_table(output_dir / TARGET_AUDIT_TABLE, target_audit, ["target_label"])
    _write_table(output_dir / MODEL_TABLE, model_table, ["model_id", "component"])
    _write_table(output_dir / SENSITIVITY_TABLE, sensitivity_table, ["analysis", "metric"])
    _write_table(output_dir / SHUFFLE_TABLE, shuffle_table, ["iteration"])
    _write_table(output_dir / MATCHED_TABLE, matched_nonmembers, ["target_label", "jcp2022"])
    _write_table(
        output_dir / QUADRANT_TABLE,
        build_quadrant_export(primary),
        ["scientific_category", "consistency_rank_within_category", "jcp2022", "target_label"],
    )
    # The renderer draws the across-target panel from these two tables, so it never
    # refits the model and never needs statsmodels or copairs.
    primary_details = model_details["min_refs_4_primary"]
    _write_table(output_dir / TARGET_SUMMARY_TABLE, primary_details["data"], ["target_label"])
    _write_table(output_dir / CURVE_TABLE, _across_target_curve(primary_details), ["mean_activity_nmap"])
    compound_summary = summarize_compounds(primary, consistent_targets)
    _write_table(output_dir / COMPOUND_SUMMARY_TABLE, compound_summary, ["jcp2022"])
    property_bins = build_property_bins(compound_summary)
    _write_table(output_dir / PROPERTY_BIN_TABLE, property_bins, ["trait_order", "bin"])
    grain_audit = build_property_grain_audit(compound_summary, primary, GRAIN_AUDIT_PERMUTATIONS, seed)
    _write_table(output_dir / GRAIN_AUDIT_TABLE, grain_audit, ["trait_order"])

    observed_counts = {
        "annotated_targets": int(memberships["target_label"].nunique()),
        "annotated_memberships": int(len(memberships)),
        "annotated_compounds": int(memberships["jcp2022"].nunique()),
        "annotated_active_compounds": int(memberships.loc[memberships["is_active"], "jcp2022"].nunique()),
        "annotated_inactive_compounds": int(memberships.loc[~memberships["is_active"], "jcp2022"].nunique()),
        "analyzed_ge5_targets": int(target_audit["eligible_analyzed_ge5"].sum()),
        "analyzed_ge5_memberships": int(
            memberships["target_label"]
            .isin(target_audit.loc[target_audit["eligible_analyzed_ge5"], "target_label"])
            .sum()
        ),
        "analyzed_ge5_queries": int(
            memberships.loc[
                memberships["target_label"].isin(
                    target_audit.loc[target_audit["eligible_analyzed_ge5"], "target_label"]
                ),
                "jcp2022",
            ].nunique()
        ),
        "exported_min3_reference_union_targets": int(membership_export["target_label"].nunique()),
        "exported_min3_reference_union_memberships": int(len(membership_export)),
        "exported_min3_reference_union_queries": int(membership_export["jcp2022"].nunique()),
        "primary_targets": observed_primary[0],
        "primary_memberships": observed_primary[1],
        "primary_queries": observed_primary[2],
        "primary_active_queries": int(primary.loc[primary["is_active"], "jcp2022"].nunique()),
        "primary_inactive_queries": int(primary.loc[~primary["is_active"], "jcp2022"].nunique()),
        "primary_distinct_membership_sets": int(
            target_audit.loc[target_audit["eligible_anchor_4"], "membership_set_id"].nunique()
        ),
        "primary_distinct_reference_sets": int(
            target_audit.loc[target_audit["eligible_anchor_4"], "anchor_set_id"].nunique()
        ),
        "primary_distinct_joint_sets": int(
            target_audit.loc[target_audit["eligible_anchor_4"], "joint_set_id"].nunique()
        ),
    }
    grain_row = grain_audit.set_index("trait_id")
    property_grain = {
        "question": ("Why a property can relate negatively to activity and positively to consistency"),
        "activity_consistency_rho_per_compound": float(grain_row["rho_activity_consistency_compound"].iloc[0]),
        "per_trait": {
            trait: {
                "rho_with_activity_per_compound": float(row["rho_property_activity_compound"]),
                "rho_with_consistency_per_compound": float(row["rho_property_consistency_compound"]),
                "mediation_only_prediction": float(row["rho_property_consistency_mediated_prediction"]),
                "partial_given_activity": float(row["partial_property_consistency_given_activity"]),
                "rho_with_consistency_per_target": float(row["rho_property_consistency_target"]),
                "aggregation_inflation": float(row["aggregation_inflation"]),
                "null_p95_abs_rho_target": float(row["null_p95_abs_rho_target"]),
                "empirical_p_target": float(row["empirical_p_target"]),
            }
            for trait, row in grain_row.iterrows()
        },
        "permutations": GRAIN_AUDIT_PERMUTATIONS,
        "interpretation": (
            "Target-level correlations are compared against a null that keeps the group sizes and "
            "assigns members at random, because averaging four to seventeen members into a target "
            "inflates a correlation whether or not an association exists."
        ),
    }
    compound_reduction = {
        "grain": "one row per compound at its best-matching retained target",
        "statistic": "maximum target-alignment nAP across the compound's retained targets",
        "compounds": int(len(compound_summary)),
        "single_target_compounds": int(compound_summary["n_retained_targets"].eq(1).sum()),
        "multi_target_compounds": int(compound_summary["n_retained_targets"].gt(1).sum()),
        "maximum_retained_targets": int(compound_summary["n_retained_targets"].max()),
        "compounds_with_consistent_best_target": int(compound_summary["best_target_consistent"].sum()),
        "distinct_best_targets": int(compound_summary["best_target_label"].nunique()),
        "selection_bias_spearman_targets_vs_max_nap": float(
            spearmanr(compound_summary["n_retained_targets"], compound_summary["best_target_alignment_nap"]).statistic
        ),
        "caveat": (
            "A maximum over k targets rises with k, so this level is not comparable to a group mAP; "
            "n_retained_targets is exported so the dependence stays visible."
        ),
    }
    expected_snapshot = {
        "annotated_compounds": 574,
        "annotated_active_compounds": 297,
        "annotated_inactive_compounds": 277,
        "analyzed_ge5_targets": 67,
        "analyzed_ge5_memberships": 634,
        "analyzed_ge5_queries": 243,
        "primary_targets": 43,
        "primary_memberships": 402,
        "primary_queries": 168,
    }
    drift = {
        key: {
            "expected": expected,
            "observed": observed_counts[key],
            "matches": observed_counts[key] == expected,
        }
        for key, expected in expected_snapshot.items()
    }
    numbers = {
        "activity_contract": {
            "dataset": DATASET,
            "columns": COLUMNS,
            "preprocessing": ACTIVITY_PREPROCESSING,
            "filter": FILTER_NAME,
            "activity_params": ACTIVITY_PARAMS,
            "active_reference_definition": f"strict corrected_p_value < {ACTIVITY_Q_THRESHOLD}",
            "query_inclusion": "all OpenTargets-annotated compounds with canonical activity, regardless of significance",
        },
        "annotation_contract": {
            "table": ANNOTATION_TABLE,
            "column": ANNOTATION_COLUMN,
            "multilabel_semantics": "pipe-split complete OpenTargets label vectors",
        },
        "profile_contract": {
            "path": str(canonical_well_profiles),
            "aggregation": "median by JCP2022 after excluding TARGET2 and positive controls",
            **profile_audit,
        },
        "observed_counts": observed_counts,
        "compound_level_reduction": compound_reduction,
        "property_grain_audit": property_grain,
        "expected_snapshot_drift_audit": drift,
        "minimum_active_reference_sensitivity": {
            str(threshold): {
                "targets": int(analytic.loc[analytic[f"eligible_anchor_{threshold}"], "target_label"].nunique()),
                "memberships": int(analytic[f"eligible_anchor_{threshold}"].sum()),
                "queries": int(analytic.loc[analytic[f"eligible_anchor_{threshold}"], "jcp2022"].nunique()),
            }
            for threshold in ANCHOR_THRESHOLDS
        },
        "primary_pair_denominators": {
            "n_pos_min": int(primary["target_alignment_n_pos_pairs"].min()),
            "n_pos_median": float(primary["target_alignment_n_pos_pairs"].median()),
            "n_pos_max": int(primary["target_alignment_n_pos_pairs"].max()),
            "n_total_min": int(primary["target_alignment_n_total_pairs"].min()),
            "n_total_median": float(primary["target_alignment_n_total_pairs"].median()),
            "n_total_max": int(primary["target_alignment_n_total_pairs"].max()),
            "identity_exclusion": "query JCP2022 differs from every positive and negative reference role",
            "independent_denominator_checks_passed": int(len(primary)),
            "positive_negative_similarity_ties": int(primary["positive_negative_similarity_ties"].sum()),
        },
        "quadrant_thresholds": quadrant_thresholds,
        "quadrant_counts": {
            key: int(value) for key, value in primary["scientific_quadrant"].value_counts().sort_index().items()
        },
        "across_target_relationship": {
            "unit": "one equally weighted row per target label",
            "target_labels": int(primary["target_label"].nunique()),
            "distinct_exact_sets": int(target_audit.loc[target_audit["eligible_anchor_4"], "joint_set_id"].nunique()),
            "spearman_all_target_labels": observed_across_target_rho,
            "spearman_distinct_exact_sets": observed_distinct_set_rho,
            "within_target_analysis": "not estimated or interpreted",
        },
        "scatter_panel_relationship": {
            "unit": "the unaggregated activity-versus-consistency scatter, at both grains it is drawn",
            "memberships": int(len(primary)),
            "compounds": int(len(compound_summary)),
            "spearman_memberships": float(
                spearmanr(primary["activity_nmap"], primary["target_alignment_nap"]).statistic
            ),
            "spearman_compounds": float(
                spearmanr(compound_summary["activity_nmap"], compound_summary["best_target_alignment_nap"]).statistic
            ),
        },
        "across_target_aggregation_null": across_target_aggregation_null(
            primary, observed_across_target_rho, GRAIN_AUDIT_PERMUTATIONS, seed
        ),
        "shuffled_label_calibration": {
            "iterations": int(len(shuffle_table)),
            "observed_equal_target_mean_nap": observed_mean,
            "null_mean_equal_target_mean_nap": float(shuffle_table["target_balanced_mean_nap"].mean()),
            "empirical_p_value": shuffle_mean_p,
            "equal_target_null_exceedances": shuffle_mean_exceedances,
            "monte_carlo_resolution": shuffle_resolution,
            "at_resolution_floor": shuffle_mean_exceedances == 0,
        },
        "matched_nonmember_calibration": {
            "memberships": int(len(matched_nonmembers)),
            "multilabel_disjoint_controls": int(matched_nonmembers["label_sets_disjoint"].sum()),
            "maximum_shared_label_count": int(matched_nonmembers["shared_label_count"].max()),
            "control_reuse_within_target": int(matched_nonmembers["control_reused_within_target"].sum()),
            "match_tiers": {
                str(key): int(value) for key, value in matched_nonmembers["match_tier"].value_counts().items()
            },
            "activity_absolute_difference_median": float(
                matched_nonmembers["activity_nmap_absolute_difference"].median()
            ),
            "activity_absolute_difference_p95": float(
                matched_nonmembers["activity_nmap_absolute_difference"].quantile(0.95)
            ),
            "activity_absolute_difference_maximum": float(
                matched_nonmembers["activity_nmap_absolute_difference"].max()
            ),
            "cell_count_percentile_absolute_difference_median": float(
                (
                    matched_nonmembers["cell_count_percentile_member"]
                    - matched_nonmembers["cell_count_percentile_nonmember"]
                )
                .abs()
                .median()
            ),
        },
        "existing_canonical_activity_filtered_group_consistency": {
            **existing_endpoint_context,
            "status": "separate endpoint; no canonical compound-activity-association or OpenTargets group-consistency output modified",
        },
        "limitations": [
            "This is observational phenotypic consistency with target annotations, not evidence that the annotated target causes the phenotype.",
            "Canonical activity nMAP is a retrieval statistic, not literal phenotype magnitude; the plate-matched cosine-distance sensitivity is separate.",
            "Bidirectional source cross-fitting removes same-source query/reference coupling from the phenotypic-consistency score, but canonical activity x uses all sources.",
            sensitivity_records[-1]["notes"],
            "Matched nonmembers are multilabel-disjoint but come from the same 574-compound OpenTargets/activity cohort and may be reused across different targets.",
            "Named examples use explicit rank-based descriptive rules and are not inferential calls; the dense low-consistency ceiling cluster is summarized in full rather than represented by selected members.",
        ],
    }
    write_summary(
        output_dir / "summary.json",
        description=(
            "Exploratory OpenTargets analysis asking whether target groups with higher mean compound activity "
            "nMAP also have higher mean phenotypic consistency with their annotated targets. Consistency is "
            "quantified from membership-level asymmetric copairs target-alignment nAP. All memberships are retained, "
            "while relationship estimates use one equally weighted row per target and no within-target analysis."
        ),
        numbers=numbers,
        files=index_files(output_dir, relative_to=output_dir),
        analysis_id=ANALYSIS_ID,
        configuration={
            "minimum_analyzed_members": MIN_ANALYZED_MEMBERS,
            "primary_minimum_active_references": PRIMARY_MIN_ACTIVE_ANCHORS,
            "minimum_active_reference_thresholds": list(ANCHOR_THRESHOLDS),
            "source_holdout": SOURCE_HOLDOUT,
            "shuffle_iterations": shuffle_iterations,
            "seed": seed,
            "copairs_api": "copairs.map.multilabel.average_precision",
            "distance": "cosine",
        },
    )
    logger.info("Wrote the complete target-alignment analysis to {}", output_dir)


if __name__ == "__main__":
    app()
