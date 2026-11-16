"""Cache the expensive pair analyses used by Supplementary Figure STRUCTURE.

The active-only structure-morphology views use copairs activity calls at 10% FDR,
valid Morgan fingerprints, and valid source-7-excluded perturbation profiles. The
global exact-Tanimoto analysis searches every valid fingerprint/profile pair and
classifies phenotypic activity states after pair discovery.

This analysis deliberately excludes presentation and the canonical notebook 2.40
and 2.32 results. The direct figure script reads those analyses as explicit inputs.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from loguru import logger
from scipy.stats import fisher_exact, spearmanr

from jump_production.gpu import has_gpu
from jump_production.queries import query_activity_results
from jump_production.statistics import permutation_null_correlation
from jump_production.summaries import write_summary

DEFAULT_DATASET = "compound_no_source7"
DEFAULT_ACTIVITY_PREPROCESSING = "activity_no_target2"
DEFAULT_ACTIVITY_FILTER = "all_sources"
DEFAULT_ACTIVITY_PARAMS = "default"
DEFAULT_MORGAN_FP = Path("data/interim/compound_featurization/morgan_fp.npz")
DEFAULT_PERTURBATION_PROFILES = Path("data/interim/anndata/compound_no_source7_perturbation.h5ad")
DEFAULT_METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")
DEFAULT_COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
DEFAULT_OUTPUT_ROOT = Path("data/processed/structure-activity-pairs")

# Cliff / overlap policy. K_PAIRS matches notebook 2.31 (activity cliffs); K_OVERLAP
# matches notebook 2.05 (structure-morphology neighbor overlap).
K_PAIRS = 100
K_OVERLAP = 50
TANIMOTO_THRESHOLD_NUMERATOR = 7
TANIMOTO_THRESHOLD_DENOMINATOR = 10
STRUCT_HIGH_THRESHOLD = TANIMOTO_THRESHOLD_NUMERATOR / TANIMOTO_THRESHOLD_DENOMINATOR
MORPH_LOW_THRESHOLD = 0.3  # cosine similarity
EXACT_TANIMOTO_GPU_BLOCK_SIZE = 4096
EXACT_TANIMOTO_CPU_BLOCK_SIZE = 128
ALL_PAIR_GPU_BLOCK_SIZE = 1024
ALL_PAIR_STRUCTURE_BINS = 20
ALL_PAIR_MORPHOLOGY_BINS = 1000
MOA_MIN_COMPOUNDS = 10
MOA_TOP_N = 20
# Target-annotation contracts for the phenotypic-cliff divergence test. Sources are
# compared within their own vocabulary (Hub gene symbols, ChEMBL UniProt accessions)
# and never pooled, so a Hub-only compound and a ChEMBL-only compound are not comparable.
TARGET_DIVERGENCE_SOURCES = {
    "hub_targets": "Drug Repurposing Hub",
    "chembl_targets": "ChEMBL",
}
TARGET_DIVERGENCE_CONTRACTS = {
    "repurposing_hub_and_chembl": ("hub_targets", "chembl_targets"),
    "chembl_only": ("chembl_targets",),
}
TARGET_DIVERGENCE_POLICY = (
    "A structurally similar pair is annotated under a contract when both compounds carry "
    "targets from the same source for at least one source in the contract, and divergent "
    "when the two compounds share no target within any such source. Phenotypic cliffs "
    "(morphology cosine < morph_low_threshold) are compared with concordant pairs "
    "(morphology cosine >= 1 - morph_low_threshold) by a two-sided Fisher exact test."
)

app = typer.Typer(add_completion=False)


def build_all_pair_similarity_histogram(
    fingerprint_matrix: np.ndarray,
    morphology_matrix: np.ndarray,
) -> pd.DataFrame:
    """Count every unordered active-compound pair in a joint similarity histogram."""
    if not has_gpu():
        raise RuntimeError("The exhaustive all-pairs similarity histogram requires a CUDA GPU")

    import cupy as cp

    fingerprints = fingerprint_matrix.astype(np.float64, copy=False)
    morphology = morphology_matrix.astype(np.float64, copy=False)
    fingerprints /= np.linalg.norm(fingerprints, axis=1, keepdims=True)
    morphology /= np.linalg.norm(morphology, axis=1, keepdims=True)
    fingerprints_gpu = cp.asarray(fingerprints)
    morphology_gpu = cp.asarray(morphology)
    n_compounds = len(fingerprints)
    histogram_gpu = cp.zeros(
        ALL_PAIR_STRUCTURE_BINS * ALL_PAIR_MORPHOLOGY_BINS,
        dtype=cp.int64,
    )
    all_columns = cp.arange(n_compounds)[None, :]

    for start in range(0, n_compounds, ALL_PAIR_GPU_BLOCK_SIZE):
        stop = min(start + ALL_PAIR_GPU_BLOCK_SIZE, n_compounds)
        structure_block = fingerprints_gpu[start:stop] @ fingerprints_gpu.T
        morphology_block = morphology_gpu[start:stop] @ morphology_gpu.T
        upper_triangle = all_columns > cp.arange(start, stop)[:, None]
        structure_values = cp.clip(structure_block[upper_triangle], 0.0, 1.0)
        morphology_values = cp.clip(morphology_block[upper_triangle], -1.0, 1.0)
        structure_bins = cp.minimum(
            (structure_values * ALL_PAIR_STRUCTURE_BINS).astype(cp.int32),
            ALL_PAIR_STRUCTURE_BINS - 1,
        )
        morphology_bins = cp.minimum(
            ((morphology_values + 1.0) * 0.5 * ALL_PAIR_MORPHOLOGY_BINS).astype(cp.int32),
            ALL_PAIR_MORPHOLOGY_BINS - 1,
        )
        histogram_gpu += cp.bincount(
            structure_bins * ALL_PAIR_MORPHOLOGY_BINS + morphology_bins,
            minlength=ALL_PAIR_STRUCTURE_BINS * ALL_PAIR_MORPHOLOGY_BINS,
        )

    histogram = cp.asnumpy(histogram_gpu).reshape(
        ALL_PAIR_STRUCTURE_BINS,
        ALL_PAIR_MORPHOLOGY_BINS,
    )
    expected_pairs = n_compounds * (n_compounds - 1) // 2
    if int(histogram.sum()) != expected_pairs:
        raise ValueError("The exhaustive similarity histogram does not contain every unordered pair")

    structure_edges = np.linspace(0.0, 1.0, ALL_PAIR_STRUCTURE_BINS + 1)
    morphology_edges = np.linspace(-1.0, 1.0, ALL_PAIR_MORPHOLOGY_BINS + 1)
    structure_indices, morphology_indices = np.indices(histogram.shape)
    return pd.DataFrame(
        {
            "structure_bin_left": structure_edges[structure_indices.ravel()],
            "structure_bin_right": structure_edges[structure_indices.ravel() + 1],
            "morphology_bin_left": morphology_edges[morphology_indices.ravel()],
            "morphology_bin_right": morphology_edges[morphology_indices.ravel() + 1],
            "n_pairs": histogram.ravel(),
        }
    )


def load_fingerprints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Morgan fingerprints (2048-bit) and their JCP2022 ids."""
    data = np.load(path, allow_pickle=True)
    fp = data["fingerprints"]
    jcp = data["jcp2022"]
    logger.info(f"Morgan fingerprints: {fp.shape[0]:,} compounds x {fp.shape[1]} bits")
    return fp, jcp


def exact_tanimoto_pairs(
    fingerprints: np.ndarray,
    *,
    use_gpu: bool | None = None,
    block_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every unordered fingerprint pair with exact Tanimoto >= 0.7.

    For binary fingerprints with intersection c and bit counts a and b,
    c / (a + b - c) >= 7 / 10 exactly when 17c >= 7(a + b). The
    integer test avoids floating-point ambiguity at the threshold.
    """
    fingerprints = np.asarray(fingerprints)
    if fingerprints.ndim != 2:
        raise ValueError("Morgan fingerprints must be a two-dimensional matrix")
    if np.any((fingerprints != 0) & (fingerprints != 1)):
        raise ValueError("Morgan fingerprints must be binary")
    bit_counts = fingerprints.sum(axis=1, dtype=np.int32)
    if np.any(bit_counts == 0):
        raise ValueError("Zero-bit fingerprints must be removed before exact pair search")

    if use_gpu is None:
        use_gpu = has_gpu()
    if block_size is None:
        block_size = EXACT_TANIMOTO_GPU_BLOCK_SIZE if use_gpu else EXACT_TANIMOTO_CPU_BLOCK_SIZE
    if block_size < 1:
        raise ValueError("Exact Tanimoto block size must be positive")

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    intersection_parts: list[np.ndarray] = []
    n_compounds = len(fingerprints)
    threshold_left = TANIMOTO_THRESHOLD_DENOMINATOR + TANIMOTO_THRESHOLD_NUMERATOR

    if use_gpu:
        import cupy as cp

        logger.info(f"Enumerating exact Tanimoto pairs on GPU in blocks of {block_size:,}")
        matrix = cp.asarray(fingerprints, dtype=cp.float32)
        counts = cp.asarray(bit_counts, dtype=cp.int32)
        for start in range(0, n_compounds, block_size):
            stop = min(start + block_size, n_compounds)
            intersections = cp.rint(matrix[start:stop] @ matrix.T).astype(cp.int32)
            close = threshold_left * intersections >= TANIMOTO_THRESHOLD_NUMERATOR * (
                counts[start:stop, None] + counts[None, :]
            )
            local_rows, cols = cp.nonzero(close)
            rows = local_rows + start
            upper = cols > rows
            if bool(cp.any(upper)):
                row_parts.append(cp.asnumpy(rows[upper]))
                col_parts.append(cp.asnumpy(cols[upper]))
                intersection_parts.append(cp.asnumpy(intersections[local_rows[upper], cols[upper]]))
    else:
        logger.warning("Enumerating exact Tanimoto pairs on CPU; this is exact but slow for the full compound set")
        matrix = fingerprints.astype(np.float32, copy=False)
        for start in range(0, n_compounds, block_size):
            stop = min(start + block_size, n_compounds)
            intersections = np.rint(matrix[start:stop] @ matrix.T).astype(np.int32)
            close = threshold_left * intersections >= TANIMOTO_THRESHOLD_NUMERATOR * (
                bit_counts[start:stop, None] + bit_counts[None, :]
            )
            local_rows, cols = np.nonzero(close)
            rows = local_rows + start
            upper = cols > rows
            if np.any(upper):
                row_parts.append(rows[upper])
                col_parts.append(cols[upper])
                intersection_parts.append(intersections[local_rows[upper], cols[upper]])

    if not row_parts:
        empty = np.array([], dtype=np.int64)
        return empty, empty.copy(), empty.astype(np.int32)
    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    cols = np.concatenate(col_parts).astype(np.int64, copy=False)
    intersections = np.concatenate(intersection_parts).astype(np.int32, copy=False)
    if np.any(cols <= rows):
        raise ValueError("Exact pair search emitted a noncanonical unordered pair")
    return rows, cols, intersections


def classify_activity_pair_states(
    status_i: np.ndarray,
    status_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify pairs from two explicit compound-level activity states."""
    status_i = np.asarray(status_i, dtype=object)
    status_j = np.asarray(status_j, dtype=object)
    if status_i.shape != status_j.shape:
        raise ValueError("Pair-member activity-state arrays must have the same shape")
    allowed = ("active", "not_detected", "unassessed")
    if not np.isin(status_i, allowed).all() or not np.isin(status_j, allowed).all():
        raise ValueError(f"Activity states must be one of {allowed}")

    on_off = ((status_i == "active") & (status_j == "not_detected")) | (
        (status_i == "not_detected") & (status_j == "active")
    )
    both_active = (status_i == "active") & (status_j == "active")
    neither_active = (status_i == "not_detected") & (status_j == "not_detected")
    pair_state = np.full(status_i.shape, "unassessed", dtype=object)
    pair_state[neither_active] = "neither_active"
    pair_state[both_active] = "both_active"
    pair_state[on_off] = "one_active_one_not_detected"
    return pair_state, on_off


def summarize_activity_pair_states(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return an exhaustive activity-state partition with explicit denominators."""
    required = {"activity_pair_state", "scaffold_relationship"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"Exact close-pair table is missing columns: {sorted(missing)}")

    state_order = ("neither_active", "one_active_one_not_detected", "both_active", "unassessed")
    state_labels = {
        "neither_active": "Neither detected as active",
        "one_active_one_not_detected": "One active, one not detected",
        "both_active": "Both active",
        "unassessed": "At least one unassessed",
    }
    if not pairs["activity_pair_state"].isin(state_order).all():
        raise ValueError("Exact close-pair table contains an unknown activity state")

    all_pair_count = len(pairs)
    assessed_pair_count = int(pairs["activity_pair_state"].ne("unassessed").sum())
    summary_rows = []
    for state in state_order:
        group = pairs[pairs["activity_pair_state"].eq(state)]
        relationship_counts = group["scaffold_relationship"].value_counts()
        summary_rows.append(
            {
                "activity_pair_state": state,
                "display_label": state_labels[state],
                "n_pairs": int(len(group)),
                "pct_of_all_close_pairs": (float(100 * len(group) / all_pair_count) if all_pair_count else np.nan),
                "pct_of_activity_assessed_close_pairs": (
                    float(100 * len(group) / assessed_pair_count)
                    if state != "unassessed" and assessed_pair_count
                    else np.nan
                ),
                "same_scaffold_pairs": int(relationship_counts.get("same", 0)),
                "different_scaffold_pairs": int(relationship_counts.get("different", 0)),
                "missing_scaffold_pairs": int(relationship_counts.get("missing", 0)),
            }
        )
    pair_summary = pd.DataFrame(summary_rows)
    if int(pair_summary["n_pairs"].sum()) != all_pair_count:
        raise ValueError("Activity-state summary does not partition every exact close pair")
    return pair_summary, assessed_pair_count


def build_global_activity_cliff_analysis(
    profiles_path: Path,
    fp: np.ndarray,
    fp_jcp: np.ndarray,
    activity: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Find exact structural neighbors globally, then classify activity-state pairs."""
    required_activity = (
        "Metadata_JCP2022",
        "mean_normalized_average_precision",
        "corrected_p_value",
        "below_corrected_p",
    )
    required_annotations = {"jcp2022", "scaffold", "smiles", "name"}
    if missing := set(required_activity) - set(activity.columns):
        raise ValueError(f"Activity results are missing columns: {sorted(missing)}")
    if missing := required_annotations - set(annotations.columns):
        raise ValueError(f"Compound annotations are missing columns: {sorted(missing)}")

    activity_calls = activity[list(required_activity)].copy()
    activity_calls["Metadata_JCP2022"] = activity_calls["Metadata_JCP2022"].astype(str)
    if not activity_calls["Metadata_JCP2022"].is_unique:
        raise ValueError("Activity results contain duplicate compounds")
    if activity_calls["below_corrected_p"].isna().any():
        raise ValueError("Activity results contain missing corrected-p classifications")
    activity_calls = activity_calls.set_index("Metadata_JCP2022")

    annotations = annotations.copy()
    annotations["jcp2022"] = annotations["jcp2022"].astype(str)
    if not annotations["jcp2022"].is_unique:
        raise ValueError("Compound annotations contain duplicate compounds")
    annotations = annotations.set_index("jcp2022")

    adata = sc.read_h5ad(profiles_path)
    profile_jcp = adata.obs["JCP2022"].astype(str).to_numpy()
    fp_jcp = np.asarray(fp_jcp).astype(str)
    if len(profile_jcp) != len(set(profile_jcp)):
        raise ValueError("Morphology profiles contain duplicate compounds")
    if len(fp_jcp) != len(set(fp_jcp)):
        raise ValueError("Morgan fingerprints contain duplicate compounds")
    profile_idx = {jcp: i for i, jcp in enumerate(profile_jcp)}
    fp_idx = {jcp: i for i, jcp in enumerate(fp_jcp)}
    common_ids = np.array(sorted(set(profile_idx) & set(fp_idx)), dtype=str)
    X_morph = np.asarray(adata.X[[profile_idx[jcp] for jcp in common_ids]], dtype=np.float32)
    X_fp = fp[[fp_idx[jcp] for jcp in common_ids]]
    morph_norms = np.linalg.norm(X_morph, axis=1)
    fp_bit_counts = X_fp.sum(axis=1, dtype=np.int32)
    valid = np.isfinite(morph_norms) & (morph_norms > 0) & (fp_bit_counts > 0)
    jcp_ids = common_ids[valid]
    X_morph = X_morph[valid]
    X_fp = X_fp[valid]
    fp_bit_counts = fp_bit_counts[valid]
    X_morph /= morph_norms[valid, None]
    logger.info(f"Exact activity-cliff pair universe: {len(jcp_ids):,} compounds with valid profile + fingerprint")

    rows, cols, intersections = exact_tanimoto_pairs(X_fp)
    denominators = fp_bit_counts[rows] + fp_bit_counts[cols] - intersections
    tanimoto = intersections.astype(np.float64) / denominators
    morphology_cosine = np.einsum("ij,ij->i", X_morph[rows], X_morph[cols]).astype(np.float32)
    morphology_cosine = np.clip(morphology_cosine, -1.0, 1.0)

    aligned_activity = activity_calls.reindex(jcp_ids)
    assessed = aligned_activity["below_corrected_p"].notna().to_numpy()
    active = aligned_activity["below_corrected_p"].eq(True).to_numpy()
    statuses = np.full(len(jcp_ids), "unassessed", dtype=object)
    statuses[assessed & ~active] = "not_detected"
    statuses[active] = "active"
    status_i = statuses[rows]
    status_j = statuses[cols]
    activity_pair_state, on_off = classify_activity_pair_states(status_i, status_j)

    aligned_annotations = annotations.reindex(jcp_ids)
    annotation_values = {
        column: aligned_annotations[column].fillna("").astype(str).to_numpy()
        for column in ("scaffold", "smiles", "name")
    }
    scaffold_i = annotation_values["scaffold"][rows]
    scaffold_j = annotation_values["scaffold"][cols]
    scaffold_missing = (scaffold_i == "") | (scaffold_j == "")
    scaffold_same = ~scaffold_missing & (scaffold_i == scaffold_j)
    scaffold_relationship = np.where(
        scaffold_missing,
        "missing",
        np.where(scaffold_same, "same", "different"),
    )

    normalized_map = aligned_activity["mean_normalized_average_precision"].to_numpy(dtype=float)
    corrected_p = aligned_activity["corrected_p_value"].to_numpy(dtype=float)
    pairs = pd.DataFrame(
        {
            "jcp_i": jcp_ids[rows],
            "jcp_j": jcp_ids[cols],
            "fingerprint_bits_i": fp_bit_counts[rows],
            "fingerprint_bits_j": fp_bit_counts[cols],
            "fingerprint_intersection_bits": intersections,
            "morgan_tanimoto": tanimoto,
            "morphology_cosine": morphology_cosine,
            "activity_status_i": status_i,
            "activity_status_j": status_j,
            "mean_normalized_average_precision_i": normalized_map[rows],
            "mean_normalized_average_precision_j": normalized_map[cols],
            "corrected_p_value_i": corrected_p[rows],
            "corrected_p_value_j": corrected_p[cols],
            "activity_pair_state": activity_pair_state,
            "is_on_off_activity_cliff": on_off,
            "scaffold_i": scaffold_i,
            "scaffold_j": scaffold_j,
            "scaffold_relationship": scaffold_relationship,
            "smiles_i": annotation_values["smiles"][rows],
            "smiles_j": annotation_values["smiles"][cols],
            "name_i": annotation_values["name"][rows],
            "name_j": annotation_values["name"][cols],
        }
    ).sort_values(
        ["is_on_off_activity_cliff", "morgan_tanimoto", "jcp_i", "jcp_j"],
        ascending=[False, False, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    if pairs.duplicated(["jcp_i", "jcp_j"]).any() or not (pairs["jcp_i"] < pairs["jcp_j"]).all():
        raise ValueError("Exact activity-cliff table contains duplicate or noncanonical pairs")
    exact_threshold = (TANIMOTO_THRESHOLD_DENOMINATOR + TANIMOTO_THRESHOLD_NUMERATOR) * pairs[
        "fingerprint_intersection_bits"
    ] >= TANIMOTO_THRESHOLD_NUMERATOR * (pairs["fingerprint_bits_i"] + pairs["fingerprint_bits_j"])
    if not exact_threshold.all():
        raise ValueError("An exact activity-cliff pair falls below the declared Tanimoto threshold")
    if not pairs["is_on_off_activity_cliff"].equals(pairs["activity_pair_state"].eq("one_active_one_not_detected")):
        raise ValueError("On/off activity-cliff flags do not reproduce the declared activity-state rule")

    pair_summary, assessed_pair_count = summarize_activity_pair_states(pairs)

    same_scaffold_on_off = pairs[pairs["is_on_off_activity_cliff"] & pairs["scaffold_relationship"].eq("same")]
    scaffold_group_rows = []
    for scaffold, group in same_scaffold_on_off.groupby("scaffold_i", sort=True):
        active_ids = pd.concat(
            [
                group.loc[group["activity_status_i"].eq("active"), "jcp_i"],
                group.loc[group["activity_status_j"].eq("active"), "jcp_j"],
            ],
            ignore_index=True,
        )
        not_detected_ids = pd.concat(
            [
                group.loc[group["activity_status_i"].eq("not_detected"), "jcp_i"],
                group.loc[group["activity_status_j"].eq("not_detected"), "jcp_j"],
            ],
            ignore_index=True,
        )
        all_ids = sorted(set(active_ids) | set(not_detected_ids))
        representative = group.sort_values(
            ["morgan_tanimoto", "jcp_i", "jcp_j"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        scaffold_group_rows.append(
            {
                "scaffold": scaffold,
                "n_on_off_activity_cliff_pairs": int(len(group)),
                "n_unique_compounds": int(len(all_ids)),
                "n_active_compounds": int(active_ids.nunique()),
                "n_not_detected_compounds": int(not_detected_ids.nunique()),
                "max_morgan_tanimoto": float(group["morgan_tanimoto"].max()),
                "median_morgan_tanimoto": float(group["morgan_tanimoto"].median()),
                "representative_jcp_i": representative["jcp_i"],
                "representative_jcp_j": representative["jcp_j"],
                "jcp_ids": "|".join(all_ids),
            }
        )
    scaffold_group_columns = (
        "scaffold",
        "n_on_off_activity_cliff_pairs",
        "n_unique_compounds",
        "n_active_compounds",
        "n_not_detected_compounds",
        "max_morgan_tanimoto",
        "median_morgan_tanimoto",
        "representative_jcp_i",
        "representative_jcp_j",
        "jcp_ids",
    )
    scaffold_groups = pd.DataFrame(scaffold_group_rows, columns=scaffold_group_columns).sort_values(
        ["n_on_off_activity_cliff_pairs", "max_morgan_tanimoto", "scaffold"],
        ascending=[False, False, True],
        kind="mergesort",
        ignore_index=True,
    )

    on_off_pairs = pairs[pairs["is_on_off_activity_cliff"]]
    active_on_off_ids = pd.concat(
        [
            on_off_pairs.loc[on_off_pairs["activity_status_i"].eq("active"), "jcp_i"],
            on_off_pairs.loc[on_off_pairs["activity_status_j"].eq("active"), "jcp_j"],
        ],
        ignore_index=True,
    )
    not_detected_on_off_ids = pd.concat(
        [
            on_off_pairs.loc[on_off_pairs["activity_status_i"].eq("not_detected"), "jcp_i"],
            on_off_pairs.loc[on_off_pairs["activity_status_j"].eq("not_detected"), "jcp_j"],
        ],
        ignore_index=True,
    )
    active_member_is_i = on_off_pairs["activity_status_i"].eq("active").to_numpy()
    active_member_normalized_ap = np.where(
        active_member_is_i,
        on_off_pairs["mean_normalized_average_precision_i"],
        on_off_pairs["mean_normalized_average_precision_j"],
    )
    not_detected_member_normalized_ap = np.where(
        active_member_is_i,
        on_off_pairs["mean_normalized_average_precision_j"],
        on_off_pairs["mean_normalized_average_precision_i"],
    )
    normalized_ap_difference = active_member_normalized_ap - not_detected_member_normalized_ap
    if not (np.isfinite(active_member_normalized_ap).all() and np.isfinite(not_detected_member_normalized_ap).all()):
        raise ValueError("Candidate on/off pairs contain a missing activity score")

    def median_or_nan(values: np.ndarray) -> float:
        return float(np.median(values)) if len(values) else np.nan

    on_off_relationship_counts = on_off_pairs["scaffold_relationship"].value_counts()
    configuration = {
        "activity_cliff_pair_compound_scope": (
            "all compounds with valid source-7-excluded perturbation profiles and nonzero Morgan fingerprints"
        ),
        "activity_cliff_pair_search": "all unordered compound pairs; exhaustive, not k-nearest-neighbor",
        "activity_cliff_pair_structural_metric": "exact Morgan ECFP4 Tanimoto (binary Jaccard)",
        "activity_cliff_pair_structural_threshold": STRUCT_HIGH_THRESHOLD,
        "activity_cliff_pair_exact_threshold_test": (
            f"{TANIMOTO_THRESHOLD_DENOMINATOR + TANIMOTO_THRESHOLD_NUMERATOR} * "
            f"intersection_bits >= {TANIMOTO_THRESHOLD_NUMERATOR} * (bits_i + bits_j)"
        ),
        "activity_cliff_pair_on_off_definition": (
            f"exact Tanimoto at least {STRUCT_HIGH_THRESHOLD:g} and exactly one compound active; "
            "both compounds must have an activity assessment"
        ),
        "activity_cliff_pair_morphology_similarity_used_for_pair_classification": False,
        "activity_cliff_pair_scaffold_used_for_selection": False,
        "activity_cliff_pair_scaffold_role": (
            "post hoc annotation as same, different, or missing Bemis-Murcko scaffold"
        ),
        "activity_cliff_pair_unassessed_policy": (
            "retain exact close pairs in the canonical table; exclude pairs containing an unassessed compound "
            "from the activity-assessed plot denominator"
        ),
        "activity_cliff_pair_count_interpretation": (
            "descriptive unordered-pair counts; compounds can contribute to more than one pair"
        ),
    }
    numbers = {
        "activity_cliff_pair_compounds_analyzed": int(len(jcp_ids)),
        "activity_cliff_pair_compounds_assessed": int((statuses != "unassessed").sum()),
        "activity_cliff_pair_compounds_unassessed": int((statuses == "unassessed").sum()),
        "activity_cliff_pair_possible_unordered_pairs": int(len(jcp_ids) * (len(jcp_ids) - 1) // 2),
        "activity_cliff_exact_close_pairs": int(len(pairs)),
        "activity_cliff_exact_close_pairs_assessed": assessed_pair_count,
        "activity_cliff_exact_close_pairs_unassessed": int(pairs["activity_pair_state"].eq("unassessed").sum()),
        "activity_cliff_pair_neither_active": int(pairs["activity_pair_state"].eq("neither_active").sum()),
        "activity_cliff_pair_both_active": int(pairs["activity_pair_state"].eq("both_active").sum()),
        "activity_cliff_on_off_pairs": int(len(on_off_pairs)),
        "activity_cliff_on_off_pairs_pct_of_assessed": (
            float(100 * len(on_off_pairs) / assessed_pair_count) if assessed_pair_count else np.nan
        ),
        "activity_cliff_on_off_unique_compounds": int(len(set(on_off_pairs["jcp_i"]) | set(on_off_pairs["jcp_j"]))),
        "activity_cliff_on_off_active_compounds": int(active_on_off_ids.nunique()),
        "activity_cliff_on_off_not_detected_compounds": int(not_detected_on_off_ids.nunique()),
        "activity_cliff_on_off_median_active_normalized_ap": median_or_nan(active_member_normalized_ap),
        "activity_cliff_on_off_median_not_detected_normalized_ap": median_or_nan(not_detected_member_normalized_ap),
        "activity_cliff_on_off_median_within_pair_normalized_ap_difference": median_or_nan(normalized_ap_difference),
        "activity_cliff_on_off_active_member_higher_normalized_ap_pairs": int((normalized_ap_difference > 0).sum()),
        "activity_cliff_on_off_active_member_higher_normalized_ap_pct": (
            float(100 * (normalized_ap_difference > 0).mean()) if len(normalized_ap_difference) else np.nan
        ),
        "activity_cliff_on_off_same_scaffold_pairs": int(on_off_relationship_counts.get("same", 0)),
        "activity_cliff_on_off_different_scaffold_pairs": int(on_off_relationship_counts.get("different", 0)),
        "activity_cliff_on_off_missing_scaffold_pairs": int(on_off_relationship_counts.get("missing", 0)),
        "activity_cliff_on_off_scaffold_groups": int(len(scaffold_groups)),
    }
    return pairs, pair_summary, scaffold_groups, configuration, numbers


def load_active_profiles(
    profiles_path: Path,
    fp: np.ndarray,
    fp_jcp: np.ndarray,
    active_jcp: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fingerprints and morphology profiles on the active compound set.

    Returns L2-comparable matrices (raw; the caller normalizes) and the JCP2022 ids,
    filtered to compounds active AND carrying a valid fingerprint AND a valid profile.
    """
    adata = sc.read_h5ad(profiles_path)
    morph_jcp = adata.obs["JCP2022"].values
    common = sorted(set(fp_jcp) & set(morph_jcp) & active_jcp)
    logger.info(f"Active with fingerprint + profile: {len(common):,}")

    fp_idx = {k: i for i, k in enumerate(fp_jcp)}
    morph_idx = {k: i for i, k in enumerate(morph_jcp)}
    X_fp = fp[[fp_idx[k] for k in common]].astype(np.float32)
    X_morph = np.asarray(adata.X[[morph_idx[k] for k in common]]).astype(np.float32)
    jcp_ids = np.array(common)

    fp_norms = np.linalg.norm(X_fp, axis=1)
    morph_norms = np.linalg.norm(X_morph, axis=1)
    valid = (fp_norms > 0) & (morph_norms > 0) & ~np.isnan(morph_norms)
    logger.info(f"Valid active compounds: {int(valid.sum()):,}")
    return X_fp[valid], X_morph[valid], jcp_ids[valid]


def _neighbors_fn():
    """Return the k-NN function, GPU-accelerated when available."""
    if has_gpu():
        import rapids_singlecell as rsc

        logger.info("Using GPU-accelerated k-NN (rapids_singlecell)")
        return rsc.pp.neighbors
    logger.info("Using CPU k-NN (scanpy)")
    return sc.pp.neighbors


def knn_graph(X_norm: np.ndarray, k: int):
    """Cosine k-NN graph on already L2-normalized rows."""
    adata = sc.AnnData(X=X_norm)
    _neighbors_fn()(adata, n_neighbors=k, use_rep="X", metric="cosine")
    return adata


def undirected_pair_indices(graph) -> tuple[np.ndarray, np.ndarray]:
    """Return each unordered pair in the union of a directed sparse graph once."""
    graph = graph.tocsr()
    directed_rows = np.repeat(np.arange(graph.shape[0], dtype=np.int64), np.diff(graph.indptr))
    directed_cols = graph.indices.astype(np.int64, copy=False)
    off_diagonal = directed_rows != directed_cols
    lower = np.minimum(directed_rows[off_diagonal], directed_cols[off_diagonal])
    upper = np.maximum(directed_rows[off_diagonal], directed_cols[off_diagonal])
    pair_keys = np.unique(lower * graph.shape[0] + upper)
    return pair_keys // graph.shape[0], pair_keys % graph.shape[0]


def parse_target_sets(annotations: pd.DataFrame, column: str) -> dict[str, frozenset[str]]:
    """Map compound id to its pipe-delimited target set, skipping compounds without one."""
    out: dict[str, frozenset[str]] = {}
    for jcp, value in zip(annotations["jcp2022"].astype(str), annotations[column]):
        if isinstance(value, str) and value:
            targets = frozenset(t for t in value.split("|") if t)
            if targets:
                out[jcp] = targets
    return out


def build_target_divergence(
    pairs: pd.DataFrame,
    cliff_mask: np.ndarray,
    concordant_mask: np.ndarray,
    target_sets: dict[str, dict[str, frozenset[str]]],
) -> tuple[pd.DataFrame, dict]:
    """Compare annotated-target divergence between phenotypic cliffs and concordant pairs.

    ``pairs`` carries ``jcp_i``/``jcp_j``; the two masks select the cliff and concordant
    rows. See TARGET_DIVERGENCE_POLICY for the annotation and comparison rules.
    """
    jcp_i = pairs["jcp_i"].astype(str).to_numpy()
    jcp_j = pairs["jcp_j"].astype(str).to_numpy()
    empty: frozenset[str] = frozenset()
    rows = []
    numbers: dict[str, dict] = {}
    for contract, sources in TARGET_DIVERGENCE_CONTRACTS.items():
        both = np.zeros(len(pairs), dtype=bool)
        shared = np.zeros(len(pairs), dtype=bool)
        for source in sources:
            lookup = target_sets[source]
            sets_i = [lookup.get(j, empty) for j in jcp_i]
            sets_j = [lookup.get(j, empty) for j in jcp_j]
            both |= np.fromiter((bool(a) and bool(b) for a, b in zip(sets_i, sets_j)), bool, len(pairs))
            shared |= np.fromiter((not a.isdisjoint(b) for a, b in zip(sets_i, sets_j)), bool, len(pairs))
        table = {}
        for group, mask in (("cliff", cliff_mask), ("concordant", concordant_mask)):
            annotated = int((both & mask).sum())
            divergent = int((both & mask & ~shared).sum())
            table[group] = [divergent, annotated - divergent]
            rows.append(
                {
                    "contract": contract,
                    "sources": " + ".join(TARGET_DIVERGENCE_SOURCES[s] for s in sources),
                    "group": group,
                    "pairs": int(mask.sum()),
                    "annotated_pairs": annotated,
                    "divergent_pairs": divergent,
                    "shared_target_pairs": annotated - divergent,
                    "pct_divergent": (100.0 * divergent / annotated) if annotated else float("nan"),
                }
            )
        odds_ratio, p_value = fisher_exact([table["cliff"], table["concordant"]], alternative="two-sided")
        odds_ratio = float(odds_ratio) if np.isfinite(odds_ratio) else None
        for row in rows[-2:]:
            row["odds_ratio"] = odds_ratio
            row["fisher_p"] = float(p_value)
        numbers[contract] = {
            "sources": [TARGET_DIVERGENCE_SOURCES[s] for s in sources],
            "cliff_annotated_pairs": table["cliff"][0] + table["cliff"][1],
            "cliff_divergent_pairs": table["cliff"][0],
            "cliff_pct_divergent": rows[-2]["pct_divergent"],
            "concordant_annotated_pairs": table["concordant"][0] + table["concordant"][1],
            "concordant_divergent_pairs": table["concordant"][0],
            "concordant_pct_divergent": rows[-1]["pct_divergent"],
            "odds_ratio": odds_ratio,
            "fisher_p": float(p_value),
        }
    return pd.DataFrame(rows), numbers


def extract_pairs(
    adata_struct,
    X_fp_norm: np.ndarray,
    X_morph_norm: np.ndarray,
    jcp_ids: np.ndarray,
) -> pd.DataFrame:
    """Undirected union of structural k-NN pairs with both cosine similarities."""
    rows, cols = undirected_pair_indices(adata_struct.obsp["distances"])
    struct_sim = np.einsum("ij,ij->i", X_fp_norm[rows], X_fp_norm[cols]).astype(np.float32)
    morph_sim = np.einsum("ij,ij->i", X_morph_norm[rows], X_morph_norm[cols]).astype(np.float32)
    return pd.DataFrame(
        {
            "jcp_i": jcp_ids[rows].astype(str),
            "jcp_j": jcp_ids[cols].astype(str),
            "struct_sim": struct_sim,
            "morph_sim": morph_sim,
        }
    )


def jaccard_overlap(adata_struct, adata_morph) -> np.ndarray:
    """Per-compound Jaccard overlap between structural and morphological k-NN sets."""
    d1 = adata_struct.obsp["distances"]
    d2 = adata_morph.obsp["distances"]
    out = np.empty(adata_struct.n_obs, dtype=np.float64)
    for i in range(adata_struct.n_obs):
        n1 = set(d1[i].nonzero()[1])
        n2 = set(d2[i].nonzero()[1])
        out[i] = len(n1 & n2) / len(n1 | n2) if (n1 | n2) else 0.0
    return out


@app.command()
def main(
    morgan_fp: Path = typer.Option(
        DEFAULT_MORGAN_FP,
        "--morgan-fp",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    perturbation_profiles: Path = typer.Option(
        DEFAULT_PERTURBATION_PROFILES,
        "--perturbation-profiles",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    metadata_db: Path = typer.Option(
        DEFAULT_METADATA_DB,
        "--metadata-db",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    copairs_db: Path = typer.Option(
        DEFAULT_COPAIRS_DB,
        "--copairs-db",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path | None = typer.Option(None, "--output", file_okay=False),
    dataset: str = typer.Option(DEFAULT_DATASET, "--dataset"),
    activity_preprocessing: str = typer.Option(
        DEFAULT_ACTIVITY_PREPROCESSING,
        "--activity-preprocessing",
    ),
    activity_filter: str = typer.Option(DEFAULT_ACTIVITY_FILTER, "--activity-filter"),
) -> None:
    """Write the deterministic cached analysis consumed by the direct figure script."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    fp, fp_jcp = load_fingerprints(morgan_fp)

    copairs_con = duckdb.connect(str(copairs_db), read_only=True)
    try:
        activity = query_activity_results(
            copairs_con,
            dataset=dataset,
            preprocessing=activity_preprocessing,
            filter_name=activity_filter,
            activity_params=DEFAULT_ACTIVITY_PARAMS,
        )
    finally:
        copairs_con.close()
    required_activity = [
        "Metadata_JCP2022",
        "mean_normalized_average_precision",
        "corrected_p_value",
        "below_corrected_p",
    ]
    if missing := set(required_activity) - set(activity.columns):
        raise ValueError(f"Activity results are missing columns: {sorted(missing)}")
    activity = activity[required_activity].copy()
    activity["Metadata_JCP2022"] = activity["Metadata_JCP2022"].astype(str)
    if not activity["Metadata_JCP2022"].is_unique:
        raise ValueError("Activity results contain duplicate compounds")
    activity = activity.sort_values("Metadata_JCP2022", kind="mergesort", ignore_index=True)
    active_jcp = set(activity.loc[activity["below_corrected_p"], "Metadata_JCP2022"])
    n_active_total = len(active_jcp)
    logger.info(f"Phenotypically active (copairs FDR): {n_active_total:,}")

    metadata_con = duckdb.connect(str(metadata_db), read_only=True)
    try:
        moa = metadata_con.execute(
            "SELECT Metadata_JCP2022 AS jcp2022, Metadata_repurposing_moa AS moa "
            "FROM compound_metadata WHERE Metadata_repurposing_moa IS NOT NULL "
            "ORDER BY Metadata_JCP2022"
        ).df()
        compound_annotations = metadata_con.execute(
            "SELECT Metadata_JCP2022 AS jcp2022, Metadata_MurckoScaffold AS scaffold, "
            "Metadata_SMILES AS smiles, Metadata_repurposing_name AS name "
            "FROM compound_metadata ORDER BY Metadata_JCP2022"
        ).df()
        target_annotations = metadata_con.execute(
            "SELECT Metadata_JCP2022 AS jcp2022, Metadata_repurposing_target AS hub_targets, "
            "Metadata_Uniprot_target AS chembl_targets "
            "FROM compound_metadata ORDER BY Metadata_JCP2022"
        ).df()
    finally:
        metadata_con.close()
    target_sets = {name: parse_target_sets(target_annotations, name) for name in TARGET_DIVERGENCE_SOURCES}

    (
        activity_cliff_pairs,
        activity_cliff_pair_summary,
        activity_cliff_scaffold_groups,
        activity_cliff_pair_configuration,
        activity_cliff_pair_numbers,
    ) = build_global_activity_cliff_analysis(
        perturbation_profiles,
        fp,
        fp_jcp,
        activity,
        compound_annotations,
    )

    X_fp, X_morph, jcp_ids = load_active_profiles(
        perturbation_profiles,
        fp,
        fp_jcp,
        active_jcp,
    )
    n_compounds = len(jcp_ids)
    X_fp_norm = X_fp / np.linalg.norm(X_fp, axis=1, keepdims=True)
    X_morph_norm = X_morph / np.linalg.norm(X_morph, axis=1, keepdims=True)
    all_pair_similarity_histogram = build_all_pair_similarity_histogram(X_fp, X_morph)

    struct_pairs_graph = knn_graph(X_fp_norm, K_PAIRS)
    identified_pairs = extract_pairs(
        struct_pairs_graph,
        X_fp_norm,
        X_morph_norm,
        jcp_ids,
    ).sort_values(["jcp_i", "jcp_j"], kind="mergesort", ignore_index=True)
    pairs = identified_pairs[["struct_sim", "morph_sim"]].copy()
    n_pairs = len(pairs)
    spearman_r, spearman_p = spearmanr(
        pairs["struct_sim"],
        pairs["morph_sim"],
    )
    correlation_null = permutation_null_correlation(
        pairs["struct_sim"].to_numpy(),
        pairs["morph_sim"].to_numpy(),
        n_permutations=200,
        subsample_size=None,
    )

    high_struct = pairs["struct_sim"] >= STRUCT_HIGH_THRESHOLD
    low_struct = ~high_struct
    high_morph = pairs["morph_sim"] >= (1 - MORPH_LOW_THRESHOLD)
    low_morph = pairs["morph_sim"] < MORPH_LOW_THRESHOLD
    quadrants = {
        "cliffs": int((high_struct & low_morph).sum()),
        "concordant": int((high_struct & high_morph).sum()),
        "reverse_cliffs": int((low_struct & high_morph).sum()),
        "neutral": int((low_struct & low_morph).sum()),
    }
    quadrants["intermediate"] = int(n_pairs - sum(quadrants.values()))
    n_struct_similar = int(high_struct.sum())
    phenotypic_cliffs = identified_pairs.loc[
        high_struct & low_morph,
        ["jcp_i", "jcp_j", "struct_sim", "morph_sim"],
    ].copy()
    phenotypic_cliffs["cliff_score"] = phenotypic_cliffs["struct_sim"] - phenotypic_cliffs["morph_sim"]
    phenotypic_cliffs = phenotypic_cliffs.sort_values(
        ["cliff_score", "jcp_i", "jcp_j"],
        ascending=[False, True, True],
        kind="mergesort",
        ignore_index=True,
    )
    if len(phenotypic_cliffs) != quadrants["cliffs"]:
        raise ValueError("The itemized phenotypic-cliff table differs from the reported cliff count")
    similar_pairs = identified_pairs.loc[high_struct, ["jcp_i", "jcp_j"]].reset_index(drop=True)
    target_divergence, target_divergence_numbers = build_target_divergence(
        similar_pairs,
        low_morph[high_struct].to_numpy(),
        high_morph[high_struct].to_numpy(),
        target_sets,
    )
    for contract, stats in target_divergence_numbers.items():
        logger.info(
            f"Target divergence [{contract}]: cliffs {stats['cliff_divergent_pairs']}/"
            f"{stats['cliff_annotated_pairs']} vs concordant {stats['concordant_divergent_pairs']}/"
            f"{stats['concordant_annotated_pairs']}, OR={stats['odds_ratio']}, p={stats['fisher_p']:.2g}"
        )

    struct_overlap_graph = knn_graph(X_fp_norm, K_OVERLAP)
    morph_overlap_graph = knn_graph(X_morph_norm, K_OVERLAP)
    overlap = jaccard_overlap(struct_overlap_graph, morph_overlap_graph)
    expected_random = K_OVERLAP / (2 * n_compounds)
    overlap_df = (
        pd.DataFrame({"jcp2022": jcp_ids.astype(str), "neighbor_overlap": overlap})
        .merge(moa, on="jcp2022", how="left")
        .sort_values("jcp2022", kind="mergesort", ignore_index=True)
    )
    moa_stats = (
        overlap_df.dropna(subset=["moa"])
        .groupby("moa")
        .agg(
            mean_overlap=("neighbor_overlap", "mean"),
            count=("neighbor_overlap", "size"),
        )
        .reset_index()
    )
    moa_stats["enrichment"] = moa_stats["mean_overlap"] / expected_random
    moa_stats = (
        moa_stats[moa_stats["count"] >= MOA_MIN_COMPOUNDS]
        .sort_values(
            ["enrichment", "moa"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(MOA_TOP_N)
        .reset_index(drop=True)
    )

    paths = {
        "pairs": output_dir / "pairs.parquet",
        "all_pair_similarity_histogram": (output_dir / "all_pair_similarity_histogram.parquet"),
        "phenotypic_cliffs": output_dir / "phenotypic_cliffs.csv",
        "phenotypic_cliff_target_divergence": (output_dir / "phenotypic_cliff_target_divergence.csv"),
        "activity_cliff_pairs": output_dir / "activity_cliff_pairs.parquet",
        "activity_cliff_pair_summary": (output_dir / "activity_cliff_pair_summary.csv"),
        "activity_cliff_scaffold_groups": (output_dir / "activity_cliff_scaffold_groups.csv"),
        "phenotypic_activity": output_dir / "phenotypic_activity.csv",
        "neighbor_overlap": output_dir / "neighbor_overlap.csv",
        "moa_coherence": output_dir / "moa_coherence.csv",
    }
    pairs.to_parquet(paths["pairs"], index=False, compression="zstd")
    all_pair_similarity_histogram.to_parquet(
        paths["all_pair_similarity_histogram"],
        index=False,
        compression="zstd",
    )
    phenotypic_cliffs.to_csv(
        paths["phenotypic_cliffs"],
        index=False,
        float_format="%.7g",
    )
    target_divergence.to_csv(
        paths["phenotypic_cliff_target_divergence"],
        index=False,
        float_format="%.17g",
    )
    activity_cliff_pairs.to_parquet(
        paths["activity_cliff_pairs"],
        index=False,
        compression="zstd",
    )
    activity_cliff_pair_summary.to_csv(
        paths["activity_cliff_pair_summary"],
        index=False,
        float_format="%.17g",
    )
    activity_cliff_scaffold_groups.to_csv(
        paths["activity_cliff_scaffold_groups"],
        index=False,
        float_format="%.17g",
    )
    activity.rename(columns={"Metadata_JCP2022": "jcp2022"}).to_csv(
        paths["phenotypic_activity"],
        index=False,
        float_format="%.17g",
    )
    overlap_df.to_csv(
        paths["neighbor_overlap"],
        index=False,
        float_format="%.17g",
    )
    moa_stats.to_csv(
        paths["moa_coherence"],
        index=False,
        float_format="%.17g",
    )

    parameters = {
        "dataset": dataset,
        "activity_preprocessing": activity_preprocessing,
        "activity_filter": activity_filter,
        "activity_params": DEFAULT_ACTIVITY_PARAMS,
        "activity_corrected_p_threshold": 0.10,
        "structural_representation": "morgan_ecfp4_2048",
        "k_pairs": K_PAIRS,
        "k_pair_graph": (
            "undirected union of all per-compound directed k-nearest-neighbor edges; one row per unordered pair"
        ),
        "k_overlap": K_OVERLAP,
        "all_pair_structure_bins": ALL_PAIR_STRUCTURE_BINS,
        "all_pair_morphology_bins": ALL_PAIR_MORPHOLOGY_BINS,
        "struct_high_threshold": STRUCT_HIGH_THRESHOLD,
        "morph_low_threshold": MORPH_LOW_THRESHOLD,
        "moa_min_compounds": MOA_MIN_COMPOUNDS,
        "moa_top_n": MOA_TOP_N,
        "target_divergence_policy": TARGET_DIVERGENCE_POLICY,
        **activity_cliff_pair_configuration,
    }
    numbers = {
        "parameters": parameters,
        "phenotypically_active_compounds": int(n_active_total),
        "active_compounds_analyzed": int(n_compounds),
        "all_possible_active_pairs": int(n_compounds * (n_compounds - 1) // 2),
        "n_pairs": int(n_pairs),
        "n_structurally_similar_pairs": n_struct_similar,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "spearman_permutation_z": float(correlation_null["z_score"]),
        "spearman_null_mean": float(correlation_null["null_mean"]),
        "spearman_null_std": float(correlation_null["null_std"]),
        "spearman_permutations": int(correlation_null["n_permutations"]),
        "cliff_pairs": quadrants["cliffs"],
        "cliff_rate": float(quadrants["cliffs"] / n_pairs),
        "cliff_rate_of_structurally_similar": float(quadrants["cliffs"] / n_struct_similar),
        "quadrants": quadrants,
        "target_divergence": target_divergence_numbers,
        "mean_jaccard_overlap": float(np.mean(overlap)),
        "median_jaccard_overlap": float(np.median(overlap)),
        "expected_random_overlap": float(expected_random),
        "ratio_vs_random": float(np.mean(overlap) / expected_random),
        "pct_nonzero_overlap": float(100 * np.mean(overlap > 0)),
        "moa_classes_reported": int(len(moa_stats)),
        **activity_cliff_pair_numbers,
    }
    file_captions = {
        "pairs": "Active-compound structural k-nearest-neighbor similarities",
        "all_pair_similarity_histogram": ("Exhaustive active-compound structure-morphology similarity histogram"),
        "phenotypic_cliffs": "Itemized active-pair morphology cliffs",
        "phenotypic_cliff_target_divergence": (
            "Annotated-target divergence of phenotypic cliffs versus concordant pairs"
        ),
        "activity_cliff_pairs": ("Global exact-Tanimoto pairs with phenotypic activity states"),
        "activity_cliff_pair_summary": "Exact-pair activity-state counts",
        "activity_cliff_scaffold_groups": ("Post hoc same-scaffold on-off activity-cliff groups"),
        "phenotypic_activity": "Canonical compound phenotypic activity calls",
        "neighbor_overlap": "Per-compound structure-morphology neighbor overlap",
        "moa_coherence": "Mechanism-of-action neighbor-overlap summary",
    }
    write_summary(
        output_dir / "summary.json",
        description=(
            "Cached pair analysis for Supplementary Figure STRUCTURE. It "
            "contains the exhaustive active-compound similarity histogram, "
            "the symmetrized structural k-nearest-neighbor estimands, and the "
            "global exact-Tanimoto phenotypic activity-state screen."
        ),
        numbers=numbers,
        files=[
            {
                "path": path.name,
                "caption": file_captions[name],
            }
            for name, path in paths.items()
        ],
    )
    logger.success(
        f"Wrote STRUCTURE pair cache to {output_dir}: "
        f"{n_compounds:,} active compounds, "
        f"{quadrants['cliffs']:,} active-pair morphology cliffs, "
        f"{activity_cliff_pair_numbers['activity_cliff_on_off_pairs']:,} "
        "exact-Tanimoto on/off pairs"
    )


if __name__ == "__main__":
    app()
