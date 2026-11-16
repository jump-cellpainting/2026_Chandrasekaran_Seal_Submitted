"""Build the phenotypic-consistency figure directly from the canonical JUMP analysis inputs.

This script owns the data selection, statistics, plots, tables, legend,
and standard summary envelope. It intentionally has no intermediate figure
substrate or catalog-frontend dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import anndata as ad
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from jump_production.queries import query_activity_results, query_consistency_results
from jump_production.summaries import write_legend, write_summary
from jump_production.visualization import attach_caption, save_figure

DEFAULT_DATASET = "compound_no_source7"
DEFAULT_ACTIVITY_PREPROCESSING = "activity_no_target2"
DEFAULT_ACTIVITY_FILTER = "all_sources"
DEFAULT_ACTIVITY_PARAMS = "default"
DEFAULT_METRIC = "cosine"
DEFAULT_WELL_UMAP = Path("data/interim/anndata/compound_no_source7_cosine_all_umap.h5ad")
DEFAULT_PERTURBATION_UMAP = Path("data/interim/anndata/compound_no_source7_perturbation_cosine_all_umap.h5ad")
DEFAULT_PERTURBATION_PROFILES = Path("data/interim/anndata/compound_no_source7_perturbation.h5ad")
DEFAULT_METADATA_DB = Path("data/interim/jump_metadata_augmented.duckdb")
DEFAULT_COPAIRS_DB = Path("data/processed/copairs_results.duckdb")
FIGURE_ID = "figure_phenotypic_consistency"
DEFAULT_OUTPUT_DIR = Path("data/processed/figures") / FIGURE_ID

ACTIVITY_FDR_THRESHOLD = 0.10
CONSISTENCY_FDR_THRESHOLD = 0.05
TOP_N = 20
MOA_RETRIEVAL_MAX_K = 100
MOA_RETRIEVAL_PERMUTATIONS = 10_000
MOA_RETRIEVAL_SEED = 20_260_714
WELL_PLOT_ORDER_SEED = 0

SELECTED_MOA_FDR_THRESHOLD = 0.05
SELECTED_MOA_MIN_ACTIVE_COMPOUNDS = 4
SELECTED_MOA_MIN_NORMALIZED_AVERAGE_PRECISION = 0.23
CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_KS = (1, 10)
CROSS_SCAFFOLD_RETRIEVAL_MAIN_TEXT_ANNOTATION_KS = (1,)
GLOBAL_MOA_MAP_PROXIMITY_SCORE_UPPER_QUANTILE = 0.98
GLOBAL_MOA_MAP_CMAP = "cividis_r"
GLOBAL_MOA_MAP_UMAP_LABEL_FONTSIZE = 18
GLOBAL_MOA_MAP_COUNT_FONTSIZE = 10
GLOBAL_MOA_MAP_COLORBAR_TICKS = (0.0, 1.0, 2.0)
GLOBAL_MOA_MAP_COLORBAR_SHRINK = 0.56
GLOBAL_MOA_MAP_COLORBAR_TICK_FONTSIZE = 12
GLOBAL_MOA_MAP_COLORBAR_LABEL_FONTSIZE = 9.5
GLOBAL_MOA_MAP_COLORBAR_LABEL_OFFSET = -0.12
COMBINED_RETRIEVAL_WSPACE = 0.2
CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_OFFSETS = {
    1: (12, -41),
    10: (16, -30),
}
COMBINED_RETRIEVAL_ANNOTATION_OFFSETS = {
    1: (26, -42),
    10: (-154, -30),
}
COMBINED_RETRIEVAL_X_LABEL = "Number of phenotypic neighbors\nexamined (k)"
CROSS_SCAFFOLD_RETRIEVAL_X_LABEL_FONTSIZE = 14
CROSS_SCAFFOLD_RETRIEVAL_Y_LABEL_FONTSIZE = 14
CROSS_SCAFFOLD_RETRIEVAL_Y_LABEL_PAD = 10
CROSS_SCAFFOLD_RETRIEVAL_TICK_FONTSIZE = 13
CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_FONTSIZE = 12
CROSS_SCAFFOLD_RETRIEVAL_LEGEND_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 12
CONTEXT_TITLE_FONTSIZE = 11.5
CONTEXT_COLORBAR_LABEL_FONTSIZE = 9
CONTEXT_COLORBAR_TICK_FONTSIZE = 8.5
COMBINED_TITLE_FONTSIZE = 13
VIEW_FIGSIZE_INCHES = {
    "profile_context_umaps": (8.0, 15.3),
    "cross_scaffold_moa_retrieval": (5.6, 5.4),
    "ranked_targets_and_moas": (14.5, 10.0),
    "global_moa_map": (6.0, 5.4),
    "selected_moa_neighborhoods": (10.2, 6.8),
    "combined_global_moa_retrieval": (11.2, 5.4),
}
VIEW_ORDER = (
    "profile_context_umaps",
    "cross_scaffold_moa_retrieval",
    "ranked_targets_and_moas",
    "global_moa_map",
    "selected_moa_neighborhoods",
    "combined_global_moa_retrieval",
)
VIEW_STEMS = {
    "profile_context_umaps": "activity_context",
    "cross_scaffold_moa_retrieval": "cross_scaffold_retrieval",
    "ranked_targets_and_moas": "ranked_targets_moas",
    "global_moa_map": "global_moa_map",
    "selected_moa_neighborhoods": "moa_neighborhoods",
    "combined_global_moa_retrieval": "map_retrieval",
}
# One legend section per rendered file, titled here and bodied by the caption the view carries.
SECTION_TITLES = {
    "profile_context_umaps": "Perturbation activity, cell count, and well-level source",
    "cross_scaffold_moa_retrieval": "Cross-scaffold MOA retrieval",
    "ranked_targets_and_moas": "Top-ranked targets and mechanisms",
    "global_moa_map": "Global map of tested compounds",
    "selected_moa_neighborhoods": "Selected mechanism neighborhoods",
    "combined_global_moa_retrieval": "Global MOA map with cross-scaffold retrieval",
}
FIGURE_OUTPUT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
PUBLICATION_TYPOGRAPHY = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

CURATED_MOAS = (
    "HSP inhibitor",
    "HMGCR inhibitor",
    "NAMPT inhibitor",
    "kinesin-like spindle protein inhibitor",
    "MEK inhibitor",
    "mTOR inhibitor",
    "tubulin polymerization inhibitor",
    "tricyclic antidepressant",
    "bromodomain inhibitor",
    "HDAC inhibitor",
    "histone lysine methyltransferase inhibitor",
)
MOA_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#CCBB44",
    "#000000",
    "#332288",
    "#AA4499",
    "#117733",
)
MOA_PALETTE = dict(zip(CURATED_MOAS, MOA_COLORS, strict=True))
SOURCE_COLORS = (
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
)
SIGNIFICANT_COLOR = "#0072B2"
NONSIGNIFICANT_COLOR = "#B8B8B8"
ACTIVE_COLOR = "#D55E00"
INACTIVE_COLOR = "#56B4E9"
NO_RESULT_COLOR = "#8C8C8C"
CONTEXT_CELL_COUNT_CMAP = "viridis"
CONTEXT_CELL_COUNT_QUANTILES = (0.01, 0.99)
CONTEXT_CELL_COUNT_POINT_SIZE = 1.2
CONTEXT_CELL_COUNT_ALPHA = 0.76
WELL_SOURCE_VIEWPORT_IQR_K = 3.0
WELL_SOURCE_VIEWPORT_NOTE_FONTSIZE = 8.5
RANKING_Y_LABEL_FONTSIZE = 10.0
RANKING_X_LABEL_FONTSIZE = 10.0
RANKING_X_TICK_FONTSIZE = 9.0
RANKING_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE
RANKING_LEGEND_FONTSIZE = 9.0
RANKING_LEGEND_TITLE_FONTSIZE = 9.5
RANKING_LABEL_FORMAT = "{label} (n = {count})"
RANKING_TITLE_X_BY_SOURCE = {
    "Chemical Probes targets": 0.5,
    "Drug Repurposing Hub MOAs": 0.38,
}
RANKING_SIZE_LEGEND_LOCATION = "lower right"
RANKING_SIZE_LEGEND_COLUMNS = 1
RANKING_SIGNIFICANCE_LEGEND_LOCATION = "outside upper center"
RANKING_SIGNIFICANCE_LEGEND_COLUMNS = 2
RANKING_LABELS = {
    "inosine monophosphate dehydrogenase inhibitor": "IMPDH inhibitor",
    "microtubule stabilizing agent": "microtubule stabilizer",
    "kinesin-like spindle protein inhibitor": "KSP inhibitor",
    "hypoxia inducible factor inhibitor": "HIF inhibitor",
    "phosphofructokinase inhibitor": "PFK inhibitor",
    "matrix metalloprotease inhibitor": "MMP inhibitor",
    "histone lysine methyltransferase inhibitor": "KMT inhibitor",
    "histone lysine demethylase inhibitor": "KDM inhibitor",
    "TGF beta receptor inhibitor": "TGF-beta receptor inhibitor",
}

app = typer.Typer(add_completion=False)


def load_embedding(path: Path, obs_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Load observation metadata and UMAP coordinates without profile features."""
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs[obs_columns].copy()
        coordinates = np.asarray(adata.obsm["X_umap"]).copy()
        random_state = int(adata.uns["umap"]["params"]["random_state"])
    finally:
        adata.file.close()
    if len(obs) != len(coordinates):
        raise AssertionError(f"Observation/coordinate mismatch in {path}")
    return obs, coordinates, random_state


def consistency_preprocessing(activity_preprocessing: str) -> str:
    """Return the paired consistency preprocessing configuration."""
    if not activity_preprocessing.startswith("activity_"):
        raise ValueError("Activity preprocessing must start with 'activity_'")
    return f"consistency_{activity_preprocessing.removeprefix('activity_')}"


def exact_moa_memberships(metadata: pd.DataFrame, embedded_ids: pd.Series) -> pd.DataFrame:
    """Expand exact pipe-delimited MOA memberships for embedded compounds."""
    metadata = metadata.rename(columns={"Metadata_JCP2022": "jcp2022"})
    metadata = metadata[metadata["jcp2022"].isin(embedded_ids)]
    memberships = metadata.assign(
        mechanism_of_action=metadata["Metadata_repurposing_moa"].fillna("").str.split("|")
    ).explode("mechanism_of_action")
    memberships = memberships[memberships["mechanism_of_action"] != ""][
        ["jcp2022", "mechanism_of_action"]
    ].drop_duplicates()
    memberships["jcp2022"] = memberships["jcp2022"].astype(str)
    return memberships.sort_values(["mechanism_of_action", "jcp2022"]).reset_index(drop=True)


def attach_moa_activity(memberships: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical phenotypic-activity calls to MOA memberships."""
    activity = activity.rename(columns={"Metadata_JCP2022": "jcp2022"})[["jcp2022", "corrected_p_value"]].copy()
    activity["jcp2022"] = activity["jcp2022"].astype(str)
    memberships = memberships.merge(activity, on="jcp2022", how="left", validate="many_to_one")
    memberships["active"] = memberships["corrected_p_value"] < ACTIVITY_FDR_THRESHOLD
    return memberships[["jcp2022", "mechanism_of_action", "active"]].sort_values(["mechanism_of_action", "jcp2022"])


def prepare_moa_consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Prepare the complete ranked MOA consistency evidence."""
    consistency = results[
        [
            "group_value",
            "n_perturbations",
            "mean_normalized_average_precision",
            "corrected_p_value",
        ]
    ].rename(columns={"group_value": "mechanism_of_action"})
    if consistency["mechanism_of_action"].duplicated().any():
        raise AssertionError("MOA consistency results contain duplicate mechanism classes")
    consistency = consistency.sort_values(
        ["mean_normalized_average_precision", "mechanism_of_action"],
        ascending=[False, True],
    ).reset_index(drop=True)
    consistency.insert(0, "rank", np.arange(1, len(consistency) + 1, dtype=int))
    return consistency


def load_selected_profiles(path: Path, selected_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load selected perturbation profiles in source-file order."""
    adata = ad.read_h5ad(path, backed="r")
    try:
        profile_ids = adata.obs["JCP2022"].astype(str).to_numpy()
        selected_positions = np.flatnonzero(np.isin(profile_ids, sorted(selected_ids)))
        ordered_ids = profile_ids[selected_positions]
        if set(ordered_ids) != selected_ids:
            missing = sorted(selected_ids - set(ordered_ids))
            raise AssertionError(f"Selected compounds missing perturbation profiles: {missing[:5]}")
        matrix = np.asarray(adata.X[selected_positions, :], dtype=np.float32)
    finally:
        adata.file.close()
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise AssertionError("Perturbation profile matrix must contain feature columns")
    return ordered_ids, matrix


def chance_adjusted_proximity_score(rank: int | None, population: int, relevant: int) -> float:
    """Score an observed first-hit rank against its class-size-aware chance probability."""
    if rank is None or relevant == 0:
        return 0.0
    if relevant >= population - rank + 1:
        return 0.0
    draws = np.arange(rank, dtype=float)
    log_no_hit = np.log1p(-relevant / (population - draws)).sum()
    chance_probability = float(-np.expm1(log_no_hit))
    return float(-np.log10(max(chance_probability, 1e-12)))


def prepare_global_moa_retrieval(
    memberships: pd.DataFrame,
    consistency: pd.DataFrame,
    annotations: pd.DataFrame,
    perturbations: pd.DataFrame,
    profile_path: Path,
    permutations: int,
    seed: int,
    max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Quantify atlas-wide same-MOA retrieval after excluding exact shared scaffolds."""
    if permutations < 1:
        raise ValueError("MOA retrieval requires at least one label permutation")
    if max_k < 1:
        raise ValueError("MOA retrieval max_k must be positive")

    tested_classes = set(consistency["mechanism_of_action"])
    active_memberships = memberships[
        memberships["active"] & memberships["mechanism_of_action"].isin(tested_classes)
    ].copy()
    label_sets = active_memberships.groupby("jcp2022", sort=True)["mechanism_of_action"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    selected_ids, matrix = load_selected_profiles(profile_path, set(label_sets.index))

    label_names = sorted(tested_classes)
    label_index = {label: index for index, label in enumerate(label_names)}
    incidence = np.zeros((len(selected_ids), len(label_names)), dtype=np.int16)
    for row_index, jcp2022 in enumerate(selected_ids):
        for label in label_sets.loc[jcp2022]:
            incidence[row_index, label_index[label]] = 1

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise AssertionError("MOA retrieval found zero-norm perturbation profiles")
    normalized = matrix / norms[:, None]
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    scaffold_frame = pd.DataFrame({"jcp2022": selected_ids}).merge(
        annotations[["Metadata_JCP2022", "Metadata_MurckoScaffold"]].rename(
            columns={
                "Metadata_JCP2022": "jcp2022",
                "Metadata_MurckoScaffold": "scaffold",
            }
        ),
        on="jcp2022",
        how="left",
        validate="one_to_one",
    )
    scaffold_values = scaffold_frame["scaffold"].fillna("").astype(str).to_numpy()
    valid_scaffold = scaffold_values != ""
    same_scaffold = (
        (scaffold_values[:, None] == scaffold_values[None, :]) & valid_scaffold[:, None] & valid_scaffold[None, :]
    )
    np.fill_diagonal(same_scaffold, True)
    similarity[same_scaffold] = -np.inf

    eligible_neighbor_counts = (~same_scaffold).sum(axis=1)
    effective_max_k = min(max_k, int(eligible_neighbor_counts.min()))
    neighbor_order = np.argsort(-similarity, axis=1, kind="stable")[:, :effective_max_k]
    rows = np.arange(len(selected_ids))[:, None]
    if same_scaffold[rows, neighbor_order].any():
        raise AssertionError("Same-scaffold compounds entered the retrieval neighborhood")

    shared_label = (incidence @ incidence.T) > 0
    np.fill_diagonal(shared_label, False)
    observed_match = shared_label[rows, neighbor_order]
    ks = np.arange(1, effective_max_k + 1)
    observed_precision = np.cumsum(observed_match, axis=1).mean(axis=0) / ks
    observed_coverage = np.maximum.accumulate(observed_match, axis=1).mean(axis=0)

    rng = np.random.default_rng(seed)
    null_precision = np.empty((permutations, effective_max_k), dtype=np.float32)
    null_coverage = np.empty((permutations, effective_max_k), dtype=np.float32)
    for permutation_index in range(permutations):
        permutation = rng.permutation(len(selected_ids))
        null_match = shared_label[permutation[:, None], permutation[neighbor_order]]
        null_precision[permutation_index] = np.cumsum(null_match, axis=1).mean(axis=0) / ks
        null_coverage[permutation_index] = np.maximum.accumulate(null_match, axis=1).mean(axis=0)

    retrieval_curve = pd.DataFrame(
        {
            "k": ks,
            "observed_coverage": observed_coverage,
            "null_coverage_mean": null_coverage.mean(axis=0),
            "null_coverage_low": np.quantile(null_coverage, 0.025, axis=0),
            "null_coverage_high": np.quantile(null_coverage, 0.975, axis=0),
            "observed_neighbor_precision": observed_precision,
            "null_neighbor_precision_mean": null_precision.mean(axis=0),
            "null_neighbor_precision_low": np.quantile(null_precision, 0.025, axis=0),
            "null_neighbor_precision_high": np.quantile(null_precision, 0.975, axis=0),
        }
    )

    compound_rows = []
    full_neighbor_order = np.argsort(-similarity, axis=1, kind="stable")
    for row_index, jcp2022 in enumerate(selected_ids):
        eligible = ~same_scaffold[row_index, full_neighbor_order[row_index]]
        ordered_neighbors = full_neighbor_order[row_index, eligible]
        matches = shared_label[row_index, ordered_neighbors]
        match_positions = np.flatnonzero(matches)
        first_rank = int(match_positions[0] + 1) if len(match_positions) else None
        partner_count = int((shared_label[row_index] & ~same_scaffold[row_index]).sum())
        population = int((~same_scaffold[row_index]).sum())
        compound_rows.append(
            {
                "jcp2022": jcp2022,
                "mechanism_of_action": "|".join(label_sets.loc[jcp2022]),
                "cross_scaffold_same_moa_partners": partner_count,
                "cross_scaffold_nearest_same_moa_rank": first_rank,
                "same_moa_proximity_score": chance_adjusted_proximity_score(first_rank, population, partner_count),
            }
        )
    retrieval_compounds = pd.DataFrame(compound_rows).merge(
        perturbations[["jcp2022", "umap_1", "umap_2"]],
        on="jcp2022",
        how="left",
        validate="one_to_one",
    )
    if retrieval_compounds[["umap_1", "umap_2"]].isna().any().any():
        raise AssertionError("MOA retrieval compounds are missing UMAP coordinates")
    retrieval_compounds = retrieval_compounds.sort_values("jcp2022").reset_index(drop=True)

    numbers = {
        "moa_retrieval_compounds": int(len(retrieval_compounds)),
        "moa_retrieval_compounds_with_cross_scaffold_partner": int(
            (retrieval_compounds["cross_scaffold_same_moa_partners"] > 0).sum()
        ),
        "moa_retrieval_membership_rows": int(len(active_memberships)),
        "moa_retrieval_tested_classes": int(len(tested_classes)),
    }
    return retrieval_compounds, retrieval_curve, numbers


def classify_missing_activity_results(
    well_obs: pd.DataFrame,
    perturbation_ids: pd.Series,
    activity: pd.DataFrame,
    plates: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, int]:
    """Classify why embedded perturbations lack canonical activity results."""
    analyzed_ids = set(activity["Metadata_JCP2022"].astype(str))
    missing_ids = set(perturbation_ids.astype(str)) - analyzed_ids
    missing_wells = well_obs[well_obs["JCP2022"].astype(str).isin(missing_ids)].copy()
    missing_wells["JCP2022"] = missing_wells["JCP2022"].astype(str)
    missing_wells = missing_wells.merge(
        plates.drop_duplicates(["Metadata_Source", "Metadata_Plate"]),
        left_on=["Source", "Plate"],
        right_on=["Metadata_Source", "Metadata_Plate"],
        how="left",
        validate="many_to_one",
    ).merge(
        controls.drop_duplicates("Metadata_JCP2022"),
        left_on="JCP2022",
        right_on="Metadata_JCP2022",
        how="left",
        validate="many_to_one",
    )
    if missing_wells["Metadata_PlateType"].isna().any():
        raise AssertionError("Missing activity classification found wells without plate metadata")

    counts = {
        "activity_missing_target2_only": 0,
        "activity_missing_insufficient_replicates": 0,
        "activity_missing_positive_controls": 0,
        "activity_missing_negative_controls": 0,
    }
    classified_ids: set[str] = set()
    for jcp2022, group in missing_wells.groupby("JCP2022", observed=True):
        perturbation_types = set(group["Metadata_pert_type"].dropna())
        if perturbation_types == {"poscon"}:
            counts["activity_missing_positive_controls"] += 1
        elif perturbation_types == {"negcon"}:
            counts["activity_missing_negative_controls"] += 1
        elif perturbation_types:
            raise AssertionError(f"Unexpected control types for {jcp2022}: {sorted(perturbation_types)}")
        else:
            eligible_wells = int((group["Metadata_PlateType"] != "TARGET2").sum())
            if eligible_wells == 0:
                counts["activity_missing_target2_only"] += 1
            elif eligible_wells < 2:
                counts["activity_missing_insufficient_replicates"] += 1
            else:
                raise AssertionError(
                    f"Unclassified missing activity result for {jcp2022}: {eligible_wells} eligible wells"
                )
        classified_ids.add(jcp2022)
    if classified_ids != missing_ids:
        absent = sorted(missing_ids - classified_ids)
        raise AssertionError(f"Missing activity IDs absent from well profiles: {absent}")
    if sum(counts.values()) != len(missing_ids):
        raise AssertionError("Missing activity reason counts do not cover every missing result")
    return counts


def prepare_rankings(
    connection: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str,
    metric: str,
) -> pd.DataFrame:
    """Select the top 20 normalized-AP rows for both annotation sources."""
    frames: list[pd.DataFrame] = []
    for group_type, source_label in (
        ("chemical_probes", "Chemical Probes targets"),
        ("moa", "Drug Repurposing Hub MOAs"),
    ):
        results = query_consistency_results(
            connection,
            dataset=dataset,
            preprocessing=preprocessing,
            filter_name=filter_name,
            group_type=group_type,
            distance=metric,
        )
        top = results.sort_values(
            ["mean_normalized_average_precision", "group_value"],
            ascending=[False, True],
        ).head(TOP_N)
        if len(top) != TOP_N:
            raise AssertionError(f"Expected {TOP_N} {source_label} rows, found {len(top)}")
        top = top[
            [
                "group_value",
                "n_perturbations",
                "mean_normalized_average_precision",
                "corrected_p_value",
            ]
        ].copy()
        top.insert(0, "annotation_source", source_label)
        top.insert(1, "rank", np.arange(1, TOP_N + 1, dtype=int))
        top["significant"] = top["corrected_p_value"] < CONSISTENCY_FDR_THRESHOLD
        if not top["mean_normalized_average_precision"].is_monotonic_decreasing:
            raise AssertionError(f"{source_label} rankings are not descending")
        frames.append(top)
    rankings = pd.concat(frames, ignore_index=True)
    counts = rankings.groupby("annotation_source", observed=True).size()
    if not (counts == TOP_N).all():
        raise AssertionError(f"Panel C row counts are incorrect: {counts.to_dict()}")
    return rankings


def select_curated_moas(memberships: pd.DataFrame, consistency: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain the historical MOA display list and validate current significance and counts."""
    membership_columns = {"jcp2022", "mechanism_of_action", "active"}
    consistency_columns = {
        "rank",
        "mechanism_of_action",
        "n_perturbations",
        "mean_normalized_average_precision",
        "corrected_p_value",
    }
    if set(memberships.columns) != membership_columns:
        raise ValueError("Unexpected MOA membership columns")
    if set(consistency.columns) != consistency_columns:
        raise ValueError("Unexpected MOA consistency columns")

    selected = memberships[memberships["mechanism_of_action"].isin(CURATED_MOAS)].copy()
    missing_memberships = set(CURATED_MOAS) - set(selected["mechanism_of_action"])
    if missing_memberships:
        raise ValueError(f"Curated MOAs missing from memberships: {sorted(missing_memberships)}")
    if selected.duplicated("jcp2022").any():
        overlaps = selected[selected.duplicated("jcp2022", keep=False)].sort_values(["jcp2022", "mechanism_of_action"])
        raise ValueError(f"Curated MOA memberships overlap:\n{overlaps.to_string(index=False)}")

    evidence = consistency.set_index("mechanism_of_action")
    if not evidence.index.is_unique:
        raise ValueError("MOA consistency evidence contains duplicate mechanism classes")
    missing_evidence = set(CURATED_MOAS) - set(evidence.index)
    if missing_evidence:
        raise ValueError(f"Curated MOAs missing from consistency results: {sorted(missing_evidence)}")
    # Keep the historical display cohort; its original nMAP cutoff is not a current
    # eligibility rule. Current per-class values are exported in selected_moa_classes.csv.

    rows = []
    for moa in CURATED_MOAS:
        group = selected[selected["mechanism_of_action"] == moa]
        result = evidence.loc[moa]
        active_count = int(group["active"].sum())
        consistency_count = int(result["n_perturbations"])
        corrected_p_value = float(result["corrected_p_value"])
        if corrected_p_value >= SELECTED_MOA_FDR_THRESHOLD:
            raise ValueError(f"Curated MOA does not meet {SELECTED_MOA_FDR_THRESHOLD:.0%} FDR (BH): {moa}")
        if consistency_count < SELECTED_MOA_MIN_ACTIVE_COMPOUNDS:
            raise ValueError(f"Curated MOA has fewer than {SELECTED_MOA_MIN_ACTIVE_COMPOUNDS} active compounds: {moa}")
        if active_count != consistency_count:
            raise ValueError(f"Activity/consistency count mismatch for {moa}: {active_count} != {consistency_count}")
        rows.append(
            {
                "mechanism_of_action": moa,
                "consistency_rank": int(result["rank"]),
                "total_annotated_compounds": int(len(group)),
                "active_compounds_used_for_consistency": consistency_count,
                "mean_normalized_average_precision": float(result["mean_normalized_average_precision"]),
                "corrected_p_value": corrected_p_value,
            }
        )
    return (
        selected.sort_values(["mechanism_of_action", "jcp2022"]).reset_index(drop=True),
        pd.DataFrame(rows),
    )


def robust_umap_bounds(coordinates: np.ndarray, iqr_k: float = 3.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return stable UMAP bounds while excluding only extreme coordinate outliers."""

    def bounds(values: np.ndarray) -> tuple[float, float]:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        return float(q1 - iqr_k * iqr), float(q3 + iqr_k * iqr)

    return bounds(coordinates[:, 0]), bounds(coordinates[:, 1])


def full_umap_bounds(coordinates: np.ndarray, padding: float = 0.04) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return bounds that retain every point with proportional padding."""
    lower = coordinates.min(axis=0).astype(float)
    upper = coordinates.max(axis=0).astype(float)
    margin = (upper - lower) * padding
    return (
        (lower[0] - margin[0], upper[0] + margin[0]),
        (lower[1] - margin[1], upper[1] + margin[1]),
    )


def style_umap_axis(axis: plt.Axes, bounds: tuple[tuple[float, float], tuple[float, float]]) -> None:
    """Apply the shared minimal UMAP style."""
    axis.set_xlim(bounds[0])
    axis.set_ylim(bounds[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def summarize_well_source_viewport(wells: pd.DataFrame) -> dict[str, object]:
    """Summarize the robust well-level viewport and coordinate rows it clips."""
    coordinates = wells[["umap_1", "umap_2"]].to_numpy(dtype=float)
    bounds = robust_umap_bounds(coordinates, iqr_k=WELL_SOURCE_VIEWPORT_IQR_K)
    finite = np.isfinite(coordinates).all(axis=1)
    included = (
        finite
        & (coordinates[:, 0] >= bounds[0][0])
        & (coordinates[:, 0] <= bounds[0][1])
        & (coordinates[:, 1] >= bounds[1][0])
        & (coordinates[:, 1] <= bounds[1][1])
    )
    included_wells = int(included.sum())
    total_wells = int(len(coordinates))
    return {
        "method": "IQR-clipped robust bounds",
        "iqr_k": float(WELL_SOURCE_VIEWPORT_IQR_K),
        "bounds": {
            "umap_1": [float(bounds[0][0]), float(bounds[0][1])],
            "umap_2": [float(bounds[1][0]), float(bounds[1][1])],
        },
        "total_wells": total_wells,
        "included_wells": included_wells,
        "excluded_wells": total_wells - included_wells,
    }


def plot_selected_moa_neighborhoods(
    perturbations: pd.DataFrame, caption: str, output_dir: Path, dpi: int
) -> list[Path]:
    """Plot all compounds and emphasize the selected mechanism classes."""
    figure = plt.figure(
        figsize=VIEW_FIGSIZE_INCHES["selected_moa_neighborhoods"],
        constrained_layout=True,
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 0.52))
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    axis.scatter(
        perturbations["umap_1"],
        perturbations["umap_2"],
        s=1.2,
        color="#D7D7D7",
        alpha=0.42,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    handles = []
    for moa in CURATED_MOAS:
        selected = perturbations[perturbations["moa_class"] == moa]
        axis.scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=34,
            color=MOA_PALETTE[moa],
            edgecolors="white",
            linewidths=0.55,
            alpha=0.96,
            rasterized=True,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=MOA_PALETTE[moa],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                label=f"{moa} (n={len(selected)})",
            )
        )
    legend_axis.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        title="Mechanism of action\n(total compounds)",
        handletextpad=0.7,
        labelspacing=0.9,
    )
    coordinates = perturbations[["umap_1", "umap_2"]].to_numpy()
    style_umap_axis(axis, full_umap_bounds(coordinates))
    axis.set_title("Selected mechanism-of-action classes", fontsize=PANEL_TITLE_FONTSIZE, pad=8)
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["selected_moa_neighborhoods"], dpi)


def draw_global_moa_map(
    figure: plt.Figure,
    umap_axis: plt.Axes,
    perturbations: pd.DataFrame,
    retrieval_compounds: pd.DataFrame,
    score_upper: float,
) -> None:
    """Draw the atlas-wide MOA proximity map on one axis."""
    umap_axis.scatter(
        perturbations["umap_1"],
        perturbations["umap_2"],
        s=1.0,
        color="#D9D9D9",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    points = umap_axis.scatter(
        retrieval_compounds["umap_1"],
        retrieval_compounds["umap_2"],
        c=retrieval_compounds["same_moa_proximity_score"],
        s=10,
        cmap=GLOBAL_MOA_MAP_CMAP,
        vmin=0,
        vmax=score_upper,
        edgecolors="none",
        alpha=0.88,
        rasterized=True,
        zorder=2,
    )
    coordinates = perturbations[["umap_1", "umap_2"]].to_numpy()
    style_umap_axis(umap_axis, full_umap_bounds(coordinates))
    umap_axis.set_xlabel("UMAP 1", fontsize=GLOBAL_MOA_MAP_UMAP_LABEL_FONTSIZE)
    umap_axis.set_ylabel("UMAP 2", fontsize=GLOBAL_MOA_MAP_UMAP_LABEL_FONTSIZE)
    umap_axis.set_title("Global map of tested compounds", fontsize=PANEL_TITLE_FONTSIZE, pad=26)
    umap_axis.text(
        0.5,
        1.01,
        f"{len(retrieval_compounds):,} active, MOA-annotated compounds",
        transform=umap_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=GLOBAL_MOA_MAP_COUNT_FONTSIZE,
        color="#444444",
        clip_on=False,
    )
    colorbar = figure.colorbar(
        points,
        ax=umap_axis,
        fraction=0.036,
        pad=0.02,
        shrink=GLOBAL_MOA_MAP_COLORBAR_SHRINK,
        aspect=14,
        ticks=GLOBAL_MOA_MAP_COLORBAR_TICKS,
    )
    colorbar.ax.tick_params(
        labelsize=GLOBAL_MOA_MAP_COLORBAR_TICK_FONTSIZE,
        length=6,
        width=0.9,
    )
    colorbar.ax.text(
        0.5,
        GLOBAL_MOA_MAP_COLORBAR_LABEL_OFFSET,
        "Same-MOA proximity\n"
        "(chance-adjusted -log10\n"
        "probability; color capped\n"
        f"at {100 * GLOBAL_MOA_MAP_PROXIMITY_SCORE_UPPER_QUANTILE:.0f}th percentile)",
        transform=colorbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=GLOBAL_MOA_MAP_COLORBAR_LABEL_FONTSIZE,
        linespacing=1.05,
    )


def draw_cross_scaffold_moa_retrieval(
    retrieval_axis: plt.Axes,
    retrieval_curve: pd.DataFrame,
    *,
    annotation_offsets: dict[int, tuple[int, int]] | None = None,
    x_label: str | None = None,
    main_text_style: bool = False,
) -> None:
    """Draw the observed and label-permuted MOA retrieval curves."""
    if annotation_offsets is None:
        annotation_offsets = CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_OFFSETS
    if x_label is None:
        x_label = "Number of phenotypic neighbors examined (k)"
    retrieval_axis.fill_between(
        retrieval_curve["k"],
        100 * retrieval_curve["null_coverage_low"],
        100 * retrieval_curve["null_coverage_high"],
        color="#BDBDBD",
        alpha=0.24 if main_text_style else 0.34,
        linewidth=0,
        label="Label-permuted 95% interval",
    )
    retrieval_axis.plot(
        retrieval_curve["k"],
        100 * retrieval_curve["null_coverage_mean"],
        color="#858585" if main_text_style else "#777777",
        linewidth=1.4 if main_text_style else 1.5,
        linestyle="--",
        label="Label-permuted mean",
    )
    retrieval_axis.plot(
        retrieval_curve["k"],
        100 * retrieval_curve["observed_coverage"],
        color="#0072B2",
        linewidth=3.0 if main_text_style else 2.4,
        label="Observed",
    )
    retrieval_axis.set_xscale("log")
    retrieval_axis.set_xlim(1, int(retrieval_curve["k"].max()))
    retrieval_axis.set_xticks([1, 2, 5, 10, 20, 50, 100])
    retrieval_axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    retrieval_axis.set_ylim(bottom=0)
    retrieval_axis.set_xlabel(x_label, fontsize=CROSS_SCAFFOLD_RETRIEVAL_X_LABEL_FONTSIZE)
    retrieval_axis.set_ylabel(
        "Compounds with a same-MOA neighbor (%)",
        fontsize=CROSS_SCAFFOLD_RETRIEVAL_Y_LABEL_FONTSIZE,
        labelpad=CROSS_SCAFFOLD_RETRIEVAL_Y_LABEL_PAD,
    )
    retrieval_axis.set_title(
        "Cross-scaffold MOA retrieval",
        fontsize=PANEL_TITLE_FONTSIZE + (1 if main_text_style else 0),
        fontweight="semibold" if main_text_style else "normal",
        pad=8,
    )
    if main_text_style:
        retrieval_axis.grid(False)
    else:
        retrieval_axis.grid(color="#E8E8E8", linewidth=0.7)
    retrieval_axis.set_axisbelow(True)
    retrieval_axis.spines[["top", "right"]].set_visible(False)
    if main_text_style:
        retrieval_axis.tick_params(
            labelsize=CROSS_SCAFFOLD_RETRIEVAL_TICK_FONTSIZE,
            length=5,
            width=0.9,
        )
        legend_handles, legend_labels = retrieval_axis.get_legend_handles_labels()
        legend_handles.reverse()
        legend_labels.reverse()
        retrieval_axis.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            frameon=False,
            fontsize=CROSS_SCAFFOLD_RETRIEVAL_LEGEND_FONTSIZE,
            handlelength=2.4,
            labelspacing=0.45,
        )
    else:
        retrieval_axis.tick_params(labelsize=CROSS_SCAFFOLD_RETRIEVAL_TICK_FONTSIZE)
        retrieval_axis.legend(loc="upper left", frameon=False, fontsize=CROSS_SCAFFOLD_RETRIEVAL_LEGEND_FONTSIZE)

    annotation_ks = (
        CROSS_SCAFFOLD_RETRIEVAL_MAIN_TEXT_ANNOTATION_KS if main_text_style else CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_KS
    )
    for k in annotation_ks:
        row = retrieval_curve.loc[retrieval_curve["k"] == k].iloc[0]
        observed = 100 * float(row["observed_coverage"])
        null = 100 * float(row["null_coverage_mean"])
        retrieval_axis.scatter(k, observed, s=24, color="#0072B2", zorder=4)
        retrieval_axis.annotate(
            f"k={k}: {observed:.1f}% observed\n{null:.1f}% label-permuted",
            xy=(k, observed),
            xytext=annotation_offsets[k],
            textcoords="offset points",
            fontsize=(
                CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_FONTSIZE + 1.5
                if main_text_style
                else CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_FONTSIZE
            ),
            color="#333333",
            bbox=None
            if main_text_style
            else {
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
            arrowprops={"arrowstyle": "-", "color": "#AEAEAE", "linewidth": 0.8},
            zorder=5,
        )


def plot_global_moa_map(
    perturbations: pd.DataFrame,
    retrieval_compounds: pd.DataFrame,
    score_upper: float,
    caption: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Render the standalone global MOA proximity map."""
    figure, umap_axis = plt.subplots(figsize=VIEW_FIGSIZE_INCHES["global_moa_map"], constrained_layout=True)
    draw_global_moa_map(figure, umap_axis, perturbations, retrieval_compounds, score_upper)
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["global_moa_map"], dpi)


def plot_cross_scaffold_moa_retrieval(
    retrieval_curve: pd.DataFrame, caption: str, output_dir: Path, dpi: int
) -> list[Path]:
    """Render the standalone cross-scaffold MOA retrieval curve."""
    figure, retrieval_axis = plt.subplots(
        figsize=VIEW_FIGSIZE_INCHES["cross_scaffold_moa_retrieval"],
        constrained_layout=True,
    )
    draw_cross_scaffold_moa_retrieval(retrieval_axis, retrieval_curve, main_text_style=True)
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["cross_scaffold_moa_retrieval"], dpi)


def plot_combined_global_moa_retrieval(
    perturbations: pd.DataFrame,
    retrieval_compounds: pd.DataFrame,
    retrieval_curve: pd.DataFrame,
    score_upper: float,
    caption: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Render the retained combined global-map and retrieval layout."""
    figure = plt.figure(
        figsize=VIEW_FIGSIZE_INCHES["combined_global_moa_retrieval"],
        constrained_layout=True,
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.05, 0.95), wspace=COMBINED_RETRIEVAL_WSPACE)
    umap_axis = figure.add_subplot(grid[0, 0])
    retrieval_axis = figure.add_subplot(grid[0, 1])
    draw_global_moa_map(figure, umap_axis, perturbations, retrieval_compounds, score_upper)
    draw_cross_scaffold_moa_retrieval(
        retrieval_axis,
        retrieval_curve,
        annotation_offsets=COMBINED_RETRIEVAL_ANNOTATION_OFFSETS,
        x_label=COMBINED_RETRIEVAL_X_LABEL,
    )
    figure.suptitle(
        "Same-MOA proximity and cross-scaffold retrieval",
        fontsize=COMBINED_TITLE_FONTSIZE,
    )
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["combined_global_moa_retrieval"], dpi)


def format_source(source: str) -> str:
    """Format one source identifier for the legend."""
    return source.replace("source_", "Source ")


def plot_profile_context_umaps(
    wells: pd.DataFrame,
    perturbations: pd.DataFrame,
    well_source_viewport: dict[str, object],
    caption: str,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Plot perturbation activity, cell count, and well-level source context."""
    figure = plt.figure(figsize=VIEW_FIGSIZE_INCHES["profile_context_umaps"], constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.0, 0.24))
    axes = [figure.add_subplot(grid[row, 0]) for row in range(3)]
    legend_axes = [figure.add_subplot(grid[row, 1]) for row in range(3)]
    for legend_axis in legend_axes:
        legend_axis.axis("off")

    activity_palette = {
        "Active": ACTIVE_COLOR,
        "Inactive": INACTIVE_COLOR,
        "No activity result": NO_RESULT_COLOR,
    }
    sizes = {"Active": 2.2, "Inactive": 0.75, "No activity result": 2.0}
    alphas = {"Active": 0.82, "Inactive": 0.34, "No activity result": 0.9}
    for activity_class in ("Inactive", "No activity result", "Active"):
        selected = perturbations[perturbations["activity_class"] == activity_class]
        axes[0].scatter(
            selected["umap_1"],
            selected["umap_2"],
            s=sizes[activity_class],
            color=activity_palette[activity_class],
            alpha=alphas[activity_class],
            linewidths=0,
            rasterized=True,
        )
    perturbation_coordinates = perturbations[["umap_1", "umap_2"]].to_numpy()
    style_umap_axis(axes[0], full_umap_bounds(perturbation_coordinates))
    axes[0].set_title(
        "Perturbation-level profiles by phenotypic activity",
        fontsize=CONTEXT_TITLE_FONTSIZE,
        pad=8,
    )
    activity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=activity_palette[label],
            markeredgecolor="none",
            markersize=6,
            label=label,
        )
        for label in ("Active", "Inactive", "No activity result")
    ]
    legend_axes[0].legend(
        handles=activity_handles,
        loc="center left",
        frameon=False,
        title=f"Significant at {ACTIVITY_FDR_THRESHOLD:.0%} FDR (BH)",
    )

    cell_count_values = perturbations["median_cell_count"].to_numpy(dtype=float)
    cell_count_limits = np.quantile(cell_count_values, CONTEXT_CELL_COUNT_QUANTILES)
    cell_count_norm = Normalize(vmin=float(cell_count_limits[0]), vmax=float(cell_count_limits[1]), clip=True)
    cell_count_points = axes[1].scatter(
        perturbations["umap_1"],
        perturbations["umap_2"],
        c=cell_count_values,
        s=CONTEXT_CELL_COUNT_POINT_SIZE,
        cmap=CONTEXT_CELL_COUNT_CMAP,
        norm=cell_count_norm,
        alpha=CONTEXT_CELL_COUNT_ALPHA,
        linewidths=0,
        rasterized=True,
    )
    style_umap_axis(axes[1], full_umap_bounds(perturbation_coordinates))
    axes[1].set_title(
        "Perturbation-level profiles by median cell count",
        fontsize=CONTEXT_TITLE_FONTSIZE,
        pad=8,
    )
    cell_count_cax = legend_axes[1].inset_axes([0.08, 0.18, 0.15, 0.64])
    cell_count_colorbar = figure.colorbar(cell_count_points, cax=cell_count_cax)
    cell_count_colorbar.set_label(
        "Median cells per well\n(color clipped to 1st-99th percentiles)",
        fontsize=CONTEXT_COLORBAR_LABEL_FONTSIZE,
        labelpad=10,
    )
    cell_count_colorbar.ax.tick_params(labelsize=CONTEXT_COLORBAR_TICK_FONTSIZE)

    sources = sorted(
        wells["source"].astype(str).unique(),
        key=lambda value: int(value.split("_")[-1]),
    )
    if len(sources) > len(SOURCE_COLORS):
        raise AssertionError(f"Source palette has {len(SOURCE_COLORS)} colors for {len(sources)} sources")
    source_palette = dict(zip(sources, SOURCE_COLORS, strict=False))
    well_coordinates = wells[["umap_1", "umap_2"]].to_numpy()
    source_values = wells["source"].astype(str).to_numpy()
    order = np.random.default_rng(WELL_PLOT_ORDER_SEED).permutation(len(wells))
    point_colors = np.array([source_palette[source] for source in source_values], dtype=object)
    axes[2].scatter(
        well_coordinates[order, 0],
        well_coordinates[order, 1],
        s=0.35,
        color=point_colors[order],
        alpha=0.46,
        linewidths=0,
        rasterized=True,
    )
    viewport_bounds = well_source_viewport["bounds"]
    if not isinstance(viewport_bounds, dict):
        raise TypeError("Well source viewport bounds must be a mapping")
    well_source_bounds = (
        tuple(viewport_bounds["umap_1"]),
        tuple(viewport_bounds["umap_2"]),
    )
    style_umap_axis(axes[2], well_source_bounds)
    axes[2].set_title(
        "Well-level profiles by data-generating source",
        fontsize=CONTEXT_TITLE_FONTSIZE,
        pad=8,
    )
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=source_palette[source],
            markeredgecolor="none",
            markersize=6,
            label=format_source(source),
        )
        for source in sources
    ]
    legend_axes[2].legend(handles=source_handles, loc="center left", frameon=False, title="Source")
    legend_axes[2].text(
        0.0,
        0.02,
        "Robust viewport\n"
        f"{well_source_viewport['included_wells']:,} of "
        f"{well_source_viewport['total_wells']:,} wells visible\n"
        f"({well_source_viewport['excluded_wells']:,} outside bounds)",
        transform=legend_axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=WELL_SOURCE_VIEWPORT_NOTE_FONTSIZE,
        color="#555555",
    )
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["profile_context_umaps"], dpi)


def point_sizes(counts: pd.Series | np.ndarray) -> np.ndarray:
    """Map group sizes to ranking-plot point areas."""
    values = np.asarray(counts, dtype=float)
    return 24.0 + 30.0 * np.sqrt(values)


def plot_ranking_axis(axis: plt.Axes, data: pd.DataFrame, title: str, title_x: float) -> None:
    """Draw one ranked normalized-AP axis."""
    y = np.arange(len(data))
    values = data["mean_normalized_average_precision"].to_numpy()
    colors = np.where(data["significant"], SIGNIFICANT_COLOR, NONSIGNIFICANT_COLOR)
    display_labels = [
        RANKING_LABEL_FORMAT.format(
            label=RANKING_LABELS.get(label, label),
            count=int(count),
        )
        for label, count in zip(data["group_value"], data["n_perturbations"], strict=True)
    ]
    axis.hlines(y, 0, values, color="#E3E3E3", linewidth=0.8, zorder=1)
    axis.scatter(
        values,
        y,
        s=point_sizes(data["n_perturbations"]),
        c=colors,
        edgecolors="white",
        linewidths=0.55,
        zorder=2,
    )
    axis.set_yticks(y, labels=display_labels, fontsize=RANKING_Y_LABEL_FONTSIZE)
    axis.invert_yaxis()
    axis.set_xlim(left=0)
    axis.set_xlabel("Mean normalized average precision", fontsize=RANKING_X_LABEL_FONTSIZE)
    axis.set_title(title, fontsize=RANKING_TITLE_FONTSIZE, x=title_x, pad=10)
    axis.grid(axis="x", color="#EAEAEA", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=RANKING_X_TICK_FONTSIZE)
    axis.tick_params(axis="y", length=0)


def plot_ranked_targets_and_moas(rankings: pd.DataFrame, caption: str, output_dir: Path, dpi: int) -> list[Path]:
    """Plot side-by-side ranked target and MOA consistency results."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=VIEW_FIGSIZE_INCHES["ranked_targets_and_moas"],
        constrained_layout=True,
    )
    for axis, (source, title, title_x) in zip(
        axes,
        (
            (
                "Chemical Probes targets",
                "Chemical Probes: targets",
                RANKING_TITLE_X_BY_SOURCE["Chemical Probes targets"],
            ),
            (
                "Drug Repurposing Hub MOAs",
                "Repurposing Hub: MOAs",
                RANKING_TITLE_X_BY_SOURCE["Drug Repurposing Hub MOAs"],
            ),
        ),
        strict=True,
    ):
        plot_ranking_axis(axis, rankings[rankings["annotation_source"] == source], title, title_x)

    significance_handles = []
    for is_significant, color, label in (
        (
            True,
            SIGNIFICANT_COLOR,
            f"Significant at {CONSISTENCY_FDR_THRESHOLD:.0%} FDR (BH)",
        ),
        (
            False,
            NONSIGNIFICANT_COLOR,
            f"Not significant at {CONSISTENCY_FDR_THRESHOLD:.0%} FDR (BH)",
        ),
    ):
        if bool((rankings["significant"] == is_significant).any()):
            significance_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markersize=7,
                    label=label,
                )
            )
    figure.legend(
        handles=significance_handles,
        loc=RANKING_SIGNIFICANCE_LEGEND_LOCATION,
        ncol=RANKING_SIGNIFICANCE_LEGEND_COLUMNS,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#D0D0D0",
        title="Point color",
        fontsize=RANKING_LEGEND_FONTSIZE,
        title_fontsize=RANKING_LEGEND_TITLE_FONTSIZE,
    )

    size_handles = [
        axes[1].scatter(
            [],
            [],
            s=point_sizes(np.array([count]))[0],
            color="#777777",
            edgecolors="white",
            linewidths=0.55,
            label=str(count),
        )
        for count in (2, 5, 10, 20)
    ]
    axes[1].legend(
        handles=size_handles,
        loc=RANKING_SIZE_LEGEND_LOCATION,
        ncol=RANKING_SIZE_LEGEND_COLUMNS,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#D0D0D0",
        title="Active compounds",
        fontsize=RANKING_LEGEND_FONTSIZE,
        title_fontsize=RANKING_LEGEND_TITLE_FONTSIZE,
    )
    attach_caption(figure, caption)
    return save_figure(figure, output_dir, VIEW_STEMS["ranked_targets_and_moas"], dpi)


def prepare_figure_data(
    *,
    well_umap: Path,
    perturbation_umap: Path,
    perturbation_profiles: Path,
    metadata_db: Path,
    copairs_db: Path,
    dataset: str,
    activity_preprocessing: str,
    activity_filter: str,
    metric: str,
    permutations: int,
    seed: int,
    max_k: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, int | float], dict[str, object]]:
    """Load the five canonical inputs and derive every table this figure needs."""
    consistency_name = consistency_preprocessing(activity_preprocessing)

    logger.info(f"Loading well-level UMAP from {well_umap}")
    well_obs, well_coordinates, well_umap_seed = load_embedding(well_umap, ["Source", "Plate", "JCP2022"])
    wells = pd.DataFrame(
        {
            "source": well_obs["Source"].astype(str).to_numpy(),
            "umap_1": well_coordinates[:, 0],
            "umap_2": well_coordinates[:, 1],
        }
    )

    logger.info(f"Loading perturbation-level UMAP from {perturbation_umap}")
    perturbation_obs, perturbation_coordinates, perturbation_umap_seed = load_embedding(perturbation_umap, ["JCP2022"])
    if well_umap_seed != perturbation_umap_seed:
        raise AssertionError(
            "Well- and perturbation-level UMAPs use different random seeds: "
            f"{well_umap_seed} != {perturbation_umap_seed}"
        )
    perturbations = pd.DataFrame(
        {
            "jcp2022": perturbation_obs["JCP2022"].astype(str).to_numpy(),
            "umap_1": perturbation_coordinates[:, 0],
            "umap_2": perturbation_coordinates[:, 1],
        }
    )
    if perturbations["jcp2022"].duplicated().any():
        raise AssertionError("Perturbation embedding contains duplicate compound identifiers")

    copairs = duckdb.connect(str(copairs_db), read_only=True)
    metadata = duckdb.connect(str(metadata_db), read_only=True)
    try:
        activity = query_activity_results(
            copairs,
            dataset=dataset,
            preprocessing=activity_preprocessing,
            filter_name=activity_filter,
            activity_params=DEFAULT_ACTIVITY_PARAMS,
        )
        if activity.empty or activity["Metadata_JCP2022"].duplicated().any():
            raise AssertionError("Activity query returned no rows or duplicate compound identifiers")
        if (
            dataset,
            activity_preprocessing,
            activity_filter,
            DEFAULT_ACTIVITY_PARAMS,
        ) == (
            DEFAULT_DATASET,
            DEFAULT_ACTIVITY_PREPROCESSING,
            DEFAULT_ACTIVITY_FILTER,
            DEFAULT_ACTIVITY_PARAMS,
        ):
            stored = activity["below_corrected_p"].astype(bool).to_numpy()
            thresholded = (activity["corrected_p_value"] < ACTIVITY_FDR_THRESHOLD).to_numpy()
            if not np.array_equal(stored, thresholded):
                raise AssertionError(
                    "Canonical activity flags do not equal corrected-p < 0.10 calls for "
                    "activity_no_target2/all_sources/default"
                )

        annotations = metadata.execute(
            "SELECT Metadata_JCP2022, Metadata_repurposing_moa, Metadata_MurckoScaffold, "
            "Metadata_median_cell_count FROM compound_metadata"
        ).df()
        activity_missing_counts = classify_missing_activity_results(
            well_obs,
            perturbations["jcp2022"],
            activity,
            metadata.execute("SELECT Metadata_Source, Metadata_Plate, Metadata_PlateType FROM plate").df(),
            metadata.execute("SELECT Metadata_JCP2022, Metadata_pert_type FROM perturbation_control").df(),
        )
        moa_memberships = exact_moa_memberships(annotations, perturbations["jcp2022"])
        moa_memberships = attach_moa_activity(moa_memberships, activity)
        moa_results = query_consistency_results(
            copairs,
            dataset=dataset,
            preprocessing=consistency_name,
            filter_name=activity_filter,
            group_type="moa",
            distance=metric,
        )
        moa_consistency = prepare_moa_consistency(moa_results)
        retrieval_compounds, retrieval_curve, retrieval_numbers = prepare_global_moa_retrieval(
            moa_memberships,
            moa_consistency,
            annotations,
            perturbations,
            perturbation_profiles,
            permutations,
            seed,
            max_k,
        )
        rankings = prepare_rankings(
            copairs,
            dataset=dataset,
            preprocessing=consistency_name,
            filter_name=activity_filter,
            metric=metric,
        )
    finally:
        copairs.close()
        metadata.close()

    activity_for_plot = activity.rename(columns={"Metadata_JCP2022": "jcp2022"})[
        ["jcp2022", "corrected_p_value"]
    ].copy()
    activity_for_plot["jcp2022"] = activity_for_plot["jcp2022"].astype(str)
    perturbations = perturbations.merge(activity_for_plot, on="jcp2022", how="left", validate="one_to_one")
    cell_counts = annotations[["Metadata_JCP2022", "Metadata_median_cell_count"]].rename(
        columns={
            "Metadata_JCP2022": "jcp2022",
            "Metadata_median_cell_count": "median_cell_count",
        }
    )
    cell_counts["jcp2022"] = cell_counts["jcp2022"].astype(str)
    perturbations = perturbations.merge(cell_counts, on="jcp2022", how="left", validate="one_to_one")
    if perturbations["median_cell_count"].isna().any():
        raise AssertionError("Perturbation embedding contains compounds without median cell counts")
    if (perturbations["median_cell_count"] <= 0).any():
        raise AssertionError("Median cell counts must be positive")
    perturbations["activity_class"] = np.select(
        [
            perturbations["corrected_p_value"] < ACTIVITY_FDR_THRESHOLD,
            perturbations["corrected_p_value"].notna(),
        ],
        ["Active", "Inactive"],
        default="No activity result",
    )

    curated_memberships, moa_classes = select_curated_moas(moa_memberships, moa_consistency)
    selected_moa_perturbations = perturbations.merge(
        curated_memberships[["jcp2022", "mechanism_of_action"]].rename(columns={"mechanism_of_action": "moa_class"}),
        on="jcp2022",
        how="left",
        validate="one_to_one",
    )
    well_source_viewport = summarize_well_source_viewport(wells)
    if int(well_source_viewport["included_wells"]) + int(well_source_viewport["excluded_wells"]) != len(wells):
        raise AssertionError("Well viewport counts do not sum to all well profiles")

    activity_counts = perturbations["activity_class"].value_counts()
    numbers: dict[str, int | float] = {
        "well_level_profiles": int(len(wells)),
        "perturbation_level_profiles": int(len(perturbations)),
        "active_perturbations": int(activity_counts.get("Active", 0)),
        "inactive_perturbations": int(activity_counts.get("Inactive", 0)),
        "perturbations_without_activity_results": int(activity_counts.get("No activity result", 0)),
        "perturbations_with_median_cell_count": int(perturbations["median_cell_count"].notna().sum()),
        "median_cell_count_min": int(perturbations["median_cell_count"].min()),
        "median_cell_count_median": float(perturbations["median_cell_count"].median()),
        "median_cell_count_max": int(perturbations["median_cell_count"].max()),
        **activity_missing_counts,
        "moa_membership_rows": int(len(moa_memberships)),
        "moa_annotated_compounds": int(moa_memberships["jcp2022"].nunique()),
        "moa_consistency_classes": int(len(moa_consistency)),
        **retrieval_numbers,
        "chemical_probes_ranking_rows": int((rankings["annotation_source"] == "Chemical Probes targets").sum()),
        "repurposing_hub_moa_ranking_rows": int((rankings["annotation_source"] == "Drug Repurposing Hub MOAs").sum()),
        "curated_moa_classes": int(len(CURATED_MOAS)),
        "curated_annotated_compounds": int(curated_memberships["jcp2022"].nunique()),
        "active_curated_compounds": int(curated_memberships["active"].sum()),
        "well_source_viewport_included_wells": int(well_source_viewport["included_wells"]),
        "well_source_viewport_excluded_wells": int(well_source_viewport["excluded_wells"]),
    }
    missing_total = (
        numbers["activity_missing_target2_only"]
        + numbers["activity_missing_insufficient_replicates"]
        + numbers["activity_missing_positive_controls"]
        + numbers["activity_missing_negative_controls"]
    )
    if missing_total != numbers["perturbations_without_activity_results"]:
        raise AssertionError("Missing-activity reason counts do not sum to all gray profiles")

    configuration: dict[str, object] = {
        "dataset": dataset,
        "activity_preprocessing": activity_preprocessing,
        "activity_filter": activity_filter,
        "activity_params": DEFAULT_ACTIVITY_PARAMS,
        "activity_corrected_p_threshold": ACTIVITY_FDR_THRESHOLD,
        "consistency_preprocessing": consistency_name,
        "consistency_corrected_p_threshold": CONSISTENCY_FDR_THRESHOLD,
        "metric": metric,
        "umap_random_seed": well_umap_seed,
        "moa_retrieval_permutations": permutations,
        "moa_retrieval_seed": seed,
        "moa_retrieval_max_k": max_k,
        "moa_retrieval_effective_max_k": int(retrieval_curve["k"].max()),
        "moa_retrieval_exact_shared_scaffolds_excluded": True,
        "well_plot_order_seed": WELL_PLOT_ORDER_SEED,
        "well_source_viewport": well_source_viewport,
        "context_cell_count_quantiles": list(CONTEXT_CELL_COUNT_QUANTILES),
        "selected_moa_curated_moas": list(CURATED_MOAS),
        "selected_moa_bh_fdr_threshold": SELECTED_MOA_FDR_THRESHOLD,
        "selected_moa_min_active_compounds": SELECTED_MOA_MIN_ACTIVE_COMPOUNDS,
        "selected_moa_min_normalized_average_precision": (SELECTED_MOA_MIN_NORMALIZED_AVERAGE_PRECISION),
    }
    tables = {
        "wells": wells,
        "perturbations": perturbations,
        "selected_moa_perturbations": selected_moa_perturbations,
        "moa_classes": moa_classes,
        "retrieval_compounds": retrieval_compounds,
        "retrieval_curve": retrieval_curve,
        "rankings": rankings,
    }
    return tables, numbers, configuration


def build_captions(
    *,
    numbers: dict[str, int | float],
    configuration: dict[str, object],
    well_source_viewport: dict[str, object],
    retrieval_curve: pd.DataFrame,
    global_moa_score_upper: float,
) -> dict[str, str]:
    """Author the six self-contained view captions from computed values."""
    annotations = {
        int(k): retrieval_curve.loc[retrieval_curve["k"] == k].iloc[0] for k in CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_KS
    }
    retrieval_k1 = annotations[1]
    retrieval_k10 = annotations[10]
    perturbation_count = int(numbers["perturbation_level_profiles"])
    active_count = int(numbers["active_perturbations"])
    no_activity_count = int(numbers["perturbations_without_activity_results"])
    missing_target2 = int(numbers["activity_missing_target2_only"])
    missing_replicates = int(numbers["activity_missing_insufficient_replicates"])
    missing_positive_controls = int(numbers["activity_missing_positive_controls"])
    missing_negative_controls = int(numbers["activity_missing_negative_controls"])
    missing_controls = missing_positive_controls + missing_negative_controls
    active_fraction = active_count / perturbation_count
    no_activity_fraction = no_activity_count / perturbation_count
    activity_gray_breakdown = (
        f"Gray marks {no_activity_count:,} profiles ({no_activity_fraction:.2%}) without an activity result: "
        f"{missing_target2:,} TARGET2-only profiles, {missing_replicates:,} profiles with one eligible well, and "
        f"{missing_controls:,} controls ({missing_positive_controls:,} positive and "
        f"{missing_negative_controls:,} negative)."
    )
    ranking_abbreviations = (
        "Abbreviations: HIF, hypoxia-inducible factor; IMPDH, inosine monophosphate "
        "dehydrogenase; KDM, histone lysine demethylase; KMT, histone lysine "
        "methyltransferase; KSP, kinesin spindle protein; MMP, matrix metalloprotease; "
        "PFK, phosphofructokinase."
    )
    retrieval_count = int(numbers["moa_retrieval_compounds"])
    permutations = int(configuration["moa_retrieval_permutations"])
    cross_scaffold_description = (
        "active, MOA-annotated compounds were ranked by high-dimensional profile cosine similarity after excluding "
        "compounds with the same Bemis-Murcko scaffold. The curve shows the cumulative percentage with a same-MOA "
        "compound among the k nearest eligible neighbors. The gray 95% interval comes from "
        f"{permutations:,} permutations of complete multilabel annotation sets. Coverage was "
        f"{retrieval_k1['observed_coverage']:.1%} observed versus "
        f"{retrieval_k1['null_coverage_mean']:.1%} permuted at k=1 and "
        f"{retrieval_k10['observed_coverage']:.1%} versus "
        f"{retrieval_k10['null_coverage_mean']:.1%} at k=10."
    )
    global_moa_description = (
        f"cosine UMAP of {perturbation_count:,} perturbation-level profiles (gray). The "
        f"{retrieval_count:,} active, MOA-annotated compounds are colored by the chance-adjusted -log10 "
        "probability of same-MOA proximity in the high-dimensional profile space. Color is capped at the "
        f"{100 * GLOBAL_MOA_MAP_PROXIMITY_SCORE_UPPER_QUANTILE:.0f}th percentile "
        f"(score {global_moa_score_upper:.2f})."
    )
    return {
        "profile_context_umaps": (
            f"Top, UMAP of {perturbation_count:,} perturbation-level profiles colored by activity at "
            f"{ACTIVITY_FDR_THRESHOLD:.0%} Benjamini-Hochberg FDR: {active_count:,} "
            f"({active_fraction:.1%}) active profiles in orange and tested profiles below threshold in light blue. "
            f"{activity_gray_breakdown} Middle, the same UMAP colored by median cells per well, clipped to the "
            "1st-99th percentiles. Bottom, well-level profiles colored by source. The viewport shows "
            f"{well_source_viewport['included_wells']:,} of {well_source_viewport['total_wells']:,} wells and omits "
            f"{well_source_viewport['excluded_wells']:,} beyond the {WELL_SOURCE_VIEWPORT_IQR_K:g}-IQR bounds. "
            "Activity was calculated from the "
            "high-dimensional profiles."
        ),
        "cross_scaffold_moa_retrieval": (f"{cross_scaffold_description[0].upper()}{cross_scaffold_description[1:]}"),
        "ranked_targets_and_moas": (
            "The 20 Chemical Probes targets and 20 Drug Repurposing Hub MOAs with the highest mean normalized "
            "average precision. Point area and labels give the number of active compounds; color marks "
            f"Benjamini-Hochberg significance at 5% FDR. {ranking_abbreviations}"
        ),
        "global_moa_map": f"{global_moa_description[0].upper()}{global_moa_description[1:]}",
        "selected_moa_neighborhoods": (
            f"The same UMAP with {len(CURATED_MOAS)} historically selected MOA classes highlighted. "
            f"Displayed classes currently pass {SELECTED_MOA_FDR_THRESHOLD:.0%} Benjamini-Hochberg FDR "
            f"and contain at least {SELECTED_MOA_MIN_ACTIVE_COMPOUNDS} active compounds. "
            "Historical selection used mean normalized average precision "
            f">= {SELECTED_MOA_MIN_NORMALIZED_AVERAGE_PRECISION:g}; this cutoff is not reapplied. "
            "Current per-class values are reported in selected_moa_classes.csv. "
            "Legend counts include all annotated compounds."
        ),
        "combined_global_moa_retrieval": (f"Left, {global_moa_description} Right, {cross_scaffold_description}"),
    }


def relative_file_records(
    output_dir: Path,
    view_paths: dict[str, list[Path]],
    table_paths: list[Path],
    legend_path: Path,
) -> list[dict[str, str]]:
    """Build a deterministic standard file index with paths relative to the output directory.

    Manifest captions are short labels; the real caption is drawn on the view and written
    to `legend.md`.
    """
    records: list[dict[str, str]] = []
    for view_id in VIEW_ORDER:
        for path in view_paths[view_id]:
            records.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "kind": "figure",
                    "caption": SECTION_TITLES[view_id],
                }
            )
    for path in table_paths:
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "kind": "table",
                "caption": path.stem.replace("_", " "),
            }
        )
    records.append(
        {
            "path": legend_path.relative_to(output_dir).as_posix(),
            "kind": "data",
            "caption": "Figure legend",
        }
    )
    return records


def write_figure_outputs(
    *,
    tables: dict[str, pd.DataFrame],
    numbers: dict[str, int | float],
    configuration: dict[str, object],
    output_dir: Path,
    dpi: int,
) -> dict:
    """Write all images, companion tables, legend, and the standard summary."""
    if dpi < 72:
        raise ValueError("Figure DPI must be at least 72")
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = tables["wells"]
    perturbations = tables["perturbations"]
    selected_moa_perturbations = tables["selected_moa_perturbations"]
    moa_classes = tables["moa_classes"]
    retrieval_compounds = tables["retrieval_compounds"]
    retrieval_curve = tables["retrieval_curve"]
    rankings = tables["rankings"]
    if not set(CROSS_SCAFFOLD_RETRIEVAL_ANNOTATION_KS).issubset(set(retrieval_curve["k"].astype(int))):
        raise ValueError("Retrieval curve must include the authored k=1 and k=10 annotations")

    global_moa_score_upper = max(
        1.0,
        float(retrieval_compounds["same_moa_proximity_score"].quantile(GLOBAL_MOA_MAP_PROXIMITY_SCORE_UPPER_QUANTILE)),
    )
    score_values = retrieval_compounds["same_moa_proximity_score"].to_numpy(dtype=float)
    cell_counts = perturbations["median_cell_count"].to_numpy(dtype=float)
    cell_count_limits = np.quantile(cell_counts, CONTEXT_CELL_COUNT_QUANTILES)
    numbers.update(
        {
            "global_moa_map_saturated_compounds": int((score_values >= global_moa_score_upper).sum()),
            "cell_count_below_color_range": int((cell_counts < cell_count_limits[0]).sum()),
            "cell_count_above_color_range": int((cell_counts > cell_count_limits[1]).sum()),
        }
    )
    configuration.update(
        {
            "dpi": dpi,
            "global_moa_map_proximity_score_upper_quantile": (GLOBAL_MOA_MAP_PROXIMITY_SCORE_UPPER_QUANTILE),
            "global_moa_map_proximity_score_upper_value": global_moa_score_upper,
            "global_moa_map_values_at_or_above_cap_share_maximum_color": True,
            "context_cell_count_color_range": [
                float(cell_count_limits[0]),
                float(cell_count_limits[1]),
            ],
            "context_cell_count_values_outside_range_share_endpoint_colors": True,
            "view_order": list(VIEW_ORDER),
            "view_figure_sizes_inches": {view_id: list(VIEW_FIGSIZE_INCHES[view_id]) for view_id in VIEW_ORDER},
        }
    )

    table_paths = [
        output_dir / "selected_moa_classes.csv",
        output_dir / "global_moa_retrieval.csv",
        output_dir / "cross_scaffold_moa_retrieval.csv",
        output_dir / "ranked_targets_and_moas.csv",
    ]
    moa_classes.to_csv(table_paths[0], index=False, float_format="%.17g")
    retrieval_compounds.sort_values("jcp2022").to_csv(table_paths[1], index=False, float_format="%.17g")
    retrieval_curve.sort_values("k").to_csv(table_paths[2], index=False, float_format="%.17g")
    rankings.sort_values(["annotation_source", "rank"], kind="stable").to_csv(
        table_paths[3], index=False, float_format="%.17g"
    )

    well_source_viewport = configuration["well_source_viewport"]
    if not isinstance(well_source_viewport, dict):
        raise TypeError("Well source viewport must be a mapping")
    # Captions come first: each is drawn under its own view and is that view's legend section.
    captions = build_captions(
        numbers=numbers,
        configuration=configuration,
        well_source_viewport=well_source_viewport,
        retrieval_curve=retrieval_curve,
        global_moa_score_upper=global_moa_score_upper,
    )
    if set(captions) != set(VIEW_ORDER):
        raise AssertionError("Every view must have exactly one caption")

    with plt.rc_context(PUBLICATION_TYPOGRAPHY | FIGURE_OUTPUT_RC):
        rendered_view_paths = {
            "global_moa_map": plot_global_moa_map(
                perturbations,
                retrieval_compounds,
                global_moa_score_upper,
                captions["global_moa_map"],
                output_dir,
                dpi,
            ),
            "selected_moa_neighborhoods": plot_selected_moa_neighborhoods(
                selected_moa_perturbations,
                captions["selected_moa_neighborhoods"],
                output_dir,
                dpi,
            ),
            "cross_scaffold_moa_retrieval": plot_cross_scaffold_moa_retrieval(
                retrieval_curve,
                captions["cross_scaffold_moa_retrieval"],
                output_dir,
                dpi,
            ),
            "profile_context_umaps": plot_profile_context_umaps(
                wells,
                perturbations,
                configuration["well_source_viewport"],
                captions["profile_context_umaps"],
                output_dir,
                dpi,
            ),
            "ranked_targets_and_moas": plot_ranked_targets_and_moas(
                rankings,
                captions["ranked_targets_and_moas"],
                output_dir,
                dpi,
            ),
            "combined_global_moa_retrieval": plot_combined_global_moa_retrieval(
                perturbations,
                retrieval_compounds,
                retrieval_curve,
                global_moa_score_upper,
                captions["combined_global_moa_retrieval"],
                output_dir,
                dpi,
            ),
        }
    view_paths = {view_id: rendered_view_paths[view_id] for view_id in VIEW_ORDER}
    legend_path = output_dir / "legend.md"
    write_legend(
        legend_path,
        figure_id=FIGURE_ID,
        title="Phenotypic activity and biological annotations in Cell Painting profile space",
        sections=[(f"{VIEW_STEMS[view_id]}.png", SECTION_TITLES[view_id], captions[view_id]) for view_id in VIEW_ORDER],
    )
    files = relative_file_records(output_dir, view_paths, table_paths, legend_path)
    summary = write_summary(
        output_dir / "summary.json",
        description=(
            "This figure directly connects canonical well- and perturbation-level Cell Painting "
            "embeddings, phenotypic activity and consistency results, compound annotations, and "
            "high-dimensional profiles to the publication views and companion tables. "
            "The selected_moa_min_normalized_average_precision configuration records the historical "
            "display-cohort cutoff; current per-class values are in selected_moa_classes.csv."
        ),
        numbers=numbers,
        files=files,
        configuration=configuration,
    )
    return summary


@app.command()
def main(
    well_umap: Annotated[Path, typer.Option("--well-umap", exists=True, dir_okay=False)] = DEFAULT_WELL_UMAP,
    perturbation_umap: Annotated[
        Path, typer.Option("--perturbation-umap", exists=True, dir_okay=False)
    ] = DEFAULT_PERTURBATION_UMAP,
    perturbation_profiles: Annotated[
        Path, typer.Option("--perturbation-profiles", exists=True, dir_okay=False)
    ] = DEFAULT_PERTURBATION_PROFILES,
    metadata_db: Annotated[Path, typer.Option("--metadata-db", exists=True, dir_okay=False)] = DEFAULT_METADATA_DB,
    copairs_db: Annotated[Path, typer.Option("--copairs-db", exists=True, dir_okay=False)] = DEFAULT_COPAIRS_DB,
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)] = (DEFAULT_OUTPUT_DIR),
    dataset: Annotated[str, typer.Option("--dataset")] = DEFAULT_DATASET,
    activity_preprocessing: Annotated[str, typer.Option("--activity-preprocessing")] = DEFAULT_ACTIVITY_PREPROCESSING,
    activity_filter: Annotated[str, typer.Option("--activity-filter")] = (DEFAULT_ACTIVITY_FILTER),
    metric: Annotated[str, typer.Option("--metric")] = DEFAULT_METRIC,
    seed: Annotated[int, typer.Option("--seed")] = MOA_RETRIEVAL_SEED,
    permutations: Annotated[int, typer.Option("--permutations", min=1)] = MOA_RETRIEVAL_PERMUTATIONS,
    max_k: Annotated[int, typer.Option("--max-k", min=10)] = MOA_RETRIEVAL_MAX_K,
    dpi: Annotated[int, typer.Option("--dpi", min=72)] = 150,
) -> None:
    """Build the phenotypic-consistency figure directly from its five canonical inputs."""
    tables, numbers, configuration = prepare_figure_data(
        well_umap=well_umap,
        perturbation_umap=perturbation_umap,
        perturbation_profiles=perturbation_profiles,
        metadata_db=metadata_db,
        copairs_db=copairs_db,
        dataset=dataset,
        activity_preprocessing=activity_preprocessing,
        activity_filter=activity_filter,
        metric=metric,
        permutations=permutations,
        seed=seed,
        max_k=max_k,
    )
    configuration["inputs"] = {
        "well_umap": str(well_umap),
        "perturbation_umap": str(perturbation_umap),
        "perturbation_profiles": str(perturbation_profiles),
        "metadata_db": str(metadata_db),
        "copairs_db": str(copairs_db),
    }
    write_figure_outputs(
        tables=tables,
        numbers=numbers,
        configuration=configuration,
        output_dir=output_dir,
        dpi=dpi,
    )
    logger.success(f"Wrote {FIGURE_ID} directly to {output_dir}")


if __name__ == "__main__":
    app()
