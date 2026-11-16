# -*- coding: utf-8 -*-
# Common helper functions for Snakemake rules
# Separated from rule files per snakemake lint recommendations

from jump_production.config import DEFAULT_DPI

# =============================================================================
# Configuration from snakemake config (copairs matrix)
# =============================================================================


def expand_consistency_matrix(matrix_config):
    """Expand dimensional consistency config into flat list.

    Supports two formats:
    1. Flat dict (current): {dataset, columns, ...} - returned wrapped in list
    2. Matrix format: {datasets: [...], targets: [{column, table}, ...], ...}
       - Expands as outer product of datasets × targets × preprocessings × distances
    """
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]

    results = []
    for item in matrix_config:
        # Flat format: has 'dataset' (singular) key
        if "dataset" in item:
            results.append(item)
            continue

        # Matrix format: has 'datasets' (plural) key
        datasets = item["datasets"]
        columns = item.get("columns", "feat_all")
        targets = item["targets"]
        preprocessings = item.get("preprocessings", [{}])
        distances = item.get("distances", ["cosine"])

        for dataset in datasets:
            for target in targets:
                for preproc in preprocessings:
                    for distance in distances:
                        results.append(
                            {
                                "dataset": dataset,
                                "columns": columns,
                                "preprocessing_02_core_cons": preproc.get("cons"),
                                "preprocessing_02_core_act": preproc.get("act"),
                                "filter": preproc.get("filter"),
                                "target_column": target["column"],
                                "target_table": target["table"],
                                "distance": distance,
                            }
                        )
    return results


def expand_activity_matrix(matrix_config):
    """Expand dimensional activity config into flat list.

    Supports two formats:
    1. Flat dict: {dataset, columns, ...} - returned as-is
    2. Matrix format: {datasets: [...], preprocessings: [...], ...}
       - Expands as outer product of datasets × preprocessings
    """
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]

    results = []
    for item in matrix_config:
        # Flat format: has 'dataset' (singular) key
        if "dataset" in item:
            results.append(item)
            continue

        # Matrix format: has 'datasets' (plural) key
        datasets = item["datasets"]
        columns = item.get("columns", "feat_all")
        preprocessings = item["preprocessings"]

        for dataset in datasets:
            for preproc in preprocessings:
                results.append({
                    "dataset": dataset,
                    "columns": columns,
                    "preprocessing_02_core": preproc["preprocessing"],
                    "filter": preproc["filter"],
                })
    return results


def expand_consistency_prefiltered_matrix(matrix_config):
    """Expand dimensional prefiltered consistency config into flat list.

    Matrix format: {datasets: [...], targets: [...], ...}
    Expands as outer product of datasets × targets
    """
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]

    results = []
    for item in matrix_config:
        # Flat format: has 'dataset' (singular) key
        if "dataset" in item:
            results.append(item)
            continue

        # Matrix format: has 'datasets' (plural) key
        datasets = item["datasets"]
        columns = item.get("columns", "feat_all")
        targets = item["targets"]
        preprocessing = item.get("preprocessing_02_core_cons", "consistency_prefiltered")
        filter_val = item.get("filter", "all_sources")
        distance = item.get("distance", "cosine")

        for dataset in datasets:
            for target in targets:
                results.append({
                    "dataset": dataset,
                    "columns": columns,
                    "preprocessing_02_core_cons": preprocessing,
                    "filter": filter_val,
                    "target_column": target,
                    "distance": distance,
                })
    return results


def expand_consistency_sweep_matrix(matrix_config):
    """Expand consistency_sweep config with threshold dimension.

    Similar to expand_consistency_matrix but adds threshold iteration.
    Used for Issue #22 threshold sensitivity analysis.
    """
    if not isinstance(matrix_config, list):
        matrix_config = [matrix_config]

    results = []
    for item in matrix_config:
        datasets = item["datasets"]
        columns = item.get("columns", "feat_all")
        targets = item["targets"]
        preprocessings = item.get("preprocessings", [{}])
        distances = item.get("distances", ["cosine"])
        thresholds = item["thresholds"]  # Required for sweep

        for dataset in datasets:
            for target in targets:
                for preproc in preprocessings:
                    for distance in distances:
                        for threshold in thresholds:
                            results.append({
                                "dataset": dataset,
                                "columns": columns,
                                "preprocessing_02_core_cons": preproc.get("cons"),
                                "preprocessing_02_core_act": preproc.get("act"),
                                "filter": preproc.get("filter"),
                                "target_column": target["column"],
                                "target_table": target["table"],
                                "distance": distance,
                                "activity_threshold": threshold,
                            })
    return results


ACTIVITY = config["activity"]
CONSISTENCY = expand_consistency_matrix(config["consistency"])
CONSISTENCY_PREFILTERED = expand_consistency_prefiltered_matrix(config.get("consistency_prefiltered", []))
CONSISTENCY_SWEEP = expand_consistency_sweep_matrix(config.get("consistency_sweep", []))
ACTIVITY_WITHINSOURCE = expand_activity_matrix(config.get("activity_withinsource", []))
ACTIVITY_CROSSSOURCE = expand_activity_matrix(config.get("activity_crosssource", []))

# =============================================================================
# Downstream analysis configuration (from analysis: section in YAML)
# =============================================================================
# Read from config with backwards-compatible defaults
ANALYSIS_CONFIG = config.get("analysis", {})

# Default dataset and activity config for downstream analyses
DEFAULT_DATASET = ANALYSIS_CONFIG.get("default_dataset", "compound_no_source7")
DEFAULT_COLUMNS = ANALYSIS_CONFIG.get("default_columns", "feat_all")
DEFAULT_PREPROCESSING = ANALYSIS_CONFIG.get("default_preprocessing", "activity_no_target2")
DEFAULT_FILTER = ANALYSIS_CONFIG.get("default_filter", "all_sources")
DEFAULT_ACTIVITY_PARAMS = ANALYSIS_CONFIG.get("default_activity_params", "default")

# Profile comparison configuration
COMPARISON_PROFILES_ACTIVITY = [
    tuple(p) for p in ANALYSIS_CONFIG.get("comparison_profiles_activity",
        [["compound_no_source7", "compound_DL_CPCNN_no_source7"]])
]
COMPARISON_PROFILES_CONSISTENCY = [
    tuple(p) for p in ANALYSIS_CONFIG.get("comparison_profiles_consistency",
        [["compound_no_source7_active_union", "compound_DL_CPCNN_no_source7_active_union"]])
]

# Target consistency visualization configuration
TARGET_CONSISTENCY_DATASETS = ANALYSIS_CONFIG.get("target_consistency_datasets",
    ["compound_no_source7", "compound_DL_CPCNN_no_source7"])

# Cross-source reproducibility analysis configuration
CROSS_SOURCE_COMBOS = [
    tuple(c) for c in ANALYSIS_CONFIG.get("cross_source_combos", [
        ["activity_only_target2", "all_sources", "activity_only_target2"],
        ["activity_only_target2", "no_source9", "activity_only_target2_no_source9"],
        ["activity_poscon_only", "all_sources", "activity_poscon_only"],
    ])
]


# =============================================================================
# Copairs helper functions
# =============================================================================
def act_name(it):
    """Generate activity run name from config item."""
    return f"{it['dataset']}__{it['columns']}__{it['preprocessing_02_core']}__{it['filter']}"


def activity_path(it, activity_params="default"):
    """Build activity output path for a given activity params config."""
    return f"data/processed/copairs/runs/activity/{act_name(it)}__{activity_params}/results/activity_map_results.csv"


def cons_name(it):
    """Generate consistency run name from config item."""
    return f"{it['dataset']}__{it['columns']}__{it['preprocessing_02_core_cons']}__{it['filter']}__{it['target_column']}__{it['distance']}"


def cons_prefiltered_name(it):
    """Name for prefiltered consistency runs (same structure, explicit function for clarity)."""
    return f"{it['dataset']}__{it['columns']}__{it['preprocessing_02_core_cons']}__{it['filter']}__{it['target_column']}__{it['distance']}"


def cons_sweep_name(it):
    """Generate consistency sweep run name including threshold."""
    return (f"{it['dataset']}__{it['columns']}__{it['preprocessing_02_core_cons']}__"
            f"{it['filter']}__{it['target_column']}__{it['distance']}__{it['activity_threshold']}")


def get_consistency_params(wildcards, config_list=None):
    """Look up config item from config_list (default: CONSISTENCY) based on wildcards."""
    if config_list is None:
        config_list = CONSISTENCY

    for item in config_list:
        if (item['dataset'] == wildcards.dataset and
            item['columns'] == wildcards.columns and
            item['preprocessing_02_core_cons'] == wildcards.preprocessing_02_core_cons and
            item['filter'] == wildcards.filter and
            item['target_column'] == wildcards.target_column and
            item['distance'] == wildcards.distance):
            # For sweep runs, also match threshold
            if 'activity_threshold' in item:
                if str(item['activity_threshold']) != getattr(wildcards, 'activity_threshold', None):
                    continue
            return item
    return None


def get_activity_input(wildcards, config_list=None):
    """Return activity file path for consistency run, or empty list if not found."""
    params = get_consistency_params(wildcards, config_list)
    if params is None:
        return []
    return f"data/processed/copairs/runs/activity/{wildcards.dataset}__{wildcards.columns}__{params['preprocessing_02_core_act']}__{wildcards.filter}__default/results/activity_map_results.csv"


# =============================================================================
# UMAP helper functions
# =============================================================================
def get_umap_combinations():
    """Generate valid (dataset, metric, filter) combinations for UMAP.

    Reads from analysis.umap_combinations in config with defaults:
    - compound_no_source7: cosine × (all, active), euclidean × all
    - compound_DL_CPCNN_no_source7: cosine × all only
    """
    default = [
        ["compound_no_source7", "cosine", "all"],
        ["compound_no_source7", "cosine", "active"],
        ["compound_no_source7", "euclidean", "all"],
        ["compound_DL_CPCNN_no_source7", "cosine", "all"],
        # Source_7-inclusive (for visual batch effect assessment)
        ["compound_with_source7", "cosine", "all"],
        ["compound_DL_CPCNN_with_source7", "cosine", "all"],
    ]
    return [tuple(c) for c in ANALYSIS_CONFIG.get("umap_combinations", default)]


UMAP_COMBOS = get_umap_combinations()


def umap_compute_inputs(wildcards):
    """Return inputs for umap_compute, adding copairs_db dependency for active filtering.

    Note: metadata_db is NOT required here - UMAP computation is metadata-independent.
    Metadata is joined at visualization time (umap_plot, target_consistency rules).
    """
    inputs = {
        "h5ad_well": f"data/interim/anndata/{wildcards.dataset}.h5ad",
        "h5ad_perturbation": f"data/interim/anndata/{wildcards.dataset}_perturbation.h5ad",
    }
    if wildcards.filter == "active":
        inputs["copairs_db"] = ancient("data/processed/copairs_results.duckdb")
    return inputs


# =============================================================================
# Analysis helper functions
# =============================================================================
def get_target_consistency_combinations():
    """Generate valid (dataset, group_type, umap_filter, distance) combinations.

    Reads from CONSISTENCY config to determine which dataset/group_type/distance
    tuples have consistency results. Only generates combinations that exist.
    Only include active filter for datasets that have active UMAP built.
    """
    valid_tuples = {
        (item["dataset"], item["target_column"], item["distance"])
        for item in CONSISTENCY
        if item["dataset"] in TARGET_CONSISTENCY_DATASETS
    }

    combos = []
    active_umap_datasets = {d for d, m, f in UMAP_COMBOS if f == "active" and m == "cosine"}
    for dataset, group_type, distance in valid_tuples:
        combos.append((dataset, group_type, "all", distance))
        if dataset in active_umap_datasets:
            combos.append((dataset, group_type, "active", distance))
    return combos


TARGET_CONSISTENCY_COMBOS = get_target_consistency_combinations()
