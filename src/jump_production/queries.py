"""DuckDB query functions for copairs results database.

Provides query functions for activity and consistency results from copairs_results.duckdb.
"""

import duckdb
import pandas as pd


def query_activity_results(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
    activity_params: str = "default",
    columns: str = "feat_all",
) -> pd.DataFrame:
    """Query activity results from copairs_results.duckdb.

    Args:
        con: DuckDB connection to copairs_results.duckdb
        dataset: Dataset name (e.g., 'compound_no_source7')
        preprocessing: Preprocessing config name
        filter_name: Filter config name
        activity_params: Activity params config (default, withinsource, crosssource)
        columns: Feature-column config name

    Returns:
        DataFrame with activity results for the specified parameters
    """
    query = """
        SELECT
            Metadata_JCP2022,
            mean_average_precision,
            mean_normalized_average_precision,
            p_value,
            corrected_p_value,
            below_p,
            below_corrected_p
        FROM activity_results
        WHERE _dataset = ?
          AND _columns = ?
          AND _preprocessing = ?
          AND _filter = ?
          AND _activity_params = ?
        ORDER BY mean_normalized_average_precision DESC
    """
    return con.execute(
        query,
        [dataset, columns, preprocessing, filter_name, activity_params],
    ).df()


def query_cross_source_reproducibility(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str,
    filter_name: str = "all_sources",
) -> pd.DataFrame:
    """Query within-source vs cross-source activity for reproducibility analysis.

    Joins within-source and cross-source activity results to compare how
    reproducible compound phenotypes are across different lab sites.

    NOTE: This is NOT "consistency" (shared biology clustering). This measures
    reproducibility of the same compound across sources.

    Args:
        con: DuckDB connection to copairs_results.duckdb
        dataset: Dataset name (e.g., "compound_no_source7", "compound_DL_CPCNN_no_source7")
        preprocessing: Preprocessing config (e.g., "activity_only_target2", "activity_poscon_only")
        filter_name: Filter config (e.g., "all_sources", "no_source9")

    Returns:
        DataFrame with columns:
        - Metadata_JCP2022, Metadata_Source: compound and source identifiers
        - nmAP_within, nmAP_cross: normalized mAP for within/cross source
        - p_within, p_cross: corrected p-values
        - sig_within, sig_cross: boolean significance flags (p < 0.1)
    """
    # Use mean_normalized_average_precision (not raw mAP) because:
    # - Within-source has ~25 positive pairs, cross-source has ~122
    # - Raw mAP depends on n_pos_pairs, normalized mAP is scale-independent
    # - Normalized mAP = (AP - mu_0) / (1 - mu_0), where mu_0 = expected AP under random
    # Column naming: nmAP = normalized mAP (to distinguish from raw mAP)
    query = """
        WITH within_src AS (
            SELECT
                Metadata_JCP2022,
                Metadata_Source,
                mean_normalized_average_precision as nmAP_within,
                corrected_p_value as p_within,
                below_corrected_p as sig_within
            FROM activity_results
            WHERE _dataset = ?
              AND _activity_params = 'withinsource'
              AND _preprocessing = ?
              AND _filter = ?
        ),
        cross_src AS (
            SELECT
                Metadata_JCP2022,
                Metadata_Source,
                mean_normalized_average_precision as nmAP_cross,
                corrected_p_value as p_cross,
                below_corrected_p as sig_cross
            FROM activity_results
            WHERE _dataset = ?
              AND _activity_params = 'crosssource'
              AND _preprocessing = ?
              AND _filter = ?
        )
        SELECT
            w.Metadata_JCP2022,
            w.Metadata_Source,
            w.nmAP_within,
            w.p_within,
            w.sig_within,
            c.nmAP_cross,
            c.p_cross,
            c.sig_cross
        FROM within_src w
        JOIN cross_src c ON w.Metadata_JCP2022 = c.Metadata_JCP2022
                        AND w.Metadata_Source = c.Metadata_Source
        ORDER BY w.Metadata_JCP2022, w.Metadata_Source
    """
    return con.execute(query, [dataset, preprocessing, filter_name, dataset, preprocessing, filter_name]).df()


def query_consistency_results(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str = "consistency_no_target2",
    filter_name: str = "all_sources",
    group_type: str = "repurposing",
    distance: str = "cosine",
    columns: str = "feat_all",
    activity_threshold: float | None = None,
) -> pd.DataFrame:
    """Query consistency results from copairs_results.duckdb.

    Note: This function excludes threshold sweep runs (preprocessing names ending in '_sweep')
    and returns only regular consistency runs. For sweep analysis, query the table
    directly with appropriate _preprocessing filtering.

    Args:
        con: DuckDB connection to copairs_results.duckdb
        dataset: Dataset name (e.g., 'compound_no_source7')
        preprocessing: Preprocessing config name (should be consistency_*)
        filter_name: Filter config name
        group_type: Target grouping type (e.g., 'repurposing', 'uniprot', 'moa')
        distance: Distance metric (cosine, abs_cosine)
        columns: Feature-column config name
        activity_threshold: Optional upstream activity threshold used to select
            compounds. ``None`` preserves the prior behavior and does not filter
            this configuration field.

    Returns:
        DataFrame with consistency results for the specified parameters
    """
    threshold_clause = ""
    parameters: list[str | float] = [
        dataset,
        columns,
        preprocessing,
        filter_name,
        group_type,
        distance,
    ]
    if activity_threshold is not None:
        threshold_clause = "AND _activity_threshold = ?"
        parameters.append(activity_threshold)
    query = f"""
        SELECT
            group_value,
            mean_average_precision,
            mean_normalized_average_precision,
            p_value,
            corrected_p_value,
            below_p,
            below_corrected_p,
            n_perturbations
        FROM consistency_results
        WHERE _dataset = ?
          AND _columns = ?
          AND _preprocessing = ?
          AND _filter = ?
          AND _group_type = ?
          AND _distance = ?
          {threshold_clause}
          AND _preprocessing NOT LIKE '%_sweep'
        ORDER BY mean_normalized_average_precision DESC
    """
    return con.execute(query, parameters).df()


def query_consistency_scores(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    preprocessing: str = "consistency_no_target2",
    filter_name: str = "all_sources",
    group_type: str = "repurposing",
    distance: str = "cosine",
    columns: str = "feat_all",
    activity_threshold: float | None = None,
) -> pd.DataFrame:
    """Query per-compound consistency scores from one exact regular run.

    A compound can appear in more than one annotation group and source. The
    returned rows therefore preserve the compound-by-group membership grain.
    As with :func:`query_consistency_results`, threshold-sweep runs are
    excluded explicitly.

    Args:
        con: DuckDB connection to copairs_results.duckdb
        dataset: Dataset name (e.g., 'compound_no_source7')
        preprocessing: Preprocessing config name (should be consistency_*)
        filter_name: Filter config name
        group_type: Target grouping type (e.g., 'repurposing', 'uniprot', 'moa')
        distance: Distance metric (cosine, abs_cosine)
        columns: Feature-column config name
        activity_threshold: Optional upstream activity threshold used to select
            compounds. ``None`` does not filter this configuration field.

    Returns:
        DataFrame with per-compound AP values and annotation-group membership
    """
    threshold_clause = ""
    parameters: list[str | float] = [
        dataset,
        columns,
        preprocessing,
        filter_name,
        group_type,
        distance,
    ]
    if activity_threshold is not None:
        threshold_clause = "AND _activity_threshold = ?"
        parameters.append(activity_threshold)
    query = f"""
        SELECT
            Metadata_JCP2022,
            group_value,
            average_precision,
            normalized_average_precision,
            n_pos_pairs,
            n_total_pairs
        FROM consistency_scores
        WHERE _dataset = ?
          AND _columns = ?
          AND _preprocessing = ?
          AND _filter = ?
          AND _group_type = ?
          AND _distance = ?
          {threshold_clause}
          AND _preprocessing NOT LIKE '%_sweep'
        ORDER BY group_value, Metadata_JCP2022
    """
    return con.execute(query, parameters).df()
