"""Build DuckDB from copairs results for unified querying.

Scans all copairs run directories, reads Hydra configs for metadata,
and builds a DuckDB with tables for both aggregated results and per-entity scores.

Tables and columns are documented via COMMENT ON statements in the DuckDB itself.
Query duckdb_tables() and duckdb_columns() to see descriptions.

Usage:
    pixi run python -m jump_production.processing.build_copairs_results_db
    pixi run python -m jump_production.processing.build_copairs_results_db --output data/processed/copairs_results.duckdb
"""

from pathlib import Path

import duckdb
import pandas as pd
import typer
from loguru import logger
from omegaconf import OmegaConf

from jump_production.config import PROCESSED_DATA_DIR

COPAIRS_RUNS_DIR = PROCESSED_DATA_DIR / "copairs/runs"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "copairs_results.duckdb"

app = typer.Typer()

# Columns that can have purely numeric values but should be strings
CSV_DTYPES = {
    "Metadata_Plate": "str",  # e.g., "1086292853" vs "UL001643"
    "Metadata_Batch": "str",  # Batch IDs may be numeric
}


def add_config_metadata(df: pd.DataFrame, config: OmegaConf, result_type: str) -> pd.DataFrame:
    """Add config metadata columns to DataFrame.

    Args:
        df: DataFrame to augment
        config: Hydra config object
        result_type: Type of result (activity, consistency)

    Returns:
        DataFrame with added metadata columns
    """
    df["_dataset"] = config.dataset.name
    df["_columns"] = config.columns.name
    df["_preprocessing"] = config.preprocessing_02_core.name
    df["_filter"] = config.filter.name

    # Extract activity_params from config (activity only)
    if hasattr(config, "activity_params"):
        df["_activity_params"] = config.activity_params.name

    # For consistency, add group info (target, disease, pathway, etc.)
    # Note: "target_column" is the Hydra config key name, but we store as _group_* for genericity
    if hasattr(config, "target_column"):
        df["_group_type"] = config.target_column.name
        df["_group_column"] = config.target_column.column
        # Normalize: extract group value from the type-specific column
        group_col = config.target_column.column
        if group_col in df.columns:
            df["group_value"] = df[group_col]

    # Extract distance metric from average_precision config (consistency only)
    if hasattr(config, "average_precision") and hasattr(config.average_precision, "params"):
        if "distance" in config.average_precision.params:
            df["_distance"] = config.average_precision.params.distance

    # Extract activity threshold from preprocessing config (consistency only)
    # Look for filter_active step in preprocessing_02_core.steps
    if hasattr(config, "preprocessing_02_core") and hasattr(config.preprocessing_02_core, "steps"):
        for step in config.preprocessing_02_core.steps:
            if step.get("type") == "filter_active":
                threshold = step.get("params", {}).get("threshold")
                if threshold is not None:
                    df["_activity_threshold"] = threshold
                break

    return df


def collect_results(result_type: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Collect all runs of a given result type with both results and scores.

    Args:
        result_type: Type of result (activity, consistency, etc.)

    Returns:
        Tuple of (map_results DataFrames, ap_scores DataFrames)
    """
    runs_dir = COPAIRS_RUNS_DIR / result_type
    if not runs_dir.exists():
        logger.warning(f"No {result_type} runs directory found: {runs_dir}")
        return [], []

    map_results_dfs = []
    ap_scores_dfs = []

    for config_path in runs_dir.glob("*/results/.hydra/config.yaml"):
        run_dir = config_path.parent.parent
        map_results_file = run_dir / f"{result_type}_map_results.csv"
        ap_scores_file = run_dir / f"{result_type}_ap_scores.csv"

        # Load Hydra config
        config = OmegaConf.load(config_path)
        run_name = run_dir.name

        # Load map_results (aggregated with p-values)
        if map_results_file.exists():
            df = pd.read_csv(map_results_file, dtype=CSV_DTYPES)
            df = add_config_metadata(df, config, result_type)

            # Compute n_perturbations from indices for consistency results
            if result_type == "consistency" and "indices" in df.columns:
                df["n_perturbations"] = df["indices"].apply(lambda x: 0 if x == "[]" else x.count(",") + 1)

            map_results_dfs.append(df)
            logger.info(f"Loaded {len(df):,} map_results rows from {run_name}")
        else:
            logger.warning(f"Missing map_results: {map_results_file}")

        # Load ap_scores (per-entity scores)
        if ap_scores_file.exists():
            df = pd.read_csv(ap_scores_file, dtype=CSV_DTYPES)
            df = add_config_metadata(df, config, result_type)
            ap_scores_dfs.append(df)
            logger.info(f"Loaded {len(df):,} ap_scores rows from {run_name}")
        else:
            logger.warning(f"Missing ap_scores: {ap_scores_file}")

    return map_results_dfs, ap_scores_dfs


def build_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    dfs: list[pd.DataFrame],
) -> None:
    """Build a table from collected DataFrames.

    Args:
        con: DuckDB connection
        table_name: Name of table to create
        dfs: List of DataFrames to combine
    """
    if not dfs:
        logger.warning(f"No data to build {table_name} table")
        return

    # Combine all DataFrames
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined):,} total rows for {table_name}")

    # Create table
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM combined")

    # Note: indexes removed — they trigger heap corruption in DuckDB 1.5.0
    # with mixed-type/nullable columns. DuckDB's columnar storage is fast
    # enough for this data size (~3M rows) without explicit indexes.

    # Add table and column comments
    add_comments(con, table_name)

    logger.success(f"Created {table_name} with {len(combined):,} rows")


# ============================================================================
# TABLE AND COLUMN COMMENTS
# ============================================================================

TABLE_COMMENTS = {
    "activity_scores": "Per-replicate Average Precision scores for phenotypic activity measurement",
    "activity_results": "Per-perturbation mean Average Precision (mAP) for phenotypic activity measurement",
    "consistency_scores": "Per-perturbation-per-group Average Precision scores for phenotypic consistency measurement",
    "consistency_results": "Per-group mean Average Precision (mAP) for phenotypic consistency measurement. Contains both regular runs and threshold sweep runs. Filter with: _preprocessing NOT LIKE '%_sweep' for regular runs only, or _preprocessing LIKE '%_sweep' for sweep runs only.",
}

# Column comments shared across tables
COLUMN_COMMENTS = {
    # Config metadata columns (from Hydra)
    "_dataset": "Dataset config name (e.g., 'compound_no_source7')",
    "_columns": "Feature columns config name (e.g., 'feat_all')",
    "_preprocessing": "Preprocessing config name (e.g., 'activity_no_target2')",
    "_filter": "Filter config name (e.g., 'all_sources', 'source_2')",
    "_activity_params": "Activity parameters config (e.g., 'default', 'withinsource', 'crosssource')",
    "_distance": "Distance metric for similarity computation (e.g., 'cosine', 'abs_cosine')",
    "_activity_threshold": "P-value threshold used for activity filtering. To distinguish sweep vs regular runs, check _preprocessing: names ending in '_sweep' are threshold sweep runs.",
    # Group columns (consistency only)
    "_group_type": "Group type config name (e.g., 'repurposing', 'uniprot', 'chemical_probes')",
    "_group_column": "Original column name in source data (e.g., 'Metadata_repurposing_target')",
    "group_value": "Normalized group value extracted from the group-specific column",
    # Copairs output columns (from mean_average_precision in map.py)
    "Metadata_JCP2022": "JUMP perturbation ID",
    "mean_average_precision": "Mean AP score for each group",
    "mean_normalized_average_precision": "Mean normalized AP score (scale-independent)",
    "p_value": "P-value comparing mAP to the null distribution",
    "corrected_p_value": "Adjusted p-value after multiple testing correction (FDR, Benjamini-Hochberg)",
    "below_p": "True if p_value < threshold",
    "below_corrected_p": "True if corrected_p_value < threshold (use this for significance)",
    "-log10(p-value)": "Negative log10 of p-value for visualization",
    "indices": "List of row indices used in the mAP aggregation",
    # Copairs output columns (from average_precision in average_precision.py)
    "average_precision": "Calculated average precision score for each profile",
    "normalized_average_precision": "Normalized AP score (scale-independent)",
    "n_pos_pairs": "Number of positive pairs for each profile",
    "n_total_pairs": "Total number of pairs (positive + negative) for each profile",
    # Computed columns
    "n_perturbations": "Number of perturbations in this group (= n_pos_pairs + 1 from scores)",
}


def add_comments(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Add COMMENT ON TABLE and COMMENT ON COLUMN statements."""
    # Table comment
    if table_name in TABLE_COMMENTS:
        comment = TABLE_COMMENTS[table_name].replace("'", "''")
        con.execute(f"COMMENT ON TABLE {table_name} IS '{comment}'")

    # Column comments
    columns = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    for col in columns:
        if col in COLUMN_COMMENTS:
            comment = COLUMN_COMMENTS[col].replace("'", "''")
            # Quote column names with special characters
            col_quoted = f'"{col}"' if not col.isidentifier() else col
            con.execute(f"COMMENT ON COLUMN {table_name}.{col_quoted} IS '{comment}'")


@app.command()
def main(
    output: Path = typer.Option(
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Output DuckDB file path",
    ),
) -> None:
    """Build copairs results DuckDB from all completed runs.

    Scans activity/ and consistency/ run directories, reads Hydra configs,
    and creates tables for both aggregated results and per-entity scores.
    """
    # Discover result types from directory structure
    result_types = []
    if COPAIRS_RUNS_DIR.exists():
        result_types = [d.name for d in COPAIRS_RUNS_DIR.iterdir() if d.is_dir()]

    if not result_types:
        logger.error(f"No result types found in {COPAIRS_RUNS_DIR}")
        raise typer.Exit(1)

    logger.info(f"Found result types: {result_types}")

    # Remove existing DB
    output.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output))

    # Build tables for each result type
    for result_type in result_types:
        logger.info(f"Processing {result_type} runs...")
        map_results_dfs, ap_scores_dfs = collect_results(result_type)

        if map_results_dfs:
            build_table(con, f"{result_type}_results", map_results_dfs)

        if ap_scores_dfs:
            build_table(con, f"{result_type}_scores", ap_scores_dfs)

    # Show summary
    tables = con.execute("SHOW TABLES").fetchall()
    logger.info("Database summary:")
    for (table_name,) in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        cols = con.execute(f"SELECT COUNT(*) FROM duckdb_columns() WHERE table_name = '{table_name}'").fetchone()[0]
        logger.info(f"  {table_name}: {count:,} rows x {cols} columns")

    con.close()
    logger.success(f"Wrote {output}")


if __name__ == "__main__":
    app()
