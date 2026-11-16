"""Filter profiles to include only source_7-exclusive compounds.

Creates a new profile parquet that includes all non-source_7 wells plus
source_7 wells for compounds that were profiled ONLY at source_7 (no
cross-site replicates). This avoids dose confounds since source_7 used
0.625 µM while other sources used 10 µM.

Usage:
    pixi run python -m jump_production.processing.filter_source7_exclusive \
        --input data/raw/profiles/compound.parquet \
        --output data/raw/profiles/compound_with_source7.parquet \
        --db data/interim/jump_metadata_augmented.duckdb
"""

from pathlib import Path

import duckdb
import pandas as pd
import typer
from loguru import logger

app = typer.Typer()

# Canonical metadata columns expected by copairs-runner (joined from DuckDB at runtime).
# Extra metadata in the parquet causes conflicts with the DB join.
CANONICAL_METADATA = {"Metadata_JCP2022", "Metadata_Plate", "Metadata_Source", "Metadata_Well"}


def get_source7_exclusive_compounds(db_path: Path) -> set[str]:
    """Query DuckDB for compounds profiled only at source_7."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        result = con.sql("""
            SELECT DISTINCT s.Metadata_JCP2022
            FROM (SELECT DISTINCT Metadata_JCP2022 FROM well WHERE Metadata_Source = 'source_7') s
            LEFT JOIN (SELECT DISTINCT Metadata_JCP2022 FROM well WHERE Metadata_Source != 'source_7') o
                USING (Metadata_JCP2022)
            WHERE o.Metadata_JCP2022 IS NULL
        """).fetchdf()
    return set(result["Metadata_JCP2022"])


@app.command()
def main(
    input_file: Path = typer.Option(..., "--input", help="Full profile parquet (with source_7)"),
    output: Path = typer.Option(..., help="Output parquet path"),
    db: Path = typer.Option(..., help="Augmented metadata DuckDB path"),
) -> int:
    """Filter profile to non-source_7 wells + source_7-exclusive compounds."""

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise typer.Exit(1)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        raise typer.Exit(1)

    exclusive = get_source7_exclusive_compounds(db)
    logger.info(f"Found {len(exclusive):,} source_7-exclusive compounds in database")

    df = pd.read_parquet(input_file)
    is_source7 = df["Metadata_Source"] == "source_7"
    is_exclusive = df["Metadata_JCP2022"].isin(exclusive)

    # Keep: all non-source_7 wells + source_7 wells for exclusive compounds
    df_filtered = df[~is_source7 | is_exclusive]

    # Standardize metadata columns to canonical set
    extra_meta = [c for c in df_filtered.columns if c.startswith("Metadata_") and c not in CANONICAL_METADATA]
    if extra_meta:
        df_filtered = df_filtered.drop(columns=extra_meta)
        logger.info(f"Dropped {len(extra_meta)} extra metadata columns: {extra_meta}")

    logger.info(f"Input: {len(df):,} rows")
    logger.info(f"Source_7 wells kept (exclusive): {(is_source7 & is_exclusive).sum():,}")
    logger.info(f"Source_7 wells dropped (non-exclusive): {(is_source7 & ~is_exclusive).sum():,}")
    logger.info(f"Output: {len(df_filtered):,} rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output, index=False)
    logger.success(f"Saved to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(app())
