"""Create pre-filtered parquet files containing union of active perturbations.

This enables comprehensive comparison of feature extraction methods by creating
datasets that contain all perturbations active in either method.

Usage:
    pixi run python -m jump_production.processing.create_union_profiles \
        --activity1 data/processed/copairs/runs/activity/compound_no_source7__feat_all__activity_no_target2__all_sources/results/activity_map_results.csv \
        --activity2 data/processed/copairs/runs/activity/compound_DL_CPCNN_no_source7__feat_all__activity_no_target2__all_sources/results/activity_map_results.csv \
        --parquet1 data/raw/profiles/compound_no_source7.parquet \
        --parquet2 data/raw/profiles/compound_DL_CPCNN_no_source7.parquet \
        --output-dir data/raw/profiles \
        --suffix active_union
"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    activity1: Path = typer.Option(..., help="Activity results CSV for first profile"),
    activity2: Path = typer.Option(..., help="Activity results CSV for second profile"),
    parquet1: Path = typer.Option(..., help="Parquet file for first profile"),
    parquet2: Path = typer.Option(..., help="Parquet file for second profile"),
    output_dir: Path = typer.Option(..., help="Output directory for filtered parquets"),
    suffix: str = typer.Option("active_union", help="Suffix for output filenames"),
    on_column: str = typer.Option("Metadata_JCP2022", help="Column to match on"),
    filter_column: str = typer.Option("below_corrected_p", help="Activity flag column"),
) -> int:
    """Create parquet files filtered to union of active perturbations."""

    # Load activity results and get active sets
    act1 = pd.read_csv(activity1)
    act2 = pd.read_csv(activity2)

    active1 = set(act1[act1[filter_column]][on_column].unique())
    active2 = set(act2[act2[filter_column]][on_column].unique())

    logger.info(f"{parquet1.stem}: {len(active1):,} active perturbations")
    logger.info(f"{parquet2.stem}: {len(active2):,} active perturbations")

    # Compute union (all perturbations active in either method)
    union = active1 | active2
    logger.info(f"Union: {len(union):,} perturbations")
    logger.info(f"  Intersection (both): {len(active1 & active2):,}")
    logger.info(f"  Only {parquet1.stem}: {len(active1 - active2):,}")
    logger.info(f"  Only {parquet2.stem}: {len(active2 - active1):,}")

    # Filter and save each parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in [parquet1, parquet2]:
        df = pd.read_parquet(parquet_path)
        original_rows = len(df)

        # Filter to union
        df_filtered = df[df[on_column].isin(union)]

        # Output path: compound_no_source7.parquet -> compound_no_source7_active_union.parquet
        output_path = output_dir / f"{parquet_path.stem}_{suffix}.parquet"
        df_filtered.to_parquet(output_path, index=False)

        logger.info(
            f"{parquet_path.stem}: {original_rows:,} -> {len(df_filtered):,} rows "
            f"({len(df_filtered) / original_rows * 100:.1f}%) -> {output_path.name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(app())
