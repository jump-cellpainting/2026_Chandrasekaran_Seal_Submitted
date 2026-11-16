#!/usr/bin/env python
"""Process EPA functional use data for JUMP compounds.

Cleans the EPA functional use CSV into a typed interim table for the augmented DB.
Source: Phillips et al. 2017 (PMID 30505234), a QSUR model predicting the probability
that a chemical serves each of 35 functional uses; data provided by Josh Harrill
(EPA/Corteva). Two kinds of labels: `functioncategory_all` is the curated use category
(from EPA CPDat), and the 35 numeric columns are predicted use probabilities (0-1).

The raw CSV encodes missing values as the literal string 'NA' in both the curated
category and the probability columns, which forces the numeric columns to object dtype.
This converts 'NA' to null, coerces the probabilities to float, and renames columns to
the Metadata_funcuse_* convention. The file is already keyed on JCP2022 (mapped upstream
via DTXSID), so no structure matching happens here; the JUMP-validity filter is applied
downstream in augment_metadata_db.sql, consistent with the other annotation sources.

Run directly:
    pixi run python -m jump_production.processing.functional_use
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

# 35 predicted functional-use probability columns (QSUR, Phillips et al. 2017), in the
# order they appear in the source CSV and the functional_use table definition.
FUP_COLS = [
    "antimicrobial", "heat_stabilizer", "catalyst", "colorant", "skin_conditioner",
    "uv_absorber", "buffer", "flavorant", "skin_protectant", "antioxidant", "fragrance",
    "flame_retardant", "oxidizer", "photoinitiator", "crosslinker", "preservative",
    "organic_pigment", "chelator", "emollient", "hair_conditioner", "humectant",
    "hair_dye", "reducer", "surfactant", "additive", "monomer", "lubricating_agent",
    "foamer", "emulsifier", "emulsion_stabilizer", "whitener", "soluble_dye",
    "antistatic_agent", "foam_boosting_agent", "vinyl",
]

DEFAULT_INPUT = EXTERNAL_DATA_DIR / "2026-04-06_JUMP_chems_func_use.csv"
DEFAULT_OUTPUT = INTERIM_DATA_DIR / "functional_use_processed.parquet"


@app.command()
def main(
    input_file: Path = typer.Option(None, "--input", "-i", help=f"Input CSV (default: {DEFAULT_INPUT})"),
    output_file: Path = typer.Option(None, "--output", "-o", help=f"Output parquet (default: {DEFAULT_OUTPUT})"),
) -> int:
    """Clean the EPA functional use CSV and write a typed interim parquet."""
    if input_file is None:
        input_file = DEFAULT_INPUT
    if output_file is None:
        output_file = DEFAULT_OUTPUT
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info(f"Loading EPA functional use data from {input_file}")
    df = pd.read_csv(input_file)

    # 'NA' is the source's missing-value token in both the category and probability columns
    df["functioncategory_all"] = df["functioncategory_all"].replace("NA", pd.NA)
    for col in FUP_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df[["JCP2022", "DTXSID", "functioncategory_all", *FUP_COLS]].rename(
        columns={
            "JCP2022": "Metadata_JCP2022",
            "DTXSID": "Metadata_funcuse_dtxsid",
            "functioncategory_all": "Metadata_funcuse_category",
            **{c: f"Metadata_funcuse_{c}" for c in FUP_COLS},
        }
    )

    out.to_parquet(output_file, index=False)
    n_fuc = out["Metadata_funcuse_category"].notna().sum()
    n_fup = out[[f"Metadata_funcuse_{c}" for c in FUP_COLS]].notna().any(axis=1).sum()
    logger.success(
        f"Saved {len(out):,} compounds ({n_fuc:,} with curated FUC, {n_fup:,} with FUP predictions) "
        f"to {output_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
