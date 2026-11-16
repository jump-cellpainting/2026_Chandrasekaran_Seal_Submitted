#!/usr/bin/env python
"""Process MitoTox mitochondrial toxicity annotations for JUMP compound matching.

MitoTox (https://www.mitotox.org/) provides binary toxicity labels and functional
mechanism annotations for compounds with mitochondrial toxicity evidence. Compounds
are classified as toxic if any experimental record shows a positive result across
8 top-level functional categories (F01-F08).

Matching to JUMP uses InChIKey14 prefix (stereoisomer-tolerant).

Run directly:
    pixi run -e cheminformatics python -m jump_production.processing.mitotox
"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import (
    generate_inchikeys_fast,
    load_jump_compounds,
    match_to_jump,
    validate_inputs,
)

app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to input parquet (default: data/external/mitotox/mitotox_compounds.parquet)",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: data/interim/mitotox_processed.csv)",
    ),
) -> int:
    """Process MitoTox annotations and return exit code."""
    logger.info("Processing MitoTox mitochondrial toxicity annotations...")

    if input_file is None:
        input_file = EXTERNAL_DATA_DIR / "mitotox" / "mitotox_compounds.parquet"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "mitotox_processed.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if not validate_inputs(input_file, compound_file, names=["MitoTox parquet", "JUMP compound metadata"]):
        return 1

    try:
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} MitoTox compounds")

        # Generate InChIKeys from SMILES
        df = generate_inchikeys_fast(df, smiles_col="smiles", source_name="MitoTox")

        # Deduplicate source data on InChIKey14 before matching — stereoisomers
        # (e.g. Doxorubicin/Epirubicin) share connectivity but may differ in labels
        df = df.dropna(subset=["standardized_inchikey"])
        df["_ik14"] = df["standardized_inchikey"].str[:14]
        n_before = len(df)
        df = df.drop_duplicates(subset=["_ik14"], keep="first").drop(columns=["_ik14"])
        logger.info(f"Deduplicated InChIKey14: {n_before} -> {len(df)}")

        # Match to JUMP compounds via InChIKey14 prefix
        jump_compounds = load_jump_compounds(compound_file)
        matched = match_to_jump(df, jump_compounds, inchikey_col="standardized_inchikey")

        # Build output with Metadata_ prefix convention
        result = pd.DataFrame(
            {
                "Metadata_JCP2022": matched["Metadata_JCP2022"],
                "Metadata_mitotox_label": matched["mitotox_label"],
                "Metadata_mitotox_toxic": (matched["mitotox_label"] == "toxic").astype(int),
                "Metadata_mitotox_mechanisms": matched["functional_mechanism"],
            }
        )

        # One row per JUMP compound (multiple source compounds may map to same JCP2022)
        result = result.drop_duplicates(subset=["Metadata_JCP2022"], keep="first")

        result.to_csv(output_file, index=False)

        toxic_count = result["Metadata_mitotox_toxic"].sum()
        logger.success(
            f"Saved {len(result)} compounds ({toxic_count} toxic, "
            f"{len(result) - toxic_count} non-toxic) to {output_file}"
        )
        return 0

    except Exception as e:
        logger.exception(f"Failed to process MitoTox data: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
