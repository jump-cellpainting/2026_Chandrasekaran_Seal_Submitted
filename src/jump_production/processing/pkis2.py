#!/usr/bin/env python
"""Ingest the PKIS2 kinase binding matrix and match it to JUMP compounds.

Background
----------
PKIS2 (Drewry et al. 2017) is the second-generation Published Kinase Inhibitor
Set: ~640 compounds profiled for % inhibition against ~470 kinases at 1 µM. Unlike
KCGS and PKIS (membership-only, ingested into the `kinase_probes` table), PKIS2's
value is the full per-compound × per-kinase binding matrix, so it is stored as a
tidy long table rather than a membership flag.

This follows the same ingest-then-query convention as the other `process_*` rules
(see `kinase_probes.py`): standardize SMILES, match to JUMP via the InChIKey
connectivity layer (first 14 chars), then write an interim long table keyed by
`JCP2022`. The augmentation step prefixes those columns and ingests the table as
`pkis2_binding`, which notebook 2.44 and downstream substrate exporters query directly.

Data source
-----------
PKIS2: Drewry et al. (2017) PLoS ONE, DOI: 10.1371/journal.pone.0181585.
Expected input: data/external/pkis2.xlsx with columns
  Compound, Smiles, Regno, Chemotype, >90, >80, >70, and one column per kinase
  (each holding % inhibition at 1 µM).

Run directly:
    pixi run -e cheminformatics python -m jump_production.processing.pkis2 --fast
    pixi run -e jump-smiles python -m jump_production.processing.pkis2   # max matches
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import (
    generate_inchikeys_fast,
    load_jump_compounds,
    match_to_jump,
    standardize_smiles,
    validate_inputs,
)

app = typer.Typer()

# Non-kinase columns in pkis2.xlsx (after renaming Compound→probe_id, Smiles→smiles);
# every other column is a kinase % inhibition column.
META_COLS = {"Regno", "probe_id", "Chemotype", "smiles", ">90", ">80", ">70"}


def load_pkis2(input_file: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load PKIS2 compounds and identify the kinase columns.

    Returns
    -------
    df : DataFrame with probe_id, smiles, and the kinase % inhibition columns
    kinase_cols : names of the kinase columns
    """
    logger.info(f"Loading PKIS2 from {input_file}")
    df = pd.read_excel(input_file)
    df = df.rename(columns={"Compound": "probe_id", "Smiles": "smiles"})
    kinase_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} PKIS2 compounds × {len(kinase_cols)} kinases")
    return df, kinase_cols


@app.command()
def main(
    pkis2_file: Path = typer.Option(None, "--pkis2", help="Path to pkis2.xlsx (default: data/external/pkis2.xlsx)"),
    compound_file: Path = typer.Option(
        None, "--compounds", help="Path to JUMP compound metadata (default: data/external/compound.csv.gz)"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Output parquet (default: data/interim/pkis2_binding.parquet)"
    ),
    num_cpu: int = typer.Option(4, "--num-cpu", "-j", help="CPUs for SMILES standardization"),
    fast: bool = typer.Option(
        False, "--fast", help="Fast mode: RDKit InChIKeys instead of jump_canonical (~fewer matches)"
    ),
) -> int:
    """Ingest PKIS2 binding matrix, match to JUMP, and write a tidy long table."""
    if pkis2_file is None:
        pkis2_file = EXTERNAL_DATA_DIR / "pkis2.xlsx"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "pkis2_binding.parquet"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not validate_inputs(pkis2_file, compound_file, names=["PKIS2 Excel file", "JUMP compound metadata"]):
        return 1

    try:
        jump_compounds = load_jump_compounds(compound_file)
        df, kinase_cols = load_pkis2(pkis2_file)

        # Standardize SMILES → InChIKey, then match to JUMP via connectivity layer
        if fast:
            df = generate_inchikeys_fast(df, "smiles", source_name="PKIS2")
        else:
            df = standardize_smiles(df, "smiles", num_cpus=num_cpu, source_name="PKIS2")
        df = df[df["standardized_inchikey"].notna()].copy()

        matched = match_to_jump(df, jump_compounds, inchikey_col="standardized_inchikey", how="inner")
        matched = matched.drop_duplicates(subset=["probe_id", "Metadata_JCP2022"]).reset_index(drop=True)
        logger.info(
            f"Matched {matched['probe_id'].nunique()} PKIS2 compounds to "
            f"{matched['Metadata_JCP2022'].nunique()} JUMP compounds"
        )

        # Melt kinase columns to tidy long format
        long = matched.melt(
            id_vars=["Metadata_JCP2022", "probe_id"],
            value_vars=kinase_cols,
            var_name="kinase",
            value_name="pct_inhibition",
        )
        long = long.rename(columns={"Metadata_JCP2022": "JCP2022"})
        long["pct_inhibition"] = pd.to_numeric(long["pct_inhibition"], errors="coerce")
        long = long.dropna(subset=["pct_inhibition"]).reset_index(drop=True)

        long.to_parquet(output_file, index=False)
        logger.success(
            f"Saved {len(long):,} (compound, kinase) binding rows for "
            f"{long['JCP2022'].nunique()} JUMP compounds to {output_file}"
        )
        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing PKIS2: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
