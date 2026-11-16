#!/usr/bin/env python
"""Process ChEMBL protein targets from wide to long format with pipe-delimited targets.

This script can be run directly:
    python -m jump_production.processing.chembl
"""

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import validate_inputs

app = typer.Typer()


@app.command()
def main() -> int:
    """Process ChEMBL protein targets and return exit code."""
    logger.info("Processing ChEMBL protein targets...")

    # Ensure output directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define input/output paths
    input_file = EXTERNAL_DATA_DIR / "targetannotations_singleproteins_jumpcompounds_all_5.csv"
    output_path = INTERIM_DATA_DIR / "chembl_protein_targets_processed.csv"

    # Validate inputs
    if not validate_inputs(input_file, names=["ChEMBL protein targets file"]):
        return 1

    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        chembl_annotations_df = pd.read_csv(input_file).set_index("JUMP_ID").drop(columns=["Standardized_SMILES"])

        # Transform from wide to long format
        chembl_annotations_df = (
            chembl_annotations_df.stack()
            .reset_index()
            .rename(columns={0: "connection", "level_1": "gene", "JUMP_ID": "Metadata_JCP2022"})
            .query("connection != 0")
            .drop(columns=["connection"])
        )

        # Group by compound and create pipe-delimited target list
        chembl_annotations_df = (
            chembl_annotations_df.groupby("Metadata_JCP2022")
            .apply(lambda x: "|".join(pd.Series(x["gene"]).unique()), include_groups=False)
            .reset_index()
            .rename(columns={0: "Metadata_Uniprot_target"})
        )

        # Save results
        chembl_annotations_df.to_csv(output_path, index=False)
        logger.success(f"Saved {len(chembl_annotations_df)} compound-target mappings to {output_path}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse input file: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing ChEMBL protein targets: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
