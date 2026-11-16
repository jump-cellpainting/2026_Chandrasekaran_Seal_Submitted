#!/usr/bin/env python
"""Map Drug Repurposing Hub annotations to JUMP IDs via InChIKey14.

Extracts clinical_phase, moa, target, disease_area, and indication annotations.

This script can be run directly:
    python -m jump_production.processing.repurposing_hub
"""

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import load_jump_compounds, validate_inputs

app = typer.Typer()


@app.command()
def main() -> int:
    """Process Drug Repurposing Hub annotations and return exit code."""
    logger.info("Processing Drug Repurposing Hub annotations...")

    # Ensure output directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define input/output paths
    samples_file = EXTERNAL_DATA_DIR / "repurposing_samples_20200324.txt"
    drugs_file = EXTERNAL_DATA_DIR / "repurposing_drugs_20200324.txt"
    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    output_path = INTERIM_DATA_DIR / "repurposing_hub_annotations_processed.tsv"

    # Validate inputs
    if not validate_inputs(
        samples_file,
        drugs_file,
        compound_file,
        names=["Repurposing Hub samples", "Repurposing Hub drugs", "JUMP compounds"],
    ):
        return 1

    try:
        # Load repurposing hub samples
        logger.info(f"Loading samples from {samples_file}")
        rephub_samples_df = (
            pd.read_csv(
                samples_file,
                sep="\t",
                comment="!",
                usecols=["InChIKey", "pert_iname"],
            )
            .assign(InChIKey14=lambda x: x["InChIKey"].str[:14])
            .drop_duplicates(subset=["pert_iname", "InChIKey14"])
            .drop(columns="InChIKey")
            .rename(columns={"InChIKey14": "Metadata_InChIKey14"})
        ).reset_index(drop=True)

        # Load repurposing hub drugs
        logger.info(f"Loading drugs from {drugs_file}")
        rephub_drugs_df = (
            pd.read_csv(
                drugs_file,
                sep="\t",
                comment="!",
                usecols=[
                    "pert_iname",
                    "clinical_phase",
                    "moa",
                    "target",
                    "disease_area",
                    "indication",
                ],
            )
            .drop_duplicates(subset="pert_iname")
            .rename(
                columns={
                    "clinical_phase": "Metadata_repurposing_clinical_phase",
                    "moa": "Metadata_repurposing_moa",
                    "target": "Metadata_repurposing_target",
                    "disease_area": "Metadata_repurposing_disease_area",
                    "indication": "Metadata_repurposing_indication",
                }
            )
            .reset_index(drop=True)
        )

        # Merge samples and drugs (keep pert_iname as compound name)
        logger.info("Merging samples and drugs...")
        rephub_merged_df = rephub_samples_df.merge(rephub_drugs_df, on="pert_iname", how="inner").rename(
            columns={"pert_iname": "Metadata_repurposing_name"}
        )

        # Load JUMP compound metadata using shared utility
        compound_metadata_df = load_jump_compounds(compound_file)

        # Map to JUMP IDs via InChIKey14
        logger.info("Mapping to JUMP IDs...")
        jcpid_target_df = (
            compound_metadata_df.merge(rephub_merged_df, on="Metadata_InChIKey14")
            .drop(columns=["Metadata_InChIKey", "Metadata_InChIKey14"])
            .dropna(subset=["Metadata_repurposing_target"])
            .reset_index(drop=True)
        )

        # Save results
        jcpid_target_df.to_csv(output_path, index=False, sep="\t")
        logger.success(f"Saved {len(jcpid_target_df)} compound annotations to {output_path}")
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
        logger.exception(f"Unexpected error processing Drug Repurposing Hub: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
