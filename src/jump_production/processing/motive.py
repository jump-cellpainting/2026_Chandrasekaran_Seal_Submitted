#!/usr/bin/env python
"""Process compound-gene annotations from the MOTIVE dataset.

MOTIVE integrates curated drug-target interaction data from 8 databases:
- BioKG, DGIdb, DrugRep, Hetionet, OpenBioLink, OpenTargets, PharmeBiNet, PrimeKG

The source data (motive_cpd_gene_annot.parquet) was processed using the
jump_compound_annotator library to match compounds to JUMP identifiers.

This script:
1. Loads curated annotations from data/external/motive_cpd_gene_annot.parquet
2. Matches to JUMP compounds via InChIKey (prefix matching for stereoisomer tolerance)
3. Aggregates genes by (JCP2022, rel_type, database) with pipe-separated gene lists
4. Outputs to data/interim/motive_annotations.csv for SQL import

The output format (one row per JCP2022/rel_type/database) enables simple SQL views
to filter by database and rel_type without complex aggregation.

Note: MOTIVE's 'drugrep' has ~30% fewer compounds than repurposing_hub_annotations
(both from Drug Repurposing Hub). Likely due to PubChem CID -> InChIKey lookup failures.
Use repurposing_hub_annotations for best Drug Repurposing Hub coverage.

Run directly:
    pixi run python -m jump_production.processing.motive
"""

from pathlib import Path

import duckdb
import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def load_jump_compounds(db_path: Path) -> pd.DataFrame:
    """Load JUMP compounds with InChIKey from metadata database."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        "SELECT Metadata_JCP2022, Metadata_InChIKey FROM compound WHERE Metadata_InChIKey IS NOT NULL"
    ).fetchdf()
    con.close()

    # Create InChIKey prefix for stereoisomer-tolerant matching
    df["inchikey_prefix"] = df["Metadata_InChIKey"].str[:14]
    return df


def load_annotations(input_file: Path) -> pd.DataFrame:
    """Load curated compound-gene annotations."""
    df = pd.read_parquet(input_file)

    # Drop internal index column if present
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    # Create InChIKey prefix for matching
    df["inchikey_prefix"] = df["inchikey"].str[:14]

    return df


def standardize_rel_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize relationship types to canonical names.

    Groups related relationship types for easier analysis.
    """
    # Map verbose/redundant rel_types to standardized names
    rel_type_mapping = {
        # Already standard
        "targets": "targets",
        "binds": "binds",
        "enzyme": "enzyme",
        "transports": "transports",
        "carries": "carries",
        "upregulates": "upregulates",
        "downregulates": "downregulates",
        "unknown": "unknown",
        # Inhibitors and blockers
        "inhibitor": "inhibitor",
        "blocker": "blocker",
        "antagonist": "antagonist",
        "inverse agonist": "inverse_agonist",
        # Activators and agonists
        "agonist": "agonist",
        "activator": "activator",
        "partial agonist": "partial_agonist",
        # Modulators
        "modulator": "modulator",
        "allosteric modulator": "allosteric_modulator",
        "positive modulator": "positive_modulator",
        "negative modulator": "negative_modulator",
        "inhibitory allosteric modulator": "inhibitory_allosteric_modulator",
        # Complex types - keep as-is but clean up
        "DRUG_CATALYSIS_GENE": "catalysis",
        "DRUG_REACTION_GENE": "reaction",
        "DRUG_INHIBITION_GENE": "inhibition",
        "DRUG_ACTIVATION_GENE": "activation",
        "DRUG_BINDACT_GENE": "binding_activation",
        "ASSOCIATES_CHaG": "associates",
        "IS_ACTIVE_ON_DNA_OR_RNA_LEVEL_CHiaodorlG": "dna_rna_active",
        "IS_ACTIVE_IN_METABOLISM_CHiaimG": "metabolism_active",
        "IS_ACTIVE_ON_CELLULAR_LEVEL_CHiaoclG": "cellular_active",
        "INCREASES_DEGENERATION_CHidG": "increases_degeneration",
        "DECREASES_DEGENERATION_CHddG": "decreases_degeneration",
    }

    # Apply mapping, keep original if not in mapping
    df["rel_type"] = df["rel_type"].map(lambda x: rel_type_mapping.get(x, x))

    # Clean up remaining edge cases (comma-separated, etc.)
    df["rel_type"] = df["rel_type"].str.replace(",", "_", regex=False)
    df["rel_type"] = df["rel_type"].str.lower()

    return df


def process_annotations(
    annotations: pd.DataFrame,
    jump_compounds: pd.DataFrame,
) -> pd.DataFrame:
    """Match annotations to JUMP compounds and aggregate genes per (JCP2022, rel_type, database).

    Returns DataFrame with:
    - Metadata_JCP2022: JUMP compound ID
    - Metadata_rel_type: Relationship type (targets, binds, etc.)
    - Metadata_database: Source database (single value per row)
    - Metadata_motive_gene: Pipe-separated gene symbols
    - Metadata_n_genes: Count of genes for this combination
    """
    # Merge annotations with JUMP compounds via InChIKey prefix
    merged = annotations.merge(
        jump_compounds[["Metadata_JCP2022", "inchikey_prefix"]],
        on="inchikey_prefix",
        how="inner",
    )

    logger.info(f"Matched {len(merged)} annotations to {merged['Metadata_JCP2022'].nunique()} JUMP compounds")

    # Standardize relationship types
    merged = standardize_rel_types(merged)

    # Aggregate genes by (JCP2022, rel_type, database)
    # Output: one row per (compound, rel_type, database) with pipe-joined genes
    aggregated = (
        merged.groupby(["Metadata_JCP2022", "rel_type", "database"])
        .agg(
            genes=("target", lambda x: "|".join(sorted(set(x)))),
            n_genes=("target", "nunique"),
        )
        .reset_index()
    )

    # Rename columns to match SQL schema
    aggregated = aggregated.rename(
        columns={
            "rel_type": "Metadata_rel_type",
            "database": "Metadata_database",
            "genes": "Metadata_motive_gene",
            "n_genes": "Metadata_n_genes",
        }
    )

    return aggregated


@app.command()
def main(
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to filtered_annotations.parquet (default: data/external/motive_cpd_gene_annot.parquet)",
    ),
    db_path: Path = typer.Option(
        None,
        "--db",
        "-d",
        help="Path to JUMP metadata database (default: data/interim/jump_metadata_augmented.duckdb)",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: data/interim/motive_annotations.csv)",
    ),
) -> int:
    """Process compound-gene annotations and return exit code."""
    logger.info("Processing compound-gene annotations from MOTIVE dataset...")

    # Set default paths
    if input_file is None:
        input_file = EXTERNAL_DATA_DIR / "motive_cpd_gene_annot.parquet"
    if db_path is None:
        db_path = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "motive_annotations.csv"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check input files exist
    if not input_file.exists():
        logger.error(f"Annotations file not found: {input_file}")
        logger.info("Expected: motive_cpd_gene_annot.parquet from MOTIVE dataset")
        return 1

    if not db_path.exists():
        logger.error(f"JUMP metadata database not found: {db_path}")
        return 1

    try:
        # Load data
        logger.info(f"Loading annotations from {input_file}")
        annotations = load_annotations(input_file)
        logger.info(f"Loaded {len(annotations)} annotations for {annotations['inchikey'].nunique()} compounds")

        logger.info(f"Loading JUMP compounds from {db_path}")
        jump_compounds = load_jump_compounds(db_path)
        logger.info(f"Loaded {len(jump_compounds)} JUMP compounds with InChIKey")

        # Process and match
        result = process_annotations(annotations, jump_compounds)

        # Log statistics
        logger.info(f"Generated {len(result)} (compound, rel_type, database) records")
        logger.info(f"Unique compounds: {result['Metadata_JCP2022'].nunique()}")
        logger.info(f"Unique databases: {result['Metadata_database'].nunique()}")
        logger.info(f"Relationship types: {result['Metadata_rel_type'].value_counts().head(10).to_dict()}")
        logger.info(f"Database counts: {result['Metadata_database'].value_counts().to_dict()}")

        # Save results
        result.to_csv(output_file, index=False)
        logger.success(f"Saved {len(result)} records to {output_file}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except duckdb.Error as e:
        logger.error(f"Database error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing annotations: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
