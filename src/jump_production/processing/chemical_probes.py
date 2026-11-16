#!/usr/bin/env python
"""Process Chemical Probes data and standardize SMILES.

This script processes the Probe & Drugs Portal chemical probes data,
standardizes SMILES strings, and generates InChI keys for compound matching.

This script can be run directly:
    # Full processing with SMILES standardization (slow):
    python jump_production/processing/chemical_probes.py

    # Quick processing without SMILES standardization (for testing):
    python jump_production/processing/chemical_probes.py --skip-standardization
"""

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import (
    load_jump_compounds,
    validate_inputs,
)
from jump_production.processing.utils import (
    standardize_smiles as standardize_smiles_util,
)

app = typer.Typer()


def process_targets_and_merge(df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    """Process target information and merge with compounds data.

    This is the fast part - aggregating targets and merging.

    Args:
        df: Compounds dataframe
        targets_df: Targets dataframe

    Returns:
        df with target_genes column added
    """
    logger.info("Aggregating target genes for each probe...")

    # Group targets by pdid and aggregate gene names
    # First, flatten any comma-separated genes within entries
    def flatten_and_dedupe_genes(gene_series):
        """Flatten comma-separated genes, deduplicate, and sort.

        Note: "-" entries represent uncharacterized proteins (e.g., "Uncharacterized protein FLJ45252").
        We remove these because for target similarity analysis, "-" would incorrectly match
        between compounds that have different uncharacterized proteins as targets.
        """
        all_genes = set()
        for gene_entry in gene_series.dropna():
            # Split on commas to handle entries like "ADORA1,ADORA2A,ADORA2B"
            genes = [g.strip() for g in gene_entry.split(",")]
            all_genes.update(genes)
        # Remove empty strings and "-" (uncharacterized proteins)
        all_genes.discard("")
        all_genes.discard("-")  # Remove to avoid false matches in target similarity
        return "|".join(sorted(all_genes)) if all_genes else None

    target_aggregation = (
        targets_df.groupby("pdid")["gene_name"].apply(flatten_and_dedupe_genes).reset_index(name="target_genes")
    )

    # Merge target genes with main compounds data
    df = df.merge(target_aggregation, on="pdid", how="left")

    # Rename to include Metadata_chmprb_ prefix for chemical probes table
    df = df.rename(columns={"target_genes": "Metadata_chmprb_target_genes"})

    # Count how many have targets
    probes_with_targets = df[df["probe"] == 1]["Metadata_chmprb_target_genes"].notna().sum()
    total_probes = (df["probe"] == 1).sum()
    logger.info(f"  {probes_with_targets}/{total_probes} probes have target gene annotations")

    return df


def standardize_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize SMILES and generate InChI keys.

    This is the slow part - can be skipped for testing.

    Args:
        df: DataFrame with 'smiles' column

    Returns:
        df with standardized_smiles and standardized_inchikey columns added
    """
    # Use shared utility for SMILES standardization
    return standardize_smiles_util(df, "smiles", source_name="chemical probes")


def deduplicate_by_inchikey(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate probes that standardize to same InChIKey.

    When multiple probes map to the same InChIKey:
    1. Merge target genes (union)
    2. Keep probe with best quality metrics
    3. Track which pdids were merged

    Args:
        df: DataFrame with standardized_inchikey column

    Returns:
        Deduplicated DataFrame
    """
    # Check for duplicates
    dup_mask = df["standardized_inchikey"].duplicated(keep=False)
    n_dups = dup_mask.sum()

    if n_dups == 0:
        logger.info("No duplicate InChIKeys found after standardization")
        return df

    logger.warning(f"Found {n_dups} probes with duplicate InChIKeys")

    # Group by standardized_inchikey
    deduped_rows = []
    merged_count = 0

    for inchikey, group in df.groupby("standardized_inchikey"):
        if len(group) > 1:
            merged_count += 1

            # Merge target genes (union of all targets)
            all_targets = set()
            for _, row in group.iterrows():
                if pd.notna(row["Metadata_chmprb_target_genes"]):
                    # Targets are already flattened and pipe-separated
                    all_targets.update(row["Metadata_chmprb_target_genes"].split("|"))

            # Remove empty strings and "-" (uncharacterized proteins), deduplicate, and sort
            all_targets.discard("")
            all_targets.discard("-")  # Remove to avoid false matches in target similarity
            merged_targets = "|".join(sorted(all_targets)) if all_targets else None

            # Pick best row based on quality metrics
            # Priority: P&D approved > experimental probe > highest QED
            # Note: Using original column names since cleaning happens after dedup
            best_row = (
                group.sort_values(["P&D approved", "experimental probe", "qed"], ascending=False, na_position="last")
                .iloc[0]
                .copy()
            )

            # Update with merged data
            best_row["Metadata_chmprb_target_genes"] = merged_targets
            best_row["no. targets"] = len(all_targets) if all_targets else 0

            # Track merge (store other pdids in a new column)
            other_pdids = group[group["pdid"] != best_row["pdid"]]["pdid"].tolist()
            if other_pdids:
                best_row["merged_from_pdids"] = "|".join(other_pdids)

            logger.info(
                f"Merged {len(group)} probes with InChIKey {inchikey}: keeping {best_row['pdid']} ({best_row['name']})"
            )

            deduped_rows.append(best_row)
        else:
            # No duplicates for this InChIKey
            deduped_rows.append(group.iloc[0])

    result_df = pd.DataFrame(deduped_rows).reset_index(drop=True)
    logger.success(f"Deduplication complete: {len(df)} → {len(result_df)} probes ({merged_count} merges)")

    return result_df


@app.command()
def main(
    skip_standardization: bool = typer.Option(
        False,
        "--skip-standardization",
        help="Skip SMILES standardization (use original InChIKeys for faster testing)",
    ),
) -> int:
    """Process chemical probes data and return exit code."""
    logger.info("Processing chemical probes data...")
    if skip_standardization:
        logger.info("⚡ FAST MODE: Skipping SMILES standardization")

    # Ensure output directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define input/output paths
    input_file = EXTERNAL_DATA_DIR / "pd_export_01_2025_875_targets_standardized.xlsx"
    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    output_probes_path = INTERIM_DATA_DIR / "chemical_probes_processed.csv"
    output_targets_path = INTERIM_DATA_DIR / "chemical_probes_targets.csv"

    # Validate input files
    if not validate_inputs(
        input_file,
        compound_file,
        names=["Chemical probes Excel file", "JUMP compound metadata"],
    ):
        return 1

    try:
        # Load chemical probes data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_excel(input_file, sheet_name="COMPOUNDS")
        targets_df = pd.read_excel(input_file, sheet_name="TARGETS")
        logger.info(f"Loaded {len(df)} chemical probes and {len(targets_df)} probe-target pairs")

        # Load JUMP compound metadata for InChIKey to JCP2022 mapping using shared utility
        jump_compounds = load_jump_compounds(compound_file)

        # STEP 1: Process targets (fast)
        df = process_targets_and_merge(df, targets_df)

        # STEP 2: Standardize SMILES (slow - can be skipped)
        if skip_standardization:
            # Use original InChIKeys for testing
            logger.info("Using original InChIKeys (not standardized)")
            df["standardized_smiles"] = df["smiles"]
            df["standardized_inchikey"] = df["inchikey"]
        else:
            df = standardize_smiles(df)

        # Process probe information - filter for actual chemical probes
        probes_df = df[df["probe"] == 1].copy()
        logger.info(f"Found {len(probes_df)} chemical probes")

        # STEP 3: Deduplicate probes with same InChIKey after standardization
        probes_df = deduplicate_by_inchikey(probes_df)

        # STEP 4: Match to JUMP compounds via InChIKey
        logger.info("Matching chemical probes to JUMP compounds via InChIKey...")
        probes_df = probes_df.merge(
            jump_compounds, left_on="standardized_inchikey", right_on="Metadata_InChIKey", how="left"
        )

        # Count matches
        matched_probes = probes_df["Metadata_JCP2022"].notna().sum()
        logger.info(f"  {matched_probes}/{len(probes_df)} chemical probes matched to JUMP compounds")

        # Drop the redundant InChIKey columns from merge (InChIKey14 is only for matching)
        probes_df = probes_df.drop(columns=["Metadata_InChIKey", "Metadata_InChIKey14"], errors="ignore")

        # STEP 5: Clean up columns and add Metadata_chmprb_ prefix
        # First rename columns to use consistent lowercase with underscores
        probes_df = probes_df.rename(
            columns={
                "experimental probe": "experimental_probe",
                "calculated probe": "calculated_probe",
                "approved drug": "approved_drug",
                "P&D approved": "pd_approved",
                "covalent binder": "covalent_binder",
                "biased GPCR ligand": "biased_gpcr_ligand",
                "structural alert": "structural_alert",
                "PAINS Family A": "pains_family_a",
                "PAINS Family B": "pains_family_b",
                "PAINS Family C": "pains_family_c",
                "Drug Status": "drug_status",
                "no. targets": "num_targets",
                "PROTAC": "protac",
                "Aggregator": "aggregator",
                "Obsolete": "obsolete",
                "Nuisance": "nuisance",
            }
        )

        # Add Metadata_chmprb_ prefix to all columns except Metadata_JCP2022 and Metadata_chmprb_target_genes
        columns_to_rename = {}
        for col in probes_df.columns:
            if col == "Metadata_JCP2022":
                # Keep universal JUMP identifier as is
                continue
            elif col == "Metadata_chmprb_target_genes":
                # Already has the correct prefix
                continue
            elif col == "Metadata_InChIKey":
                # Drop this redundant column from the merge
                continue
            else:
                # Add prefix to all other columns
                columns_to_rename[col] = f"Metadata_chmprb_{col}"

        probes_df = probes_df.rename(columns=columns_to_rename)

        # Convert binary flags to proper integers (0/1) instead of floats
        binary_columns = [
            "Metadata_chmprb_probe",
            "Metadata_chmprb_experimental_probe",
            "Metadata_chmprb_calculated_probe",
            "Metadata_chmprb_available",
            "Metadata_chmprb_approved_drug",
            "Metadata_chmprb_pd_approved",
            "Metadata_chmprb_protac",
            "Metadata_chmprb_covalent_binder",
            "Metadata_chmprb_biased_gpcr_ligand",
            "Metadata_chmprb_inorganic",
            "Metadata_chmprb_structural_alert",
            "Metadata_chmprb_pains_family_a",
            "Metadata_chmprb_pains_family_b",
            "Metadata_chmprb_pains_family_c",
            "Metadata_chmprb_aggregator",
            "Metadata_chmprb_obsolete",
            "Metadata_chmprb_nuisance",
        ]

        for col in binary_columns:
            if col in probes_df.columns:
                # Convert to integer, treating NaN as 0
                probes_df[col] = probes_df[col].fillna(0).astype(int)

        # Add computed quality flags with Metadata_chmprb_ prefix
        probes_df["Metadata_chmprb_has_quality_alerts"] = (
            (probes_df.get("Metadata_chmprb_structural_alert", 0) > 0)
            | (probes_df.get("Metadata_chmprb_pains_family_a", 0) > 0)
            | (probes_df.get("Metadata_chmprb_pains_family_b", 0) > 0)
            | (probes_df.get("Metadata_chmprb_pains_family_c", 0) > 0)
            | (probes_df.get("Metadata_chmprb_aggregator", 0) > 0)
            | (probes_df.get("Metadata_chmprb_obsolete", 0) > 0)
            | (probes_df.get("Metadata_chmprb_nuisance", 0) > 0)
        ).astype(int)

        probes_df["Metadata_chmprb_is_high_quality"] = (
            (probes_df["Metadata_chmprb_experimental_probe"] == 1)
            & (probes_df["Metadata_chmprb_pd_approved"] == 1)
            & (probes_df["Metadata_chmprb_qed"] >= 0.5)
            & (probes_df["Metadata_chmprb_has_quality_alerts"] == 0)
        ).astype(int)

        # Add flag for whether probe is in JUMP compound library
        probes_df["Metadata_chmprb_in_jump"] = probes_df["Metadata_JCP2022"].notna().astype(int)

        # Reorganize columns: put Metadata_JCP2022 first if it exists
        cols = list(probes_df.columns)
        if "Metadata_JCP2022" in cols:
            cols.remove("Metadata_JCP2022")
            cols = ["Metadata_JCP2022"] + cols
            probes_df = probes_df[cols]

        # Save main processed probes file with target genes
        probes_df.to_csv(output_probes_path, index=False)
        logger.success(f"Saved {len(probes_df)} chemical probes to {output_probes_path}")

        # Save detailed targets file (filter to only probes that remain after deduplication)
        # Also include pdids that were merged into other probes
        remaining_pdids = set(probes_df["Metadata_chmprb_pdid"])
        # Add pdids that were merged (if any)
        if "Metadata_chmprb_merged_from_pdids" in probes_df.columns:
            remaining_pdids.update(probes_df["Metadata_chmprb_merged_from_pdids"].dropna().str.split("|").explode())

        targets_filtered = targets_df[targets_df["pdid"].isin(remaining_pdids)].copy()

        # Select relevant columns for targets file and add Metadata_chmprb_ prefix
        targets_output = targets_filtered[
            ["pdid", "name", "gene_name", "target_name", "target_type", "moa", "activity_biochemical"]
        ].rename(
            columns={
                "pdid": "Metadata_chmprb_pdid",
                "name": "Metadata_chmprb_probe_name",
                "gene_name": "Metadata_chmprb_gene_name",
                "target_name": "Metadata_chmprb_target_name",
                "target_type": "Metadata_chmprb_target_type",
                "moa": "Metadata_chmprb_moa",
                "activity_biochemical": "Metadata_chmprb_activity_biochemical",
            }
        )

        targets_output.to_csv(output_targets_path, index=False)
        logger.success(f"Saved {len(targets_output)} probe-target pairs to {output_targets_path}")

        # Summary statistics
        logger.info("\nChemical probes summary:")
        logger.info(f"  Total probes processed: {len(probes_df)}")
        logger.info(f"  Total probe-target pairs: {len(targets_output)}")
        logger.info(f"  Unique target genes: {targets_df['gene_name'].nunique()}")

        # Distribution of targets per probe
        target_counts = probes_df["Metadata_chmprb_target_genes"].str.count(r"\|").fillna(-1) + 1
        logger.info(f"  Single-target probes: {(target_counts == 1).sum()}")
        logger.info(f"  Multi-target probes (2-5): {((target_counts >= 2) & (target_counts <= 5)).sum()}")
        logger.info(f"  Promiscuous probes (>5): {(target_counts > 5).sum()}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except ValueError as e:
        logger.error(f"SMILES processing error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing chemical probes data: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
