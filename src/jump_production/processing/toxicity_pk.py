#!/usr/bin/env python
"""Process toxicity and pharmacokinetic annotations for JUMP compound matching.

Integrates three SMILES-based datasets:
- DILI (Drug-Induced Liver Injury): binary classification from DILIrank v2
  Source: https://pubmed.ncbi.nlm.nih.gov/41005561/
- DICT (Drug-Induced CardioToxicity): binary classification from DICTrank
  Source: https://srijitseal.com/DICTrank_Predictor/datasets.html
- PK (Pharmacokinetic parameters): regression values from PKSmart
  Source: https://srijitseal.com/PKSmart/datasets.html

All datasets use SMILES for compound identification, matched to JUMP via
jump_smiles standardization and InChIKey14 prefix matching.

Run directly:
    pixi run -e jump-smiles python -m jump_production.processing.toxicity_pk
"""

import zipfile
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from jump_production.processing.utils import (
    generate_inchikeys_fast,
    load_jump_compounds,
    standardize_smiles,
    validate_inputs,
)

app = typer.Typer()


def process_dili(
    zip_path: Path | None = None,
    csv_path: Path | None = None,
    num_cpus: int | None = None,
    fast: bool = False,
) -> pd.DataFrame:
    """Process DILI (Drug-Induced Liver Injury) annotations.

    DILIrank v2 provides concern levels for drug-induced liver injury.

    Args:
        zip_path: Path to zip file containing DILIrankv2_smiles.csv
        csv_path: Direct path to CSV (alternative to zip)
        num_cpus: Number of CPUs for standardization
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized SMILES and DILI concern level
    """
    if zip_path is not None:
        logger.info(f"Loading DILI data from {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if "DILIrank" in f and f.endswith(".csv")]
            if not csv_files:
                raise ValueError("Could not find DILIrankv2_smiles.csv in zip")
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    elif csv_path is not None:
        logger.info(f"Loading DILI data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("Either zip_path or csv_path must be provided")
    logger.info(f"Loaded {len(df)} DILI records")

    # Generate InChIKeys using shared utilities
    if fast:
        df = generate_inchikeys_fast(df, "standardized_smiles", source_name="DILI")
    else:
        df = standardize_smiles(df, "standardized_smiles", num_cpus=num_cpus, source_name="DILI")

    # Map concern levels to numeric values for easier filtering
    # Original values: vMOST-DILI-concern, vMost-DILI-concern, vLess-DILI-concern,
    #                  vNo-DILI-concern, Ambiguous-DILI-concern
    # Note: vMOST and vMost appear to be the same (case inconsistency in source)
    concern_map = {
        "vMOST-DILI-concern": "most",
        "vMost-DILI-concern": "most",
        "vLess-DILI-concern": "less",
        "vNo-DILI-concern": "no",
        "Ambiguous-DILI-concern": "ambiguous",
    }
    df["Metadata_dili_concern"] = df["vDILI-Concern"].map(concern_map)

    # Add binary classification (most = positive, others = negative/excluded)
    df["Metadata_dili_positive"] = (df["Metadata_dili_concern"] == "most").astype(int)

    # Keep only relevant columns
    result = df[["standardized_inchikey", "Metadata_dili_concern", "Metadata_dili_positive"]].copy()
    result = result.dropna(subset=["standardized_inchikey"])

    logger.info(f"DILI concern distribution:\n{df['Metadata_dili_concern'].value_counts()}")

    return result


def process_dict(
    zip_path: Path | None = None,
    csv_path: Path | None = None,
    num_cpus: int | None = None,
    fast: bool = False,
) -> pd.DataFrame:
    """Process DICT (Drug-Induced CardioToxicity) annotations.

    DICTrank provides concern levels for drug-induced cardiotoxicity.

    Args:
        zip_path: Path to zip file containing DICTrank_smiles.csv
        csv_path: Direct path to CSV (alternative to zip)
        num_cpus: Number of CPUs for standardization
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized SMILES and DICT concern level
    """
    if zip_path is not None:
        logger.info(f"Loading DICT data from {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if "DICTrank" in f and f.endswith(".csv")]
            if not csv_files:
                raise ValueError("Could not find DICTrank_smiles.csv in zip")
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    elif csv_path is not None:
        logger.info(f"Loading DICT data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("Either zip_path or csv_path must be provided")
    logger.info(f"Loaded {len(df)} DICT records")

    # Generate InChIKeys using shared utilities
    if fast:
        df = generate_inchikeys_fast(df, "Standardized_SMILES", source_name="DICT")
    else:
        df = standardize_smiles(df, "Standardized_SMILES", num_cpus=num_cpus, source_name="DICT")

    # Map concern levels (already lowercase: most, less, no, ambiguous)
    df["Metadata_dict_concern"] = df["DICT _ Concern"].str.lower().str.strip()

    # Add binary classification (most = positive, others = negative/excluded)
    df["Metadata_dict_positive"] = (df["Metadata_dict_concern"] == "most").astype(int)

    # Keep only relevant columns
    result = df[["standardized_inchikey", "Metadata_dict_concern", "Metadata_dict_positive"]].copy()
    result = result.dropna(subset=["standardized_inchikey"])

    logger.info(f"DICT concern distribution:\n{df['Metadata_dict_concern'].value_counts()}")

    return result


def process_pk(
    zip_path: Path | None = None,
    csv_path: Path | None = None,
    num_cpus: int | None = None,
    fast: bool = False,
) -> pd.DataFrame:
    """Process PK (Pharmacokinetic) annotations.

    PKSmart provides human pharmacokinetic parameters:
    - VDss: Volume of distribution at steady state (L/kg)
    - CL: Clearance (mL/min/kg)
    - fup: Fraction unbound in plasma
    - MRT: Mean residence time (h)
    - t1/2: Half-life (h)

    Args:
        zip_path: Path to zip file containing Human_PK_data.csv
        csv_path: Direct path to CSV (alternative to zip)
        num_cpus: Number of CPUs for standardization
        fast: Use RDKit directly instead of jump_smiles (faster but fewer matches)

    Returns:
        DataFrame with standardized SMILES and PK parameters
    """
    if zip_path is not None:
        logger.info(f"Loading PK data from {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if "Human_PK" in f and f.endswith(".csv")]
            if not csv_files:
                raise ValueError("Could not find Human_PK_data.csv in zip")
            with zf.open(csv_files[0]) as f:
                df = pd.read_csv(f)
    elif csv_path is not None:
        logger.info(f"Loading PK data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("Either zip_path or csv_path must be provided")
    logger.info(f"Loaded {len(df)} PK records")

    # Generate InChIKeys using shared utilities
    if fast:
        df = generate_inchikeys_fast(df, "smiles_r", source_name="PK")
    else:
        df = standardize_smiles(df, "smiles_r", num_cpus=num_cpus, source_name="PK")

    # Rename columns with Metadata_pk_ prefix
    result = df[
        [
            "standardized_inchikey",
            "human_VDss_L_kg",
            "human_CL_mL_min_kg",
            "human_fup",
            "human_mrt",
            "human_thalf",
        ]
    ].copy()

    result = result.rename(
        columns={
            "human_VDss_L_kg": "Metadata_pk_vdss_l_kg",
            "human_CL_mL_min_kg": "Metadata_pk_cl_ml_min_kg",
            "human_fup": "Metadata_pk_fup",
            "human_mrt": "Metadata_pk_mrt_h",
            "human_thalf": "Metadata_pk_thalf_h",
        }
    )

    result = result.dropna(subset=["standardized_inchikey"])

    # Log PK parameter statistics
    logger.info("PK parameter statistics:")
    for col in result.columns:
        if col != "standardized_inchikey":
            non_null = result[col].notna().sum()
            logger.info(f"  {col}: {non_null} values")

    return result


def match_dataset_to_jump(
    df: pd.DataFrame,
    jump_compounds: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Match a dataset to JUMP compounds via InChIKey14 prefix.

    Args:
        df: DataFrame with standardized_inchikey column
        jump_compounds: JUMP compounds with Metadata_InChIKey14 column
        dataset_name: Name for logging

    Returns:
        DataFrame with Metadata_JCP2022 column added
    """
    # Add InChIKey14 prefix for matching
    df = df.copy()
    df["inchikey_prefix"] = df["standardized_inchikey"].str[:14]

    # Match to JUMP compounds
    matched = df.merge(
        jump_compounds[["Metadata_JCP2022", "Metadata_InChIKey14"]],
        left_on="inchikey_prefix",
        right_on="Metadata_InChIKey14",
        how="inner",
    )

    # Clean up temporary columns
    matched = matched.drop(columns=["inchikey_prefix", "Metadata_InChIKey14", "standardized_inchikey"])

    logger.info(f"{dataset_name}: {matched['Metadata_JCP2022'].nunique()} JUMP compounds matched")
    return matched


@app.command()
def main(
    input_zip: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to input zip file (default: data/external/toxicity_pk_data.zip)",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: data/interim/toxicity_pk_processed.csv)",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Fast mode: use RDKit directly instead of jump_smiles (~5s vs ~10min)",
    ),
) -> int:
    """Process toxicity and PK annotations and return exit code."""
    logger.info("Processing toxicity and pharmacokinetic annotations...")
    if fast:
        logger.info("Fast mode enabled: using RDKit InChIKeys without jump_canonical")

    # Set default paths
    if input_zip is None:
        input_zip = EXTERNAL_DATA_DIR / "toxicity_pk_data.zip"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "toxicity_pk_processed.csv"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if not validate_inputs(
        input_zip,
        compound_file,
        names=["Toxicity/PK data zip", "JUMP compound metadata"],
    ):
        return 1

    try:
        # Load JUMP compound metadata using shared utility
        jump_compounds = load_jump_compounds(compound_file)

        # Determine number of CPUs (only used in non-fast mode)
        import os

        num_cpus = max((os.cpu_count() or 1) // 2, 1) if not fast else 1

        # Process each dataset from the zip file
        logger.info("\n" + "=" * 60)
        logger.info("Processing DILI annotations")
        logger.info("=" * 60)
        dili_df = process_dili(zip_path=input_zip, num_cpus=num_cpus, fast=fast)

        logger.info("\n" + "=" * 60)
        logger.info("Processing DICT annotations")
        logger.info("=" * 60)
        dict_df = process_dict(zip_path=input_zip, num_cpus=num_cpus, fast=fast)

        logger.info("\n" + "=" * 60)
        logger.info("Processing PK annotations")
        logger.info("=" * 60)
        pk_df = process_pk(zip_path=input_zip, num_cpus=num_cpus, fast=fast)

        # Match each dataset to JUMP compounds
        logger.info("\n" + "=" * 60)
        logger.info("Matching to JUMP compounds via InChIKey14 prefix")
        logger.info("=" * 60)

        dili_matched = match_dataset_to_jump(dili_df, jump_compounds, "DILI")
        dict_matched = match_dataset_to_jump(dict_df, jump_compounds, "DICT")
        pk_matched = match_dataset_to_jump(pk_df, jump_compounds, "PK")

        # Combine all matched data (outer join on Metadata_JCP2022)
        # Start with unique JCP2022 IDs from all datasets
        all_jcp_ids = (
            set(dili_matched["Metadata_JCP2022"])
            | set(dict_matched["Metadata_JCP2022"])
            | set(pk_matched["Metadata_JCP2022"])
        )
        result = pd.DataFrame({"Metadata_JCP2022": sorted(all_jcp_ids)})

        # Merge DILI columns
        dili_cols = dili_matched[
            ["Metadata_JCP2022", "Metadata_dili_concern", "Metadata_dili_positive"]
        ].drop_duplicates(subset=["Metadata_JCP2022"])
        result = result.merge(dili_cols, on="Metadata_JCP2022", how="left")

        # Merge DICT columns
        dict_cols = dict_matched[
            ["Metadata_JCP2022", "Metadata_dict_concern", "Metadata_dict_positive"]
        ].drop_duplicates(subset=["Metadata_JCP2022"])
        result = result.merge(dict_cols, on="Metadata_JCP2022", how="left")

        # Merge PK columns (take first if duplicates from stereoisomers)
        pk_cols = pk_matched[
            [
                "Metadata_JCP2022",
                "Metadata_pk_vdss_l_kg",
                "Metadata_pk_cl_ml_min_kg",
                "Metadata_pk_fup",
                "Metadata_pk_mrt_h",
                "Metadata_pk_thalf_h",
            ]
        ].drop_duplicates(subset=["Metadata_JCP2022"])
        result = result.merge(pk_cols, on="Metadata_JCP2022", how="left")

        # Save results
        result.to_csv(output_file, index=False)
        logger.success(f"Saved {len(result)} records to {output_file}")

        # Summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        logger.info(f"Total JUMP compounds with any toxicity/PK annotation: {len(result)}")
        logger.info(f"  - DILI annotations: {result['Metadata_dili_concern'].notna().sum()}")
        logger.info(f"  - DICT annotations: {result['Metadata_dict_concern'].notna().sum()}")
        logger.info(f"  - PK annotations: {result['Metadata_pk_vdss_l_kg'].notna().sum()}")

        # Overlap statistics
        has_dili = result["Metadata_dili_concern"].notna()
        has_dict = result["Metadata_dict_concern"].notna()
        has_pk = result["Metadata_pk_vdss_l_kg"].notna()

        logger.info("\nOverlap statistics:")
        logger.info(f"  - DILI + DICT: {(has_dili & has_dict).sum()}")
        logger.info(f"  - DILI + PK: {(has_dili & has_pk).sum()}")
        logger.info(f"  - DICT + PK: {(has_dict & has_pk).sum()}")
        logger.info(f"  - All three: {(has_dili & has_dict & has_pk).sum()}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing toxicity/PK data: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
