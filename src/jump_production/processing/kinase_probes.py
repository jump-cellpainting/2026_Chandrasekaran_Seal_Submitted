#!/usr/bin/env python
"""Process kinase probe sets (KCGS and PKIS) and match to JUMP compounds.

Background
----------
KCGS and PKIS are gold-standard kinase inhibitor probe sets designed for phenotypic
screening. They are ideal for validating Cell Painting morphological profiling because:

1. Kinase inhibitors produce strong, detectable morphological phenotypes
2. Their biochemical targets are well-characterized (ground truth for validation)
3. They enable mechanism-of-action prediction validation

Literature support:
- "KCGS is the most highly annotated set of selective kinase inhibitors available
   to researchers for use in phenotypic screens" (PMID: 33429995)
- "Among molecular target subtypes, kinase targets appeared to benefit the most
   from Cell Painting-based approaches" (Nat Comms 2024, DOI: 10.1038/s41467-024-47171-1)

Data Sources
------------
KCGS (Kinase Chemogenomic Set) v2.0:
- Source: Zenodo DOI: 10.5281/zenodo.7830352 (official SGC-UNC distribution)
- 295 compounds with Kd < 100 nM and S10 < 0.04 (highly selective)
- Includes selectivity scores and target binding data from DiscoveRx/Nanosyn panels
- Reference: Elkins et al. (2021) IJMS, DOI: 10.3390/ijms22020566

PKIS (Published Kinase Inhibitor Set):
- Source: ChEMBL document CHEMBL2303647 (GSK official deposition)
- 367 compounds from GSK drug discovery programs, screened against 224 kinases
- Distributed to 300+ laboratories for validation studies
- Reference: Drewry et al. (2014) CTM, DOI: 10.2174/1568026613666131127160819

Processing Steps
----------------
1. Load KCGS from downloaded Excel file
2. Query PKIS from ChEMBL API (with caching)
3. Standardize SMILES using jump_smiles (jump_canonical method) for consistent matching
4. Match to JUMP compounds via InChIKey connectivity layer (first 14 chars)
   - Connectivity layer matching tolerates stereochemistry differences between labs
5. Output CSV with probe set membership for SQL import

Run directly:
    pixi run -e jump-smiles python -m jump_production.processing.kinase_probes
"""

import json
import urllib.error
import urllib.request
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

# ChEMBL API endpoint for PKIS compounds
# CHEMBL2303647 is GSK's official deposition of PKIS - curated structures with activity data
# Using ChEMBL ensures we get standardized, validated SMILES rather than scraping from other sources
# Reference: Drewry et al. (2014) "Seeding Collaborations to Advance Kinase Science with the GSK
# Published Kinase Inhibitor Set (PKIS)" DOI: 10.2174/1568026613666131127160819
CHEMBL_BASE_URL = "https://www.ebi.ac.uk"
CHEMBL_PKIS_URL = f"{CHEMBL_BASE_URL}/chembl/api/data/molecule.json?document_chembl_id=CHEMBL2303647&limit=1000"


def load_kcgs(input_file: Path) -> pd.DataFrame:
    """Load KCGS compounds from Zenodo Excel file.

    Args:
        input_file: Path to KCGS Excel file

    Returns:
        DataFrame with compound names and SMILES
    """
    logger.info(f"Loading KCGS from {input_file}")
    df = pd.read_excel(input_file, sheet_name="Data summary")

    # Filter out empty SMILES
    df = df[df["SMILES string"].notna() & (df["SMILES string"] != "EMPTY")].copy()

    # Convert yes/no to 1/0 for is_original_kcgs (may not exist in CSV)
    is_original_col = df.get("member of original KCGS", pd.Series([0] * len(df)))
    is_original = is_original_col.map({"yes": 1, "no": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Extract relevant columns
    result = pd.DataFrame(
        {
            "probe_id": df["compound"],
            "probe_name": df["compound"],
            "smiles": df["SMILES string"],
            "probe_set": "KCGS",
            "selectivity_s10": df.get("S10 (1 uM) from Discoverx data"),
            "is_original_kcgs": is_original,
            "target_info": df.get(
                "target data from our lab: generally  Discoverx Kd<100 nM and/or Nanosyn %I>90 (screened at 1 uM); some rows have literature data only, or a combination of literature and our data"
            ),
        }
    )

    logger.info(f"Loaded {len(result)} KCGS compounds")
    return result


def load_pkis_raw(cache_file: Path | None = None, max_compounds: int = 2000) -> list:
    """Load raw PKIS molecule data from cache or ChEMBL API.

    Args:
        cache_file: Optional path to cache the API response
        max_compounds: Maximum number of compounds to fetch (default: 2000)

    Returns:
        List of raw molecule dicts from ChEMBL API
    """
    # Check cache first
    if cache_file and cache_file.exists():
        logger.info(f"Loading PKIS from cache: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    logger.info("Fetching PKIS from ChEMBL API...")
    molecules = []

    # Fetch pages up to max_compounds
    url = CHEMBL_PKIS_URL
    while url and len(molecules) < max_compounds:
        logger.debug(f"Fetching: {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode())
            molecules.extend(data.get("molecules", []))
            next_url = data.get("page_meta", {}).get("next")
            # Handle relative URLs from ChEMBL API
            if next_url and not next_url.startswith("http"):
                url = f"{CHEMBL_BASE_URL}{next_url}"
            else:
                url = next_url

    # Trim to max_compounds
    molecules = molecules[:max_compounds]
    logger.info(f"Fetched {len(molecules)} molecules from ChEMBL (limited to {max_compounds})")

    # Cache the response
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(molecules, f)
        logger.info(f"Cached PKIS data to {cache_file}")

    return molecules


def fetch_pkis_from_chembl(cache_file: Path | None = None, max_compounds: int = 2000) -> pd.DataFrame:
    """Fetch PKIS compounds from ChEMBL API.

    Note: The PKIS set contains ~367 compounds, but the ChEMBL document query
    returns all molecules with activity data against PKIS targets. We limit
    to max_compounds to avoid downloading excessive data.

    Args:
        cache_file: Optional path to cache the API response
        max_compounds: Maximum number of compounds to fetch (default: 2000)

    Returns:
        DataFrame with ChEMBL IDs and SMILES
    """
    molecules = load_pkis_raw(cache_file, max_compounds)

    # Extract SMILES and IDs
    pkis_data = []
    for mol in molecules:
        chembl_id = mol.get("molecule_chembl_id")
        structures = mol.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        pref_name = mol.get("pref_name")

        if smiles:
            pkis_data.append(
                {
                    "probe_id": chembl_id,
                    "probe_name": pref_name or chembl_id,
                    "smiles": smiles,
                    "probe_set": "PKIS",
                    "selectivity_s10": None,
                    "is_original_kcgs": 0,
                    "target_info": None,
                }
            )

    result = pd.DataFrame(pkis_data)
    logger.info(f"Loaded {len(result)} PKIS compounds with SMILES")
    return result


def generate_inchikeys_fast_for_kinase_probes(kcgs_df: pd.DataFrame, pkis_mols: list) -> pd.DataFrame:
    """Generate InChIKeys without jump_canonical standardization (fast mode).

    Fast mode (~1 sec vs ~4 min) with ~12% fewer matches:
    - KCGS: Uses RDKit to generate InChIKey directly from SMILES
    - PKIS: Uses ChEMBL's pre-computed standard_inchi_key

    This trades completeness for speed - useful for development/testing.
    For production, use standardize_smiles() with jump_canonical method.

    Args:
        kcgs_df: KCGS DataFrame with 'smiles' column
        pkis_mols: Raw PKIS molecule list from ChEMBL API

    Returns:
        Combined DataFrame with 'standardized_inchikey' column
    """
    logger.info("Fast mode: generating InChIKeys without jump_canonical standardization")
    logger.warning("Fast mode finds ~12% fewer matches (106 vs 120) but runs 240x faster")

    # KCGS: Use shared utility for RDKit InChIKey generation
    kcgs_with_inchikey = generate_inchikeys_fast(kcgs_df, "smiles", source_name="KCGS")

    # PKIS: Use ChEMBL's standard_inchi_key directly (pre-computed)
    pkis_data = []
    for mol in pkis_mols:
        chembl_id = mol.get("molecule_chembl_id")
        structures = mol.get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        inchikey = structures.get("standard_inchi_key")
        pref_name = mol.get("pref_name")

        if smiles:
            pkis_data.append(
                {
                    "probe_id": chembl_id,
                    "probe_name": pref_name or chembl_id,
                    "smiles": smiles,
                    "probe_set": "PKIS",
                    "selectivity_s10": None,
                    "is_original_kcgs": 0,
                    "target_info": None,
                    "standardized_inchikey": inchikey,
                }
            )
    pkis_df = pd.DataFrame(pkis_data)
    logger.info(f"PKIS: {pkis_df['standardized_inchikey'].notna().sum()}/{len(pkis_df)} with ChEMBL InChIKey")

    return pd.concat([kcgs_with_inchikey, pkis_df], ignore_index=True)


def match_probes_to_jump(df: pd.DataFrame, jump_compounds: pd.DataFrame) -> pd.DataFrame:
    """Match probe compounds to JUMP via InChIKey connectivity layer.

    Args:
        df: DataFrame with standardized_inchikey column
        jump_compounds: DataFrame with Metadata_JCP2022 and Metadata_InChIKey14

    Returns:
        DataFrame with matched JUMP compound IDs
    """
    # Filter to valid InChIKeys
    df = df[df["standardized_inchikey"].notna()].copy()

    # Create connectivity layer (first 14 chars) for stereoisomer-tolerant matching
    df["connectivity"] = df["standardized_inchikey"].str[:14]

    # Merge
    matched = df.merge(
        jump_compounds[["Metadata_JCP2022", "Metadata_InChIKey14"]],
        left_on="connectivity",
        right_on="Metadata_InChIKey14",
        how="inner",
    )

    # Deduplicate (same probe might match multiple JUMP compounds via stereoisomers)
    matched = matched.drop_duplicates(subset=["probe_id", "Metadata_JCP2022"])

    # Clean up
    matched = matched.drop(columns=["connectivity", "Metadata_InChIKey14"])

    logger.info(
        f"Matched {matched['probe_id'].nunique()} probes to {matched['Metadata_JCP2022'].nunique()} JUMP compounds"
    )

    return matched


@app.command()
def main(
    kcgs_file: Path = typer.Option(
        None,
        "--kcgs",
        help="Path to KCGS Excel file (default: data/external/kcgs_v2.xlsx)",
    ),
    compound_file: Path = typer.Option(
        None,
        "--compounds",
        help="Path to JUMP compound metadata CSV (default: data/external/compound.csv.gz)",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV path (default: data/interim/kinase_probes.csv)",
    ),
    pkis_cache: Path = typer.Option(
        None,
        "--pkis-cache",
        help="Path to cache PKIS API response (default: data/external/pkis_chembl_cache.json)",
    ),
    num_cpu: int = typer.Option(
        4,
        "--num-cpu",
        "-j",
        help="Number of CPUs for SMILES standardization",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Fast mode: skip jump_canonical standardization (~1s vs ~4min, ~12% fewer matches)",
    ),
) -> int:
    """Process KCGS and PKIS kinase probe sets and match to JUMP compounds."""
    logger.info("Processing kinase probe sets (KCGS and PKIS)...")
    if fast:
        logger.info("Fast mode enabled: using RDKit/ChEMBL InChIKeys without jump_canonical")

    # Set default paths
    if kcgs_file is None:
        kcgs_file = EXTERNAL_DATA_DIR / "kcgs_v2.xlsx"
    if compound_file is None:
        compound_file = EXTERNAL_DATA_DIR / "compound.csv.gz"
    if output_file is None:
        output_file = INTERIM_DATA_DIR / "kinase_probes.csv"
    if pkis_cache is None:
        pkis_cache = EXTERNAL_DATA_DIR / "pkis_chembl_cache.json"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not validate_inputs(
        kcgs_file,
        compound_file,
        names=["KCGS Excel file", "JUMP compound metadata"],
    ):
        logger.info("Run: pixi run python -m jump_production.downloading.download_data")
        return 1

    try:
        # Load JUMP compounds using shared utility
        jump_compounds = load_jump_compounds(compound_file)

        # Load KCGS
        kcgs_df = load_kcgs(kcgs_file)

        # Load/fetch PKIS data
        pkis_raw = load_pkis_raw(cache_file=pkis_cache)

        if fast:
            # Fast mode: use RDKit for KCGS, ChEMBL InChIKey for PKIS
            # ~240x faster (~1s vs ~4min) but finds ~12% fewer matches
            combined = generate_inchikeys_fast_for_kinase_probes(kcgs_df, pkis_raw)
        else:
            # Full mode: standardize all SMILES with jump_canonical for maximum matches
            pkis_df = fetch_pkis_from_chembl(cache_file=pkis_cache)
            combined = pd.concat([kcgs_df, pkis_df], ignore_index=True)
            logger.info(f"Combined: {len(combined)} compounds ({len(kcgs_df)} KCGS + {len(pkis_df)} PKIS)")
            combined = standardize_smiles(combined, "smiles", num_cpus=num_cpu, source_name="kinase probes")

        # Match to JUMP
        matched = match_probes_to_jump(combined, jump_compounds)

        # Create output with probe set flags
        output = matched[
            [
                "Metadata_JCP2022",
                "probe_set",
                "probe_id",
                "probe_name",
                "standardized_inchikey",
                "selectivity_s10",
                "is_original_kcgs",
                "target_info",
            ]
        ].copy()

        # Rename columns with Metadata_ prefix
        output = output.rename(
            columns={
                "probe_set": "Metadata_kinase_probe_set",
                "probe_id": "Metadata_kinase_probe_id",
                "probe_name": "Metadata_kinase_probe_name",
                "standardized_inchikey": "Metadata_kinase_probe_inchikey",
                "selectivity_s10": "Metadata_kinase_selectivity_s10",
                "is_original_kcgs": "Metadata_kinase_is_original_kcgs",
                "target_info": "Metadata_kinase_target_info",
            }
        )

        # Save
        output.to_csv(output_file, index=False)
        logger.success(f"Saved {len(output)} kinase probe annotations to {output_file}")

        # Summary statistics
        logger.info("\nKinase probes summary:")
        kcgs_matched = output[output["Metadata_kinase_probe_set"] == "KCGS"]
        pkis_matched = output[output["Metadata_kinase_probe_set"] == "PKIS"]
        # pkis_raw has SMILES for all, so count those with SMILES
        pkis_with_smiles = sum(1 for m in pkis_raw if (m.get("molecule_structures") or {}).get("canonical_smiles"))
        logger.info(f"  KCGS: {len(kcgs_matched)} probes matched to JUMP ({len(kcgs_df)} total)")
        logger.info(f"  PKIS: {len(pkis_matched)} probes matched to JUMP ({pkis_with_smiles} total)")
        logger.info(f"  Unique JUMP compounds: {output['Metadata_JCP2022'].nunique()}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e.filename}")
        return 1
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        return 1
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch PKIS from ChEMBL API: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error processing kinase probes: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
