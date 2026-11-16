#!/usr/bin/env python
"""
Download external data and profile files using Pooch.
Downloads to data/external/ and data/raw/profiles/ folders.

Two sets of profiles are downloaded:
- Recipe-Harmony profiles: batch-corrected by jump-profiling-recipe (default, used by
  copairs/anndata). Saved as {dataset}.parquet.
- Pre-Harmony profiles: post-featselect, pre-batch-correction. Used as input for
  in-repo batch correction via batch_correct.py. Saved as {dataset}_pre_harmony.parquet.
  In-repo Harmony outputs use the _rsc suffix (e.g., compound_no_source7_rsc.parquet).
"""

import pooch
import typer
from loguru import logger

from jump_production.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

PROFILES_DIR = RAW_DATA_DIR / "profiles"

EXTERNAL_FILES = {
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt": (
        "repurposing_drugs_20200324.txt",
        "9c3a08b9d4369fe4257dc9aa763d682fc5ef6b7dd0615534bb44f92b48bc287b",
    ),
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt": (
        "repurposing_samples_20200324.txt",
        "3324dfac93d19ceaeafbe7d9e466127709328777cb2f6c5a465740cd327088e9",
    ),
    "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz": (
        "compound.csv.gz",
        "8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a",
    ),
    "https://github.com/jump-cellpainting/datasets/raw/main/metadata/plate.csv.gz": (
        "plate.csv.gz",
        "541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319",
    ),
    "https://github.com/jump-cellpainting/datasets/raw/main/metadata/well.csv.gz": (
        "well.csv.gz",
        "fde9fe0678b8d9973f17e106e4cf7e62017110ee3b700f10e27122e1d5f291a1",
    ),
    "https://github.com/jump-cellpainting/datasets/releases/download/v0.12/jump_metadata.duckdb": (
        "jump_metadata.duckdb",
        "ea77a4234cd01060bdf0c8c7f4a18e64dac283c8630a25de4a205936413f6a12",
    ),
    # Cell counts from jump-profiling-recipe (pinned to commit d2512d9)
    "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/all_cell_counts.csv.gz": (
        "all_cell_counts.csv.gz",
        "bf0b2252f316ae96cf857052f1e7739fd1aa6a1f1fee92eb9b1bcf89c89e6897",
    ),
    "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/compound_cell_counts.csv.gz": (
        "compound_cell_counts.csv.gz",
        "799a8be0ff54becb86f731467147f1ac0bfbe1936c6fefdf0964dd7876ef9676",
    ),
    "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/crispr_cell_counts.csv.gz": (
        "crispr_cell_counts.csv.gz",
        "67a137376cfb0725c22e5d71ae2b2543cc38efd2ad435f4d24de0b5078b7cbb0",
    ),
    "https://github.com/broadinstitute/jump-profiling-recipe/raw/d2512d978ca17aafead0e99de66386337e6312e4/inputs/cell_counts/orf_cell_counts.csv.gz": (
        "orf_cell_counts.csv.gz",
        "809be69372a5e19c53f3c921b49d20f768ee660c3a037e437a57bfecc7deb691",
    ),
    # MOTIVE compound-gene annotations from Zenodo (DOI: 10.5281/zenodo.18197517)
    # Curated drug-target interactions from 8 databases: BioKG, DGIdb, DrugRep, Hetionet, OpenBioLink, OpenTargets, PharmeBiNet, PrimeKG
    "https://zenodo.org/records/18197517/files/annotations_compound_gene_curated.parquet?download=1": (
        "motive_cpd_gene_annot.parquet",
        "164057bf957c7829ab929d5598b605e00b98024691406742101d8b5a32f36536",
    ),
    # UniChem ID mappings for JUMP compounds (from same Zenodo record)
    # Maps InChIKey to external database IDs (ChEMBL, DrugBank, PubChem, etc.) for 91% of JUMP compounds
    # Use for compound matching when adding new annotation sources
    "https://zenodo.org/records/18197517/files/mappings_pointers.csv?download=1": (
        "unichem_pointers.csv",
        "03400b14411b561d4a8a2eb4a90b71c393bffc849dcaa52b653d37907cedbf89",
    ),
    # EPA ToxCast invitrodb v4.3 summary files (August 2025)
    # Contains hitcall and AC50 data for ~9,400 chemicals across ~1,600 assay endpoints
    # Citation: US EPA (2025). ToxCast invitrodb version 4.3. https://doi.org/10.23645/epacomptox.6062623.v14
    "https://clowder.edap-cluster.com/files/68af6b70e4b02565fc7c3a98/blob": (
        "toxcast_invitrodb_v4_3_summary.zip",
        "c61d46cf685d5d84e72d0d4f6f46746ab1714d6f924e43480d390925ae827c59",
    ),
    # EPA DSSTox chemical identifiers (December 2025)
    # Maps DTXSID to InChIKey for matching ToxCast chemicals to JUMP compounds
    "https://clowder.edap-cluster.com/files/69529775e4b0731a616efc4b/blob": (
        "dsstox_identifiers.zip",
        "66fc9d4d3bda053ab60ee01e2ff0ed4ba030339588c40711b6939faede6cf4c3",
    ),
    # KCGS v2.0 (Kinase Chemogenomic Set) from SGC-UNC via Zenodo (DOI: 10.5281/zenodo.7830352)
    # "The most highly annotated set of selective kinase inhibitors available for phenotypic screens"
    # 295 compounds with Kd < 100 nM and S10 < 0.04 (highly selective, narrow-spectrum)
    # Zenodo is the official distribution channel; provides stable DOI-versioned URL
    # Reference: Elkins et al. (2021) IJMS, DOI: 10.3390/ijms22020566
    # Website: https://www.sgc-unc.org/kcgs
    "https://zenodo.org/records/7830352/files/KCGS2.0%20plate%20map%20and%20kinase%20data%20and%20coverage%20summary.xlsx?download=1": (
        "kcgs_v2.xlsx",
        "4b2ab160808c8491d3bc1d830166db7d767771855b3a23a9bfc5cbde4d8cd61a",
    ),
    # ChEMBL release 33 compound-target activity matrix for JUMP compounds.
    # Archived with the other paper inputs at DOI: 10.5281/zenodo.21800106.
    # See jump_production.downloading.chembl_targets_sqlite for its verified
    # provenance and optional exact reconstruction from the official SQLite release.
    "https://zenodo.org/records/21800106/files/targetannotations_singleproteins_jumpcompounds_all_5.csv?download=1": (
        "targetannotations_singleproteins_jumpcompounds_all_5.csv",
        "b299aabfd07a4b6685c9eea0327d762a3d20b71434c82a646a24d76c32c46480",
    ),
    # DILIrank v2, DICTrank, and PKSmart toxicity/pharmacokinetic inputs.
    "https://zenodo.org/records/21800106/files/toxicity_pk_data.zip?download=1": (
        "toxicity_pk_data.zip",
        "80316a23707f6fb0fbabfcaf133dda1957459f753ad87f9b72d14de5ee4eb61b",
    ),
    # EPA functional-use categories and QSUR-predicted use probabilities.
    "https://zenodo.org/records/21800106/files/2026-04-06_JUMP_chems_func_use.csv?download=1": (
        "2026-04-06_JUMP_chems_func_use.csv",
        "27f78d857e8b094837a9998addbf701cef666099ba4c35f2b40c62cf5c542b0b",
    ),
    # PKIS2 compound-by-kinase inhibition matrix (Drewry et al. 2017).
    # Canonical PLOS ONE S4 supplement, DOI: 10.1371/journal.pone.0181585.s004.
    "https://journals.plos.org/plosone/article/file?type=supplementary&id=info:doi/10.1371/journal.pone.0181585.s004": (
        "pkis2.xlsx",
        "48ead22a1f860cd0d5096fa87d5acd329f722fe8d65e693bb0be682a333e2a2c",
    ),
}

MANUAL_FILES = [
    "pd_export_01_2025_875_targets_standardized.xlsx",  # Chemical Probes (Probes & Drugs Portal)
    "molport_batch_search.zip",  # Molport commercial availability search results
]

# Recipe-Harmony profile files from CellPainting Gallery (batch-corrected by jump-profiling-recipe)
# These are the default profiles used by copairs/anndata downstream
# Note: compound.parquet (with source_7) is synced via `just get-inputs` from team S3, not here
PROFILE_FILES = {
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect_harmony.parquet": (
        "compound_no_source7.parquet",
        "8e1e5d9e50c8c7c95ed406981b02adb57c7ff9d253192ed52e661115379be6f1",
    ),
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN/v1.0/profiles_dropna_var_mad_int_featselect_harmony.parquet": (
        "compound_DL_CPCNN.parquet",
        "3fbf22bb0a8f45aefb924ce024f6b3abf64e33f4b1dc17a6ef916a2534e966bc",
    ),
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN_no_source7/v1.0/profiles_dropna_var_mad_int_featselect_harmony.parquet": (
        "compound_DL_CPCNN_no_source7.parquet",
        "da9485973e74e22d22f48239ede414fe872efb53fab28aa4eb1e17022a062b00",
    ),
}

# Pre-Harmony profile files from CellPainting Gallery (post-featselect, pre-batch-correction)
# Used as input for in-repo batch correction via batch_correct.py
# In-repo Harmony outputs use the _rsc suffix (e.g., compound_no_source7_rsc.parquet)
PRE_HARMONY_FILES = {
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_no_source7/v1.0/profiles_var_mad_int_featselect.parquet": (
        "compound_no_source7_pre_harmony.parquet",
        "30655596ee3b6b06ce18a13342f429249dbd7b2351ea0649609384721571d02b",
    ),
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN_no_source7/v1.0/profiles_dropna_var_mad_int_featselect.parquet": (
        "compound_DL_CPCNN_no_source7_pre_harmony.parquet",
        "f0ac49023c6c21e94352e468806e8edd2e4a3e93a403c60fa10dbf009b68e4a3",
    ),
    # Full profiles (with source_7) — needed for _with_source7 variants via filter_source7_exclusive
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/COMPOUND/v1.0a/profiles_var_mad_int_featselect.parquet": (
        "compound_pre_harmony.parquet",
        "9ee4862e8bfc9b08ba5d8675f50cd224f7a4d1ec83848b9fe4396224960b2936",
    ),
    "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/profiles_assembled/compound_DL_CPCNN/v1.0/profiles_dropna_var_mad_int_featselect.parquet": (
        "compound_DL_CPCNN_pre_harmony.parquet",
        "843fabab93495e69c22bd7e52102a9c70212e7e099a4c008f4f9c3d740b15f70",
    ),
}


@app.command()
def main() -> int:
    """Download external data and profile files using Pooch."""
    # Download external files
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading external files to {EXTERNAL_DATA_DIR}")

    for url, (filename, hash_value) in EXTERNAL_FILES.items():
        logger.info(f"Downloading {filename}")
        pooch.retrieve(url=url, known_hash=hash_value, path=EXTERNAL_DATA_DIR, fname=filename)

    for filename in MANUAL_FILES:
        if not (EXTERNAL_DATA_DIR / filename).exists():
            logger.warning(f"{filename} missing - add manually")

    # Download recipe-Harmony profile files (default, used by copairs/anndata)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading recipe-Harmony profile files to {PROFILES_DIR}")

    for url, (filename, hash_value) in PROFILE_FILES.items():
        logger.info(f"Downloading {filename}")
        pooch.retrieve(url=url, known_hash=hash_value, path=PROFILES_DIR, fname=filename)

    # Download pre-Harmony profiles (input for in-repo batch correction)
    logger.info(f"Downloading pre-Harmony profile files to {PROFILES_DIR}")

    for url, (filename, hash_value) in PRE_HARMONY_FILES.items():
        logger.info(f"Downloading {filename}")
        pooch.retrieve(url=url, known_hash=hash_value, path=PROFILES_DIR, fname=filename)

    logger.success("All downloads complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
