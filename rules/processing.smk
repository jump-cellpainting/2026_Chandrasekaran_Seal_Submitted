# -*- coding: utf-8 -*-
# Processing rules: external/raw data → interim
# ChEMBL, Repurposing Hub, Chemical Probes, Motive, ToxCast, DB augmentation

rule process_chembl:
    input:
        "data/external/targetannotations_singleproteins_jumpcompounds_all_5.csv",
        "src/jump_production/processing/chembl.py"
    output:
        "data/interim/chembl_protein_targets_processed.csv"
    log:
        "logs/processing/chembl.log"
    shell:
        "pixi run python -m jump_production.processing.chembl 2>&1 | tee {log}"

rule process_repurposing_hub:
    input:
        "data/external/repurposing_samples_20200324.txt",
        "data/external/repurposing_drugs_20200324.txt",
        "data/external/compound.csv.gz",
        "src/jump_production/processing/repurposing_hub.py"
    output:
        "data/interim/repurposing_hub_annotations_processed.tsv"
    log:
        "logs/processing/repurposing_hub.log"
    shell:
        "pixi run python -m jump_production.processing.repurposing_hub 2>&1 | tee {log}"

rule process_chemical_probes:
    input:
        "data/external/pd_export_01_2025_875_targets_standardized.xlsx",
        "src/jump_production/processing/chemical_probes.py"
    output:
        "data/interim/chemical_probes_processed.csv",
        "data/interim/chemical_probes_targets.csv"
    log:
        "logs/processing/chemical_probes.log"
    shell:
        "pixi run -e jump-smiles python -m jump_production.processing.chemical_probes 2>&1 | tee {log}"

rule process_kinase_probes:
    # Uses --fast mode: ~1s vs ~4min, finds ~12% fewer matches (106 vs 120)
    # Fast mode uses RDKit InChIKey (KCGS) + ChEMBL InChIKey (PKIS), skipping jump_canonical
    # For maximum matches, remove --fast and use jump-smiles env instead
    input:
        kcgs="data/external/kcgs_v2.xlsx",
        compounds="data/external/compound.csv.gz",
        script="src/jump_production/processing/kinase_probes.py"
    output:
        "data/interim/kinase_probes.csv"
    log:
        "logs/processing/kinase_probes.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.kinase_probes --fast 2>&1 | tee {log}"

rule process_pkis2:
    # PKIS2 per-compound x per-kinase binding matrix matched to JUMP compounds.
    # Ingested into the augmented DB as the pkis2_binding table (see augment_metadata_db).
    # Source data/external/pkis2.xlsx is tracked in team S3 (`just get-inputs`).
    input:
        pkis2="data/external/pkis2.xlsx",
        compounds="data/external/compound.csv.gz",
        script="src/jump_production/processing/pkis2.py"
    output:
        "data/interim/pkis2_binding.parquet"
    log:
        "logs/processing/pkis2.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.pkis2 --fast 2>&1 | tee {log}"

rule process_functional_use:
    # EPA functional use categories + predicted use probabilities (Phillips et al. 2017).
    # Cleans the raw CSV ('NA' -> null, typing, Metadata_funcuse_* renaming) into a typed
    # interim parquet; ingested into the functional_use table by augment_metadata_db.
    # Source CSV is tracked in team S3 (`just get-inputs`).
    input:
        csv="data/external/2026-04-06_JUMP_chems_func_use.csv",
        script="src/jump_production/processing/functional_use.py"
    output:
        "data/interim/functional_use_processed.parquet"
    log:
        "logs/processing/functional_use.log"
    shell:
        "pixi run python -m jump_production.processing.functional_use 2>&1 | tee {log}"

rule process_motive:
    input:
        "data/external/motive_cpd_gene_annot.parquet",
        "data/external/jump_metadata.duckdb",
        "src/jump_production/processing/motive.py"
    output:
        "data/interim/motive_annotations.csv"
    log:
        "logs/processing/motive.log"
    shell:
        "pixi run python -m jump_production.processing.motive --db data/external/jump_metadata.duckdb 2>&1 | tee {log}"

rule process_toxcast:
    input:
        "data/external/toxcast_invitrodb_v4_3_summary.zip",
        "data/external/dsstox_identifiers.zip",
        "data/external/compound.csv.gz",
        "src/jump_production/processing/toxcast.py"
    output:
        "data/interim/toxcast_processed.parquet"
    log:
        "logs/processing/toxcast.log"
    shell:
        "pixi run python -m jump_production.processing.toxcast --apply-filters 2>&1 | tee {log}"

rule process_toxicity_pk:
    # Toxicity and pharmacokinetic annotations:
    # - DILI: Drug-Induced Liver Injury from DILIrank v2
    # - DICT: Drug-Induced CardioToxicity from DICTrank
    # - PK: Pharmacokinetic parameters from PKSmart
    # Uses --fast mode: ~5s vs ~10min, uses RDKit directly instead of jump_smiles
    # For maximum matches, remove --fast and use jump-smiles env instead
    input:
        "data/external/toxicity_pk_data.zip",
        "data/external/compound.csv.gz",
        "src/jump_production/processing/toxicity_pk.py"
    output:
        "data/interim/toxicity_pk_processed.csv"
    log:
        "logs/processing/toxicity_pk.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.toxicity_pk --fast 2>&1 | tee {log}"

# Compute physicochemical properties from SMILES using RDKit
# Run manually: snakemake compound_properties -j 1
rule compound_properties:
    input:
        db="data/external/jump_metadata.duckdb",
        script="src/jump_production/processing/compound_properties.py"
    output:
        parquet="data/interim/compound_featurization/properties.parquet"
    log:
        "logs/processing/compound_properties.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.compound_properties properties --force 2>&1 | tee {log}"

# Compute Morgan fingerprints (2048-bit, radius 2)
# Run manually: snakemake compound_morgan -j 1
rule compound_morgan:
    input:
        db="data/external/jump_metadata.duckdb",
        script="src/jump_production/processing/compound_properties.py"
    output:
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz"
    log:
        "logs/processing/compound_morgan.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.compound_properties morgan --force 2>&1 | tee {log}"

# Compute ChemBERTa-77M-MLM embeddings (384-dim)
# Uses dedicated chemberta env with CUDA PyTorch
# Run manually: snakemake compound_chemberta -j 1
rule compound_chemberta:
    input:
        db="data/external/jump_metadata.duckdb",
        script="src/jump_production/processing/compound_properties.py"
    output:
        chemberta="data/interim/compound_featurization/chemberta_77m_mlm.npz"
    log:
        "logs/processing/compound_chemberta.log"
    shell:
        "pixi run -e chemberta python -m jump_production.processing.compound_properties chemberta --device cuda --force 2>&1 | tee {log}"

rule process_mitotox:
    # MitoTox mitochondrial toxicity labels matched to JUMP compounds
    # Uses RDKit InChIKey generation (fast mode, ~5s)
    input:
        ancient("data/external/mitotox/mitotox_compounds.parquet"),
        "data/external/compound.csv.gz",
        "src/jump_production/processing/mitotox.py"
    output:
        "data/interim/mitotox_processed.csv"
    log:
        "logs/processing/mitotox.log"
    shell:
        "pixi run -e cheminformatics python -m jump_production.processing.mitotox 2>&1 | tee {log}"

# Download MitoTox database (compounds with mitochondrial toxicity labels)
# Standalone download - not part of main pipeline, run manually when needed
# Uses cached API responses if available (won't re-download unless --refresh)
# Run manually: snakemake download_mitotox -j 1
rule download_mitotox:
    input:
        script="src/jump_production/downloading/mitotox.py"
    output:
        "data/external/mitotox/mitotox_compounds.parquet"
    log:
        "logs/downloading/mitotox.log"
    shell:
        "pixi run python -m jump_production.downloading.mitotox 2>&1 | tee {log}"

rule fetch_image_dimensions:
    input:
        "scripts/fetch_image_dimensions.sh"
    output:
        "data/interim/image_dimensions.csv"
    log:
        "logs/processing/fetch_image_dimensions.log"
    shell:
        "./scripts/fetch_image_dimensions.sh > {output} 2> {log}"

rule augment_metadata_db:
    input:
        db="data/external/jump_metadata.duckdb",
        chembl="data/interim/chembl_protein_targets_processed.csv",
        repurposing="data/interim/repurposing_hub_annotations_processed.tsv",
        probes="data/interim/chemical_probes_processed.csv",
        kinase_probes="data/interim/kinase_probes.csv",
        pkis2="data/interim/pkis2_binding.parquet",
        motive="data/interim/motive_annotations.csv",
        toxcast="data/interim/toxcast_processed.parquet",
        funcuse="data/interim/functional_use_processed.parquet",
        toxicity_pk="data/interim/toxicity_pk_processed.csv",
        mitotox="data/interim/mitotox_processed.csv",
        compound_props="data/interim/compound_featurization/properties.parquet",
        cell_counts="data/external/compound_cell_counts.csv.gz",
        image_dims="data/interim/image_dimensions.csv",
        sql="scripts/augment_metadata_db.sql"
    output:
        "data/interim/jump_metadata_augmented.duckdb"
    log:
        "logs/processing/augment_metadata_db.log"
    shell:
        """
        set -e
        cp {input.db} {output}
        pixi run python -c 'import sys, duckdb; duckdb.connect(sys.argv[1]).execute(open(sys.argv[2]).read())' {output} {input.sql}
        """ + " 2>&1 | tee {log}"
