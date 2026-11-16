# -*- coding: utf-8 -*-
# Srijit's analysis notebooks (phase 2) - phenotype prediction and SAR analysis
#
# These rules orchestrate Srijit's analysis notebooks for predicting compound
# properties from Cell Painting profiles.
#
# Run all: snakemake srijit -j 4
# Individual: snakemake srijit_2_30 -j 1
#
# Tip: Run `pixi run snakemake --touch -j1` to prevent unnecessary re-runs after:
#   - Downloading results from S3 (just get-results)
#   - Pulling code changes that modify scripts
#   - Manually running scripts outside Snakemake

# Memory of the notebooks that run worker processes (2.30, 2.34, 2.40): about 2 GB for
# each worker and for the notebook itself, measured on 2.30. See COPAIRS_CONSISTENCY_MEM_MB.
SRIJIT_WORKER_MEM_MB = 2000

# =============================================================================
# Aggregate target: run all Srijit's analysis notebooks
# =============================================================================

rule srijit:
    """Run all of Srijit's analysis notebooks."""
    input:
        f"data/processed/pains-prediction/{DEFAULT_DATASET}/summary.json",
        f"data/processed/activity-cliffs/{DEFAULT_DATASET}/summary.json",
        f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/summary.json",
        f"data/processed/sar-vignette/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/summary.json",
        f"data/processed/toxicity-pk-prediction/{DEFAULT_DATASET}/summary.json",
        f"data/processed/commercial-compounds/summary.json",
        f"data/processed/mitotox-analysis/{DEFAULT_DATASET}/summary.json",
        "data/processed/mmp9-inhibitors/summary.json",
        f"data/processed/substructure-morphology/{DEFAULT_DATASET}/summary.json",
        f"data/processed/polypharmacology/{DEFAULT_DATASET}/summary.json",
        f"data/processed/functional-use-classification/summary.json",
        f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/summary.json",
        f"data/processed/kinase-chemical-probes/{DEFAULT_DATASET}/summary.json",
        f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/summary.json",


# =============================================================================
# 2.30: PAINS Prediction
# =============================================================================

# Cap BLAS threads for scaffold cross-validation.
rule srijit_2_30:
    """Predict PAINS structural alerts from morphological profiles.

    Uses scaffold splitting for realistic generalization assessment.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.30-seal-pains-prediction.py",
    output:
        summary=f"data/processed/pains-prediction/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        sample=1.0,
    log:
        f"logs/srijit/2.30-pains-prediction/{DEFAULT_DATASET}.log"
    # One process per fit, each on one thread; each cross-validation has 10 slow fits.
    threads: 10
    resources:
        mem_mb=lambda wildcards, threads: SRIJIT_WORKER_MEM_MB * (threads + 1)
    priority: 10  # long: start it as soon as its inputs exist
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "pixi run python {input.script} --dataset {params.dataset} --sample {params.sample} "
        "--workers {threads} 2>&1 | tee {log}"


# =============================================================================
# 2.31: Activity Cliffs Analysis
# =============================================================================

rule srijit_2_31:
    """Activity cliffs analysis using structural and morphological similarity.

    Identifies compound pairs that are structurally similar but
    morphologically different (and vice versa).
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        fingerprints="data/interim/compound_featurization/morgan_fp.npz",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.31-seal-activity-cliffs-analysis.py",
    output:
        summary=f"data/processed/activity-cliffs/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        sample=1.0,
    log:
        f"logs/srijit/2.31-activity-cliffs/{DEFAULT_DATASET}.log"
    resources:
        gpu=1,
        # 58 GB of host memory measured with a 24 GB GPU (7 GB with a 48 GB GPU).
        mem_mb=60000
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} --dataset {params.dataset} --sample {params.sample} 2>&1 | tee {log}"


# =============================================================================
# 2.32: Activity Cliffs PhenoSeeker Style
# =============================================================================

rule srijit_2_32:
    """Transfer the released PhenoSeeker scaffold-series method to CellProfiler profiles.

    Groups compounds by Bemis-Murcko scaffold and identifies scaffold series
    that split into morphologically divergent clusters under the released rule.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        fingerprints="data/interim/compound_featurization/morgan_fp.npz",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.32-seal-activity-cliffs-phenoseeker.py",
    output:
        summary=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/summary.json",
        scaffold_analysis=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/scaffold_analysis.parquet",
        scaffold_series=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/cliff_scaffolds.csv",
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/srijit/2.32-activity-cliffs-phenoseeker/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} "
        "--min-compounds 6 --n-clusters 2 --separation-threshold 0.2 "
        "--top-scaffolds 0 --skip-plots "
        "--output $(dirname {output.summary}) 2>&1 | tee {log}"


# =============================================================================
# 2.33: SAR Vignette (BRICS-based structural alerts)
# =============================================================================

rule srijit_2_33:
    """Discover structural alerts using BRICS fragment enumeration.

    Identifies molecular substructures associated with phenotypically
    active compounds in Cell Painting.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        chem_umap="data/processed/chemical-space/all/structure_umap.h5ad",
        script="notebooks/2.33-seal-sar-vignette.py",
    output:
        summary=f"data/processed/sar-vignette/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
    log:
        f"logs/srijit/2.33-sar-vignette/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} 2>&1 | tee {log}"


# =============================================================================
# 2.34: Toxicity and PK Prediction
# =============================================================================

rule srijit_2_34:
    """Predict DILI, DICT, MitoTox, and PK endpoints from morphological profiles.

    All toxicity and PK labels come from the augmented metadata database.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        script="notebooks/2.34-seal-toxicity-pk-prediction.py",
    output:
        summary=f"data/processed/toxicity-pk-prediction/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/srijit/2.34-toxicity-pk-prediction/{DEFAULT_DATASET}.log"
    # One process per cross-validation fold, each on one thread.
    threads: 5
    resources:
        mem_mb=lambda wildcards, threads: SRIJIT_WORKER_MEM_MB * (threads + 1)
    priority: 10  # long: start it as soon as its inputs exist
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} "
        "--workers {threads} 2>&1 | tee {log}"


# =============================================================================
# 2.36: Commercial Compound Curation
# =============================================================================

rule srijit_2_36:
    """Curate commercially available compounds with distinctive profiles.

    Clusters active Molport-available compounds and selects diverse
    representatives across 6 cluster configurations.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        pheno_umap=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation_cosine_all_umap.h5ad",
        molport_zip=ancient("data/external/molport_batch_search.zip"),
        chem_umap="data/processed/chemical-space/all/structure_umap.h5ad",
        script="notebooks/2.36-seal-commercial-compounds.py",
    output:
        summary="data/processed/commercial-compounds/summary.json",
    resources:
        gpu=1
    log:
        "logs/srijit/2.36-commercial-compounds.log"
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids-cheminformatics python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 2.37: MitoTox Morphology Analysis
# =============================================================================

rule srijit_2_37:
    """Analyze MitoTox compounds by morphological mechanism.

    Tests whether functional mechanisms separate in morphological space.
    MitoTox labels come from the augmented database.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        morph_umap=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation_cosine_all_umap.h5ad",
        chem_umap="data/processed/chemical-space/all/structure_umap.h5ad",
        script="notebooks/2.37-seal-mitotox-morphology.py",
    output:
        summary=f"data/processed/mitotox-analysis/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/srijit/2.37-mitotox-morphology/{DEFAULT_DATASET}.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} 2>&1 | tee {log}"


# =============================================================================
# 2.38: MMP9 Inhibitor Identification
# =============================================================================

rule srijit_2_38:
    """Identify MMP9 inhibitors among JUMP compounds with MoA context and UMAP.

    Queries all annotation sources for target inhibitors, adds activity
    status from copairs, and visualizes in morphological/chemical space.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        morph_umap=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation_cosine_all_umap.h5ad",
        script="notebooks/2.38-seal-mmp9-inhibitors.py",
    output:
        summary="data/processed/mmp9-inhibitors/summary.json",
    log:
        "logs/srijit/2.38-mmp9-inhibitors.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 2.40: Substructure → Morphology (scaffold coherence + cluster enrichment)
# =============================================================================

rule srijit_2_40:
    """Link chemical substructure to morphological phenotype.

    Tests Murcko scaffold coherence in morphological space and enrichment of
    BRICS fragments across Leiden clusters of the profile UMAP.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        umap=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation_cosine_all_umap.h5ad",
        script="notebooks/2.40-seal-substructure-morphology.py",
    output:
        summary=f"data/processed/substructure-morphology/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/srijit/2.40-substructure-morphology/{DEFAULT_DATASET}.log"
    # One process per worker for the permutation and BRICS steps, each on one thread.
    threads: 8
    resources:
        mem_mb=lambda wildcards, threads: SRIJIT_WORKER_MEM_MB * (threads + 1)
    priority: 10  # long: start it as soon as its inputs exist
    shell:
        # OpenBLAS in the KMeans/PCA step crashes on high-core boxes ("precompiled NUM_THREADS
        # exceeded"); cap threads per the repo convention for heavy compute notebooks.
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} "
        "--workers {threads} 2>&1 | tee {log}"


rule srijit_2_40_active:
    """Run the canonical active-compound scaffold analysis used by the structure figure."""
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.40-seal-substructure-morphology.py",
    output:
        scaffold=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/scaffold_coherence.csv",
        cluster_enrichment=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/cluster_enrichment.csv",
        cluster_assignments=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/cluster_assignments.csv",
        cluster_features=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/cluster_features.csv",
        browser=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/browser.html",
        summary=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        output_dir=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active",
    log:
        f"logs/srijit/2.40-substructure-morphology/{DEFAULT_DATASET}_active.log"
    shell:
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} "
        "--output {params.output_dir} --active-only --skip-brics --skip-plots "
        "2>&1 | tee {log}"


# =============================================================================
# 2.41: Polypharmacology Profile Additivity
# =============================================================================

rule srijit_2_41:
    """Test whether dual-target compounds produce profiles resembling a blend
    of single-target profiles.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.41-seal-polypharmacology-additivity.py",
    output:
        summary=f"data/processed/polypharmacology/{DEFAULT_DATASET}/summary.json",
    log:
        f"logs/srijit/2.41-polypharmacology/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 2.42: Functional Use Classification (EPA functional use categories)
# =============================================================================

rule srijit_2_42:
    """Characterize EPA functional use categories (FUCs) for JUMP compounds.

    Tests enrichment/depletion of phenotypic activity by functional use
    category and predicted functional use probability. Activity comes from the
    copairs DB; functional-use categories/probabilities come from the
    `functional_use` table in the augmented metadata DB. The current augmented
    DB is required because the declared enrichment tables are downstream inputs.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/2.42-seal-functional-use-classification.py",
    output:
        summary="data/processed/functional-use-classification/summary.json",
        fup_enrichment="data/processed/functional-use-classification/fup_enrichment.csv",
        fuc_enrichment="data/processed/functional-use-classification/fuc_enrichment.csv",
        activity_nmap="data/processed/functional-use-classification/functional_use_activity_nmap.csv",
        activity_distributions="data/processed/functional-use-classification/functional_use_activity_distributions.parquet",
    params:
        dataset=DEFAULT_DATASET,
        columns=DEFAULT_COLUMNS,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
        fup_threshold=0.5,
        min_compounds=10,
    log:
        "logs/srijit/2.42-functional-use-classification.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --columns {params.columns} "
        "--preprocessing {params.preprocessing} --filter {params.filter_name} "
        "--activity-params {params.activity_params} "
        "--fup-threshold {params.fup_threshold} --min-compounds {params.min_compounds} "
        "2>&1 | tee {log}"


# =============================================================================
# 2.43: Trait → Activity / Consistency Enrichment
# =============================================================================

rule srijit_2_43:
    """Trait → phenotypic activity / consistency enrichment.

    For each compound trait family (RDKit named functional groups, BRICS
    fragments, Murcko scaffolds, target/MoA classes), tests whether compounds
    carrying the trait are over- or under-represented among phenotypically
    active and phenotypically consistent compounds (Fisher's exact, BH FDR).

    Analogous to MorphMap's gene-group enrichment, but for chemistry.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/2.43-seal-trait-activity-enrichment.py"
    output:
        activity_csv=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/enrichment_activity.csv",
        consistency_csv=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/enrichment_consistency.csv",
        structural_activity_nmap=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_class_activity_nmap.csv",
        structural_activity_distributions=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_class_activity_distributions.parquet",
        structural_all_significant=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_classes_all_significant_5pct_fdr.csv",
        consistency_property_effects=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/consistency_property_effects.csv",
        consistency_group_properties=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/consistency_group_properties.csv",
        summary_json=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        columns=DEFAULT_COLUMNS,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
        consistency_preprocessing="consistency_no_target2",
        consistency_filter="all_sources",
        consistency_distance="cosine",
    log:
        f"logs/srijit/2.43-trait-activity-enrichment/{DEFAULT_DATASET}.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} "
        "--columns {params.columns} --preprocessing {params.preprocessing} "
        "--filter {params.filter_name} --activity-params {params.activity_params} "
        "--consistency-preprocessing {params.consistency_preprocessing} "
        "--consistency-filter {params.consistency_filter} "
        "--consistency-distance {params.consistency_distance} "
        "--output $(dirname {output.summary_json}) 2>&1 | tee {log}"


# =============================================================================
# 2.44: Kinase / Chemical-Probe Library Overlap
# =============================================================================

rule srijit_2_44:
    """Visualize JUMP-overlap subsets of KCGS, PKIS, Chemical Probes (and PKIS2).

    For each probe set, reports: overlap with JUMP, % phenotypically active, %
    consistent. Plots UMAP overlay (small-multiples), a KCGS compound×compound
    similarity heatmap, and (when available) a PKIS2 compound×kinase binding heatmap
    with row order from Cell Painting clustering.

    Probe-set membership and PKIS2 binding are queried from the augmented metadata DB
    (kinase_probes, chemical_probes, and pkis2_binding tables) - no raw probe files are
    read. When the pkis2_binding table is absent (older DB), the PKIS2 panels are
    skipped and the rest runs normally.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        umap=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation_cosine_all_umap.h5ad",
        script="notebooks/2.44-seal-kinase-chemical-probes.py"
    output:
        summary=f"data/processed/kinase-chemical-probes/{DEFAULT_DATASET}/summary.json",
        overlap_csv=f"data/processed/kinase-chemical-probes/{DEFAULT_DATASET}/probe_overlap_summary.csv",
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/srijit/2.44-kinase-chemical-probes/{DEFAULT_DATASET}.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --dataset {params.dataset} --output $(dirname {output.summary}) 2>&1 | tee {log}"


# =============================================================================
# 2.45: Compound properties associated with phenotypic activity
# =============================================================================

rule srijit_2_45:
    """Consolidate compound-trait odds ratios using explicit eligible populations.

    Binary chemistry, clinical, annotation, and curated-library traits are read from
    the augmented metadata DB. Multi-class results consume eligible-population counts
    from 2.42 and 2.43, then recompute Fisher tests and within-family BH values.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        fup_enrichment=rules.srijit_2_42.output.fup_enrichment,
        fuc_enrichment=rules.srijit_2_42.output.fuc_enrichment,
        functional_use_summary=rules.srijit_2_42.output.summary,
        trait_enrichment=rules.srijit_2_43.output.activity_csv,
        trait_summary=rules.srijit_2_43.output.summary_json,
        script="notebooks/2.45-seal-property-activity-oddsratios.py",
    output:
        intrinsic_activity_nmap=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/intrinsic_activity_nmap_bins.csv",
        selected_categorical_activity_nmap=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/selected_categorical_activity_nmap.csv",
        compound_activity_nmap_distributions=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/compound_activity_nmap_distributions.parquet",
        binary_csv=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/binary_trait_oddsratios.csv",
        multiclass_csv=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/multiclass_trait_oddsratios.csv",
        binary_png=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_binary_traits.png",
        binary_pdf=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_binary_traits.pdf",
        binary_full_png=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_binary_traits_full.png",
        binary_full_pdf=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_binary_traits_full.pdf",
        multiclass_png=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_multiclass_activity.png",
        multiclass_pdf=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/forest_multiclass_activity.pdf",
        summary=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/summary.json",
    params:
        dataset=DEFAULT_DATASET,
        columns=DEFAULT_COLUMNS,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
        activity_threshold=0.10,
        fup_threshold=0.5,
        min_compounds=10,
        output_dir=lambda wildcards, output: str(Path(output.summary).parent),
    log:
        f"logs/srijit/2.45-property-activity-oddsratios/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} "
        "--dataset {params.dataset} --columns {params.columns} "
        "--preprocessing {params.preprocessing} "
        "--filter {params.filter_name} --activity-params {params.activity_params} "
        "--activity-threshold {params.activity_threshold} "
        "--fup-threshold {params.fup_threshold} --min-compounds {params.min_compounds} "
        "--trait-enrichment {input.trait_enrichment} --trait-summary {input.trait_summary} "
        "--fup-enrichment {input.fup_enrichment} --fuc-enrichment {input.fuc_enrichment} "
        "--functional-use-summary {input.functional_use_summary} --output {params.output_dir} "
        "2>&1 | tee {log}"
