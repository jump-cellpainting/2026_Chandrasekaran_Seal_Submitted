# -*- coding: utf-8 -*-
# Downstream analysis rules
# Target consistency visualization, profile comparison, and reporting
#
# Helper functions: get_target_consistency_combinations (in common.smk)
# Config variables: COMPARISON_PROFILES_ACTIVITY, COMPARISON_PROFILES_CONSISTENCY,
#                   TARGET_CONSISTENCY_DATASETS, TARGET_CONSISTENCY_COMBOS,
#                   CROSS_SOURCE_COMBOS (in common.smk)


rule all_target_consistency:
    input:
        [f"data/processed/target-consistency/{d}/{g}/{dist}/umap_grid_{f}.png"
         for d, g, f, dist in TARGET_CONSISTENCY_COMBOS]
    output:
        touch("data/processed/target-consistency/.complete")


rule target_consistency_base:
    """Generate target consistency visualizations (volcano, dotplot, summary, UMAP grid).

    This rule generates all base outputs plus the 'all' UMAP grid.
    Metadata is joined at visualization time from metadata_db.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        umap_h5ad="data/interim/anndata/{dataset}_perturbation_cosine_all_umap.h5ad",
        script="notebooks/2.03-ss-target-consistency.py"
    output:
        volcano="data/processed/target-consistency/{dataset}/{group_type}/{distance}/volcano.png",
        dotplot="data/processed/target-consistency/{dataset}/{group_type}/{distance}/dotplot.png",
        csv="data/processed/target-consistency/{dataset}/{group_type}/{distance}/results.csv",
        umap_grid="data/processed/target-consistency/{dataset}/{group_type}/{distance}/umap_grid_all.png",
    log:
        "logs/analysis/target_consistency/{dataset}__{group_type}__{distance}.log"
    shell:
        "pixi run python {input.script} --dataset {wildcards.dataset} --group-type {wildcards.group_type} --umap-filter all --distance {wildcards.distance} 2>&1 | tee {log}"


rule target_consistency_umap_active:
    """Generate active-only UMAP grid for target consistency.

    Uses --umap-only to skip regenerating volcano, dotplot, summary.
    Metadata is joined at visualization time from metadata_db.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        umap_h5ad="data/interim/anndata/{dataset}_perturbation_cosine_active_umap.h5ad",
        # Depend on base outputs to ensure they exist
        base_volcano="data/processed/target-consistency/{dataset}/{group_type}/{distance}/volcano.png",
        script="notebooks/2.03-ss-target-consistency.py"
    output:
        "data/processed/target-consistency/{dataset}/{group_type}/{distance}/umap_grid_active.png"
    log:
        "logs/analysis/target_consistency_umap_active/{dataset}__{group_type}__{distance}.log"
    shell:
        "pixi run python {input.script} --dataset {wildcards.dataset} --group-type {wildcards.group_type} --umap-filter active --umap-only --distance {wildcards.distance} 2>&1 | tee {log}"


# Profile-comparison consistency leaves: the runner (scripts/profile_comparison_runner.py) emits one
# per (group_type, distance) it finds in the copairs DB for the union-profile pair. In practice that is
# the 9 canonical target-consistency sources at cosine distance (canonical list in copairs_runs.yaml;
# see CLAUDE.md "Annotation Sources"). Declaring them lets the dataset-reference hub collate each
# annotation's CP-vs-DL agreement (concordance + overlap counts + the threshold-curve figure). If the
# DB's consistency combos ever change, run-all fails loud here - update this list to match.
PROFILE_CMP_CONSISTENCY_GROUPS = [
    "repurposing", "uniprot", "chemical_probes", "moa", "disease_area",
    "motive_biokg", "motive_opentargets", "motive_primekg", "toxcast_assay",
]
_PROFILE_CMP_CONSISTENCY_DIR = (
    "data/processed/profile-comparison/"
    "compound_no_source7_active_union_vs_compound_DL_CPCNN_no_source7_active_union/consistency"
)


rule profile_comparison:
    """Run pairwise profile comparisons for activity and consistency."""
    input:
        db="data/processed/copairs_results.duckdb",
        comparison_script="notebooks/1.01-jfh-profile-comparison.py",
        runner_script="scripts/profile_comparison_runner.py"
    output:
        # The activity comparison (CP vs DL) leaf is deterministic; declare its envelope so the hub
        # discovers it. The consistency leaves (one per annotation, cosine) are declared via expand so
        # the hub collates each annotation panel too.
        activity_summary="data/processed/profile-comparison/compound_no_source7_vs_compound_DL_CPCNN_no_source7/activity/default/summary.json",
        consistency_summaries=expand(
            _PROFILE_CMP_CONSISTENCY_DIR + "/{grp}/cosine/summary.json",
            grp=PROFILE_CMP_CONSISTENCY_GROUPS,
        ),
        complete=touch("data/processed/profile-comparison/.complete")
    log:
        "logs/analysis/profile_comparison.log"
    shell:
        "pixi run python {input.runner_script} --db {input.db} --script {input.comparison_script} 2>&1 | tee {log}"


rule profile_comparison_report:
    """Generate PDF report for profile comparisons."""
    input:
        "data/processed/profile-comparison/.complete",
        script="scripts/1.01-collate-profile-comparison-report.sh"
    output:
        "data/processed/profile-comparison/report.pdf"
    log:
        "logs/analysis/profile_comparison_report.log"
    shell:
        "bash {input.script} 2>&1 | tee {log}"


rule target_consistency_report:
    """Generate PDF report for target consistency."""
    input:
        "data/processed/target-consistency/.complete",
        script="scripts/2.03-collate-target-consistency-report.sh"
    output:
        "data/processed/target-consistency/{dataset}/report.pdf"
    log:
        "logs/analysis/target_consistency_report/{dataset}.log"
    shell:
        "bash {input.script} {wildcards.dataset} 2>&1 | tee {log}"


rule data_quality:
    """Analyze cell density vs activity and edge effects.

    Cell density analysis progression (diagnoses batch effect confounding):
    1. Raw: Simple median cell density per compound (baseline)
    2. Exclude source_2: Test if outlier source drives the pattern
    3. Normalized: Regress out source + z-score by plate (proper adjustment)

    Also analyzes edge effects in cell counts across plate positions.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/2.02b-ss-cell-count-quality.py"
    output:
        # Key outputs (many more files generated) - hierarchical path
        hexbin_raw=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/cell_density_activity/cell_density_vs_map_hexbin.png",
        hexbin_normalized=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/cell_density_activity/cell_density_vs_map_hexbin_normalized.png",
        scatter_normalized=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/cell_density_activity/cell_density_vs_map_scatter_normalized.png",
        distribution_comparison=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/cell_density_activity/cell_density_distribution_comparison.png",
        edge_effects_heatmap=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/edge_effects/cell_count_heatmap_by_source.png",
        edge_effects_summary=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/edge_effects/summary.csv",
        summary=f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/summary.json",
        complete=touch(f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/.complete")
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
    log:
        f"logs/analysis/data_quality/{DEFAULT_DATASET}__{DEFAULT_PREPROCESSING}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} --filter {params.filter_name} --activity-params {params.activity_params} --output $(dirname {output.complete}) 2>&1 | tee {log}"


rule cross_source_reproducibility:
    """Analyze cross-source reproducibility of compound activity.

    Compares within-source vs cross-source activity (normalized mAP) to measure
    how reproducible compound phenotypes are across different lab sites.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        script="notebooks/2.02a-ss-cross-source-reproducibility.py"
    output:
        expand(f"data/processed/cross-source-reproducibility/{DEFAULT_DATASET}/{{subdir}}/within_vs_cross_scatter.png",
               subdir=[c[2] for c in CROSS_SOURCE_COMBOS]),
        expand(f"data/processed/cross-source-reproducibility/{DEFAULT_DATASET}/{{subdir}}/summary.json",
               subdir=[c[2] for c in CROSS_SOURCE_COMBOS]),
        complete=touch(f"data/processed/cross-source-reproducibility/{DEFAULT_DATASET}/.complete")
    params:
        dataset=DEFAULT_DATASET,
        # Pass combos as colon-separated triples for shell parsing
        combos=" ".join([f"{p}:{f}:{s}" for p, f, s in CROSS_SOURCE_COMBOS]),
    log:
        f"logs/analysis/cross_source_reproducibility/{DEFAULT_DATASET}.log"
    shell:
        """
        OUTPUT_DIR=$(dirname {output.complete})
        for combo in {params.combos}; do
            IFS=':' read -r preprocessing filter_name subdir <<< "$combo"
            echo "Running: preprocessing=$preprocessing filter=$filter_name subdir=$subdir"
            pixi run python {input.script} --dataset {params.dataset} --preprocessing "$preprocessing" --filter "$filter_name" --output "$OUTPUT_DIR"
        done 2>&1 | tee {log}
        """


rule all_reports:
    """Generate all PDF reports."""
    input:
        "data/processed/profile-comparison/report.pdf",
        expand("data/processed/target-consistency/{dataset}/report.pdf",
               dataset=TARGET_CONSISTENCY_DATASETS)


rule chemical_space:
    """Characterize chemical space of JUMP compound collection.

    Generates physicochemical property distributions, annotation coverage,
    scaffold diversity, PAINS alerts, structure UMAP, and summary statistics.

    Structure UMAP uses L2 normalization + cosine similarity as a proxy for
    Tanimoto/Jaccard (achieves 99.8% rank correlation, see issue #26).
    Runs in rapids environment for GPU-accelerated UMAP.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        script="notebooks/2.04a-ss-chemical-space.py"
    output:
        combined_png="data/processed/chemical-space/all/chemical_space_combined.png",
        combined_pdf="data/processed/chemical-space/all/chemical_space_combined.pdf",
        summary_json="data/processed/chemical-space/all/chemical_space_summary.json",
        umap_h5ad="data/processed/chemical-space/all/structure_umap.h5ad",
        umap_png="data/processed/chemical-space/all/structure_umap.png",
        umap_gallery_png="data/processed/chemical-space/all/structure_umap_gallery.png",
        umap_gallery_pdf="data/processed/chemical-space/all/structure_umap_gallery.pdf",
        complete=touch("data/processed/chemical-space/all/.complete")
    resources:
        gpu=1
    priority: 10  # srijit_2_33 (5 min) waits for it, so do not leave it last in the GPU queue
    log:
        "logs/analysis/chemical_space.log"
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} 2>&1 | tee {log}"


rule compound_traits:
    """Analyze compound traits vs phenotypic activity.

    Tests which physicochemical properties are associated with higher activity rates:
    MW, LogP, TPSA, QED, Lipinski violations, PAINS, target annotations.

    Observed association: less drug-like compounds (high LogP, Ro5 violations, PAINS)
    show higher activity rates and broader cellular perturbations.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/2.04c-ss-compound-traits.py"
    output:
        combined_png=f"data/processed/compound-traits/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/compound_traits_combined.png",
        summary_json=f"data/processed/compound-traits/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/compound_traits_summary.json",
        complete=touch(f"data/processed/compound-traits/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/.complete")
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
    log:
        f"logs/analysis/compound_traits/{DEFAULT_DATASET}__{DEFAULT_PREPROCESSING}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} --filter {params.filter_name} --activity-params {params.activity_params} --output $(dirname {output.complete}) 2>&1 | tee {log}"


rule structure_morphology:
    """Analyze structure-morphology relationship for a specific structural representation.

    Computes k-NN overlap between chemical structure and morphology spaces
    to test whether structurally similar compounds are also morphologically similar.

    Supports two structural representations:
    - morgan: Morgan fingerprints (2048-bit) - L2 norm + cosine ≈ Tanimoto (ρ=0.998)
    - chemberta: ChemBERTa-77M-MLM embeddings (384-dim) - learned SMILES representations

    Both use the same compound set (intersection of valid compounds) for fair comparison.

    Runs in rapids environment for GPU-accelerated k-NN when available.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        chemberta="data/interim/compound_featurization/chemberta_77m_mlm.npz",
        profiles=f"data/interim/anndata/{DEFAULT_DATASET}_perturbation.h5ad",
        script="notebooks/2.05-ss-structure-morphology.py"
    output:
        combined_png=f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/structure_morphology_combined.png",
        summary_json=f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/structure_morphology_summary.json",
        overlap_parquet=f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/neighbor_overlap.parquet",
        complete=touch(f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/.complete")
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
    resources:
        gpu=1
    log:
        f"logs/analysis/structure_morphology/{DEFAULT_DATASET}__{DEFAULT_PREPROCESSING}/{{structure_rep}}.log"
    wildcard_constraints:
        structure_rep="morgan|chemberta"
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} --filter {params.filter_name} --activity-params {params.activity_params} --structure {wildcards.structure_rep} 2>&1 | tee {log}"


rule all_structure_morphology:
    """Run structure-morphology analysis for all structural representations."""
    input:
        expand(f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/.complete",
               structure_rep=["morgan", "chemberta"])


rule phenotype_prediction:
    """Predict phenotypic outcomes from chemical structure.

    Uses scaffold split (MoleculeNet standard) vs random split to quantify
    generalization gap. XGBoost classifier/regressor with GPU acceleration.

    Structural representations:
    - morgan: Morgan fingerprints (2048-bit) - recommended, best performance (0.769 ROC-AUC)
    - chemberta: ChemBERTa-77M-MLM embeddings - SKIPPED (underperforms properties baseline)
      Note: If running ChemBERTa, use --use-mlp flag (MLP: 0.745 > XGBoost: 0.718)

    Key finding: Properties baseline (MW, LogP, TPSA) achieves 0.733 ROC-AUC,
    capturing 87 of the signal. Morgan FP adds only +0.036 improvement.
    See GitHub issue #29 for signal decomposition analysis.

    Runs in deepchem environment for DeepChem ScaffoldSplitter and GPU XGBoost.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        chemberta="data/interim/compound_featurization/chemberta_77m_mlm.npz",
        script="notebooks/2.06-ss-phenotype-prediction.py"
    output:
        comparison_png=f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/comparison_plot.png",
        summary_json=f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/summary.json",
        complete=touch(f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/.complete")
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
    resources:
        gpu=1  # Single GPU for XGBoost training
    log:
        f"logs/analysis/phenotype_prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}.log"
    wildcard_constraints:
        structure_rep="morgan|chemberta"
    shell:
        "scripts/gpu_lease.sh pixi run -e deepchem python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} --filter {params.filter_name} --activity-params {params.activity_params} --structure {wildcards.structure_rep} --sample 0.2 2>&1 | tee {log}"


rule all_phenotype_prediction:
    """Run phenotype prediction for Morgan fingerprints only.

    ChemBERTa is excluded: underperforms properties baseline (0.718 vs 0.733).
    To re-enable, add "chemberta" to structure_rep list and use --use-mlp flag.
    """
    input:
        expand(f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/{{structure_rep}}/.complete",
               structure_rep=["morgan"])


rule structure_activity_pairs:
    """Run notebook 2.07's expensive all-pairs and neighbor analyses."""
    input:
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        perturbation_profiles="data/interim/anndata/compound_no_source7_perturbation.h5ad",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        script="notebooks/2.07-ss-structure-activity-pairs.py"
    output:
        pairs=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/pairs.parquet",
        all_pair_similarity_histogram=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/all_pair_similarity_histogram.parquet",
        phenotypic_cliffs=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/phenotypic_cliffs.csv",
        phenotypic_cliff_target_divergence=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/phenotypic_cliff_target_divergence.csv",
        activity_cliff_pairs=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/activity_cliff_pairs.parquet",
        activity_cliff_pair_summary=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/activity_cliff_pair_summary.csv",
        activity_cliff_scaffold_groups=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/activity_cliff_scaffold_groups.csv",
        phenotypic_activity=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/phenotypic_activity.csv",
        neighbor_overlap=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/neighbor_overlap.csv",
        moa_coherence=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/moa_coherence.csv",
        summary=f"data/processed/structure-activity-pairs/{DEFAULT_DATASET}/summary.json"
    log:
        f"logs/analysis/structure-activity-pairs/{DEFAULT_DATASET}.log"
    resources:
        gpu=1
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} "
        "--morgan-fp {input.morgan_fp} --perturbation-profiles {input.perturbation_profiles} "
        "--metadata-db {input.metadata_db} --copairs-db {input.copairs_db} "
        "--output $(dirname {output.summary}) --dataset {DEFAULT_DATASET} "
        "--activity-preprocessing activity_no_target2 --activity-filter all_sources "
        "2>&1 | tee {log}"


rule opentargets_target_alignment:
    """Score asymmetric OpenTargets target alignment for its figure panels.

    Uses well-level profiles to reproduce the consistency_no_target2 compound
    consensus and calls the public copairs multilabel API programmatically.
    """
    input:
        canonical_well_profiles="data/interim/anndata/compound_no_source7.h5ad",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        script="notebooks/2.46-ss-opentargets-target-alignment.py",
    output:
        memberships=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_memberships.csv",
        target_audit=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_reference_audit.csv",
        models=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_models.csv",
        sensitivities=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_sensitivities.csv",
        shuffled_calibration=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_shuffled_label_calibration.csv",
        matched_nonmembers=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_matched_nonmembers.csv",
        quadrant_table=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_activity_consistency_quadrants.csv",
        target_summary=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_target_summary.csv",
        fitted_curve=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_fitted_curve.csv",
        compound_summary=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_compound_summary.csv",
        property_bins=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_target_alignment_property_bins.csv",
        grain_audit=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/opentargets_property_grain_audit.csv",
        summary=f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/summary.json",
    log:
        f"logs/analysis/opentargets-target-alignment/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} "
        "--canonical-well-profiles {input.canonical_well_profiles} "
        "--metadata-db {input.metadata_db} --copairs-db {input.copairs_db} "
        "--output-dir $(dirname {output.summary}) 2>&1 | tee {log}"


rule fingerprint_metrics:
    """Evaluate fingerprint transformations for Tanimoto approximation.

    Tests L2 normalization, popcount normalization, IDF weighting, MinHash,
    and random features to find methods where cosine/Euclidean approximate
    Tanimoto/Jaccard similarity.

    Key finding: Simple L2 normalization achieves ρ=0.998 correlation with
    Tanimoto for JUMP compounds due to homogeneous bit densities (CV=20.4%).
    See Issue #26.
    """
    input:
        morgan_fp="data/interim/compound_featurization/morgan_fp.npz",
        script="notebooks/2.04b-ss-fingerprint-metrics.py"
    output:
        correlation_png="data/processed/fingerprint-metrics/fingerprint_metrics_correlation.png",
        summary_png="data/processed/fingerprint-metrics/fingerprint_metrics_summary.png",
        results_json="data/processed/fingerprint-metrics/fingerprint_metrics_summary.json",
        complete=touch("data/processed/fingerprint-metrics/.complete")
    log:
        "logs/analysis/fingerprint_metrics.log"
    shell:
        "pixi run python {input.script} --sample-size 3000 --skip-slow 2>&1 | tee {log}"


rule chemical_space_analyzed:
    """Chemical-space summary for the canonical analyzed slice (excl source 7).

    Same analysis as `chemical_space`, filtered to the analyzed compounds via the canonical
    key from analysis.yaml (default_dataset + default_preprocessing, read inside 2.04a). Skips
    the structure UMAP (GPU): only the summary is needed downstream, consumed by
    `dataset_reference` (analyzed annotated_any / Ro5 / PAINS / scaffolds / launched).
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        analysis_config="configs/snakemake/analysis.yaml",
        script="notebooks/2.04a-ss-chemical-space.py"
    output:
        summary_json=f"data/processed/chemical-space/{DEFAULT_DATASET}/chemical_space_summary.json"
    log:
        "logs/analysis/chemical_space_analyzed.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --analyzed-only --skip-umap 2>&1 | tee {log}"


rule compound_collection_facts:
    """Emit the computed compound-collection facts as one audited summary.

    Core chemistry values come from the canonical analyzed chemical-space summary. Target/MOA,
    MolPort, dual-sourcing, and EPA functional-use counts are recomputed against the four-key
    analyzed roster. A tracked contract pins the expected values and narrative assertions.
    """
    input:
        chemical_summary=f"data/processed/chemical-space/{DEFAULT_DATASET}/chemical_space_summary.json",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        functional_use_csv="data/external/2026-04-06_JUMP_chems_func_use.csv",
        analysis_config="configs/snakemake/analysis.yaml",
        molport_archive="data/external/molport_batch_search.zip",
        numerical_contract="references/compound_collection_contract.json",
        script="notebooks/0.16-ss-compound-collection-facts.py"
    output:
        summary="data/processed/compound-collection-facts/summary.json"
    log:
        "logs/analysis/compound_collection_facts.log"
    shell:
        "pixi run -e cheminformatics python {input.script} 2>&1 | tee {log}"


rule unichem_mapping:
    """External-database mapping coverage via the UniChem/MyChem ID pointers.

    Counts how many JUMP compounds resolve to at least one external chemical database (and the
    per-database split), emitting the standard {description, numbers, files} envelope so
    `dataset_reference` collates it like any other node.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        pointers="data/external/unichem_pointers.csv",
        script="notebooks/0.17-ss-unichem-mapping.py"
    output:
        summary="data/processed/unichem-mapping/summary.json"
    log:
        "logs/analysis/unichem_mapping.log"
    shell:
        "pixi run -e marimo python {input.script} 2>&1 | tee {log}"


rule dataset_reference_manifest:
    """Write the one canonical manifest of summaries included in the paper report.

    The summary inputs below are the sole membership definition. The manifest records each concrete
    path and its producing rule; both the in-repo and standalone notebook read this same document.
    """
    input:
        summaries=[
            *[
                f"data/processed/umap/{dataset}/{metric}_{filter_name}/summary.json"
                for dataset, metric, filter_name in UMAP_COMBOS
            ],
            f"data/processed/exploration/0.04/{DEFAULT_DATASET}/summary.json",
            "data/processed/profile-comparison/compound_no_source7_vs_compound_DL_CPCNN_no_source7/activity/default/summary.json",
            *expand(
                "data/processed/profile-comparison/compound_no_source7_active_union_vs_compound_DL_CPCNN_no_source7_active_union/consistency/{grp}/cosine/summary.json",
                grp=PROFILE_CMP_CONSISTENCY_GROUPS,
            ),
            f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/summary.json",
            *expand(
                f"data/processed/cross-source-reproducibility/{DEFAULT_DATASET}/{{subdir}}/summary.json",
                subdir=[combo[2] for combo in CROSS_SOURCE_COMBOS],
            ),
            "data/processed/chemical-space/all/chemical_space_summary.json",
            f"data/processed/chemical-space/{DEFAULT_DATASET}/chemical_space_summary.json",
            "data/processed/compound-collection-facts/summary.json",
            "data/processed/activity-summary/summary.json",
            f"data/processed/compound-traits/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/compound_traits_summary.json",
            f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/morgan/structure_morphology_summary.json",
            f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/chemberta/structure_morphology_summary.json",
            f"data/processed/activity-cliffs/{DEFAULT_DATASET}/summary.json",
            f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/summary.json",
            f"data/processed/substructure-morphology/{DEFAULT_DATASET}/summary.json",
            f"data/processed/sar-vignette/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/summary.json",
            f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/summary.json",
            f"data/processed/opentargets-target-alignment/{DEFAULT_DATASET}/summary.json",
            "data/processed/commercial-compounds/summary.json",
            f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/morgan/summary.json",
            "data/processed/fingerprint-metrics/fingerprint_metrics_summary.json",
            "data/processed/dataset_structure/summary.json",
            "data/processed/unichem-mapping/summary.json",
        ],
        analysis_config="configs/snakemake/analysis.yaml",
        script="src/jump_production/processing/build_dataset_reference_manifest.py",
    output:
        manifest="data/processed/dataset_reference/dataset_reference_manifest.json"
    params:
        default_dataset=DEFAULT_DATASET,
        default_columns=DEFAULT_COLUMNS,
        default_preprocessing=DEFAULT_PREPROCESSING,
        default_filter=DEFAULT_FILTER,
        default_activity_params=DEFAULT_ACTIVITY_PARAMS,
    log:
        "logs/analysis/dataset_reference_manifest.log"
    shell:
        "pixi run python {input.script} "
        "--output {output.manifest} "
        "--default-dataset {params.default_dataset} "
        "--default-columns {params.default_columns} "
        "--default-preprocessing {params.default_preprocessing} "
        "--default-filter {params.default_filter} "
        "--default-activity-params {params.default_activity_params} "
        "{input.summaries} 2>&1 | tee {log}"


rule dataset_reference:
    """Collate every number named by the canonical dataset-reference manifest.

    The notebook computes no scientific values. It reads the same manifest in every environment,
    resolving that manifest and each listed envelope from local storage or its exact S3 mirror.
    """
    input:
        manifest=rules.dataset_reference_manifest.output.manifest,
        script="notebooks/dataset_reference.py"
    output:
        json="data/processed/dataset_reference/dataset_reference.json",
        csv="data/processed/dataset_reference/dataset_reference.csv"
    log:
        "logs/analysis/dataset_reference.log"
    shell:
        "uv run {input.script} 2>&1 | tee {log}"


rule dataset_structure:
    """Regenerate Figure 1A's structural numbers and emit them as a machine-readable artifact.

    The schematic Figure 1A (dataset/source overlaps) is "not precise to scale"; this node
    recomputes every structural quantity in the Figure 1 narrative from the augmented metadata
    DB - headline counts, both wave-overlap framings (nominated vs screened), the ORF/CRISPR
    gene overlap, per-site compound counts, and replicate breadth.

    Like dataset_reference, the target is a marimo notebook run as a plain script: it reads its
    declared input at a fixed path and writes the JSON plus the three panel PNGs. Freshness is the
    DAG's job - declaring the metadata DB as an input puts this rule downstream of its producing
    rule. The PNGs require matplotlib-venn (in the marimo env); script execution renders them as a
    side effect of running every cell.
    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        all_cell_counts="data/external/all_cell_counts.csv.gz",
        script="notebooks/0.12-ss-dataset-structure-venn.py"
    output:
        json="data/processed/dataset_structure/dataset_structure.json",
        summary="data/processed/dataset_structure/summary.json",
        upset="data/processed/dataset_structure/figures/upset_site_combinations.png",
        exchange="data/processed/dataset_structure/figures/compound_exchange.png",
        network="data/processed/dataset_structure/figures/exchange_network.png",
        venn="data/processed/dataset_structure/figures/orf_crispr_venn.png"
    log:
        "logs/analysis/dataset_structure.log"
    shell:
        "pixi run -e marimo python {input.script} 2>&1 | tee {log}"


rule activity_summary:
    """Phenotypic-activity headline numbers for the canonical analyzed slice, as an envelope summary.

    The activity counts (analyzed / phenotypically active / rate) live only in the copairs results
    store, which is an aggregate across many copairs runs - so they belong to no single analysis
    notebook. This node queries the store at the canonical key (from analysis.yaml) and emits the
    standard {description, numbers, files} envelope, so `dataset_reference` views it like any other
    node instead of special-casing a DB query.
    """
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        analysis_config="configs/snakemake/analysis.yaml",
        profile=f"data/raw/profiles/{DEFAULT_DATASET}.parquet",
        script="notebooks/0.13-ss-activity-summary.py"
    output:
        summary="data/processed/activity-summary/summary.json"
    log:
        "logs/analysis/activity_summary.log"
    shell:
        "pixi run -e marimo python {input.script} 2>&1 | tee {log}"
