# Publication figures are ordinary scripts over canonical analysis outputs.

# Every figure draws its caption band and writes its legend through these two helpers, so a change
# to either has to rebuild the figures. Without this a helper fix left stale PNGs on disk and the
# validator still passed them.
FIGURE_HELPERS = [
    "src/jump_production/summaries.py",
    "src/jump_production/visualization.py",
]

rule figure_compound_collection:
    """Build Figure 1 directly through the chemical-space analysis script."""
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        analysis_config="configs/snakemake/analysis.yaml",
        dataset_structure_summary="data/processed/dataset_structure/summary.json",
        dataset_structure_asset="assets/figure_compound_collection/dataset_structure.png",
        script="notebooks/2.04a-ss-chemical-space.py",
        helpers=FIGURE_HELPERS
    output:
        dataset_structure_png="data/processed/figures/figure_compound_collection/dataset_structure.png",
        chemical_space_png="data/processed/figures/figure_compound_collection/chemical_space.png",
        chemical_space_pdf="data/processed/figures/figure_compound_collection/chemical_space.pdf",
        legend="data/processed/figures/figure_compound_collection/legend.md",
        summary="data/processed/figures/figure_compound_collection/summary.json"
    log:
        "logs/figures/figure_compound_collection.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --publication-only "
        "--publication-figure figure_compound_collection --output $(dirname {output.summary}) "
        "--dataset-structure-asset {input.dataset_structure_asset} --dpi {DEFAULT_DPI} "
        "2>&1 | tee {log}"


rule figure_phenotypic_consistency:
    """Build Figure 2 directly from its five canonical analysis inputs.

    Runs in the default Pixi environment rather than `cheminformatics`: this figure draws
    embeddings and retrieval curves and never touches RDKit, unlike the other four.
    """
    input:
        well_umap="data/interim/anndata/compound_no_source7_cosine_all_umap.h5ad",
        perturbation_umap="data/interim/anndata/compound_no_source7_perturbation_cosine_all_umap.h5ad",
        perturbation_profiles="data/interim/anndata/compound_no_source7_perturbation.h5ad",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        script="notebooks/figure-phenotypic-consistency.py",
        helpers=FIGURE_HELPERS
    output:
        moa_neighborhoods_png="data/processed/figures/figure_phenotypic_consistency/moa_neighborhoods.png",
        moa_neighborhoods_pdf="data/processed/figures/figure_phenotypic_consistency/moa_neighborhoods.pdf",
        map_retrieval_png="data/processed/figures/figure_phenotypic_consistency/map_retrieval.png",
        map_retrieval_pdf="data/processed/figures/figure_phenotypic_consistency/map_retrieval.pdf",
        global_moa_map_png="data/processed/figures/figure_phenotypic_consistency/global_moa_map.png",
        global_moa_map_pdf="data/processed/figures/figure_phenotypic_consistency/global_moa_map.pdf",
        activity_context_png="data/processed/figures/figure_phenotypic_consistency/activity_context.png",
        activity_context_pdf="data/processed/figures/figure_phenotypic_consistency/activity_context.pdf",
        cross_scaffold_retrieval_png="data/processed/figures/figure_phenotypic_consistency/cross_scaffold_retrieval.png",
        cross_scaffold_retrieval_pdf="data/processed/figures/figure_phenotypic_consistency/cross_scaffold_retrieval.pdf",
        ranked_targets_moas_png="data/processed/figures/figure_phenotypic_consistency/ranked_targets_moas.png",
        ranked_targets_moas_pdf="data/processed/figures/figure_phenotypic_consistency/ranked_targets_moas.pdf",
        selected_moa_classes_csv="data/processed/figures/figure_phenotypic_consistency/selected_moa_classes.csv",
        global_moa_retrieval_csv="data/processed/figures/figure_phenotypic_consistency/global_moa_retrieval.csv",
        cross_scaffold_moa_retrieval_csv="data/processed/figures/figure_phenotypic_consistency/cross_scaffold_moa_retrieval.csv",
        ranked_targets_and_moas_csv="data/processed/figures/figure_phenotypic_consistency/ranked_targets_and_moas.csv",
        legend="data/processed/figures/figure_phenotypic_consistency/legend.md",
        summary="data/processed/figures/figure_phenotypic_consistency/summary.json"
    log:
        "logs/figures/figure_phenotypic_consistency.log"
    shell:
        "pixi run python {input.script} --well-umap {input.well_umap} "
        "--perturbation-umap {input.perturbation_umap} "
        "--perturbation-profiles {input.perturbation_profiles} --metadata-db {input.metadata_db} "
        "--copairs-db {input.copairs_db} --output-dir $(dirname {output.summary}) "
        "--dataset compound_no_source7 --activity-preprocessing activity_no_target2 "
        "--activity-filter all_sources --metric cosine --dpi {DEFAULT_DPI} 2>&1 | tee {log}"


rule figure_compound_activity_associations:
    """Render the compound-activity-association composite, modular panels, and separate figures."""
    input:
        functional_summary="data/processed/functional-use-classification/summary.json",
        functional_table="data/processed/functional-use-classification/functional_use_activity_nmap.csv",
        functional_distributions="data/processed/functional-use-classification/functional_use_activity_distributions.parquet",
        trait_summary=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/summary.json",
        structural_table=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_class_activity_nmap.csv",
        structural_distributions=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_class_activity_distributions.parquet",
        structural_all_significant=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/structural_classes_all_significant_5pct_fdr.csv",
        consistency_table=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/consistency_property_effects.csv",
        consistency_group_table=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/consistency_group_properties.csv",
        property_summary=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/summary.json",
        intrinsic_table=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/intrinsic_activity_nmap_bins.csv",
        categorical_table=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/selected_categorical_activity_nmap.csv",
        compound_distributions=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/compound_activity_nmap_distributions.parquet",
        alignment_summary=rules.opentargets_target_alignment.output.summary,
        alignment_memberships=rules.opentargets_target_alignment.output.memberships,
        alignment_target_summary=rules.opentargets_target_alignment.output.target_summary,
        alignment_curve=rules.opentargets_target_alignment.output.fitted_curve,
        alignment_compounds=rules.opentargets_target_alignment.output.compound_summary,
        alignment_property_bins=rules.opentargets_target_alignment.output.property_bins,
        alignment_grain_audit=rules.opentargets_target_alignment.output.grain_audit,
        script="notebooks/figure-compound-activity-associations.py",
        helpers=FIGURE_HELPERS
    output:
        composite_png="data/processed/figures/figure_compound_activity_associations/compound_properties_activity.png",
        composite_pdf="data/processed/figures/figure_compound_activity_associations/compound_properties_activity.pdf",
        intrinsic_properties_png="data/processed/figures/figure_compound_activity_associations/intrinsic_properties.png",
        intrinsic_properties_pdf="data/processed/figures/figure_compound_activity_associations/intrinsic_properties.pdf",
        lipinski_violations_png="data/processed/figures/figure_compound_activity_associations/lipinski_violations.png",
        lipinski_violations_pdf="data/processed/figures/figure_compound_activity_associations/lipinski_violations.pdf",
        structural_classes_png="data/processed/figures/figure_compound_activity_associations/structural_classes.png",
        structural_classes_pdf="data/processed/figures/figure_compound_activity_associations/structural_classes.pdf",
        clinical_phase_png="data/processed/figures/figure_compound_activity_associations/clinical_phase.png",
        clinical_phase_pdf="data/processed/figures/figure_compound_activity_associations/clinical_phase.pdf",
        curated_collections_png="data/processed/figures/figure_compound_activity_associations/pains_and_curated_collections.png",
        curated_collections_pdf="data/processed/figures/figure_compound_activity_associations/pains_and_curated_collections.pdf",
        functional_use_png="data/processed/figures/figure_compound_activity_associations/functional_use.png",
        functional_use_pdf="data/processed/figures/figure_compound_activity_associations/functional_use.pdf",
        phenotypic_consistency_png="data/processed/figures/figure_compound_activity_associations/phenotypic_consistency.png",
        phenotypic_consistency_pdf="data/processed/figures/figure_compound_activity_associations/phenotypic_consistency.pdf",
        phenotypic_consistency_heatmap_png="data/processed/figures/figure_compound_activity_associations/phenotypic_consistency_heatmap.png",
        phenotypic_consistency_heatmap_pdf="data/processed/figures/figure_compound_activity_associations/phenotypic_consistency_heatmap.pdf",
        target_alignment_png="data/processed/figures/figure_compound_activity_associations/target_alignment.png",
        target_alignment_pdf="data/processed/figures/figure_compound_activity_associations/target_alignment.pdf",
        compound_consistency_png="data/processed/figures/figure_compound_activity_associations/compound_consistency_properties.png",
        compound_consistency_pdf="data/processed/figures/figure_compound_activity_associations/compound_consistency_properties.pdf",
        cohort_activity_png="data/processed/figures/figure_compound_activity_associations/compound_cohort_activity_properties.png",
        cohort_activity_pdf="data/processed/figures/figure_compound_activity_associations/compound_cohort_activity_properties.pdf",
        alignment_membership_table="data/processed/figures/figure_compound_activity_associations/opentargets_target_alignment_memberships.csv",
        alignment_target_table="data/processed/figures/figure_compound_activity_associations/opentargets_target_alignment_target_summary.csv",
        alignment_curve_table="data/processed/figures/figure_compound_activity_associations/opentargets_target_alignment_fitted_curve.csv",
        alignment_compound_table="data/processed/figures/figure_compound_activity_associations/opentargets_target_alignment_compound_summary.csv",
        alignment_property_bin_table="data/processed/figures/figure_compound_activity_associations/opentargets_target_alignment_property_bins.csv",
        alignment_grain_audit_table="data/processed/figures/figure_compound_activity_associations/opentargets_property_grain_audit.csv",
        intrinsic_table="data/processed/figures/figure_compound_activity_associations/intrinsic_activity_nmap_bins.csv",
        categorical_table="data/processed/figures/figure_compound_activity_associations/selected_categorical_activity_nmap.csv",
        compound_distributions="data/processed/figures/figure_compound_activity_associations/compound_activity_nmap_distributions.parquet",
        structural_table="data/processed/figures/figure_compound_activity_associations/structural_class_activity_nmap.csv",
        structural_distributions="data/processed/figures/figure_compound_activity_associations/structural_class_activity_distributions.parquet",
        structural_all_significant="data/processed/figures/figure_compound_activity_associations/structural_classes_all_significant_5pct_fdr.csv",
        functional_table="data/processed/figures/figure_compound_activity_associations/functional_use_activity_nmap.csv",
        functional_distributions="data/processed/figures/figure_compound_activity_associations/functional_use_activity_distributions.parquet",
        consistency_table="data/processed/figures/figure_compound_activity_associations/consistency_property_effects.csv",
        consistency_group_table="data/processed/figures/figure_compound_activity_associations/consistency_group_properties.csv",
        legend="data/processed/figures/figure_compound_activity_associations/legend.md",
        summary="data/processed/figures/figure_compound_activity_associations/summary.json"
    log:
        "logs/figures/figure_compound_activity_associations.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --dataset {DEFAULT_DATASET} "
        "--functional-use $(dirname {input.functional_summary}) "
        "--trait-enrichment $(dirname {input.trait_summary}) "
        "--property-oddsratios $(dirname {input.property_summary}) "
        "--target-alignment $(dirname {input.alignment_summary}) "
        "--output $(dirname {output.summary}) --dpi 600 2>&1 | tee {log}"


rule figure_opentargets_group_consistency:
    """Render OpenTargets target-group consistency views and audit tables."""
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        group_properties=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/consistency_group_properties.csv",
        group_properties_summary=f"data/processed/trait-activity-enrichment/{DEFAULT_DATASET}/summary.json",
        compound_properties=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/compound_activity_nmap_distributions.parquet",
        compound_properties_summary=f"data/processed/property-activity-oddsratios/{DEFAULT_DATASET}/summary.json",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/figure-opentargets-group-consistency.py",
        helpers=FIGURE_HELPERS
    output:
        primary_png="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_consistency.png",
        primary_pdf="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_consistency.pdf",
        deduplicated_png="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_consistency_deduplicated.png",
        deduplicated_pdf="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_consistency_deduplicated.pdf",
        companion_png="data/processed/figures/figure_opentargets_group_consistency/opentargets_activity_consistency.png",
        companion_pdf="data/processed/figures/figure_opentargets_group_consistency/opentargets_activity_consistency.pdf",
        group_activity_png="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_mean_activity.png",
        group_activity_pdf="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_mean_activity.pdf",
        compound_activity_png="data/processed/figures/figure_opentargets_group_consistency/opentargets_matched_compound_activity.png",
        compound_activity_pdf="data/processed/figures/figure_opentargets_group_consistency/opentargets_matched_compound_activity.pdf",
        group_summary="data/processed/figures/figure_opentargets_group_consistency/opentargets_group_summary.csv",
        continuous_table="data/processed/figures/figure_opentargets_group_consistency/continuous_group_properties.csv",
        categorical_table="data/processed/figures/figure_opentargets_group_consistency/categorical_group_composition.csv",
        memberships="data/processed/figures/figure_opentargets_group_consistency/opentargets_memberships.csv",
        matched_compounds="data/processed/figures/figure_opentargets_group_consistency/opentargets_matched_compounds.csv",
        legend="data/processed/figures/figure_opentargets_group_consistency/legend.md",
        summary="data/processed/figures/figure_opentargets_group_consistency/summary.json"
    log:
        "logs/figures/figure_opentargets_group_consistency.log"
    shell:
        "pixi run python {input.script} --copairs-db {input.copairs_db} "
        "--group-properties {input.group_properties} "
        "--group-properties-summary {input.group_properties_summary} "
        "--compound-properties {input.compound_properties} "
        "--compound-properties-summary {input.compound_properties_summary} "
        "--metadata-db {input.metadata_db} "
        "--output-dir $(dirname {output.summary}) --dpi 600 2>&1 | tee {log}"


rule figure_suppl_structure:
    """Render Supplementary Figure STRUCTURE from one cache and two canonical analyses."""
    input:
        pair_cache_summary=rules.structure_activity_pairs.output.summary,
        pair_cache_pairs=rules.structure_activity_pairs.output.pairs,
        pair_cache_histogram=rules.structure_activity_pairs.output.all_pair_similarity_histogram,
        pair_cache_phenotypic_cliffs=rules.structure_activity_pairs.output.phenotypic_cliffs,
        pair_cache_target_divergence=rules.structure_activity_pairs.output.phenotypic_cliff_target_divergence,
        pair_cache_activity_pairs=rules.structure_activity_pairs.output.activity_cliff_pairs,
        pair_cache_pair_summary=rules.structure_activity_pairs.output.activity_cliff_pair_summary,
        pair_cache_scaffold_groups=rules.structure_activity_pairs.output.activity_cliff_scaffold_groups,
        pair_cache_activity=rules.structure_activity_pairs.output.phenotypic_activity,
        pair_cache_overlap=rules.structure_activity_pairs.output.neighbor_overlap,
        pair_cache_moa=rules.structure_activity_pairs.output.moa_coherence,
        substructure_summary=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/summary.json",
        scaffold_coherence=f"data/processed/substructure-morphology/{DEFAULT_DATASET}_active/scaffold_coherence.csv",
        scaffold_series_summary=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/summary.json",
        scaffold_analysis=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/scaffold_analysis.parquet",
        cliff_scaffolds=f"data/processed/activity-cliffs-phenoseeker/{DEFAULT_DATASET}/cliff_scaffolds.csv",
        script="notebooks/figure-structure-activity.py",
        helpers=FIGURE_HELPERS
    output:
        similarity_map_png="data/processed/figures/figure_suppl_structure/similarity_map.png",
        similarity_map_pdf="data/processed/figures/figure_suppl_structure/similarity_map.pdf",
        similar_structure_morphology_png="data/processed/figures/figure_suppl_structure/similar_structure_morphology.png",
        similar_structure_morphology_pdf="data/processed/figures/figure_suppl_structure/similar_structure_morphology.pdf",
        neighbor_overlap_png="data/processed/figures/figure_suppl_structure/neighbor_overlap.png",
        neighbor_overlap_pdf="data/processed/figures/figure_suppl_structure/neighbor_overlap.pdf",
        moa_coherence_png="data/processed/figures/figure_suppl_structure/moa_coherence.png",
        moa_coherence_pdf="data/processed/figures/figure_suppl_structure/moa_coherence.pdf",
        pair_classes_png="data/processed/figures/figure_suppl_structure/pair_classes.png",
        pair_classes_pdf="data/processed/figures/figure_suppl_structure/pair_classes.pdf",
        thresholds_png="data/processed/figures/figure_suppl_structure/thresholds.png",
        thresholds_pdf="data/processed/figures/figure_suppl_structure/thresholds.pdf",
        scaffold_coherence_png="data/processed/figures/figure_suppl_structure/scaffold_coherence.png",
        scaffold_coherence_pdf="data/processed/figures/figure_suppl_structure/scaffold_coherence.pdf",
        activity_cliff_pairs_png="data/processed/figures/figure_suppl_structure/activity_cliff_pairs.png",
        activity_cliff_pairs_pdf="data/processed/figures/figure_suppl_structure/activity_cliff_pairs.pdf",
        moa_coherence_csv="data/processed/figures/figure_suppl_structure/moa_coherence.csv",
        similarity_quadrants_csv="data/processed/figures/figure_suppl_structure/similarity_quadrants.csv",
        threshold_sensitivity_csv="data/processed/figures/figure_suppl_structure/threshold_sensitivity.csv",
        scaffold_coherence_csv="data/processed/figures/figure_suppl_structure/scaffold_coherence.csv",
        phenotypic_cliffs_csv="data/processed/figures/figure_suppl_structure/phenotypic_cliffs.csv",
        activity_cliff_pairs_parquet="data/processed/figures/figure_suppl_structure/activity_cliff_pairs.parquet",
        activity_cliff_pair_summary_csv="data/processed/figures/figure_suppl_structure/activity_cliff_pair_summary.csv",
        activity_cliff_threshold_sensitivity_csv="data/processed/figures/figure_suppl_structure/activity_cliff_threshold_sensitivity.csv",
        activity_cliff_scaffold_groups_csv="data/processed/figures/figure_suppl_structure/activity_cliff_scaffold_groups.csv",
        phenoseeker_procedure_transfer_audit_csv="data/processed/figures/figure_suppl_structure/phenoseeker_procedure_transfer_audit.csv",
        activity_cliff_scaffold_groups_metadata="data/processed/figures/figure_suppl_structure/activity_cliff_scaffold_groups__metadata.yml",
        legend="data/processed/figures/figure_suppl_structure/legend.md",
        summary="data/processed/figures/figure_suppl_structure/summary.json"
    log:
        "logs/figures/figure_suppl_structure.log"
    shell:
        "pixi run -e cheminformatics python {input.script} "
        "--pair-cache-summary {input.pair_cache_summary} "
        "--substructure-summary {input.substructure_summary} "
        "--scaffold-coherence {input.scaffold_coherence} "
        "--scaffold-series-summary {input.scaffold_series_summary} "
        "--scaffold-analysis {input.scaffold_analysis} --cliff-scaffolds {input.cliff_scaffolds} "
        "--output $(dirname {output.summary}) --dpi {DEFAULT_DPI} 2>&1 | tee {log}"


rule figure_suppl_chemical_properties:
    """Build supplementary chemical properties through the chemical-space analysis script."""
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db="data/processed/copairs_results.duckdb",
        analysis_config="configs/snakemake/analysis.yaml",
        structure_umap="data/processed/chemical-space/all/structure_umap.h5ad",
        script="notebooks/2.04a-ss-chemical-space.py",
        helpers=FIGURE_HELPERS
    output:
        figure_png="data/processed/figures/figure_suppl_chemical_properties/supp_chemical_properties.png",
        figure_pdf="data/processed/figures/figure_suppl_chemical_properties/supp_chemical_properties.pdf",
        top_scaffolds_csv="data/processed/figures/figure_suppl_chemical_properties/top_scaffolds.csv",
        legend="data/processed/figures/figure_suppl_chemical_properties/legend.md",
        summary="data/processed/figures/figure_suppl_chemical_properties/summary.json"
    log:
        "logs/figures/figure_suppl_chemical_properties.log"
    shell:
        "pixi run -e cheminformatics python {input.script} --publication-only "
        "--publication-figure figure_suppl_chemical_properties "
        "--output $(dirname {output.summary}) --structure-umap-input {input.structure_umap} "
        "--dpi {DEFAULT_DPI} 2>&1 | tee {log}"


rule figure_positive_control_reproducibility:
    """Build the positive-control heatmap as an explicit target."""
    input:
        copairs_db="data/processed/copairs_results.duckdb",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/figure-positive-control-reproducibility.py",
        query="src/jump_production/queries.py",
        helpers=FIGURE_HELPERS
    output:
        png="data/processed/figures/figure_positive_control_reproducibility/poscon_heatmap.png",
        pdf="data/processed/figures/figure_positive_control_reproducibility/poscon_heatmap.pdf",
        table="data/processed/figures/figure_positive_control_reproducibility/poscon_heatmap.csv",
        legend="data/processed/figures/figure_positive_control_reproducibility/legend.md",
        summary="data/processed/figures/figure_positive_control_reproducibility/summary.json"
    log:
        "logs/figures/figure_positive_control_reproducibility.log"
    shell:
        "pixi run python {input.script:q} --copairs-db {input.copairs_db:q} "
        "--metadata-db {input.metadata_db:q} --dpi {DEFAULT_DPI} 2>&1 | tee {log:q}"


rule all_figures:
    """Build the six paper-figure directories without a catalog landing page."""
    input:
        rules.figure_compound_collection.output,
        rules.figure_phenotypic_consistency.output,
        rules.figure_compound_activity_associations.output,
        rules.figure_opentargets_group_consistency.output,
        rules.figure_suppl_structure.output,
        rules.figure_suppl_chemical_properties.output
