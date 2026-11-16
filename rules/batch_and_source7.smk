# -*- coding: utf-8 -*-
# Batch correction and source 7 comparison sidebar analysis
# Investigation: include/exclude source 7, recipe vs in-repo Harmony, DL batch effects
#
# Related GitHub issues: #52, #54, #55, #56
# Run with: snakemake batch_and_source7 -j 4
#
# Helper functions and config variables from common.smk are available via include chain


rule source7_4way_comparison:
    """4-way source 7 comparison: CP vs DL x no_s7 vs with_s7 (#54).

    Union-based comparison on identical compound pools.
    Conclusion: include source 7, use CP (DL too batch-structured).
    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/1.02-ss-source7-4way-comparison.py",
    output:
        dilution="data/processed/source7-comparison/dilution_summary.csv",
        head_to_head="data/processed/source7-comparison/head_to_head.csv",
    log:
        "logs/analysis/source7_4way_comparison.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


rule source7_consistency_comparison:
    """Compare target consistency between no_source7 and with_source7 datasets (#52).

    Superseded by #54 but preserved for reference.
    Reads pre-computed target consistency CSVs.
    """
    input:
        "data/processed/target-consistency/.complete",
        script="scripts/compare_source7_consistency.py",
    output:
        "data/processed/source7-comparison/source7_consistency_comparison.md"
    log:
        "logs/analysis/source7_consistency_comparison.log"
    shell:
        "pixi run python {input.script} 1> {output} 2> {log}"


rule batch_effect_quantification:
    """Quantify DL batch effects via iLISI and kBET (#55).

    Confirms DL 3.5x worse batch mixing than CP by local metrics.
    Source 7 has no impact on batch structure.
    Requires GPU (rapids env).
    """
    input:
        expand("data/interim/anndata/{dataset}_cosine_all_umap.h5ad",
               dataset=["compound_no_source7", "compound_DL_CPCNN_no_source7",
                        "compound_with_source7", "compound_DL_CPCNN_with_source7"]),
        script="notebooks/2.02c-ss-source7-batch.py",
    output:
        csv="data/processed/batch-effect/batch_effect_summary.csv",
        plot="data/processed/batch-effect/batch_effect_comparison.png",
        summary="data/processed/batch-effect/summary.json",
    resources:
        gpu=1
    log:
        "logs/analysis/batch_effect_quantification.log"
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} 2>&1 | tee {log}"


rule harmony_comparison:
    """Compare Harmony batch correction variants for CP and DL features (#56).

    6-way comparison (2 feature types x 3 Harmony variants).
    Conclusion: recipe-Harmony is sufficient; CP outperforms DL on activity
    across all variants.
    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/1.03-ss-harmony-comparison.py",
    output:
        "data/processed/harmony-comparison/summary.json"
    log:
        "logs/analysis/harmony_comparison.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# =============================================================================
# Aggregate target
# =============================================================================

rule batch_and_source7:
    """Aggregate rule for all batch correction and source 7 comparison analyses."""
    input:
        rules.source7_4way_comparison.output,
        rules.source7_consistency_comparison.output,
        rules.batch_effect_quantification.output,
        rules.harmony_comparison.output,
