# -*- coding: utf-8 -*-
# Exploration notebooks (phase 0) - issue investigations and one-off analyses
#
# These rules provide reproducibility for exploration notebooks without
# including them in the main pipeline. Run with: snakemake explore -j 4
#
# Individual notebooks can be run with: snakemake explore_0_01 -j 1
#
# Uses DEFAULT_* constants from common.smk for dataset/preprocessing configuration

# Note: DEFAULT_DATASET, DEFAULT_PREPROCESSING, DEFAULT_FILTER, DEFAULT_ACTIVITY_PARAMS
# are defined in common.smk and available here via Snakefile include chain

# =============================================================================
# Aggregate target: run all exploration notebooks
# =============================================================================

rule explore:
    """Run all exploration notebooks."""
    input:
        "data/processed/exploration/.complete"


rule all_explore:
    """Aggregate all exploration outputs."""
    input:
        f"data/processed/exploration/0.01/{DEFAULT_DATASET}/.complete",
        "data/processed/exploration/0.02/.complete",  # No dataset (synthetic demo)
        f"data/processed/exploration/0.04/{DEFAULT_DATASET}/.complete",
        f"data/processed/exploration/0.05/{DEFAULT_DATASET}/.complete",
        f"data/processed/exploration/0.06/{DEFAULT_DATASET}/.complete",
        f"data/processed/exploration/0.07/{DEFAULT_DATASET}/.complete",
        "data/processed/circos-plots/compound_plate_metadata_circos.png",
        "data/processed/gene-compound-matching/summary.json",
        "data/processed/dataset-storage-sizes/summary.json",
    output:
        touch("data/processed/exploration/.complete")


# =============================================================================
# 0.01: Cosine vs Absolute Cosine comparison (Issue #22)
# =============================================================================

rule explore_0_01:
    """Compare cosine vs absolute cosine distance metrics for consistency.

    Creates scatter plot showing nMAP values for both metrics.

    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/0.01-ss-cosine-vs-abs-cosine.py",
    output:
        plot=f"data/processed/exploration/0.01/{DEFAULT_DATASET}/cosine_vs_abs_cosine_scatter.png",
        summary=f"data/processed/exploration/0.01/{DEFAULT_DATASET}/summary.json",
        complete=touch(f"data/processed/exploration/0.01/{DEFAULT_DATASET}/.complete"),
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/exploration/0.01/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} 2>&1 | tee {log}"


# =============================================================================
# 0.02: Stratification effect demo (Issue #29)
# =============================================================================

rule explore_0_02:
    """Demo: Why stratification had minimal effect on ROC-AUC.

    Pure synthetic demo showing law of large numbers effect.

    """
    input:
        script="notebooks/0.02-ss-stratification-effect.py",
    output:
        summary="data/processed/exploration/0.02/summary.json",
        complete=touch("data/processed/exploration/0.02/.complete")
    log:
        "logs/exploration/0.02-stratification-effect.log"
    conda:
        None  # Uses pixi environment
    shell:
        "pixi run -e deepchem python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 0.04: UMAP seed stability scan (Issue #17)
# =============================================================================

rule explore_0_04:
    """UMAP seed stability scan: compare layouts across random seeds.

    Runs 5 seeds × 2 metrics (cosine, euclidean) to verify layout stability.

    """
    input:
        # Profiles loaded via load_profiles() - declare db as proxy for data availability
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/0.04-ss-umap-seed-scan.py",
    output:
        plot=f"data/processed/exploration/0.04/{DEFAULT_DATASET}/seed_comparison.png",
        summary=f"data/processed/exploration/0.04/{DEFAULT_DATASET}/summary.json",
        complete=touch(f"data/processed/exploration/0.04/{DEFAULT_DATASET}/.complete"),
    params:
        dataset=DEFAULT_DATASET,
    log:
        f"logs/exploration/0.04/{DEFAULT_DATASET}.log"
    threads:
        1
    resources:
        gpu=1
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python {input.script} --dataset {params.dataset} 2>&1 | tee {log}"


# =============================================================================
# 0.05: Activity threshold sensitivity sweep (Issue #22)
# =============================================================================

rule explore_0_05:
    """Analyze activity threshold sensitivity for consistency results.

    Sweeps activity p-value thresholds (0.05-0.30) to measure impact on
    consistency results. Helps determine optimal threshold selection.

    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/0.05-ss-threshold-sweep-analysis.py",
    output:
        summary_plot=f"data/processed/exploration/0.05/{DEFAULT_DATASET}/threshold_sweep_summary.png",
        tradeoff_plot=f"data/processed/exploration/0.05/{DEFAULT_DATASET}/threshold_sweep_tradeoff.png",
        summary=f"data/processed/exploration/0.05/{DEFAULT_DATASET}/summary.json",
        complete=touch(f"data/processed/exploration/0.05/{DEFAULT_DATASET}/.complete"),
    params:
        dataset=DEFAULT_DATASET,
        outdir=f"data/processed/exploration/0.05/{DEFAULT_DATASET}",
    log:
        f"logs/exploration/0.05/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --outdir {params.outdir} 2>&1 | tee {log}"


# =============================================================================
# 0.06: Activity vs consistency relationship (Issue #35)
# =============================================================================

rule explore_0_06:
    """Analyze relationship between activity p-value and consistency contribution.

    Explores whether marginally active compounds still contribute meaningful
    consistency signal. Includes effect size vs replicate analysis.

    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/0.06-ss-activity-consistency-relationship.py",
    output:
        plot=f"data/processed/exploration/0.06/{DEFAULT_DATASET}/activity_vs_consistency.png",
        summary=f"data/processed/exploration/0.06/{DEFAULT_DATASET}/summary.json",
        complete=touch(f"data/processed/exploration/0.06/{DEFAULT_DATASET}/.complete"),
    params:
        dataset=DEFAULT_DATASET,
        outdir=f"data/processed/exploration/0.06/{DEFAULT_DATASET}",
    log:
        f"logs/exploration/0.06/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --outdir {params.outdir} 2>&1 | tee {log}"


# =============================================================================
# 0.07: Statistical power analysis (Issue #35)
# =============================================================================

rule explore_0_07:
    """Analyze statistical power for activity significance.

    Shows that high-consistency marginal compounds have moderate effect sizes
    but insufficient statistical power due to low replicates.

    """
    input:
        db="data/processed/copairs_results.duckdb",
        script="notebooks/0.07-ss-power-analysis.py",
    output:
        plot=f"data/processed/exploration/0.07/{DEFAULT_DATASET}/power_analysis.png",
        summary=f"data/processed/exploration/0.07/{DEFAULT_DATASET}/summary.json",
        complete=touch(f"data/processed/exploration/0.07/{DEFAULT_DATASET}/.complete"),
    params:
        dataset=DEFAULT_DATASET,
        outdir=f"data/processed/exploration/0.07/{DEFAULT_DATASET}",
    log:
        f"logs/exploration/0.07/{DEFAULT_DATASET}.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --outdir {params.outdir} 2>&1 | tee {log}"


# =============================================================================
# 0.10: Circos plots of experiment metadata (Issue #10)
# =============================================================================

rule explore_0_10:
    """Circos plots of plate metadata and experiment properties.

    Visualizes JUMP experiment structure: plate counts, batches, plate types,
    microscopes, imaging modes, CellProfiler versions.

    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/0.10-nc-circos-plots.py",
    output:
        plate_plot="data/processed/circos-plots/compound_plate_metadata_circos.png",
        props_plot="data/processed/circos-plots/compound_experiment_properties_circos.png",
        summary="data/processed/circos-plots/summary.json",
    log:
        "logs/exploration/0.10.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 0.14: Direct gene-compound matching is at the random null (motivates MOTIVE)
# =============================================================================

rule explore_0_14:
    """Direct gene-compound morphological matching is ~random, motivating MOTIVE's GNN.

    Compounds and ORF genes co-embedded in ONE sphering+Harmony space (canonical JUMP
    'all' v1.0b profile, split by PlateType + perturbation-median) with MOTIVE's curated
    ground truth. Per gene, is a true compound in its top-1% by cosine? vs a label-
    permutation null. Self-fetches the canonical profile + MOTIVE maps from public S3
    (no-sign-request) and caches the 342 MB perturbation aggregate; first run pays the
    one-time 2.5 GB scan, later runs are seconds. Numbers-only node (no figure).

    """
    input:
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        script="notebooks/0.14-ss-gene-compound-matching.py",
    output:
        summary="data/processed/gene-compound-matching/summary.json",
    log:
        "logs/exploration/0.14.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# =============================================================================
# 0.15: Storage footprint of single-cell + well-level profiles on CPG
# =============================================================================

rule explore_0_15:
    """Storage sizes of JUMP single-cell (.sqlite) and well-level (.csv) profiles.

    Sums the raw per-plate backend/ files on the Cell Painting Gallery (cpg0016-jump,
    per-source) at two scopes - ALL released sources and COMPOUND only (matching the
    1.53B single-cell count). Replaces the stale "36 GB" well-level figure. S3 listing
    only (no download), public no-sign-request. Numbers-only node.
    Related: paper data-availability sizes.
    """
    input:
        script="notebooks/0.15-ss-dataset-storage-sizes.py",
    output:
        summary="data/processed/dataset-storage-sizes/summary.json",
    log:
        "logs/exploration/0.15.log"
    shell:
        "pixi run python {input.script} 2>&1 | tee {log}"


# 0.19 graduated to `rule opentargets_target_alignment` in rules/analysis.smk when its
# panels joined the compound-activity-association figure; the script is now
# notebooks/2.46-ss-opentargets-target-alignment.py.
