# -*- coding: utf-8 -*-
# UMAP computation and visualization (two-stage pipeline)
# Stage 1: Compute UMAP embeddings (expensive, GPU) → minimal h5ad files
#          Saves only obs metadata + obsm embeddings (no X matrix) to save disk
# Stage 2: Generate plots from minimal h5ad (cheap, CPU) → PNG files
# Supports multiple distance metrics (cosine, euclidean) via wildcard
#
# Helper functions: get_umap_combinations, umap_compute_inputs (in common.smk)
# Config variables: UMAP_COMBOS (in common.smk)
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: ancient() on copairs_results.duckdb
# ─────────────────────────────────────────────────────────────────────────────
# UMAP rules use ancient() on copairs_results.duckdb to avoid costly rebuilds
# (~10 min GPU compute) when copairs is frequently updated.
#
# What UMAP actually needs from copairs:
#   - umap_compute (active filter): activity_results.below_corrected_p to decide
#     which perturbations to include → affects UMAP geometry
#   - umap_plot: activity_results.below_corrected_p for coloring points
#
# Risk: If activity results change (new p-values, different significance calls),
# UMAPs may show stale data. Consistency results changes are safe to ignore.
#
# To force UMAP rebuild after activity changes:
#   snakemake --forcerun umap_compute umap_plot -j4
#
# Alternative (not implemented): Create sentinel files with hashes of relevant
# activity_results rows, have UMAP depend on those instead of the full DB.
# ─────────────────────────────────────────────────────────────────────────────


rule all_umap:
    input:
        well=[f"data/processed/umap/{d}/{m}_{f}/well_umap_combined.png" for d, m, f in UMAP_COMBOS],
        perturbation=[f"data/processed/umap/{d}/{m}_{f}/perturbation_umap_combined.png" for d, m, f in UMAP_COMBOS]
    output:
        touch("data/processed/umap/.complete")


rule umap_compute:
    """Compute UMAP embeddings for JUMP compound profiles.

    Handles both regular and active-only UMAPs via {filter} wildcard.
    Active filtering requires copairs_results.duckdb for significance data.
    Uses GPU acceleration via rapids_singlecell when available.
    """
    input:
        unpack(umap_compute_inputs)
    output:
        well="data/interim/anndata/{dataset}_{metric}_{filter}_umap.h5ad",
        perturbation="data/interim/anndata/{dataset}_perturbation_{metric}_{filter}_umap.h5ad"
    wildcard_constraints:
        metric="cosine|euclidean",
        filter="all|active"
    params:
        filter_flag=lambda w: "--filter-active" if w.filter == "active" else ""
    resources:
        gpu=1
    log:
        "logs/umap/compute/{dataset}__{metric}__{filter}.log"
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python -m jump_production.processing.umap {wildcards.dataset} --metric {wildcards.metric} {params.filter_flag} 2>&1 | tee {log}"


rule umap_plot:
    """Generate plots from UMAP embeddings.

    Handles both regular and active-only UMAPs via {filter} wildcard.
    Metadata is joined at visualization time from metadata_db.
    """
    input:
        h5ad_well="data/interim/anndata/{dataset}_{metric}_{filter}_umap.h5ad",
        h5ad_perturbation="data/interim/anndata/{dataset}_perturbation_{metric}_{filter}_umap.h5ad",
        metadata_db="data/interim/jump_metadata_augmented.duckdb",
        copairs_db=ancient("data/processed/copairs_results.duckdb"),
        script="notebooks/2.01-ss-umap-visualization.py"
    output:
        # Track combined plots as sentinels (script generates additional plots)
        "data/processed/umap/{dataset}/{metric}_{filter}/well_umap_combined.png",
        "data/processed/umap/{dataset}/{metric}_{filter}/perturbation_umap_combined.png",
        "data/processed/umap/{dataset}/{metric}_{filter}/summary.json"
    wildcard_constraints:
        metric="cosine|euclidean",
        filter="all|active"
    log:
        "logs/umap/plot/{dataset}__{metric}__{filter}.log"
    shell:
        "pixi run python {input.script} {wildcards.dataset} --metric {wildcards.metric} --filter {wildcards.filter} 2>&1 | tee {log}"
