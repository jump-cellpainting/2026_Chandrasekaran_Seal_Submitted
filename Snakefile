# -*- coding: utf-8 -*-
# Run core (db + copairs): snakemake core -j 4
# Run all (everything):    snakemake -j 4
# Run one activity run:    snakemake data/processed/copairs/runs/activity/compound_no_source7__feat_all__activity_no_target2__all_sources/results -j 4

import os
from pathlib import Path

# Load matrices & settings (Snakemake merges multiple configfiles)
configfile: "configs/snakemake/copairs_runs.yaml"
configfile: "configs/snakemake/analysis.yaml"

# ---------------------------------------------------------------------
# Include modular rule files
# ---------------------------------------------------------------------
include: "rules/common.smk"  # Helper functions and config variables (must be first)
include: "rules/profiles.smk"
include: "rules/processing.smk"
include: "rules/anndata.smk"
include: "rules/umap.smk"
include: "rules/copairs.smk"
include: "rules/analysis.smk"
include: "rules/figures.smk"
include: "rules/batch_and_source7.smk"
include: "rules/exploration.smk"
include: "rules/srijit.smk"

# ---------------------------------------------------------------------
# Core target: augmented database + copairs metrics
# ---------------------------------------------------------------------
rule core:
    input:
        # key intermediates
        "data/interim/jump_metadata_augmented.duckdb",
        # copairs metrics (activity + consistency)
        "data/processed/copairs/runs/.complete",

# ---------------------------------------------------------------------
# Default target: everything (core + all downstream analysis)
# Uses DEFAULT_DATASET and DEFAULT_PREPROCESSING from common.smk (set via config)
# ---------------------------------------------------------------------
rule all:
    input:
        # core pipeline (db + copairs)
        rules.core.input,
        # profile comparisons
        "data/processed/profile-comparison/.complete",
        # UMAP visualizations
        "data/processed/umap/.complete",
        # target consistency visualizations
        "data/processed/target-consistency/.complete",
        # paper figures (direct notebook scripts over canonical analysis outputs)
        rules.all_figures.input,
        # data quality analysis (cell density vs activity, edge effects) - hierarchical path
        f"data/processed/data-quality/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/.complete",
        # cross-source reproducibility analysis - hierarchical path
        f"data/processed/cross-source-reproducibility/{DEFAULT_DATASET}/.complete",
        # chemical space characterization (no activity config - uses metadata only)
        "data/processed/chemical-space/all/.complete",
        # fingerprint metrics evaluation (L2 norm ≈ Tanimoto, see Issue #26)
        "data/processed/fingerprint-metrics/.complete",
        # compound traits vs activity - hierarchical path
        f"data/processed/compound-traits/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/.complete",
        # structure-morphology relationships (both morgan and chemberta) - hierarchical path
        f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/morgan/.complete",
        f"data/processed/structure-morphology/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/chemberta/.complete",
        # phenotype prediction from chemical structure (Morgan FP only, see Issue #29)
        f"data/processed/phenotype-prediction/{DEFAULT_DATASET}/{DEFAULT_PREPROCESSING}/morgan/.complete",
        # phenotypic-activity headline numbers (envelope summary from the copairs store)
        "data/processed/activity-summary/summary.json",
        # consolidated ground-truth reference numbers (capstone - downstream of the summaries it reads)
        "data/processed/dataset_reference/dataset_reference.json",
        # Figure 1A structural numbers + Venn-diagram inputs (downstream of the metadata DB)
        "data/processed/dataset_structure/dataset_structure.json",
        # PDF reports
        rules.all_reports.input,

# ---------------------------------------------------------------------
# Note: Profile data should be downloaded before running the pipeline
# Use: just get-inputs (from team S3) or just get-from-sources (from original sources)
# This ensures profiles are treated as input data, similar to external/
# ---------------------------------------------------------------------
