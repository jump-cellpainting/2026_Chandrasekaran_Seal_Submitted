# -*- coding: utf-8 -*-
# Copairs activity and consistency metrics
# Also includes union profile creation and results database aggregation
#
# Helper functions: act_name, activity_path, cons_name, cons_prefiltered_name,
#                   get_consistency_params, get_activity_input (in common.smk)
# Config variables: ACTIVITY, CONSISTENCY, CONSISTENCY_PREFILTERED,
#                   ACTIVITY_WITHINSOURCE, ACTIVITY_CROSSSOURCE (in common.smk)


# Prefer prefiltered rule for *_active_union datasets
ruleorder: copairs_consistency_prefiltered > copairs_consistency

# Peak memory of one consistency job, measured at 11 GB with the null cache restored and
# 9 to 13 GB without it on 16 CPUs. `just reproduce` keeps the sum over running jobs
# under the machine's memory; without a limit on the command line this changes nothing.
COPAIRS_CONSISTENCY_MEM_MB = 12000


rule all_copairs:
    """Aggregate rule for all copairs outputs."""
    input:
        "data/processed/copairs_results.duckdb"
    output:
        touch("data/processed/copairs/runs/.complete")


rule create_union_profiles:
    """Create parquet files filtered to union of active perturbations.

    This enables comprehensive comparison between CellProfiler and Deep Learning
    by evaluating both methods on all perturbations active in either method.
    """
    input:
        activity1="data/processed/copairs/runs/activity/compound_no_source7__feat_all__activity_no_target2__all_sources__default/results/activity_map_results.csv",
        activity2="data/processed/copairs/runs/activity/compound_DL_CPCNN_no_source7__feat_all__activity_no_target2__all_sources__default/results/activity_map_results.csv",
        parquet1="data/raw/profiles/compound_no_source7.parquet",
        parquet2="data/raw/profiles/compound_DL_CPCNN_no_source7.parquet",
    output:
        parquet1="data/raw/profiles/compound_no_source7_active_union.parquet",
        parquet2="data/raw/profiles/compound_DL_CPCNN_no_source7_active_union.parquet",
    log:
        "logs/copairs/create_union_profiles.log"
    shell:
        """
        pixi run python -m jump_production.processing.create_union_profiles \
            --activity1 {input.activity1} \
            --activity2 {input.activity2} \
            --parquet1 {input.parquet1} \
            --parquet2 {input.parquet2} \
            --output-dir data/raw/profiles \
            --suffix active_union \
            2>&1 | tee {log}
        """


rule create_union_profiles_with_source7:
    """Create parquet files filtered to union of active perturbations (with source_7).

    Same as create_union_profiles but for with_source7 datasets.
    """
    input:
        activity1="data/processed/copairs/runs/activity/compound_with_source7__feat_all__activity_no_target2__all_sources__default/results/activity_map_results.csv",
        activity2="data/processed/copairs/runs/activity/compound_DL_CPCNN_with_source7__feat_all__activity_no_target2__all_sources__default/results/activity_map_results.csv",
        parquet1="data/raw/profiles/compound_with_source7.parquet",
        parquet2="data/raw/profiles/compound_DL_CPCNN_with_source7.parquet",
    output:
        parquet1="data/raw/profiles/compound_with_source7_active_union.parquet",
        parquet2="data/raw/profiles/compound_DL_CPCNN_with_source7_active_union.parquet",
    log:
        "logs/copairs/create_union_profiles_with_source7.log"
    shell:
        """
        pixi run python -m jump_production.processing.create_union_profiles \
            --activity1 {input.activity1} \
            --activity2 {input.activity2} \
            --parquet1 {input.parquet1} \
            --parquet2 {input.parquet2} \
            --output-dir data/raw/profiles \
            --suffix active_union \
            2>&1 | tee {log}
        """


rule filter_source7_exclusive:
    """Filter full profiles to exclude non-exclusive source_7 compounds.

    Keeps all non-source_7 wells plus source_7 wells for compounds that
    were profiled ONLY at source_7 (3,191 compounds). This avoids dose
    confounds (source_7 used 0.625 µM vs 10 µM at other sources).
    """
    input:
        parquet="data/raw/profiles/{dataset}.parquet",
        db="data/interim/jump_metadata_augmented.duckdb",
    output:
        "data/raw/profiles/{dataset,(compound|compound_DL_CPCNN)}_with_source7.parquet"
    log:
        "logs/copairs/filter_source7_exclusive_{dataset}.log"
    shell:
        """
        pixi run python -m jump_production.processing.filter_source7_exclusive \
            --input {input.parquet} \
            --output {output} \
            --db {input.db} \
            2>&1 | tee {log}
        """


rule copairs_activity:
    """Activity analysis with different parameter configurations.

    activity_params='default': Standard activity (default grouping)
    activity_params='withinsource': Positive pairs must match on source
    activity_params='crosssource': Positive pairs must differ on source
    """
    input:
        parquet=lambda w: f"data/raw/profiles/{w.dataset}.parquet",
        activity_cfg="configs/copairs/metrics/activity.yaml",
        activity_params_cfg=lambda w: f"configs/copairs/metrics/activity_params/{w.activity_params}.yaml",
        dataset_cfg=lambda w: f"configs/copairs/metrics/dataset/{w.dataset}.yaml",
        columns_cfg=lambda w: f"configs/copairs/metrics/columns/{w.columns}.yaml",
        preprocessing_01_metadata_cfg="configs/copairs/metrics/preprocessing_01_metadata/jump_standard.yaml",
        preprocessing_02_core_cfg=lambda w: f"configs/copairs/metrics/preprocessing_02_core/{w.preprocessing_02_core}.yaml",
        filter_cfg=lambda w: f"configs/copairs/metrics/filter/{w.filter}.yaml",
        db="data/interim/jump_metadata_augmented.duckdb"
    output:
        "data/processed/copairs/runs/activity/{dataset}__{columns}__{preprocessing_02_core}__{filter}__{activity_params}/results/activity_map_results.csv"
    wildcard_constraints:
        activity_params="(default|withinsource|crosssource)"
    params:
        tag=lambda w: f"{w.dataset}__{w.columns}__{w.preprocessing_02_core}__{w.filter}"
    resources:
        mem_mb=4000
    log:
        "logs/copairs/activity/{dataset}__{columns}__{preprocessing_02_core}__{filter}__{activity_params}.log"
    shell:
        r"""
        export COPAIRS_OUTPUT="$(pwd)"
        pixi run copairs-runner \
          --config-dir configs/copairs/metrics --config-name activity \
          activity_params={wildcards.activity_params} \
          dataset={wildcards.dataset} columns={wildcards.columns} \
          preprocessing_02_core={wildcards.preprocessing_02_core} filter={wildcards.filter} \
          2>&1 | tee {log}
        """


rule copairs_consistency:
    input:
        parquet=lambda w: f"data/raw/profiles/{w.dataset}.parquet",
        consistency_cfg="configs/copairs/metrics/consistency.yaml",
        dataset_cfg=lambda w: f"configs/copairs/metrics/dataset/{w.dataset}.yaml",
        columns_cfg=lambda w: f"configs/copairs/metrics/columns/{w.columns}.yaml",
        preprocessing_01_metadata_cfg="configs/copairs/metrics/preprocessing_01_metadata/jump_with_annotations.yaml",
        preprocessing_02_core_cfg=lambda w: f"configs/copairs/metrics/preprocessing_02_core/{w.preprocessing_02_core_cons}.yaml",
        filter_cfg=lambda w: f"configs/copairs/metrics/filter/{w.filter}.yaml",
        target_cfg=lambda w: f"configs/copairs/metrics/target_column/{w.target_column}.yaml",
        # Depends on augmented database
        db="data/interim/jump_metadata_augmented.duckdb",
        # require matching activity run (empty list for prefiltered datasets)
        activity=get_activity_input
    output:
        "data/processed/copairs/runs/consistency/{dataset}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}/results/consistency_map_results.csv"
    wildcard_constraints:
        # Exclude prefiltered datasets (handled by copairs_consistency_prefiltered rule)
        dataset="(?!.*_active_union$).*"
    params:
        tag=lambda w: f"{w.dataset}__{w.columns}__{w.preprocessing_02_core_cons}__{w.filter}__{w.target_column}__{w.distance}"
    resources:
        mem_mb=COPAIRS_CONSISTENCY_MEM_MB
    log:
        "logs/copairs/consistency/{dataset}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}.log"
    shell:
        r"""
        export COPAIRS_OUTPUT="$(pwd)"
        pixi run copairs-runner \
          --config-dir configs/copairs/metrics --config-name consistency \
          dataset={wildcards.dataset} columns={wildcards.columns} \
          preprocessing_02_core={wildcards.preprocessing_02_core_cons} \
          filter={wildcards.filter} target_column={wildcards.target_column} \
          distance={wildcards.distance} \
          2>&1 | tee {log}
        """


rule copairs_consistency_prefiltered:
    """Consistency for prefiltered datasets (no activity dependency).

    These datasets are pre-filtered to union of active perturbations.
    """
    input:
        parquet=lambda w: f"data/raw/profiles/{w.dataset}.parquet",
        consistency_cfg="configs/copairs/metrics/consistency.yaml",
        dataset_cfg=lambda w: f"configs/copairs/metrics/dataset/{w.dataset}.yaml",
        columns_cfg=lambda w: f"configs/copairs/metrics/columns/{w.columns}.yaml",
        preprocessing_01_metadata_cfg="configs/copairs/metrics/preprocessing_01_metadata/jump_with_annotations.yaml",
        preprocessing_02_core_cfg=lambda w: f"configs/copairs/metrics/preprocessing_02_core/{w.preprocessing_02_core_cons}.yaml",
        filter_cfg=lambda w: f"configs/copairs/metrics/filter/{w.filter}.yaml",
        target_cfg=lambda w: f"configs/copairs/metrics/target_column/{w.target_column}.yaml",
        # Depends on augmented database
        db="data/interim/jump_metadata_augmented.duckdb",
        # No activity dependency - dataset is pre-filtered to active perturbations
    output:
        "data/processed/copairs/runs/consistency/{dataset,(compound_no_source7_active_union|compound_DL_CPCNN_no_source7_active_union|compound_with_source7_active_union|compound_DL_CPCNN_with_source7_active_union)}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}/results/consistency_map_results.csv"
    params:
        tag=lambda w: f"{w.dataset}__{w.columns}__{w.preprocessing_02_core_cons}__{w.filter}__{w.target_column}__{w.distance}"
    resources:
        mem_mb=COPAIRS_CONSISTENCY_MEM_MB
    log:
        "logs/copairs/consistency_prefiltered/{dataset}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}.log"
    shell:
        r"""
        export COPAIRS_OUTPUT="$(pwd)"
        pixi run copairs-runner \
          --config-dir configs/copairs/metrics --config-name consistency \
          dataset={wildcards.dataset} columns={wildcards.columns} \
          preprocessing_02_core={wildcards.preprocessing_02_core_cons} \
          filter={wildcards.filter} target_column={wildcards.target_column} \
          distance={wildcards.distance} \
          2>&1 | tee {log}
        """


rule copairs_consistency_sweep:
    """Run consistency with threshold sweep for sensitivity analysis (Issue #22).

    Output goes to consistency/ folder (not consistency_sweep/) since we use
    --config-name consistency, which produces files named consistency_map_results.csv.
    The activity_threshold wildcard differentiates sweep runs from regular runs.
    """
    input:
        parquet=lambda w: f"data/raw/profiles/{w.dataset}.parquet",
        consistency_cfg="configs/copairs/metrics/consistency.yaml",
        dataset_cfg=lambda w: f"configs/copairs/metrics/dataset/{w.dataset}.yaml",
        columns_cfg=lambda w: f"configs/copairs/metrics/columns/{w.columns}.yaml",
        preprocessing_01_metadata_cfg="configs/copairs/metrics/preprocessing_01_metadata/jump_with_annotations.yaml",
        preprocessing_02_core_cfg=lambda w: f"configs/copairs/metrics/preprocessing_02_core/{w.preprocessing_02_core_cons}.yaml",
        filter_cfg=lambda w: f"configs/copairs/metrics/filter/{w.filter}.yaml",
        target_cfg=lambda w: f"configs/copairs/metrics/target_column/{w.target_column}.yaml",
        db="data/interim/jump_metadata_augmented.duckdb",
        activity=lambda w: get_activity_input(w, CONSISTENCY_SWEEP)
    output:
        "data/processed/copairs/runs/consistency/{dataset}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}__{activity_threshold}/results/consistency_map_results.csv"
    wildcard_constraints:
        activity_threshold=r"\d+\.\d+"
    params:
        tag=lambda w: f"{w.dataset}__{w.columns}__{w.preprocessing_02_core_cons}__{w.filter}__{w.target_column}__{w.distance}__{w.activity_threshold}"
    resources:
        mem_mb=COPAIRS_CONSISTENCY_MEM_MB
    log:
        "logs/copairs/consistency_sweep/{dataset}__{columns}__{preprocessing_02_core_cons}__{filter}__{target_column}__{distance}__{activity_threshold}.log"
    shell:
        r"""
        export COPAIRS_OUTPUT="$(pwd)"
        pixi run copairs-runner \
          --config-dir configs/copairs/metrics --config-name consistency \
          dataset={wildcards.dataset} columns={wildcards.columns} \
          preprocessing_02_core={wildcards.preprocessing_02_core_cons} \
          filter={wildcards.filter} target_column={wildcards.target_column} \
          distance={wildcards.distance} \
          preprocessing_02_core.activity_threshold={wildcards.activity_threshold} \
          'hydra.run.dir=${{oc.env:COPAIRS_OUTPUT,.}}/data/processed/copairs/runs/consistency/{wildcards.dataset}__{wildcards.columns}__{wildcards.preprocessing_02_core_cons}__{wildcards.filter}__{wildcards.target_column}__{wildcards.distance}__{wildcards.activity_threshold}/results/' \
          2>&1 | tee {log}
        """


rule build_copairs_results_db:
    """Aggregate all activity/consistency results into unified DuckDB."""
    input:
        # All activity outputs (using activity_path helper for consistency)
        [activity_path(it) for it in ACTIVITY],
        # Within-source activity outputs
        [activity_path(it, "withinsource") for it in ACTIVITY_WITHINSOURCE],
        # Cross-source activity outputs
        [activity_path(it, "crosssource") for it in ACTIVITY_CROSSSOURCE],
        # All consistency outputs
        [f"data/processed/copairs/runs/consistency/{cons_name(it)}/results/consistency_map_results.csv" for it in CONSISTENCY],
        # All prefiltered consistency outputs (for fair profile comparison)
        [f"data/processed/copairs/runs/consistency/{cons_prefiltered_name(it)}/results/consistency_map_results.csv" for it in CONSISTENCY_PREFILTERED],
        # All threshold sweep consistency outputs (Issue #22 sensitivity analysis)
        # Lives in consistency/ folder since it uses --config-name consistency
        [f"data/processed/copairs/runs/consistency/{cons_sweep_name(it)}/results/consistency_map_results.csv" for it in CONSISTENCY_SWEEP],
    output:
        "data/processed/copairs_results.duckdb"
    log:
        "logs/copairs/build_results_db.log"
    shell:
        "pixi run python -m jump_production.processing.build_copairs_results_db --output {output} 2>&1 | tee {log}"
