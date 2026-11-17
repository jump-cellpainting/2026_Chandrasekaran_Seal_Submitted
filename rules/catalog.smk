# -*- coding: utf-8 -*-
# Results catalog: the index of analyses that the publication data browser reads.
#
# configs/results_catalog.yaml is the authored registry and the sole membership definition. Its
# entries span the `all`, `batch_and_source7`, `explore` and `srijit` targets, so this rule is its
# own target and is not part of `rule all`: building the catalog builds every analysis it lists.

import yaml

RESULTS_CATALOG_REGISTRY = "configs/results_catalog.yaml"
with open(RESULTS_CATALOG_REGISTRY) as _stream:
    _RESULTS_CATALOG_ENTRIES = yaml.safe_load(_stream)["entries"]


rule results_catalog:
    """Write the catalog of analyses, their descriptions and their declared files.

    The catalog computes no scientific value. It joins the registry's titles and sections with the
    summary envelopes, records each producing rule and script, and fails if a registered summary or
    declared file is missing. `just verify-baseline` then requires every envelope to be cataloged.
    """
    input:
        summaries=[entry["summary"] for entry in _RESULTS_CATALOG_ENTRIES if "summary" in entry],
        resources=[
            entry.get("requires", entry["path"])
            for entry in _RESULTS_CATALOG_ENTRIES
            if "summary" not in entry
        ],
        registry=RESULTS_CATALOG_REGISTRY,
        analysis_config="configs/snakemake/analysis.yaml",
        builder="src/jump_production/catalog.py",
        script="src/jump_production/processing/build_results_catalog.py",
    output:
        catalog="data/processed/catalog.json"
    params:
        default_dataset=DEFAULT_DATASET,
        default_columns=DEFAULT_COLUMNS,
        default_preprocessing=DEFAULT_PREPROCESSING,
        default_filter=DEFAULT_FILTER,
        default_activity_params=DEFAULT_ACTIVITY_PARAMS,
    log:
        "logs/analysis/results_catalog.log"
    shell:
        "pixi run python {input.script} "
        "--registry {input.registry} "
        "--output {output.catalog} "
        "--default-dataset {params.default_dataset} "
        "--default-columns {params.default_columns} "
        "--default-preprocessing {params.default_preprocessing} "
        "--default-filter {params.default_filter} "
        "--default-activity-params {params.default_activity_params} 2>&1 | tee {log}"
