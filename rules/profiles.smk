# -*- coding: utf-8 -*-
# Profile batch correction (in-repo, via rapids_singlecell)
# Pre-Harmony profiles are downloaded via `just get-inputs`. This rule runs PCA +
# Harmony via rapids_singlecell to produce batch-corrected profile variants.
# These are separate from the default recipe-Harmony profiles (also downloaded).
#
# Variants (each has its own config file):
#   _rsc        — Metadata_Batch (122 batches, fine-grained)
#   _rsc_source — Metadata_Source (10 sources, coarse)

import yaml

# Map variant suffix → config file
BATCH_CORRECT_VARIANTS = {
    "rsc":        "configs/batch_correction.yaml",
    "rsc_source": "configs/batch_correction_source.yaml",
}

BATCH_CORRECT_DATASETS = [
    "compound_no_source7",
    "compound_DL_CPCNN_no_source7",
    "compound",
    "compound_DL_CPCNN",
]


rule batch_correct:
    """Batch-correct profiles using PCA + Harmony (GPU).

    Reads pre-Harmony parquet, joins batch metadata from DuckDB, runs PCA
    then Harmony via rapids_singlecell, saves corrected profiles.
    Variant suffix determines which config file (and batch key) is used.
    Output is separate from the default recipe-Harmony profiles.
    """
    input:
        parquet="data/raw/profiles/{dataset}_pre_harmony.parquet",
        db="data/interim/jump_metadata_augmented.duckdb",
        config=lambda w: BATCH_CORRECT_VARIANTS[w.variant],
        script="src/jump_production/processing/batch_correct.py",
    output:
        parquet="data/raw/profiles/{dataset}_{variant}.parquet",
    wildcard_constraints:
        dataset="|".join(BATCH_CORRECT_DATASETS),
        variant="|".join(BATCH_CORRECT_VARIANTS)
    log:
        "logs/profiles/batch_correct_{dataset}_{variant}.log"
    resources:
        gpu=1
    shell:
        """
        scripts/gpu_lease.sh pixi run -e rapids python -m jump_production.processing.batch_correct \
            --input {input.parquet} \
            --output {output.parquet} \
            --db {input.db} \
            --config {input.config} \
            2>&1 | tee {log}
        """
