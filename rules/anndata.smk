# -*- coding: utf-8 -*-
# Profile conversion to AnnData (raw parquet → interim h5ad)
# Creates both well-level and perturbation-level (aggregated) h5ad files

rule profiles_to_anndata:
    input:
        parquet="data/raw/profiles/{dataset}.parquet",
        script="src/jump_production/processing/profiles_to_anndata.py"
    output:
        well="data/interim/anndata/{dataset}.h5ad",
        perturbation="data/interim/anndata/{dataset}_perturbation.h5ad"
    wildcard_constraints:
        # UMAP outputs share this directory and also end in .h5ad. Without this
        # exclusion, their full stem can be captured as a raw profile dataset.
        dataset=r"(?!.*_(?:cosine|euclidean)_(?:all|active)_umap)[^/]+"
    log:
        "logs/anndata/profiles_to_anndata/{dataset}.log"
    resources:
        gpu=1
    shell:
        "scripts/gpu_lease.sh pixi run -e rapids python -m jump_production.processing.profiles_to_anndata {input.parquet} {output.well} 2>&1 | tee {log}"
