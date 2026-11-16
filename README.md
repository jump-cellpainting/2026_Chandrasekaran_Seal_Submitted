<h1 align="center">JUMP Cell Painting</h1>
<p align="center">Compound profiles, analyses, and reproducible results</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.1101/2023.03.23.534023v4">Read the paper</a> &nbsp; | &nbsp;
  <a href="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/index.html">Explore the results</a> &nbsp; | &nbsp;
  <a href="docs/reproduce.md">Reproduce the analysis</a>
</p>

This repository contains supporting code for the compound analyses of the JUMP Cell Painting dataset, from profile processing and compound annotations to activity, consistency, and downstream analyses.
The dataset is described in:

> **[JUMP Cell Painting chemical dataset: Morphological map of 115,796 compounds in human cells](https://www.biorxiv.org/content/10.1101/2023.03.23.534023v4)** > > Chandrasekaran, Seal et al. | *bioRxiv* (2026) | [doi:10.1101/2023.03.23.534023](https://doi.org/10.1101/2023.03.23.534023)

Please cite this preprint when using the JUMP dataset.
The analyses here focus on the chemical perturbation dataset.

<p align="center">
  <a href="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore/processed/figures/figure_compound_collection/dataset_structure.png">
    <img src="https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore/processed/figures/figure_compound_collection/dataset_structure.png" width="640" alt="JUMP dataset schematic showing compound collections across partner sites, compound exchange between sites, and the related ORF and CRISPR collections.">
  </a>
</p>

*Dataset overview from this repository's publication outputs.
Circle sizes and overlaps are schematic; select the image to view it at full resolution.*

## Explore the results

The [publication data browser](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/index.html) provides figures, analysis summaries, profiles, and result tables hosted in the Cell Painting Gallery.
Preview images, read summaries, and explore CSV and Parquet tables directly in your browser.
No installation or AWS account is needed to browse.

## Reproduce the analysis

Follow the [reproduction guide](docs/reproduce.md) to run the pipeline on AWS and compare its outputs with the pinned public baseline.
It covers prerequisites, instance setup, execution, verification, and cleanup.
Running the full workflow requires your own AWS account and incurs compute and storage costs.

For an existing development environment, see the [setup and pipeline guide](docs/development.md).
The workflow uses Snakemake with pinned environments and public input data.

## Find your way around

| Resource | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | Analyses and figure generation |
| [`src/jump_production/`](src/jump_production/) | Profile processing, metadata, and query utilities |
| [`configs/`](configs/) and [`rules/`](rules/) | Analysis settings and Snakemake workflows |
| [Dataset reference](docs/paper/dataset_reference.md) | How reported numbers and analysis summaries are collected |

For questions or reproducibility problems, [open an issue](https://github.com/jump-cellpainting/2026_Chandrasekaran_Seal_Submitted/issues).
