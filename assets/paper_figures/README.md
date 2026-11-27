# Static panels for assembled paper figures

These files are inputs to `just paper-figures` that no rule in this repository can regenerate.
Layouts in `configs/paper_figures/` mark every panel taken from here as `origin: static`.
See [figure assembly](../../docs/paper/figure_assembly.md).

- `posted/`: regions of existing paper figures that have no pipeline source, so the values they show are not checked against pipeline outputs.
  `example_micrographs.png` is the strip of example micrographs, cropped from the first main figure of the working manuscript.
  `alternative_analytics.png` and `activity_prediction.png` are two supplementary figures, byte-identical to the files in the supplement of preprint version 4; layouts crop the mean-versus-median aggregation tables from the first and the assay-endpoint prediction curves from the second.
- `replicate_number/`: the panels of the replicate-number analysis and its summary table.
  The analysis code is not in this repository.
  The panels were produced from the copairs results database, the well-level profiles and the metadata database of the published datastore.
