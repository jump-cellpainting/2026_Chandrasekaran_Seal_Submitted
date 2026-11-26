# Static panels for assembled paper figures

These files are inputs to `just paper-figures` that no rule in this repository can regenerate.
Layouts in `configs/paper_figures/` mark every panel taken from here as `origin: static`.
See [figure assembly](../../docs/paper/figure_assembly.md).

- `posted/`: three figures as posted with the preprint, from each of which one region is cropped.
  `compound_collection.png` supplies the example micrographs, `alternative_analytics.png` the mean-versus-median aggregation tables, and `activity_prediction.png` the assay-endpoint prediction curves.
  None of these has a pipeline source, so the values they show are not checked against pipeline outputs.
- `replicate_number/`: the panels of the replicate-number analysis and its summary table.
  The analysis code is not in this repository.
  The panels were produced from the copairs results database, the well-level profiles and the metadata database of the published datastore.
