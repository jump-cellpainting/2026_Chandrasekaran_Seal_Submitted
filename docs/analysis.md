# Analysis data and output contracts

## Data sources

- Drug Repurposing Hub, ChEMBL target annotations, the Chemical Probes database, and JUMP Cell Painting metadata and profiles from public S3.
- Source 7 is dropped from profiles because of a concentration difference.
- Source 15 is excluded from circos plots and visualizations: its imaging merged two canonical Cell Painting channels.
- Source 4 (mainly ORFs) and source 13 (CRISPR) are excluded from compound-specific visualizations.
- Target consistency uses nine annotation sources, listed canonically in `configs/snakemake/copairs_runs.yaml`: repurposing, uniprot, chemical_probes, moa, disease_area, motive_biokg, motive_opentargets, motive_primekg and toxcast_assay.
  BindingDB and DrugBank are not used.

## Databases

Both DuckDB databases carry table and column comments; read them with `SELECT * FROM duckdb_tables()` and `SELECT * FROM duckdb_columns()`.

- **`data/interim/jump_metadata_augmented.duckdb`** holds compound annotations, target mappings (ChEMBL, Repurposing Hub, Chemical Probes, Motive, ToxCast), compound properties, cell counts and image dimensions.
  `rules/processing.smk` builds it with `scripts/augment_metadata_db.sql`.
  Use it for metadata lookups and for joining annotations to profiles.
- **`data/processed/copairs_results.duckdb`** holds the tables `activity_results`, `activity_scores`, `consistency_results` and `consistency_scores` for every copairs run configuration.
  `rules/copairs.smk` builds it with `src/jump_production/processing/build_copairs_results_db.py`.
  Query it through `jump_production.queries`, which handles the four-part key (`_dataset`, `_preprocessing`, `_filter`, `_activity_params`).
  In raw SQL, filter on every `_*` column you need; [the copairs configuration guide](copairs/CONFIG.md) maps them to config keys, and `configs/snakemake/copairs_runs.yaml` lists the configurations that exist.
  Consistency sweep runs have `_preprocessing` names ending in `_sweep`; select or exclude them with `LIKE '%_sweep'`.

## Conventions

- Compare mAP with `mean_normalized_average_precision`, not raw mAP, and decide significance with `corrected_p_value`, not the raw p-value.
- Load profiles with `jump_production.io.load_profiles(dataset, level="well"|"perturbation")`.
- The AnnData `obs` column is `JCP2022`; DuckDB tables use the `Metadata_` prefix and h5ad files do not.
- `median_cell_count` is in the `compound_metadata` DuckDB view for every compound but not in `adata.obs`; fetch it with SQL.
- For the activity of a small compound set, filter in DuckDB with `Metadata_JCP2022 = ANY(?)`; `query_activity_results()` returns all compounds.
- Harmony profiles from the Cell Painting Gallery recipe are the default and use canonical names such as `compound_no_source7.parquet`.
  In-repository Harmony outputs carry the `_rsc` suffix and their pre-Harmony inputs the `_pre_harmony` suffix.
  `configs/batch_correction.yaml` sets the in-repository Harmony parameters, which run in the `rapids` environment.

## Summary envelopes

Every analysis writes one summary envelope, a JSON object `{description, numbers, files}`, and its Snakemake rule declares that file as an output.

- Write it with `write_summary` and `index_files` from `jump_production.summaries`.
  Notebooks that must run in a sandbox, such as 0.12, write the same dictionary inline.
- A path in `files` is relative to the checkout (`data/processed/...`) or, in a publication package, to the envelope's own directory.
  `write_summary` rewrites absolute paths inside the checkout, so an envelope never names the machine that wrote it.
  Readers should use the catalog's resolved paths instead of interpreting the two forms.
- Cast numpy scalars to `int` or `float` in `numbers`; the writer would otherwise store them as strings.
- State reported populations explicitly, for example the full released collection versus the analyzed subset.
  The canonical analyzed key is in `configs/snakemake/analysis.yaml`.

Two indexes are built from the envelopes.
The [dataset-reference report](paper/dataset_reference.md) collates the numbers of the summaries named in `rule dataset_reference_manifest`.
The [results catalog](data-browser.md#adding-an-analysis) lists every analysis for the data browser, and every summary must be registered there.

## Determinism

The reproduction is checked against a pinned baseline with [exact hashes and stated tolerances](reproduce.md#what-is-pinned-and-checked), so outputs must not vary between runs.

- Give every `STRING_AGG` or `GROUP_CONCAT`, and every DataFrame written to CSV, an explicit `ORDER BY` or `sort_values` tiebreaker.
  Without one, scan order or `PYTHONHASHSEED` changes the row or string order: identical values, different bytes.
- DuckDB files are not byte-reproducible across rebuilds, because compressed string columns are serialized differently each time.
  Compare table content with `EXCEPT` and row counts, never file hashes.
- copairs mAP CSVs hold deterministic mAP columns beside permutation p-values that depend on the null distributions.
  Compare runs that did not share a null cache by column, not by file hash.
- GPU model, UMAP initialization and the null cache all change results; see [outcomes on other GPU hardware](reproduce.md#outcomes-on-other-gpu-hardware).
