# Analysis data and output contracts

## Data sources

The pipeline integrates data from the following sources:

- Drug Repurposing Hub (Broad Institute)
- ChEMBL target annotations
- Chemical Probes database
- JUMP Cell Painting metadata and profiles from S3
- **Source exclusions**: Source 7 dropped from profiles due to concentration difference.
  Source 15 excluded from circos plots/visualizations - their imaging merged 2 canonical Cell Painting channels.
  Source 4 (Broad, mainly ORFs) and Source 13 (CRISPR) excluded from compound-specific visualizations

## Databases

Two DuckDB databases contain rich queryable data:

1. **`data/interim/jump_metadata_augmented.duckdb`**: Augmented metadata database - Built by `rules/processing.smk` via `scripts/augment_metadata_db.sql` - Contains: compound annotations, target mappings (ChEMBL, Repurposing Hub, Chemical Probes, Motive, ToxCast), compound properties, cell counts, image dimensions - Use this for any metadata lookups or joining annotations to profiles - Tables and columns have descriptive comments - query `SELECT * FROM duckdb_tables()` and `SELECT * FROM duckdb_columns()` to see them

2. **`data/processed/copairs_results.duckdb`**: Aggregated copairs analysis results - Built by `rules/copairs.smk` via `src/jump_production/processing/build_copairs_results_db.py` - Contains tables: `activity_results`, `activity_scores`, `consistency_results`, `consistency_scores` - Each table includes config metadata columns (prefixed with `_`) to identify run parameters - **Query utilities**: Use `jump_production.queries` module (not raw SQL) - it handles the 4-dimensional key (`_dataset`, `_preprocessing`, `_filter`, `_activity_params`) - Tables and columns have descriptive comments - query `SELECT * FROM duckdb_tables()` and `SELECT * FROM duckdb_columns()` to see them (very helpful for understanding the schema) - **Important**: This database contains results from many different run configurations.
   Always filter carefully using the `_*` columns (`_dataset`, `_preprocessing`, `_filter`, `_group_type`, `_distance`, `_activity_params`, etc.) to get the exact subset you need.
   To understand which configurations exist and what they mean, first read `configs/snakemake/copairs_runs.yaml` which defines all the run combinations. - **Consistency sweep vs regular**: Use `_preprocessing` column to distinguish them - regular runs have names like `consistency_no_target2`, sweep runs have names ending in `_sweep` (e.g., `consistency_no_target2_sweep`).
   Filter with: `_preprocessing NOT LIKE '%_sweep'` for regular runs, `_preprocessing LIKE '%_sweep'` for sweep runs

## Copairs configuration

See `docs/copairs/CONFIG.md` for comprehensive documentation on config keys, datasets, DuckDB column mappings, and performance tuning.
The run matrix is defined in `configs/snakemake/copairs_runs.yaml`.

## Annotation sources

- 9 target consistency sources (canonical list in `copairs_runs.yaml`): repurposing, uniprot, chemical_probes, moa, disease_area, motive_biokg, motive_opentargets, motive_primekg, toxcast_assay
- BindingDB and DrugBank are NOT in the current pipeline (removed in earlier refactor)

## Profiles, summaries, and reproducibility

- **summary.json envelope convention**: every analysis notebook emits one summary envelope `{description, numbers, files}` via `from jump_production.summaries import write_summary, index_files`; each Snakemake rule must declare its `summary.json` as an output, and paper-report membership is the `summaries` input of `rule dataset_reference_manifest`.
  The generated manifest is the only discovery path used locally and on S3.
  Cast numpy scalars to int/float in `numbers` (the writer uses `json.dumps(default=str)`, which would stringify them).
  Sandbox-pure notebooks (e.g. 0.12) inline the envelope dict instead of importing `jump_production`.

- Batch correction: `configs/batch_correction.yaml` controls in-repo Harmony params (batch_key, n_clusters, etc.) - requires rapids env: `direnv exec . pixi run -e rapids`

- Two Harmony paths: recipe-Harmony profiles (from CellPainting Gallery, default) use canonical names (e.g., `compound_no_source7.parquet`); in-repo Harmony outputs use `_rsc` suffix (e.g., `compound_no_source7_rsc.parquet`).
  Pre-Harmony inputs use `_pre_harmony` suffix.

- For metadata queries, use `data/interim/jump_metadata_augmented.duckdb`

- For copairs results queries, use `data/processed/copairs_results.duckdb`

- When comparing mAP values, always use `mean_normalized_average_precision` (not raw mAP) and `corrected_p_value` (not raw p-value)

- Load profiles: `jump_production.io.load_profiles(dataset, level="well"|"perturbation")`

- AnnData obs column is `JCP2022` (not `Metadata_JCP2022`) - DuckDB tables use the `Metadata_` prefix, h5ad files do not

- `median_cell_count` is in the `compound_metadata` DuckDB view (100% coverage) but NOT in `adata.obs` - must be fetched via SQL

- Activity lookups for small compound sets: use server-side `Metadata_JCP2022 = ANY(?)` filter in DuckDB rather than `query_activity_results()` which returns all compounds

- **Reproducibility**: Use the pinned cached L40S comparison and tolerances in [the reproduction guide](reproduce.md).
  The release baseline checks exact artifacts and keyed claims; GPU, initialization and null-cache choices matter.

- **DuckDB files are NOT byte-reproducible** across rebuilds even for identical content (verified v1.5.2: string-compression codecs FSST/dictionary emit run-varying bytes; integer-only or `force_compression='uncompressed'` tables ARE byte-stable).
  Our DBs have compressed string columns (IDs, SMILES, names), so file hashes always differ - compare table content via `EXCEPT` + row counts, never sha256.
  Copairs mAP CSVs separately mix deterministic mAP columns with stochastic permutation p-values in one file - compare those by column value, not file hash.

- **Determinism gotcha**: any `STRING_AGG`/`GROUP_CONCAT` and any DataFrame written to CSV needs an explicit `ORDER BY` / `sort_values` tiebreaker, or row/string order varies run-to-run (scan order, or `PYTHONHASHSEED` over dict/set iteration) - identical values, non-reproducible bytes.
  Fixed two such cases 2026-06-19 (a05d7e9 compound_metadata view, c015876 2.43 enrichment).
