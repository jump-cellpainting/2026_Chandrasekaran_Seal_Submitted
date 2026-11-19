# copairs configuration

Activity and consistency are computed with [copairs](https://github.com/cytomining/copairs), following the mAP framework of [Kalinin et al.
(2025)](https://doi.org/10.1038/s41467-025-60306-2), and run through `copairs-runner` with Hydra configurations.
Both packages are pinned to exact revisions in `pyproject.toml` and `pixi.lock`; read their own documentation at those revisions for the API.
The configuration files are the source of truth, so inspect them instead of relying on the examples here.

- **Activity** asks whether replicates of a perturbation retrieve each other against negative controls.
- **Consistency** asks whether active perturbations that share an annotation, such as a target or mechanism, retrieve each other.

Compare runs with `mean_normalized_average_precision` and decide significance with `corrected_p_value`.

## Layout

```text
configs/copairs/metrics/
├── activity.yaml              # main configs: defaults and output path
├── consistency.yaml
├── dataset/                   # one profile set each (name, path)
├── columns/                   # feature subset
├── preprocessing_01_metadata/ # metadata merging, runs first
├── preprocessing_02_core/     # filtering and aggregation steps, runs second
├── filter/                    # source filter (name, query)
├── activity_params/           # replicate pairing and mAP parameters
└── target_column/             # consistency annotation (name, column)
configs/snakemake/copairs_runs.yaml   # the run matrix
```

A dataset is one embedding of the same compounds:

| Dataset | Profiles |
| --- | --- |
| `compound_no_source7` | CellProfiler features, source 7 excluded |
| `compound_DL_CPCNN_no_source7` | Deep-learning CPCNN embeddings, source 7 excluded |
| `compound_no_source7_rsc` | CellProfiler features with in-repository RAPIDS Harmony |
| `*_active_union` | The same, pre-filtered to union-active compounds |

`rules/copairs.smk` reads the run matrix, runs every activity and consistency combination and collects the results in `data/processed/copairs_results.duckdb`.
`just run` builds all of it and `just dry` previews it.

## Config keys and database columns

Each config key becomes a `_*` column in `copairs_results.duckdb`.
Filter on these columns to select exactly one configuration.

| Config key | DuckDB column | Meaning |
| --- | --- | --- |
| `dataset` | `_dataset` | Profile set |
| `columns` | `_columns` | Feature subset, such as `feat_all` |
| `preprocessing_02_core` | `_preprocessing` | Activity or consistency preprocessing |
| `filter` | `_filter` | Source filter: `all_sources`, `source_2`, `no_source9` |
| `activity_params` | `_activity_params` | Replicate pairing: `default`, `withinsource`, `crosssource` |
| `target_column` | `_group_type` | Consistency annotation: `repurposing`, `uniprot`, `moa` and others |
| `distance` | `_distance` | Similarity: `cosine`, `abs_cosine` |

## One run

```bash
direnv exec . pixi run copairs-runner \
  --config-dir configs/copairs/metrics --config-name activity \
  dataset=compound_no_source7 \
  columns=feat_all \
  preprocessing_02_core=activity_no_target2 \
  filter=all_sources \
  activity_params=default
```

Hydra loads `activity.yaml`, merges the chosen option of each group and resolves interpolations such as `${dataset.path}`.
`activity_params` files use `# @package _global_` to set `average_precision` and `mean_average_precision` at the root.
The output directory is named from the options:

```text
data/processed/copairs/runs/activity/{dataset}__{columns}__{preprocessing}__{filter}__{activity_params}/results/
```

It holds `activity_ap_scores.csv`, `activity_map_results.csv`, a plot and the resolved configuration in `.hydra/config.yaml`.
Consistency runs follow the same pattern under `runs/consistency/`, with `{target_column}__{distance}` in place of the activity parameters.

## Adding configurations

After adding any option, add the runs that use it to `configs/snakemake/copairs_runs.yaml`.

- **Dataset**: `dataset/<name>.yaml` with `name`, `path` and `description`.
- **Filter**: `filter/<name>.yaml` with `name`, a `query` such as `"Metadata_Source != 'source_9'"` and `description`.
- **Preprocessing**: `preprocessing_02_core/<name>.yaml` with `name` and an ordered list of `steps`.
  This lists the step types that the pinned `copairs-runner` supports:

  ```bash
  direnv exec . pixi run python -c "
  from copairs_runner import CopairsRunner
  print('\n'.join(sorted(m.removeprefix('_preprocess_') for m in dir(CopairsRunner) if m.startswith('_preprocess_'))))"
  ```

- **Activity parameters**: `activity_params/<name>.yaml`, modeled on `crosssource.yaml`.
- **Target column**: make sure the column exists in the augmented metadata database, add its merge step to `preprocessing_01_metadata/jump_with_annotations.yaml`, and create `target_column/<name>.yaml` with `name`, `column` and `description`.

## Slow runs

A run that hangs usually has too many positive pairs, which grow quadratically with replicates per group.
As a rule of thumb, under 1 million pairs take seconds, 10 million take minutes, 100 million take tens of minutes, and more calls for aggregation.

Positive controls are the dense case: about 55,000 profiles gave 333 million cross-source pairs and a run of about 100 minutes.
`activity_poscon_only` first takes the median per compound, source and plate, leaving about 11,000 profiles, 14 million pairs and a 20-second run:

```yaml
- type: aggregate_replicates
  params:
    groupby:
      - Metadata_JCP2022
      - Metadata_Source
      - Metadata_Plate
      - Metadata_pert_type
```

Aggregate for positive controls and other densely replicated groups.
Do not aggregate ordinary compound activity, where most compounds have about one replicate per plate, or analyses that need well-level resolution.

To find which source drives a cross-source difference, group mAP by compound and source (`sameby: ["Metadata_JCP2022", "Metadata_Source"]`, as in `crosssource.yaml`) and compare within-source with cross-source results per source.
