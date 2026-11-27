# Development

This page is for working in an existing development environment.
To reproduce the results from nothing, follow [the reproduction guide](reproduce.md) instead.

## Environment

```bash
direnv allow                       # authorize the committed .envrc (`use flake .`), once
direnv exec . pixi install
direnv exec . just get-inputs      # public inputs, no credentials
direnv exec . just verify-inputs
```

`direnv allow` authorizes the environment; it does not activate it for later commands in a non-interactive shell.
Agents, SSH commands and scripts must prefix every project command with `direnv exec .`, including `just`, `pixi`, `uv`, Snakemake, Python and marimo.
Interactive shells with the direnv hook can omit the prefix.
Commands that do not depend on the environment, such as `git`, `rg` and `jq`, never need it.

The flake supplies the pinned `pixi` and adds the NVIDIA driver libraries (`/run/opengl-driver/lib`) to `LD_LIBRARY_PATH`.
Without them CUDA fails in ways that do not name the cause, even when `nvidia-smi` works: `cudaErrorInsufficientDriver`, `driverGetVersion() == 0`, no enumerable GPUs, or a `rapids-singlecell` build that resolves `CMAKE_CUDA_ARCHITECTURES=native` to an architecture too old for `atomicAdd(double*, double)`.
Check activation before changing CUDA packages or blaming the host driver:

```bash
direnv exec . pixi run -e rapids python -c \
  'import cupy as cp; print(cp.cuda.runtime.driverGetVersion(), cp.cuda.runtime.getDeviceCount())'
```

## Inputs and caches

- `just get-inputs` synchronizes `data/external/` and `data/raw/profiles/` with the public bucket over anonymous S3 access, and `just verify-inputs` checks them against `configs/inputs.sha256`.
  It is a sync: local files missing from the remote are deleted, up to `RCLONE_MAX_DELETE`.
  Generated profile families (`*_rsc*`, `*_with_source7*`, `*_active_union`) are excluded so that outputs are not overwritten.
  Do not add `--s3-env-auth` to anonymous downloads: a missing credential can produce an empty listing.
- `just get-null-cache` restores the pinned copairs null distributions into `~/.copairs` and checks every file against the pinned manifest.
  It replaces files of the same name and leaves the others.
  The cache suits only the copairs revision in `pyproject.toml`; see [running without the null cache](reproduce.md#running-without-the-null-cache) for the alternative.
- `just get-from-sources` refreshes inputs from their upstream sources, which can differ from the pinned reproduction inputs.

The published datastore, the null cache and the baseline snapshot are read-only inputs, and this repository has no upload command.
Keep trial renders and logs under `scratch/`, outside `data/`.

## Pipeline targets

`just --list` shows every command.

```bash
just run                     # core pipeline: databases and copairs
just run-all                 # copairs and the analyses in `all`
just explore                 # exploration notebooks, not part of run-all
just srijit                  # the -seal- analyses; run-all includes only those the paper report needs
just catalog                 # the results catalog and everything it lists
just dry                     # preview what would run
just dataset-reference       # build the dataset-reference report (`dry` previews, `view` opens it)
just reproduce               # complete rebuild, figure validation and the baseline check
just unlock                  # release a stale Snakemake lock; confirm no workflow is running first
```

A complete rebuild is the targets `all`, `batch_and_source7`, `explore`, `srijit` and `results_catalog`.
The catalog lists analyses from the other four, so it is a target of its own and not part of `all`.

## Notebooks

Build results with Snakemake before opening their notebooks.
`marimo run` shows a notebook with its cells executed and `marimo edit` modifies it.
An editor can start with stale cells, so run all cells and inspect errors before accepting changes, and do not edit a notebook file externally while a live editor has it open.

```bash
# Pure-Python notebooks with PEP 723 dependencies
direnv exec . uv run --with marimo marimo run --sandbox notebooks/dataset_reference.py --host 127.0.0.1

# GPU notebooks that import CuPy or RAPIDS: the pinned RAPIDS environment, without --sandbox
direnv exec . pixi run -e rapids python -m marimo run notebooks/2.01-ss-umap-visualization.py --host 127.0.0.1

# Scripts that need RDKit
direnv exec . pixi run -e cheminformatics python notebooks/0.16-ss-compound-collection-facts.py
```

`just reproduce` caps BLAS and OpenMP threads to avoid oversubscription during scaffold cross-validation; set `OMP_NUM_THREADS=1` when running those analyses on their own.
If uv reports too many open files, run `ulimit -n 65536` first.

### Positive-control reproducibility heatmap

This figure is opt-in and not part of the default run:

```bash
direnv exec . just build-figure figure_positive_control_reproducibility
direnv exec . just validate-figure figure_positive_control_reproducibility
direnv exec . just catalog
```

It is built from the copairs and augmented metadata databases, and its directory holds the PNG, vector PDF, plotted values, caption and summary.
`just catalog` adds it to `data/processed/catalog.json` as an optional entry.
Its files are not in the baseline roster, so build it in a separate output tree when you compare a reproduction with the baseline.

### Assembled paper figures

`just paper-figures` assembles multi-panel paper figures from pipeline panels into `data/paper_figures/`.
It is opt-in, writes nothing under `data/processed/`, and is described in [figure assembly](paper/figure_assembly.md).

## Code checks

```bash
pixi run ruff check src/jump_production
pixi run ruff format src/jump_production
just test            # pytest lives in the `test` environment; a bare `pixi run pytest` does not find it
just test-browser    # the data browser, in Chrome or Chromium
just check-release   # the reviewed source allowlist
```

## Dependencies

- Add Python packages to `[project] dependencies` in `pyproject.toml`; `[tool.pixi.dependencies]` is for conda packages.
- `copairs-runner` comes from a monorepo subdirectory and must be pinned to a commit hash.
  A branch name makes uv fail to re-solve, because the monorepo root has no `pyproject.toml`.

## GPU rules

Scripts detect the GPU themselves with `from jump_production.gpu import has_gpu`; do not add `use_gpu` parameters or `--no-gpu` flags.
`has_gpu()` checks only CuPy and CUDA, so scripts that need `rapids_singlecell` also gate on that import, as `profiles_to_anndata.py` and `umap.py` do.
`config.py` exports `JUMP_CPUS` from the CPUs available to the process, which respects Slurm and cgroups.

A GPU rule needs two things, and `tests/test_gpu_rules.py` fails if a rule has only one:

- **`resources: gpu=1`** limits how many GPU jobs run at once.
  Snakemake treats an undeclared resource as unlimited, so the limit exists only because the Justfile passes `--resources gpu=N`, with N from `scripts/gpu_lease.sh --count`.
- **`scripts/gpu_lease.sh` as the command prefix** chooses the card.
  Snakemake never assigns a device, so without the lease every job would use device 0.
  The lease locks one file per GPU and exports `CUDA_VISIBLE_DEVICES` for the job, so concurrent jobs land on distinct cards and extra jobs wait.

The count and the cards come from the same script, so the limit cannot exceed the cards available.
Do not count GPUs with `nvidia-smi` in the Justfile: it ignores `CUDA_VISIBLE_DEVICES`.

- **Confine a run** with `GPUS=1 just run-all` or `CUDA_VISIBLE_DEVICES=0,1 just run-all`.
- **Run on CPU** with `CUDA_VISIBLE_DEVICES=""`.
  `GPUS=0` does not do this: Snakemake reads `--resources gpu=0` as never satisfiable, and every GPU rule fails to schedule.
- **The locks are per user.** The lease arbitrates between your own jobs; two people on one machine can still pick the same card, and `JUMP_GPU_LOCKDIR` does not change that.
