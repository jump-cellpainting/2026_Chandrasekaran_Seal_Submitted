# Development and GPU execution

Use the sections relevant to the requested setup, pipeline, or GPU change.
Commands below run inside `direnv exec .`; include that prefix in non-interactive shells.

## Environment setup

```bash
# 1. Authorize the committed environment (one time)
direnv allow

# 2. Run project commands inside the flake environment
direnv exec . pixi install

# 3. Download input data from team S3 (works without AWS credentials)
direnv exec . just get-inputs

# Optional upstream refresh (not the pinned reproduction inputs)
direnv exec . just get-from-sources
```

**`direnv allow` authorizes `.envrc`; it does not activate the environment for later commands in a non-interactive shell.** Interactive users with the direnv shell hook can authorize the committed `.envrc` (just `use flake .`) once and then use the commands below normally.
Agents, SSH commands, escalated commands, and other non-interactive processes must run project commands through:

```bash
direnv exec . <command>
```

This applies to `just`, `pixi`, `uv`, Snakemake, Python, marimo, CUDA, and GPU commands.
It is not needed for environment-independent inspection commands such as `git`, `rg`, `jq`, and `ls`.

The flake devShell supplies the pinned `pixi` and sets `LD_LIBRARY_PATH` to include `/run/opengl-driver/lib`.
Without those driver libraries, CUDA can fail during installation or at runtime.
Misleading symptoms include `cudaErrorInsufficientDriver`, `driverGetVersion() == 0`, an inability to enumerate GPUs, or an incorrect `CMAKE_CUDA_ARCHITECTURES=native` result even when `nvidia-smi` works.
Verify flake activation before changing CUDA packages or diagnosing the host driver:

```bash
direnv exec . pixi run -e rapids python -c \
  'import cupy as cp; print(cp.cuda.runtime.driverGetVersion(), cp.cuda.runtime.getDeviceCount())'
```

Without the driver libraries, `rapids-singlecell` (always built from sdist - `pixi.lock` pins the tarball) can resolve `CMAKE_CUDA_ARCHITECTURES=native` to an architecture too old for `atomicAdd(double*, double)`.
The error names CUDA rather than direnv, so check activation first.

## Dependencies

- Add Python packages to `[project] dependencies` in `pyproject.toml`, not `[tool.pixi.dependencies]` (which is for conda packages)
- **`copairs-runner` git dep**: Uses monorepo subdirectory - must pin to a commit hash, not a branch name.
  Branch refs cause uv re-solve failures (no `pyproject.toml` at monorepo root).
  Update hash manually when upgrading.

## Pipeline targets

See `just --list` for all commands.
Key workflows:

```bash
just run          # Core pipeline (db + copairs)
just run-all      # Full pipeline (copairs + all analysis)
just explore      # Exploration notebooks (phase 0 - not included in run-all)
just srijit       # Full -seal- analysis aggregate; run-all includes selected paper-report dependencies only
just dry          # Preview what will run
just dataset-reference       # Build the paper's dataset-reference report
just dataset-reference dry   # Preview the report's Snakemake dependencies
just dataset-reference view  # View the report as a marimo notebook
```

## Code checks

```bash
# Run linter (ruff)
pixi run ruff check src/jump_production

# Run formatter
pixi run ruff format src/jump_production

# Run tests (pytest lives in the `test` pixi env, so a bare `pixi run pytest` won't find it)
just test
```

## GPU scheduling

- CPU count: `config.py` exports `JUMP_CPUS` via `os.sched_getaffinity(0)` (respects Slurm/cgroups)
- GPU detection: `from jump_production.gpu import has_gpu; HAS_GPU = has_gpu()` - probes CUDA driver via cupy, cached after first call
- Pattern for new scripts: `from jump_production.gpu import has_gpu; HAS_GPU = has_gpu()` - no `use_gpu` parameters or `--no-gpu` flags
- **Important**: `has_gpu()` only checks cupy/CUDA.
  Scripts needing `rapids_singlecell` must also gate on that import (see `profiles_to_anndata.py`, `umap.py`)

**How GPU jobs are scheduled - two halves, both required:**
- **How many:** a GPU rule declares `resources: gpu=1`.
  Snakemake treats an *undeclared* resource as **unlimited**, so the cap only exists if you pass `--resources gpu=N`.
  The Justfile does this for you, taking N from `scripts/gpu_lease.sh --count`.
- **Which card:** Snakemake never assigns a device id, so every job would default to device 0 and could exhaust its memory while the rest idle.
  Every GPU rule therefore wraps its command in `scripts/gpu_lease.sh`, which `flock`s one lock file per leasable GPU and exports `CUDA_VISIBLE_DEVICES` for that job.
  Concurrent jobs land on distinct cards, and a card is never double-booked even under a bare `snakemake` with no `--resources` cap (excess jobs just wait).
- The count and the cards come from the *same* script, so the cap can never exceed the cards actually available.
  (Don't reintroduce an `nvidia-smi` count in the Justfile - `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so the cap would silently disagree with it.)
- **Confining a run:** `GPUS=1 just run-all`, or `CUDA_VISIBLE_DEVICES=0,1 just run-all` (both the cap and the lease honour it).
  `CUDA_VISIBLE_DEVICES=""` means CPU-only, per the CUDA convention: the lease runs the job unpinned and `has_gpu()` sends it down the CPU path.
  **`GPUS=0` is not the CPU-only knob** - it sets `--resources gpu=0`, which Snakemake reads as "never satisfiable", so every `gpu=1` rule fails to schedule.
  Use `CUDA_VISIBLE_DEVICES=""`.
- **Scope:** the lease's locks are per-user, so it arbitrates between *your own* jobs - which is all it is meant to do.
  Two people running the pipeline on one box can still pick the same card and OOM each other.
  (`JUMP_GPU_LOCKDIR` does not buy you cross-user coordination: the lock files are opened for writing, so one user cannot lock another's file.
  Supporting that needs a code change, not an env var.)
- **Adding a GPU rule:** you must do *both* - declare `resources: gpu=1` **and** prefix the shell command with `scripts/gpu_lease.sh`.
  A rule that merely uses `pixi run -e rapids`/`-e deepchem` without declaring the resource escapes the cap entirely (this was the `profiles_to_anndata` bug).
  `tests/test_gpu_rules.py` enforces the pairing - it fails if a GPU rule has only one half.
  It keys off the **lease**, not the env name, so a GPU rule running in some other env (e.g.
  `srijit_2_36` under cheminformatics) is still covered.
