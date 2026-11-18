# Agent instructions

This repository reproduces JUMP Cell Painting compound analyses.
Read [CONSTITUTION.md](CONSTITUTION.md) and use the Justfile, workflow rules and tests as the source of truth.
Direct user instructions take precedence.

For independent reproduction, follow [docs/reproduce.md](docs/reproduce.md).
The AWS launcher runs directly in the local shell; it installs the pinned environment on the instance.
For development, run project commands through `direnv exec .`, including Python, pixi, just and Snakemake.
Use `direnv exec . pixi run -e test pytest` for tests.

Preserve scientific calculations, canonical inputs, eligibility rules and baseline tolerances.
Use `mean_normalized_average_precision` and `corrected_p_value` with the exact copairs configuration.
GPU rules require both `resources: gpu=1` and `scripts/gpu_lease.sh`.
Read [development](docs/development.md), [analysis contracts](docs/analysis.md) and [copairs configuration](docs/copairs/CONFIG.md) when changing these parts.

Put analysis Python in numbered `notebooks/` files and reusable processing in `src/jump_production/`.
Use Typer for command arguments and Loguru for processor logging.
Snakemake rules declare actual output files, including analysis summary envelopes.
Report membership is defined by `rule dataset_reference_manifest`; see [dataset reference](docs/paper/dataset_reference.md).
Register every summary in `configs/results_catalog.yaml`; changes to the registry, the catalog or `index.html` follow the [data browser guide](docs/data-browser.md) and `just test-browser`.
Output paths under `data/processed`, summary file names and the published locations are a fixed [layout contract](docs/data-browser.md#layout-contract) with the data browser and with shared links; read it before renaming, moving or removing an output.
Figure changes follow the [figure checklist](docs/paper/figure_review_checklist.md).

Run relevant checks and repair affected callers before declaring a change complete.
Source releases must pass `just check-release`; see [release contract](docs/release.md).
Keep generated data, local notes and intermediate work under ignored paths.
Changes to the release allowlist require content review, including comments, docstrings, embedded links and generated prose.
Do not publish data or merge changes without explicit authorization.
