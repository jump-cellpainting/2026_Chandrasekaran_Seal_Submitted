# Reproduction data

Run commands through `direnv exec .` in non-interactive development shells.
For the standalone AWS workflow, use [the reproduction guide](reproduce.md).

`just get-inputs` downloads the public inputs and `just verify-inputs` checks their pinned SHA256 values.
The download uses anonymous S3 access and requires no maintainer credentials.
It synchronizes `data/external/` and `data/raw/profiles/`, so local files missing from the remote source can be deleted, subject to `RCLONE_MAX_DELETE`.
Generated profile families (`*_rsc*`, `*_with_source7*`, `*_active_union`) are excluded from this input synchronization so outputs are not overwritten.
Do not add `--s3-env-auth` to anonymous downloads: a missing credential can produce an empty listing.

`just get-null-cache` restores the pinned copairs null distributions into `~/.copairs` with `rclone copy`.
This does not delete unrelated cache files, but replaces matching names.
Every downloaded file is checked against the pinned manifest; the cache is specific to the copairs revision in `pyproject.toml`.
See the reproduction guide for the uncached diagnostic option and its different verification contract.

A complete rebuild runs `all`, `batch_and_source7`, `explore`, `srijit` and `results_catalog`.
The catalog indexes analyses from all four groups, so it is its own target and is not part of `all`.
`just reproduce` orchestrates these targets, validates figures and verifies the scientific output contract.
`just unlock` releases a stale Snakemake lock after a terminated run; confirm no workflow is running first.
Keep trial renders and logs under `scratch/`, outside `data/`.

The existing publication datastore, null cache and baseline snapshot are read-only inputs to this workflow.
This source release has no upload or synchronization-to-S3 command.
