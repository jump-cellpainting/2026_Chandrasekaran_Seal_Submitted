# Publication data browser

`index.html` is the browser behind "Explore the results".
It is one static file with no build step and no server of its own: it reads public objects from the Cell Painting Gallery bucket directly.
This page describes what it reads and how the workflow produces it.

## What a reader gets

Each analysis writes a `summary.json` envelope with a description, its reported numbers and the files it produced.
The envelope describes one analysis in place.
It does not say which analyses exist, what each is called or how they relate, and folder names such as `exploration/0.05` do not either.

The results catalog supplies that.
`data/processed/catalog.json` lists every analysis with a title and a section, the envelope's own description, the rule and script that produced it, and its declared files.
The browser reads this one file to show an overview, to search across analyses and to open the right files.

## How the catalog is built

`rule results_catalog` in `rules/catalog.smk` writes the catalog from two inputs.

`configs/results_catalog.yaml` is the authored registry and the membership definition.
It defines the sections and gives each analysis a title, a section and, optionally, headline numbers.
A headline is a key into that envelope's `numbers`; the value is read at build time and the build fails if the key is missing.
Results that have no envelope, such as the copairs results database, are described in the registry directly.

The summary envelopes supply everything computed.
The catalog restates no scientific value and no description of its own for an analysis that has an envelope.

The build resolves every declared file to a path relative to the catalog and fails if a file was not written.
All catalog paths are relative to the catalog's directory, so the output tree can be mirrored anywhere unchanged.
The catalog also records the source repository and the commit that produced the results, or no commit when tracked files had uncommitted changes.

The catalog lists analyses from the `all`, `batch_and_source7`, `explore` and `srijit` targets, so it is a target of its own.
`just reproduce` builds it last; `just catalog` builds it and everything it lists.

## Adding an analysis

Register the new summary in `configs/results_catalog.yaml` with a title and a section.
Three checks keep the registry and the workflow together:

- `tests/test_catalog.py` requires every registered file to have exactly one producing rule and every declared summary output to be registered.
- The catalog build fails on a missing summary, a missing declared file or an unknown headline key.
- `just verify-baseline` requires the catalog, and fails if any envelope in the output tree is not cataloged.

Titles describe what a result is.
They do not cite manuscript figure numbers or issue numbers, which change.

## Envelope paths

A path in an envelope's `files` is relative to the checkout (`data/processed/...`) or, in a publication package, to the envelope's own directory.
`write_summary` rewrites absolute paths that lie inside the checkout, so an envelope never names the machine that wrote it.
Readers of envelopes should use the catalog's resolved paths instead of interpreting these two forms themselves.

## Publication

The source release has no upload command, and this repository does not write to the bucket.
Publishing `index.html` or an output tree is a separate, authorized step; see [the release contract](release.md).
