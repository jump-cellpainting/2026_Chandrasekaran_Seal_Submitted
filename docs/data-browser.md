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

## What the page does

The overview lists every result set under its section, and one search box covers analyses, their declared files and the numbers they report.
An analysis page shows its title and description, headline numbers, the producing rule, a link to the analysis source at the revision that produced the results, figure previews with captions, the figure legend, and every file with the actions that suit it.
"Reported numbers" lists every value in the dataset-reference report and links each to the summary that reports it.
"All files" is the plain folder view of the publication prefix, including the profiles and inputs.

Tables open in Datasette Lite, which loads a whole table into browser memory.
Tables up to 170 MB opened in testing and a 770 MB table failed, so above 200 MB the page offers a download and the S3 path instead.
JSON, Markdown, YAML, text and log files open in the page; everything is rendered as text, never as markup.

The page shows what is published and adds nothing to it.
Titles, sections and headline choices come from the catalog; descriptions and values come from the envelopes; file sizes and dates come from the bucket listing.

### Results without a catalog

A results tree published before the catalog existed has none.
The page then lists the tree, reads every `*summary.json` it finds and builds the overview from those.
Titles come from the analysis folder or the envelope's own `task`, headline numbers are the top-level results an analysis reports in its own order, and a declared file that is not in the bucket is labeled as missing instead of linked.
This is how the page reads the datastore that predates this catalog.

`RESULTS_PREFIXES` in `index.html` lists the results trees to try, preferred first.
The first tree that publishes a `catalog.json` is shown with its catalog; if none does, the last tree is read from its summary files.
Publishing a new output tree beside an old one therefore needs one added line there, and the old tree stays browsable as files.

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

## Testing the page

`just test-browser` drives `index.html` in Chrome or Chromium over small output trees written by the real envelope writer and catalog builder, answering the page's bucket requests locally.
It covers the cataloged and the uncataloged case, both path conventions, missing and foreign paths, the size limit, text rendering and address handling.
`JUMP_BROWSER_LIVE=1 just test-browser` also reads the public bucket.
Playwright is fetched by `uv` for that command and is not part of the pinned environments, so plain `just test` skips these tests.

## Publication

The source release has no upload command, and this repository does not write to the bucket.
Publishing `index.html` or an output tree is a separate, authorized step; see [the release contract](release.md).
