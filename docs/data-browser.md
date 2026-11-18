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
Some of those folders hold tens of thousands of objects, so a folder is listed 5,000 entries at a time and its rows are drawn 200 at a time.

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

## Layout contract

The page does not work out where results are.
It reads fixed locations in the bucket and relies on a few naming rules inside a results tree.
That layout is deliberate and stays as it is: the links readers share carry full bucket paths, and nothing redirects an old path to a new one.
Read this section before renaming, moving or removing an output, and before changing a location in `index.html`.

The page reads a results tree in one of two ways.
The published page currently reads the datastore, which has no catalog, from its summary files, so the naming rules below are what it works from.
A tree published with a `catalog.json` is read from that catalog, and the page does not check the catalog against the bucket: nothing inside such a tree changes without rebuilding and republishing its catalog.

| Fixed | Relied on by | Check that fails when it changes | Runs in |
| --- | --- | --- | --- |
| The bucket, the publication prefix, and under it `jump_production_datastore/processed/` for results and `jump_production_datastore/raw/profiles/` for profiles | The location constants that open the script in `index.html`, `S3_PROJECT_PATH` in the Justfile and the links in `README.md` | The browser tests, whose trees are laid out at these locations | `just test-browser` |
| A results tree is the checkout's `data/processed/`, published in a folder named `processed/` | Envelope paths written as `data/processed/...` | The browser tests | `just test-browser` |
| Every output path under `data/processed/` | Shared links, the registry and the dataset-reference report | The registry test, for a registered path whose rule no longer writes it; the baseline verifier, for the complete roster of output paths | `just test`; `just verify-baseline` |
| One analysis per folder, described by one envelope whose file name ends in `summary.json` | Finding analyses without a catalog; the test that every summary is registered | The registry test | `just test` |
| The envelope shape `{description, numbers, files}`, each file named as `data/processed/...` or relative to the envelope | Every view of an analysis | The catalog build and catalog validation | The catalog build; `just verify-baseline` |
| A figure package is `figures/figure_<name>/` with a `legend.md`, and registry section `figures` holds exactly the entries under `figures/` | The figure section and label, the legend on a figure page and `just validate-figures` | The registry test; figure validation | `just test`; `just reproduce` |
| The reported numbers are `dataset_reference/dataset_reference.json` at the root of the tree, each row with `claim`, `value` and `source` | "Reported numbers" and searching by value | The registry test, which names the rule that writes this file; the baseline verifier | `just test`; `just verify-baseline` |
| The catalog is `catalog.json` at the root of the tree, with paths relative to itself | Reading a tree from its catalog | The registry test; catalog validation | `just test`; `just verify-baseline` |
| The addresses `?prefix=`, `?file=`, `?q=`, `?view=numbers` and `&filter=` | Shared links and bookmarks | The browser tests | `just test-browser` |

Without a catalog, the first folder under `processed/` gives an analysis its title, and deeper folders such as the profile set qualify it.
The folder notes, profile-set descriptions and word casing written into the page are cosmetic: they can lag behind a new folder without breaking anything.

### When something has to change

Adding an analysis changes nothing above.
Give it its own folder and an envelope named `summary.json`, and register it as described under [Adding an analysis](#adding-an-analysis).

Renaming, moving or removing an output changes the contract.
The baseline verifier pins the complete roster of output paths and keys every reported number by the path of the summary that reports it, so such a change fails `just verify-baseline`.
A retired or an added file can be named in `configs/scientific_baseline.json`, where the reason is reviewed.
A renamed analysis that reports numbers cannot, because that policy has no entry for a changed summary path: it needs a new baseline snapshot.
The rule and the registry change together, and the workflow then rebuilds the dataset-reference report and the catalog.
Published objects do not move when the source does: the published tree keeps its old layout, and a link to a renamed path stops working once a tree with the new layout replaces it.

Moving the results or the page means editing the location constants in `index.html` and in `tests/test_data_browser.py`, running `just test-browser` and publishing the page again.

### What the checks do not cover

These checks compare what the source writes with what the page expects.
They do not watch the bucket.
This repository has no upload command, so an object that is renamed, overwritten or deleted in the bucket passes every check above.

`JUMP_BROWSER_LIVE=1 just test-browser -k published_bucket` is the check for the bucket.
It drives the page in this checkout over the published bucket: the overview and its count, whether every result set it lists is in the bucket, search, an analysis page, the profiles, every figure package with its legend, and whether every reported number names an analysis the overview lists.
Run it after any change to the bucket.
It finds a broken layout after the fact and cannot prevent one.
Recovering an overwritten or deleted object depends on the bucket keeping earlier versions, which it did when the page was last published and which this repository does not control.

The other limits are these:

- Where the page reads a tree from its summary files, as it does today, an analysis that reports no numbers and is renamed in the bucket is listed under its new name and passes that check, and only links to its old name break.
- Under a catalog that check compares the listed folders with the bucket, and `just verify-baseline` is what compares every cataloged file, on the local tree before it is published.
- The browser tests need Chrome or Chromium and are not part of `just test`, and no hosted check runs on a pull request, so they protect a change only when its author runs them.
- The baseline verifier runs after a full reproduction, so an output renamed consistently in its rule and the registry passes `just test` and is caught only there.
- The published page should be the `index.html` on `main`, byte for byte: its SHA256 is the `index.html` entry in `release-files.json`, and nothing compares the two automatically.
- The rows of `dataset_reference.json` are written from the same list as the CSV whose columns the verifier checks, and the JSON keys have no check of their own.

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

An entry marked `optional: true` belongs to an opt-in target that the default run does not build.
It is cataloged when its summary exists and the catalog never causes it to be built, so the pinned output roster is unaffected.
The [positive-control heatmap](notebooks.md#positive-control-reproducibility-heatmap) is registered this way: build it, run `just catalog`, and it appears under the paper figure assets.
The page also shows a folder whose summary the catalog does not list as an analysis, described by that summary, so an opt-in figure published beside an older catalog still reads properly.

## Adding an analysis

Give the analysis its own folder under `data/processed/` and name its envelope `summary.json`, as the [layout contract](#layout-contract) requires.
Register the new summary in `configs/results_catalog.yaml` with a title and a section.
Three checks keep the registry and the workflow together:

- `tests/test_catalog.py` requires every registered file to have exactly one producing rule and every declared summary output to be registered, as optional when no default target builds it.
  It also requires a summary's file name to end in `summary.json`, and section `figures` to hold exactly the entries under `figures/`.
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
It covers the cataloged and the uncataloged case, both path conventions, missing and foreign paths, the size limit, very large folders, text rendering and address handling.
`JUMP_BROWSER_LIVE=1 just test-browser` also reads the public bucket.
Playwright is fetched by `uv` for that command and is not part of the pinned environments, so plain `just test` skips these tests.

## Publication

The source release has no upload command, and this repository does not write to the bucket.
Publishing `index.html` or an output tree is a separate, authorized step; see [the release contract](release.md).
After either, run the published-bucket check and compare the published page with `index.html`, as described under [What the checks do not cover](#what-the-checks-do-not-cover).
