# Publication data browser

`index.html` is the page behind "Explore the results".
It is one static file with no build step and no server: it reads public objects from the Cell Painting Gallery bucket directly.

## How the page reads results

Each analysis writes a [summary envelope](analysis.md#summary-envelopes).
The results catalog, `data/processed/catalog.json`, lists every analysis with a title, a section, its producing rule and script, and its files.

- A results tree with a `catalog.json` is read from that one file.
  The page does not compare the catalog with the bucket, so a change inside such a tree needs a rebuilt and republished catalog.
- A tree without one is read from its summary files: every `*summary.json` becomes an analysis titled from its folder, and a declared file that is not in the bucket is labeled as missing.
  The published datastore predates the catalog and is read this way today.

`RESULTS_PREFIXES` in `index.html` lists the trees to try, preferred first: the first with a catalog is shown from it, and if none has one the last is read from its summaries.
Publishing a new tree beside an old one needs one added line there.

The page adds nothing to what is published: titles, sections and headline choices come from the catalog, descriptions and values from the envelopes, and sizes and dates from the bucket listing.
Tables open in Datasette Lite, which loads a whole table into browser memory, so above 200 MB the page offers a download instead.

## Layout contract

The page reads fixed locations in the bucket and relies on a few naming rules inside a results tree.
Shared links carry full bucket paths and nothing redirects an old path to a new one, so this layout stays as it is.
Read this section before renaming, moving or removing an output, and before changing a location in `index.html`.

| Fixed | Relied on by | Checked by |
| --- | --- | --- |
| The bucket, the publication prefix, and under it `jump_production_datastore/processed/` for results and `jump_production_datastore/raw/profiles/` for profiles | The location constants that open the script in `index.html`, `S3_PROJECT_PATH` in the Justfile and the links in `README.md` | `just test-browser` |
| A results tree is the checkout's `data/processed/`, published in a folder named `processed/` | Envelope paths written as `data/processed/...` | `just test-browser` |
| Every output path under `data/processed/` | Shared links, the registry and the dataset-reference report | `just test`, for a registered path that no rule writes; `just verify-baseline`, for the complete roster |
| One analysis per folder, described by one envelope whose file name ends in `summary.json` | Finding analyses without a catalog | `just test` |
| The envelope shape `{description, numbers, files}`, each file named as `data/processed/...` or relative to the envelope | Every view of an analysis | The catalog build; `just verify-baseline` |
| A figure package is `figures/figure_<name>/` with a `legend.md`, and registry section `figures` holds exactly the entries under `figures/` | The figure section and label, the legend on a figure page and `just validate-figures` | `just test`; figure validation in `just reproduce` |
| The reported numbers are `dataset_reference/dataset_reference.json` at the root of the tree, each row with `claim`, `value` and `source` | "Reported numbers" and searching by value | `just test`; `just verify-baseline` |
| The catalog is `catalog.json` at the root of the tree, with paths relative to itself | Reading a tree from its catalog | `just test`; `just verify-baseline` |
| The addresses `?prefix=`, `?file=`, `?q=`, `?view=numbers` and `&filter=` | Shared links and bookmarks | `just test-browser` |

The folder notes, profile-set descriptions and word casing written into the page are cosmetic and can lag behind a new folder without breaking anything.

### Changing the layout

- **Adding an analysis** changes nothing above; see [Adding an analysis](#adding-an-analysis).
- **Renaming, moving or removing an output** breaks shared links and fails `just verify-baseline`, which pins the complete roster of output paths and keys every reported number by the path of its summary.
  A retired or added file can be named in `configs/scientific_baseline.json`, where the reason is reviewed.
  A renamed analysis that reports numbers cannot: it needs a new baseline snapshot.
  Change the rule and the registry together.
  Published objects do not move when the source does, so a link to the old path stops working once a tree with the new layout replaces the old one.
- **Moving the results or the page** means editing the location constants in `index.html` and in `tests/test_data_browser.py`, running `just test-browser` and publishing the page again.

### What the checks do not cover

The checks compare what the source writes with what the page expects.
None of them watches the bucket, and this repository has no upload command, so an object renamed, overwritten or deleted in the bucket passes all of them.

After any change to the bucket, run `JUMP_BROWSER_LIVE=1 just test-browser -k published_bucket`.
It drives this checkout's page over the published bucket: the overview and every result set it lists, search, an analysis page, the profiles, every figure package with its legend, and whether every reported number names a listed analysis.
It finds a broken layout after the fact; recovering an overwritten or deleted object depends on bucket versioning, which this repository does not control.

Other limits:

- The published-bucket check misses an analysis that reports no numbers and is renamed in a tree read from its summaries: it appears under its new name and only links to the old one break.
  In a tree read from its catalog it compares the listed folders, not every cataloged file; `just verify-baseline` checks those on the local tree before it is published.
- The browser tests need Chrome or Chromium, are not part of `just test` and do not run on pull requests, so they protect a change only when its author runs them.
- An output renamed consistently in its rule and the registry passes `just test`; only `just verify-baseline`, after a full reproduction, catches it.
- Nothing compares the published page with the `index.html` on `main`, which it should equal byte for byte.
  Its SHA256 is the `index.html` entry in `release-files.json`.
- The keys of `dataset_reference.json` have no check of their own; the file is written from the same rows as the CSV that the verifier checks.

## The catalog and its registry

`rule results_catalog` in `rules/catalog.smk` joins two inputs:

- `configs/results_catalog.yaml`, the authored registry, defines membership and sections and gives each analysis a title, a section and optional headline numbers.
  A headline is a key into the envelope's `numbers`.
  Results without an envelope, such as the copairs results database, are described there directly.
- The summary envelopes supply everything computed.
  The catalog restates no scientific value.

The build fails on a missing summary, a declared file that was not written or an unknown headline key.
Catalog paths are relative to the catalog's directory, so the tree can be mirrored anywhere unchanged.
The catalog records the source repository and the commit that produced the results, or no commit when tracked files had uncommitted changes.

The catalog lists analyses from the `all`, `batch_and_source7`, `explore` and `srijit` targets, so it is a target of its own: `just reproduce` builds it last, and `just catalog` builds it with everything it lists.

An entry marked `optional: true` belongs to an opt-in target outside the default run.
It is cataloged when its summary exists and the catalog never causes it to be built, so the pinned output roster is unaffected.
The [positive-control heatmap](development.md#positive-control-reproducibility-heatmap) is registered this way.
The page also shows a folder whose summary the catalog does not list, described by that summary, so an opt-in figure published beside an older catalog still reads properly.

## Adding an analysis

1. Give the analysis its own folder under `data/processed/`, name its envelope `summary.json` and declare it as an output of its rule.
2. Register the summary in `configs/results_catalog.yaml` with a title and a section.
   A title says what the result is; it does not cite manuscript figure numbers or issue numbers, which change.

Three checks keep the registry and the workflow together:

- `tests/test_catalog.py` requires every registered file to have exactly one producing rule and every declared summary output to be registered, as optional when no default target builds it.
  It also enforces the two naming rules in the layout contract.
- The catalog build fails as described above.
- `just verify-baseline` requires the catalog and fails if any envelope in the output tree is not cataloged.

## Testing and publishing

`just test-browser` drives `index.html` in Chrome or Chromium over small output trees written by the real envelope writer and catalog builder, answering the page's bucket requests locally.
`JUMP_BROWSER_LIVE=1 just test-browser` also reads the public bucket.
Playwright is fetched by `uv` for that command and is not part of the pinned environments.

This repository does not write to the bucket.
Publishing `index.html` or an output tree is a separate, authorized step.
After either, run the published-bucket check and compare the published page with `index.html`.
