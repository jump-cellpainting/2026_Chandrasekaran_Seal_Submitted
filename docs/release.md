# Source-release contract

A source release contains the pipeline, the pinned environments and input checksums, the analysis notebooks, method references, the results-browser page, this documentation and the tests.
Generated reports and local working material are excluded.
The public data, the null cache and the baseline snapshot are separate, read-only resources: a release never overwrites or republishes them.

## Baseline projection

The baseline snapshot is immutable.
`configs/scientific_baseline.json` lists every deliberate difference between that snapshot and what the current source writes, and the verifier applies those differences and no others.
[The reproduction guide](reproduce.md#what-is-pinned-and-checked) describes the checks and tolerances.

| Check | Snapshot | Current source |
| --- | ---: | ---: |
| Required processed paths | 2,005 | 2,002 retained + 1 added |
| Artifact checks | 1,146 | 998 exact hashes + 146 Hydra configurations |
| Dataset-reference claim rows | 1,581 | 1,571 |
| Curated compound tables | 4 | 4 |
| Separately compared scientific summaries | 2 | 2 |

The differences are:

- **Three retired outputs**: the files of an editorial data-availability audit.
  Reintroducing one fails.
- **Ten removed claim rows**: narrative assertions and document pointers in the compound-collection summary, which are not computed values.
- **Seven replaced claim values or keys**: six are provenance metadata of the compound-collection summary, such as its public input URL and the `numerical_contract` key names; the seventh is the provenance hash described below.
- **Four replaced exact hashes**: the compound-collection summary; the binary-trait odds-ratio table, in which one display-selection column was renamed; and the two summaries described below.
- **One added output**: `data/processed/catalog.json`, the [results catalog](data-browser.md).
  It is derived from the summary envelopes and an authored registry, restates no scientific value, and is validated against the output tree instead of against baseline bytes.

No scientific count, value, threshold or tolerance differs, and any other missing or extra output or claim still fails.

Summary envelopes that are recorded elsewhere by content hash need care.
The compound-trait odds-ratio summary and the compound-activity figure summary store the SHA256 of the functional-use summary as provenance, and the first also reports it as a claim.
Any change to the bytes of the functional-use summary therefore moves two pinned files and one claim value; the projection carries those three replacements because that summary now declares its files relative to the checkout.
Re-emitting only the changed envelope does not show such a dependency.

Checking retained outputs with `just verify-baseline` validates the verifier and the projection without recomputing anything, so a change to executable source still needs a full GPU reproduction.

## Reviewed source snapshot

`release-files.json` lists every source path with its reviewed SHA256, excluding its own hash.
`just check-release` rejects extra or missing tracked files, symlinks, unmerged entries, unstaged content and changed reviewed bytes.
It never regenerates the allowlist.
Review each changed file's full content, including embedded prose and links, before updating its entry.
Untracked and ignored local files are not part of a release.

After review and explicit approval to publish, prepare an empty destination and export only the reviewed commit's tree:

```bash
direnv exec . just check-release
git diff --quiet
git diff --cached --quiet
git archive --format=tar HEAD > /absolute/path/outside-checkout/reviewed-source.tar
```

Inspect the archive against the allowlist, extract it into an empty directory and initialize a new Git repository there.
Create a signed root commit and verify that it has no parent.
Use that repository's URL and release commit in the reproduction guide.
Do not copy `.git`, create a Git bundle from the working repository, mirror refs, import branches or tags, or change the working repository's visibility.
The EC2 launcher transfers a Git bundle for execution, so run it from the fresh-history repository when reproducing a public release.
This procedure does not revoke previously downloaded copies.
