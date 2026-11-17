# Source-release contract

The source tree contains the scientific pipeline, pinned environments and inputs, analysis notebooks, figures, method references, results-browser source, reproduction instructions and validation tests.
Generated reports and local working material are excluded from source releases.
Public data and the existing baseline remain separate, read-only resources.

## Baseline projection

`configs/scientific_baseline.json` defines a finite projection of the immutable cached L40S snapshot.
It retires the three outputs of the editorial data-availability audit.
The compound-collection report retains every computed fact and input checksum, while removing non-computable narrative assertions and document pointers.
Its numerical contract uses a public input URL and factual provenance.
The binary-trait table renames one display-selection column; its values, row membership, estimates and intervals are unchanged.

| Check | Original snapshot | Source release |
| --- | ---: | ---: |
| Required processed paths | 2,005 | 2,002 retained + 1 added |
| Artifact checks | 1,146 | 998 exact hashes + 146 Hydra configurations |
| Dataset-reference claim rows | 1,581 | 1,571 |
| Curated compound tables | 4 | 4 |
| Separately compared scientific summary objects | 2 | 2 |

Two exact-check hashes are replaced with hashes of the deterministically rewritten metadata artifacts.
The compound-collection summary's envelope description is also rewritten as factual coverage text and is covered by its replacement hash, outside the claim-row count.
Of the remaining checks, 996 exact hashes are unchanged and 146 Hydra configurations are compared against authenticated baseline files after normalizing only their runtime checkout roots.
All configuration choices, relative path suffixes and other fields remain exact.
The timing-derived fingerprint `fastest.name` is treated like its runtime; the claim must still be present, and scientific transformation results remain checked.
Six metadata replacements are explicit in the projection; scientific counts, values, thresholds and tolerances are unchanged.
Missing or extra outputs and claims still fail, including reintroduction of retired outputs.

The projection names one output that postdates the snapshot: `data/processed/catalog.json`, the [results catalog](data-browser.md).
It is derived from the summary envelopes and an authored registry, and restates no scientific value.
The verifier requires it, checks that every entry and file it lists exists and that every summary envelope in the output tree is cataloged, and compares it with no baseline bytes.
Any other path absent from the snapshot still fails.
Fifteen summary envelopes used to declare their files by absolute path and now declare them relative to the checkout.
Their bytes already depended on where the repository was cloned, so none was among the exact-hash checks; their descriptions and `numbers` are unchanged.
All 998 exact hashes, 50 summary envelopes among them, the 146 Hydra configurations and the 1,571 claim rows are checked exactly as before.
The original snapshot, input checksum manifest and null cache are not overwritten or republished.

Pre-publication source revision `7404bb75a420e11dc673a7a6ed55c2d61a86bee8` passed a fresh L40S reproduction, including every projected check above; the exact machine, timing and run markers are recorded in [the reproduction guide](reproduce.md).
The public repository has a new signed root commit with byte-identical executable source, scientific configurations and dependency/input pins; only documentation and its allowlist hashes were adapted for publication.
Its stable well ordering before cell-density regression restores the existing exact CSV baseline without changing scientific calculations or tolerances.
A local projected-snapshot check validates the adapter and reporting contract without recomputing scientific results, so executable changes still require a full GPU reproduction.

## Reviewed source snapshot

`release-files.json` lists every source path with its reviewed SHA256, excluding its own recursive hash.
`just check-release` rejects extra or missing tracked files, symlinks, unmerged entries, unstaged content and changed reviewed bytes.
It never automatically regenerates the allowlist.
Review each changed file's full content, including embedded prose and links, before updating its entry.
Untracked ignored local files are not in the release index and must not be copied into a release.

After review and explicit approval to publish, prepare an empty destination and export only the reviewed commit's tree:

```bash
direnv exec . just check-release
git diff --quiet
git diff --cached --quiet
git archive --format=tar HEAD > /absolute/path/outside-checkout/reviewed-source.tar
```

Inspect the archive against the allowlist, extract into an empty directory, and initialize a new Git repository there.
Create a signed root commit and verify that it has no parent.
Use that new repository's URL and release commit in the reproduction guide.
Do not copy `.git`, create a Git bundle from the working repository, mirror refs, import branches or tags, or change the working repository's visibility.
The EC2 launcher transfers a Git bundle for execution, so use it from the fresh-history source repository when running a public release.
This procedure does not revoke previously downloaded copies.
