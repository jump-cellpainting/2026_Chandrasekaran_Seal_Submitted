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

Four exact-check hashes are replaced with hashes of the deterministically rewritten metadata artifacts.
The compound-collection summary's envelope description is also rewritten as factual coverage text and is covered by its replacement hash, outside the claim-row count.
Of the remaining checks, 994 exact hashes are unchanged and 146 Hydra configurations are compared against authenticated baseline files after normalizing only their runtime checkout roots.
All configuration choices, relative path suffixes and other fields remain exact.
The timing-derived fingerprint `fastest.name` is treated like its runtime; the claim must still be present, and scientific transformation results remain checked.
Seven metadata replacements are explicit in the projection; scientific counts, values, thresholds and tolerances are unchanged.
Missing or extra outputs and claims still fail, including reintroduction of retired outputs.

The projection names one output that postdates the snapshot: `data/processed/catalog.json`, the [results catalog](data-browser.md).
It is derived from the summary envelopes and an authored registry, and restates no scientific value.
The verifier requires it, checks that every entry and file it lists exists and that every summary envelope in the output tree is cataloged, and compares it with no baseline bytes.
Any other path absent from the snapshot still fails.
Fifteen summary envelopes used to declare their files by absolute path and now declare them relative to the checkout.
Their bytes already depended on where the repository was cloned, so none was among the exact-hash checks; their descriptions and `numbers` are unchanged.

One of the fifteen is recorded elsewhere by content hash.
The compound-trait odds-ratio summary stores the SHA256 of the functional-use summary as provenance and reports it as a claim, and the compound-activity figure summary stores the hashes of both.
Those two envelopes are exact-checked, so rewriting the 20 declared paths of the functional-use summary changes one provenance hash, and through it two pinned files and one claim value.
The projection replaces those two checksums and that claim value, which accounts for two of the four replaced hashes and one of the seven metadata replacements above.
The replacements were derived from the authenticated snapshot bytes alone: rewrite the declared paths, then substitute the dependent hashes.
A fresh reproduction wrote exactly those bytes, so no description, number or other field differs from the snapshot.
That provenance hash used to depend on where the repository was cloned; it no longer does.
A local preflight that re-emitted only the fifteen envelopes did not show this dependency, and the first fresh reproduction of the change failed on it.
The original snapshot, input checksum manifest and null cache are not overwritten or republished.

Pre-publication source revision `7404bb75a420e11dc673a7a6ed55c2d61a86bee8` passed a fresh L40S reproduction of the projection as it then stood; the exact machine, timing and run markers are recorded in [the reproduction guide](reproduce.md).
Public revision `58a5b12c442d430613aabf7d1f96b658b0da1fce` passed a fresh L40S reproduction of the projection as described above, including the results catalog and all four replaced hashes; the same guide records that run and the failed run that preceded it.
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
