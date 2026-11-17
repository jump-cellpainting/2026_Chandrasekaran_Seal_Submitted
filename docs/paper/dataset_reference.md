# Dataset reference

The dataset-reference report collates pipeline `summary.json` envelopes without recomputing scientific values.
Each row records its claim key, value, source path, producing rule and derivation key.

`rule dataset_reference_manifest` in `rules/analysis.smk` owns the exact summary membership.
`jump_production.workflow` maps each declared summary path to one producer, rejecting missing, ambiguous or duplicate entries.
`notebooks/dataset_reference.py` reads that manifest and emits `dataset_reference.json` and `dataset_reference.csv`.
In a checkout all summaries must exist locally, so historical remote results cannot silently replace a missing computation.
The standalone notebook can display the existing published S3 reports; those historical reports are separate from this release's rebuilt contract.

An envelope has the shape `{description, numbers, files}`.
Use `jump_production.summaries.write_summary()` and `index_files()` to write it.
Keep reported populations explicit, for example the full released collection versus the analyzed subset.
The canonical analyzed key is defined in `configs/snakemake/analysis.yaml`.

The compound-collection analysis validates its computed values against `references/compound_collection_contract.json`.
The input file checksum and public location identify EPA functional-use provenance; curated FUC and predicted FUP populations can overlap.
MolPort values use the archived quote inputs and the `Price, USD <= 200` filter, rather than current stock availability.

```bash
direnv exec . just dataset-reference
direnv exec . just dataset-reference dry
direnv exec . just dataset-reference view
```

To extend the report, emit the value in its owning analysis, declare its summary as a rule output and add the summary to the manifest rule.
Register every new summary in `configs/results_catalog.yaml` as well, whether or not it joins the report; see the [data browser](../data-browser.md).
Generated reports remain ignored and are rebuilt from the declared inputs.
