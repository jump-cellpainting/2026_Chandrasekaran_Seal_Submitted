# Dataset reference

The dataset-reference report collates the `numbers` of pipeline [summary envelopes](../analysis.md#summary-envelopes) without recomputing any value.
Each row records its claim key, value, source path, producing rule and derivation key.
It is the list of reported numbers that the baseline check compares and the data browser shows.

```bash
direnv exec . just dataset-reference        # build
direnv exec . just dataset-reference dry    # preview the Snakemake dependencies
direnv exec . just dataset-reference view   # open as a marimo notebook
```

`rule dataset_reference_manifest` in `rules/analysis.smk` is the only definition of which summaries belong to the report.
`jump_production.workflow` maps each declared summary to exactly one producing rule and rejects missing, ambiguous or duplicate entries.
`notebooks/dataset_reference.py` reads that manifest and writes `dataset_reference.json` and `dataset_reference.csv`.
In a checkout every summary must exist locally, so published results cannot stand in for a missing computation.
Run standalone, the notebook can also display the reports already published on S3, which belong to the published datastore and not to this checkout.
Generated reports are ignored by Git and rebuilt from the declared inputs.

To add a number, emit it in the summary of the analysis that computes it, declare that summary as a rule output and add it to the manifest rule.
Every summary is also [registered for the data browser](../data-browser.md#adding-an-analysis), whether or not it joins the report.

The compound-collection analysis checks its computed values against `references/compound_collection_contract.json`.
The input checksum and public location identify the EPA functional-use source; curated FUC and predicted FUP populations can overlap.
MolPort values use the archived quote inputs and the filter `Price, USD <= 200`, not current stock availability.
