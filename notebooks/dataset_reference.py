# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "cloudpathlib[s3]",
# ]
# ///
#
# One manifest per report:
#   - Snakemake writes data/processed/dataset_reference/dataset_reference_manifest.json.
#   - In a checkout, the manifest and every summary must exist locally.
#   - A standalone copy can display the historical public S3 report.
# Keep path resolution self-contained rather than importing jump_production.config; the standalone
# notebook must not depend on the rest of the repository.

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from cloudpathlib import AnyPath, S3Client

    REPO = Path(mo.notebook_dir()).parent
    IS_CHECKOUT = (REPO / "Snakefile").exists()
    MANIFEST_PATH = "data/processed/dataset_reference/dataset_reference_manifest.json"

    S3_BUCKET = "cellpainting-gallery"
    S3_KEY_PREFIX = (
        "cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore"
    )
    S3_DATASTORE = f"s3://{S3_BUCKET}/{S3_KEY_PREFIX}"
    S3Client(no_sign_request=True).set_as_default_client()

    def read_json(rel_path):
        """Read local results; allow historical S3 reads only outside a checkout."""
        if not rel_path.startswith("data/"):
            raise ValueError(f"Manifest path must start with data/: {rel_path}")
        local = REPO / rel_path
        if local.exists():
            return json.loads(local.read_text()), rel_path, "local"

        if IS_CHECKOUT:
            raise FileNotFoundError(f"Build the local summary through Snakemake: {rel_path}")

        remote = AnyPath(f"{S3_DATASTORE}/{rel_path.removeprefix('data/')}")
        try:
            return json.loads(remote.read_text()), str(remote), "s3"
        except Exception as exc:
            raise FileNotFoundError(f"Dataset-reference input exists neither locally nor on S3: {rel_path}") from exc

    MANIFEST, MANIFEST_LOCATION, MANIFEST_SOURCE = read_json(MANIFEST_PATH)
    if MANIFEST.get("schema_version") != 1:
        raise ValueError(f"Unsupported dataset-reference manifest schema: {MANIFEST.get('schema_version')}")
    if not isinstance(MANIFEST.get("summaries"), list):
        raise ValueError("Dataset-reference manifest must contain a summaries list")

    REQUIRED_COMPOUND_COLLECTION_FACTS = (
        "population.analyzed_compounds",
        "chemistry.lipinski_rule_of_5_compliant.count",
        "chemistry.lipinski_rule_of_5_compliant.percent",
        "chemistry.lipinski_rule_of_5_compliant.definition",
        "chemistry.unique_murcko_scaffolds.count",
        "chemistry.unique_murcko_scaffolds.definition",
        "chemistry.pains_flagged.count",
        "chemistry.pains_flagged.definition",
        "clinical_development.categories.Preclinical",
        "clinical_development.categories.Phase 1",
        "clinical_development.categories.Phase 2",
        "clinical_development.categories.Phase 3",
        "clinical_development.categories.Launched",
        "clinical_development.definition",
        "annotations.target_or_moa.count",
        "annotations.target_or_moa.percent",
        "annotations.target_or_moa.definition",
        "annotations.target_or_moa.distinct_from_annotated_any",
        "commercial_availability.molport.count",
        "commercial_availability.molport.percent",
        "commercial_availability.molport.definition",
        "dual_sourcing.count",
        "dual_sourcing.percent",
        "dual_sourcing.definition",
        "functional_use.curated_fuc.count",
        "functional_use.curated_fuc.memberships",
        "functional_use.curated_fuc.represented_categories",
        "functional_use.predicted_fup.count",
        "functional_use.predicted_fup.classes",
        "functional_use.predicted_fup.values",
        "functional_use.predicted_fup.non_null_probabilities_per_compound.min",
        "functional_use.predicted_fup.non_null_probabilities_per_compound.max",
        "functional_use.predicted_fup.coverage_semantics",
        "functional_use.predicted_fup.dominant_class_method",
        "functional_use.top_dominant_fup_classes",
        "functional_use.full_delivery.compounds",
        "functional_use.enrichment_analysis_compounds",
        "functional_use.provenance_status",
        "functional_use.raw_input_available",
        "functional_use.raw_input_path",
        "functional_use.raw_input_sha256",
        "functional_use.source_url",
        "functional_use.dataset",
        "functional_use.source7_excluded",
        "functional_use.ranking_evidence_status",
        "functional_use.interpretation_caveat",
    )

    def fact(claim, value, source, rule, how):
        return {
            "claim": claim,
            "value": value,
            "source": source,
            "rule": rule,
            "how": how,
        }

    def fmt(value):
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.1f}"
        return str(value)

    def show(facts, cols=("claim", "value", "source")):
        return pl.DataFrame(
            [{col: (fmt(item["value"]) if col == "value" else item[col]) for col in cols} for item in facts]
        )

    _key = MANIFEST.get("canonical_key", {})
    _key_text = "/".join(
        str(_key.get(name, "?"))
        for name in (
            "default_dataset",
            "default_columns",
            "default_preprocessing",
            "default_filter",
            "default_activity_params",
        )
    )
    mo.md(f"**Manifest:** `{MANIFEST_LOCATION}` ({MANIFEST_SOURCE}) - key `{_key_text}`")
    return (
        IS_CHECKOUT,
        MANIFEST,
        Path,
        REPO,
        REQUIRED_COMPOUND_COLLECTION_FACTS,
        S3_BUCKET,
        S3_KEY_PREFIX,
        fact,
        json,
        mo,
        pl,
        read_json,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # JUMP dataset - ground-truth reference numbers

    One consolidated table of verifiable numbers, each tagged with its **source** and producing
    **Snakemake rule**. The notebook computes no scientific values. Every number comes from an
    upstream `summary.json` envelope (`{description, numbers, files}`).

    Membership comes from one Snakemake-authored manifest. Both this checkout and the standalone
    notebook read that same manifest; only the storage location varies between local files and their
    exact public S3 mirrors.
    """)
    return


@app.cell
def _(MANIFEST, Path, REQUIRED_COMPOUND_COLLECTION_FACTS, mo, read_json):
    from collections import Counter

    _seen = set()
    _loaded = []
    for _entry in MANIFEST["summaries"]:
        if not (
            isinstance(_entry, dict) and isinstance(_entry.get("rule"), str) and isinstance(_entry.get("path"), str)
        ):
            raise ValueError(f"Invalid dataset-reference manifest entry: {_entry!r}")
        if _entry["path"] in _seen:
            raise ValueError(f"Duplicate dataset-reference manifest path: {_entry['path']}")
        _seen.add(_entry["path"])

        _doc, _location, _source = read_json(_entry["path"])
        if not (
            isinstance(_doc, dict)
            and {"description", "numbers", "files"} <= set(_doc)
            and isinstance(_doc["numbers"], dict)
            and isinstance(_doc["files"], list)
        ):
            raise ValueError(f"Invalid summary envelope: {_entry['path']}")
        _loaded.append(
            {
                "rule": _entry["rule"],
                "path": _entry["path"],
                "location": _location,
                "source": _source,
                "doc": _doc,
            }
        )

    _rule_counts = Counter(item["rule"] for item in _loaded)

    def _leaf_label(path):
        directory = Path(path).parent
        return f"{directory.parent.name}/{directory.name}"

    SUMMARIES = [
        {
            **item,
            "label": (
                item["rule"] if _rule_counts[item["rule"]] == 1 else f"{item['rule']} ({_leaf_label(item['path'])})"
            ),
        }
        for item in _loaded
    ]
    SUMMARIES.sort(key=lambda item: (item["rule"], item["path"]))

    _compound_path = "data/processed/compound-collection-facts/summary.json"
    _compound_summaries = [item for item in SUMMARIES if item["path"] == _compound_path]
    if len(_compound_summaries) != 1:
        raise ValueError(
            "Dataset-reference manifest must contain exactly one compound-collection "
            f"facts summary; found {len(_compound_summaries)}"
        )

    def _has_path(mapping, dotted_path):
        value = mapping
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
        return True

    _compound_numbers = _compound_summaries[0]["doc"]["numbers"]
    _missing_claims = [claim for claim in REQUIRED_COMPOUND_COLLECTION_FACTS if not _has_path(_compound_numbers, claim)]
    if _missing_claims:
        raise ValueError(
            "Compound-collection facts are missing from dataset-reference: " + ", ".join(_missing_claims)
        )

    _lines = [f"**Summary envelopes: {len(SUMMARIES)}**"]
    _lines.append(
        f"**Compound-collection contract:** {len(REQUIRED_COMPOUND_COLLECTION_FACTS)} required facts present"
    )
    _lines += [f"- `{item['rule']}` ({item['source']}) -> `{item['path']}`" for item in SUMMARIES]
    mo.md("\n".join(_lines))
    return (SUMMARIES,)


@app.cell
def _(Path, REPO, S3_BUCKET, S3_KEY_PREFIX, SUMMARIES, fact, mo, show):
    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}

    def _figure_source(rel_path):
        if not rel_path.startswith("data/"):
            return None
        local = REPO / rel_path
        if local.exists():
            return str(local)
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{S3_KEY_PREFIX}/{rel_path.removeprefix('data/')}"

    def _flatten(values, prefix=""):
        rows = []
        for key, value in values.items():
            claim = f"{prefix}{key}"
            if isinstance(value, dict):
                rows += _flatten(value, prefix=f"{claim}.")
            elif isinstance(value, (list, tuple)):
                import json as _json

                rows.append((claim, _json.dumps(value)))
            else:
                rows.append((claim, value))
        return rows

    loop_facts = []
    _blocks = []
    for _summary in SUMMARIES:
        _doc = _summary["doc"]
        _rule = _summary["rule"]
        _facts = [
            fact(key, value, _summary["path"], _rule, f"{_rule}: numbers.{key}")
            for key, value in _flatten(_doc["numbers"])
        ]
        loop_facts += _facts

        _images = {}
        for _file in _doc["files"]:
            if not (
                isinstance(_file, dict) and isinstance(_file.get("path"), str) and isinstance(_file.get("caption"), str)
            ):
                raise ValueError(f"Invalid file entry in {_summary['path']}: {_file!r}")
            if Path(_file["path"]).suffix.lower() in _IMAGE_SUFFIXES:
                _url = _figure_source(_file["path"])
                if _url:
                    _images[_file["caption"]] = mo.image(_url)
        _figure_widget = mo.accordion(_images) if _images else mo.md("_no inline figures available_")
        _blocks.append(
            mo.vstack(
                [
                    mo.md(f"## {_summary['label']}"),
                    mo.md(f"*{_doc['description']}*"),
                    mo.md(f"`{_summary['location']}` ({_summary['source']})"),
                    mo.ui.table(show(_facts, cols=("claim", "value")), selection=None),
                    mo.md(
                        f"**Files ({len(_doc['files'])}):** "
                        + ", ".join(f"`{Path(item['path']).name}`" for item in _doc["files"])
                    ),
                    _figure_widget,
                ]
            )
        )
    mo.vstack(_blocks)
    return (loop_facts,)


@app.cell
def _(IS_CHECKOUT, REPO, json, loop_facts, mo, pl, show):
    GOLD = loop_facts
    if IS_CHECKOUT:
        _output = REPO / "data/processed/dataset_reference/dataset_reference.json"
        _output.parent.mkdir(parents=True, exist_ok=True)
        _output.write_text(json.dumps(GOLD, indent=2) + "\n")
        _csv = REPO / "data/processed/dataset_reference/dataset_reference.csv"
        pl.DataFrame(
            [
                {
                    "claim": item["claim"],
                    "value": str(item["value"]),
                    "source": item["source"],
                    "rule": item["rule"],
                    "how": item["how"],
                }
                for item in GOLD
            ]
        ).write_csv(_csv)
        _note = "Wrote `data/processed/dataset_reference/dataset_reference.{json,csv}`."
    else:
        _note = "Standalone view - artifacts not written."
    mo.vstack(
        [
            mo.md(f"**Consolidated reference artifact - {len(GOLD)} numbers.** {_note}"),
            mo.ui.table(show(GOLD, cols=("claim", "value", "source", "rule")), selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
