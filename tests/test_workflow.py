import pytest

from jump_production.workflow import (
    build_dataset_reference_manifest,
    declared_json_outputs_from_rules,
    declared_outputs_from_rules,
)

DECLARATIONS = [
    {"rule": "exact", "output": "data/processed/exact/summary.json"},
    {
        "rule": "wildcard",
        "output": "data/processed/by-config/{representation}/summary.json",
    },
]
KEY = {
    "default_dataset": "compound_no_source7",
    "default_columns": "feat_all",
    "default_preprocessing": "activity_no_target2",
    "default_filter": "all_sources",
    "default_activity_params": "default",
}


class _Rule:
    def __init__(self, name, *outputs, **inputs):
        self.name = name
        self.output = outputs
        self.input = inputs


def test_declared_json_outputs_from_parsed_rules():
    rules = [
        _Rule("summary", "data/processed/example/summary.json", "example.csv"),
        _Rule("other", "data/processed/other/summary.json"),
    ]

    assert declared_json_outputs_from_rules(rules) == [
        {"rule": "summary", "output": "data/processed/example/summary.json"},
        {"rule": "other", "output": "data/processed/other/summary.json"},
    ]


def test_declared_outputs_name_the_analysis_script():
    rules = [
        _Rule("notebook", "out/summary.json", "out/table.csv", db="results.duckdb", script="notebooks/a.py"),
        _Rule("runner", "out/compare.json", comparison_script="notebooks/b.py", runner_script="scripts/c.py"),
        _Rule("aggregate", "out/.complete"),
        _Rule("by_function", "out/other.json", script=lambda wildcards: "notebooks/d.py"),
    ]

    assert declared_outputs_from_rules(rules) == [
        {"rule": "notebook", "output": "out/summary.json", "script": "notebooks/a.py"},
        {"rule": "notebook", "output": "out/table.csv", "script": "notebooks/a.py"},
        {"rule": "runner", "output": "out/compare.json", "script": "notebooks/b.py"},
        {"rule": "aggregate", "output": "out/.complete", "script": None},
        {"rule": "by_function", "output": "out/other.json", "script": None},
    ]


def test_build_dataset_reference_manifest_maps_producers():
    manifest = build_dataset_reference_manifest(
        [
            "data/processed/exact/summary.json",
            "data/processed/by-config/morgan/summary.json",
        ],
        canonical_key=KEY,
        declarations=DECLARATIONS,
    )

    assert manifest == {
        "schema_version": 1,
        "canonical_key": KEY,
        "summaries": [
            {"rule": "exact", "path": "data/processed/exact/summary.json"},
            {
                "rule": "wildcard",
                "path": "data/processed/by-config/morgan/summary.json",
            },
        ],
    }


@pytest.mark.parametrize(
    ("paths", "declarations", "message"),
    [
        (
            ["data/processed/missing/summary.json"],
            DECLARATIONS,
            "Expected exactly one producer",
        ),
        (
            ["data/processed/exact/summary.json"],
            [*DECLARATIONS, {"rule": "duplicate", "output": "data/processed/exact/{name}.json"}],
            "Expected exactly one producer",
        ),
        (
            ["data/processed/exact/summary.json"] * 2,
            DECLARATIONS,
            "must be unique",
        ),
        (["configs/summary.json"], DECLARATIONS, "not a data JSON path"),
    ],
)
def test_build_dataset_reference_manifest_rejects_invalid_membership(paths, declarations, message):
    with pytest.raises(ValueError, match=message):
        build_dataset_reference_manifest(
            paths,
            canonical_key=KEY,
            declarations=declarations,
        )
