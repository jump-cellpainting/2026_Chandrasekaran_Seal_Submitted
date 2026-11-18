"""Contract tests for the results catalog read by the publication data browser."""

import json
from pathlib import Path

import pytest
import yaml

from jump_production.catalog import (
    build_catalog,
    load_registry,
    optional_summaries,
    registry_inputs,
    validate_catalog,
    write_catalog,
)
from jump_production.config import PROJ_ROOT
from jump_production.summaries import index_files, write_summary
from jump_production.workflow import declared_outputs, matches_output

KEY = {"default_dataset": "compound_no_source7"}
DECLARATIONS = [
    {"rule": "activity", "output": "data/processed/activity/summary.json", "script": "notebooks/activity.py"},
    {"rule": "figure", "output": "data/processed/figures/{name}/summary.json", "script": "notebooks/figure.py"},
    {"rule": "runs", "output": "data/processed/runs/.complete", "script": None},
    {"rule": "store", "output": "data/processed/results.duckdb", "script": None},
]
REGISTRY = {
    "schema_version": 1,
    "repository": "https://example.org/source",
    "sections": [
        {"id": "figures", "title": "Figures", "description": "Rendered figures."},
        {"id": "activity", "title": "Activity", "description": "Activity results."},
    ],
    "entries": [
        {"summary": "data/processed/figures/collection/summary.json", "section": "figures", "title": "Collection"},
        {
            "summary": "data/processed/activity/summary.json",
            "section": "activity",
            "title": "Activity headline",
            "highlights": [{"key": "counts.active", "label": "Active compounds"}],
        },
        {
            "path": "data/processed/runs/",
            "requires": "data/processed/runs/.complete",
            "section": "activity",
            "title": "Per-run outputs",
            "description": "One folder\n  per run.",
            "files": ["first/results.csv"],
        },
        {
            "path": "data/processed/results.duckdb",
            "section": "activity",
            "title": "Results database",
            "description": "Every run in one database.",
        },
    ],
}


def build_tree(root: Path) -> Path:
    """Write a small output tree with the real envelope writer, in both path conventions."""
    processed = root / "data/processed"
    figure = processed / "figures/collection"
    figure.mkdir(parents=True)
    (figure / "panel.png").write_bytes(b"png\n")
    (figure / "legend.md").write_text("# legend\n")
    write_summary(
        figure / "summary.json",
        description="A figure package.",
        numbers={"panels": 1},
        files=[*index_files(figure, relative_to=figure), {"path": "legend.md", "kind": "data", "caption": "Legend"}],
    )
    activity = processed / "activity"
    (activity / "tables").mkdir(parents=True)
    (activity / "tables/rates.csv").write_text("rate\n0.15\n")
    write_summary(
        activity / "summary.json",
        description="Activity counts.",
        numbers={"counts": {"analyzed": 10, "active": 3}, "notes": ["a", "b"]},
        files=[
            {"path": "data/processed/activity/tables/rates.csv", "caption": "Activity rates"},
            "data/processed/activity/summary.json",
        ],
    )
    (processed / "runs/first").mkdir(parents=True)
    (processed / "runs/.complete").write_text("")
    (processed / "runs/first/results.csv").write_text("value\n1\n")
    (processed / "results.duckdb").write_bytes(b"db\n")
    return root


def catalog_for(root: Path, registry=REGISTRY) -> dict:
    return build_catalog(registry, root=root, declarations=DECLARATIONS, canonical_key=KEY, revision="a" * 40)


def test_catalog_joins_registry_titles_with_envelope_content(tmp_path):
    catalog = catalog_for(build_tree(tmp_path))

    assert catalog["source"] == {"repository": "https://example.org/source", "revision": "a" * 40}
    assert catalog["canonical_key"] == KEY
    assert [section["id"] for section in catalog["sections"]] == ["figures", "activity"]
    figure, activity, runs, store = catalog["entries"]

    # Both path conventions resolve to paths relative to the catalog's own directory.
    assert figure["id"] == "figures/collection" and figure["path"] == "figures/collection/"
    assert figure["summary"] == "figures/collection/summary.json"
    assert figure["files"] == [
        {"path": "figures/collection/panel.png", "kind": "figure", "caption": "panel"},
        {"path": "figures/collection/legend.md", "kind": "data", "caption": "Legend"},
    ]
    assert (figure["rule"], figure["script"]) == ("figure", "notebooks/figure.py")

    # Descriptions and values come from the envelope; the self-listing is dropped.
    assert activity["title"] == "Activity headline" and activity["description"] == "Activity counts."
    assert activity["files"] == [{"path": "activity/tables/rates.csv", "kind": "table", "caption": "Activity rates"}]
    assert activity["highlights"] == [{"key": "counts.active", "label": "Active compounds", "value": 3}]
    assert activity["numbers"] == 4

    assert runs["summary"] is None and runs["path"] == "runs/" and runs["description"] == "One folder per run."
    assert runs["files"] == [{"path": "runs/first/results.csv", "kind": "table", "caption": "results.csv"}]
    assert (store["id"], store["path"], store["rule"]) == ("results.duckdb", "", "store")
    assert store["files"] == [{"path": "results.duckdb", "kind": "data", "caption": "results.duckdb"}]


def test_written_catalog_validates_against_its_tree(tmp_path):
    root = build_tree(tmp_path)
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(REGISTRY))

    write_catalog(
        root / "data/processed/catalog.json",
        registry,
        root=root,
        declarations=DECLARATIONS,
        canonical_key=KEY,
        revision=None,
    )

    assert validate_catalog(root) == (4, 5)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("declared_missing", "was not written"),
        ("declared_elsewhere", "another machine"),
        ("declared_outside", "normalized path under"),
        ("not_envelope", "not a described envelope"),
        ("no_headline", "has no number"),
        ("headline_group", "not a single value"),
        ("no_producer", "No workflow rule declares"),
        ("resource_unbuilt", "was not built"),
        ("resource_member", "was not written"),
        ("duplicate_id", "share an id"),
    ],
)
def test_catalog_build_fails_closed(tmp_path, damage, message):
    root = build_tree(tmp_path)
    registry = json.loads(json.dumps(REGISTRY))
    activity = root / "data/processed/activity/summary.json"
    document = json.loads(activity.read_text())
    if damage == "declared_missing":
        (root / "data/processed/activity/tables/rates.csv").unlink()
    elif damage == "declared_elsewhere":
        document["files"][0]["path"] = "/home/someone/checkout/data/processed/activity/tables/rates.csv"
    elif damage == "declared_outside":
        document["files"][0]["path"] = "data/raw/profiles/compound.parquet"
    elif damage == "not_envelope":
        document = {"numbers": {}}
    elif damage == "no_headline":
        registry["entries"][1]["highlights"] = [{"key": "counts.inactive", "label": "Inactive"}]
    elif damage == "headline_group":
        registry["entries"][1]["highlights"] = [{"key": "counts", "label": "Counts"}]
    elif damage == "no_producer":
        registry["entries"][1]["summary"] = "data/processed/unknown/summary.json"
        (root / "data/processed/unknown").mkdir()
        (root / "data/processed/unknown/summary.json").write_text(activity.read_text())
    elif damage == "resource_unbuilt":
        (root / "data/processed/runs/.complete").unlink()
    elif damage == "resource_member":
        (root / "data/processed/runs/first/results.csv").unlink()
    elif damage == "duplicate_id":
        registry["entries"].append(dict(registry["entries"][0]))
    activity.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        catalog_for(root, registry)


def test_opt_in_analysis_is_cataloged_only_when_it_was_built(tmp_path):
    root = build_tree(tmp_path)
    registry = json.loads(json.dumps(REGISTRY))
    optional = {
        "summary": "data/processed/figures/heatmap/summary.json",
        "section": "figures",
        "title": "Opt-in heatmap",
        "optional": True,
        "highlights": [{"key": "pairs", "label": "Pairs"}],
    }
    registry["entries"].insert(1, optional)

    # Never a reason to build it: the workflow's required inputs do not name it.
    assert optional["summary"] not in registry_inputs(registry)
    assert optional_summaries(registry) == [optional["summary"]]
    assert [entry["id"] for entry in catalog_for(root, registry)["entries"]] == [
        "figures/collection", "activity", "runs", "results.duckdb",
    ]

    heatmap = root / "data/processed/figures/heatmap"
    heatmap.mkdir()
    (heatmap / "heatmap.png").write_bytes(b"png\n")
    write_summary(heatmap / "summary.json", description="A heatmap.", numbers={"pairs": 71}, files=["heatmap.png"])
    catalog = catalog_for(root, registry)
    assert [entry["id"] for entry in catalog["entries"]][:2] == ["figures/collection", "figures/heatmap"]
    assert catalog["entries"][1]["highlights"] == [{"key": "pairs", "label": "Pairs", "value": 71}]
    (root / "data/processed/catalog.json").write_text(json.dumps(catalog))
    assert validate_catalog(root) == (5, 6)

    # Once built it is held to the same checks as any other analysis.
    (heatmap / "heatmap.png").unlink()
    with pytest.raises(ValueError, match="was not written"):
        catalog_for(root, registry)


def test_absolute_paths_inside_the_checkout_are_still_resolved(tmp_path):
    root = build_tree(tmp_path)
    activity = root / "data/processed/activity/summary.json"
    document = json.loads(activity.read_text())
    document["files"][0]["path"] = str(root.resolve() / "data/processed/activity/tables/rates.csv")
    activity.write_text(json.dumps(document))

    assert catalog_for(root)["entries"][1]["files"][0]["path"] == "activity/tables/rates.csv"


@pytest.mark.parametrize("damage", ["uncataloged_envelope", "missing_file", "missing_summary", "schema"])
def test_validation_requires_every_envelope_and_file(tmp_path, damage):
    root = build_tree(tmp_path)
    processed = root / "data/processed"
    (processed / "catalog.json").write_text(json.dumps(catalog_for(root)))
    assert validate_catalog(root) == (4, 5)

    if damage == "uncataloged_envelope":
        (processed / "new").mkdir()
        (processed / "new/summary.json").write_text((processed / "activity/summary.json").read_text())
    elif damage == "missing_file":
        (processed / "figures/collection/panel.png").unlink()
    elif damage == "missing_summary":
        (processed / "activity/summary.json").unlink()
    elif damage == "schema":
        (processed / "catalog.json").write_text(json.dumps({"schema_version": 2}))

    with pytest.raises((ValueError, OSError)):
        validate_catalog(root)


@pytest.mark.parametrize(
    "change",
    [
        lambda registry: registry.update(schema_version=2),
        lambda registry: registry.update(repository="git@example.org:source"),
        lambda registry: registry["sections"].append({"id": "empty", "title": "Empty", "description": "Unused."}),
        lambda registry: registry["sections"].append(dict(registry["sections"][0])),
        lambda registry: registry["entries"][0].update(section="unknown"),
        lambda registry: registry["entries"][0].update(path="data/processed/figures/"),
        lambda registry: registry["entries"][0].pop("title"),
        lambda registry: registry["entries"][0].update(description="Authored text for an envelope"),
        lambda registry: registry["entries"][1].update(highlights=[{"key": "counts.active"}]),
        lambda registry: registry["entries"][3].pop("description"),
        lambda registry: registry["entries"][0].update(optional="yes"),
        lambda registry: registry["entries"][3].update(optional=True),
        # A section must not depend on an opt-in entry to have something to show.
        lambda registry: registry["entries"][0].update(optional=True),
    ],
)
def test_registry_rejects_entries_the_browser_could_not_present(tmp_path, change):
    registry = json.loads(json.dumps(REGISTRY))
    change(registry)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry))

    with pytest.raises(ValueError):
        load_registry(path)


def layout_problems(registry: dict) -> list[str]:
    """Return registry paths that break the names the published page reads without a catalog.

    The page finds analyses in such a tree by summary file name and figure packages by folder, and
    it labels a cataloged entry as a figure by section id. See the layout contract in
    docs/data-browser.md.
    """
    problems = []
    for entry in registry["entries"]:
        location = entry.get("summary", entry.get("path"))
        if "summary" in entry and not location.endswith("summary.json"):
            problems.append(f"summary file name must end in summary.json: {location}")
        if location.startswith("data/processed/figures/") != (entry["section"] == "figures"):
            problems.append(f"section figures holds exactly the entries under data/processed/figures/: {location}")
    return problems


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda entries: entries[1].update(summary="data/processed/activity/results.json"), "must end in summary.json"),
        (lambda entries: entries[1].update(section="figures"), "section figures holds exactly"),
        (lambda entries: entries[0].update(summary="data/processed/panels/collection/summary.json"), "section figures holds exactly"),
    ],
)
def test_layout_check_names_what_the_browser_could_not_find_without_a_catalog(change, message):
    registry = json.loads(json.dumps(REGISTRY))
    assert layout_problems(registry) == []
    change(registry["entries"])

    assert [problem for problem in layout_problems(registry) if message in problem]


def test_committed_registry_matches_the_workflow():
    """Every registered file has one producing rule, and every declared summary is registered."""
    registry = load_registry(PROJ_ROOT / "configs/results_catalog.yaml")
    assert layout_problems(registry) == []
    declarations = declared_outputs()
    inputs = [*registry_inputs(registry), *optional_summaries(registry)]
    assert len(inputs) == len(set(inputs))

    for path in inputs:
        producers = {d["rule"] for d in declarations if matches_output(d["output"], path)}
        assert len(producers) == 1, f"{path}: {sorted(producers)}"

    registered = [entry["summary"] for entry in registry["entries"] if "summary" in entry]
    summary_outputs = {
        d["output"]
        for d in declarations
        if d["output"].startswith("data/processed/") and d["output"].endswith("summary.json")
    }
    unregistered = sorted(
        output for output in summary_outputs if not any(matches_output(output, path) for path in registered)
    )
    assert not unregistered

    # The page reads two files by fixed name at the root of a results tree: the catalog and the
    # reported numbers. Neither moves without a change to the page and a new publication.
    fixed = {"data/processed/catalog.json", "data/processed/dataset_reference/dataset_reference.json"}
    assert {d["output"]: d["rule"] for d in declarations if d["output"] in fixed} == {
        "data/processed/catalog.json": "results_catalog",
        "data/processed/dataset_reference/dataset_reference.json": "dataset_reference",
    }

    # An entry is optional exactly when no default target builds it. The positive-control heatmap
    # is the one opt-in figure: cataloging it must not pull it into the pinned output roster.
    assert optional_summaries(registry) == [
        "data/processed/figures/figure_positive_control_reproducibility/summary.json"
    ]
