"""Build the canonical manifest consumed by the dataset-reference notebook.

Snakemake owns membership: the summary inputs of ``rule dataset_reference_manifest`` are the sole
list of envelopes included in the paper report. This module maps each concrete input path to exactly
one producing rule and serializes that mapping. The notebook reads the serialized manifest both in a
checkout and standalone, so it never performs its own workflow or S3 discovery.
"""

import json
import re
from pathlib import Path

from jump_production import config as _cfg


def _rule_script(rule) -> str | None:
    """Return the analysis script a rule declares among its named inputs, if any."""
    names = list(rule.input.keys())
    for name in ("script", *(name for name in names if name.endswith("_script"))):
        value = rule.input.get(name) if name in names else None
        if isinstance(value, str):
            return str(value)
    return None


def declared_outputs_from_rules(rules) -> list[dict]:
    """Return every output declaration, with the producing rule and its analysis script."""
    return [
        {"rule": rule.name, "output": str(out), "script": _rule_script(rule)}
        for rule in rules
        for out in rule.output
    ]


def declared_json_outputs_from_rules(rules) -> list[dict]:
    """Return JSON output declarations from an already-parsed rule collection."""
    return [
        {"rule": rule.name, "output": str(out)}
        for rule in rules
        for out in rule.output
        if str(out).endswith(".json")
    ]


def _parsed_rules(snakefile=None) -> list:
    """Parse the Snakefile in-process via the Snakemake API and return its rules (no DAG build)."""
    from snakemake.api import (
        ConfigSettings,
        DeploymentSettings,
        OutputSettings,
        ResourceSettings,
        SnakemakeApi,
        StorageSettings,
        WorkflowSettings,
    )

    snakefile = Path(snakefile or (_cfg.PROJ_ROOT / "Snakefile")).resolve()
    with SnakemakeApi(OutputSettings(printshellcmds=False)) as api:
        workflow = api.workflow(
            snakefile=snakefile,
            resource_settings=ResourceSettings(),
            config_settings=ConfigSettings(),
            storage_settings=StorageSettings(),
            workflow_settings=WorkflowSettings(),
            deployment_settings=DeploymentSettings(),
        )._workflow  # property parses the Snakefile on access; no DAG is built
        return list(workflow.rules)


def declared_json_outputs(snakefile=None) -> list[dict]:
    """Return ``[{"rule", "output"}, ...]`` for every `.json` output declared in the workflow.

    Parses the Snakefile in-process via the Snakemake API (no DAG build). `output` is the declared
    path with f-strings resolved; wildcard rules yield an unexpanded pattern.
    """
    return declared_json_outputs_from_rules(_parsed_rules(snakefile))


def declared_outputs(snakefile=None) -> list[dict]:
    """Return ``[{"rule", "output", "script"}, ...]`` for every output declared in the workflow."""
    return declared_outputs_from_rules(_parsed_rules(snakefile))


_WILDCARD = re.compile(r"\{[^{}]+\}")


def matches_output(pattern: str, path: str) -> bool:
    """Return whether a concrete path matches a declared Snakemake output pattern."""
    pieces = []
    start = 0
    for match in _WILDCARD.finditer(pattern):
        pieces.append(re.escape(pattern[start : match.start()]))
        pieces.append(r"[^/]+")
        start = match.end()
    pieces.append(re.escape(pattern[start:]))
    return re.fullmatch("".join(pieces), path) is not None


def build_dataset_reference_manifest(summary_paths, *, canonical_key, declarations=None) -> dict:
    """Build and validate the manifest for concrete summary paths.

    ``declarations`` is injectable so producer mapping can be tested without parsing Snakemake.
    """
    declarations = declarations if declarations is not None else declared_json_outputs()
    paths = [str(path) for path in summary_paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Dataset-reference summary paths must be unique")

    summaries = []
    for path in paths:
        if not (path.startswith("data/") and path.endswith(".json")):
            raise ValueError(f"Dataset-reference input is not a data JSON path: {path}")
        producers = {decl["rule"] for decl in declarations if matches_output(str(decl["output"]), path)}
        if len(producers) != 1:
            raise ValueError(f"Expected exactly one producer for {path}, found {sorted(producers)}")
        summaries.append({"rule": producers.pop(), "path": path})

    return {
        "schema_version": 1,
        "canonical_key": dict(canonical_key),
        "summaries": summaries,
    }


def write_dataset_reference_manifest(output_path, summary_paths, *, canonical_key, declarations=None) -> dict:
    """Write the canonical dataset-reference manifest and return its document."""
    manifest = build_dataset_reference_manifest(
        summary_paths,
        canonical_key=canonical_key,
        declarations=declarations,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(declared_json_outputs()))
