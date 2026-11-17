import csv
import json

import pytest

from jump_production.processing.verify_baseline import verify_claims, verify_curated


def write_claims(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["claim", "value", "source", "rule", "how"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def claims():
    return [
        dict(claim="active_count", value="17078", source="activity.json", rule="activity", how="count"),
        dict(
            claim="transformations",
            value=json.dumps([dict(name="L2", recall=0.95, transform_time_full_s=1, transform_time_sample_s=0.1)]),
            source="data/processed/fingerprint-metrics/fingerprint_metrics_summary.json",
            rule="fingerprint_metrics", how="transformations",
        ),
    ]


@pytest.mark.parametrize("change", ["timing", "science", "missing", "extra", "duplicate", "missing_timing_key", "integer"])
def test_claim_agreement_preserves_roster_and_science(tmp_path, claims, change):
    baseline, actual = tmp_path / "baseline.csv", tmp_path / "actual.csv"
    if change == "integer":
        claims[0]["value"] = str(10**15)
    write_claims(baseline, claims)
    transformation = json.loads(claims[1]["value"])
    if change == "timing":
        transformation[0]["transform_time_full_s"] = 200
    elif change == "science":
        transformation[0]["recall"] = 0.94
    elif change == "missing_timing_key":
        del transformation[0]["transform_time_sample_s"]
    claims[1]["value"] = json.dumps(transformation)
    if change == "missing":
        claims.pop(0)
    elif change == "extra":
        claims.append(dict(claims[0], claim="new_claim"))
    elif change == "duplicate":
        claims.append(claims[0])
    elif change == "integer":
        claims[0]["value"] = str(10**15 + 1)
    write_claims(actual, claims)
    if change == "timing":
        assert verify_claims(baseline, actual) == 2
    else:
        with pytest.raises(ValueError):
            verify_claims(baseline, actual)


@pytest.mark.parametrize("change", ["roundoff", "distance", "membership", "activity"])
def test_curated_membership_and_science_are_checked(tmp_path, change):
    baseline, actual = tmp_path / "baseline.csv", tmp_path / "actual.csv"
    header = "JCP2022,centroid_distance,corrected_p_value\n"
    baseline.write_text(header + "JCP2022_1,0.5,0.01\n")
    row = {
        "roundoff": "JCP2022_1,0.5000001,0.01\n",
        "distance": "JCP2022_1,0.500001,0.01\n",
        "membership": "JCP2022_2,0.5,0.01\n",
        "activity": "JCP2022_1,0.5,0.02\n",
    }[change]
    actual.write_text(header + row)
    if change == "roundoff":
        verify_curated(baseline, actual)
    else:
        with pytest.raises(ValueError):
            verify_curated(baseline, actual)


@pytest.mark.parametrize("change", ["winner", "missing", "other_source"])
def test_fastest_method_is_timing_metadata(tmp_path, change):
    baseline, actual = tmp_path / "baseline.csv", tmp_path / "actual.csv"
    row = dict(claim="fastest.name", value="Popcount Normalize",
               source="data/processed/fingerprint-metrics/fingerprint_metrics_summary.json",
               rule="fingerprint_metrics", how="fingerprint_metrics: numbers.fastest.name")
    if change == "other_source":
        row["source"] = "science.json"
    write_claims(baseline, [row])
    write_claims(actual, [] if change == "missing" else [dict(row, value="Concat Popcount")])
    if change == "winner":
        assert verify_claims(baseline, actual) == 1
    else:
        with pytest.raises(ValueError):
            verify_claims(baseline, actual)


@pytest.mark.parametrize("change", ["metadata_only", "count", "missing", "retired_returned", "new_metadata"])
def test_migration_preserves_science_and_exact_claim_roster(tmp_path, change):
    baseline, actual = tmp_path / "baseline.csv", tmp_path / "actual.csv"
    rows = [
        dict(claim="count", value="112462", source="facts.json", rule="facts", how="facts: numbers.count"),
        dict(claim="retired", value="legacy", source="facts.json", rule="facts", how="facts: numbers.retired"),
        dict(claim="old_path", value="old.json", source="facts.json", rule="facts", how="facts: numbers.old_path"),
    ]
    write_claims(baseline, rows)
    policy = {
        "claim_removals": [dict(source="facts.json", rule="facts", claim="retired")],
        "claim_replacements": [dict(source="facts.json", rule="facts", claim="old_path", new_claim="path", value="public.json")],
    }
    observed = [rows[0].copy(), dict(rows[2], claim="path", value="public.json", how="facts: numbers.path")]
    if change == "count":
        observed[0]["value"] = "112461"
    elif change == "missing":
        observed.pop(0)
    elif change == "retired_returned":
        observed.append(rows[1])
    elif change == "new_metadata":
        observed[1]["value"] = "unknown.json"
    write_claims(actual, observed)
    if change == "metadata_only":
        assert verify_claims(baseline, actual, policy) == 2
    else:
        with pytest.raises(ValueError):
            verify_claims(baseline, actual, policy)


@pytest.mark.parametrize("change", ["none", "scientific_bytes", "missing", "extra", "symlink", "root_symlink", "manifest",
                                    "hydra_root", "hydra_choice", "hydra_suffix", "hydra_missing", "hydra_reference"])
def test_file_projection_keeps_exact_scientific_checks(tmp_path, change):
    from jump_production.processing.verify_baseline import (
        CURATED_TABLES,
        SCIENTIFIC_SUMMARIES,
        sha256,
        verify_files,
    )
    baseline, actual = tmp_path / "baseline", tmp_path / "actual"
    baseline.mkdir()
    (actual / "data/processed").mkdir(parents=True)
    kept, retired = "data/processed/values.csv", "data/processed/retired.txt"
    (actual / kept).write_text("count\n112462\n")
    checks = f"{sha256(actual / kept)}  {kept}\n"
    hydra = "data/processed/copairs/runs/example/results/.hydra/hydra.yaml"
    config = """hydra:
  runtime:
    cwd: /home/ubuntu/jump_production
    output_dir: /home/ubuntu/jump_production/data/processed/copairs/runs/example/results
    config_sources:
    - path: /home/ubuntu/jump_production/configs/copairs/metrics
      schema: file
      provider: command-line
    - path: copairs_runner
      schema: pkg
      provider: main
    choices:
      dataset: compound_no_source7
"""
    for root in (baseline, actual):
        (root / hydra).parent.mkdir(parents=True, exist_ok=True)
        (root / hydra).write_text(config)
    checks += f"{sha256(baseline / hydra)}  {hydra}\n"
    (baseline / "CHECKS.sha256").write_text(checks)
    (baseline / "CLAIMS.csv").write_text("fixture\n")
    for name in (*CURATED_TABLES, *SCIENTIFIC_SUMMARIES):
        (baseline / name).parent.mkdir(parents=True, exist_ok=True)
        (baseline / name).write_text("fixture\n")
        (actual / name).parent.mkdir(parents=True, exist_ok=True)
        (actual / name).write_text("fixture\n")
    records = {name: sha256(baseline / name) for name in ("CHECKS.sha256", "CLAIMS.csv", *CURATED_TABLES, *SCIENTIFIC_SUMMARIES)}
    records.update({kept: sha256(actual / kept), retired: "0" * 64, hydra: sha256(baseline / hydra)})
    (baseline / "MANIFEST.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in records.items()))
    policy = {"baseline_manifest_sha256": sha256(baseline / "MANIFEST.sha256"), "removed_outputs": [retired], "checksum_replacements": {}}
    if change == "scientific_bytes":
        (actual / kept).write_text("count\n112461\n")
    elif change == "missing":
        (actual / kept).rename(tmp_path / "saved.csv")
    elif change == "extra":
        (actual / retired).write_text("retired output returned\n")
    elif change == "symlink":
        (actual / kept).rename(tmp_path / "saved.csv")
        (actual / kept).symlink_to(tmp_path / "saved.csv")
    elif change == "root_symlink":
        (actual / "data/processed").rename(tmp_path / "saved-processed")
        (actual / "data/processed").symlink_to(tmp_path / "saved-processed", target_is_directory=True)
    elif change == "manifest":
        (baseline / "MANIFEST.sha256").write_text("changed\n")
    elif change == "hydra_root":
        (actual / hydra).write_text(config.replace("/home/ubuntu/jump_production", "/scratch/job 123/reproduction"))
    elif change == "hydra_choice":
        (actual / hydra).write_text(config.replace("compound_no_source7", "different_dataset"))
    elif change == "hydra_suffix":
        (actual / hydra).write_text(config.replace("configs/copairs/metrics", "configs/different"))
    elif change == "hydra_missing":
        (actual / hydra).write_text(config.replace("    cwd: /home/ubuntu/jump_production\n", ""))
    elif change == "hydra_reference":
        (baseline / hydra).write_text(config.replace("compound_no_source7", "different_dataset"))
    if change in ("none", "hydra_root"):
        assert verify_files(baseline, actual, policy) == (8, 2)
    else:
        with pytest.raises(ValueError):
            verify_files(baseline, actual, policy)
