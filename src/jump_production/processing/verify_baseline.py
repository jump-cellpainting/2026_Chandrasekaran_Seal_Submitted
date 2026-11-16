"""Compare the complete claim roster with a pinned cached-run baseline."""

import csv
import hashlib
import json
import math
from pathlib import Path

import typer
from loguru import logger

CURATED_TABLES = tuple(
    f"data/processed/commercial-compounds/curated_compounds_{size}.csv"
    for size in ("160c_1pc", "160c_2pc", "320c_1pc", "320c_2pc")
)
SCIENTIFIC_SUMMARIES = (
    "data/processed/batch-effect/summary.json",
    "data/processed/mitotox-analysis/compound_no_source7/summary.json",
)


def load_claims(path: Path) -> dict:
    claims = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != 5 or set(reader.fieldnames or []) != {"claim", "value", "source", "rule", "how"}:
            raise ValueError(f"Unexpected claim columns: {path}")
        for row in reader:
            key = (row["source"], row["rule"], row["claim"])
            if key in claims:
                raise ValueError(f"Duplicate claim: {key}")
            try:
                value = json.loads(row["value"])
            except json.JSONDecodeError:
                value = row["value"]
            if key[:2] == (
                "data/processed/fingerprint-metrics/fingerprint_metrics_summary.json",
                "fingerprint_metrics",
            ):
                if key[2] == "fastest.transform_time_full_s":
                    value = None
                elif key[2] == "transformations":
                    for transformation in value:
                        for timing in ("transform_time_full_s", "transform_time_sample_s"):
                            if timing in transformation:
                                transformation[timing] = None
            claims[key] = (row["how"], value)
    return claims


def equal_values(left, right) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        if type(left) is int or type(right) is int:
            return left == right
        # Accepted float64 roundoff; no absolute tolerance for tiny p-values.
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=0)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_values(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_values(a, b) for a, b in zip(left, right))
    return left == right


def verify_claims(baseline: Path, actual: Path, policy: dict | None = None) -> int:
    expected, observed = load_claims(baseline), load_claims(actual)
    if policy is not None:
        for change in policy["claim_removals"]:
            key = (change["source"], change["rule"], change["claim"])
            if key not in expected:
                raise ValueError(f"Retired claim absent from baseline: {key}")
            del expected[key]
        for change in policy["claim_replacements"]:
            key = (change["source"], change["rule"], change["claim"])
            if key not in expected:
                raise ValueError(f"Changed metadata absent from baseline: {key}")
            how, _ = expected.pop(key)
            new_claim = change.get("new_claim", key[2])
            new_key = (key[0], key[1], new_claim)
            if new_key in expected:
                raise ValueError(f"Duplicate migrated claim: {new_key}")
            if new_claim != key[2]:
                how = f"{key[1]}: numbers.{new_claim}"
            expected[new_key] = (how, change["value"])
    if expected.keys() != observed.keys():
        raise ValueError(
            f"Claim roster changed: missing={sorted(expected.keys() - observed.keys())}, "
            f"extra={sorted(observed.keys() - expected.keys())}"
        )
    changed = [key for key in expected if not equal_values(expected[key], observed[key])]
    if changed:
        raise ValueError(f"Unexpected claim changes: {changed}")
    return len(expected)


def verify_curated(baseline: Path, actual: Path) -> None:
    def load(path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            keyed = {row["JCP2022"]: row for row in rows}
            if len(keyed) != len(rows):
                raise ValueError(f"Duplicate curated compound: {path}")
            return reader.fieldnames, keyed

    columns, expected = load(baseline)
    actual_columns, observed = load(actual)
    if columns != actual_columns or expected.keys() != observed.keys():
        raise ValueError(f"Curated compound roster/columns changed: {actual}")
    for compound, row in expected.items():
        for column, value in row.items():
            other = observed[compound][column]
            agrees = (
                math.isclose(float(value), float(other), rel_tol=0, abs_tol=2e-7)
                if column == "centroid_distance" else value == other
            )
            if not agrees:
                raise ValueError(f"Curated compound changed: {actual}, {compound}, {column}")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line[:64], line[66:]
        if len(line) < 67 or line[64:66] != "  " or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid checksum line: {path}")
        if name in result or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Duplicate or unsafe checksum path: {name}")
        result[name] = digest
    return result


def verify_files(baseline: Path, actual: Path, policy: dict) -> tuple[int, int]:
    manifest_path = baseline / "MANIFEST.sha256"
    if sha256(manifest_path) != policy["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest checksum changed")
    manifest = read_checksums(manifest_path)
    for name in ("CHECKS.sha256", "CLAIMS.csv", *CURATED_TABLES, *SCIENTIFIC_SUMMARIES):
        if sha256(baseline / name) != manifest[name]:
            raise ValueError(f"Unauthenticated baseline input: {name}")
    expected = {p for p in manifest if p.startswith("data/processed/")}
    removed = set(policy["removed_outputs"])
    if not removed <= expected or len(removed) != len(policy["removed_outputs"]):
        raise ValueError("Invalid retired output roster")
    expected -= removed
    observed = set()
    output_root = actual / "data/processed"
    if output_root.is_symlink() or (actual / "data").is_symlink():
        raise ValueError(f"Symlink in output tree: {output_root}")
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in output tree: {path}")
        if path.is_file():
            observed.add(path.relative_to(actual).as_posix())
    if expected != observed:
        raise ValueError(f"Output roster changed: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    checks = read_checksums(baseline / "CHECKS.sha256")
    if not set(checks) <= set(manifest):
        raise ValueError("Checks contain a path outside the authenticated manifest")
    for name, digest in checks.items():
        if digest != manifest[name]:
            raise ValueError(f"Check differs from authenticated manifest: {name}")
    replacements = policy["checksum_replacements"]
    if not set(replacements) <= set(checks) - removed:
        raise ValueError("Replacement checksum does not name a retained exact check")
    checks = {name: replacements.get(name, digest) for name, digest in checks.items() if name not in removed}
    for name, digest in checks.items():
        if sha256(actual / name) != digest:
            raise ValueError(f"Artifact checksum changed: {name}")
    return len(expected), len(checks)


def main(baseline: Path, actual: Path) -> None:
    policy = json.loads((Path(__file__).resolve().parents[3] / "configs/scientific_baseline.json").read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported scientific baseline policy")
    try:
        file_count, exact_count = verify_files(baseline, actual, policy)
        count = verify_claims(baseline / "CLAIMS.csv", actual / "data/processed/dataset_reference/dataset_reference.csv", policy)
        for table in CURATED_TABLES:
            verify_curated(baseline / table, actual / table)
        for summary in SCIENTIFIC_SUMMARIES:
            expected = json.loads((baseline / summary).read_text())["numbers"]
            observed = json.loads((actual / summary).read_text())["numbers"]
            if not equal_values(expected, observed):
                raise ValueError(f"Scientific summary changed: {summary}")
    except (ValueError, OSError, KeyError) as error:
        logger.error(str(error))
        raise typer.Exit(1) from error
    logger.success("{} output paths and {} exact artifact checks agree", file_count, exact_count)
    logger.success("{} claims agree; only named fingerprint timings and float64 roundoff allowed", count)
    logger.success("Four curated tables agree; only centroid_distance allows absolute drift <= 2e-7")
    logger.success("Batch and MitoTox scientific summaries agree")


if __name__ == "__main__":
    typer.run(main)
