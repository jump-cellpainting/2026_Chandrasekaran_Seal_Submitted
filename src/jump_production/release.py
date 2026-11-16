"""Check the exact reviewed source tree before a fresh-history release."""

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import typer

ALLOWLIST = "release-files.json"


def check_release(root: Path) -> int:
    root = root.resolve()
    document = json.loads((root / ALLOWLIST).read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("files"), dict):
        raise ValueError("Invalid release allowlist")
    expected = document["files"]
    for name, digest in expected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or str(path) != name or ".." in path.parts or ".git" in path.parts:
            raise ValueError(f"Unsafe release path: {name}")
        if name == ALLOWLIST or not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Invalid release digest: {name}")
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).decode()
    entries = {}
    for entry in raw.split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, oid, stage = metadata.split()
        if stage != "0" or name in entries or mode not in {"100644", "100755"}:
            raise ValueError(f"Release requires regular, merged files: {name}")
        entries[name] = oid
    if set(entries) != set(expected) | {ALLOWLIST}:
        raise ValueError(
            f"Release roster differs: extra={sorted(set(entries) - set(expected) - {ALLOWLIST})}, "
            f"missing={sorted((set(expected) | {ALLOWLIST}) - set(entries))}"
        )
    for name in entries:
        path = root / name
        if any(parent.is_symlink() for parent in [path, *path.parents]) or not path.is_file():
            raise ValueError(f"Release file missing or traverses symlink: {name}")
        indexed = subprocess.check_output(["git", "show", f":{name}"], cwd=root)
        if path.read_bytes() != indexed:
            raise ValueError(f"Unstaged release change: {name}")
        if name != ALLOWLIST and hashlib.sha256(indexed).hexdigest() != expected[name]:
            raise ValueError(f"Unreviewed content: {name}")
    return len(entries)


def main(root: Path = Path(".")) -> None:
    try:
        count = check_release(root)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"release: ok ({count} reviewed source files; index matches working tree)")


if __name__ == "__main__":
    typer.run(main)
