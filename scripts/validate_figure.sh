#!/usr/bin/env bash

set -euo pipefail

die() {
    echo "validate-figure: $*" >&2
    exit 1
}

[[ $# -eq 1 ]] || die "usage: validate_figure.sh FIGURE_ID"
figure_id="$1"
[[ "$figure_id" =~ ^figure[A-Za-z0-9_]+$ ]] || \
    die "unsafe figure id '$figure_id'; expected a path-safe ID beginning with figure"

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

: "${PROCESSED_DIR:=data/processed}"
entry_dir="$PROCESSED_DIR/figures/$figure_id"
summary_path="$entry_dir/summary.json"

[[ -d "$entry_dir" && ! -L "$entry_dir" ]] || \
    die "missing regular figure directory: $entry_dir"
[[ -f "$summary_path" && ! -L "$summary_path" ]] || \
    die "missing regular summary: $summary_path"

declared_count=$(
python3 - "$entry_dir" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

entry_dir = Path(sys.argv[1])
summary_path = entry_dir / "summary.json"
reserved_directories = {"_marimo", "_proto"}
legacy_files = {"figure.json", "index.html"}

try:
    summary = json.loads(summary_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid summary {summary_path}: {error}") from error

if not isinstance(summary, dict):
    raise SystemExit(f"summary must be an object: {summary_path}")
description = summary.get("description")
numbers = summary.get("numbers")
files = summary.get("files")
if not isinstance(description, str) or not description.strip():
    raise SystemExit(f"summary description must be non-empty text: {summary_path}")
if not isinstance(numbers, dict):
    raise SystemExit(f"summary numbers must be an object: {summary_path}")
if not isinstance(files, list) or not files:
    raise SystemExit(f"summary files must be a non-empty list: {summary_path}")

declared = set()
for index, record in enumerate(files):
    if not isinstance(record, dict):
        raise SystemExit(f"summary file record {index} must be an object")
    raw_path = record.get("path")
    kind = record.get("kind")
    caption = record.get("caption")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"summary file record {index} has an invalid path")
    if kind not in {"figure", "table", "data"}:
        raise SystemExit(f"summary file record {index} has an invalid kind")
    if not isinstance(caption, str) or not caption.strip():
        raise SystemExit(f"summary file record {index} has an invalid caption")
    if "\\" in raw_path or any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise SystemExit(f"unsafe declared path: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[0] in reserved_directories
        or raw_path in legacy_files | {"summary.json"}
    ):
        raise SystemExit(f"unsafe or non-normalized declared path: {raw_path!r}")
    if raw_path in declared:
        raise SystemExit(f"duplicate declared path: {raw_path!r}")

    artifact = entry_dir.joinpath(*relative.parts)
    cursor = entry_dir
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise SystemExit(f"declared artifact traverses a symlink: {raw_path!r}")
    try:
        artifact_stat = artifact.lstat()
    except FileNotFoundError as error:
        raise SystemExit(f"missing declared artifact: {artifact}") from error
    if not stat.S_ISREG(artifact_stat.st_mode):
        raise SystemExit(f"declared artifact is not a regular file: {artifact}")
    if artifact_stat.st_size == 0:
        raise SystemExit(f"declared artifact is empty: {artifact}")
    declared.add(raw_path)

actual = {"summary.json"}
for root, directory_names, file_names in os.walk(entry_dir, topdown=True, followlinks=False):
    root_path = Path(root)
    for directory_name in directory_names:
        directory = root_path / directory_name
        if directory.is_symlink():
            raise SystemExit(f"undeclared symlink directory: {directory}")
    for file_name in file_names:
        artifact = root_path / file_name
        if artifact.is_symlink():
            raise SystemExit(f"undeclared symlink file: {artifact}")
        if not stat.S_ISREG(artifact.lstat().st_mode):
            raise SystemExit(f"undeclared non-regular path: {artifact}")
        actual.add(artifact.relative_to(entry_dir).as_posix())

allowed = declared | {"summary.json"}
undeclared = sorted(actual - allowed)
if undeclared:
    raise SystemExit(f"undeclared files in figure directory: {undeclared}")
missing = sorted(allowed - actual)
if missing:
    raise SystemExit(f"declared files missing from figure directory: {missing}")

# One caption per rendered figure, drawn under it and reused verbatim as its legend section, so a
# legend cannot describe something the figure stopped showing. Sections name files because
# manuscript figure numbers drift.
MANUSCRIPT_NUMBER = re.compile(r"\b(?:Supplementary Figure|Figure \d)")
# Manuscript panel letters drift the same way when a figure is rearranged: "panel C", a letter
# range like "A-E", and a sentence opening on a bare letter, as in "C, clinical phase:".
PANEL_LETTER = re.compile(r"\bpanels? [A-H]\b|\b[A-H]-[A-H]\b|(?:^|(?<=[.:;] ))[A-H], ")
rendered = sorted(
    record["path"] for record in files if record["kind"] == "figure" and record["path"].endswith(".png")
)
if rendered:
    if "legend.md" not in declared:
        raise SystemExit(f"figure directory declares rendered figures but no legend.md: {entry_dir}")
    legend_text = (entry_dir / "legend.md").read_text()
    numbered = [line for line in legend_text.splitlines() if MANUSCRIPT_NUMBER.search(line)]
    if numbered:
        raise SystemExit(f"legend.md names manuscript figure numbers, which drift: {numbered}")
    lettered = [line for line in legend_text.splitlines() if PANEL_LETTER.search(line)]
    if lettered:
        raise SystemExit(f"legend.md names manuscript panel letters, which drift; key panels by position: {lettered}")
    headings = re.findall(r"^## .*$", legend_text, flags=re.MULTILINE)
    sections = re.findall(r"^## .*\(`([^`]+)`\)\s*$", legend_text, flags=re.MULTILINE)
    if len(sections) != len(headings):
        raise SystemExit(f"every legend.md section must name its file in backticks: {entry_dir}")
    if sorted(sections) != rendered:
        raise SystemExit(f"legend.md sections {sorted(sections)} do not match the rendered figures {rendered}")

print(len(declared))
PY
)

echo "Validated $figure_id: $declared_count declared files plus summary.json"
