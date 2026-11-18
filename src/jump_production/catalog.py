"""Build and check the results catalog that the publication data browser reads.

`summary.json` envelopes describe one analysis each, in place. Nothing told a reader which analyses
exist, what they are called, or how they relate, so a browser had to walk folders and guess. The
catalog is that missing index: one `data/processed/catalog.json`, written by the workflow, listing
every analysis with an authored title and section, its envelope description, its producing rule
and script, and its declared files resolved to paths that are known to exist.

Two inputs define it. `configs/results_catalog.yaml` is the authored registry: sections, and one
entry per analysis giving its title, section and optional headline numbers. An entry marked
`optional` belongs to an opt-in target: it is cataloged when its summary exists and is never a
reason to build it. The envelopes supply everything computed. The catalog never recomputes or restates a scientific value; a headline is a
key into an envelope's `numbers`, and the build fails if the key is absent.

Every path in the catalog is relative to the catalog's own directory, so the tree can be mirrored
anywhere without rewriting it.
"""

import json
import subprocess
from pathlib import Path, PurePosixPath

import yaml

from jump_production.summaries import _infer_kind
from jump_production.workflow import matches_output

SCHEMA_VERSION = 1
PROCESSED = "data/processed"
CATALOG_PATH = f"{PROCESSED}/catalog.json"
FILE_KINDS = ("figure", "table", "data")
SCALARS = (str, int, float, bool)


def is_envelope(document) -> bool:
    """Return whether a parsed JSON document has the `{description, numbers, files}` shape."""
    return (
        isinstance(document, dict)
        and isinstance(document.get("description"), str)
        and isinstance(document.get("numbers"), dict)
        and isinstance(document.get("files"), list)
    )


def load_registry(path) -> dict:
    """Read the authored registry and reject anything the catalog could not present."""
    registry = yaml.safe_load(Path(path).read_text())
    if not isinstance(registry, dict) or registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported results-catalog registry: {path}")
    if not isinstance(registry.get("repository"), str) or not registry["repository"].startswith("https://"):
        raise ValueError("Results-catalog registry needs an https repository URL")

    sections = registry.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Results-catalog registry needs a non-empty sections list")
    section_ids = []
    for section in sections:
        if not isinstance(section, dict) or set(section) != {"id", "title", "description"}:
            raise ValueError(f"Section needs exactly id, title and description: {section!r}")
        if not all(isinstance(section[key], str) and section[key].strip() for key in section):
            raise ValueError(f"Section fields must be non-empty text: {section!r}")
        section_ids.append(section["id"])
    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Results-catalog section ids must be unique")

    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Results-catalog registry needs a non-empty entries list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Registry entry must be a mapping: {entry!r}")
        if ("summary" in entry) == ("path" in entry):
            raise ValueError(f"Registry entry needs exactly one of summary or path: {entry!r}")
        allowed = {"summary", "section", "title", "highlights", "optional"} if "summary" in entry else {
            "path", "requires", "section", "title", "description", "files",
        }
        required = {"summary", "section", "title"} if "summary" in entry else {
            "path", "section", "title", "description",
        }
        if not required <= set(entry) <= allowed:
            raise ValueError(f"Registry entry has missing or unknown fields: {entry!r}")
        for key in required | ({"requires"} & set(entry)):
            if not isinstance(entry[key], str) or not entry[key].strip():
                raise ValueError(f"Registry entry field {key} must be non-empty text: {entry!r}")
        if entry["section"] not in section_ids:
            raise ValueError(f"Registry entry names an unknown section: {entry!r}")
        if not isinstance(entry.get("optional", False), bool):
            raise ValueError(f"Registry entry field optional must be true or false: {entry!r}")
        for highlight in entry.get("highlights", []):
            if not isinstance(highlight, dict) or set(highlight) != {"key", "label"}:
                raise ValueError(f"Highlight needs exactly key and label: {highlight!r}")
        if not all(isinstance(name, str) and name for name in entry.get("files", [])):
            raise ValueError(f"Resource files must be non-empty paths: {entry!r}")
    used = {entry["section"] for entry in entries if not entry.get("optional")}
    if used != set(section_ids):
        raise ValueError(f"Sections without entries: {sorted(set(section_ids) - used)}")
    return registry


def registry_inputs(registry: dict) -> list[str]:
    """Return the files the workflow must build before the catalog can be written."""
    return [
        entry["summary"] if "summary" in entry else entry.get("requires", entry["path"])
        for entry in registry["entries"]
        if not entry.get("optional")
    ]


def optional_summaries(registry: dict) -> list[str]:
    """Return summaries of opt-in analyses: cataloged when they were built, never required."""
    return [entry["summary"] for entry in registry["entries"] if entry.get("optional")]


def _inside_processed(path: str) -> PurePosixPath:
    """Return a checkout-relative `data/processed/...` path, rejecting anything outside it."""
    location = PurePosixPath(path)
    if (
        location.is_absolute()
        or location.as_posix() != path.rstrip("/")
        or ".." in location.parts
        or location.parts[:2] != ("data", "processed")
        or len(location.parts) < 3
    ):
        raise ValueError(f"Catalog path must be a normalized path under {PROCESSED}/: {path}")
    return location


def _catalog_relative(location: PurePosixPath) -> str:
    return PurePosixPath(*location.parts[2:]).as_posix()


def _regular_file(root: Path, location: PurePosixPath) -> bool:
    """Return whether a checkout-relative path is a regular file reached without symlinks."""
    cursor = root
    for part in location.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return False
    return cursor.is_file()


def _declared_file(root: Path, summary: PurePosixPath, raw_path) -> PurePosixPath:
    """Resolve one envelope `files` path to a checkout-relative location.

    Envelopes use two conventions: `data/...` relative to the checkout, and bare paths relative to
    the envelope's own directory. An absolute path is accepted only when it points into this
    checkout, which is what envelopes written before paths were made portable contain.
    """
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"Invalid declared path in {summary}: {raw_path!r}")
    if raw_path.startswith("/"):
        try:
            relative = Path(raw_path).relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"{summary} declares a path on another machine; rebuild the analysis: {raw_path}"
            ) from error
    elif raw_path.startswith("data/"):
        relative = raw_path
    else:
        relative = (summary.parent / raw_path).as_posix()
    return _inside_processed(relative)


def _lookup(numbers: dict, dotted_key: str):
    value = numbers
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def _count_leaves(value) -> int:
    if isinstance(value, dict):
        return sum(_count_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_leaves(item) for item in value)
    return 1


def _producer(declarations: list[dict], path: str) -> dict | None:
    """Return the single declaration producing `path`, or None when there is none."""
    producers = {
        (declaration["rule"], declaration.get("script"))
        for declaration in declarations
        if matches_output(str(declaration["output"]), path)
    }
    if len(producers) > 1:
        raise ValueError(f"Expected at most one producer for {path}, found {sorted(producers, key=str)}")
    if not producers:
        return None
    rule, script = producers.pop()
    return {"rule": rule, "script": script}


def _summary_entry(root: Path, entry: dict, declarations: list[dict]) -> dict:
    summary = _inside_processed(entry["summary"])
    if summary.suffix != ".json" or not _regular_file(root, summary):
        raise ValueError(f"Registered summary is not a regular JSON file: {summary}")
    document = json.loads((root / summary).read_text())
    if not is_envelope(document) or not document["description"].strip():
        raise ValueError(f"Registered summary is not a described envelope: {summary}")
    producer = _producer(declarations, summary.as_posix())
    if producer is None:
        raise ValueError(f"No workflow rule declares the registered summary: {summary}")

    files, seen = [], set()
    for record in document["files"]:
        if not isinstance(record, dict) or record.get("kind") not in FILE_KINDS:
            raise ValueError(f"Invalid file record in {summary}: {record!r}")
        if not isinstance(record.get("caption"), str) or not record["caption"].strip():
            raise ValueError(f"File record without a caption in {summary}: {record!r}")
        location = _declared_file(root, summary, record.get("path"))
        if location == summary:
            continue  # an envelope listing itself adds nothing a reader can open
        if location in seen:
            raise ValueError(f"{summary} declares a file twice: {location}")
        if not _regular_file(root, location):
            raise ValueError(f"{summary} declares a file that was not written: {location}")
        seen.add(location)
        files.append({"path": _catalog_relative(location), "kind": record["kind"], "caption": record["caption"]})

    highlights = []
    for highlight in entry.get("highlights", []):
        try:
            value = _lookup(document["numbers"], highlight["key"])
        except KeyError as error:
            raise ValueError(f"{summary} has no number {highlight['key']!r} for its headline") from error
        if not isinstance(value, SCALARS):
            raise ValueError(f"Headline {highlight['key']!r} of {summary} is not a single value")
        highlights.append({"key": highlight["key"], "label": highlight["label"], "value": value})

    return {
        "id": _catalog_relative(summary.parent),
        "section": entry["section"],
        "title": entry["title"],
        "description": document["description"],
        "path": _catalog_relative(summary.parent) + "/",
        "summary": _catalog_relative(summary),
        **producer,
        "numbers": _count_leaves(document["numbers"]),
        "highlights": highlights,
        "files": files,
    }


def _resource_entry(root: Path, entry: dict, declarations: list[dict]) -> dict:
    """Describe results that have no envelope: a database, or a directory of per-run outputs."""
    is_directory = entry["path"].endswith("/")
    location = _inside_processed(entry["path"])
    anchor = _inside_processed(entry.get("requires", entry["path"]))
    if not _regular_file(root, anchor):
        raise ValueError(f"Registered resource was not built: {anchor}")
    target = root / location
    if is_directory:
        if target.is_symlink() or not target.is_dir():
            raise ValueError(f"Registered resource is not a directory: {location}")
        if anchor.parts[: len(location.parts)] != location.parts:
            raise ValueError(f"Resource {location} must contain the file that requires it: {anchor}")
    elif anchor != location:
        raise ValueError(f"A file resource is its own requirement: {location}")

    if entry.get("files") and not is_directory:
        raise ValueError(f"Only a directory resource lists member files: {location}")
    files = []
    for name in entry.get("files", []):
        member = _inside_processed(f"{location.as_posix()}/{name}")
        if not _regular_file(root, member):
            raise ValueError(f"Registered resource file was not written: {member}")
        files.append({"path": _catalog_relative(member), "kind": _infer_kind(name), "caption": PurePosixPath(name).name})
    if not is_directory:
        files.append({"path": _catalog_relative(location), "kind": _infer_kind(location), "caption": location.name})

    relative = _catalog_relative(location)
    parent = _catalog_relative(location.parent) if len(location.parts) > 3 else ""
    return {
        "id": relative,
        "section": entry["section"],
        "title": entry["title"],
        "description": " ".join(entry["description"].split()),
        "path": f"{relative}/" if is_directory else (f"{parent}/" if parent else ""),
        "summary": None,
        **(_producer(declarations, anchor.as_posix()) or {"rule": None, "script": None}),
        "numbers": 0,
        "highlights": [],
        "files": files,
    }


def build_catalog(registry: dict, *, root, declarations: list[dict], canonical_key: dict, revision: str | None) -> dict:
    """Join the authored registry with the envelopes on disk into the catalog document."""
    root = Path(root).resolve()
    entries = [
        (_summary_entry if "summary" in entry else _resource_entry)(root, entry, declarations)
        for entry in registry["entries"]
        # An opt-in analysis that was not built is simply absent from the catalog.
        if not (entry.get("optional") and not (root / entry["summary"]).exists())
    ]
    identifiers = [entry["id"] for entry in entries]
    duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicates:
        raise ValueError(f"Catalog entries share an id: {duplicates}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"repository": registry["repository"], "revision": revision},
        "canonical_key": dict(canonical_key),
        "sections": [dict(section) for section in registry["sections"]],
        "entries": entries,
    }


def source_revision(root) -> str | None:
    """Return the commit that produced the results, or None when that cannot be stated.

    A checkout with uncommitted changes to tracked files did not run any one commit, so it reports
    no revision rather than a misleading one.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return revision if len(revision) == 40 and not dirty else None


def write_catalog(output_path, registry_path, *, root, declarations, canonical_key, revision) -> dict:
    """Write the catalog document and return it."""
    catalog = build_catalog(
        load_registry(registry_path),
        root=root,
        declarations=declarations,
        canonical_key=canonical_key,
        revision=revision,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def validate_catalog(root) -> tuple[int, int]:
    """Check a written catalog against the output tree it describes.

    Every cataloged summary, directory and file must exist, and every envelope in the tree must be
    cataloged, so an analysis cannot be published without a title and section. Returns the number
    of entries and of cataloged files.
    """
    root = Path(root)
    processed = root / PROCESSED
    catalog = json.loads((root / CATALOG_PATH).read_text())
    if not isinstance(catalog, dict) or catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported results catalog")
    sections = [section.get("id") for section in catalog.get("sections", [])]
    entries = catalog.get("entries")
    if not sections or not isinstance(entries, list) or not entries:
        raise ValueError("Results catalog has no sections or entries")

    cataloged, file_count = set(), 0
    for entry in entries:
        if entry.get("section") not in sections or not str(entry.get("title", "")).strip():
            raise ValueError(f"Catalog entry lacks a known section or a title: {entry.get('id')}")
        if not (processed / entry["path"]).is_dir():
            raise ValueError(f"Catalog entry directory is missing: {entry['path']}")
        if entry["summary"] is not None:
            if not is_envelope(json.loads((processed / entry["summary"]).read_text())):
                raise ValueError(f"Cataloged summary is not an envelope: {entry['summary']}")
            cataloged.add(entry["summary"])
        for record in entry["files"]:
            location = _inside_processed(f"{PROCESSED}/{record['path']}")
            if not _regular_file(root, location):
                raise ValueError(f"Cataloged file is missing: {record['path']}")
            file_count += 1

    envelopes = set()
    for path in processed.rglob("*.json"):
        if path.is_file() and not path.is_symlink():
            try:
                document = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if is_envelope(document):
                envelopes.add(path.relative_to(processed).as_posix())
    if envelopes != cataloged:
        raise ValueError(
            f"Catalog and summary envelopes differ: uncataloged={sorted(envelopes - cataloged)}, "
            f"missing={sorted(cataloged - envelopes)}"
        )
    return len(entries), file_count
