"""Standardized analysis-summary envelope for the JUMP pipeline.

Every analysis node writes one `summary.json` describing what it produced, in a single shape
so the dataset-reference hub can read them all uniformly:

    {
      "description": "<one or two sentences: what this analysis is and why>",
      "numbers":     { ... the values, nested however the analysis likes ... },
      "files":       [ {"path": ..., "kind": "figure|table|data", "caption": ...}, ... ]
    }

`description` carries authored intent (better than reverse-engineering it from code); `numbers` is
whatever dict the analysis builds; `files` is an index of everything the analysis generated
(figures, tables) so the hub can render or link them. Use `index_files()` to build that index by
globbing the output directory - the notebook knows exactly what it wrote.
"""

import json
from pathlib import Path

_FIGURE_EXT = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
_TABLE_EXT = {".csv", ".tsv", ".parquet"}


def _infer_kind(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in _FIGURE_EXT:
        return "figure"
    if ext in _TABLE_EXT:
        return "table"
    return "data"


def index_files(directory: str | Path, relative_to: str | Path | None = None) -> list[dict]:
    """Index the figures and tables generated in `directory` for a summary's `files` manifest.

    Non-recursive: lists the figure/table files written directly into the analysis output dir,
    skipping the summary.json itself. Captions default to a prettified file stem.

    Paths are emitted as written by default (`directory` joined with the file name), which is what
    analyses that gather files from several directories need, since bare basenames could collide.
    Pass `relative_to` to emit paths relative to a package root instead. **Publication packages must
    do this** - `scripts/validate_figure.sh` resolves every declared path under the figure directory
    and rejects absolute or non-normalized ones, so a paper figure indexes its own output with
    `index_files(output_dir, relative_to=output_dir)`.
    """
    directory = Path(directory)
    root = Path(relative_to) if relative_to is not None else None
    keep = _FIGURE_EXT | _TABLE_EXT
    items = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in keep:
            path = p.relative_to(root).as_posix() if root is not None else str(p)
            items.append({"path": path, "kind": _infer_kind(p), "caption": p.stem.replace("_", " ")})
    return items


def write_summary(path: str | Path, *, description: str, numbers: dict, files=None, **extra) -> dict:
    """Write a standardized summary envelope to `path` and return it.

    `files` may be a list of dicts ({"path", optional "kind"/"caption"}) or bare paths; `kind` is
    inferred from the extension and `caption` defaults to the file stem. Extra keyword arguments are
    merged into the envelope for analysis-specific fields.
    """
    items = []
    for f in files or []:
        if isinstance(f, (str, Path)):
            f = {"path": str(f)}
        f = dict(f)
        f.setdefault("kind", _infer_kind(f["path"]))
        f.setdefault("caption", Path(f["path"]).stem.replace("_", " "))
        items.append({"path": str(f["path"]), "kind": f["kind"], "caption": f["caption"]})

    summary = {"description": description, "numbers": numbers, "files": items, **extra}
    Path(path).write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def write_legend(path: str | Path, *, figure_id: str, title: str, sections) -> str:
    """Write a figure directory's `legend.md` from the captions its renderers drew.

    `sections` is `[(png_file_name, section_title, caption)]` in declared order. The caption is the
    same string drawn under the figure, so the legend cannot describe something the figure does not
    show. Headings name files, never manuscript figure numbers or panel letters, which drift.
    `scripts/validate_figure.sh` re-checks both properties before anything is published.
    """
    body = "\n\n".join(
        f"## {section_title} (`{file_name}`)\n\n{caption}" for file_name, section_title, caption in sections
    )
    text = f"# {figure_id}: {title}\n\n{body}\n"
    Path(path).write_text(text)
    return text
