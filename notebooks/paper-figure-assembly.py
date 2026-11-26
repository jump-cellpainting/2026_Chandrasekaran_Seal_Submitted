#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.4.6",
#     "pillow==12.3.0",
#     "pymupdf==2.2",
#     "pyyaml==6.0.3",
#     "typer==0.21.2",
# ]
# ///
"""Assemble one paper figure from pipeline panels, as described by a layout file.

A layout (`configs/paper_figures/<figure_id>.yaml`) lists rows of panels. Each panel names a source
file relative to the checkout, an optional clip box in page fractions, and a final width in inches.
A row entry that is itself a list is a vertical stack. Panel letters are assigned in reading order.

Vector sources stay vector: the PDF is assembled with PyMuPDF and the PNG is rasterised from that
same PDF, so the two cannot disagree. PyMuPDF is fetched by `uv` for this script and is not part of
the pinned Pixi environments, which is why the script keeps no import from `jump_production`.

The package it writes is a standard figure directory: `assembled.png`, `assembled.pdf`, `legend.md`
and a `summary.json` envelope. The legend is built from the layout and from the captions that the
source renderers computed, keyed by position; it names no manuscript figure number or panel letter.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pymupdf
import typer
import yaml
from PIL import Image

PT = 72.0
LETTER_PT = 11.0
TITLE_PT = 8.5
FONT_BOLD = "hebo"  # Helvetica-Bold, a PDF base-14 font
RASTER_SOURCE_DPI = 300.0
ORIGINS = {
    "pipeline": "pipeline output",
    "static": "static input that the pipeline cannot regenerate",
}
PANEL_KEYS = {"src", "title", "clip", "width", "letter", "label", "mask", "origin", "note"}
FIGURE_KEYS = {"id", "title", "width", "justify", "rows"}

app = typer.Typer(add_completion=False, help=__doc__.splitlines()[0])


def load_layout(path: Path) -> dict:
    """Read a layout and normalise every row entry to a list of panels (a stack of one or more)."""
    layout = yaml.safe_load(path.read_text())
    if not isinstance(layout, dict) or set(layout) - FIGURE_KEYS or not {"id", "title", "rows"} <= set(layout):
        raise ValueError(f"{path}: a layout needs id, title and rows, and allows only {sorted(FIGURE_KEYS)}")
    if layout["id"] != path.stem:
        raise ValueError(f"{path}: id {layout['id']!r} must equal the file name")
    rows = []
    for row in layout["rows"]:
        cells = []
        for cell in row:
            stack = cell if isinstance(cell, list) else [cell]
            for panel in stack:
                if not isinstance(panel, dict) or set(panel) - PANEL_KEYS or "src" not in panel or "width" not in panel:
                    raise ValueError(
                        f"{path}: a panel needs src and width, and allows only {sorted(PANEL_KEYS)}: {panel}"
                    )
                panel.setdefault("origin", "pipeline")
                if panel["origin"] not in ORIGINS:
                    raise ValueError(f"{path}: origin must be one of {sorted(ORIGINS)}: {panel['src']}")
            cells.append(stack)
        rows.append(cells)
    layout["rows"] = rows
    return layout


def panels_of(layout: dict):
    """Yield (row, column, stack position, stack size, panel) in reading order."""
    for r, row in enumerate(layout["rows"]):
        for c, stack in enumerate(row):
            for s, panel in enumerate(stack):
                yield r, c, s, len(stack), panel


def layout_problems(layout: dict, root: Path) -> list[str]:
    """Everything that would make the assembly wrong or impossible, without drawing anything."""
    problems = []
    for _, _, _, _, panel in panels_of(layout):
        src = panel["src"]
        if Path(src).is_absolute() or ".." in Path(src).parts:
            problems.append(f"source must be relative to the checkout: {src}")
            continue
        if not (root / src).is_file():
            problems.append(f"missing panel source: {src}")
        if not panel["width"] > 0:
            problems.append(f"width must be positive: {src}")
        for name, box in [("clip", panel.get("clip"))] + [("mask", m) for m in panel.get("mask") or []]:
            if box is None:
                continue
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                problems.append(f"{name} box must lie inside the page as [x0, y0, x1, y1]: {src} {box}")
    return problems


def position(r: int, c: int, s: int, layout: dict) -> str:
    """Where a panel sits, in words: letters drift when a figure is rearranged, positions are read off it."""
    row, stack = layout["rows"][r], layout["rows"][r][c]

    def nth(index, total, names):
        if total == 1:
            return ""
        if total <= 3:
            return names[total][index]
        return f"{names['item']} {index + 1}"

    row_names = {2: ["Top row", "Bottom row"], 3: ["Top row", "Middle row", "Bottom row"], "item": "Row"}
    column_names = {2: ["left", "right"], 3: ["left", "centre", "right"], "item": "position"}
    stack_names = {2: ["upper", "lower"], 3: ["upper", "middle", "lower"], "item": "stacked"}
    parts = [
        nth(r, len(layout["rows"]), row_names),
        nth(c, len(row), column_names),
        nth(s, len(stack), stack_names),
    ]
    text = ", ".join(part for part in parts if part)
    return (text[0].upper() + text[1:]) if text else "Whole figure"


def source_caption(root: Path, src: str) -> str:
    """The caption the source renderer computed for this file, from the envelope beside it."""
    summary = root / Path(src).parent / "summary.json"
    if not summary.is_file():
        return ""
    try:
        files = json.loads(summary.read_text()).get("files", [])
    except json.JSONDecodeError:
        return ""
    stem = Path(src).stem
    for record in files:
        if isinstance(record, dict) and Path(str(record.get("path", ""))).stem == stem and record.get("caption"):
            return " ".join(str(record["caption"]).split())
    return ""


def build_caption(layout: dict, root: Path) -> str:
    sentences = []
    for r, c, s, _, panel in panels_of(layout):
        where = position(r, c, s, layout)
        clause = f"{where}: " + (f"{panel['title']}; " if panel.get("title") else "")
        clause += f"from `{panel['src']}`" + (", clipped" if panel.get("clip") else "")
        clause += f" ({ORIGINS[panel['origin']]})."
        caption = source_caption(root, panel["src"])
        if caption:
            clause += f" Source caption: {caption}" + ("" if caption.endswith(".") else ".")
        sentences.append(clause)
    return " ".join(sentences)


def open_source(path: Path) -> tuple[pymupdf.Document, bool]:
    if path.suffix.lower() == ".pdf":
        return pymupdf.open(path), True
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image)
        image = white
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page(width=image.width / RASTER_SOURCE_DPI * PT, height=image.height / RASTER_SOURCE_DPI * PT)
    page.insert_image(page.rect, stream=buffer.getvalue())
    return document, False


def fraction_box(page_rect: pymupdf.Rect, box) -> pymupdf.Rect:
    x0, y0, x1, y1 = box
    return pymupdf.Rect(
        page_rect.x0 + x0 * page_rect.width,
        page_rect.y0 + y0 * page_rect.height,
        page_rect.x0 + x1 * page_rect.width,
        page_rect.y0 + y1 * page_rect.height,
    )


def ink_box(document: pymupdf.Document, box: pymupdf.Rect, pad: float = 2.0) -> pymupdf.Rect:
    """Bounding box of non-white ink inside `box`, in page points."""
    zoom = 2.0
    pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=box, alpha=False)
    data = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    ink = (data[:, :, :3] < 247).any(axis=2)
    if not ink.any():
        return box
    ys, xs = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    found = pymupdf.Rect(
        box.x0 + xs[0] / zoom - pad,
        box.y0 + ys[0] / zoom - pad,
        box.x0 + (xs[-1] + 1) / zoom + pad,
        box.y0 + (ys[-1] + 1) / zoom + pad,
    )
    return found & box


def assemble(layout: dict, root: Path, out_dir: Path, dpi: int = 300) -> dict:
    """Draw the figure and return what was placed."""
    problems = layout_problems(layout, root)
    if problems:
        raise ValueError(f"{layout['id']}: " + "; ".join(problems))
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    placed = {}
    for r, c, s, _, panel in panels_of(layout):
        document, is_vector = open_source(root / panel["src"])
        page_rect = document[0].rect
        for box in panel.get("mask") or []:
            document[0].draw_rect(fraction_box(page_rect, box), color=(1, 1, 1), fill=(1, 1, 1), width=0)
        box = ink_box(document, fraction_box(page_rect, panel["clip"]) if panel.get("clip") else page_rect)
        factor = panel["width"] * PT / box.width
        if "label" in panel:
            letter = str(panel["label"])
        elif panel.get("letter", True):
            letter = next(letters)
        else:
            letter = ""
        placed[(r, c, s)] = {
            "document": document,
            "vector": is_vector,
            "box": box,
            "size": (box.width * factor, box.height * factor),
            "letter": letter,
        }

    margin, hgap, vgap, title_gap, stack_gap = 0.08 * PT, 0.16 * PT, 0.16 * PT, 0.03 * PT, 0.07 * PT

    def head(key, panel) -> float:
        return (LETTER_PT + title_gap) if (placed[key]["letter"] or panel.get("title")) else 0.0

    def cell_size(r, c):
        stack = layout["rows"][r][c]
        width = max(placed[(r, c, s)]["size"][0] for s in range(len(stack)))
        height = sum(placed[(r, c, s)]["size"][1] + head((r, c, s), panel) for s, panel in enumerate(stack))
        return width, height + stack_gap * (len(stack) - 1)

    rows = layout["rows"]
    row_widths = [
        sum(cell_size(r, c)[0] for c in range(len(row))) + hgap * (len(row) - 1) for r, row in enumerate(rows)
    ]
    row_heights = [max(cell_size(r, c)[1] for c in range(len(row))) for r, row in enumerate(rows)]
    inner_width = layout["width"] * PT - 2 * margin if layout.get("width") else max(row_widths)
    overflow = [r for r, width in enumerate(row_widths) if width > inner_width + 0.5]
    if overflow:
        raise ValueError(f"{layout['id']}: rows {[r + 1 for r in overflow]} are wider than the figure")

    output = pymupdf.open()
    page = output.new_page(
        width=inner_width + 2 * margin, height=sum(row_heights) + vgap * (len(rows) - 1) + 2 * margin
    )
    records = []
    y = margin
    for r, row in enumerate(rows):
        gap = hgap
        if layout.get("justify") and len(row) > 1 and row_widths[r] < inner_width:
            gap = hgap + (inner_width - row_widths[r]) / (len(row) - 1)
        x = margin
        for c, stack in enumerate(row):
            cell_width, _ = cell_size(r, c)
            cy = y
            for s, panel in enumerate(stack):
                item = placed[(r, c, s)]
                width, height = item["size"]
                top = cy + head((r, c, s), panel)
                page.show_pdf_page(pymupdf.Rect(x, top, x + width, top + height), item["document"], 0, clip=item["box"])
                baseline, text_x = cy + LETTER_PT * 0.82, x
                if item["letter"]:
                    page.insert_text((text_x, baseline), item["letter"], fontname=FONT_BOLD, fontsize=LETTER_PT)
                    text_x += pymupdf.get_text_length(item["letter"] + " ", fontname=FONT_BOLD, fontsize=LETTER_PT)
                if panel.get("title"):
                    page.insert_text((text_x, baseline), panel["title"], fontname=FONT_BOLD, fontsize=TITLE_PT)
                    title_width = pymupdf.get_text_length(panel["title"], fontname=FONT_BOLD, fontsize=TITLE_PT)
                    if text_x + title_width > x + cell_width + gap - 2:
                        raise ValueError(f"{layout['id']}: title wider than its panel: {panel['title']!r}")
                records.append(
                    {
                        "position": position(r, c, s, layout),
                        "drawn_letter": item["letter"],
                        "title": panel.get("title", ""),
                        "source": panel["src"],
                        "origin": panel["origin"],
                        "vector": item["vector"],
                        "clip": panel.get("clip"),
                        "width_in": round(width / PT, 3),
                        "height_in": round(height / PT, 3),
                        "note": panel.get("note", ""),
                    }
                )
                cy = top + height + stack_gap
            x += cell_width + gap
        y += row_heights[r] + vgap

    out_dir.mkdir(parents=True, exist_ok=True)
    output.save(out_dir / "assembled.pdf", deflate=True, garbage=3)
    zoom = dpi / PT
    output[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).save(out_dir / "assembled.png")
    return {"width_in": round(page.rect.width / PT, 2), "height_in": round(page.rect.height / PT, 2), "panels": records}


@app.command()
def compose(
    layout_path: Path = typer.Argument(..., help="Layout file, configs/paper_figures/<figure_id>.yaml"),
    output_root: Path = typer.Option(
        Path("data/paper_figures/figures"), help="Directory that holds the figure packages"
    ),
    root: Path = typer.Option(Path("."), help="Checkout root that layout sources are relative to"),
    dpi: int = typer.Option(300, help="Resolution of the PNG"),
) -> None:
    """Write one figure package: assembled.png, assembled.pdf, legend.md and summary.json."""
    layout = load_layout(layout_path)
    out_dir = output_root / layout["id"]
    info = assemble(layout, root, out_dir, dpi)
    caption = build_caption(layout, root)
    (out_dir / "legend.md").write_text(
        f"# {layout['id']}: {layout['title']}\n\n## Assembled figure (`assembled.png`)\n\n{caption}\n"
    )
    panels = info["panels"]
    summary = {
        "description": (
            f"{layout['title']}: a paper figure assembled from pipeline panels by "
            f"notebooks/paper-figure-assembly.py according to {layout_path.as_posix()}. "
            "The assembly computes no value; every number shown is drawn by the source panels."
        ),
        "numbers": {
            "panels": len(panels),
            "vector_panels": sum(panel["vector"] for panel in panels),
            "pipeline_panels": sum(panel["origin"] == "pipeline" for panel in panels),
            "static_panels": sum(panel["origin"] == "static" for panel in panels),
            "width_in": info["width_in"],
            "height_in": info["height_in"],
        },
        "files": [
            {"path": "assembled.png", "kind": "figure", "caption": caption},
            {"path": "assembled.pdf", "kind": "figure", "caption": caption},
            {"path": "legend.md", "kind": "data", "caption": "The caption of the assembled figure."},
        ],
        "layout": layout_path.as_posix(),
        "panels": panels,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.echo(f"{layout['id']}: {info['width_in']} x {info['height_in']} in, {len(panels)} panels")


@app.command()
def check(
    layout_paths: list[Path] = typer.Argument(..., help="Layout files to check"),
    root: Path = typer.Option(Path("."), help="Checkout root that layout sources are relative to"),
) -> None:
    """Report missing sources and boxes outside the page, without drawing anything."""
    failed = False
    for path in layout_paths:
        for problem in layout_problems(load_layout(path), root):
            failed = True
            typer.echo(f"{path}: {problem}", err=True)
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
