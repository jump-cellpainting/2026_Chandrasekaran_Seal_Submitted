"""The paper-figure assembler places panels as the layout says and refuses a layout it cannot honour.

PyMuPDF is not part of the pinned environments, so these tests are skipped by `just test` and run
by `just test-paper-figures`, which fetches it with uv.
"""

import json
import os
import subprocess

import pytest
import yaml
from conftest import load_notebook

pymupdf = pytest.importorskip("pymupdf")
PIL_Image = pytest.importorskip("PIL.Image")

assembly = load_notebook("paper-figure-assembly.py")
FIGURE_ID = "figure_paper_fixture"


def write_sources(root):
    """One vector page with ink in a known place and one raster panel, each beside an envelope."""
    plots = root / "data/processed/plots"
    plots.mkdir(parents=True)
    document = pymupdf.open()
    page = document.new_page(width=288, height=144)
    page.draw_rect(pymupdf.Rect(20, 20, 124, 124), color=(0, 0, 1), fill=(0, 0, 1))
    page.draw_rect(pymupdf.Rect(164, 20, 268, 124), color=(1, 0, 0), fill=(1, 0, 0))
    document.save(plots / "pair.pdf")
    PIL_Image.new("RGB", (600, 300), (0, 128, 0)).save(plots / "block.png")
    (plots / "summary.json").write_text(
        json.dumps(
            {
                "description": "Fixture plots.",
                "numbers": {},
                "files": [{"path": "pair.pdf", "kind": "figure", "caption": "Two squares"}],
            }
        )
    )


def write_layout(root, rows, **extra):
    path = root / f"{FIGURE_ID}.yaml"
    path.write_text(yaml.safe_dump({"id": FIGURE_ID, "title": "Fixture figure", "width": 5.0, "rows": rows, **extra}))
    return path


def default_rows():
    pair = "data/processed/plots/pair.pdf"
    return [
        [
            {"src": pair, "width": 2.0, "clip": [0, 0, 0.5, 1]},
            {"src": pair, "width": 2.0, "clip": [0.5, 0, 1, 1], "title": "Right half"},
        ],
        [{"src": "data/processed/plots/block.png", "width": 3.0, "letter": False}],
    ]


def test_assembly_writes_a_valid_figure_package(tmp_path):
    write_sources(tmp_path)
    layout = assembly.load_layout(write_layout(tmp_path, default_rows()))
    out_dir = tmp_path / "tree/figures" / FIGURE_ID
    info = assembly.assemble(layout, tmp_path, out_dir)

    panels = info["panels"]
    assert [panel["drawn_letter"] for panel in panels] == ["A", "B", ""]
    assert [panel["position"] for panel in panels] == ["Top row, left", "Top row, right", "Bottom row"]
    assert [panel["vector"] for panel in panels] == [True, True, False]
    assert info["width_in"] == 5.0
    # Each half is trimmed to its square, so both come out square at the requested width.
    assert panels[0]["height_in"] == pytest.approx(2.0, abs=0.05)

    with pymupdf.open(out_dir / "assembled.pdf") as document:
        text = document[0].get_text()
        assert document[0].rect.width == pytest.approx(5.0 * 72)
    assert "A" in text and "Right half" in text
    with PIL_Image.open(out_dir / "assembled.png") as image:
        assert image.width == pytest.approx(5.0 * 300, abs=2)
        colours = {colour for _, colour in image.convert("RGB").getcolors(maxcolors=1 << 20)}
    assert {(0, 0, 255), (255, 0, 0), (0, 128, 0)} <= colours

    caption = assembly.build_caption(layout, tmp_path)
    assert (
        "Top row, left: from `data/processed/plots/pair.pdf`, clipped (pipeline output). Source caption: Two squares."
        in caption
    )


def test_compose_command_passes_figure_validation(tmp_path):
    write_sources(tmp_path)
    layout_path = write_layout(tmp_path, default_rows())
    tree = tmp_path / "tree"
    assembly.compose(layout_path, output_root=tree / "figures", root=tmp_path, dpi=150)

    summary = json.loads((tree / "figures" / FIGURE_ID / "summary.json").read_text())
    assert summary["numbers"]["panels"] == 3 and summary["numbers"]["static_panels"] == 0
    assert {record["path"] for record in summary["files"]} == {"assembled.png", "assembled.pdf", "legend.md"}
    repository = assembly.Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", "scripts/validate_figure.sh", FIGURE_ID],
        cwd=repository,
        env={**os.environ, "PROCESSED_DIR": str(tree)},
        check=True,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[0][0].update(src="data/processed/plots/absent.pdf"), "missing panel source"),
        (lambda rows: rows[0][0].update(clip=[0, 0, 1.2, 1]), "clip box must lie inside the page"),
        (lambda rows: rows[0][0].update(clip=[0.6, 0, 0.5, 1]), "clip box must lie inside the page"),
        (lambda rows: rows[0][0].update(src="../outside.pdf"), "relative to the checkout"),
        (lambda rows: rows[1][0].update(width=6.0), "wider than the figure"),
        (lambda rows: rows[0][1].update(title="A title far too long to sit above a panel this narrow"), "title wider"),
    ],
)
def test_assembly_refuses_a_layout_it_cannot_honour(tmp_path, change, message):
    write_sources(tmp_path)
    rows = default_rows()
    change(rows)
    layout = assembly.load_layout(write_layout(tmp_path, rows))
    with pytest.raises(ValueError, match=message):
        assembly.assemble(layout, tmp_path, tmp_path / "out")


@pytest.mark.parametrize(
    "damage",
    [
        lambda layout: layout.update(id="another_name"),
        lambda layout: layout["rows"][0][0].pop("width"),
        lambda layout: layout["rows"][0][0].update(origin="borrowed"),
        lambda layout: layout["rows"][0][0].update(scale=2),
    ],
)
def test_layout_schema_is_closed(tmp_path, damage):
    document = {"id": FIGURE_ID, "title": "Fixture figure", "rows": default_rows()}
    damage(document)
    path = tmp_path / f"{FIGURE_ID}.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        assembly.load_layout(path)


def test_stacked_panels_are_described_by_position(tmp_path):
    write_sources(tmp_path)
    block = {"src": "data/processed/plots/block.png", "width": 2.0}
    layout = assembly.load_layout(write_layout(tmp_path, [[[dict(block), dict(block)], dict(block)]]))
    assert [assembly.position(r, c, s, layout) for r, c, s, _, _ in assembly.panels_of(layout)] == [
        "Left, upper",
        "Left, lower",
        "Right",
    ]
