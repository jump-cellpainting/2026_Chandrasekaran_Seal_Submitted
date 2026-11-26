"""Paper-figure layouts name only sources that a rule writes or that the repository versions."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from jump_production.config import PROJ_ROOT
from jump_production.workflow import declared_outputs, matches_output

LAYOUT_DIR = PROJ_ROOT / "configs/paper_figures"
LAYOUTS = sorted(LAYOUT_DIR.glob("figure_paper_*.yaml"))
TRACKED = set(subprocess.check_output(["git", "ls-files", "assets/paper_figures"], cwd=PROJ_ROOT, text=True).split())
# The same drift rules that scripts/validate_figure.sh applies to a legend.
MANUSCRIPT_NUMBER = re.compile(r"\b(?:Supplementary Figure|Figure \d|Fig\.? ?S?\d)")
PANEL_LETTER = re.compile(r"\bpanels? [A-H]\b|\b[A-H]-[A-H]\b")


def panels(layout):
    for row in layout["rows"]:
        for cell in row:
            yield from cell if isinstance(cell, list) else [cell]


def test_layouts_exist():
    assert LAYOUTS


@pytest.mark.parametrize("path", LAYOUTS, ids=lambda path: path.stem)
def test_layout_sources_are_produced_or_pinned(path):
    layout = yaml.safe_load(path.read_text())
    assert layout["id"] == path.stem
    declarations = declared_outputs()
    for panel in panels(layout):
        src, origin = panel["src"], panel.get("origin", "pipeline")
        if src.startswith("assets/paper_figures/"):
            assert origin == "static", src
            assert src in TRACKED, f"{src} is not tracked; *.pdf is ignored outside assets/paper_figures/"
            continue
        assert origin == "pipeline", src
        if src.startswith("data/paper_figures/panels/"):
            producers = {d["rule"] for d in declarations if matches_output(d["output"], src)}
            assert len(producers) == 1, f"{src}: {sorted(producers)}"
        else:
            # Analyses declare a summary or a completion marker, not every plot. An undeclared plot
            # needs one of the markers that `paper_inputs` in rules/paper_figures.smk waits for.
            assert src.startswith("data/processed/"), src
            folder = Path(src).parent
            candidates = [src, f"{folder}/summary.json", f"{folder}/.complete", f"{folder.parent}/.complete"]
            assert any(matches_output(d["output"], candidate) for d in declarations for candidate in candidates), (
                f"no rule declares {src} or a marker for its analysis"
            )


@pytest.mark.parametrize("path", LAYOUTS, ids=lambda path: path.stem)
def test_layout_text_names_no_manuscript_number_or_panel_letter(path):
    layout = yaml.safe_load(path.read_text())
    text = [layout["title"]] + [panel.get(key, "") for panel in panels(layout) for key in ("title", "note")]
    assert not [line for line in text if MANUSCRIPT_NUMBER.search(line) or PANEL_LETTER.search(line)]
