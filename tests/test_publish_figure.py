"""Behavioral tests for exact paper-figure directory validation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_figure.sh"
LEGEND = "# figure_test: Test figure\n\n## Test panel (`panel.png`)\n\n{caption}\n"


@pytest.fixture
def figure_dir(tmp_path: Path) -> Path:
    """Create one minimal standard-summary figure directory that validates."""
    figure_dir = tmp_path / "processed" / "figures" / "figure_test"
    (figure_dir / "tables").mkdir(parents=True)
    (figure_dir / "panel.png").write_bytes(b"png fixture\n")
    (figure_dir / "tables" / "values.csv").write_text("value\n1\n")
    (figure_dir / "legend.md").write_text(LEGEND.format(caption="The caption."))
    summary = {
        "description": "Synthetic validation fixture.",
        "numbers": {"rows": 1},
        "configuration": {"seed": 7},
        "files": [
            {"path": "panel.png", "kind": "figure", "caption": "Test panel."},
            {"path": "tables/values.csv", "kind": "table", "caption": "Test values."},
            {"path": "legend.md", "kind": "data", "caption": "Authored legend."},
        ],
    }
    (figure_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    return figure_dir


def validate(processed_dir: Path, figure_id: str = "figure_test") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VALIDATOR), figure_id],
        cwd=ROOT,
        env={**os.environ, "PROCESSED_DIR": str(processed_dir)},
        check=False,
        capture_output=True,
        text=True,
    )


def assert_rejected(figure_dir: Path, error: str) -> None:
    result = validate(figure_dir.parents[1])
    assert result.returncode != 0
    assert error in result.stderr


def declare_path(path: str):
    return lambda summary: summary["files"][0].update(path=path)


def link_panel_to_legend(figure_dir: Path) -> None:
    (figure_dir / "panel.png").unlink()
    (figure_dir / "panel.png").symlink_to(figure_dir / "legend.md")


def test_validate_accepts_an_exact_summary_declared_directory(figure_dir: Path) -> None:
    result = validate(figure_dir.parents[1])

    assert result.returncode == 0, result.stderr
    assert "Validated figure_test: 3 declared files plus summary.json" in result.stdout


@pytest.mark.parametrize(
    ("transform", "error"),
    [
        (lambda summary: summary.update(description=""), "summary description"),
        (lambda summary: summary.update(numbers=[]), "summary numbers"),
        (lambda summary: summary.update(files=[]), "summary files"),
        (lambda summary: summary["files"][0].pop("kind"), "invalid kind"),
        (lambda summary: summary["files"][0].update(caption=""), "invalid caption"),
        # Traversal, reserved and non-normalized declared paths.
        *(
            (declare_path(path), "declared path")
            for path in (
                "../panel.png",
                "/tmp/panel.png",
                "nested//panel.png",
                "nested/../panel.png",
                "summary.json",
                "_proto/draft.png",
                "_marimo/main.js",
                "figure.json",
                "index.html",
            )
        ),
    ],
)
def test_validate_rejects_malformed_summaries(figure_dir: Path, transform, error: str) -> None:
    path = figure_dir / "summary.json"
    summary = json.loads(path.read_text())
    transform(summary)
    path.write_text(json.dumps(summary) + "\n")

    assert_rejected(figure_dir, error)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda figure_dir: (figure_dir / "notes.txt").write_text("undeclared\n"),
            "undeclared files in figure directory",
        ),
        (link_panel_to_legend, "traverses a symlink"),
        (lambda figure_dir: (figure_dir / "panel.png").write_bytes(b""), "declared artifact is empty"),
    ],
)
def test_validate_rejects_directory_contents_outside_the_declaration(figure_dir: Path, mutate, error: str) -> None:
    mutate(figure_dir)

    assert_rejected(figure_dir, error)


@pytest.mark.parametrize(
    "figure_id",
    ["../figure_test", "figure/test", "Figure_test", "figure", "2.08"],
)
def test_validator_rejects_unsafe_or_nonfigure_ids(tmp_path: Path, figure_id: str) -> None:
    result = validate(tmp_path / "processed", figure_id)

    assert result.returncode != 0
    assert "unsafe figure id" in result.stderr


def test_validate_requires_existing_directory_and_summary(tmp_path: Path) -> None:
    figure_dir = tmp_path / "processed" / "figures" / "figure_test"
    assert_rejected(figure_dir, "missing regular figure directory")

    figure_dir.mkdir(parents=True)
    assert_rejected(figure_dir, "missing regular summary")


@pytest.mark.parametrize(
    ("legend", "error"),
    [
        # One section per rendered figure, each naming its file, without manuscript numbering.
        (LEGEND.format(caption="The caption.").replace("panel.png", "other.png"), "do not match the rendered figures"),
        (LEGEND.format(caption="The caption.").replace(" (`panel.png`)", ""), "must name its file in backticks"),
        (LEGEND.format(caption="The caption.").replace("figure_test:", "Figure 1."), "manuscript figure numbers"),
        *(
            (LEGEND.format(caption=caption), "manuscript panel letters")
            for caption in (
                "C, clinical phase: the counts.",
                "The counts. D, Lipinski violations follow.",
                "Described in panel C of the figure.",
                "Panels F-I are not part of this A-E build.",
            )
        ),
        # Captions keyed by position, and prose that merely resembles panel letters, pass.
        *(
            (LEGEND.format(caption=caption), None)
            for caption in (
                "Left, clinical phase. Right, Lipinski violations.",
                "Top left, property space. Bottom right, the structure UMAP.",
                "The AKT1/AKT2/AKT3 and PIK3CG/PIK3R1 sets repeat.",
                "Bins hold 658-23.2k compounds at 5% FDR.",
            )
        ),
    ],
)
def test_validate_checks_legend_sections_and_captions(figure_dir: Path, legend: str, error: str | None) -> None:
    (figure_dir / "legend.md").write_text(legend)

    if error is None:
        result = validate(figure_dir.parents[1])
        assert result.returncode == 0, result.stderr
    else:
        assert_rejected(figure_dir, error)
