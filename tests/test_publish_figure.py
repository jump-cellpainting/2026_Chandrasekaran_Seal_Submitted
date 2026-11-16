"""Behavioral tests for exact paper-figure directory validation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_figure.sh"


def write_valid_figure(processed_dir: Path, figure_id: str = "figure_test") -> Path:
    """Create one minimal standard-summary figure directory."""
    figure_dir = processed_dir / "figures" / figure_id
    (figure_dir / "tables").mkdir(parents=True)
    (figure_dir / "panel.png").write_bytes(b"png fixture\n")
    (figure_dir / "tables" / "values.csv").write_text("value\n1\n")
    (figure_dir / "legend.md").write_text("# figure_test: Test figure\n\n## Test panel (`panel.png`)\n\nThe caption.\n")
    summary = {
        "description": "Synthetic validation fixture.",
        "numbers": {"rows": 1},
        "configuration": {"seed": 7},
        "files": [
            {"path": "panel.png", "kind": "figure", "caption": "Test panel."},
            {
                "path": "tables/values.csv",
                "kind": "table",
                "caption": "Test values.",
            },
            {"path": "legend.md", "kind": "data", "caption": "Authored legend."},
        ],
    }
    (figure_dir / "summary.json").write_text(json.dumps(summary) + "\n")
    return figure_dir


def run_validator(
    tmp_path: Path,
    *,
    figure_id: str = "figure_test",
    prepare_figure: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the validator against isolated files."""
    processed_dir = tmp_path / "processed"
    figure_dir = processed_dir / "figures" / figure_id
    if prepare_figure:
        figure_dir = write_valid_figure(processed_dir, figure_id)
    result = subprocess.run(
        [str(VALIDATOR), figure_id],
        cwd=ROOT,
        env={**os.environ, "PROCESSED_DIR": str(processed_dir)},
        check=False,
        capture_output=True,
        text=True,
    )
    return result, figure_dir


def rewrite_summary(figure_dir: Path, transform) -> None:
    """Apply one test mutation to the standard summary."""
    path = figure_dir / "summary.json"
    summary = json.loads(path.read_text())
    transform(summary)
    path.write_text(json.dumps(summary) + "\n")


def rerun_validator(figure_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(VALIDATOR), figure_dir.name],
        cwd=ROOT,
        env={**os.environ, "PROCESSED_DIR": str(figure_dir.parents[1])},
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_accepts_an_exact_summary_declared_directory(tmp_path: Path) -> None:
    result, _figure_dir = run_validator(tmp_path)

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
    ],
)
def test_validate_rejects_malformed_standard_summary_fields(
    tmp_path: Path,
    transform,
    error: str,
) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    rewrite_summary(figure_dir, transform)

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert error in retried.stderr


@pytest.mark.parametrize(
    "figure_id",
    ["../figure_test", "figure/test", "Figure_test", "figure", "2.08"],
)
def test_validator_rejects_unsafe_or_nonfigure_ids(tmp_path: Path, figure_id: str) -> None:
    result, _figure_dir = run_validator(
        tmp_path,
        figure_id=figure_id,
        prepare_figure=False,
    )

    assert result.returncode != 0
    assert "unsafe figure id" in result.stderr


@pytest.mark.parametrize(
    "declared_path",
    [
        "../panel.png",
        "/tmp/panel.png",
        "nested//panel.png",
        "nested/../panel.png",
        "summary.json",
        "_proto/draft.png",
        "_marimo/main.js",
        "figure.json",
        "index.html",
    ],
)
def test_validate_rejects_traversal_reserved_and_nonnormalized_paths(
    tmp_path: Path,
    declared_path: str,
) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    rewrite_summary(
        figure_dir,
        lambda summary: summary["files"][0].update(path=declared_path),
    )

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "declared path" in retried.stderr


def test_validate_rejects_undeclared_regular_files(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "notes.txt").write_text("undeclared\n")

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "undeclared files in figure directory" in retried.stderr


def test_validate_rejects_symlinks(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "panel.png").unlink()
    (figure_dir / "panel.png").symlink_to(figure_dir / "legend.md")

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "traverses a symlink" in retried.stderr


def test_validate_rejects_empty_declared_artifacts(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "panel.png").write_bytes(b"")

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "declared artifact is empty" in retried.stderr


def test_validate_requires_existing_directory_and_summary(tmp_path: Path) -> None:
    missing, _figure_dir = run_validator(tmp_path, prepare_figure=False)
    assert missing.returncode != 0
    assert "missing regular figure directory" in missing.stderr

    figure_dir = tmp_path / "processed" / "figures" / "figure_test"
    figure_dir.mkdir(parents=True)
    no_summary = rerun_validator(figure_dir)
    assert no_summary.returncode != 0
    assert "missing regular summary" in no_summary.stderr


def test_validate_requires_one_legend_section_per_rendered_figure(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "legend.md").write_text("# figure_test: Test figure\n\n## Test panel (`other.png`)\n\nThe caption.\n")

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "do not match the rendered figures" in retried.stderr


def test_validate_requires_every_legend_section_to_name_its_file(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "legend.md").write_text("# figure_test: Test figure\n\n## Test panel\n\nThe caption.\n")

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "must name its file in backticks" in retried.stderr


def test_validate_rejects_manuscript_figure_numbers(tmp_path: Path) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "legend.md").write_text(
        "# Figure 1. Test figure\n\n## Test panel (`panel.png`)\n\nThe caption.\n",
    )

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "manuscript figure numbers" in retried.stderr


@pytest.mark.parametrize(
    "caption",
    [
        "C, clinical phase: the counts.",
        "The counts. D, Lipinski violations follow.",
        "Described in panel C of the figure.",
        "Panels F-I are not part of this A-E build.",
    ],
)
def test_validate_rejects_manuscript_panel_letters(tmp_path: Path, caption: str) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "legend.md").write_text(
        f"# figure_test: Test figure\n\n## Test panel (`panel.png`)\n\n{caption}\n",
    )

    retried = rerun_validator(figure_dir)

    assert retried.returncode != 0
    assert "manuscript panel letters" in retried.stderr


@pytest.mark.parametrize(
    "caption",
    [
        "Left, clinical phase. Right, Lipinski violations.",
        "Top left, property space. Bottom right, the structure UMAP.",
        "The AKT1/AKT2/AKT3 and PIK3CG/PIK3R1 sets repeat.",
        "Bins hold 658-23.2k compounds at 5% FDR.",
    ],
)
def test_validate_accepts_captions_keyed_by_position(tmp_path: Path, caption: str) -> None:
    result, figure_dir = run_validator(tmp_path)
    assert result.returncode == 0
    (figure_dir / "legend.md").write_text(
        f"# figure_test: Test figure\n\n## Test panel (`panel.png`)\n\n{caption}\n",
    )

    retried = rerun_validator(figure_dir)

    assert retried.returncode == 0, retried.stderr
