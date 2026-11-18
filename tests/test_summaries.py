"""Contract tests for the shared summary-envelope helpers."""

from pathlib import Path

import pytest

from jump_production.config import PROJ_ROOT
from jump_production.summaries import index_files, write_summary


def build_output_dir(tmp_path: Path) -> Path:
    """Write one output directory with a figure, a table, and ignored extras."""
    output_dir = tmp_path / "figure_test"
    (output_dir / "tables").mkdir(parents=True)
    (output_dir / "panel.png").write_bytes(b"png fixture\n")
    (output_dir / "activity_rates.csv").write_text("value\n1\n")
    (output_dir / "tables" / "nested.csv").write_text("value\n2\n")
    (output_dir / "legend.md").write_text("# legend\n")
    (output_dir / "summary.json").write_text("{}\n")
    return output_dir


def test_index_files_keeps_joined_paths_by_default(tmp_path: Path) -> None:
    output_dir = build_output_dir(tmp_path)

    records = index_files(output_dir)

    assert [record["path"] for record in records] == [
        str(output_dir / "activity_rates.csv"),
        str(output_dir / "panel.png"),
    ]
    assert [record["kind"] for record in records] == ["table", "figure"]
    assert [record["caption"] for record in records] == ["activity rates", "panel"]


def test_paths_inside_the_checkout_never_name_the_machine(tmp_path: Path) -> None:
    # Analyses that build their output directory from jump_production.config hold absolute paths.
    # The same analysis must write the same envelope wherever the repository is cloned.
    output_dir = PROJ_ROOT / "data" / "processed" / "example"

    summary = write_summary(
        tmp_path / "summary.json",
        description="Synthetic analysis.",
        numbers={"rows": 1},
        files=[
            output_dir / "figure.png",
            {"path": str(output_dir / "tables" / "rates.csv"), "caption": "Rates"},
            "data/processed/example/already_relative.csv",
        ],
    )

    assert [record["path"] for record in summary["files"]] == [
        "data/processed/example/figure.png",
        "data/processed/example/tables/rates.csv",
        "data/processed/example/already_relative.csv",
    ]
    assert str(PROJ_ROOT) not in (tmp_path / "summary.json").read_text()


def test_index_files_emits_package_relative_paths_for_publication(tmp_path: Path) -> None:
    output_dir = build_output_dir(tmp_path)

    records = index_files(output_dir, relative_to=output_dir)

    # publish_figure.sh resolves declared paths under the figure directory, so a published
    # package must declare bare relative paths rather than repository-relative ones.
    assert [record["path"] for record in records] == ["activity_rates.csv", "panel.png"]
    for record in records:
        assert not Path(record["path"]).is_absolute()
        assert ".." not in record["path"]


def test_index_files_rejects_a_root_that_does_not_contain_the_directory(tmp_path: Path) -> None:
    output_dir = build_output_dir(tmp_path)

    with pytest.raises(ValueError):
        index_files(output_dir, relative_to=tmp_path / "elsewhere")


def test_write_summary_preserves_relative_declared_paths(tmp_path: Path) -> None:
    output_dir = build_output_dir(tmp_path)

    summary = write_summary(
        output_dir / "summary.json",
        description="Synthetic publication package.",
        numbers={"rows": 1},
        files=index_files(output_dir, relative_to=output_dir),
    )

    assert [record["path"] for record in summary["files"]] == ["activity_rates.csv", "panel.png"]
