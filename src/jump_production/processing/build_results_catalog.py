"""Build the results catalog read by the publication data browser."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from jump_production.catalog import source_revision, write_catalog
from jump_production.config import PROJ_ROOT
from jump_production.workflow import declared_outputs


def main(
    registry: Annotated[Path, typer.Option(help="Authored registry of sections and entries")],
    output: Annotated[Path, typer.Option(help="Catalog output path")],
    default_dataset: Annotated[str, typer.Option()],
    default_columns: Annotated[str, typer.Option()],
    default_preprocessing: Annotated[str, typer.Option()],
    default_filter: Annotated[str, typer.Option()],
    default_activity_params: Annotated[str, typer.Option()],
) -> None:
    catalog = write_catalog(
        output,
        registry,
        root=PROJ_ROOT,
        declarations=declared_outputs(),
        canonical_key={
            "default_dataset": default_dataset,
            "default_columns": default_columns,
            "default_preprocessing": default_preprocessing,
            "default_filter": default_filter,
            "default_activity_params": default_activity_params,
        },
        revision=source_revision(PROJ_ROOT),
    )
    files = sum(len(entry["files"]) for entry in catalog["entries"])
    logger.success("Wrote {} with {} entries and {} files", output, len(catalog["entries"]), files)


if __name__ == "__main__":
    typer.run(main)
