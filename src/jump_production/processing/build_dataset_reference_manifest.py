"""Build the canonical dataset-reference manifest in a standalone process."""

from pathlib import Path
from typing import Annotated

import typer

from jump_production.workflow import write_dataset_reference_manifest


def main(
    summaries: Annotated[list[Path], typer.Argument(help="Summary envelopes to include")],
    output: Annotated[Path, typer.Option(help="Manifest output path")],
    default_dataset: Annotated[str, typer.Option()],
    default_columns: Annotated[str, typer.Option()],
    default_preprocessing: Annotated[str, typer.Option()],
    default_filter: Annotated[str, typer.Option()],
    default_activity_params: Annotated[str, typer.Option()],
) -> None:
    manifest = write_dataset_reference_manifest(
        output,
        summaries,
        canonical_key={
            "default_dataset": default_dataset,
            "default_columns": default_columns,
            "default_preprocessing": default_preprocessing,
            "default_filter": default_filter,
            "default_activity_params": default_activity_params,
        },
    )
    typer.echo(f"Wrote {output} with {len(manifest['summaries'])} summaries")


if __name__ == "__main__":
    typer.run(main)
