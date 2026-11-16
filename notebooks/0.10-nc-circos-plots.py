"""Create circos plots visualizing JUMP experiment metadata.

Migrated from jump-cellpainting/jump-production (02.visualization/00.0.create-circos-plot.ipynb).
Generates two circos plots:
1. Plate metadata: sectors sized by plate count, rings for batches and plate types
2. Experiment properties: microscope, imaging mode, CellProfiler version, well count, sites/well

Usage:
    pixi run python notebooks/0.10-nc-circos-plots.py
"""

from pathlib import Path

import duckdb
import pandas as pd
import typer
from loguru import logger
from matplotlib.patches import Patch
from pycirclize import Circos

from jump_production.config import DEFAULT_DPI, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from jump_production.summaries import index_files, write_summary

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROCESSED_DATA_DIR / "circos-plots"

COLOR_MAP = {
    "COMPOUND": "#000000",
    "TARGET2": "#0072b2",
    "DMSO": "#009e73",
    "COMPOUND_EMPTY": "#d55e00",
    "POSCON8": "#cc79a7",
}

RMAP = {
    "COMPOUND": (87.5, 92.5),
    "TARGET2": (80, 85),
    "DMSO": (72.5, 77.5),
    "COMPOUND_EMPTY": (65, 70),
    "POSCON8": (57.5, 62.5),
}

app = typer.Typer()


def zero_pad_source(source: str) -> str:
    """Convert 'source_1' to 'source_01' for display."""
    return f"source_{int(source.split('_')[1]):02d}"


def load_data():
    """Load plate metadata and experiment properties from DuckDB."""
    con = duckdb.connect(str(METADATA_DB), read_only=True)

    plate_df = con.sql("""
        SELECT Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_PlateType
        FROM plate
        WHERE Metadata_Source NOT IN ('source_4', 'source_13', 'source_15')
    """).fetchdf()

    microscope_df = con.sql("""
        SELECT Metadata_Source, Metadata_Microscope_Name,
               Metadata_Widefield_vs_Confocal, Metadata_Sites_Per_Well
        FROM microscope_config
    """).fetchdf()

    cp_version_df = con.sql("SELECT * FROM cellprofiler_version").fetchdf()

    con.close()

    plate_df["Metadata_Source"] = plate_df["Metadata_Source"].apply(zero_pad_source)
    plate_df = plate_df.sort_values("Metadata_Source").reset_index(drop=True)

    cp_scope_df = microscope_df.merge(cp_version_df, on="Metadata_Source", how="inner")
    cp_scope_df["Metadata_Source"] = cp_scope_df["Metadata_Source"].apply(zero_pad_source)
    cp_scope_df = cp_scope_df.assign(
        Metadata_Well_Count=lambda x: x["Metadata_Source"].map(
            lambda s: 1536 if s in ("source_01", "source_09") else 384
        )
    )

    cp_scope_df = cp_scope_df[cp_scope_df["Metadata_Source"].isin(plate_df["Metadata_Source"].unique())].reset_index(
        drop=True
    )

    return plate_df, cp_scope_df


def plot_plate_metadata(plate_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Create circos plot of plate metadata (batches and plate types)."""
    n_plates_pbs = (
        plate_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_PlateType"])
        .agg({"Metadata_Plate": "count"})
        .reset_index()
    )
    n_plates_bs = (
        n_plates_pbs.groupby(["Metadata_Source", "Metadata_Batch"]).agg({"Metadata_Plate": "sum"}).reset_index()
    )
    n_plates_s = n_plates_bs.groupby("Metadata_Source").agg({"Metadata_Plate": "sum"}).reset_index()

    sectors = dict(zip(n_plates_s.Metadata_Source, n_plates_s.Metadata_Plate))
    circos = Circos(sectors, space=2, start=0, end=360)

    for sector in circos.sectors:
        sector.text(sector.name)
        batch_counter = 0
        plate_counter = 0

        for batch in n_plates_bs.query("Metadata_Source==@sector.name").Metadata_Batch.unique():
            n_batch = n_plates_bs.query("Metadata_Source==@sector.name & Metadata_Batch==@batch").Metadata_Plate.values[
                0
            ]

            sector.rect(
                start=batch_counter,
                end=batch_counter + n_batch,
                r_lim=(95, 100),
                fc="#bdbdbd",
                ec="white",
                lw=0.5,
            )
            batch_counter += n_batch

            for plate_type in n_plates_pbs.query(
                "Metadata_Source==@sector.name & Metadata_Batch==@batch"
            ).Metadata_PlateType:
                n_pt = n_plates_pbs.query(
                    "Metadata_Source==@sector.name & Metadata_Batch==@batch & Metadata_PlateType==@plate_type"
                ).Metadata_Plate.values[0]

                sector.rect(
                    start=plate_counter,
                    end=plate_counter + n_pt,
                    r_lim=RMAP[plate_type],
                    fc=COLOR_MAP[plate_type],
                    ec=COLOR_MAP[plate_type],
                    lw=0.5,
                )
                plate_counter += n_pt

    fig = circos.plotfig()

    handles = [Patch(color="#bdbdbd", label="Batches of plates")]
    for pt, color in COLOR_MAP.items():
        handles.append(Patch(color=color, label=f"{pt} plates"))

    circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.5), loc="center", fontsize=12)

    for fmt in ("png", "svg"):
        fig.savefig(
            output_dir / f"compound_plate_metadata_circos.{fmt}",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
        )
    logger.success(f"Saved plate metadata circos to {output_dir}")


def plot_experiment_properties(cp_scope_df: pd.DataFrame, sources: list[str], output_dir: Path, dpi: int) -> None:
    """Create circos plot of experiment properties per source."""
    sectors = dict(zip(sources, [1] * len(sources)))
    circos = Circos(sectors, space=2, start=0, end=320)

    track_defs = [
        ((90, 100), "Metadata_Microscope_Name", 95, 352, "Microscope"),
        ((75, 85), "Metadata_Widefield_vs_Confocal", 83, 344, "Widefield vs Confocal"),
        ((60, 70), "Metadata_CellProfiler_Version", 68, 343, "CellProfiler version"),
        ((45, 55), "Metadata_Well_Count", 51, 347, "Well count"),
        ((30, 40), "Metadata_Sites_Per_Well", 37, 338, "Sites per well"),
    ]

    cp_indexed = cp_scope_df.set_index("Metadata_Source")

    for sector in circos.sectors:
        sector.text(sector.name)
        row = cp_indexed.loc[sector.name]
        for r_lim, col, *_ in track_defs:
            track = sector.add_track(r_lim)
            track.axis()
            track.text(row[col], size=9, color="black")

    for _, _, r, deg, label in track_defs:
        circos.text(label, r=r, deg=deg, size=10, color="black")

    fig = circos.plotfig()

    for fmt in ("png", "svg"):
        fig.savefig(
            output_dir / f"compound_experiment_properties_circos.{fmt}",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
        )
    logger.success(f"Saved experiment properties circos to {output_dir}")


@app.command()
def main(
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi", help="Plot DPI"),
):
    """Generate circos plots of JUMP experiment metadata."""
    output = output_dir or OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)

    plate_df, cp_scope_df = load_data()
    sources = sorted(plate_df["Metadata_Source"].unique())

    plot_plate_metadata(plate_df, output, dpi)
    plot_experiment_properties(cp_scope_df, sources, output, dpi)

    # Emit the standard summary envelope so the dataset-reference hub discovers this node.
    write_summary(
        output / "summary.json",
        description=(
            "Circos visualizations of JUMP experiment metadata: a plate-metadata plot (sectors per "
            "source sized by plate count, rings for batches and plate types) and an experiment-"
            "properties plot (microscope, imaging mode, CellProfiler version, well count, sites per "
            "well). Primarily a figure node; reports the plate and source counts it visualizes."
        ),
        numbers={"n_plates": int(len(plate_df)), "n_sources": int(len(sources))},
        files=index_files(output),
    )

    return 0


if __name__ == "__main__":
    app()
