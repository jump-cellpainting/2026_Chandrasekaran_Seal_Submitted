"""Render the positive-control heatmap from the cross-source copairs analysis."""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import typer

from jump_production.config import DEFAULT_DPI
from jump_production.queries import query_cross_source_reproducibility
from jump_production.summaries import write_legend, write_summary
from jump_production.visualization import attach_caption

FIGURE_ID = "figure_positive_control_reproducibility"
FIGURE_TITLE = "Positive-control reproducibility across sources"
DATASET = "compound_no_source7"
PREPROCESSING = "activity_poscon_only"
FILTER = "all_sources"


def main(
    copairs_db: Path = typer.Option(Path("data/processed/copairs_results.duckdb"), exists=True, dir_okay=False),
    metadata_db: Path = typer.Option(Path("data/interim/jump_metadata_augmented.duckdb"), exists=True, dir_okay=False),
    output: Path = Path("data/processed/figures") / FIGURE_ID,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Render within-source, cross-source, and difference heatmaps for the JUMP positive controls."""
    with duckdb.connect(str(copairs_db), read_only=True) as con:
        scores = query_cross_source_reproducibility(con, DATASET, PREPROCESSING, FILTER)
    if scores.empty:
        raise typer.BadParameter(f"No copairs results for {DATASET}/{PREPROCESSING}/{FILTER}")

    with duckdb.connect(str(metadata_db), read_only=True) as con:
        names = dict(
            con.execute(
                "SELECT Metadata_JCP2022, Metadata_Name FROM perturbation_control WHERE Metadata_pert_type = 'poscon'"
            ).fetchall()
        )
    compounds = sorted(scores["Metadata_JCP2022"].unique())
    unnamed = [compound for compound in compounds if not names.get(compound)]
    if unnamed:
        raise typer.BadParameter(f"Missing positive-control names for: {', '.join(unnamed)}")
    codes = {
        compound: f"C{index}"
        for index, compound in enumerate(sorted(compounds, key=lambda compound: names[compound].casefold()), start=1)
    }
    sources = sorted(scores["Metadata_Source"].unique(), key=lambda source: int(source.split("_")[1]))
    source_labels = {source: source.replace("source_", "S") for source in sources}
    within, cross = (
        scores.pivot(index="Metadata_Source", columns="Metadata_JCP2022", values=score_column)
        .reindex(index=sources, columns=compounds)
        .rename(index=source_labels, columns=codes)
        for score_column in ("nmAP_within", "nmAP_cross")
    )
    panels = (within, cross, within - cross)
    saturated = sum(int((values.abs() > 1).sum().sum()) for values in panels)

    key = ", ".join(f"{code} {names[compound]}" for compound, code in codes.items())
    caption = (
        f"Normalized mean Average Precision (mAP) for the {len(compounds)} JUMP positive controls across "
        f"{len(sources)} sources in the {DATASET} dataset, using cosine similarity. "
        "Left, within-source normalized mAP. Middle, cross-source normalized mAP. "
        "Right, within-source minus cross-source normalized mAP. "
        "All matched compound-source scores are shown without a significance filter. "
        "Gray cells have no score. "
        f"{saturated} values outside [-1, 1] use the endpoint colors and retain their numeric labels. "
        f"Compound key: {key}."
    )

    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(6.5, 2.1), layout="constrained")
    for index, (axis, values) in enumerate(zip(axes, panels, strict=True)):
        axis.set_facecolor("#dddddd")
        sns.heatmap(
            values,
            ax=axis,
            cmap="RdBu_r" if index < 2 else "PuOr_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6.5},
            cbar=False,
            linewidths=0.35,
            linecolor="white",
        )
        axis.set(xlabel="", ylabel="")
        axis.tick_params(axis="both", labelsize=8, length=0)
        axis.tick_params(axis="y", labelleft=index == 0, labelrotation=0)
        axis.tick_params(axis="x", labelrotation=0)

    for axis, group, label, aspect in (
        (axes[0], axes[:2], "Normalized mAP", 40),
        (axes[2], axes[2], "Difference in normalized mAP", 20),
    ):
        colorbar = figure.colorbar(
            axis.collections[0],
            ax=group,
            location="bottom",
            shrink=0.9,
            pad=0.12,
            aspect=aspect,
            ticks=[-1, -0.5, 0, 0.5, 1],
        )
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label(label, fontsize=8)

    output.mkdir(parents=True, exist_ok=True)
    attach_caption(figure, caption)
    figure.savefig(output / "poscon_heatmap.png", dpi=dpi)
    figure.savefig(output / "poscon_heatmap.pdf")
    plt.close(figure)

    columns = ["Metadata_Source", "Metadata_JCP2022", "nmAP_within", "nmAP_cross"]
    scores = scores[columns].assign(
        compound_name=scores["Metadata_JCP2022"].map(names),
        compound_code=scores["Metadata_JCP2022"].map(codes),
        nmAP_difference=scores["nmAP_within"] - scores["nmAP_cross"],
    )
    scores.to_csv(output / "poscon_heatmap.csv", index=False)
    write_legend(
        output / "legend.md",
        figure_id=FIGURE_ID,
        title=FIGURE_TITLE,
        sections=[("poscon_heatmap.png", FIGURE_TITLE, caption)],
    )
    write_summary(
        output / "summary.json",
        description=FIGURE_TITLE,
        numbers={
            "dataset": DATASET,
            "preprocessing": PREPROCESSING,
            "filter": FILTER,
            "compounds": len(compounds),
            "sources": len(sources),
            "matched_compound_source_pairs": len(scores),
            "values_outside_color_range": saturated,
        },
        files=["poscon_heatmap.png", "poscon_heatmap.pdf", "poscon_heatmap.csv", "legend.md"],
    )


if __name__ == "__main__":
    typer.run(main)
