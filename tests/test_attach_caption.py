"""Contract tests for the caption band attached to a finished figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from jump_production.visualization import attach_caption  # noqa: E402

CAPTION = "word " * 60


def build_figure() -> plt.Figure:
    """Return a constrained-layout figure carrying both figure-legend anchoring styles."""
    figure, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), constrained_layout=True)
    for axis in axes:
        axis.plot([0, 1], [0, 1])
    handles = [Line2D([], [], marker="o", linestyle="", label="thing")]
    figure.legend(handles=handles, loc="lower left")
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    figure.suptitle("Title")
    return figure


def anchors(figure: plt.Figure) -> list[tuple[float, float, float, float]]:
    return [legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted()).bounds for legend in figure.legends]


def test_caption_grows_the_canvas_without_shrinking_the_panels() -> None:
    figure = build_figure()
    try:
        # Settle the layout engine first: before it runs, the positions are placeholder defaults.
        figure.draw_without_rendering()
        width_in, height_in = figure.get_size_inches()
        panel_heights_in = [axis.get_position().height * height_in for axis in figure.axes]

        attach_caption(figure, CAPTION)

        new_width_in, new_height_in = figure.get_size_inches()
        assert new_width_in == width_in
        assert new_height_in > height_in
        for axis, before_in in zip(figure.axes, panel_heights_in, strict=True):
            assert axis.get_position().height * new_height_in == pytest.approx(before_in)
    finally:
        plt.close(figure)


def test_a_figure_anchored_legend_keeps_its_whole_rectangle() -> None:
    """A legend placed by `loc` alone anchors to the figure; collapsing that to its corner
    threw it off the left edge."""
    figure = build_figure()
    try:
        attach_caption(figure, CAPTION)

        by_loc, by_anchor = anchors(figure)
        assert by_loc[2] == pytest.approx(1.0)
        assert by_loc[0] == pytest.approx(0.0)
        assert by_loc[1] > 0.0
        assert by_anchor[0] == pytest.approx(0.98)
        assert by_anchor[1] > 0.98
    finally:
        plt.close(figure)


def test_the_caption_and_the_title_land_on_opposite_edges() -> None:
    figure = build_figure()
    try:
        attach_caption(figure, CAPTION)

        title, caption = figure.texts[0], figure.texts[-1]
        assert title.get_text() == "Title"
        assert title.get_position()[1] > 0.9
        assert caption.get_position()[1] < 0.1
        assert caption.get_text().splitlines()[0].startswith("word")
    finally:
        plt.close(figure)
