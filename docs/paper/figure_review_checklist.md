# Figure review checklist

These rules apply when generating, editing or reviewing figures, not to numerical analyses, tables or manuscript prose.
Use the sections relevant to the change.

An attached figure and a plain request such as "fix this legend" are sufficient.
When no path is given, locate the renderer from visible labels, output file names and figure summaries before asking for it.

## Build and validate

Paper figures are ordinary scripts under `notebooks/` that read canonical processed inputs and write PNG and PDF views, tables, `legend.md` and a standard `summary.json` directly under `data/processed/figures/<figure-id>/`.

```bash
direnv exec . just build-figure figure_suppl_structure
direnv exec . just validate-figure figure_suppl_structure
```

- Validation requires the directory to contain exactly `summary.json` and the files it declares; undeclared legacy files and `_proto` scratch fail.
- Confirm that the named Snakemake target resolves and declares every public artifact.
- Confirm that `summary.json` uses relative paths and reports the important counts and configuration.
- Run focused behavioral tests for changed scientific contracts, and the full suite once at the end of a broad migration, not after each figure edit.

## Evidence

- Trace plotted values to the canonical input or the output of the analysis that owns them.
- Use `mean_normalized_average_precision` for copairs comparisons and `corrected_p_value` for stored activity decisions.
- State the actual multiple-testing procedure, threshold, denominator, eligible population and missing-data treatment.
- Distinguish an exhaustive rule from a curated display selection.
- Keep expensive or reused computation in a named analysis cache and ordinary figure preparation in the figure script.
- Presentation-only changes must not alter data, estimates, intervals, thresholds, denominators, eligibility rules or statistical results.
- Preserve deterministic plotting order, table order, label maps, palettes and seeds where they matter.

## Captions and legends

- Every rendered figure file carries exactly one caption, built once from computed values.
  That string is drawn under the figure and is the body of that file's `legend.md` section.
  Never write legend prose separately from the caption, and attach prose that belongs to no single file to the file it is most about.
- `legend.md` is `# <figure_id>: <title>` followed by one ``## <Section title> (`<file>.png`)`` per rendered file and nothing else. Write it with `write_legend` in `src/jump_production/summaries.py`, passing the caption strings the renderers drew.
- An in-figure caption is transport only: it rides into a drawing tool and is cropped off.
  Size the panels first, let the caption wrap to that width and grow the canvas downward.
  A caption is raw material for writing up, so a long one is fine and a divergent one is not.
- Omit panel titles; put them in the caption, keyed by position ("Top,", "Bottom,").
- Never name a figure by manuscript number or panel letter, in a caption, heading, cross-reference or summary description.
  Manuscript numbering drifts and file names do not, so refer to files by name and to panels by position.
- Name the profile level, distance or similarity, thresholds, selection rules and important exclusions.
- Separate what the figure shows from what it suggests.
- Disclose clipped viewports and saturated scales, with included and excluded counts.

## Layout and inspection

- Legends, colorbars and annotation boxes must not obscure plotted data.
- Prefer direct labels or dedicated space outside the data axes.
  A legend inside the axes is acceptable only after inspection confirms that it sits in an empty region at both original resolution and likely manuscript size.
- If no safe region exists, enlarge or rearrange the panel or add a dedicated legend axis.
  Do not make an overlay translucent and leave it covering data.
- Style sheets and shared plotting helpers may set mechanical defaults such as fonts, colors, line widths and legend appearance.
  Legend placement stays a per-figure decision, because a style sheet cannot know where the data are.
- Inspect every changed PNG and PDF at original resolution and at likely manuscript width.
  Check titles, category labels, ticks, legends, annotations, gutters and margins for clipping or collisions.
