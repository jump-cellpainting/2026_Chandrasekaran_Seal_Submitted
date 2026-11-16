# Figure review checklist

Use only the sections relevant to the requested change.

## Evidence

- Trace plotted values to the canonical input or owning analysis output.
- Use `mean_normalized_average_precision` for copairs comparisons and `corrected_p_value` for stored activity decisions.
- State the actual multiple-testing procedure, threshold, denominator, eligible population, and missing-data treatment.
- Distinguish an exhaustive rule from a curated display selection.
- Keep expensive or reused computation in a named analysis cache; keep ordinary figure preparation in the descriptive notebook script.

## Rendering contract

Paper figures are ordinary scripts under `notebooks/` that read canonical processed inputs and write PNG/PDF views, tables, `legend.md`, and a standard `summary.json` directly under `data/processed/figures/<figure-id>/`.
Build and validate one exact figure ID with:

```bash
direnv exec . just build-figure figure_suppl_structure
direnv exec . just validate-figure figure_suppl_structure
```

Figure validation requires the directory to contain exactly `summary.json` and the files it declares.
Undeclared legacy files and `_proto` scratch fail validation.

These rules apply only when generating, editing, or reviewing figures.
They do not apply to numerical analyses, tables, or manuscript prose.

- An attached figure and a plain request such as "fix this legend" are sufficient.
  When a path is not given, locate the responsible renderer from visible labels, panel titles, output filenames, and figure manifests before asking the user to identify the file.
- Legends, colorbars, and annotation boxes must not obscure plotted data.
- Prefer direct labels or dedicated space outside the data axes.
  An inside-axes legend is acceptable only after visual inspection confirms that it occupies an empty region at both original resolution and likely manuscript size.
- If no safe region exists, enlarge or rearrange the panel, or add a dedicated legend axis.
  Do not make an overlay translucent and leave it covering data.
- An in-figure caption is transport only: it rides into a drawing tool and is cropped off.
  Size the panels first; let the caption wrap to that width and grow the canvas downward.
- Every rendered figure file carries exactly one caption, built once from computed values.
  That same string is drawn under the figure and is the body of that file's `legend.md` section.
  Never author legend prose separately from the caption, and attach prose that belongs to no single file to the file it is most about.
  A caption is raw material for writing up rather than final text, so a long one is fine and a divergent one is not.
- `legend.md` is `# <figure_id>: <title>` followed by one ``## <Section title> (`<file>.png`)`` per rendered file and nothing else. Write it with `write_legend` in `src/jump_production/summaries.py`, passing the same caption strings the renderers drew.
- Omit panel titles.
  Put them in the caption instead, keyed by position ("Top,", "Bottom,").
- Never name a figure by manuscript number or panel letter, in a caption, a heading, a cross-reference, or a summary description.
  Manuscript numbering drifts and filenames do not, so refer to files by name and to panels by position.
- Inspect every changed PNG and PDF at original resolution and likely manuscript width before considering the figure complete.
- Presentation-only changes must not alter data, estimates, intervals, thresholds, denominators, eligibility rules, or statistical results.
- Matplotlib style sheets and shared plotting helpers may define mechanical defaults such as fonts, colors, line widths, and legend appearance.
  Legend placement remains a per-figure layout decision because a style sheet cannot determine where the plotted data are.

## Legend and visual inspection

- Name profile level, distance or similarity, thresholds, selection rules, and important exclusions.
- Separate what the figure shows from what it suggests.
- Inspect each changed raster at original resolution and likely manuscript width.
- Check titles, category labels, ticks, legends, annotations, gutters, and margins for clipping or collisions.
- Disclose clipped viewports and saturated scales, including included and excluded counts.
- Preserve deterministic plotting order, table order, label maps, palettes, and seeds where they matter.

## Validation

- Run focused behavioral tests for changed scientific contracts.
- Confirm the named Snakemake target resolves and declares every public artifact.
- Confirm `summary.json` uses relative paths and reports the important counts and configuration.
- Run the full repository suite once at the end of a broad migration, not after each figure edit.
