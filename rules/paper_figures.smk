# -*- coding: utf-8 -*-
# Paper figures assembled from pipeline panels. An opt-in target: nothing here is part of `all`,
# of `just reproduce` or of the results catalog, and nothing is written under data/processed/.
#
# A layout in configs/paper_figures/ names every panel of one figure by its source file. The rules
# below produce those sources in three ways:
#   data/processed/...               ordinary pipeline outputs, used as they are (clipped if needed)
#   data/paper_figures/panels/...    the figure renderers run again with JUMP_FIGURE_NO_CAPTION=1,
#                                    so the panel carries no caption band, the tables and plots drawn
#                                    by notebooks/paper-figure-tables.py, and print-resolution copies
#                                    of analysis plots
#   assets/paper_figures/...         versioned inputs that no rule can regenerate
# Outputs of the panel rules are read off the layouts, so a layout cannot name a panel that no rule
# writes without failing the dry run.

import glob

import yaml
from snakemake.exceptions import WorkflowError

from jump_production.workflow import matches_output

PAPER_FIGURES = "data/paper_figures"
PAPER_LAYOUTS = {
    Path(path).stem: path for path in sorted(glob.glob("configs/paper_figures/figure_paper_*.yaml"))
}


def paper_sources(layout_path):
    """Every panel source one layout names, in reading order."""
    with open(layout_path) as stream:
        layout = yaml.safe_load(stream)
    sources = []
    for row in layout["rows"]:
        for cell in row:
            sources += [panel["src"] for panel in (cell if isinstance(cell, list) else [cell])]
    return list(dict.fromkeys(sources))


def paper_panels(folder, summary=True):
    """The files under one panel folder that some layout uses, plus that folder's envelope."""
    prefix = f"{PAPER_FIGURES}/panels/{folder}/"
    used = sorted({src for path in PAPER_LAYOUTS.values() for src in paper_sources(path) if src.startswith(prefix)})
    return used + ([f"{prefix}summary.json"] if summary else [])


def paper_inputs(wildcards):
    """What one figure waits for: each source a rule declares, else the marker of the analysis that writes it.

    Analyses declare a summary or a completion marker, not every plot. Naming an undeclared plot as
    an input would stop the workflow in a checkout where it has not been built yet, so the figure
    waits for the marker instead and the assembler reports a source that is still missing.
    """
    declared = [str(output) for rule in workflow.rules for output in rule.output]

    def is_declared(path):
        return any(matches_output(pattern, path) for pattern in declared)

    inputs = []
    for src in paper_sources(PAPER_LAYOUTS[wildcards.figure_id]):
        if not src.startswith("data/processed/") or is_declared(src):
            inputs.append(src)
            continue
        folder = Path(src).parent
        markers = [folder / "summary.json", folder / ".complete", folder.parent / ".complete"]
        marker = next((str(m) for m in markers if is_declared(str(m))), None)
        if marker is None:
            raise WorkflowError(f"{src}: no rule declares this file or a marker for its analysis")
        inputs.append(marker)
    return list(dict.fromkeys(inputs))


# A caption-free render repeats a figure rule's command with a different output directory, so each
# rule below takes that rule's inputs whole. The caption text is still computed and written to the
# panel folder's summary.json, which is where the assembled figure's legend reads it from.
NO_CAPTION = "JUMP_FIGURE_NO_CAPTION=1"
# Print resolution for raster panels and for the assembled PNG; the pipeline default is a screen resolution.
PAPER_DPI = 300


rule paper_panels_compound_collection:
    input:
        **rules.figure_compound_collection.input
    output:
        paper_panels("figure_compound_collection")
    log:
        "logs/paper_figures/panels_compound_collection.log"
    shell:
        "{NO_CAPTION} pixi run -e cheminformatics python {input.script} --publication-only "
        "--publication-figure figure_compound_collection --output {PAPER_FIGURES}/panels/figure_compound_collection "
        "--dataset-structure-asset {input.dataset_structure_asset} --dpi {PAPER_DPI} 2>&1 | tee {log}"


rule paper_panels_suppl_chemical_properties:
    input:
        **rules.figure_suppl_chemical_properties.input
    output:
        paper_panels("figure_suppl_chemical_properties")
    log:
        "logs/paper_figures/panels_suppl_chemical_properties.log"
    shell:
        "{NO_CAPTION} pixi run -e cheminformatics python {input.script} --publication-only "
        "--publication-figure figure_suppl_chemical_properties "
        "--output {PAPER_FIGURES}/panels/figure_suppl_chemical_properties "
        "--structure-umap-input {input.structure_umap} --dpi {PAPER_DPI} 2>&1 | tee {log}"


rule paper_panels_phenotypic_consistency:
    input:
        **rules.figure_phenotypic_consistency.input
    output:
        paper_panels("figure_phenotypic_consistency")
    log:
        "logs/paper_figures/panels_phenotypic_consistency.log"
    shell:
        "{NO_CAPTION} pixi run python {input.script} --well-umap {input.well_umap} "
        "--perturbation-umap {input.perturbation_umap} "
        "--perturbation-profiles {input.perturbation_profiles} --metadata-db {input.metadata_db} "
        "--copairs-db {input.copairs_db} --output-dir {PAPER_FIGURES}/panels/figure_phenotypic_consistency "
        "--dataset compound_no_source7 --activity-preprocessing activity_no_target2 "
        "--activity-filter all_sources --metric cosine --dpi {PAPER_DPI} 2>&1 | tee {log}"


rule paper_panels_compound_activity_associations:
    input:
        **rules.figure_compound_activity_associations.input
    output:
        paper_panels("figure_compound_activity_associations")
    log:
        "logs/paper_figures/panels_compound_activity_associations.log"
    shell:
        "{NO_CAPTION} pixi run -e cheminformatics python {input.script} --dataset {DEFAULT_DATASET} "
        "--functional-use $(dirname {input.functional_summary}) "
        "--trait-enrichment $(dirname {input.trait_summary}) "
        "--property-oddsratios $(dirname {input.property_summary}) "
        "--target-alignment $(dirname {input.alignment_summary}) "
        "--output {PAPER_FIGURES}/panels/figure_compound_activity_associations --dpi 600 2>&1 | tee {log}"


rule paper_panels_suppl_structure:
    input:
        **rules.figure_suppl_structure.input
    output:
        paper_panels("figure_suppl_structure")
    log:
        "logs/paper_figures/panels_suppl_structure.log"
    shell:
        "{NO_CAPTION} pixi run -e cheminformatics python {input.script} "
        "--pair-cache-summary {input.pair_cache_summary} "
        "--substructure-summary {input.substructure_summary} "
        "--scaffold-coherence {input.scaffold_coherence} "
        "--scaffold-series-summary {input.scaffold_series_summary} "
        "--scaffold-analysis {input.scaffold_analysis} --cliff-scaffolds {input.cliff_scaffolds} "
        "--output {PAPER_FIGURES}/panels/figure_suppl_structure --dpi {PAPER_DPI} 2>&1 | tee {log}"


rule paper_panels_positive_control_reproducibility:
    input:
        **rules.figure_positive_control_reproducibility.input
    output:
        paper_panels("figure_positive_control_reproducibility")
    log:
        "logs/paper_figures/panels_positive_control_reproducibility.log"
    shell:
        "{NO_CAPTION} pixi run python {input.script:q} --copairs-db {input.copairs_db:q} "
        "--metadata-db {input.metadata_db:q} "
        "--output {PAPER_FIGURES}/panels/figure_positive_control_reproducibility "
        "--dpi {PAPER_DPI} 2>&1 | tee {log:q}"


rule paper_panels_cross_source_reproducibility:
    """Vector copies of the cross-source plots; the analysis itself writes PNG only."""
    input:
        **rules.cross_source_reproducibility.input
    output:
        paper_panels("cross-source-reproducibility", summary=False),
        expand(
            f"{PAPER_FIGURES}/panels/cross-source-reproducibility/{{subdir}}/summary.json",
            subdir=[combo[2] for combo in CROSS_SOURCE_COMBOS],
        )
    params:
        combos=" ".join(f"{p}:{f}" for p, f, _ in CROSS_SOURCE_COMBOS),
    log:
        "logs/paper_figures/panels_cross_source_reproducibility.log"
    shell:
        """
        for combo in {params.combos}; do
            IFS=':' read -r preprocessing filter_name <<< "$combo"
            pixi run python {input.script} --dataset {DEFAULT_DATASET} --preprocessing "$preprocessing" \
                --filter "$filter_name" --output {PAPER_FIGURES}/panels/cross-source-reproducibility \
                --pdf --dpi {PAPER_DPI}
        done 2>&1 | tee {log}
        """


rule paper_panels_cell_density:
    """Print-resolution copy of the cell-density plots; the analysis writes them at a screen resolution."""
    input:
        **rules.data_quality.input
    output:
        paper_panels("data-quality")
    params:
        dataset=DEFAULT_DATASET,
        preprocessing=DEFAULT_PREPROCESSING,
        filter_name=DEFAULT_FILTER,
        activity_params=DEFAULT_ACTIVITY_PARAMS,
    log:
        "logs/paper_figures/panels_cell_density.log"
    shell:
        "pixi run python {input.script} --dataset {params.dataset} --preprocessing {params.preprocessing} "
        "--filter {params.filter_name} --activity-params {params.activity_params} "
        "--output {PAPER_FIGURES}/panels/data-quality --dpi {PAPER_DPI} 2>&1 | tee {log}"


rule paper_panels_tables:
    input:
        activity=f"data/processed/profile-comparison/{DEFAULT_DATASET}_vs_compound_DL_CPCNN_no_source7/activity/default/summary.json",
        consistency="data/processed/profile-comparison/.complete",
        batch_effect="data/processed/batch-effect/batch_effect_summary.csv",
        cosine=f"data/processed/exploration/0.01/{DEFAULT_DATASET}/summary.json",
        trios=f"data/processed/polypharmacology/{DEFAULT_DATASET}/trio_categories.csv",
        classification=f"data/processed/toxicity-pk-prediction/{DEFAULT_DATASET}/classification_results.csv",
        regression=f"data/processed/toxicity-pk-prediction/{DEFAULT_DATASET}/regression_results.csv",
        script="notebooks/paper-figure-tables.py",
        helpers=FIGURE_HELPERS
    output:
        paper_panels("tables")
    log:
        "logs/paper_figures/panels_tables.log"
    shell:
        "pixi run python {input.script} --processed data/processed "
        "--output {PAPER_FIGURES}/panels/tables 2>&1 | tee {log}"


rule paper_figure:
    """Assemble one figure package from its layout. PyMuPDF comes from uv, not the pinned environments."""
    input:
        layout=lambda wildcards: PAPER_LAYOUTS[wildcards.figure_id],
        sources=paper_inputs,
        script="notebooks/paper-figure-assembly.py",
        validator="scripts/validate_figure.sh"
    output:
        png=f"{PAPER_FIGURES}/figures/{{figure_id}}/assembled.png",
        pdf=f"{PAPER_FIGURES}/figures/{{figure_id}}/assembled.pdf",
        legend=f"{PAPER_FIGURES}/figures/{{figure_id}}/legend.md",
        summary=f"{PAPER_FIGURES}/figures/{{figure_id}}/summary.json"
    wildcard_constraints:
        figure_id="figure_paper_[a-z_]+"
    log:
        "logs/paper_figures/{figure_id}.log"
    shell:
        "(uv run --script {input.script} compose {input.layout} --output-root {PAPER_FIGURES}/figures "
        "--dpi {PAPER_DPI} && PROCESSED_DIR={PAPER_FIGURES} bash {input.validator} {wildcards.figure_id}) "
        "2>&1 | tee {log}"


rule paper_figures:
    """Assemble every paper figure that has a layout, and describe the tree for readers."""
    input:
        summaries=expand(f"{PAPER_FIGURES}/figures/{{figure_id}}/summary.json", figure_id=PAPER_LAYOUTS),
        layouts=list(PAPER_LAYOUTS.values())
    output:
        readme=f"{PAPER_FIGURES}/README.md"
    run:
        lines = [
            "# Assembled paper figures",
            "",
            "Multi-panel paper figures assembled in code from pipeline panels.",
            "Each folder under `figures/` holds `assembled.png`, `assembled.pdf`, a `legend.md` that names",
            "the source of every panel by position, and a `summary.json`.",
            "`panels/` holds the caption-free panel renders and tables the figures are assembled from.",
            "The layouts and the method are in `configs/paper_figures/` and",
            "`docs/paper/figure_assembly.md` of the analysis repository.",
            "",
        ]
        for figure_id, path in PAPER_LAYOUTS.items():
            with open(path) as stream:
                lines.append(f"- `figures/{figure_id}/`: {yaml.safe_load(stream)['title']}")
        with open(output.readme, "w") as stream:
            stream.write("\n".join(lines) + "\n")
