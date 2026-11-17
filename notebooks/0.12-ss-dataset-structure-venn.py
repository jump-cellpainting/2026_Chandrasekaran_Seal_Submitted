# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb",
#     "matplotlib",
#     "matplotlib-venn",
#     "pandas",
# ]
# ///
"""Figure 1A dataset structure - regenerate the numbers, draw the Venn diagrams.

Self-contained: locates the augmented metadata DB from the repo root, queries it live, and
draws area-proportional Venn diagrams. It needs only PyPI packages, so `--sandbox` works.

Dual-use:
- Interactive view:  uv run --with marimo marimo run --sandbox notebooks/0.12-ss-dataset-structure-venn.py
- Emit artifact:     uv run --with marimo --with duckdb --with matplotlib --with matplotlib-venn \
                         --with pandas python notebooks/0.12-ss-dataset-structure-venn.py

Either path writes the persistent JSON artifact to
`data/processed/dataset_structure/dataset_structure.json`, the machine-readable record of every
regenerated number (headline counts, both wave-overlap framings, the ORF/CRISPR gene overlap,
per-site counts, and replicate breadth).
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import duckdb
    import matplotlib.pyplot as plt
    import pandas as pd

    # Venn rendering is optional: the JSON emit (this notebook's pipeline output) needs no plotting.
    # When run in the `marimo` pixi env without matplotlib-venn the figures are skipped but the
    # artifact still writes; the sandbox env installs it via the PEP 723 header and renders them.
    try:
        from matplotlib_venn import venn2

        HAS_VENN = True
    except ImportError:
        venn2 = None
        HAS_VENN = False

    # Repo root = notebooks/.. ; no jump_production import so this runs in an isolated sandbox env.
    REPO = Path(__file__).resolve().parents[1]
    METADATA_DB = REPO / "data" / "interim" / "jump_metadata_augmented.duckdb"
    # Full-resource per-well cell counts (all sources incl. ORF + CRISPR). The augmented DB's
    # cell_counts table only carries the 10 compound sources; this CSV is the whole release.
    ALL_CELL_COUNTS = REPO / "data" / "external" / "all_cell_counts.csv.gz"

    # Dedicated output folder - the JSON artifact downstream code reads programmatically.
    OUT_DIR = REPO / "data" / "processed" / "dataset_structure"
    OUT_JSON = OUT_DIR / "dataset_structure.json"
    # Standard summary envelope ({description, numbers, files}) so the dataset-reference hub
    # discovers this node like any other. Written inline (not via jump_production.summaries) to
    # keep the notebook importable in an isolated --sandbox env.
    OUT_SUMMARY = OUT_DIR / "summary.json"

    # Wave membership and special sites - read off Figure 1A and the Methods (Experimental Design).
    # Wave 1 = 7 partners, Wave 2 = 3 partners. S7 (lower-dose bioactives, excluded from analysis)
    # and S4 (Broad pilots/ORFs) sit outside the two waves.
    WAVE1 = ["source_1", "source_2", "source_3", "source_6", "source_8", "source_10", "source_15"]
    WAVE2 = ["source_5", "source_9", "source_11"]

    # Colors keyed to the Figure 1A legend (Compound / ORF / CRISPR).
    COMPOUND_COLOR = "#5b9bd5"
    ORF_COLOR = "#70ad47"
    CRISPR_COLOR = "#ed7d31"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 1A - dataset structure, regenerated from the metadata DB

    **What this is.**

    Data and visualization of how `cpg0016-jump` is built from partner sites and how it overlaps the genetic-perturbation data.
    It reads from `data/interim/jump_metadata_augmented.duckdb`.

    **Two compound framings - do not mix them:**

    - **nominated** (`compound_source`): which site *contributed* a compound.
    - **screened** (`well` x `perturbation`): which site *imaged* a compound.

    Note that `cpg0012` (30,616 bioactive compounds, Bray et al. 2017) is an external dataset, not in this DB, and cannot be obtained through a query.
    """)
    return


@app.function
def connect():
    """Read-only connection to the augmented metadata DB."""
    return duckdb.connect(str(METADATA_DB), read_only=True)


@app.function
def nominated_compounds(sources):
    """JCP IDs nominated by any of `sources` (compound provenance / exchange frame)."""
    con = connect()
    placeholders = ", ".join(["?"] * len(sources))
    rows = con.execute(
        f"SELECT DISTINCT Metadata_JCP2022 FROM compound_source "
        f"WHERE Metadata_Compound_Source IN ({placeholders})",
        sources,
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


@app.function
def screened_compounds(sources):
    """JCP IDs imaged at any of `sources` (well-level screening frame, compounds only)."""
    con = connect()
    placeholders = ", ".join(["?"] * len(sources))
    rows = con.execute(
        f"SELECT DISTINCT w.Metadata_JCP2022 FROM well w "
        f"JOIN plate pl USING (Metadata_Source, Metadata_Plate) "
        f"JOIN perturbation p USING (Metadata_JCP2022) "
        f"WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' "
        f"AND w.Metadata_Source IN ({placeholders})",
        sources,
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


@app.function
def gene_set(table):
    """Distinct NCBI gene IDs for a genetic-perturbation table ('orf' or 'crispr')."""
    con = connect()
    rows = con.execute(
        f"SELECT DISTINCT Metadata_NCBI_Gene_ID FROM {table} "
        f"WHERE Metadata_NCBI_Gene_ID IS NOT NULL"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


@app.function
def plot_venn2(set_a, set_b, label_a, label_b, color_a, color_b, title):
    """Area-proportional 2-set Venn with raw counts in each region."""
    fig, ax = plt.subplots(figsize=(6, 5))
    v = venn2(
        [set_a, set_b],
        set_labels=(label_a, label_b),
        set_colors=(color_a, color_b),
        alpha=0.55,
        ax=ax,
    )
    for region in ("10", "01", "11"):
        lbl = v.get_label_by_id(region)
        if lbl is not None:
            lbl.set_text(f"{int(lbl.get_text()):,}")
    ax.set_title(title)
    fig.tight_layout()
    return fig


@app.function
def build_reference():
    """Recompute every structural number into a plain JSON-serializable dict.
    """
    con = connect()
    # Imaging totals from per-source microscope config x well counts. Image count = wells x sites x
    # 5 fluorescent Cell Painting channels (matches the paper's 43,072,410 exactly). Data size is a
    # derived estimate at 16-bit/pixel, bracketed by min/max brightfield z-planes per source; the
    # released ~127 TB (Cell Painting Gallery, cpg0016-jump) falls inside this range.
    _img = con.execute(
        "WITH w AS (SELECT Metadata_Source, COUNT(*) n_wells FROM well GROUP BY 1) "
        "SELECT SUM(w.n_wells * m.Metadata_Sites_Per_Well * 5), "
        "  SUM(w.n_wells * m.Metadata_Sites_Per_Well * (5 + COALESCE(m.Metadata_N_Brightfield_Planes_Min,0)) "
        "      * m.Metadata_Image_Width_Pixels * m.Metadata_Image_Height_Pixels * 2), "
        "  SUM(w.n_wells * m.Metadata_Sites_Per_Well * (5 + COALESCE(m.Metadata_N_Brightfield_Planes_Max,0)) "
        "      * m.Metadata_Image_Width_Pixels * m.Metadata_Image_Height_Pixels * 2) "
        "FROM microscope_config m JOIN w USING (Metadata_Source)"
    ).fetchone()
    headline = {
        "compounds_perturbation": int(
            con.execute(
                "SELECT COUNT(*) FROM perturbation WHERE Metadata_perturbation_modality = 'compound'"
            ).fetchone()[0]
        ),
        "compounds_nominated": int(
            con.execute("SELECT COUNT(DISTINCT Metadata_JCP2022) FROM compound_source").fetchone()[0]
        ),
        "tested_wells": int(con.execute("SELECT COUNT(*) FROM well").fetchone()[0]),
        # Full released resource: all sources incl. ORF + CRISPR (source_15 has no counts and is
        # excluded everywhere). This is the headline "1.8 billion single-cell profiles" number.
        "single_cell_profiles": int(
            con.execute(
                f"SELECT SUM(Metadata_Count_Cells) FROM read_csv_auto('{ALL_CELL_COUNTS}')"
            ).fetchone()[0]
        ),
        # Compound profiles only (the 10 compound sources in the cell_counts table), for the
        # compound-scoped framing.
        "single_cell_profiles_compound": int(
            con.execute("SELECT SUM(Metadata_Count_Cells) FROM cell_counts").fetchone()[0]
        ),
        "total_images": int(_img[0]),
        "data_size_tb_min": round(_img[1] / 1e12, 1),
        "data_size_tb_max": round(_img[2] / 1e12, 1),
        "compounds_excl_source7": int(
            con.execute(
                "SELECT COUNT(DISTINCT w.Metadata_JCP2022) FROM well w "
                "JOIN perturbation p USING (Metadata_JCP2022) "
                "WHERE p.Metadata_perturbation_modality = 'compound' AND w.Metadata_Source <> 'source_7'"
            ).fetchone()[0]
        ),
    }
    per_source = {
        r[0]: int(r[1])
        for r in con.execute(
            "SELECT w.Metadata_Source, COUNT(DISTINCT w.Metadata_JCP2022) "
            "FROM well w JOIN plate pl USING (Metadata_Source, Metadata_Plate) JOIN perturbation p USING (Metadata_JCP2022) "
            "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' "
            "GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }
    replicate_breadth = [
        {"n_sites": int(r[0]), "n_compounds": int(r[1])}
        for r in con.execute(
            "WITH c AS (SELECT w.Metadata_JCP2022, COUNT(DISTINCT w.Metadata_Source) nsrc "
            "FROM well w JOIN plate pl USING (Metadata_Source, Metadata_Plate) JOIN perturbation p USING (Metadata_JCP2022) "
            "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' GROUP BY 1) "
            "SELECT nsrc, COUNT(*) FROM c GROUP BY 1 ORDER BY 1"
        ).fetchall()
    ]
    con.close()

    def overlap(a, b):
        return {"a": len(a), "b": len(b), "intersection": len(a & b), "union": len(a | b)}

    orf_g, crispr_g = gene_set("orf"), gene_set("crispr")
    return {
        "source_db": str(METADATA_DB.relative_to(REPO)),
        "generated_by": "notebooks/0.12-ss-dataset-structure-venn.py",
        "screening_frame": "per_source_screened, replicate_breadth and wave_overlap.screened count COMPOUND-plate wells only (JUMP-Target-2 control plates excluded); headline counts are released-collection (all wells)",
        "wave1_sources": WAVE1,
        "wave2_sources": WAVE2,
        "headline": headline,
        "wave_overlap": {
            "nominated": overlap(nominated_compounds(WAVE1), nominated_compounds(WAVE2)),
            "screened": overlap(screened_compounds(WAVE1), screened_compounds(WAVE2)),
        },
        "gene_overlap": overlap(orf_g, crispr_g),
        "per_source_screened": per_source,
        "replicate_breadth": replicate_breadth,
    }


@app.cell(hide_code=True)
def _(mo):
    # Emit the persistent JSON artifact. Runs on every execution (interactive) and when the
    # notebook is run as a script: `python notebooks/0.12-ss-dataset-structure-venn.py`.
    reference = build_reference()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(reference, indent=2) + "\n")

    # Also emit the standard summary envelope for the dataset-reference hub. `numbers` carries the
    # quantitative facts (the metadata keys - source_db, wave membership, etc. - stay in the detailed
    # artifact above); `files` indexes the four panel figures the later cells render.
    _summary_numbers = {
        "headline": reference["headline"],
        "wave_overlap": reference["wave_overlap"],
        "gene_overlap": reference["gene_overlap"],
        "per_source_screened": reference["per_source_screened"],
        "replicate_breadth": reference["replicate_breadth"],
    }
    _fig_dir = OUT_DIR / "figures"
    # Numbers in the prose below are interpolated from the computed headline, never hand-typed, so the
    # description can't drift from the values it describes. ~127 TB is the external released size (Cell
    # Painting Gallery) that data_size_tb_min/max bracket, so it stays a literal reference.
    _h = reference["headline"]
    _summary = {
        "description": (
            "Figure 1A dataset structure, regenerated from the augmented metadata DB: headline "
            "collection counts (compounds in perturbation/nominated/screened framings, tested wells, "
            "single-cell profiles - single_cell_profiles is the full released resource across all "
            f"sources incl. ORF + CRISPR (~{_h['single_cell_profiles'] / 1e9:.2f}B); "
            "single_cell_profiles_compound is the 10 compound "
            f"sources only (~{_h['single_cell_profiles_compound'] / 1e9:.2f}B, the paper's "
            "compound-scoped number); source_15 has no cell counts "
            f"and is excluded from both; total_images = wells x sites x 5 channels = "
            f"{_h['total_images']:,} "
            "matching the paper, and data_size_tb_min/max bracket the released ~127 TB at 16-bit "
            "incl. brightfield z-planes), Wave 1 vs Wave 2 compound overlap under both nominated and "
            "screened framings, the ORF/CRISPR gene overlap, per-site screened compound counts, and "
            "replicate breadth (compounds by number of sites they were screened at)."
        ),
        "numbers": _summary_numbers,
        "files": [
            # Relative to the checkout, like every other envelope: no machine-specific prefix.
            {"path": (_fig_dir / name).relative_to(REPO).as_posix(), "kind": "figure", "caption": caption}
            for name, caption in (
                ("orf_crispr_venn.png", "ORF / CRISPR gene overlap"),
                ("upset_site_combinations.png", "site-combination structure (UpSet)"),
                ("compound_exchange.png", "compound exchange: nominated vs screened"),
                ("exchange_network.png", "compound-exchange network (directed)"),
            )
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(_summary, indent=2) + "\n")

    mo.vstack(
        [
            mo.md(
                f"## Persistent artifacts\n\nWrote `{OUT_JSON.relative_to(REPO)}` "
                f"and `{OUT_SUMMARY.relative_to(REPO)}` (hub envelope)"
            ),
            mo.md(f"```json\n{json.dumps(reference, indent=2)}\n```"),
        ]
    )
    return


@app.cell
def _(mo):
    frame = mo.ui.dropdown(
        options=["nominated (exchange frame)", "screened (imaging frame)"],
        value="nominated (exchange frame)",
        label="Compound framing for the Wave 1 / Wave 2 Venn",
    )
    frame
    return (frame,)


@app.cell(hide_code=True)
def _(frame, mo):
    # Wave 1 vs Wave 2 compound overlap - the compound-exchange Venn.
    _use_nom = frame.value.startswith("nominated")
    _f = nominated_compounds if _use_nom else screened_compounds
    w1, w2 = _f(WAVE1), _f(WAVE2)
    _md = mo.md(
        f"## Wave 1 / Wave 2 compound overlap\n\n"
        f"Wave 1 = {', '.join(s.replace('source_', 'S') for s in WAVE1)}; "
        f"Wave 2 = {', '.join(s.replace('source_', 'S') for s in WAVE2)}. "
        f"Overlap = **{len(w1 & w2):,}** compounds "
    )
    _body = (
        plot_venn2(
            w1,
            w2,
            f"Wave 1\n({len(WAVE1)} sites)",
            f"Wave 2\n({len(WAVE2)} sites)",
            COMPOUND_COLOR,
            "#c55a11",
            f"Compound overlap between waves ({'nominated' if _use_nom else 'screened'})",
        )
        if HAS_VENN
        else mo.md("_matplotlib-venn not installed - showing counts only._")
    )
    mo.vstack([_md, _body])
    return


@app.cell(hide_code=True)
def _(mo):
    # ORF vs CRISPR gene overlap.
    orf_genes, crispr_genes = gene_set("orf"), gene_set("crispr")
    _md = mo.md(
        f"## ORF / CRISPR gene overlap\n\n"
        f"ORF = **{len(orf_genes):,}** genes, CRISPR = **{len(crispr_genes):,}** genes, "
        f"intersection = **{len(orf_genes & crispr_genes):,}**, "
        f"union = **{len(orf_genes | crispr_genes):,}** "
    )
    _body = (
        plot_venn2(
            orf_genes,
            crispr_genes,
            "ORF\n(S4)",
            "CRISPR\n(S13)",
            ORF_COLOR,
            CRISPR_COLOR,
            "Genetic perturbations: genes targeted",
        )
        if HAS_VENN
        else mo.md("_matplotlib-venn not installed - showing counts only._")
    )
    if HAS_VENN:
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        _body.savefig(OUT_DIR / "figures" / "orf_crispr_venn.png", dpi=200, bbox_inches="tight")
    mo.vstack([_md, _body])
    return


@app.cell(hide_code=True)
def _(mo):
    # Per-source compound counts (circle sizes) and replicate breadth (modal site count).
    _con = connect()
    per_source = _con.execute(
        "SELECT w.Metadata_Source AS source, "
        "COUNT(DISTINCT w.Metadata_JCP2022) AS screened "
        "FROM well w JOIN plate pl USING (Metadata_Source, Metadata_Plate) JOIN perturbation p USING (Metadata_JCP2022) "
        "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' "
        "GROUP BY 1 ORDER BY screened DESC"
    ).df()
    breadth = _con.execute(
        "WITH c AS (SELECT w.Metadata_JCP2022, COUNT(DISTINCT w.Metadata_Source) nsrc "
        "FROM well w JOIN plate pl USING (Metadata_Source, Metadata_Plate) JOIN perturbation p USING (Metadata_JCP2022) "
        "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' GROUP BY 1) "
        "SELECT nsrc AS n_sites, COUNT(*) AS n_compounds FROM c GROUP BY 1 ORDER BY 1"
    ).df()
    _con.close()

    _mode_sites = int(breadth.loc[breadth["n_compounds"].idxmax(), "n_sites"])

    # Compare the empirical mode against the paper's reference of five sites.
    _paper_sites = 5
    _agreement = "matches" if _mode_sites == _paper_sites else f"differs from (data: {_mode_sites})"

    # Per-site screened counts, summarized as ~k ranges straight from the data. S15 is the
    # Wave 1 outlier (it screened far fewer than the other six), so report it separately.
    _ps = dict(zip(per_source["source"], per_source["screened"]))


    def _krange(srcs):
        _vals = [int(_ps[s]) for s in srcs if s in _ps]
        _lo, _hi = round(min(_vals) / 1000), round(max(_vals) / 1000)
        return f"~{_lo}k" if _lo == _hi else f"~{_lo}-{_hi}k"


    _w1_main_range = _krange([s for s in WAVE1 if s != "source_15"])
    _s15_range = _krange(["source_15"])
    _w2_range = _krange(WAVE2)
    _s7_range = _krange(["source_7"])

    fig_src, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(per_source["source"].str.replace("source_", "S"), per_source["screened"], color=COMPOUND_COLOR)
    ax1.set_ylabel("compounds screened")
    ax1.set_title("Compounds screened per site")
    ax1.tick_params(axis="x", rotation=45)
    ax2.bar(breadth["n_sites"].astype(str), breadth["n_compounds"], color=COMPOUND_COLOR)
    ax2.set_xlabel("number of sites a compound was screened at")
    ax2.set_ylabel("compounds")
    ax2.set_title(f"Replicate breadth (mode = {_mode_sites} sites)")
    fig_src.tight_layout()

    mo.vstack(
        [
            mo.md(
                "## Per-site counts and replicate breadth\n\n"
                f"Wave 1 sites screen {_w1_main_range} compounds each (S15 is the exception at {_s15_range}); "
                f"Wave 2 {_w2_range}; S7 {_s7_range} (excluded); "
                f"S4/S13 are the genetic-pilot sites. The replicate-breadth mode at {_mode_sites} sites "
                f"{_agreement} the paper\'s claim that "
                "\"each compound was typically screened in five replicates across five sites.\""
            ),
            fig_src,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    # UpSet data: per-compound set of screening sites, aggregated to site-combination counts.
    from collections import Counter as _Counter

    _con = connect()
    _rows = _con.execute(
        "SELECT w.Metadata_JCP2022, list(DISTINCT w.Metadata_Source) AS srcs "
        "FROM well w JOIN plate pl USING (Metadata_Source, Metadata_Plate) JOIN perturbation p USING (Metadata_JCP2022) "
        "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' GROUP BY 1"
    ).fetchall()
    _con.close()

    upset_counts = _Counter(tuple(sorted(s.replace("source_", "S") for s in r[1])) for r in _rows)
    upset_src_total = _Counter()
    for _c, _n in upset_counts.items():
        for _s in _c:
            upset_src_total[_s] += _n
    return upset_counts, upset_src_total


@app.cell(hide_code=True)
def _(mo, upset_counts):
    upset_min = mo.ui.slider(
        200, 5000, value=1000, step=200, label="min compounds per site-combination shown"
    )
    _site_count = {}
    for _c, _n in upset_counts.items():
        _site_count[len(_c)] = _site_count.get(len(_c), 0) + _n
    _mode_sites = max(_site_count, key=_site_count.get)
    _nsites = len({s for _c in upset_counts for s in _c})
    mo.vstack(
        [
            mo.md(
                "## Site-combination structure (UpSet)\n\n"
                f"A Venn cannot show {_nsites} sites, "
                "so this shows the actual set structure. Each top bar is the number of compounds screened "
                "at *exactly* that set of sites; the dot-matrix says which sites; the left bars are per-site "
                "totals. The exchange design shows up directly - one big Wave 2 bar (S5+S9+S11) and a band "
                f"of {_mode_sites}-site Wave 1 patterns.\n\n"
                "Scope: cpg0016 compound production set only. The cpg0000/0001/0002 Broad "
                "pilots and cpg0012 are separate datasets - not labeled in this DB and not part "
                "of the multi-site exchange, so they are left to the schematic."
            ),
            upset_min,
        ]
    )
    return (upset_min,)


@app.cell(hide_code=True)
def _(mo, upset_counts, upset_min, upset_src_total):
    import numpy as _np
    from matplotlib.gridspec import GridSpec as _GridSpec

    _thr = upset_min.value
    _items = sorted((kv for kv in upset_counts.items() if kv[1] >= _thr), key=lambda kv: -kv[1])
    _combos = [c for c, _ in _items]
    _sizes = [n for _, n in _items]
    _total = sum(upset_counts.values())

    if not _combos:
        _out = mo.md(f"_No site-combination has >= {_thr:,} compounds - lower the slider._")
    else:
        _sources = sorted({s for c in _combos for s in c}, key=lambda s: -upset_src_total[s])
        _ncol, _nrow = len(_combos), len(_sources)
        _y_of = {s: i for i, s in enumerate(_sources)}
        _x = _np.arange(_ncol)

        _fig = plt.figure(figsize=(max(11, _ncol * 0.62), 7))
        _gs = _GridSpec(2, 2, width_ratios=[1, 4.5], height_ratios=[3, 2.2], hspace=0.06, wspace=0.06)
        _ax_bar = _fig.add_subplot(_gs[0, 1])
        _ax_mat = _fig.add_subplot(_gs[1, 1])
        _ax_set = _fig.add_subplot(_gs[1, 0])

        _ax_bar.bar(_x, _sizes, color=COMPOUND_COLOR)
        for _xi, _s in zip(_x, _sizes):
            _ax_bar.text(_xi, _s, f"{_s/1000:.1f}k", ha="center", va="bottom", fontsize=7, rotation=0)
        _ax_bar.set_ylim(0, max(_sizes) * 1.22)
        _ax_bar.set_xlim(-0.6, _ncol - 0.4)
        _ax_bar.set_xticks([])
        _ax_bar.set_ylabel("compounds in combination")
        _ax_bar.spines[["top", "right"]].set_visible(False)

        for _yi in range(_nrow):
            _ax_mat.scatter(_x, [_yi] * _ncol, color="#dddddd", s=55, zorder=1)
        for _xi, _c in zip(_x, _combos):
            _ys = [_y_of[s] for s in _c]
            _ax_mat.scatter([_xi] * len(_ys), _ys, color="#222222", s=55, zorder=3)
            if len(_ys) > 1:
                _ax_mat.plot([_xi, _xi], [min(_ys), max(_ys)], color="#222222", lw=1.6, zorder=2)
        _ax_mat.set_xlim(-0.6, _ncol - 0.4)
        _ax_mat.set_ylim(-0.6, _nrow - 0.4)
        _ax_mat.invert_yaxis()
        _ax_mat.set_yticks(range(_nrow))
        _ax_mat.set_yticklabels(_sources)
        _ax_mat.set_xticks([])
        _ax_mat.set_xlabel("site combination (ordered by number of compounds)")

        _ax_set.barh(range(_nrow), [upset_src_total[s] for s in _sources], color="#9bbfe0")
        _ax_set.set_ylim(-0.6, _nrow - 0.4)
        _ax_set.invert_yaxis()
        _ax_set.invert_xaxis()
        _ax_set.set_yticks([])
        _ax_set.set_xlabel("per site")
        _ax_set.spines[["top", "left"]].set_visible(False)
        _pos = _ax_set.get_position()
        _ax_set.set_position([_pos.x0 - 0.02, _pos.y0, _pos.width, _pos.height])
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        _fig.savefig(OUT_DIR / "figures" / "upset_site_combinations.png", dpi=200, bbox_inches="tight")

        _cov = sum(_sizes)
        _out = mo.vstack(
            [
                mo.md(
                    f"Showing **{_ncol}** combinations (>= {_thr:,} compounds), covering "
                    f"**{_cov:,}** of {_total:,} compounds ({100 * _cov / _total:.0f}%). "
                    "S4/S13 (genetic-pilot sites) have no compound-production plates and do not appear."
                ),
                _fig,
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo, upset_counts):
    import numpy as _np
    from matplotlib.colors import LogNorm as _LogNorm

    # Compound exchange = the directed flow behind the schematic's arrows: for each compound,
    # who nominated it (compound_source) vs where it was screened (well).
    _con = connect()
    _xm = _con.execute(
        "SELECT cs.Metadata_Compound_Source AS nominator, w.Metadata_Source AS screener, "
        "COUNT(DISTINCT w.Metadata_JCP2022) AS n FROM compound_source cs "
        "JOIN well w USING (Metadata_JCP2022) "
        "JOIN plate pl USING (Metadata_Source, Metadata_Plate) "
        "JOIN perturbation p ON p.Metadata_JCP2022 = cs.Metadata_JCP2022 "
        "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' GROUP BY 1, 2"
    ).df()
    _con.close()
    _xm["nominator"] = _xm["nominator"].str.replace("source_", "S")
    _xm["screener"] = _xm["screener"].str.replace("source_", "S")
    _piv = _xm.pivot_table(index="nominator", columns="screener", values="n", fill_value=0).astype(int)

    _order = ["S1", "S2", "S3", "S6", "S8", "S10", "S15", "S5", "S9", "S11", "S7", "S4"]
    _rows = [s for s in _order if s in _piv.index]
    _cols = [s for s in _order if s in _piv.columns]
    _piv = _piv.reindex(index=_rows, columns=_cols).fillna(0).astype(int)
    _vals = _piv.values
    _masked = _np.ma.masked_where(_vals == 0, _vals)

    _fig, _ax = plt.subplots(figsize=(9, 7.5))
    _cmap = plt.cm.Blues.copy()
    _cmap.set_bad("white")
    _im = _ax.imshow(_masked, cmap=_cmap, norm=_LogNorm(vmin=1, vmax=_vals.max()), aspect="auto")
    _ax.set_xticks(range(len(_cols)), _cols)
    _ax.set_yticks(range(len(_rows)), _rows)
    _ax.set_xlabel("screened at (site)")
    _ax.set_ylabel("nominated by (site)")
    for _i in range(len(_rows)):
        for _j in range(len(_cols)):
            _v = _vals[_i, _j]
            if _v == 0:
                continue
            _t = f"{_v/1000:.0f}k" if _v >= 1000 else str(_v)
            _ax.text(_j, _i, _t, ha="center", va="center", fontsize=6,
                     color="white" if _im.norm(_v) > 0.5 else "#333333")
    _n1 = sum(s in ["S1", "S2", "S3", "S6", "S8", "S10", "S15"] for s in _rows)
    _ax.axhline(_n1 - 0.5, color="crimson", lw=1)
    _ax.axvline(_n1 - 0.5, color="crimson", lw=1)
    _ax.set_title("Compound exchange: nominated by (row) vs screened at (column)")
    _fig.colorbar(_im, ax=_ax, label="compounds (log scale)", shrink=0.7)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    _fig.savefig(OUT_DIR / "figures" / "compound_exchange.png", dpi=200, bbox_inches="tight")
    _W1L = ["S1", "S2", "S3", "S6", "S8", "S10", "S15"]
    _W2L = ["S5", "S11", "S9"]
    _diag = [int(_piv.loc[s, s]) for s in _W1L if s in _piv.index and s in _piv.columns]
    _w1_nom = int(_np.median(_diag)) if _diag else 0
    _withinw = [int(_piv.loc[a, b]) for a in _W1L for b in _W1L if a != b and a in _piv.index and b in _piv.columns]
    _within_med = int(_np.median(_withinw)) if _withinw else 0
    _sc = {}
    for _c, _n in upset_counts.items():
        _sc[len(_c)] = _sc.get(len(_c), 0) + _n
    _partners = max(_sc, key=_sc.get) - 1
    _cross = sum(_n for _c, _n in upset_counts.items() if (set(_c) & set(_W1L)) and (set(_c) & set(_W2L)))

    mo.vstack(
        [
            mo.md(
                "## Compound exchange\n\n"
                "The directed flow the schematic drew as arrows. Rows = the site that nominated a compound; "
                "columns = the sites where it was screened. The diagonal is home-site screening; the two "
                "blocks (red lines split Wave 1 from Wave 2 + S7) show each wave traded almost entirely "
                f"within itself - every Wave 1 site shared its ~{round(_w1_nom / 1000)}k compounds with "
                f"{_partners} Wave 1 partners. The "
                f"faint off-block cells are the {_cross:,} compounds shared across waves to tie the two together. "
                f"Log color scale so the cross-wave trickle stays visible next to the ~{round(_within_med / 1000)}k within-wave blocks."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    network_min = mo.ui.slider(
        10, 13000, value=1000, step=10, label="min compounds exchanged for an edge"
    )
    network_layout = mo.ui.dropdown(
        ["two clusters (by wave)", "circular (by wave)"],
        value="circular (by wave)",
        label="layout",
    )
    mo.vstack(
        [
            mo.md(
                "## Compound-exchange network (directed)\n\n"
                "An arrow A -> B means site A nominated compounds that were screened at site B; edge weight "
                "is the number of such compounds. Drop weak exchanges with the slider; switch node layout "
                "with the dropdown."
            ),
            network_min,
            network_layout,
        ]
    )
    return network_layout, network_min


@app.cell(hide_code=True)
def _(mo, network_layout, network_min, upset_counts, upset_src_total):
    import numpy as _np
    from matplotlib.patches import FancyArrowPatch as _FancyArrowPatch

    # Directed exchange: A (nominator) -> B (screener), off-diagonal of the nominator x screener matrix.
    _con = connect()
    _rows = _con.execute(
        "SELECT cs.Metadata_Compound_Source a, w.Metadata_Source b, COUNT(DISTINCT w.Metadata_JCP2022) n "
        "FROM compound_source cs JOIN well w USING (Metadata_JCP2022) "
        "JOIN plate pl USING (Metadata_Source, Metadata_Plate) "
        "JOIN perturbation p ON p.Metadata_JCP2022 = cs.Metadata_JCP2022 "
        "WHERE p.Metadata_perturbation_modality = 'compound' AND pl.Metadata_PlateType = 'COMPOUND' "
        "GROUP BY 1, 2"
    ).fetchall()
    _con.close()
    _edges = {(a.replace("source_", "S"), b.replace("source_", "S")): int(n) for a, b, n in _rows if a != b}

    _thr = network_min.value
    _shown = {ab: n for ab, n in _edges.items() if n >= _thr}

    if not _shown:
        _out = mo.md(f"_No exchange >= {_thr:,} compounds - lower the slider._")
    else:
        _W1 = ["S1", "S2", "S3", "S6", "S8", "S10", "S15"]
        _W2 = ["S5", "S11", "S9"]
        _present = {x for ab in _shown for x in ab}
        _order = [s for s in _W1 + _W2 + ["S7", "S4"] if s in _present] + sorted(
            _present - set(_W1 + _W2 + ["S7", "S4"])
        )
        def _ring(members, cx, cy, r, pos):
            _m = [s for s in members if s in _order]
            for _i, _s in enumerate(_m):
                _a = _np.pi / 2 - 2 * _np.pi * _i / max(1, len(_m))
                pos[_s] = (cx + r * _np.cos(_a), cy + r * _np.sin(_a))

        if network_layout.value == "circular (by wave)":
            _N = len(_order)
            _pos = {s: (_np.cos(_np.pi / 2 - 2 * _np.pi * i / _N),
                        _np.sin(_np.pi / 2 - 2 * _np.pi * i / _N)) for i, s in enumerate(_order)}
        else:  # two clusters (by wave)
            _pos = {}
            _ring(_W1, -0.85, 0.15, 0.6, _pos)
            _ring(_W2, 0.9, 0.15, 0.45, _pos)
            _others = [s for s in _order if s not in _W1 + _W2]
            for _i, _s in enumerate(_others):
                _pos[_s] = ((-0.3 + 0.6 * _i / max(1, len(_others) - 1)) if len(_others) > 1 else 0.0, -1.1)
        _maxw = max(_edges.values())
        _maxsrc = max(upset_src_total.values())
        _node_sz = {s: 250 + 1300 * upset_src_total.get(s, 0) / _maxsrc for s in _order}

        def _radius_pts(s):
            return (_node_sz[s] / _np.pi) ** 0.5 + .01  # marker radius in points + margin

        def _ncolor(s):
            return COMPOUND_COLOR if s in _W1 else "#c55a11" if s in _W2 else "#999999"

        _fig, _ax = plt.subplots(figsize=(8, 8))
        # Collapse reciprocal directions into one double-headed arrow; single-headed where one-way.
        _bypair = {}
        for (_a, _b), _n in _shown.items():
            _bypair.setdefault(tuple(sorted((_a, _b))), {})[(_a, _b)] = _n
        _wmax = max((sum(d.values()) for d in _bypair.values()), default=1)
        for (_u, _v), _dirs in _bypair.items():
            _ab, _ba = _dirs.get((_u, _v)), _dirs.get((_v, _u))
            _tot = (_ab or 0) + (_ba or 0)
            if _ab and _ba:
                _style, _src, _dst = "<|-|>", _u, _v
            elif _ab:
                _style, _src, _dst = "-|>", _u, _v
            else:
                _style, _src, _dst = "-|>", _v, _u
            _ax.add_patch(
                _FancyArrowPatch(
                    _pos[_src], _pos[_dst], connectionstyle="arc3,rad=0.0", arrowstyle=_style,
                    mutation_scale=13, lw=0.3 + 3 * _tot / _wmax, alpha=0.2 + 0.55 * _tot / _wmax,
                    color="#777777", shrinkA=_radius_pts(_src), shrinkB=_radius_pts(_dst), zorder=2,
                )
            )
        for _s in _order:
            _ax.scatter(*_pos[_s], s=_node_sz[_s], color=_ncolor(_s), edgecolor="white", lw=1.5, zorder=3)
            _ax.annotate(_s, _pos[_s], ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        _xs = [p[0] for p in _pos.values()]
        _ys = [p[1] for p in _pos.values()]
        _ax.set_xlim(min(_xs) - 0.35, max(_xs) + 0.35)
        _ax.set_ylim(min(_ys) - 0.35, max(_ys) + 0.35)
        _ax.set_aspect("equal")
        _ax.axis("off")
        _ax.set_title(f"Compound exchange (directed): arrow A->B = A nominated, B screened (>= {_thr:,})")
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        _fig.savefig(OUT_DIR / "figures" / "exchange_network.png", dpi=200, bbox_inches="tight")

        _cross = sum(_n for _c, _n in upset_counts.items() if (set(_c) & set(_W1)) and (set(_c) & set(_W2)))
        _out = mo.vstack(
            [
                mo.md(
                    f"Arrow **A -> B** = compounds A nominated that were screened at B; a **double-headed** "
                    f"arrow marks reciprocal exchange (both directions present). Width/opacity scale "
                    f"with total volume on the link. {len(_bypair)} site-pairs shown at >= {_thr:,}. "
                    "Wave 1 (blue) and Wave 2 (orange) exchange densely within wave; thin links crossing "
                    f"between them are the {_cross:,} bridge compounds."
                ),
                _fig,
            ]
        )
    _out
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
