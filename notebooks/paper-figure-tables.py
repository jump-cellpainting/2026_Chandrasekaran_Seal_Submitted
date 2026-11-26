"""Tables and one bar chart that paper figures show beside pipeline plots, drawn from pipeline outputs.

Each panel restates values that an analysis already wrote under `data/processed/`; nothing is
computed here beyond the percentages, means and standard deviations that the tables print. No
caption, panel letter or figure number is drawn: `notebooks/paper-figure-assembly.py` places these
panels in a figure and adds the letters.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import typer

from jump_production.summaries import write_summary

CAPTIONS = {
    "cp_dl_activity_table": "Active compounds with CellProfiler (CP) and deep-learning (DL) profiles.",
    "cp_dl_consistency_table": "Share of annotation groups that are consistent with CP and with DL profiles, by annotation source.",
    "batch_mixing_ilisi": "Normalized iLISI batch mixing for the CellProfiler and CP-CNN profile sets without Source 7.",
    "cosine_vs_abs_cosine_table": "Consistency with cosine against absolute-cosine similarity, for disease-area and target groups.",
    "polypharmacology_top_delta_table": "The five dual-target trios with the largest gain of the blend over the better single-target compound.",
    "toxicity_classification_table": "Toxicity classification from morphology against a label-shuffled baseline; mean and sample standard deviation over splits.",
    "pk_regression_table": "Pharmacokinetic regression from morphology against a baseline; mean and sample standard deviation over splits.",
}


def main(
    processed: Path = typer.Option(Path("data/processed"), help="Results tree to read"),
    output: Path = typer.Option(Path("data/paper_figures/panels/tables"), help="Directory for the panels"),
) -> None:
    REL, OUT = processed, output
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    VALUES = {}  # every value drawn, written to the summary so it can be checked

    def table(stem, columns, rows, widths, fontsize=8, row_height=0.24, align=None):
        width = sum(widths)
        height = row_height * (len(rows) + 1)
        figure, axis = plt.subplots(figsize=(width, height))
        axis.axis("off")
        cells = axis.table(
            cellText=rows,
            colLabels=columns,
            colWidths=[w / width for w in widths],
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )
        cells.auto_set_font_size(False)
        cells.set_fontsize(fontsize)
        for (r, c), cell in cells.get_celld().items():
            cell.set_edgecolor("#b8c4cc")
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#eef2f5")
            if align and r > 0:
                cell.set_text_props(ha=align[c])
                cell._loc = align[c]
        figure.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)
        figure.savefig(OUT / f"{stem}.pdf")
        figure.savefig(OUT / f"{stem}.png", dpi=300)
        plt.close(figure)
        VALUES[stem] = {"columns": columns, "rows": rows}

    # ---- CP vs DL: activity table and consistency table --------------------------------------
    cmp_root = REL / "profile-comparison"
    act = json.loads(
        (cmp_root / "compound_no_source7_vs_compound_DL_CPCNN_no_source7/activity/default/summary.json").read_text()
    )["numbers"]
    oc = act["overlap_counts"]
    n = act["n_items"]
    cp = oc["both"] + oc["only_compound_no_source7"]
    dl = oc["both"] + oc["only_compound_DL_CPCNN_no_source7"]
    table(
        "cp_dl_activity_table",
        ["Profile", "% Active", "Active compounds"],
        [["CP", f"{100 * cp / n:.1f}%", f"{cp:,}"], ["DL", f"{100 * dl / n:.1f}%", f"{dl:,}"]],
        [1.0, 1.0, 1.6],
    )
    VALUES["cp_dl_activity"] = {
        **oc,
        "n_items": n,
        "cp_active": cp,
        "dl_active": dl,
        "concordance": act["concordance_fraction"],
        "union": cp + dl - oc["both"],
    }
    NAMES = [
        ("uniprot", "UniProt"),
        ("motive_biokg", "MOTIVE BioKG"),
        ("repurposing", "Repurposing Hub"),
        ("moa", "MOA"),
        ("chemical_probes", "Chemical Probes"),
        ("motive_opentargets", "MOTIVE OpenTargets"),
        ("motive_primekg", "MOTIVE PrimeKG"),
        ("disease_area", "Disease Area"),
    ]
    rows = []
    for key, label in NAMES:
        s = json.loads(
            (
                cmp_root
                / "compound_no_source7_active_union_vs_compound_DL_CPCNN_no_source7_active_union/consistency"
                / key
                / "cosine/summary.json"
            ).read_text()
        )["numbers"]
        o = s["overlap_counts"]
        total = s["n_items"]
        a = o.get("both", 0) + o.get("only_compound_no_source7_active_union", 0)
        b = o.get("both", 0) + o.get("only_compound_DL_CPCNN_no_source7_active_union", 0)
        rows.append(
            [
                label,
                str(total),
                f"{100 * a / total:.1f}%",
                f"{100 * b / total:.1f}%",
                f"{100 * s['concordance_fraction']:.1f}%",
            ]
        )
    table(
        "cp_dl_consistency_table",
        ["Source", "N", "CP %", "DL %", "Concordance"],
        rows,
        [1.7, 0.6, 0.8, 0.8, 1.1],
        align=["left", "center", "center", "center", "center"],
    )

    # ---- batch mixing bar (the two profile sets without Source 7) ----------------------------
    be = pd.read_csv(REL / "batch-effect/batch_effect_summary.csv").set_index("name")
    figure, axis = plt.subplots(figsize=(1.9, 2.6))
    vals = [be.loc["CP_no_s7", "iLISI_norm"], be.loc["DL_no_s7", "iLISI_norm"]]
    bars = axis.bar(["CellProfiler", "CP-CNN"], vals, color=["#3d85c6", "#e8702a"], width=0.7)
    for bar, value in zip(bars, vals):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}", ha="center", fontsize=7)
    axis.set_ylabel("iLISI (normalized)", fontsize=8)
    axis.set_ylim(0, 0.35)
    axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUT / "batch_mixing_ilisi.pdf")
    figure.savefig(OUT / "batch_mixing_ilisi.png", dpi=300)
    plt.close(figure)
    VALUES["batch_effect"] = be[["iLISI_norm", "kBET_acceptance"]].to_dict("index")

    # ---- cosine vs absolute cosine table -----------------------------------------------------
    ab = json.loads((REL / "exploration/0.01/compound_no_source7/summary.json").read_text())["numbers"][
        "per_annotation"
    ]
    d, t = ab["disease_area"], ab["repurposing"]

    def pct(k, x):
        return f"{x[k]} ({100 * x[k] / x['n_groups']:.1f}%)"

    table(
        "cosine_vs_abs_cosine_table",
        ["Metric", "Disease Area", "Target"],
        [
            ["N groups", str(d["n_groups"]), str(t["n_groups"])],
            ["Correlation", f"r = {d['nmap_correlation_r']:.3f}", f"r = {t['nmap_correlation_r']:.3f}"],
            ["Groups improved", pct("groups_improved", d), pct("groups_improved", t)],
            ["Groups worse", pct("groups_worse", d), pct("groups_worse", t)],
            [
                "Mean diff (abs - cos)",
                f"{d['mean_nmap_diff_abs_minus_cosine']:.4f}",
                f"{t['mean_nmap_diff_abs_minus_cosine']:.4f}",
            ],
            ["Lost significance", str(d["sig_lost_cosine_to_abs"]), str(t["sig_lost_cosine_to_abs"])],
            ["Gained significance", str(d["sig_gained"]), str(t["sig_gained"])],
        ],
        [1.7, 1.2, 1.2],
        align=["left", "center", "center"],
    )

    # ---- polypharmacology: top five trios by delta -------------------------------------------
    tr = pd.read_csv(REL / "polypharmacology/compound_no_source7/trio_categories.csv")
    tr["delta"] = tr["sim_AB_blend"] - tr[["sim_AB_A", "sim_AB_B"]].max(axis=1)
    top = tr.sort_values(["delta", "target_pair"], ascending=[False, True]).head(5)
    SHORT = {
        "Additive polypharmacology": "Additive",
        "Convergent targets": "Convergent",
        "Dominant target": "Dominant",
        "Divergent from shared": "Divergent",
        "Emergent phenotype": "Emergent",
    }
    table(
        "polypharmacology_top_delta_table",
        ["Target pair", "AB", "A", "B", "sim(AB, blend)", "sim(AB, A)", "sim(AB, B)", "sim(A, B)", "delta", "category"],
        [
            [
                r.target_pair,
                r.name_ab,
                r.name_a,
                r.name_b,
                f"{r.sim_AB_blend:.2f}",
                f"{r.sim_AB_A:.2f}",
                f"{r.sim_AB_B:.2f}",
                f"{r.sim_A_B:.2f}",
                f"{r.delta:+.2f}",
                SHORT[r.category],
            ]
            for r in top.itertuples()
        ],
        [1.0, 0.95, 0.95, 1.05, 0.85, 0.75, 0.75, 0.7, 0.5, 0.8],
        fontsize=7,
    )
    VALUES["polypharmacology"] = {
        "category_counts": tr["category"].value_counts().to_dict(),
        "n_pairs": int(len(tr)),
        "min_trio_nmap": float(tr[["nmap_ab", "nmap_a", "nmap_b"]].min().min()),
        "n_positive_delta": int((tr["delta"] > 0).sum()),
    }

    # The toxicity and PK outputs carry no cell-count baseline, so the tables have no such column.
    # ---- toxicity / PK tables ----------------------------------------------------------------
    tox = REL / "toxicity-pk-prediction/compound_no_source7"
    cl = pd.read_csv(tox / "classification_results.csv")
    rg = pd.read_csv(tox / "regression_results.csv")
    VALUES["toxpk_columns"] = {
        "classification": list(cl.columns),
        "regression": list(rg.columns),
        "class_modalities": sorted(cl.modality.unique()),
        "reg_modalities": sorted(rg.modality.unique()),
    }
    rows = []
    for task in ["DICT", "DILI", "MitoTox"]:
        m = cl[(cl.task == task) & (cl.modality == "morphology")]
        n_cmp = int((m.n_train + m.n_test).iloc[0])
        lift = m.model_auc.mean() - m.baseline_auc.mean()
        rows.append(
            [
                task,
                str(n_cmp),
                f"{m.model_auc.mean():.3f} ± {m.model_auc.std():.3f}",
                f"{m.baseline_auc.mean():.3f}",
                f"{lift:+.3f}",
            ]
        )
    table(
        "toxicity_classification_table",
        ["Task", "N", "Model AUC (morphology)", "Baseline AUC", "Lift (morph - baseline)"],
        rows,
        [0.8, 0.5, 1.7, 1.0, 1.6],
        fontsize=7.5,
    )
    LABEL = {
        "human_VDss_L_kg": "VDss (L/kg)",
        "human_CL_mL_min_kg": "CL (mL/min/kg)",
        "human_fup": "fup",
        "human_mrt": "MRT (h)",
        "human_thalf": "t1/2 (h)",
    }
    rows = []
    for target, label in LABEL.items():
        m = rg[(rg.target == target) & (rg.modality == "morphology")]
        rows.append(
            [
                label,
                str(int((m.n_train + m.n_test).iloc[0])),
                f"{m.model_r2.mean():.3f} ± {m.model_r2.std():.3f}",
                f"{m.model_spearman.mean():.3f}",
                f"{m.baseline_r2.mean():.3f}",
            ]
        )
    table(
        "pk_regression_table",
        ["Endpoint", "N", "Model R2 (morphology)", "Model rho", "Baseline R2"],
        rows,
        [1.2, 0.5, 1.7, 0.9, 1.0],
        fontsize=7.5,
    )

    files = [
        {"path": f"{stem}.{suffix}", "kind": "figure", "caption": caption}
        for stem, caption in CAPTIONS.items()
        for suffix in ("png", "pdf")
    ]
    write_summary(
        OUT / "summary.json",
        description=(
            "Tables and one bar chart drawn from pipeline outputs for use as panels of assembled paper "
            "figures. Every value restates an analysis output; none is computed here."
        ),
        numbers=VALUES,
        files=files,
    )


if __name__ == "__main__":
    typer.run(main)
