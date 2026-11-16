"""Compound-trait associations with Cell Painting phenotypic activity.

Tests which compound properties are associated with a detectable Cell Painting
phenotype for binary compound traits:

  1) Clinical development stage (NEW here): Launched, Phase 3/2/1, and any-clinical
     (Phase 1 through Launched) vs other compounds with a Repurposing Hub stage annotation.
  2) Drug-likeness liabilities: 1 / 2 / 3 Lipinski Ro5 violations (vs 0), and PAINS-flagged
     (vs clean) -- re-derived here so they share one activity definition and CI method.
  3) Kinase / chemical-probe libraries: KCGS, PKIS, PKIS2, and the Probes & Drugs
     collection that subsumes SGC chemical probes.

Each trait is a 2x2 Fisher's exact test {member, non-member} x {active, inactive} with a
Haldane-Anscombe odds ratio and Woolf 95% CI. Missing measurements and annotations are
excluded through explicit trait-specific eligible populations. Activity is called at
copairs corrected p < 0.10.

This is the binary complement to the continuous-trait (2.04c, issue #27) and multi-class
(2.43, issue #60; 2.42, issue #57) analyses.

Usage:
    pixi run -e cheminformatics python notebooks/2.45-seal-property-activity-oddsratios.py

Outputs: data/processed/property-activity-oddsratios/{dataset}/
    - binary_trait_oddsratios.csv
    - multiclass_trait_oddsratios.csv
    - intrinsic_activity_nmap_bins.csv (plot-ready intrinsic-property contract)
    - selected_categorical_activity_nmap.csv (clinical phase and fixed collections)
    - compound_activity_nmap_distributions.parquet (compound-level boxplot contract)
    - forest_binary_traits.{png,pdf}
    - forest_binary_traits_full.{png,pdf}
    - forest_multiclass_activity.{png,pdf}
    - optional multivariate_coefficients.csv and forest_multivariate.{png,pdf}
    - summary.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.special import expit
from scipy.stats import fisher_exact, norm, spearmanr
from statsmodels.stats.multitest import multipletests

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, METADATA_DB
from jump_production.queries import query_activity_results
from jump_production.summaries import write_summary

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

OUTPUT_DIR = PROCESSED_DATA_DIR / "property-activity-oddsratios"

MULTICLASS_FAMILIES = {
    "rdkit_fg": ("RDKit functional group", "#1f77b4"),
    "brics": ("BRICS fragment", "#ff7f0e"),
    "scaffold": ("Murcko scaffold", "#2ca02c"),
    "target_class": ("Target / MoA class", "#d62728"),
    "epa_fup": ("EPA functional use (pred.)", "#9467bd"),
    "epa_fuc": ("EPA functional use (curated)", "#8c564b"),
}

TRAIT_MIN_COMPOUNDS = {
    "rdkit_fg": 10,
    "brics": 20,
    "scaffold": 5,
    "target_class": 5,
}

# Physicochemical + structural predictors for the optional exploratory model.
MULTIVARIATE_FEATURES = {
    "logp": "LogP",
    "mw": "Molecular weight",
    "tpsa": "TPSA",
    "qed": "QED (drug-likeness)",
    "hbd": "H-bond donors",
    "hba": "H-bond acceptors",
    "rotatable": "Rotatable bonds",
    "num_rings": "Ring count",
    "num_aromatic": "Aromatic ring count",
    "pains": "PAINS-flagged",
}

ACTIVITY_THRESHOLD = 0.10  # copairs mAP corrected-p cutoff for "phenotypically active"
FDR_THRESHOLD = 0.05
MIN_NUMERIC_N = 50
MIN_CELL_COUNT = 5

INTRINSIC_TRAITS = (
    ("chem.molecular_weight", "mw", "Molecular weight", "Molecular weight (Da)"),
    ("chem.logp", "logp", "LogP", "Calculated LogP"),
    ("chem.tpsa", "tpsa", "TPSA", "TPSA (angstrom^2)"),
    ("chem.hydrogen_bond_donors", "hbd", "Hydrogen-bond donors", "Hydrogen-bond donors"),
    ("chem.hydrogen_bond_acceptors", "hba", "Hydrogen-bond acceptors", "Hydrogen-bond acceptors"),
    ("chem.rotatable_bonds", "rotatable", "Rotatable bonds", "Rotatable bonds"),
    ("chem.ring_count", "num_rings", "Ring count", "Ring count"),
    ("chem.heavy_atom_count", "heavy_atom_count", "Heavy-atom count", "Heavy-atom count"),
    ("chem.aromatic_ring_count", "num_aromatic", "Aromatic-ring count", "Aromatic-ring count"),
    ("chem.fraction_csp3", "fraction_csp3", "Fraction Csp3", "Fraction Csp3"),
    ("chem.lipinski_violations", "lipinski", "Lipinski violations", "Lipinski violations"),
    ("chem.qed", "qed", "QED", "QED"),
)

INTRINSIC_DISPLAY_TRAITS = (
    "chem.logp",
    "chem.molecular_weight",
    "chem.rotatable_bonds",
    "chem.qed",
    "chem.tpsa",
    "chem.fraction_csp3",
)
BIN_COUNT = 10
BIN_TRIM_QUANTILES = (0.01, 0.99)
# Relative, so a bin floor exists at screen scale without emptying a small input.
MIN_BIN_FRACTION = 0.001

# The compound-level contract carries the displayed properties too, so a renderer can
# show the screen unbinned instead of only at bin centers.
DISPLAY_TRAIT_COLUMNS = tuple(
    column for trait_id, column, _, _ in INTRINSIC_TRAITS if trait_id in INTRINSIC_DISPLAY_TRAITS
)

SELECTED_CLINICAL_PHASES = ("Phase 1", "Phase 2", "Phase 3", "Launched")
SELECTED_COLLECTION_SPECS = (
    ("KCGS", "collection.kcgs"),
    ("PKIS", "collection.pkis"),
    ("PKIS2", "collection.pkis2"),
    ("Chemical Probes (includes SGC)", "collection.chemical_probes_sgc"),
)

COMPOUND_DISTRIBUTION_TABLE = "compound_activity_nmap_distributions.parquet"

# Category colours + display order for the forest plot
CATEGORY_COLORS = {
    "Clinical development": "#1f77b4",
    "Drug-likeness liability": "#d62728",
    "Structural": "#8c564b",
    "Kinase / chemical-probe library": "#2ca02c",
    "Annotation presence": "#9467bd",
}
CATEGORY_ORDER = list(CATEGORY_COLORS.keys())

# Annotation-presence traits: (label, table). Each is a "does the compound carry ANY
# annotation from source X" binary. NOTE: these are partly circular -- compounds are
# curated into these sources BECAUSE they are known bioactives -- so a positive OR is
# expected and should be read as a sanity check, not an independent driver.
ANNOTATION_TABLES = [
    ("Has ChEMBL target", "chembl_protein_targets"),
    ("In Repurposing Hub", "repurposing_hub_annotations"),
    ("Has MOTIVE drug-target annotation", "motive_annotations"),
    ("Has ToxCast annotation", "toxcast_annotations"),
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one provenance input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    """Raise when a required upstream schema is incomplete."""
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def save_figure(fig, path: Path, dpi: int) -> None:
    """Save a plot without clock-dependent PDF metadata."""
    kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if path.suffix == ".pdf" else {}
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)


def _haldane_or_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Haldane-Anscombe odds ratio with a Woolf 95% confidence interval."""
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = a_ * d_ / (b_ * c_)
    log_or = np.log(odds_ratio)
    standard_error = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    z_value = float(norm.ppf(0.975))
    return (
        float(odds_ratio),
        float(np.exp(log_or - z_value * standard_error)),
        float(np.exp(log_or + z_value * standard_error)),
    )


def logistic_or_per_iqr(values: pd.Series, outcomes: pd.Series) -> tuple[float, float, float, float, float]:
    """Fit a two-parameter logistic model and return an odds ratio per IQR."""
    x_values = np.asarray(values, dtype=float)
    y_values = np.asarray(outcomes, dtype=float)
    iqr = float(np.subtract(*np.percentile(x_values, [75, 25])))
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The numeric trait has no positive interquartile range")
    scaled = (x_values - np.median(x_values)) / iqr
    design = np.column_stack([np.ones(len(scaled)), scaled])
    coefficients = np.zeros(2, dtype=float)
    for _ in range(100):
        fitted = expit(np.clip(design @ coefficients, -35, 35))
        weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
        information = design.T @ (design * weights[:, None])
        score = design.T @ (y_values - fitted)
        step = np.linalg.solve(information, score)
        coefficients += step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        raise ValueError("Logistic model did not converge")
    fitted = expit(np.clip(design @ coefficients, -35, 35))
    weights = np.clip(fitted * (1.0 - fitted), 1e-9, None)
    covariance = np.linalg.inv(design.T @ (design * weights[:, None]))
    standard_error = float(np.sqrt(covariance[1, 1]))
    coefficient = float(coefficients[1])
    z_value = float(norm.ppf(0.975))
    return (
        float(np.exp(coefficient)),
        float(np.exp(coefficient - z_value * standard_error)),
        float(np.exp(coefficient + z_value * standard_error)),
        float(2 * norm.sf(abs(coefficient / standard_error))),
        iqr,
    )


def wilson_bounds(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided Wilson 95% interval for one binomial proportion."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z_value = float(norm.ppf(0.975))
    rate = successes / total
    denominator = 1 + z_value**2 / total
    center = (rate + z_value**2 / (2 * total)) / denominator
    half_width = z_value * np.sqrt(rate * (1 - rate) / total + z_value**2 / (4 * total**2)) / denominator
    return float(center - half_width), float(center + half_width)


def build_intrinsic_activity_nmap_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Build intrinsic-property plot rows from the canonical activity population.

    The displayed six traits use quantile-bin summaries. Their activity odds
    ratios are corrected together with every tested intrinsic chemistry trait,
    including the PAINS indicator, matching the original figure's FDR family.
    """
    required = {
        "Metadata_JCP2022",
        "mean_normalized_average_precision",
        "active",
        "valid_mol",
        "pains",
        *(column for _, column, _, _ in INTRINSIC_TRAITS),
    }
    require_columns(frame, required, "intrinsic activity input")
    if frame["Metadata_JCP2022"].duplicated().any():
        raise ValueError("Intrinsic activity input must contain one row per compound")
    valid = frame.loc[frame["valid_mol"].fillna(False).astype(bool)].copy()
    if valid.empty:
        raise ValueError("Intrinsic activity input has no valid structures")

    inference_rows: list[dict[str, object]] = []
    for trait_id, column, title, xlabel in INTRINSIC_TRAITS:
        measured = valid[[column, "active", "mean_normalized_average_precision"]].dropna().copy()
        measured[column] = pd.to_numeric(measured[column], errors="raise")
        active = measured["active"].astype(bool)
        row: dict[str, object] = {
            "trait_id": trait_id,
            "title": title,
            "xlabel": xlabel,
            "status": "excluded",
            "exclusion_reason": "",
            "estimate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "rho": np.nan,
            "n_total": int(len(measured)),
        }
        if len(measured) < MIN_NUMERIC_N:
            row["exclusion_reason"] = f"fewer than {MIN_NUMERIC_N} measured compounds"
        elif min(int(active.sum()), int((~active).sum())) < MIN_CELL_COUNT:
            row["exclusion_reason"] = f"fewer than {MIN_CELL_COUNT} active or inactive compounds"
        elif measured[column].nunique() < 2:
            row["exclusion_reason"] = "trait has one observed value"
        else:
            try:
                estimate, ci_lower, ci_upper, p_value, _ = logistic_or_per_iqr(measured[column], active)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                row["exclusion_reason"] = str(error)
            else:
                correlation = spearmanr(
                    measured[column], measured["mean_normalized_average_precision"], nan_policy="omit"
                )
                row.update(
                    {
                        "status": "tested",
                        "estimate": estimate,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "p_value": p_value,
                        "rho": float(correlation.statistic),
                    }
                )
        inference_rows.append(row)

    pains = valid[["pains", "active", "mean_normalized_average_precision"]].dropna().copy()
    exposed = pains["pains"].astype(bool)
    active = pains["active"].astype(bool)
    cells = (
        int((active & exposed).sum()),
        int((~active & exposed).sum()),
        int((active & ~exposed).sum()),
        int((~active & ~exposed).sum()),
    )
    pains_row: dict[str, object] = {
        "trait_id": "chem.has_pains",
        "title": "PAINS structural alert",
        "xlabel": "PAINS present versus absent",
        "status": "excluded",
        "exclusion_reason": "",
        "estimate": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "p_value": np.nan,
        "rho": np.nan,
        "n_total": int(len(pains)),
    }
    if len(pains) < MIN_NUMERIC_N:
        pains_row["exclusion_reason"] = f"fewer than {MIN_NUMERIC_N} measured compounds"
    elif min(cells) < MIN_CELL_COUNT:
        pains_row["exclusion_reason"] = f"a contingency cell contains fewer than {MIN_CELL_COUNT} compounds"
    else:
        estimate, ci_lower, ci_upper = _haldane_or_ci(*cells)
        _, p_value = fisher_exact([[cells[0], cells[1]], [cells[2], cells[3]]], alternative="two-sided")
        correlation = spearmanr(exposed.astype(float), pains["mean_normalized_average_precision"], nan_policy="omit")
        pains_row.update(
            {
                "status": "tested",
                "estimate": estimate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": float(p_value),
                "rho": float(correlation.statistic),
            }
        )
    inference_rows.append(pains_row)

    inference = pd.DataFrame(inference_rows)
    inference["fdr_bh"] = np.nan
    tested = inference["status"].eq("tested")
    if tested.any():
        inference.loc[tested, "fdr_bh"] = multipletests(inference.loc[tested, "p_value"], method="fdr_bh")[1]
    inference["significant_at_5pct_fdr"] = inference["fdr_bh"].lt(FDR_THRESHOLD).fillna(False)
    inference = inference.set_index("trait_id")

    rows: list[dict[str, object]] = []
    rho_p_values: dict[str, float] = {}
    for trait_order, trait_id in enumerate(INTRINSIC_DISPLAY_TRAITS):
        _, column, _, _ = next(spec for spec in INTRINSIC_TRAITS if spec[0] == trait_id)
        result = inference.loc[trait_id]
        if result["status"] != "tested":
            raise ValueError(f"Displayed intrinsic trait {trait_id} was not testable: {result['exclusion_reason']}")
        measured = (
            valid[[column, "active", "mean_normalized_average_precision"]]
            .dropna()
            .sort_values(column, kind="stable")
            .reset_index(drop=True)
        )
        # Equal-width bins over the trimmed window, not quantiles. A quantile curve is
        # drawn at bin medians, so it stops two half-bins short of the data and reads as
        # truncated wherever the axis is not scaled to the curve itself. Regular
        # intervals span their window and carry explicit edges. The trim keeps a handful
        # of extreme compounds (LogP runs to 18, molecular weight to 2,880) from
        # collapsing every real compound into two bins; inference below is unaffected,
        # because odds ratios and rho use every measured compound.
        # The correlation's own p, so a renderer can label it the same way the cohort
        # figures do. At this sample size it is a foregone conclusion, which is the point.
        # It uses every measured compound, because it is inference rather than display.
        rho_p_values[trait_id] = float(
            spearmanr(measured[column], measured["mean_normalized_average_precision"]).pvalue
        )
        low, high = measured[column].quantile(BIN_TRIM_QUANTILES)
        edges = np.linspace(float(low), float(high), BIN_COUNT + 1)
        windowed = measured.loc[measured[column].between(float(low), float(high))].copy()
        # labels=False gives the bin's position in `edges`, so the published edges are the
        # linspace cut points themselves. Interval labels would carry pandas' display
        # rounding (precision=3) instead, which is not where the compounds were actually
        # split.
        windowed["bin"] = pd.cut(windowed[column], bins=edges, include_lowest=True, labels=False)
        # The linear reading of the same relationship, fitted on the same window the bins
        # span so a renderer never fits anything itself. Fitting on every measured
        # compound instead would let the trimmed tails - the few compounds the window
        # exists to exclude, and the ones least squares weights hardest - set the slope of
        # a line drawn where they are not.
        fit_slope, fit_intercept = np.polyfit(windowed[column], windowed["mean_normalized_average_precision"], 1)
        # A bin needs two compounds before its standard error means anything, so the floor
        # is never below two however small the input.
        bin_floor = max(2, round(MIN_BIN_FRACTION * len(measured)))
        for bin_code, group in windowed.groupby("bin", sort=True):
            if len(group) < bin_floor:
                continue
            bin_lower, bin_upper = float(edges[bin_code]), float(edges[bin_code + 1])
            # Position in the window, not a running count: an empty or under-floor bin
            # leaves a gap in the numbering rather than renumbering the bins above it.
            bin_index = int(bin_code) + 1
            active_count = int(group["active"].astype(bool).sum())
            lower, upper = wilson_bounds(active_count, len(group))
            rows.append(
                {
                    "trait_order": trait_order,
                    "trait_id": trait_id,
                    "title": result["title"],
                    "xlabel": result["xlabel"],
                    "bin": bin_index,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "x": (bin_lower + bin_upper) / 2,
                    "n": int(len(group)),
                    "rate": active_count / len(group),
                    "lower": lower,
                    "upper": upper,
                    "nmap_median": float(group["mean_normalized_average_precision"].median()),
                    "nmap_mean": float(group["mean_normalized_average_precision"].mean()),
                    "nmap_mean_se": float(group["mean_normalized_average_precision"].std(ddof=1) / np.sqrt(len(group))),
                    "nmap_q25": float(group["mean_normalized_average_precision"].quantile(0.25)),
                    "nmap_q75": float(group["mean_normalized_average_precision"].quantile(0.75)),
                    "estimate": float(result["estimate"]),
                    "ci_lower": float(result["ci_lower"]),
                    "ci_upper": float(result["ci_upper"]),
                    "rho": float(result["rho"]),
                    "rho_p_value": rho_p_values[trait_id],
                    "fit_slope": float(fit_slope),
                    "fit_intercept": float(fit_intercept),
                    "n_total": int(result["n_total"]),
                    "fdr_bh": float(result["fdr_bh"]),
                    "significant_at_5pct_fdr": bool(result["significant_at_5pct_fdr"]),
                }
            )
    if not rows:
        raise ValueError("Intrinsic activity binning produced no rows")
    result = pd.DataFrame(rows).sort_values(["trait_order", "bin"], kind="stable").reset_index(drop=True)
    # One Benjamini-Hochberg family over the six displayed correlations, matching how the
    # cohort figures label theirs.
    traits = list(dict.fromkeys(result["trait_id"]))
    corrected = dict(zip(traits, multipletests([rho_p_values[t] for t in traits], method="fdr_bh")[1], strict=True))
    result["rho_q_value"] = result["trait_id"].map(corrected)
    return result


def selected_group_summary(
    frame: pd.DataFrame,
    *,
    view: str,
    view_order: int,
    row_order: int,
    label: str,
    trait_id: str,
    level: str,
    universe_id: str,
    comparator: str,
    member: pd.Series,
    eligible: pd.Series,
) -> dict[str, object]:
    """Summarize one fixed category against its declared reference population."""
    if not member.index.equals(frame.index) or not eligible.index.equals(frame.index):
        raise ValueError(f"Selected category masks are not aligned for {trait_id}:{level}")
    eligible = eligible.fillna(False).astype(bool)
    member = member.fillna(False).astype(bool) & eligible
    reference = eligible & ~member
    if not member.any() or not reference.any():
        raise ValueError(f"Selected category has an empty member or reference set: {trait_id}:{level}")
    member_frame = frame.loc[member]
    reference_frame = frame.loc[reference]
    active_member = int(member_frame["active"].astype(bool).sum())
    active_reference = int(reference_frame["active"].astype(bool).sum())
    lower, upper = wilson_bounds(active_member, len(member_frame))
    return {
        "view_order": view_order,
        "view": view,
        "row_order": row_order,
        "label": label,
        "trait_id": trait_id,
        "level": level,
        "universe_id": universe_id,
        "comparator": comparator,
        "n_eligible": int(eligible.sum()),
        "n": int(len(member_frame)),
        "active": active_member,
        "rate": 100 * active_member / len(member_frame),
        "rate_lower": 100 * lower,
        "rate_upper": 100 * upper,
        "reference_n": int(len(reference_frame)),
        "reference_active": active_reference,
        "reference_rate": 100 * active_reference / len(reference_frame),
        "nmap_median": float(member_frame["mean_normalized_average_precision"].median()),
        "nmap_q25": float(member_frame["mean_normalized_average_precision"].quantile(0.25)),
        "nmap_q75": float(member_frame["mean_normalized_average_precision"].quantile(0.75)),
        "reference_nmap_median": float(reference_frame["mean_normalized_average_precision"].median()),
    }


def build_selected_categorical_activity_nmap(
    frame: pd.DataFrame,
    library_sets: dict[str, set[str]],
) -> pd.DataFrame:
    """Build the two fixed categorical primary-view contracts from d2bba44."""
    required = {
        "Metadata_JCP2022",
        "mean_normalized_average_precision",
        "active",
        "valid_mol",
        "pains",
        "phase",
    }
    require_columns(frame, required, "selected categorical activity input")
    if frame["Metadata_JCP2022"].duplicated().any():
        raise ValueError("Selected categorical input must contain one row per compound")
    missing_collections = sorted({label for label, _ in SELECTED_COLLECTION_SPECS} - set(library_sets))
    if missing_collections:
        raise ValueError(f"Selected categorical input is missing collection sets: {missing_collections}")

    rows: list[dict[str, object]] = []
    phase = frame["phase"].astype("string")
    phase_eligible = phase.notna() & phase.ne("")
    phase_tokens = phase.fillna("").apply(
        lambda value: {piece.strip() for piece in str(value).split("|") if piece.strip()}
    )
    phase_comparator = "Other compounds with a Repurposing Hub clinical-stage annotation"
    for row_order, label in enumerate(SELECTED_CLINICAL_PHASES):
        rows.append(
            selected_group_summary(
                frame,
                view="clinical_phase",
                view_order=0,
                row_order=row_order,
                label=label,
                trait_id="annotation.repurposing_clinical_phase",
                level=label,
                universe_id="annotated::annotation.repurposing_clinical_phase",
                comparator=phase_comparator,
                member=phase_tokens.apply(lambda values, selected=label: selected in values),
                eligible=phase_eligible,
            )
        )

    valid_structure = frame["valid_mol"].fillna(False).astype(bool) & frame["pains"].notna()
    rows.append(
        selected_group_summary(
            frame,
            view="curated_collections",
            view_order=1,
            row_order=0,
            label="PAINS",
            trait_id="chem.has_pains",
            level="1",
            universe_id="valid_structure",
            comparator="Valid structures without a PAINS alert",
            member=frame["pains"].fillna(False).astype(bool),
            eligible=valid_structure,
        )
    )
    jcp2022 = frame["Metadata_JCP2022"]
    all_compounds = pd.Series(True, index=frame.index)
    for row_order, (label, trait_id) in enumerate(SELECTED_COLLECTION_SPECS, start=1):
        rows.append(
            selected_group_summary(
                frame,
                view="curated_collections",
                view_order=1,
                row_order=row_order,
                label=label,
                trait_id=trait_id,
                level="member",
                universe_id="analyzed_compounds",
                comparator="Analyzed compounds outside this curated collection",
                member=jcp2022.isin(library_sets[label]),
                eligible=all_compounds,
            )
        )
    result = pd.DataFrame(rows).sort_values(["view_order", "row_order"], kind="stable")
    if result.groupby("view").size().to_dict() != {"clinical_phase": 4, "curated_collections": 5}:
        raise ValueError("Selected categorical contract must contain four phase and five collection rows")
    return result.reset_index(drop=True)


def build_compound_activity_nmap_distributions(
    frame: pd.DataFrame,
    library_sets: dict[str, set[str]],
) -> pd.DataFrame:
    """Build the compound-level contract used by the selected nMAP boxplots."""
    required = {
        "Metadata_JCP2022",
        "mean_normalized_average_precision",
        "active",
        "lipinski",
        "valid_mol",
        "pains",
        "phase",
        *DISPLAY_TRAIT_COLUMNS,
    }
    require_columns(frame, required, "compound activity nMAP distribution input")
    if frame["Metadata_JCP2022"].duplicated().any():
        raise ValueError("Compound activity nMAP distributions require one row per compound")
    missing_collections = sorted({label for label, _ in SELECTED_COLLECTION_SPECS} - set(library_sets))
    if missing_collections:
        raise ValueError(f"Compound activity nMAP distributions are missing collection sets: {missing_collections}")

    result = frame[
        [
            "Metadata_JCP2022",
            "mean_normalized_average_precision",
            "active",
            "lipinski",
            "valid_mol",
            "pains",
            "phase",
            *DISPLAY_TRAIT_COLUMNS,
        ]
    ].rename(
        columns={
            "Metadata_JCP2022": "jcp2022",
            "mean_normalized_average_precision": "nmap",
            "lipinski": "lipinski_violations",
            "valid_mol": "valid_molecule",
            "pains": "has_pains",
            "phase": "clinical_phase",
        }
    )
    identifiers = result["jcp2022"]
    membership_columns = {
        "KCGS": "in_kcgs",
        "PKIS": "in_pkis",
        "PKIS2": "in_pkis2",
        "Chemical Probes (includes SGC)": "in_chemical_probes_sgc",
    }
    for label, column in membership_columns.items():
        result[column] = identifiers.isin(library_sets[label])
    return result.sort_values("jcp2022", kind="stable").reset_index(drop=True)


# =============================================================================
# Odds-ratio helper
# =============================================================================


def odds_ratio_row(
    label: str,
    category: str,
    member: pd.Series,
    active: pd.Series,
    eligible: pd.Series | None = None,
    core: bool = False,
    comparator: str = "",
    correction_group: str | None = None,
) -> dict:
    """Calculate one eligible-population Fisher test and corrected OR interval.

    `member`, `active`, and `eligible` are aligned over the same compound set.
    Missing values must be excluded by `eligible`; they are never negative calls.
    """
    if not member.index.equals(active.index):
        raise ValueError(f"Trait {label} member and activity rows are not aligned")
    if eligible is None:
        eligible = pd.Series(True, index=active.index)
    if not eligible.index.equals(active.index):
        raise ValueError(f"Trait {label} eligibility and activity rows are not aligned")

    eligible = eligible.fillna(False).astype(bool)
    tested = pd.DataFrame({"member": member, "active": active}).loc[eligible]
    if tested[["member", "active"]].isna().any().any():
        raise ValueError(f"Trait {label} contains missing values inside its eligible population")
    tested = tested.astype(bool)

    a = int((tested["member"] & tested["active"]).sum())
    b = int((tested["member"] & ~tested["active"]).sum())
    c = int((~tested["member"] & tested["active"]).sum())
    d = int((~tested["member"] & ~tested["active"]).sum())
    n_member = a + b
    n_reference = c + d
    if n_member == 0 or n_reference == 0:
        raise ValueError(
            f"Trait {label} needs at least one member and one reference compound; observed {n_member} and {n_reference}"
        )

    _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    odds_ratio, ci_lower, ci_upper = _haldane_or_ci(a, b, c, d)

    return {
        "category": category,
        "trait": label,
        "candidate_display": core,
        "correction_group": correction_group or category,
        "comparator": comparator,
        "n_eligible": int(len(tested)),
        "n_excluded": int(len(active) - len(tested)),
        "n_member": n_member,
        "n_reference": n_reference,
        "n_active_member": a,
        "n_active_reference": c,
        "frac_active_member": a / n_member if n_member else np.nan,
        "frac_active_reference": c / n_reference if n_reference else np.nan,
        "odds_ratio": odds_ratio,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": float(p_value),
    }


# =============================================================================
# Trait assembly
# =============================================================================


def build_binary_traits(
    df: pd.DataFrame,
    annotation_sets: dict[str, set[str]],
    library_sets: dict[str, set[str]],
) -> pd.DataFrame:
    """Compute binary-trait associations using explicit eligible populations.

    `df` has one row per compound with columns: Metadata_JCP2022, active (bool),
    phase, lipinski, pains, mw, logp, hbd, hba, num_rings, scaffold.
    Traits in the candidate display set are tagged `candidate_display`.
    """
    if df["Metadata_JCP2022"].duplicated().any():
        raise ValueError("Binary-trait input must contain one row per JCP2022 compound")
    active = df["active"].astype(bool)
    all_compounds = pd.Series(True, index=df.index)
    valid_structure = df["valid_mol"].fillna(False).astype(bool)
    rows: list[dict] = []

    # 1) Clinical development stage, among compounds with a phase annotation.
    phase = df["phase"].astype("string")
    phase_eligible = phase.notna()
    phase_tokens = phase.fillna("").str.split("|")

    def has_phase(label: str) -> pd.Series:
        """Match one exact pipe-delimited phase token, preserving slash levels."""
        return phase_tokens.apply(lambda values: label in values)

    is_launched = has_phase("Launched")
    is_p3 = has_phase("Phase 3")
    is_p2 = has_phase("Phase 2")
    is_p1 = has_phase("Phase 1")
    is_withdrawn = has_phase("Withdrawn")
    developed_phases = {
        "Launched",
        "Phase 3",
        "Phase 2/Phase 3",
        "Phase 2",
        "Phase 1/Phase 2",
        "Phase 1",
    }
    any_clinical = phase_tokens.apply(lambda values: bool(developed_phases.intersection(values)))
    cat = "Clinical development"
    phase_comparator = "Other compounds with a Repurposing Hub clinical-stage annotation"
    for label, member, core in (
        ("Launched", is_launched, True),
        ("Phase 3", is_p3, True),
        ("Phase 2", is_p2, True),
        ("Phase 1", is_p1, True),
        ("Any clinical (Phase 1-Launched)", any_clinical, True),
        ("Withdrawn drug", is_withdrawn, False),
    ):
        rows.append(
            odds_ratio_row(
                label,
                cat,
                member,
                active,
                eligible=phase_eligible,
                core=core,
                comparator=phase_comparator,
            )
        )

    # 2) Drug-likeness liabilities, among compounds assessed for each property.
    cat = "Drug-likeness liability"
    lip = df["lipinski"]
    zero_viol = lip == 0
    for k in (1, 2, 3):
        eligible = valid_structure & (zero_viol | (lip == k))
        rows.append(
            odds_ratio_row(
                f"{k} Lipinski violation{'s' if k > 1 else ''} (vs 0)",
                cat,
                lip == k,
                active,
                eligible=eligible,
                core=True,
                comparator="Compounds with 0 Lipinski violations",
            )
        )
    rows.append(
        odds_ratio_row(
            "PAINS-flagged (vs clean)",
            cat,
            df["pains"],
            active,
            eligible=valid_structure & df["pains"].notna(),
            core=True,
            comparator="Compounds assessed as having no PAINS alert",
        )
    )
    for label, column, threshold in (
        ("MW > 500", "mw", 500),
        ("LogP > 5", "logp", 5),
        ("HBD > 5", "hbd", 5),
        ("HBA > 10", "hba", 10),
    ):
        values = df[column]
        rows.append(
            odds_ratio_row(
                label,
                cat,
                values > threshold,
                active,
                eligible=valid_structure & values.notna(),
                comparator=f"Assessed compounds with {column} <= {threshold}",
            )
        )

    # 3) Structural properties, among compounds with a valid structure.
    cat = "Structural"
    rows.append(
        odds_ratio_row(
            "Acyclic (no rings)",
            cat,
            df["num_rings"] == 0,
            active,
            eligible=valid_structure & df["num_rings"].notna(),
            comparator="Compounds with at least one ring",
        )
    )
    scaf_counts = df["scaffold"].value_counts()
    singleton = df["scaffold"].map(scaf_counts).fillna(0) == 1
    rows.append(
        odds_ratio_row(
            "Singleton scaffold (unique in JUMP)",
            cat,
            singleton,
            active,
            eligible=valid_structure,
            comparator="All other compounds with a valid structure",
        )
    )

    # 4) Annotation presence is itself the exposure, so all analyzed compounds are eligible.
    cat = "Annotation presence"
    jcp = df["Metadata_JCP2022"]
    for label, members in annotation_sets.items():
        rows.append(
            odds_ratio_row(
                label,
                cat,
                jcp.isin(members),
                active,
                eligible=all_compounds,
                comparator="Analyzed compounds without an entry in this annotation source",
            )
        )

    # 5) Curated library membership, queried directly from the augmented database.
    cat = "Kinase / chemical-probe library"
    for label, members in library_sets.items():
        rows.append(
            odds_ratio_row(
                label,
                cat,
                jcp.isin(members),
                active,
                eligible=all_compounds,
                core=True,
                comparator="Analyzed compounds outside this curated collection",
            )
        )

    results = pd.DataFrame(rows)
    results.insert(0, "display_order", np.arange(len(results)))
    corrected = pd.Series(index=results.index, dtype=float)
    for _, indices in results.groupby("correction_group", sort=False).groups.items():
        corrected.loc[indices] = multipletests(results.loc[indices, "p_value"], method="fdr_bh")[1]
    results["corrected_p_value"] = corrected
    results["significant"] = results["corrected_p_value"] < FDR_THRESHOLD
    return results


def load_annotation_memberships(meta_con) -> dict[str, set[str]]:
    """Return {trait label -> set of member JCP2022} for annotation-presence tables."""
    sets: dict[str, set] = {}
    for label, table in ANNOTATION_TABLES:
        jcps = meta_con.execute(
            f"SELECT DISTINCT Metadata_JCP2022 FROM {table} WHERE Metadata_JCP2022 IS NOT NULL"
        ).df()["Metadata_JCP2022"]
        sets[label] = set(jcps)
    return sets


def load_library_memberships(meta_con) -> dict[str, set[str]]:
    """Load curated probe-library memberships from the augmented database."""
    kinase = meta_con.execute(
        """
        SELECT DISTINCT Metadata_JCP2022, Metadata_kinase_probe_set
        FROM kinase_probes
        WHERE Metadata_JCP2022 IS NOT NULL
          AND Metadata_kinase_probe_set IN ('KCGS', 'PKIS')
        """
    ).df()
    chemical = set(
        meta_con.execute(
            """
            SELECT DISTINCT Metadata_JCP2022
            FROM chemical_probes
            WHERE Metadata_JCP2022 IS NOT NULL
            """
        ).df()["Metadata_JCP2022"]
    )
    pkis2 = set(
        meta_con.execute(
            """
            SELECT DISTINCT Metadata_JCP2022
            FROM pkis2_binding
            WHERE Metadata_JCP2022 IS NOT NULL
            """
        ).df()["Metadata_JCP2022"]
    )
    return {
        "KCGS": set(kinase.loc[kinase["Metadata_kinase_probe_set"].eq("KCGS"), "Metadata_JCP2022"]),
        "PKIS": set(kinase.loc[kinase["Metadata_kinase_probe_set"].eq("PKIS"), "Metadata_JCP2022"]),
        "Chemical Probes (includes SGC)": chemical,
        "PKIS2": pkis2,
    }


# =============================================================================
# Forest plot
# =============================================================================


def plot_forest(results: pd.DataFrame, output_dir: Path, dpi: int, filename: str, title: str) -> None:
    """Forest plot: odds ratio (log x) per trait, grouped/coloured by category."""
    results = results.copy()
    results["_cat_rank"] = results["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)})
    results = results.sort_values(["_cat_rank", "display_order"], kind="stable").reset_index(drop=True)

    y = np.arange(len(results))[::-1]  # top trait at top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(results) + 1.5))

    for yi, (_, r) in zip(y, results.iterrows()):
        color = CATEGORY_COLORS.get(r["category"], "#333333")
        ax.plot(
            [r["ci_lower"], r["ci_upper"]],
            [yi, yi],
            color=color,
            lw=1.5,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            r["odds_ratio"],
            yi,
            color=color,
            s=45,
            zorder=2,
            edgecolors="white",
            linewidths=0.6,
        )
        # annotate OR and n
        ax.text(
            r["ci_upper"] * 1.05,
            yi,
            f"OR {r['odds_ratio']:.2f}  (n={r['n_member']}, {100 * r['frac_active_member']:.0f}% active)",
            va="center",
            fontsize=7,
            color="#333333",
        )

    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.7, zorder=0)  # OR = 1 (no enrichment)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(results["trait"], fontsize=9)
    ax.set_xlabel("Odds ratio for phenotypic activity (log scale)")
    ax.set_title(title)

    # category legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CATEGORY_COLORS[c],
            markersize=8,
            label=c,
        )
        for c in CATEGORY_ORDER
        if c in set(results["category"])
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.25)

    plt.tight_layout()
    for ext in (".png", ".pdf"):
        save_figure(fig, output_dir / f"{filename}{ext}", dpi)
    plt.close(fig)
    logger.info(f"Saved: {output_dir / (filename + '.png')}")


# =============================================================================
# Multi-class traits (issue #60 substructures/targets + #57 EPA functional use)
# =============================================================================


def build_multiclass_table(
    trait_enrichment: pd.DataFrame,
    fup_enrichment: pd.DataFrame,
    fuc_enrichment: pd.DataFrame,
    n_total: int,
    n_active: int,
    fup_threshold: float,
    min_compounds: int,
) -> pd.DataFrame:
    """Normalize 2.42 and 2.43 counts under one explicit test contract.

    Each upstream table supplies its eligible-population contingency counts.
    Fisher p-values and within-family BH values are recomputed here because a
    denominator change must never retain inferential statistics from a
    different population.
    """
    require_columns(
        trait_enrichment,
        {
            "family",
            "trait",
            "n_eligible",
            "n_missing",
            "n_with_trait",
            "n_without_trait",
            "n_flag_with_trait",
            "n_flag_without_trait",
            "n_flag_total",
            "odds_ratio",
            "ci_lower",
            "ci_upper",
            "p_value",
        },
        "2.43 activity enrichment",
    )

    rows: list[dict] = []
    for record in trait_enrichment.itertuples(index=False):
        a = int(record.n_flag_with_trait)
        b = int(record.n_with_trait) - a
        c = int(record.n_flag_without_trait)
        d = int(record.n_without_trait) - c
        n_eligible = int(record.n_eligible)
        n_excluded = n_total - n_eligible
        if min(a, b, c, d, n_excluded) < 0:
            raise ValueError(f"2.43 row {record.family}:{record.trait} has impossible contingency counts")
        if a + b + c + d != n_eligible:
            raise ValueError(f"2.43 row {record.family}:{record.trait} does not sum to n_eligible")
        if int(record.n_missing) != n_excluded:
            raise ValueError(f"2.43 row {record.family}:{record.trait} has an inconsistent n_missing")
        if a + c != int(record.n_flag_total):
            raise ValueError(f"2.43 row {record.family}:{record.trait} has an inconsistent n_flag_total")
        if a + b == 0 or c + d == 0:
            raise ValueError(f"2.43 row {record.family}:{record.trait} needs members and references")
        _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        if not np.isclose(p_value, float(record.p_value), rtol=1e-10, atol=1e-300):
            raise ValueError(f"2.43 row {record.family}:{record.trait} has a p-value from different counts")
        odds_ratio, ci_lower, ci_upper = _haldane_or_ci(a, b, c, d)
        if not np.allclose(
            [odds_ratio, ci_lower, ci_upper],
            [record.odds_ratio, record.ci_lower, record.ci_upper],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"2.43 row {record.family}:{record.trait} is inconsistent with its 2x2 counts")
        family = str(record.family)
        if family not in TRAIT_MIN_COMPOUNDS:
            raise ValueError(f"2.43 row uses unknown trait family: {family}")
        if a + b < TRAIT_MIN_COMPOUNDS[family]:
            raise ValueError(
                f"2.43 row {family}:{record.trait} has fewer than {TRAIT_MIN_COMPOUNDS[family]} member compounds"
            )
        rows.append(
            {
                "family": family,
                "trait": str(record.trait),
                "source_analysis": "2.43",
                "correction_group": family,
                "membership_definition": "trait present",
                "membership_threshold": np.nan,
                "comparator": (
                    "Other compounds annotated by the same target or MoA source"
                    if family == "target_class"
                    else "Other compounds with a valid structure"
                ),
                "n_eligible": n_eligible,
                "n_excluded": n_excluded,
                "n_member": int(record.n_with_trait),
                "n_reference": int(record.n_without_trait),
                "n_active_member": a,
                "n_active_reference": c,
                "frac_active_member": a / int(record.n_with_trait),
                "frac_active_reference": c / (c + d),
                "odds_ratio": odds_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": float(p_value),
            }
        )

    required_functional_use = {
        "category",
        "n_eligible",
        "n_missing",
        "n_high",
        "n_rest",
        "n_active_high",
        "n_active_rest",
        "odds_ratio",
        "ci_lower",
        "ci_upper",
        "p_value",
        "corrected_p_value",
    }
    for frame, family, source_label in (
        (fup_enrichment, "epa_fup", "2.42 FUP"),
        (fuc_enrichment, "epa_fuc", "2.42 FUC"),
    ):
        required_columns = required_functional_use | ({"threshold"} if family == "epa_fup" else set())
        require_columns(frame, required_columns, source_label)
        for record in frame.itertuples(index=False):
            a = int(record.n_active_high)
            b = int(record.n_high) - a
            c = int(record.n_active_rest)
            d = int(record.n_rest) - c
            n_eligible = int(record.n_eligible)
            n_excluded = n_total - n_eligible
            if min(a, b, c, d, n_excluded) < 0:
                raise ValueError(f"{source_label} row {record.category} has impossible contingency counts")
            if a + b + c + d != n_eligible:
                raise ValueError(f"{source_label} row {record.category} does not sum to n_eligible")
            if a + b == 0 or c + d == 0:
                raise ValueError(f"{source_label} row {record.category} needs members and references")
            if a + b < min_compounds:
                raise ValueError(
                    f"{source_label} row {record.category} has fewer than {min_compounds} member compounds"
                )
            if family == "epa_fup" and not np.isclose(float(record.threshold), fup_threshold):
                raise ValueError(
                    f"{source_label} row {record.category} threshold={record.threshold}; expected {fup_threshold}"
                )
            odds_ratio, ci_lower, ci_upper = _haldane_or_ci(a, b, c, d)
            if not np.allclose(
                [odds_ratio, ci_lower, ci_upper],
                [record.odds_ratio, record.ci_lower, record.ci_upper],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"{source_label} row {record.category} is inconsistent with its 2x2 counts")
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            if not np.isclose(p_value, float(record.p_value), rtol=1e-10, atol=1e-300):
                raise ValueError(f"{source_label} row {record.category} has a p-value from different counts")
            rows.append(
                {
                    "family": family,
                    "trait": str(record.category),
                    "source_analysis": "2.42",
                    "correction_group": family,
                    "membership_definition": (
                        f"predicted functional-use probability >= {fup_threshold:g}"
                        if family == "epa_fup"
                        else "curated functional-use category token present"
                    ),
                    "membership_threshold": fup_threshold if family == "epa_fup" else np.nan,
                    "comparator": (
                        "Other compounds with a non-null prediction for this endpoint"
                        if family == "epa_fup"
                        else "Other compounds with at least one curated functional-use category"
                    ),
                    "n_eligible": n_eligible,
                    "n_excluded": n_excluded,
                    "n_member": int(record.n_high),
                    "n_reference": int(record.n_rest),
                    "n_active_member": a,
                    "n_active_reference": c,
                    "frac_active_member": a / int(record.n_high),
                    "frac_active_reference": c / int(record.n_rest),
                    "odds_ratio": odds_ratio,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_value": float(p_value),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No multi-class tests were produced")
    if result.duplicated(["family", "trait"]).any():
        raise ValueError("Multi-class family/trait identifiers must be unique")
    corrected = pd.Series(index=result.index, dtype=float)
    for _, indices in result.groupby("correction_group", sort=False).groups.items():
        corrected.loc[indices] = multipletests(result.loc[indices, "p_value"], method="fdr_bh")[1]
    result["corrected_p_value"] = corrected
    result["significant"] = result["corrected_p_value"] < FDR_THRESHOLD

    cell_sum = result[["n_member", "n_reference"]].sum(axis=1)
    active_sum = result[["n_active_member", "n_active_reference"]].sum(axis=1)
    if not cell_sum.eq(result["n_eligible"]).all():
        raise ValueError("Multi-class member and reference counts do not sum to n_eligible")
    if (active_sum > n_active).any():
        raise ValueError("Multi-class active counts exceed the selected activity population")
    if not (result["n_eligible"] + result["n_excluded"]).eq(n_total).all():
        raise ValueError("Multi-class eligible and excluded counts do not sum to the activity population")
    finite_positive = result[["odds_ratio", "ci_lower", "ci_upper"]].gt(0).all(axis=1) & np.isfinite(
        result[["odds_ratio", "ci_lower", "ci_upper"]]
    ).all(axis=1)
    if not finite_positive.all():
        raise ValueError("Multi-class odds ratios and confidence limits must be finite and positive")
    if not ((result["ci_lower"] < result["odds_ratio"]) & (result["odds_ratio"] < result["ci_upper"])).all():
        raise ValueError("Multi-class confidence intervals must contain their odds ratios")
    return result.sort_values(["family", "corrected_p_value", "trait"], kind="stable").reset_index(drop=True)


def plot_multiclass_forest(m: pd.DataFrame, output_dir: Path, dpi: int, top_k: int = 10, min_n: int = 5) -> None:
    """Forest of top-k enriched + top-k depleted multi-class traits per family."""
    cand = m[(m["n_member"] >= min_n) & m["significant"] & np.isfinite(m["odds_ratio"]) & (m["odds_ratio"] > 0)]
    sel = []
    for fam in MULTICLASS_FAMILIES:
        g = cand[cand["family"] == fam]
        sel.append(
            pd.concat(
                [
                    g[g["odds_ratio"] > 1].nsmallest(top_k, "corrected_p_value"),
                    g[g["odds_ratio"] < 1].nsmallest(top_k, "corrected_p_value"),
                ]
            )
        )
    sel = pd.concat(sel, ignore_index=True)
    sel["_fr"] = sel["family"].map({f: i for i, f in enumerate(MULTICLASS_FAMILIES)})
    sel = sel.sort_values(["_fr", "odds_ratio", "trait"], ascending=[True, False, True], kind="stable").reset_index(
        drop=True
    )

    def short(t: str) -> str:
        t = str(t)
        return t if len(t) <= 32 else t[:29] + "..."

    y = np.arange(len(sel))[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.3 * len(sel) + 1.5))
    for yi, (_, r) in zip(y, sel.iterrows()):
        col = MULTICLASS_FAMILIES[r["family"]][1]
        ax.plot([r["ci_lower"], r["ci_upper"]], [yi, yi], color=col, lw=1.1, alpha=0.8, zorder=1)
        ax.scatter(r["odds_ratio"], yi, color=col, s=26, edgecolors="white", linewidths=0.5, zorder=2)
    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([short(t) for t in sel["trait"]], fontsize=6.5)
    ax.set_xlabel("Odds ratio for phenotypic activity (log scale)")
    ax.set_title(
        f"Multi-class compound traits vs Cell Painting activity (top {top_k} enriched + {top_k} depleted per family)"
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=7,
            label=label,
        )
        for _, (label, color) in MULTICLASS_FAMILIES.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        save_figure(fig, output_dir / f"forest_multiclass_activity{ext}", dpi)
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'forest_multiclass_activity.png'} ({len(sel)} traits shown)")


# =============================================================================
# Multivariate model (which properties matter, controlling for the others)
# =============================================================================


def fit_multivariate(df: pd.DataFrame, output_dir: Path, dpi: int) -> dict:
    """Fit an exploratory complete-case association model in valid structures.

    Continuous features are centered at their median and scaled by their interquartile
    range, so their coefficients represent a one-IQR difference. PAINS is a 0/1
    indicator. The model is descriptive and does not support causal interpretation.
    """
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    cols = list(MULTIVARIATE_FEATURES)
    d = df.loc[df["valid_mol"].fillna(False)].dropna(subset=cols + ["active"]).copy()
    X = d[cols].astype(float).copy()
    cont = [c for c in cols if c != "pains"]
    medians = X[cont].median()
    iqrs = (X[cont].quantile(0.75) - X[cont].quantile(0.25)).replace(0, 1)
    X[cont] = (X[cont] - medians) / iqrs
    y = d["active"].astype(int).values

    # statsmodels for coefficients + CIs
    Xc = sm.add_constant(X)
    res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    coef = res.params.drop("const")
    ci = res.conf_int().drop("const")
    ci.columns = ["lo", "hi"]

    # cross-validated AUC for context
    auc = float(np.mean(cross_val_score(LogisticRegression(max_iter=500), X, y, cv=5, scoring="roc_auc")))
    logger.info(f"Multivariate logistic: n={len(y):,}, 5-fold AUC={auc:.3f}")

    out_rows = []
    for f in cols:
        out_rows.append(
            {
                "feature": MULTIVARIATE_FEATURES[f],
                "coef_logodds": float(coef[f]),
                "ci_lower": float(ci.loc[f, "lo"]),
                "ci_upper": float(ci.loc[f, "hi"]),
                "odds_ratio": float(np.exp(coef[f])),
                "increment": "0 to 1" if f == "pains" else "one IQR",
                "p_value": float(res.pvalues[f]),
            }
        )
    coefs = pd.DataFrame(out_rows).sort_values("coef_logodds")
    coefs.to_csv(output_dir / "multivariate_coefficients.csv", index=False)

    # Forest of IQR-scaled continuous coefficients and the binary PAINS coefficient.
    y_pos = np.arange(len(coefs))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(coefs) + 1.8))
    for yi, (_, r) in zip(y_pos, coefs.iterrows()):
        ax.plot(
            [r["ci_lower"], r["ci_upper"]],
            [yi, yi],
            color="#555",
            lw=1.4,
            zorder=1,
        )
        ax.scatter(
            r["coef_logodds"],
            yi,
            color="#2ca02c" if r["coef_logodds"] > 0 else "#d62728",
            s=55,
            zorder=2,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs["feature"], fontsize=9)
    ax.set_xlabel("Exploratory logistic coefficient (continuous: one IQR; PAINS: 0 to 1)")
    ax.set_title(f"Exploratory multivariable associations with Cell Painting activity (5-fold AUC = {auc:.2f})")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        save_figure(fig, output_dir / f"forest_multivariate{ext}", dpi)
    plt.close(fig)
    logger.info(f"Saved: {output_dir / 'forest_multivariate.png'}")

    return {
        "n": int(len(y)),
        "cv_auc": auc,
        "pseudo_r2": float(res.prsquared),
        "eligible_population": "valid structures with complete predictor data",
        "feature_scaling": "continuous predictors per IQR; PAINS from 0 to 1",
        "interpretation": "descriptive association; not causal",
        "coefficients": {r["feature"]: round(r["coef_logodds"], 4) for r in out_rows},
    }


def load_summary_numbers(path: Path, label: str) -> dict:
    """Load one standard summary envelope and return its numbers mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} summary not found: {path}")
    envelope = json.loads(path.read_text())
    required = {"description", "numbers", "files"}
    if not required <= set(envelope):
        raise ValueError(f"{label} is not a standard summary envelope: {path}")
    if not isinstance(envelope["numbers"], dict):
        raise ValueError(f"{label} summary numbers must be a mapping: {path}")
    return envelope["numbers"]


def validate_upstream_contracts(
    trait_summary: Path,
    functional_use_summary: Path,
    *,
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
    activity_threshold: float,
    fup_threshold: float,
    min_compounds: int,
    n_activity_compounds: int,
    n_active: int,
) -> None:
    """Confirm that the declared upstream tables use the selected activity slice."""
    trait_numbers = load_summary_numbers(trait_summary, "2.43")
    expected_trait = {
        "dataset": dataset,
        "columns": columns,
        "preprocessing": preprocessing,
        "filter_name": filter_name,
        "activity_params": activity_params,
        "activity_threshold": activity_threshold,
        "n_activity_compounds": n_activity_compounds,
        "n_active": n_active,
        "skip_brics": False,
        "min_compounds_by_family": TRAIT_MIN_COMPOUNDS,
    }
    for key, expected in expected_trait.items():
        observed = trait_numbers.get(key)
        if observed != expected:
            raise ValueError(f"2.43 summary {key}={observed!r}; expected {expected!r} for 2.45")

    functional_numbers = load_summary_numbers(functional_use_summary, "2.42")
    expected_functional = {
        "dataset": dataset,
        "columns": columns,
        "preprocessing": preprocessing,
        "filter_name": filter_name,
        "activity_params": activity_params,
        "activity_threshold": activity_threshold,
        "fup_threshold": fup_threshold,
        "min_compounds": min_compounds,
    }
    for key, expected in expected_functional.items():
        observed = functional_numbers.get(key)
        if observed != expected:
            raise ValueError(f"2.42 summary {key}={observed!r}; expected {expected!r} for 2.45")


# =============================================================================
# Main
# =============================================================================


@app.command()
def main(
    dataset: str = typer.Option("compound_no_source7", "--dataset", "-d"),
    columns: str = typer.Option("feat_all", "--columns"),
    preprocessing: str = typer.Option("activity_no_target2", "--preprocessing"),
    filter_name: str = typer.Option("all_sources", "--filter"),
    activity_params: str = typer.Option("default", "--activity-params"),
    activity_threshold: float = typer.Option(ACTIVITY_THRESHOLD, "--activity-threshold"),
    fup_threshold: float = typer.Option(0.5, "--fup-threshold"),
    min_compounds: int = typer.Option(10, "--min-compounds"),
    trait_enrichment_csv: Path | None = typer.Option(None, "--trait-enrichment", help="2.43 activity-enrichment CSV"),
    trait_summary: Path | None = typer.Option(None, "--trait-summary", help="2.43 summary envelope"),
    fup_enrichment_csv: Path | None = typer.Option(None, "--fup-enrichment", help="2.42 FUP enrichment CSV"),
    fuc_enrichment_csv: Path | None = typer.Option(None, "--fuc-enrichment", help="2.42 FUC enrichment CSV"),
    functional_use_summary: Path | None = typer.Option(None, "--functional-use-summary", help="2.42 summary envelope"),
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    multivariate: bool = typer.Option(
        False,
        "--multivariate",
        help="Also fit an optional multivariate logistic model",
    ),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi"),
) -> None:
    """Binary + multi-class compound-trait odds ratios for phenotypic activity."""
    out = output_dir or (OUTPUT_DIR / dataset)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out}")

    trait_dir = PROCESSED_DATA_DIR / "trait-activity-enrichment" / dataset
    functional_dir = PROCESSED_DATA_DIR / "functional-use-classification"
    trait_enrichment_csv = trait_enrichment_csv or trait_dir / "enrichment_activity.csv"
    trait_summary = trait_summary or trait_dir / "summary.json"
    fup_enrichment_csv = fup_enrichment_csv or functional_dir / "fup_enrichment.csv"
    fuc_enrichment_csv = fuc_enrichment_csv or functional_dir / "fuc_enrichment.csv"
    functional_use_summary = functional_use_summary or functional_dir / "summary.json"
    provenance_inputs = {
        "trait_enrichment": trait_enrichment_csv,
        "trait_summary": trait_summary,
        "fup_enrichment": fup_enrichment_csv,
        "fuc_enrichment": fuc_enrichment_csv,
        "functional_use_summary": functional_use_summary,
    }
    for label, path in provenance_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} input not found: {path}")
    # 1) Load the canonical activity slice.
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        act = query_activity_results(
            con,
            dataset,
            preprocessing=preprocessing,
            filter_name=filter_name,
            activity_params=activity_params,
            columns=columns,
        )
    finally:
        con.close()
    act = act[
        [
            "Metadata_JCP2022",
            "mean_normalized_average_precision",
            "corrected_p_value",
        ]
    ].copy()
    if act.empty or act["Metadata_JCP2022"].duplicated().any():
        raise ValueError("Activity query must return one non-empty row per JCP2022 compound")
    act["active"] = act["corrected_p_value"] < activity_threshold
    logger.info(f"Activity: {len(act):,} compounds, {act['active'].sum():,} active ({100 * act['active'].mean():.1f}%)")
    validate_upstream_contracts(
        trait_summary,
        functional_use_summary,
        dataset=dataset,
        columns=columns,
        preprocessing=preprocessing,
        filter_name=filter_name,
        activity_params=activity_params,
        activity_threshold=activity_threshold,
        fup_threshold=fup_threshold,
        min_compounds=min_compounds,
        n_activity_compounds=len(act),
        n_active=int(act["active"].sum()),
    )

    # 2) Load binary traits and memberships from the augmented metadata database.
    meta_con = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        meta = meta_con.execute(
            "SELECT Metadata_JCP2022 AS Metadata_JCP2022, "
            "Metadata_ValidMol AS valid_mol, "
            "Metadata_repurposing_clinical_phase AS phase, "
            "Metadata_Lipinski_Violations AS lipinski, "
            "Metadata_HasPAINS AS pains, "
            "Metadata_MW AS mw, Metadata_LogP AS logp, Metadata_TPSA AS tpsa, "
            "Metadata_QED AS qed, Metadata_HBD AS hbd, Metadata_HBA AS hba, "
            "Metadata_RotatableBonds AS rotatable, "
            "Metadata_NumRings AS num_rings, Metadata_NumAromaticRings AS num_aromatic, "
            "Metadata_NumHeavyAtoms AS heavy_atom_count, "
            "Metadata_FractionCSP3 AS fraction_csp3, "
            "Metadata_MurckoScaffold AS scaffold "
            "FROM compound_metadata"
        ).df()
        annotation_sets = load_annotation_memberships(meta_con)
        library_sets = load_library_memberships(meta_con)
    finally:
        meta_con.close()

    if meta["Metadata_JCP2022"].duplicated().any():
        raise ValueError("compound_metadata must contain one row per JCP2022 compound")
    df = act.merge(meta, on="Metadata_JCP2022", how="left")
    if len(df) != len(act):
        raise ValueError("Metadata merge changed the canonical activity population")
    df["pains"] = df["pains"].astype("boolean")

    intrinsic_bins = build_intrinsic_activity_nmap_bins(df)
    intrinsic_path = out / "intrinsic_activity_nmap_bins.csv"
    intrinsic_bins.to_csv(intrinsic_path, index=False, lineterminator="\n")
    logger.info(f"Saved: {intrinsic_path} ({len(intrinsic_bins)} equal-width bins)")

    selected_categorical = build_selected_categorical_activity_nmap(df, library_sets)
    selected_categorical_path = out / "selected_categorical_activity_nmap.csv"
    selected_categorical.to_csv(selected_categorical_path, index=False, lineterminator="\n")
    logger.info(f"Saved: {selected_categorical_path} ({len(selected_categorical)} fixed phase and collection rows)")

    compound_distributions = build_compound_activity_nmap_distributions(df, library_sets)
    compound_distributions_path = out / COMPOUND_DISTRIBUTION_TABLE
    compound_distributions.to_parquet(compound_distributions_path, index=False)
    logger.info(f"Saved: {compound_distributions_path} ({len(compound_distributions):,} compound-level boxplot rows)")

    # 3) Compute binary associations with explicit eligible populations.
    binary_results = build_binary_traits(df, annotation_sets, library_sets)
    binary_path = out / "binary_trait_oddsratios.csv"
    binary_results.to_csv(binary_path, index=False, lineterminator="\n")
    logger.info(
        f"Saved: {binary_path} ({len(binary_results)} traits; "
        f"{int(binary_results['candidate_display'].sum())} in the candidate display set)"
    )

    # 4) Render the candidate and complete binary forests.
    core = binary_results[binary_results["candidate_display"]]
    plot_forest(
        core,
        out,
        dpi,
        filename="forest_binary_traits",
        title="Binary compound traits vs Cell Painting activity (candidate set)",
    )
    plot_forest(
        binary_results,
        out,
        dpi,
        filename="forest_binary_traits_full",
        title="Binary compound traits vs Cell Painting activity (complete set)",
    )

    # 5) Consolidate declared 2.42 and 2.43 eligible-population counts.
    multiclass_results = build_multiclass_table(
        pd.read_csv(trait_enrichment_csv),
        pd.read_csv(fup_enrichment_csv),
        pd.read_csv(fuc_enrichment_csv),
        n_total=len(df),
        n_active=int(df["active"].sum()),
        fup_threshold=fup_threshold,
        min_compounds=min_compounds,
    )
    multiclass_path = out / "multiclass_trait_oddsratios.csv"
    multiclass_results.to_csv(multiclass_path, index=False, lineterminator="\n")
    logger.info(f"Saved: {multiclass_path} ({len(multiclass_results)} traits, all families)")
    plot_multiclass_forest(multiclass_results, out, dpi)

    # 6) Optional exploratory multivariate model.
    mv_stats = {}
    if multivariate:
        mv_stats = fit_multivariate(df, out, dpi)

    # 7) Write the canonical summary envelope with concrete provenance.
    enriched = binary_results[(binary_results["odds_ratio"] > 1) & binary_results["significant"]]
    depleted = binary_results[(binary_results["odds_ratio"] < 1) & binary_results["significant"]]
    multiclass_stats = {
        "n_traits": int(len(multiclass_results)),
        "n_significant": int(multiclass_results["significant"].sum()),
        "per_family": {
            family: {
                "n_tested": int(multiclass_results["family"].eq(family).sum()),
                "n_significant": int(
                    (multiclass_results["family"].eq(family) & multiclass_results["significant"]).sum()
                ),
                "n_eligible_min": int(
                    multiclass_results.loc[multiclass_results["family"].eq(family), "n_eligible"].min()
                ),
                "n_eligible_max": int(
                    multiclass_results.loc[multiclass_results["family"].eq(family), "n_eligible"].max()
                ),
            }
            for family in MULTICLASS_FAMILIES
        },
    }
    numbers = {
        "dataset": dataset,
        "columns": columns,
        "preprocessing": preprocessing,
        "filter_name": filter_name,
        "activity_params": activity_params,
        "activity_threshold": activity_threshold,
        "fup_threshold": fup_threshold,
        "functional_use_min_compounds": min_compounds,
        "trait_min_compounds_by_family": TRAIT_MIN_COMPOUNDS,
        "fdr_threshold": FDR_THRESHOLD,
        "n_compounds": int(len(df)),
        "n_active": int(df["active"].sum()),
        "background_active_rate": float(df["active"].mean()),
        "intrinsic_activity_nmap_bins": {
            "design": "candidate_22",
            "n_rows": int(len(intrinsic_bins)),
            "n_traits": int(intrinsic_bins["trait_id"].nunique()),
            "activity_inference": "Logistic odds ratio per IQR",
            "multiple_testing": "Benjamini-Hochberg across 13 tested intrinsic chemistry traits",
            "nmap_summary": (
                f"Median and IQR within up to {BIN_COUNT} equal-width bins spanning the "
                f"{BIN_TRIM_QUANTILES[0]:.0%}-{BIN_TRIM_QUANTILES[1]:.0%} property window, "
                f"keeping bins holding at least {MIN_BIN_FRACTION:.1%} of measured compounds"
            ),
        },
        "selected_categorical_activity_nmap": {
            "n_rows": int(len(selected_categorical)),
            "clinical_phase_rows": int(selected_categorical["view"].eq("clinical_phase").sum()),
            "curated_collection_rows": int(selected_categorical["view"].eq("curated_collections").sum()),
            "clinical_phase_design": "all_four_exact_repurposing_hub_stages",
            "curated_collections_design": "all_five_fixed_categories_regardless_of_significance",
            "chemical_probes_members": int(
                selected_categorical.loc[
                    selected_categorical["trait_id"].eq("collection.chemical_probes_sgc"), "n"
                ].item()
            ),
        },
        "compound_activity_nmap_distributions": {
            "n_rows": int(len(compound_distributions)),
            "grain": "one analyzed compound",
            "nmap_column": "mean_normalized_average_precision",
            "activity_column": "corrected_p_value < 0.1",
        },
        "n_binary_traits_tested": int(len(binary_results)),
        "n_binary_significant": int(binary_results["significant"].sum()),
        "n_binary_enriched": int(len(enriched)),
        "n_binary_depleted": int(len(depleted)),
        "binary_multiple_testing": "Benjamini-Hochberg within binary trait family",
        "multiclass_multiple_testing": "Benjamini-Hochberg within each multi-class trait family",
        "multiclass": multiclass_stats,
        "multivariate": mv_stats,
        "binary_traits": {
            r["trait"]: {
                "category": r["category"],
                "comparator": r["comparator"],
                "n_eligible": int(r["n_eligible"]),
                "n_excluded": int(r["n_excluded"]),
                "n_member": int(r["n_member"]),
                "odds_ratio": float(r["odds_ratio"]),
                "ci_lower": float(r["ci_lower"]),
                "ci_upper": float(r["ci_upper"]),
                "frac_active_member": float(r["frac_active_member"]),
                "frac_active_reference": float(r["frac_active_reference"]),
                "corrected_p_value": float(r["corrected_p_value"]),
            }
            for _, r in binary_results.iterrows()
        },
        "provenance": {
            label: {"path": str(path), "sha256": sha256_file(path)} for label, path in provenance_inputs.items()
        },
    }
    output_files = [
        intrinsic_path,
        selected_categorical_path,
        compound_distributions_path,
        binary_path,
        multiclass_path,
        out / "forest_binary_traits.png",
        out / "forest_binary_traits.pdf",
        out / "forest_binary_traits_full.png",
        out / "forest_binary_traits_full.pdf",
        out / "forest_multiclass_activity.png",
        out / "forest_multiclass_activity.pdf",
    ]
    if multivariate:
        output_files.extend(
            [
                out / "multivariate_coefficients.csv",
                out / "forest_multivariate.png",
                out / "forest_multivariate.pdf",
            ]
        )
    write_summary(
        out / "summary.json",
        description=(
            "Compound-trait associations with Cell Painting phenotypic activity, combining "
            "binary clinical, chemistry, annotation, and curated-library traits with declared "
            "2.42 functional-use and 2.43 substructure/target enrichments. Each comparison "
            "uses its explicit assessed or annotated population, a two-sided Fisher exact "
            "test, a Haldane-Anscombe odds ratio, and a Woolf 95% confidence interval."
        ),
        numbers=numbers,
        files=output_files,
    )
    logger.info(f"Saved: {out / 'summary.json'}")
    logger.success(f"Binary-trait odds-ratio analysis complete: {out}")


if __name__ == "__main__":
    app()
