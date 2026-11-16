"""Functional use classification and phenotypic activity enrichment.

Characterizes the EPA functional use categories (FUCs) and predicted functional
use probabilities (FUPs) for JUMP compounds mapped to DTXSID identifiers.
Based on Phillips et al 2017 (PMID: 30505234), data provided by Josh Harrill
(EPA/Corteva).

Two analyses:
1. Distribution of compounds across functional use categories
2. Enrichment/depletion of phenotypic activity within explicit non-null eligible populations

Usage:
    pixi run python notebooks/2.42-seal-functional-use-classification.py
    pixi run python notebooks/2.42-seal-functional-use-classification.py --fup-threshold 0.7

Outputs: data/processed/functional-use-classification/
    - functional_use_activity_nmap.csv (plot-ready functional-use contract)
    - functional_use_activity_distributions.parquet (displayed compound-level boxes)
    - existing coverage, enrichment, figure, and summary outputs
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.special import expit
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.io import COPAIRS_RESULTS_DB, METADATA_DB
from jump_production.queries import query_activity_results
from jump_production.summaries import index_files, write_summary

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = PROCESSED_DATA_DIR / "functional-use-classification"

ACTIVITY_P_THRESHOLD = 0.10
DEFAULT_FUP_THRESHOLD = 0.5
FIGURE_FDR_THRESHOLD = 0.05
FIGURE_MIN_CATEGORY_MEMBERS = 20
FIGURE_MIN_NUMERIC_N = 50
FIGURE_MIN_CELL_COUNT = 5
FIGURE_MIN_UNIQUE_PROBABILITIES = 2

FUP_COLS = [
    "antimicrobial",
    "heat_stabilizer",
    "catalyst",
    "colorant",
    "skin_conditioner",
    "uv_absorber",
    "buffer",
    "flavorant",
    "skin_protectant",
    "antioxidant",
    "fragrance",
    "flame_retardant",
    "oxidizer",
    "photoinitiator",
    "crosslinker",
    "preservative",
    "organic_pigment",
    "chelator",
    "emollient",
    "hair_conditioner",
    "humectant",
    "hair_dye",
    "reducer",
    "surfactant",
    "additive",
    "monomer",
    "lubricating_agent",
    "foamer",
    "emulsifier",
    "emulsion_stabilizer",
    "whitener",
    "soluble_dye",
    "antistatic_agent",
    "foam_boosting_agent",
    "vinyl",
]

app = typer.Typer()


# =============================================================================
# Data Loading
# =============================================================================


def load_functional_use_data(meta_con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load functional use data from the `functional_use` table in the augmented DB.

    Columns are aliased back to their source names (JCP2022, functioncategory_all,
    and the FUP_COLS) so downstream code is unchanged. Curated categories arrive as
    NULL (the ingest already maps 'NA' -> NULL).

    The current augmented database is required because the Snakemake rule declares
    the enrichment CSVs as downstream inputs.
    """
    fup_select = ", ".join(f"Metadata_funcuse_{c} AS {c}" for c in FUP_COLS)
    try:
        df = meta_con.execute(
            "SELECT Metadata_JCP2022 AS JCP2022, "
            "Metadata_funcuse_dtxsid AS DTXSID, "
            "Metadata_funcuse_category AS functioncategory_all, "
            f"{fup_select} FROM functional_use"
        ).df()
    except duckdb.CatalogException as exc:
        raise RuntimeError(
            "functional_use table not in augmented DB; run just get-inputs, then rebuild the augmented metadata DB"
        ) from exc
    if df.empty:
        raise ValueError("functional_use table is empty")
    for col in FUP_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_with_fuc = df["functioncategory_all"].notna().sum()
    n_with_fup = df[FUP_COLS].notna().any(axis=1).sum()
    logger.info(
        f"Loaded functional use data: {len(df):,} compounds, "
        f"{n_with_fuc:,} with curated FUC, {n_with_fup:,} with FUP predictions"
    )
    return df


def load_activity_data(
    dataset: str,
    columns: str,
    preprocessing: str,
    filter_name: str,
    activity_params: str,
) -> pd.DataFrame:
    """Load activity results from copairs database."""
    con = duckdb.connect(str(COPAIRS_RESULTS_DB), read_only=True)
    try:
        activity_df = query_activity_results(
            con,
            dataset,
            preprocessing=preprocessing,
            filter_name=filter_name,
            activity_params=activity_params,
            columns=columns,
        )
    finally:
        con.close()
    activity_df["is_active"] = activity_df["corrected_p_value"] < ACTIVITY_P_THRESHOLD
    n_active = activity_df["is_active"].sum()
    logger.info(
        f"Activity data: {len(activity_df):,} compounds, {n_active:,} active ({100 * n_active / len(activity_df):.1f}%)"
    )
    return activity_df


# =============================================================================
# Analysis
# =============================================================================


def compute_enrichment(
    merged_df: pd.DataFrame,
    category_col: str,
    threshold: float,
    min_compounds: int,
) -> pd.DataFrame | None:
    """Compute activity enrichment for a single FUP category.

    Returns one row with a Haldane-adjusted odds ratio and exact Fisher p-value.
    """
    eligible = merged_df.loc[merged_df[category_col].notna()].copy()
    has_value = eligible[category_col] >= threshold
    n_eligible = len(eligible)
    n_missing = len(merged_df) - n_eligible
    n_high = int(has_value.sum())
    n_rest = n_eligible - n_high
    if n_high < min_compounds:
        return None

    high_active = int((has_value & eligible["is_active"]).sum())
    high_inactive = int((has_value & ~eligible["is_active"]).sum())
    rest_active = int((~has_value & eligible["is_active"]).sum())
    rest_inactive = int((~has_value & ~eligible["is_active"]).sum())

    table = [[high_active, high_inactive], [rest_active, rest_inactive]]
    _, p_value = fisher_exact(table, alternative="two-sided")

    # Woolf method for CI with Haldane correction
    a, b, c, d = high_active + 0.5, high_inactive + 0.5, rest_active + 0.5, rest_inactive + 0.5
    log_or = np.log(a * d / (b * c))
    odds_ratio = np.exp(log_or)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z_value = float(norm.ppf(0.975))
    ci_lower = np.exp(log_or - z_value * se)
    ci_upper = np.exp(log_or + z_value * se)

    return pd.DataFrame(
        [
            {
                "category": category_col,
                "threshold": threshold,
                "n_eligible": n_eligible,
                "n_missing": n_missing,
                "n_high": n_high,
                "n_rest": n_rest,
                "n_active_high": high_active,
                "n_active_rest": rest_active,
                "frac_active_high": high_active / n_high if n_high > 0 else 0,
                "frac_active_rest": rest_active / n_rest if n_rest > 0 else 0,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log_or / np.log(2),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
            }
        ]
    )


def compute_all_fup_enrichments(
    merged_df: pd.DataFrame,
    fup_threshold: float,
    min_compounds: int,
) -> pd.DataFrame:
    """Compute activity enrichment for all 35 FUP categories."""
    results = []
    for col in FUP_COLS:
        row = compute_enrichment(merged_df, col, fup_threshold, min_compounds)
        if row is not None:
            results.append(row)

    if not results:
        logger.warning("No FUP categories met the minimum compound threshold")
        return pd.DataFrame()

    enrichment_df = pd.concat(results, ignore_index=True)

    _, corrected_p, _, _ = multipletests(enrichment_df["p_value"], method="fdr_bh")
    enrichment_df["corrected_p_value"] = corrected_p
    enrichment_df["significant"] = enrichment_df["corrected_p_value"] < 0.05

    n_sig = enrichment_df["significant"].sum()
    n_enriched = ((enrichment_df["significant"]) & (enrichment_df["odds_ratio"] > 1)).sum()
    n_depleted = ((enrichment_df["significant"]) & (enrichment_df["odds_ratio"] < 1)).sum()
    logger.info(
        f"FUP enrichment: {n_sig} significant ({n_enriched} enriched, {n_depleted} depleted) "
        f"out of {len(enrichment_df)} tested"
    )
    return enrichment_df.sort_values(["odds_ratio", "category"], ascending=[False, True], kind="stable")


def compute_fuc_enrichments(
    merged_df: pd.DataFrame,
    min_compounds: int,
) -> pd.DataFrame:
    """Compute activity enrichment for curated FUC categories."""
    eligible = merged_df.loc[merged_df["functioncategory_all"].notna()].copy()
    eligible["_fuc_tokens"] = eligible["functioncategory_all"].map(
        lambda value: tuple(sorted({token.strip() for token in str(value).split("|") if token.strip()}))
    )
    all_cats = eligible["_fuc_tokens"].explode().value_counts()
    eligible_cats = all_cats[all_cats >= min_compounds].index.tolist()
    n_eligible = len(eligible)
    n_missing = len(merged_df) - n_eligible

    if not eligible_cats:
        logger.warning("No curated FUC categories met the minimum compound threshold")
        return pd.DataFrame()

    results = []
    for cat in eligible_cats:
        has_cat = eligible["_fuc_tokens"].map(lambda tokens: cat in tokens)
        n_high = int(has_cat.sum())
        n_rest = n_eligible - n_high
        if n_high < min_compounds:
            continue

        high_active = int((has_cat & eligible["is_active"]).sum())
        high_inactive = int((has_cat & ~eligible["is_active"]).sum())
        rest_active = int((~has_cat & eligible["is_active"]).sum())
        rest_inactive = int((~has_cat & ~eligible["is_active"]).sum())

        table = [[high_active, high_inactive], [rest_active, rest_inactive]]
        _, p_value = fisher_exact(table, alternative="two-sided")

        a, b, c, d = high_active + 0.5, high_inactive + 0.5, rest_active + 0.5, rest_inactive + 0.5
        log_or = np.log(a * d / (b * c))
        odds_ratio = np.exp(log_or)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

        results.append(
            {
                "category": cat,
                "n_eligible": n_eligible,
                "n_missing": n_missing,
                "n_high": n_high,
                "n_rest": n_rest,
                "n_active_high": high_active,
                "n_active_rest": rest_active,
                "frac_active_high": high_active / n_high if n_high > 0 else 0,
                "frac_active_rest": rest_active / n_rest if n_rest > 0 else 0,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log_or / np.log(2),
                "ci_lower": np.exp(log_or - float(norm.ppf(0.975)) * se),
                "ci_upper": np.exp(log_or + float(norm.ppf(0.975)) * se),
                "p_value": p_value,
            }
        )

    if not results:
        return pd.DataFrame()

    fuc_df = pd.DataFrame(results)
    _, corrected_p, _, _ = multipletests(fuc_df["p_value"], method="fdr_bh")
    fuc_df["corrected_p_value"] = corrected_p
    fuc_df["significant"] = fuc_df["corrected_p_value"] < 0.05

    n_sig = fuc_df["significant"].sum()
    logger.info(f"FUC enrichment: {n_sig} significant out of {len(fuc_df)} tested")
    return fuc_df.sort_values(["odds_ratio", "category"], ascending=[False, True], kind="stable")


def functional_display_name(value: str) -> str:
    """Return the stable reader-facing label used by candidate_05."""
    text = str(value).replace("epa.fup::", "").replace("_", " ").title()
    return text.replace("Uv ", "UV ").replace("Ph ", "pH ").replace("Epa", "EPA")


def functional_wilson_bounds(successes: int, total: int) -> tuple[float, float]:
    """Return a two-sided Wilson 95% interval as percentages."""
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z_value = float(norm.ppf(0.975))
    rate = successes / total
    denominator = 1 + z_value**2 / total
    center = (rate + z_value**2 / (2 * total)) / denominator
    half_width = z_value * np.sqrt(rate * (1 - rate) / total + z_value**2 / (4 * total**2)) / denominator
    return 100 * float(center - half_width), 100 * float(center + half_width)


def functional_logistic_or_per_iqr(values: pd.Series, outcomes: pd.Series) -> tuple[float, float, float, float]:
    """Fit the candidate_05 continuous FUP activity model."""
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
    )


def functional_group_summary(merged_df: pd.DataFrame, member: pd.Series, eligible: pd.Series) -> dict[str, float | int]:
    """Summarize one displayed class and its within-source reference."""
    eligible = eligible.fillna(False).astype(bool)
    member = member.fillna(False).astype(bool) & eligible
    reference = eligible & ~member
    member_frame = merged_df.loc[member]
    reference_frame = merged_df.loc[reference]
    if member_frame.empty or reference_frame.empty:
        raise ValueError("Functional-use summary requires exposed and reference compounds")
    active_exposed = int(member_frame["is_active"].astype(bool).sum())
    active_reference = int(reference_frame["is_active"].astype(bool).sum())
    exposed_low, exposed_high = functional_wilson_bounds(active_exposed, len(member_frame))
    reference_low, reference_high = functional_wilson_bounds(active_reference, len(reference_frame))
    return {
        "n_exposed": int(len(member_frame)),
        "n_reference": int(len(reference_frame)),
        "active_exposed": active_exposed,
        "active_reference": active_reference,
        "rate_exposed": 100 * active_exposed / len(member_frame),
        "rate_reference": 100 * active_reference / len(reference_frame),
        "rate_ci_low": exposed_low,
        "rate_ci_high": exposed_high,
        "reference_ci_low": reference_low,
        "reference_ci_high": reference_high,
        "nmap_median": float(member_frame["mean_normalized_average_precision"].median()),
        "nmap_q25": float(member_frame["mean_normalized_average_precision"].quantile(0.25)),
        "nmap_q75": float(member_frame["mean_normalized_average_precision"].quantile(0.75)),
        "reference_nmap_median": float(reference_frame["mean_normalized_average_precision"].median()),
        "reference_nmap_q25": float(reference_frame["mean_normalized_average_precision"].quantile(0.25)),
        "reference_nmap_q75": float(reference_frame["mean_normalized_average_precision"].quantile(0.75)),
    }


def build_functional_use_activity_nmap(
    merged_df: pd.DataFrame,
    fup_display_threshold: float = DEFAULT_FUP_THRESHOLD,
) -> pd.DataFrame:
    """Build functional-use FUC and continuous-FUP plot rows.

    FUC inference is one-vs-rest Fisher testing among compounds with any curated
    FUC. FUP inference is a logistic odds ratio per IQR of continuous
    probability. The FUP threshold is used only for displayed group summaries,
    never for the inferential estimate or its FDR value.
    """
    if not 0 <= fup_display_threshold <= 1:
        raise ValueError("FUP display threshold must be between 0 and 1")
    required = {
        "JCP2022",
        "functioncategory_all",
        "is_active",
        "mean_normalized_average_precision",
        *FUP_COLS,
    }
    missing = sorted(required - set(merged_df.columns))
    if missing:
        raise ValueError(f"Functional-use plot input is missing columns: {missing}")
    if merged_df["JCP2022"].duplicated().any():
        raise ValueError("Functional-use plot input must contain one row per compound")

    fuc_eligible = merged_df["functioncategory_all"].notna()
    tokens = (
        merged_df["functioncategory_all"]
        .fillna("")
        .map(lambda value: tuple(sorted({item.strip() for item in str(value).split("|") if item.strip()})))
    )
    fuc_rows: list[dict[str, object]] = []
    for level in sorted(set(tokens.loc[fuc_eligible].explode())):
        member = tokens.map(lambda values: level in values) & fuc_eligible
        reference = fuc_eligible & ~member
        active = merged_df["is_active"].astype(bool)
        cells = (
            int((member & active).sum()),
            int((member & ~active).sum()),
            int((reference & active).sum()),
            int((reference & ~active).sum()),
        )
        if min(int(member.sum()), int(reference.sum())) < FIGURE_MIN_CATEGORY_MEMBERS:
            continue
        if min(cells) < FIGURE_MIN_CELL_COUNT:
            continue
        _, p_value = fisher_exact([[cells[0], cells[1]], [cells[2], cells[3]]], alternative="two-sided")
        adjusted = np.asarray(cells, dtype=float) + 0.5
        estimate = float(adjusted[0] * adjusted[3] / (adjusted[1] * adjusted[2]))
        fuc_rows.append(
            {
                "section_order": 0,
                "section": "FUC",
                "category": functional_display_name(level),
                "source_level": level,
                "p_value": float(p_value),
                "inference_estimate": estimate,
                **functional_group_summary(merged_df, member, fuc_eligible),
            }
        )

    fup_rows: list[dict[str, object]] = []
    for column in FUP_COLS:
        eligible = merged_df[column].notna()
        tested = merged_df.loc[eligible, [column, "is_active"]].copy()
        active = tested["is_active"].astype(bool)
        if len(tested) < FIGURE_MIN_NUMERIC_N:
            continue
        if min(int(active.sum()), int((~active).sum())) < FIGURE_MIN_CELL_COUNT:
            continue
        if tested[column].nunique() < FIGURE_MIN_UNIQUE_PROBABILITIES:
            continue
        try:
            estimate, _, _, p_value = functional_logistic_or_per_iqr(tested[column], active)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        member = merged_df[column].ge(fup_display_threshold) & eligible
        if not member.any() or not (eligible & ~member).any():
            raise ValueError(
                f"FUP {column} cannot form the displayed probability >= {fup_display_threshold:g} contrast"
            )
        fup_rows.append(
            {
                "section_order": 1,
                "section": "FUP",
                "category": functional_display_name(column),
                "source_level": column,
                "p_value": p_value,
                "inference_estimate": estimate,
                **functional_group_summary(merged_df, member, eligible),
            }
        )

    sections = []
    for rows in (fuc_rows, fup_rows):
        if not rows:
            raise ValueError("Functional-use plot-ready contract has an empty inference family")
        frame = pd.DataFrame(rows)
        frame["q_value"] = multipletests(frame["p_value"], method="fdr_bh")[1]
        frame["significant"] = frame["q_value"] < FIGURE_FDR_THRESHOLD
        frame = frame.sort_values(
            ["inference_estimate", "category"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        frame.insert(2, "row_order", np.arange(len(frame)))
        frame["rate_delta"] = frame["rate_exposed"] - frame["rate_reference"]
        frame["nmap_delta"] = frame["nmap_median"] - frame["reference_nmap_median"]
        sections.append(frame.drop(columns="p_value"))
    return (
        pd.concat(sections, ignore_index=True)
        .sort_values(["section_order", "row_order"], kind="stable")
        .reset_index(drop=True)
    )


def build_functional_use_activity_distributions(
    merged_df: pd.DataFrame,
    summary: pd.DataFrame,
    fup_display_threshold: float = DEFAULT_FUP_THRESHOLD,
) -> pd.DataFrame:
    """Build compound-level nMAP rows for every displayed FUC and FUP box."""
    if not 0 <= fup_display_threshold <= 1:
        raise ValueError("FUP display threshold must be between 0 and 1")
    required = {
        "JCP2022",
        "functioncategory_all",
        "is_active",
        "mean_normalized_average_precision",
        *FUP_COLS,
    }
    missing = sorted(required - set(merged_df.columns))
    if missing:
        raise ValueError(f"Functional-use distribution input is missing columns: {missing}")
    require_summary = {"section", "category", "source_level", "significant", "n_exposed", "active_exposed"}
    missing = sorted(require_summary - set(summary.columns))
    if missing:
        raise ValueError(f"Functional-use summary is missing distribution columns: {missing}")
    tokens = (
        merged_df["functioncategory_all"]
        .fillna("")
        .map(lambda value: frozenset(item.strip() for item in str(value).split("|") if item.strip()))
    )
    blocks: list[pd.DataFrame] = []
    displayed = summary.loc[summary["section"].eq("FUC") | summary["significant"].astype(bool)]
    for row in displayed.itertuples(index=False):
        if row.section == "FUC":
            member = tokens.map(lambda values: str(row.source_level) in values)
        else:
            member = merged_df[str(row.source_level)].ge(fup_display_threshold)
        block = merged_df.loc[
            member,
            ["JCP2022", "mean_normalized_average_precision", "is_active"],
        ].rename(
            columns={
                "JCP2022": "jcp2022",
                "mean_normalized_average_precision": "nmap",
                "is_active": "active",
            }
        )
        if len(block) != int(row.n_exposed) or int(block["active"].sum()) != int(row.active_exposed):
            raise ValueError(f"Functional-use distribution disagrees with {row.section} {row.category}")
        block.insert(0, "source_level", str(row.source_level))
        block.insert(0, "category", str(row.category))
        block.insert(0, "section", str(row.section))
        blocks.append(block)
    return (
        pd.concat(blocks, ignore_index=True)
        .sort_values(["section", "category", "jcp2022"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_functional_use_activity(
    frame: pd.DataFrame,
    *,
    fup_display_threshold: float,
) -> dict[str, object]:
    """Return the plot-ready functional-use contract from its computed rows."""
    required = {"section", "significant", "inference_estimate"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Functional-use plot summary is missing columns: {missing}")
    fuc = frame.loc[frame["section"].eq("FUC")]
    fup = frame.loc[frame["section"].eq("FUP")]
    significant_fup = fup.loc[fup["significant"].astype(bool)]
    return {
        "design": "candidate_05",
        "n_rows": int(len(frame)),
        "n_fuc_tested": int(len(fuc)),
        "n_fuc_significant": int(fuc["significant"].astype(bool).sum()),
        "n_fup_tested": int(len(fup)),
        "n_fup_significant": int(len(significant_fup)),
        "n_fup_higher": int(significant_fup["inference_estimate"].gt(1).sum()),
        "n_fup_lower": int(significant_fup["inference_estimate"].lt(1).sum()),
        "fdr_threshold": FIGURE_FDR_THRESHOLD,
        "fuc_inference": "Two-sided one-vs-rest Fisher exact test",
        "fuc_eligible_population": "Compounds with at least one curated FUC annotation",
        "fuc_testability": {
            "minimum_members_per_group": FIGURE_MIN_CATEGORY_MEMBERS,
            "minimum_contingency_cell_count": FIGURE_MIN_CELL_COUNT,
        },
        "fup_inference": "Logistic odds ratio per IQR of continuous probability",
        "fup_eligible_population": "Compounds with a non-null prediction for the individual endpoint",
        "fup_testability": {
            "minimum_non_null_compounds": FIGURE_MIN_NUMERIC_N,
            "minimum_outcomes_per_activity_class": FIGURE_MIN_CELL_COUNT,
            "minimum_unique_probabilities": FIGURE_MIN_UNIQUE_PROBABILITIES,
            "requires_positive_iqr": True,
        },
        "fup_display_threshold": float(fup_display_threshold),
        "fup_display_membership": f"Probability >= {fup_display_threshold:g}",
        "multiple_testing": "Benjamini-Hochberg separately across tested FUC and FUP relationships",
    }


# =============================================================================
# Figures
# =============================================================================


def plot_coverage_summary(
    n_total: int,
    n_dtxsid: int,
    n_in_analysis: int,
    n_fuc: int,
    n_fup: int,
    output_dir: Path,
    dpi: int,
) -> None:
    """Plot canonical, delivered, overlap, and annotated functional-use coverage."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = [
        "Canonical activity compounds\n(no source 7 profiles)",
        "DTXSID-mapped\n(full delivery)",
        "Functional-use overlap\n(canonical activity)",
        "With curated FUC",
        "With FUP predictions",
    ]
    values = [n_total, n_dtxsid, n_in_analysis, n_fuc, n_fup]
    colors = ["#4472C4", "#5B9BD5", "#70AD47", "#ED7D31", "#FFC000"]

    bars = ax.barh(range(len(labels)), values, color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of compounds")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, max(values) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"coverage_summary{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: coverage_summary.png")


def plot_fuc_distribution(
    func_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    top_n: int = 25,
) -> None:
    """Bar chart of top curated functional use categories."""
    fuc_series = func_df["functioncategory_all"].dropna().str.split("|")
    all_cats = fuc_series.explode().str.strip().value_counts()

    top_cats = all_cats.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_cats) + 1))
    y_pos = range(len(top_cats))
    ax.barh(y_pos, top_cats.values, color="#4472C4", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cats.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of compounds")
    ax.set_title(f"Top {top_n} curated functional use categories (of {len(all_cats)} total)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(top_cats.values):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"fuc_distribution{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fuc_distribution.png")


def plot_fup_distribution(
    func_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Box plots of FUP probability distributions and coverage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: box plots of FUP probabilities
    fup_data = []
    fup_labels = []
    medians = []
    for col in FUP_COLS:
        vals = func_df[col].dropna().values
        if len(vals) > 0:
            fup_data.append(vals)
            fup_labels.append(col.replace("_", " ").title())
            medians.append(np.median(vals))

    sort_idx = np.argsort(medians)[::-1]
    fup_data = [fup_data[i] for i in sort_idx]
    fup_labels = [fup_labels[i] for i in sort_idx]

    bp = ax1.boxplot(
        fup_data,
        vert=False,
        tick_labels=fup_labels,
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 2, "alpha": 0.3},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#5B9BD5")
        patch.set_alpha(0.7)

    ax1.set_xlabel("Predicted probability")
    ax1.set_title("A. FUP probability distributions")
    ax1.invert_yaxis()
    ax1.tick_params(axis="y", labelsize=8)

    # Panel B: coverage (n compounds with non-NA)
    coverage = []
    for col in FUP_COLS:
        coverage.append(func_df[col].notna().sum())
    coverage_sorted = [
        coverage[FUP_COLS.index(fup_labels[i].replace(" ", "_").lower())] for i in range(len(fup_labels))
    ]

    ax2.barh(range(len(fup_labels)), coverage_sorted, color="#70AD47", height=0.6)
    ax2.set_yticks([])
    ax2.set_xlabel("Compounds with prediction")
    ax2.set_title("B. Coverage")
    ax2.invert_yaxis()

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"fup_distribution{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fup_distribution.png")


def plot_dominant_class(
    func_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    """Bar chart of the highest available non-null FUP class per compound."""
    fup_subset = func_df[FUP_COLS].copy()
    has_any = fup_subset.notna().any(axis=1)
    fup_valid = fup_subset[has_any]

    dominant = fup_valid.idxmax(axis=1)
    dominant_counts = dominant.value_counts()

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(dominant_counts) + 1))
    labels = [c.replace("_", " ").title() for c in dominant_counts.index]
    ax.barh(range(len(labels)), dominant_counts.values, color="#ED7D31", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of compounds")
    ax.set_title(f"Highest available predicted functional class (n={len(fup_valid):,})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(dominant_counts.values):
        ax.text(v + max(dominant_counts.values) * 0.01, i, f"{v:,}", va="center", fontsize=8)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"dominant_class_distribution{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: dominant_class_distribution.png")


def plot_forest(
    enrichment_df: pd.DataFrame,
    title: str,
    filename: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Forest plot of log2(OR) with 95% CI."""
    if enrichment_df.empty:
        logger.warning(f"Skipping forest plot ({filename}): no data")
        return

    df = enrichment_df.sort_values("log2_odds_ratio").copy()

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.5))
    y_pos = range(len(df))

    colors = []
    for _, row in df.iterrows():
        if row["significant"] and row["odds_ratio"] > 1:
            colors.append("#D62728")
        elif row["significant"] and row["odds_ratio"] < 1:
            colors.append("#1F77B4")
        else:
            colors.append("#999999")

    labels = [c.replace("_", " ").title() for c in df["category"]]

    for i, (_, row) in enumerate(df.iterrows()):
        ci_lo = np.log2(row["ci_lower"]) if row["ci_lower"] > 0 else -5
        ci_hi = np.log2(row["ci_upper"]) if row["ci_upper"] > 0 else 5
        ax.plot([ci_lo, ci_hi], [i, i], color=colors[i], linewidth=1.5, zorder=1)

    ax.scatter(df["log2_odds_ratio"].values, y_pos, c=colors, s=40, zorder=2, edgecolors="white", linewidth=0.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("log2(Haldane-adjusted odds ratio)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"{filename}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {filename}.png")


def plot_volcano(
    enrichment_df: pd.DataFrame,
    title: str,
    filename: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Volcano plot: log2(OR) vs -log10(corrected p-value)."""
    if enrichment_df.empty:
        logger.warning(f"Skipping volcano plot ({filename}): no data")
        return

    df = enrichment_df.copy()
    df["neg_log10_p"] = -np.log10(df["corrected_p_value"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        if row["significant"] and row["odds_ratio"] > 1:
            color = "#D62728"
        elif row["significant"] and row["odds_ratio"] < 1:
            color = "#1F77B4"
        else:
            color = "#999999"
        ax.scatter(
            row["log2_odds_ratio"], row["neg_log10_p"], c=color, s=50, edgecolors="white", linewidth=0.5, zorder=2
        )

        if row["significant"]:
            label = row["category"].replace("_", " ").title()
            ax.annotate(
                label,
                (row["log2_odds_ratio"], row["neg_log10_p"]),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-np.log10(0.05), color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("log2(Haldane-adjusted odds ratio)")
    ax.set_ylabel("-log₁₀(FDR-corrected p-value)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(output_dir / f"{filename}{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {filename}.png")


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
    fup_threshold: float = typer.Option(
        DEFAULT_FUP_THRESHOLD, "--fup-threshold", help="FUP probability threshold for high classification"
    ),
    min_compounds: int = typer.Option(10, "--min-compounds", help="Minimum compounds per category for enrichment test"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi"),
) -> None:
    """Functional use classification and activity enrichment analysis."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    meta = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        func_df = load_functional_use_data(meta)
    finally:
        meta.close()
    activity_df = load_activity_data(
        dataset,
        columns,
        preprocessing,
        filter_name,
        activity_params,
    )

    # 2. Merge
    merged_df = func_df.merge(
        activity_df[["Metadata_JCP2022", "mean_normalized_average_precision", "corrected_p_value", "is_active"]],
        left_on="JCP2022",
        right_on="Metadata_JCP2022",
        how="inner",
    )
    merged_df.drop(columns=["Metadata_JCP2022"], inplace=True)

    n_with_fuc = merged_df["functioncategory_all"].notna().sum()
    n_with_fup = merged_df[FUP_COLS].notna().any(axis=1).sum()
    overlap_activity_rate = merged_df["is_active"].mean()
    logger.info(
        f"Merged: {len(merged_df):,} compounds in analysis "
        f"({n_with_fuc:,} with FUC, {n_with_fup:,} with FUP), "
        f"functional-use overlap activity rate: {overlap_activity_rate:.1%}"
    )

    # 3. Coverage summary figure
    n_total_activity = len(activity_df)
    plot_coverage_summary(
        n_total=n_total_activity,
        n_dtxsid=len(func_df),
        n_in_analysis=len(merged_df),
        n_fuc=n_with_fuc,
        n_fup=n_with_fup,
        output_dir=out,
        dpi=dpi,
    )

    # 4. FUC distribution
    plot_fuc_distribution(merged_df, out, dpi)

    # 5. FUP distribution
    plot_fup_distribution(merged_df, out, dpi)

    # 6. Dominant predicted class
    plot_dominant_class(merged_df, out, dpi)

    # 7. FUP enrichment
    fup_enrichment = compute_all_fup_enrichments(merged_df, fup_threshold, min_compounds)
    if not fup_enrichment.empty:
        fup_enrichment.to_csv(out / "fup_enrichment.csv", index=False)
        fup_title = "FUP Activity Enrichment (other compounds with a non-null prediction)"
        plot_forest(fup_enrichment, fup_title, "fup_enrichment_forest", out, dpi)
        plot_volcano(fup_enrichment, fup_title, "fup_enrichment_volcano", out, dpi)

    # 8. FUC enrichment
    fuc_enrichment = compute_fuc_enrichments(merged_df, min_compounds)
    if not fuc_enrichment.empty:
        fuc_enrichment.to_csv(out / "fuc_enrichment.csv", index=False)
        fuc_title = "FUC Activity Enrichment (other compounds with any curated FUC)"
        plot_forest(fuc_enrichment, fuc_title, "fuc_enrichment_forest", out, dpi)
        plot_volcano(fuc_enrichment, fuc_title, "fuc_enrichment_volcano", out, dpi)

    functional_plot_table = build_functional_use_activity_nmap(
        merged_df,
        fup_display_threshold=fup_threshold,
    )
    functional_plot_path = out / "functional_use_activity_nmap.csv"
    functional_plot_table.to_csv(functional_plot_path, index=False, lineterminator="\n")
    logger.info(f"Saved: {functional_plot_path} ({len(functional_plot_table)} tested relationships)")
    functional_distributions = build_functional_use_activity_distributions(
        merged_df,
        functional_plot_table,
        fup_display_threshold=fup_threshold,
    )
    functional_distribution_path = out / "functional_use_activity_distributions.parquet"
    functional_distributions.to_parquet(functional_distribution_path, index=False)
    logger.info(f"Saved: {functional_distribution_path} ({len(functional_distributions):,} displayed memberships)")

    # 9. Summary
    full_fuc_cats = func_df["functioncategory_all"].dropna().str.split("|").explode().str.strip().value_counts()
    fuc_cats = merged_df["functioncategory_all"].dropna().str.split("|").explode().str.strip().value_counts()
    summary = {
        "dataset": dataset,
        "columns": columns,
        "preprocessing": preprocessing,
        "filter_name": filter_name,
        "activity_params": activity_params,
        "activity_threshold": ACTIVITY_P_THRESHOLD,
        "fup_threshold": fup_threshold,
        "min_compounds": min_compounds,
        "n_canonical_activity_compounds": n_total_activity,
        "n_dtxsid_mapped_full_delivery": len(func_df),
        "n_functional_use_overlap": len(merged_df),
        "n_with_curated_fuc_in_analysis": int(n_with_fuc),
        "n_with_fup_prediction_in_analysis": int(n_with_fup),
        "n_fuc_categories_full_delivery": len(full_fuc_cats),
        "n_fuc_categories_in_analysis": len(fuc_cats),
        "top_10_fuc_categories_in_analysis": {k: int(v) for k, v in fuc_cats.head(10).items()},
        "activity_rate_functional_use_overlap": float(overlap_activity_rate),
        "functional_use_activity_nmap": summarize_functional_use_activity(
            functional_plot_table,
            fup_display_threshold=fup_threshold,
        ),
        "functional_use_activity_distributions": {
            "n_rows": int(len(functional_distributions)),
            "n_displayed_groups": int(functional_distributions[["section", "source_level"]].drop_duplicates().shape[0]),
            "grain": "one displayed functional-use group membership by compound",
        },
        "enrichment_method": {
            "test": "Two-sided Fisher exact test within each eligible population",
            "odds_ratio": "Haldane-Anscombe corrected point estimate with Woolf 95% confidence interval",
            "multiple_testing": "Benjamini-Hochberg correction separately across tested FUP and FUC classes",
        },
        "fup_enrichment": {
            "n_tested": len(fup_enrichment) if not fup_enrichment.empty else 0,
            "n_significant": int(fup_enrichment["significant"].sum()) if not fup_enrichment.empty else 0,
            "n_enriched": int(((fup_enrichment["significant"]) & (fup_enrichment["odds_ratio"] > 1)).sum())
            if not fup_enrichment.empty
            else 0,
            "n_depleted": int(((fup_enrichment["significant"]) & (fup_enrichment["odds_ratio"] < 1)).sum())
            if not fup_enrichment.empty
            else 0,
            "significant_enriched_categories": sorted(
                fup_enrichment.loc[
                    fup_enrichment["significant"] & (fup_enrichment["odds_ratio"] > 1), "category"
                ].tolist()
            )
            if not fup_enrichment.empty
            else [],
            "significant_depleted_categories": sorted(
                fup_enrichment.loc[
                    fup_enrichment["significant"] & (fup_enrichment["odds_ratio"] < 1), "category"
                ].tolist()
            )
            if not fup_enrichment.empty
            else [],
            "eligible_population": "Non-null prediction for the individual FUP endpoint",
            "n_eligible_min": int(fup_enrichment["n_eligible"].min()) if not fup_enrichment.empty else 0,
            "n_eligible_max": int(fup_enrichment["n_eligible"].max()) if not fup_enrichment.empty else 0,
        },
        "fuc_enrichment": {
            "n_tested": len(fuc_enrichment) if not fuc_enrichment.empty else 0,
            "n_significant": int(fuc_enrichment["significant"].sum()) if not fuc_enrichment.empty else 0,
            "n_enriched": int(((fuc_enrichment["significant"]) & (fuc_enrichment["odds_ratio"] > 1)).sum())
            if not fuc_enrichment.empty
            else 0,
            "n_depleted": int(((fuc_enrichment["significant"]) & (fuc_enrichment["odds_ratio"] < 1)).sum())
            if not fuc_enrichment.empty
            else 0,
            "significant_enriched_categories": sorted(
                fuc_enrichment.loc[
                    fuc_enrichment["significant"] & (fuc_enrichment["odds_ratio"] > 1), "category"
                ].tolist()
            )
            if not fuc_enrichment.empty
            else [],
            "significant_depleted_categories": sorted(
                fuc_enrichment.loc[
                    fuc_enrichment["significant"] & (fuc_enrichment["odds_ratio"] < 1), "category"
                ].tolist()
            )
            if not fuc_enrichment.empty
            else [],
            "eligible_population": "Compounds with at least one curated FUC",
            "n_eligible": int(n_with_fuc),
        },
    }

    write_summary(
        out / "summary.json",
        description=(
            "Functional use classification and phenotypic activity enrichment for JUMP compounds "
            "mapped to DTXSID identifiers, using EPA functional use categories (FUCs) and predicted "
            "functional use probabilities (FUPs) from Phillips et al 2017. Reports compound coverage "
            "across functional use categories and Fisher-exact enrichment/depletion of phenotypic "
            "activity by category within explicit non-null eligible populations."
        ),
        numbers=summary,
        files=index_files(out),
    )
    logger.info("Saved: summary.json")
    logger.info("Done!")


if __name__ == "__main__":
    app()
