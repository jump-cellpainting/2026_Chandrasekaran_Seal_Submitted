"""Render compound-activity association composites and modular panels.

This ordinary script consumes plot-ready tables owned by analyses 2.42, 2.43,
and 2.45. It does not query databases or recompute scientific estimands.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image as PILImage
from PIL import ImageDraw
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from jump_production.config import DEFAULT_DPI, PROCESSED_DATA_DIR
from jump_production.summaries import write_legend, write_summary
from jump_production.visualization import (
    PROPERTY_FIGURE_COLOR,
    PROPERTY_FIGURE_PANEL_HEIGHT_IN,
    PROPERTY_FIGURE_TYPE,
    PROPERTY_FIGURE_WIDTH_IN,
    property_figure_caption_layout,
)

RDLogger.DisableLog("rdApp.*")

FIGURE_ID = "figure_compound_activity_associations"
OUTPUT_DIR = PROCESSED_DATA_DIR / "figures" / FIGURE_ID
COMPOSITE_OUTPUT_STEM = "compound_properties_activity"
CONSISTENCY_HEATMAP_STEM = "phenotypic_consistency_heatmap"
# The same two panels at membership grain and at compound grain.
TARGET_ALIGNMENT_STEM = "target_alignment"

# The two outcomes the alignment cohort can be scored on. Same compounds, same
# properties, same layout: only the column read for the share row, the outcome, and
# the correlation changes, so the pair isolates the outcome from the population.
COHORT_OUTCOMES = {
    "compound_consistency_properties": {
        "outcome": "best_target_alignment_nap",
        "rho": "rho",
        "outcome_label": "Phenotypic\nconsistency",
        "title": "Compound properties and phenotypic consistency",
    },
    "compound_cohort_activity_properties": {
        "outcome": "activity_nmap",
        "rho": "rho_activity",
        "outcome_label": "Phenotypic\nactivity",
        "title": "Compound properties and phenotypic activity, alignment cohort",
    },
}


MODULAR_PANEL_STEMS = (
    "intrinsic_properties",
    "lipinski_violations",
    "clinical_phase",
    "pains_and_curated_collections",
    "phenotypic_consistency",
)
PROPERTY_DRAWERS_STEM = "property_drawers"
SEPARATE_FIGURE_STEMS = (
    "structural_classes",
    "functional_use",
    CONSISTENCY_HEATMAP_STEM,
    TARGET_ALIGNMENT_STEM,
    *COHORT_OUTCOMES,
    PROPERTY_DRAWERS_STEM,
)
FIGURE_OUTPUT_STEMS = (COMPOSITE_OUTPUT_STEM, *MODULAR_PANEL_STEMS, *SEPARATE_FIGURE_STEMS)
# One legend section per rendered file, titled here and bodied by the caption the renderer drew.
SECTION_TITLES = {
    COMPOSITE_OUTPUT_STEM: "Compound properties and annotations by phenotypic activity",
    "intrinsic_properties": "Intrinsic compound properties and phenotypic activity",
    "lipinski_violations": "Lipinski violations and phenotypic activity",
    "clinical_phase": "Clinical phase and phenotypic activity",
    "pains_and_curated_collections": "PAINS and curated collections",
    "phenotypic_consistency": "Compound attributes and phenotypic consistency",
    "structural_classes": "Structural classes and phenotypic activity",
    "functional_use": "Functional-use annotations and phenotypic activity",
    CONSISTENCY_HEATMAP_STEM: "Compound attributes across annotation sources",
    TARGET_ALIGNMENT_STEM: "Phenotypic activity and consistency across OpenTargets targets",
    "compound_consistency_properties": "Compound properties and per-compound phenotypic consistency",
    "compound_cohort_activity_properties": "The same cohort scored on phenotypic activity",
    PROPERTY_DRAWERS_STEM: "The same six properties read three ways",
}
OUTPUTS_BY_ROLE = {
    "composite": [COMPOSITE_OUTPUT_STEM],
    "modular_panel": list(MODULAR_PANEL_STEMS),
    "separate_figure": list(SEPARATE_FIGURE_STEMS),
}
COMPOSITE_MEMBERS = list(MODULAR_PANEL_STEMS)

INTRINSIC_TABLE = "intrinsic_activity_nmap_bins.csv"
CATEGORICAL_TABLE = "selected_categorical_activity_nmap.csv"
COMPOUND_DISTRIBUTION_TABLE = "compound_activity_nmap_distributions.parquet"
STRUCTURAL_TABLE = "structural_class_activity_nmap.csv"
STRUCTURAL_DISTRIBUTION_TABLE = "structural_class_activity_distributions.parquet"
STRUCTURAL_ALL_SIGNIFICANT_TABLE = "structural_classes_all_significant_5pct_fdr.csv"
FUNCTIONAL_TABLE = "functional_use_activity_nmap.csv"
FUNCTIONAL_DISTRIBUTION_TABLE = "functional_use_activity_distributions.parquet"
CONSISTENCY_TABLE = "consistency_property_effects.csv"
CONSISTENCY_GROUP_TABLE = "consistency_group_properties.csv"
ALIGNMENT_MEMBERSHIP_TABLE = "opentargets_target_alignment_memberships.csv"
ALIGNMENT_TARGET_TABLE = "opentargets_target_alignment_target_summary.csv"
ALIGNMENT_CURVE_TABLE = "opentargets_target_alignment_fitted_curve.csv"
ALIGNMENT_COMPOUND_TABLE = "opentargets_target_alignment_compound_summary.csv"
ALIGNMENT_PROPERTY_BIN_TABLE = "opentargets_target_alignment_property_bins.csv"
ALIGNMENT_GRAIN_AUDIT_TABLE = "opentargets_property_grain_audit.csv"
CLINICAL_STAGE_TRAIT_PREFIX = "derived.clinical_phase."

DRAWER_TYPE_SCALE = 1.7
PROPERTY_DRAWERS_WIDTH_IN = 22.0
# Drawer height as a fraction of drawer width.
DRAWER_ASPECT = 2 / 3
CATEGORY_DRAWER_BLOCKS = (
    ("Lipinski violations", "Number of violations"),
    ("Clinical phase", "Clinical development stage"),
    ("PAINS/collections", "Curated collection"),
)
# Property column -> the one cohort compound left out of that facet's drawn points.
DRAWER_EXCLUDED_COMPOUNDS = {"rotatable": "JCP2022_115083"}
DRAWER_EXCLUDED_NOTE = (
    "Dronedarone (18 rotatable bonds; next highest, 12) is omitted from the two cohort drawers in that column "
    "but retained in the full-screen drawer and all correlations."
)
ALIGNMENT_FIGURE_WIDTH = 3.6
ALIGNMENT_PANEL_HEIGHT = 3.15
# The two scatters share an x axis, so the upper one spends no height on tick labels
# or an axis label and can sit shorter than the across-target panel below them.
ALIGNMENT_SCATTER_HEIGHT = 2.65
ALIGNMENT_CAPTION_POINTS = 6.0

# Same six properties as the activity panels, in the same reading order.
# The full collection names collide on a nine-panel grid.
COMPOUND_CATEGORY_SHORT_LABELS = {
    "Chemical Probes\n(includes SGC)": "Probes",
    "Everything": "All",
    "All other": "Other",
}
# The alignment tables key properties by bare column name, the 2.45 intrinsic table by
# namespaced trait id. The drawers figure is the only place both grains meet.
SCREEN_TRAIT_BY_COLUMN = {
    "logp": "chem.logp",
    "mw": "chem.molecular_weight",
    "rotatable": "chem.rotatable_bonds",
    "qed": "chem.qed",
    "tpsa": "chem.tpsa",
    "fraction_csp3": "chem.fraction_csp3",
}
COMPOUND_PROPERTY_TRAITS = (
    ("logp", "LogP", "Calculated LogP"),
    ("mw", "Molecular weight", "Molecular weight (Da)"),
    ("rotatable", "Rotatable bonds", "Rotatable bonds"),
    ("qed", "QED", "QED"),
    ("tpsa", "TPSA", "TPSA (angstrom^2)"),
    ("fraction_csp3", "Fraction Csp3", "Fraction Csp3"),
)

CONSISTENCY_HEATMAP_TRAIT_IDS = (
    "chem.logp",
    "chem.molecular_weight",
    "chem.rotatable_bonds",
    "chem.qed",
    "chem.tpsa",
    "chem.fraction_csp3",
    "chem.has_pains",
    "chem.lipinski_violations",
    "cell_painting.normalized_cell_density",
    "derived.clinical_phase.phase_1",
    "derived.clinical_phase.phase_2",
    "derived.clinical_phase.phase_3",
    "derived.clinical_phase.launched",
)
CONSISTENCY_HEATMAP_LABELS = {
    "chem.logp": "LogP",
    "chem.molecular_weight": "MW",
    "chem.rotatable_bonds": "Rot. bonds",
    "chem.qed": "QED",
    "chem.tpsa": "TPSA",
    "chem.fraction_csp3": "Fraction\nCsp3",
    "chem.has_pains": "PAINS",
    "chem.lipinski_violations": "Lipinski\nviolations",
    "cell_painting.normalized_cell_density": "Cell\ndensity",
    "derived.clinical_phase.phase_1": "Phase\n1",
    "derived.clinical_phase.phase_2": "Phase\n2",
    "derived.clinical_phase.phase_3": "Phase\n3",
    "derived.clinical_phase.launched": "Launched",
}
CONSISTENCY_HEATMAP_FIGURE_SIZE = (12.0, 9.4)
CONSISTENCY_COLORMAP = "RdBu"
CONSISTENCY_UNTESTED_COLOR = "#D9D9D9"

# The shared property-figure system lives in jump_production.visualization so this
# renderer and figure-opentargets-group-consistency.py cannot drift apart.
FAMILY_BLUE = PROPERTY_FIGURE_COLOR["blue"]
FAMILY_DARK_GRAY = PROPERTY_FIGURE_COLOR["dark_gray"]
FAMILY_GRAY = PROPERTY_FIGURE_COLOR["gray"]
FAMILY_LIGHT_GRAY = PROPERTY_FIGURE_COLOR["light_gray"]
FAMILY_TITLE_PT = PROPERTY_FIGURE_TYPE["title"]
FAMILY_AXIS_LABEL_PT = PROPERTY_FIGURE_TYPE["axis_label"]
FAMILY_Y_LABEL_PT = PROPERTY_FIGURE_TYPE["y_label"]
FAMILY_TICK_PT = PROPERTY_FIGURE_TYPE["tick"]
FAMILY_ANNOTATION_PT = PROPERTY_FIGURE_TYPE["annotation"]
FAMILY_CAPTION_PT = PROPERTY_FIGURE_TYPE["caption"]
FAMILY_CATEGORY_TICK_PT = 7.0
FAMILY_SUPTITLE_PT = PROPERTY_FIGURE_TYPE["suptitle"]

HIGHER_ASSOCIATION_FILL = "#D9EBF6"
HIGHER_ASSOCIATION_EDGE = "#1769A5"
LOWER_ASSOCIATION_FILL = "#FCE8D2"
LOWER_ASSOCIATION_EDGE = "#B45309"
NEUTRAL_BOX_FILL = "#F3F4F6"
NEUTRAL_BOX_EDGE = "#6B7280"
# Hue by outcome, not by role: both activity drawers read green and the consistency
# drawer purple, through the points, the fit, the boxes, and the y label. Both hues come
# from the Okabe-Ito set, so the pair separates under the common colour-vision types.
# A category holding thousands of compounds draws as a solid block that swallows the
# box and its neighbours, so above this many the box and the count stand alone.
CATEGORY_DOT_LIMIT = 1_000
ACTIVITY_INK, ACTIVITY_TINT = "#009E73", "#D5F0E6"
CONSISTENCY_INK, CONSISTENCY_TINT = "#CC79A7", "#F7E2EE"
DRAWER_COLORS = {
    "screen": (ACTIVITY_INK, ACTIVITY_TINT),
    "activity": (ACTIVITY_INK, ACTIVITY_TINT),
    "consistency": (CONSISTENCY_INK, CONSISTENCY_TINT),
}

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one provenance file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise a clear contract error for missing plot columns."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_analysis_table(directory: Path, filename: str) -> tuple[pd.DataFrame, dict]:
    """Load one table only when its owning analysis declares it in summary.json."""
    summary_path = directory / "summary.json"
    table_path = directory / filename
    if not summary_path.is_file():
        raise FileNotFoundError(f"Required analysis summary not found: {summary_path}")
    if not table_path.is_file():
        raise FileNotFoundError(f"Required plot-ready table not found: {table_path}. Run its owning analysis first.")
    summary = json.loads(summary_path.read_text())
    declared = {Path(item["path"]).name for item in summary.get("files", [])}
    if filename not in declared:
        raise ValueError(f"{summary_path} does not declare required table {filename}")
    frame = pd.read_parquet(table_path) if table_path.suffix == ".parquet" else pd.read_csv(table_path)
    if frame.empty:
        raise ValueError(f"Required plot-ready table is empty: {table_path}")
    return frame, summary


def validate_analysis_summaries(
    functional_summary: dict,
    trait_summary: dict,
    property_summary: dict,
    requested_dataset: str,
) -> None:
    """Require the three owning analyses to describe one activity slice."""
    summaries = {
        "2.42": functional_summary.get("numbers", {}),
        "2.43": trait_summary.get("numbers", {}),
        "2.45": property_summary.get("numbers", {}),
    }
    for key in ("dataset", "columns", "preprocessing", "filter_name", "activity_params"):
        values = {label: numbers.get(key) for label, numbers in summaries.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"Owning analysis summaries disagree on {key}: {values}")
    if summaries["2.45"]["dataset"] != requested_dataset:
        raise ValueError(
            f"Requested dataset {requested_dataset!r} does not match owning analyses ({summaries['2.45']['dataset']!r})"
        )
    thresholds = {label: numbers.get("activity_threshold") for label, numbers in summaries.items()}
    if any(value is None for value in thresholds.values()) or not np.allclose(
        list(thresholds.values()), thresholds["2.45"]
    ):
        raise ValueError(f"Owning analysis summaries disagree on activity_threshold: {thresholds}")
    compound_counts = {
        "2.42": summaries["2.42"].get("n_canonical_activity_compounds"),
        "2.43": summaries["2.43"].get("n_activity_compounds"),
        "2.45": summaries["2.45"].get("n_compounds"),
    }
    if None in compound_counts.values() or len(set(compound_counts.values())) != 1:
        raise ValueError(f"Owning analysis summaries disagree on analyzed compounds: {compound_counts}")
    active_counts = {
        "2.43": summaries["2.43"].get("n_active"),
        "2.45": summaries["2.45"].get("n_active"),
    }
    if None in active_counts.values() or len(set(active_counts.values())) != 1:
        raise ValueError(f"Owning analysis summaries disagree on active compounds: {active_counts}")


def require_number_contract(summary: dict, key: str, label: str) -> dict:
    """Return one required nested numbers contract from an owning summary."""
    contract = summary.get("numbers", {}).get(key)
    if not isinstance(contract, dict):
        raise ValueError(f"{label} summary is missing numbers.{key}")
    return contract


def summarize_structural_class_contract(frame: pd.DataFrame, trait_summary: dict) -> dict[str, object]:
    """Summarize the structural screen from its complete significant-class table."""
    require_columns(
        frame,
        {
            "family_title",
            "class",
            "direction",
            "selected_for_figure",
            "n_tested_family",
        },
        STRUCTURAL_ALL_SIGNIFICANT_TABLE,
    )
    source = require_number_contract(trait_summary, "structural_class_activity_nmap", "Trait-enrichment")
    required_source = {
        "selection",
        "maximum_rows_per_direction",
        "fdr_threshold",
        "minimum_members_per_group",
        "minimum_contingency_cell_count",
        "test",
        "multiple_testing",
        "eligible_population",
        "reference_population",
    }
    missing = sorted(required_source - set(source))
    if missing:
        raise ValueError(f"Trait-enrichment structural contract is missing fields: {missing}")
    maximum_rows = int(source["maximum_rows_per_direction"])
    families: dict[str, dict[str, object]] = {}
    for title, group in frame.groupby("family_title", sort=False):
        if group["n_tested_family"].nunique() != 1:
            raise ValueError(f"Structural family {title} has multiple tested-family denominators")
        selected = group.loc[group["selected_for_figure"].astype(bool)]
        selected_by_direction = {
            str(direction).lower(): int(len(direction_rows))
            for direction, direction_rows in selected.groupby("direction", sort=False)
        }
        if any(count > maximum_rows for count in selected_by_direction.values()):
            raise ValueError(f"Structural family {title} exceeds its per-direction display maximum")
        families[str(title)] = {
            "n_tested": int(group["n_tested_family"].iloc[0]),
            "n_significant": int(len(group)),
            "n_selected": int(len(selected)),
            "n_selected_by_direction": selected_by_direction,
        }
    return {
        "families": families,
        "n_significant_total": int(len(frame)),
        "n_selected": int(frame["selected_for_figure"].astype(bool).sum()),
        "selection": str(source["selection"]),
        "maximum_rows_per_direction": maximum_rows,
        "fdr_threshold": float(source["fdr_threshold"]),
        "testability": {
            "minimum_members_per_group": int(source["minimum_members_per_group"]),
            "minimum_contingency_cell_count": int(source["minimum_contingency_cell_count"]),
        },
        "test": str(source["test"]),
        "multiple_testing": str(source["multiple_testing"]),
        "eligible_population": str(source["eligible_population"]),
        "reference_population": str(source["reference_population"]),
        "context_row_is_comparator": False,
    }


def summarize_functional_use_contract(frame: pd.DataFrame, functional_summary: dict) -> dict[str, object]:
    """Summarize FUC and FUP inference from the plot-ready rows and owning contract."""
    require_columns(
        frame,
        {
            "section",
            "row_order",
            "category",
            "source_level",
            "inference_estimate",
            "n_exposed",
            "n_reference",
            "active_exposed",
            "rate_exposed",
            "significant",
        },
        FUNCTIONAL_TABLE,
    )
    source = require_number_contract(functional_summary, "functional_use_activity_nmap", "Functional-use")
    required_source = {
        "n_fuc_tested",
        "n_fuc_significant",
        "n_fup_tested",
        "n_fup_significant",
        "n_fup_higher",
        "n_fup_lower",
        "fdr_threshold",
        "fuc_inference",
        "fuc_eligible_population",
        "fuc_testability",
        "fup_inference",
        "fup_eligible_population",
        "fup_testability",
        "fup_display_threshold",
        "multiple_testing",
    }
    missing = sorted(required_source - set(source))
    if missing:
        raise ValueError(f"Functional-use plot contract is missing fields: {missing}")

    fuc = frame.loc[frame["section"].eq("FUC")].sort_values("row_order", kind="stable")
    fup = frame.loc[frame["section"].eq("FUP")].sort_values("row_order", kind="stable")
    significant_fuc = fuc.loc[fuc["significant"].astype(bool)]
    significant_fup = fup.loc[fup["significant"].astype(bool)]
    higher = significant_fup.loc[significant_fup["inference_estimate"].gt(1)]
    lower = significant_fup.loc[significant_fup["inference_estimate"].lt(1)]
    observed_counts = {
        "n_fuc_tested": int(len(fuc)),
        "n_fuc_significant": int(len(significant_fuc)),
        "n_fup_tested": int(len(fup)),
        "n_fup_significant": int(len(significant_fup)),
        "n_fup_higher": int(len(higher)),
        "n_fup_lower": int(len(lower)),
    }
    declared_counts = {key: int(source[key]) for key in observed_counts}
    if observed_counts != declared_counts:
        raise ValueError(
            f"Functional-use table disagrees with its owning summary: observed={observed_counts}, "
            f"declared={declared_counts}"
        )
    fuc_eligible = (fuc["n_exposed"] + fuc["n_reference"]).astype(int)
    if fuc_eligible.empty or fuc_eligible.nunique() != 1:
        raise ValueError("FUC rows must share one annotated-compound eligible population")
    declared_fuc_eligible = functional_summary.get("numbers", {}).get("n_with_curated_fuc_in_analysis")
    if declared_fuc_eligible is None or int(declared_fuc_eligible) != int(fuc_eligible.iloc[0]):
        raise ValueError("FUC table disagrees with the owning summary's annotated-compound count")
    declared_fup_coverage = functional_summary.get("numbers", {}).get("n_with_fup_prediction_in_analysis")
    if declared_fup_coverage is None:
        raise ValueError("Functional-use summary is missing n_with_fup_prediction_in_analysis")
    fup_eligible = (fup["n_exposed"] + fup["n_reference"]).astype(int)
    if fup_eligible.empty:
        raise ValueError("FUP contract requires at least one tested endpoint")

    def displayed_rate(row: pd.Series) -> dict[str, object]:
        active = int(row["active_exposed"])
        total = int(row["n_exposed"])
        percent = 100 * active / total
        if not np.isclose(percent, float(row["rate_exposed"])):
            raise ValueError(f"Functional-use rate disagrees for {row['source_level']}")
        return {
            "endpoint": str(row["source_level"]),
            "active": active,
            "n": total,
            "active_percent": float(percent),
        }

    highest_rate = (
        displayed_rate(significant_fup.loc[significant_fup["rate_exposed"].idxmax()])
        if not significant_fup.empty
        else None
    )
    lowest_rate = (
        displayed_rate(significant_fup.loc[significant_fup["rate_exposed"].idxmin()])
        if not significant_fup.empty
        else None
    )
    return {
        "curated_fuc": {
            "n_compounds_with_annotation": int(fuc_eligible.iloc[0]),
            "n_tested": observed_counts["n_fuc_tested"],
            "n_significant": observed_counts["n_fuc_significant"],
            "test": str(source["fuc_inference"]),
            "eligible_population": str(source["fuc_eligible_population"]),
            "missing_annotation_is_absence": False,
            "testability": source["fuc_testability"],
        },
        "predicted_fup": {
            "n_compounds_with_any_prediction": int(declared_fup_coverage),
            "n_tested": observed_counts["n_fup_tested"],
            "n_significant": observed_counts["n_fup_significant"],
            "n_higher": observed_counts["n_fup_higher"],
            "n_lower": observed_counts["n_fup_lower"],
            "higher_endpoints": higher["source_level"].astype(str).tolist(),
            "lower_endpoints": lower["source_level"].astype(str).tolist(),
            "inference": str(source["fup_inference"]),
            "eligible_population": str(source["fup_eligible_population"]),
            "eligible_n_min": int(fup_eligible.min()),
            "eligible_n_max": int(fup_eligible.max()),
            "testability": source["fup_testability"],
            "display_threshold": float(source["fup_display_threshold"]),
            "display_threshold_defines_inference": False,
            "highest_display_active_rate": highest_rate,
            "lowest_display_active_rate": lowest_rate,
        },
        "fdr_threshold": float(source["fdr_threshold"]),
        "multiple_testing": str(source["multiple_testing"]),
        "context_row_is_comparator": False,
    }


def validate_consistency_group_audit(frame: pd.DataFrame, trait_summary: dict) -> None:
    """Validate the declared one-source-group-by-property audit grain."""
    keys = ["annotation_source", "annotation_group", "trait_id"]
    require_columns(frame, set(keys), CONSISTENCY_GROUP_TABLE)
    if frame.duplicated(keys).any():
        raise ValueError(f"{CONSISTENCY_GROUP_TABLE} contains duplicate source-group-property rows")
    contract = trait_summary.get("numbers", {}).get("consistency_group_properties", {})
    observed = {
        "n_rows": len(frame),
        "n_groups": frame[["annotation_source", "annotation_group"]].drop_duplicates().shape[0],
        "n_traits": frame["trait_id"].nunique(),
    }
    expected = {key: contract.get(key) for key in observed}
    if None in expected.values() or observed != expected:
        raise ValueError(
            f"{CONSISTENCY_GROUP_TABLE} disagrees with its owning summary: observed={observed}, declared={expected}"
        )


def summarize_consistency_audit(groups: pd.DataFrame, effects: pd.DataFrame) -> dict[str, int]:
    """Return the denominator and modeled-sample audit declared by this figure."""
    group_keys = ["annotation_source", "annotation_group"]
    require_columns(
        groups,
        {
            *group_keys,
            "trait_id",
            "n_membership_compounds",
            "n_property_compounds",
        },
        CONSISTENCY_GROUP_TABLE,
    )
    require_columns(
        effects,
        {"trait_id", "effect_scope", "status", "n_memberships", "n_groups"},
        CONSISTENCY_TABLE,
    )
    membership_variants = groups.groupby(group_keys, sort=False)["n_membership_compounds"].nunique()
    if membership_variants.gt(1).any():
        raise ValueError("Consistency-group membership counts must be invariant across traits")

    eligible_by_trait = groups.groupby("trait_id", sort=False)["n_property_compounds"].sum().astype(int)
    clinical_eligible = eligible_by_trait.loc[eligible_by_trait.index.str.startswith(CLINICAL_STAGE_TRAIT_PREFIX)]
    if clinical_eligible.empty or clinical_eligible.nunique() != 1:
        raise ValueError("Clinical-stage traits must share one non-empty consistency eligibility count")

    # These are analytic-sample denominators, including rows later excluded from
    # inference because the fitted model did not meet the testing contract.
    modeled = effects.loc[effects["effect_scope"].eq("between_group")]
    modeled_memberships = modeled.groupby("trait_id", sort=False)["n_memberships"].sum().astype(int)
    modeled_groups = modeled.groupby("trait_id", sort=False)["n_groups"].sum().astype(int)
    if modeled_memberships.empty or modeled_groups.empty:
        raise ValueError("Consistency audit requires tested between-group models")

    group_once = groups.drop_duplicates(group_keys)
    return {
        "consistency_group_audit_rows": int(len(groups)),
        "consistency_annotation_sources": int(groups["annotation_source"].nunique()),
        "consistency_groups": int(len(group_once)),
        "consistency_unique_memberships_total": int(group_once["n_membership_compounds"].sum()),
        "consistency_trait_eligible_memberships_min": int(eligible_by_trait.min()),
        "consistency_trait_eligible_memberships_max": int(eligible_by_trait.max()),
        "consistency_clinical_stage_eligible_memberships": int(clinical_eligible.iloc[0]),
        "consistency_modeled_memberships_min": int(modeled_memberships.min()),
        "consistency_modeled_memberships_max": int(modeled_memberships.max()),
        "consistency_modeled_groups_min": int(modeled_groups.min()),
        "consistency_modeled_groups_max": int(modeled_groups.max()),
    }


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    """Save deterministic PNG and PDF variants.

    Deliberately local rather than `jump_production.visualization.save_figure`: these panels are
    assembled from variable-width molecule renderings, so they need tight cropping, and the PDF
    timestamps are scrubbed so a rebuild that changes nothing produces a byte-identical file.
    The shared saver keeps fixed dimensions and default metadata, which the other figures rely on.
    """
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == ".pdf" else {}
        figure.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def format_count(value: float | int) -> str:
    """Format a compact panel count."""
    value = int(value)
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def style_boxplot(boxplot: dict[str, list[plt.Artist]], edgecolors: list[str]) -> None:
    """Apply the selected minimal neutral boxplot grammar."""
    for box, edgecolor in zip(boxplot["boxes"], edgecolors, strict=True):
        box.set(facecolor="white", edgecolor=edgecolor, linewidth=1.15)
    for median in boxplot["medians"]:
        median.set(color="#111827", linewidth=1.65)
    for index, edgecolor in enumerate(edgecolors):
        for whisker in boxplot["whiskers"][2 * index : 2 * index + 2]:
            whisker.set(color=edgecolor, linewidth=1.0)
        for cap in boxplot["caps"][2 * index : 2 * index + 2]:
            cap.set(color=edgecolor, linewidth=1.0)


def summarize_box_distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    """Return the count and activity annotation for one displayed box."""
    if frame.empty:
        raise ValueError("Cannot render an empty compound distribution")
    active = int(frame["active"].astype(bool).sum())
    return {
        "n": int(len(frame)),
        "active": active,
        "active_rate": 100 * active / len(frame),
    }


def build_boxplot_distributions(
    compounds: pd.DataFrame,
    selected: pd.DataFrame | None = None,
) -> tuple[
    list[tuple[str, pd.DataFrame]],
    list[tuple[str, pd.DataFrame]],
    list[tuple[str, pd.DataFrame]],
]:
    """Build the selected Lipinski, clinical-phase, and collection distributions."""
    required = {
        "jcp2022",
        "nmap",
        "active",
        "lipinski_violations",
        "valid_molecule",
        "has_pains",
        "clinical_phase",
        "in_kcgs",
        "in_pkis",
        "in_pkis2",
        "in_chemical_probes_sgc",
    }
    require_columns(compounds, required, COMPOUND_DISTRIBUTION_TABLE)
    if compounds["jcp2022"].duplicated().any():
        raise ValueError(f"{COMPOUND_DISTRIBUTION_TABLE} must contain one row per compound")
    if compounds[["nmap", "active"]].isna().any().any():
        raise ValueError(f"{COMPOUND_DISTRIBUTION_TABLE} contains missing activity values")

    lipinski_values = compounds["lipinski_violations"].dropna().astype(int)
    levels = sorted(lipinski_values.unique())
    if not levels or levels[0] != 0 or levels[-1] > 4 or levels != list(range(levels[-1] + 1)):
        raise ValueError(f"Expected contiguous Lipinski violation levels beginning at 0, found {levels}")
    lipinski = [(str(level), compounds.loc[compounds["lipinski_violations"].eq(level)]) for level in levels]

    phase_tokens = (
        compounds["clinical_phase"]
        .fillna("")
        .apply(lambda value: {piece.strip() for piece in str(value).split("|") if piece.strip()})
    )
    phase_labels = ("Phase 1", "Phase 2", "Phase 3", "Launched")
    clinical = [
        (label, compounds.loc[phase_tokens.apply(lambda values, selected_label=label: selected_label in values)])
        for label in phase_labels
    ]
    displayed_phase = phase_tokens.apply(lambda values: bool(values.intersection(phase_labels)))
    clinical.append(("All other", compounds.loc[~displayed_phase]))

    valid_pains = compounds["valid_molecule"].fillna(False).astype(bool) & compounds["has_pains"].notna()
    collection_masks = (
        ("PAINS", valid_pains & compounds["has_pains"].fillna(False).astype(bool)),
        ("KCGS", compounds["in_kcgs"].fillna(False).astype(bool)),
        ("PKIS", compounds["in_pkis"].fillna(False).astype(bool)),
        ("PKIS2", compounds["in_pkis2"].fillna(False).astype(bool)),
        ("Chemical Probes\n(includes SGC)", compounds["in_chemical_probes_sgc"].fillna(False).astype(bool)),
    )
    collections = [(label, compounds.loc[mask]) for label, mask in collection_masks]
    collections.append(("Everything", compounds))

    # Only the full analyzed family has 2.45 counts to reconcile against; a restricted
    # cohort reuses the same category rules with no upstream ledger to check.
    if selected is not None:
        expected = selected.set_index("label")
        for label, distribution in clinical[:-1] + collections[:-1]:
            lookup = label.replace("\n", " ")
            observed = summarize_box_distribution(distribution)
            recorded = expected.loc[lookup]
            if (observed["n"], observed["active"]) != (int(recorded["n"]), int(recorded["active"])):
                raise ValueError(f"Displayed category counts changed for {lookup}")
    return lipinski, clinical, collections


def save_exact_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    """Save a figure without cropping so its requested print dimensions stay exact."""
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == ".pdf" else {}
        figure.savefig(path, dpi=dpi, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def validate_intrinsic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the six intrinsic traits in display order."""
    required = {
        "trait_order",
        "trait_id",
        "title",
        "xlabel",
        "bin",
        "bin_lower",
        "bin_upper",
        "x",
        "n",
        "rate",
        "lower",
        "upper",
        "nmap_median",
        "nmap_q25",
        "nmap_q75",
        "rho",
        "rho_q_value",
        "fit_slope",
        "fit_intercept",
        "fdr_bh",
        "significant_at_5pct_fdr",
    }
    require_columns(frame, required, INTRINSIC_TABLE)
    traits = frame[["trait_order", "trait_id"]].drop_duplicates().sort_values("trait_order", kind="stable")
    if len(traits) != 6:
        raise ValueError(f"{INTRINSIC_TABLE} must contain exactly six intrinsic properties")
    return traits


def draw_intrinsic_panels(
    figure: plt.Figure,
    outer,
    frame: pd.DataFrame,
) -> None:
    """Draw the reusable 2x3 intrinsic-property block into a supplied grid."""
    traits = validate_intrinsic_frame(frame)
    gray = "#6B7280"
    for panel_index, trait in enumerate(traits.itertuples(index=False)):
        row = panel_index // 3
        column = panel_index % 3
        curve = frame.loc[frame["trait_id"].eq(trait.trait_id)].sort_values("bin", kind="stable")
        first = curve.iloc[0]
        nmap_axis = figure.add_subplot(outer[row, column])
        x_values = curve["x"].to_numpy(dtype=float)
        nmap_axis.set_title(str(first["title"]), fontsize=FAMILY_TITLE_PT, fontweight="bold", loc="left")
        nmap_axis.fill_between(
            x_values,
            curve["nmap_q25"].to_numpy(dtype=float),
            curve["nmap_q75"].to_numpy(dtype=float),
            color="#E5E7EB",
            alpha=0.82,
        )
        nmap_axis.plot(
            x_values,
            curve["nmap_median"],
            color="#4B5563",
            marker="s",
            markersize=1.6,
            linewidth=0.9,
        )
        nmap_axis.axhline(0, color="#D1D5DB", linestyle="--", linewidth=0.5)
        nmap_axis.set_ylim(-0.06, 0.55)
        nmap_axis.set_yticks([0.0, 0.2, 0.4])
        nmap_axis.set_xlabel(str(first["xlabel"]), fontsize=FAMILY_AXIS_LABEL_PT, labelpad=1.5)
        nmap_axis.text(
            0.02,
            0.94,
            f"rho={float(first['rho']):.2f}",
            transform=nmap_axis.transAxes,
            ha="left",
            va="top",
            fontsize=FAMILY_ANNOTATION_PT,
            color=gray,
        )
        if column == 0:
            nmap_axis.set_ylabel(
                "Phenotypic\nactivity (nMAP)", color=FAMILY_DARK_GRAY, fontsize=FAMILY_Y_LABEL_PT, labelpad=1.5
            )
        nmap_axis.tick_params(labelsize=FAMILY_TICK_PT, length=2.0, pad=1.2)
        nmap_axis.spines[["top", "right"]].set_visible(False)
        nmap_axis.spines[["left", "bottom"]].set_linewidth(0.6)


def intrinsic_caption(frame: pd.DataFrame, analyzed_compounds: int) -> str:
    """Return the shared caption for the six intrinsic-property panels."""
    significance_note = (
        "all six intrinsic associations pass q < 0.05"
        if frame["significant_at_5pct_fdr"].astype(bool).all()
        else "intrinsic q values come from the prespecified activity models"
    )
    return (
        f"Phenotypic activity nMAP across {analyzed_compounds:,} compounds. Ten equal-width bins span the "
        f"1st-99th percentile of each property and contain {format_count(frame['n'].min())}-"
        f"{format_count(frame['n'].max())} compounds. Curves and bands show median and interquartile-range nMAP; "
        f"labels give Spearman rho. {significance_note.capitalize()} in prespecified activity models."
    )


def plot_intrinsic_properties(
    frame: pd.DataFrame,
    analyzed_compounds: int,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render only the six intrinsic physicochemical-property panels."""
    panel_height = PROPERTY_FIGURE_PANEL_HEIGHT_IN * 2 / 3
    caption = intrinsic_caption(frame, analyzed_compounds)
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        panel_height_in=panel_height,
    )
    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, figure_height))
    outer = figure.add_gridspec(2, 3, hspace=0.62, wspace=0.34)
    draw_intrinsic_panels(figure, outer, frame)
    figure.suptitle("Intrinsic properties and phenotypic activity", fontsize=8.5, fontweight="bold", x=0.015, ha="left")
    figure.text(
        0.50,
        0.015,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=FAMILY_CAPTION_PT,
        color=FAMILY_GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.30 / figure_height,
    )
    return save_exact_figure(figure, output_dir, "intrinsic_properties", dpi), {"intrinsic_properties": caption}


def compound_properties_caption(
    frame: pd.DataFrame,
    analyzed_compounds: int,
    active_compounds: int,
    activity_threshold: float,
) -> str:
    """Return the caption for the activity-panel composite."""
    significance = (
        "all six intrinsic associations pass q < 0.05"
        if frame["significant_at_5pct_fdr"].astype(bool).all()
        else "intrinsic q values come from the prespecified activity models"
    )
    active_percent = 100 * active_compounds / analyzed_compounds
    return (
        f"Phenotypic activity and compound annotations for {analyzed_compounds:,} compounds. Activity is replicate "
        "retrieval nMAP against same-plate negative controls; "
        f"{active_compounds:,} compounds ({active_percent:.1f}%) passed Benjamini-Hochberg corrected p < "
        f"{activity_threshold:g}. "
        "Curves and boxplots summarize compound-level nMAP, percentages give active compounds, and labels give "
        f"Spearman rho. Ten equal-width bins span each property's 1st-99th percentile and contain "
        f"{format_count(frame['n'].min())}-{format_count(frame['n'].max())} compounds; {significance}. "
        "Continuous associations use univariate logistic regression and report odds ratios per interquartile-range "
        "increase. Boolean and categorical associations use two-sided Fisher exact tests with eligible, measured "
        "compounds as comparators. Tested rows are adjusted by Benjamini-Hochberg within each trait family at 5% "
        "FDR; intervals are pointwise 95% confidence intervals. The consistency sidebar uses the adjusted "
        "multivariable model described below. Probes combines Chemical Probes and SGC."
    )


def plot_compound_properties_activity(
    frame: pd.DataFrame,
    lipinski: list[tuple[str, pd.DataFrame]],
    clinical: list[tuple[str, pd.DataFrame]],
    collections: list[tuple[str, pd.DataFrame]],
    analyzed_compounds: int,
    active_compounds: int,
    activity_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render the nine activity panels on the shared property-figure geometry.

    The consistency sidebar used to ride along on the right of this canvas while also
    shipping as its own export; it is now only the standalone `phenotypic_consistency`.
    """
    validate_intrinsic_frame(frame)
    blue = FAMILY_BLUE
    gray = FAMILY_GRAY
    caption_text = compound_properties_caption(frame, analyzed_compounds, active_compounds, activity_threshold)
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(caption_text)
    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, figure_height))
    outer = figure.add_gridspec(3, 3, hspace=0.62, wspace=0.34)
    draw_intrinsic_panels(figure, outer, frame)

    def draw_box_panel(
        axis: plt.Axes,
        distributions: list[tuple[str, pd.DataFrame]],
        title: str,
        tick_labels: list[str],
        xlabel: str,
        *,
        show_ylabel: bool = False,
    ) -> None:
        positions = np.arange(1, len(distributions) + 1)
        summaries = [summarize_box_distribution(distribution) for _, distribution in distributions]
        boxplot = axis.boxplot(
            [distribution["nmap"].to_numpy(dtype=float) for _, distribution in distributions],
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        style_boxplot(boxplot, [blue] * len(distributions))
        for box in boxplot["boxes"]:
            box.set_facecolor("#D9EBF6")
        for position, summary in zip(positions, summaries, strict=True):
            axis.text(
                position,
                1.065,
                f"{summary['active_rate']:.0f}%",
                ha="center",
                va="center",
                fontsize=FAMILY_ANNOTATION_PT,
                color=gray,
            )
        axis.set_xticks(positions, tick_labels)
        axis.set_ylim(-0.14, 1.13)
        axis.axhline(0, color="#D1D5DB", linestyle="--", linewidth=0.5, zorder=0)
        axis.set_title(title, fontsize=FAMILY_TITLE_PT, fontweight="bold", loc="left")
        axis.set_xlabel(xlabel, fontsize=FAMILY_AXIS_LABEL_PT, labelpad=1.5)
        if show_ylabel:
            axis.set_ylabel(
                "Phenotypic\nactivity (nMAP)", color=FAMILY_DARK_GRAY, fontsize=FAMILY_Y_LABEL_PT, labelpad=1.5
            )
        axis.tick_params(labelsize=FAMILY_CATEGORY_TICK_PT, length=2.0, pad=1.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.6)

    lipinski_counts = [format_count(len(distribution)) for _, distribution in lipinski]
    draw_box_panel(
        figure.add_subplot(outer[2, 0]),
        lipinski,
        "Lipinski violations",
        [f"{label}\n({count})" for (label, _), count in zip(lipinski, lipinski_counts, strict=True)],
        "Number of violations",
        show_ylabel=True,
    )
    clinical_names = ["Phase 1", "Phase 2", "Phase 3", "Launched", "All other"]
    clinical_counts = [format_count(len(distribution)) for _, distribution in clinical]
    draw_box_panel(
        figure.add_subplot(outer[2, 1]),
        clinical,
        "Clinical phase",
        [f"{label}\n({count})" for label, count in zip(clinical_names, clinical_counts, strict=True)],
        "Clinical development stage",
    )
    collection_names = ["PAINS", "KCGS", "PKIS", "PKIS2", "Probes", "All"]
    collection_counts = [format_count(len(distribution)) for _, distribution in collections]
    draw_box_panel(
        figure.add_subplot(outer[2, 2]),
        collections,
        "PAINS/collections",
        [f"{label}\n({count})" for label, count in zip(collection_names, collection_counts, strict=True)],
        "Curated collection",
    )
    figure.suptitle(
        "Compound properties and phenotypic activity",
        fontsize=FAMILY_SUPTITLE_PT,
        fontweight="bold",
        x=0.065,
        y=1 - 0.30 / figure_height,
        ha="left",
    )
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=FAMILY_CAPTION_PT,
        color=FAMILY_GRAY,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.30 / figure_height,
    )
    return save_exact_figure(figure, output_dir, COMPOSITE_OUTPUT_STEM, dpi), {COMPOSITE_OUTPUT_STEM: caption_text}


def plot_standalone_boxplot(
    distributions: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    stem: str,
    title: str,
    xlabel: str,
    dpi: int,
    *,
    figsize: tuple[float, float],
    caption: str,
    annotation_fontsize: float = 6.6,
    tick_fontsize: float = 8.0,
    title_fontsize: float = 13.0,
    left_margin: float = 0.09,
    axis_fontsize: float = 9.0,
    xlabel_fontsize: float = 8.0,
    footer_fontsize: float = 5.8,
    include_active_word: bool = True,
) -> list[Path]:
    """Render one selected nMAP-only category view with activity annotations."""
    gray = "#6B7280"
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=figsize[0],
        panel_height_in=figsize[1],
        caption_points=footer_fontsize,
    )
    figure, axis = plt.subplots(figsize=(figsize[0], figure_height))
    positions = np.arange(1, len(distributions) + 1)
    summaries = [summarize_box_distribution(distribution) for _, distribution in distributions]
    boxplot = axis.boxplot(
        [distribution["nmap"].to_numpy(dtype=float) for _, distribution in distributions],
        positions=positions,
        widths=0.68,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
    )
    style_boxplot(boxplot, [gray] * len(distributions))
    for position, summary in zip(positions, summaries, strict=True):
        axis.text(
            position,
            1.105,
            f"{summary['active_rate']:.1f}%{' active' if include_active_word else ''}\nn={format_count(summary['n'])}",
            ha="center",
            va="top",
            fontsize=annotation_fontsize,
            color=gray,
        )
    axis.set_xticks(positions, [label for label, _ in distributions])
    axis.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)
    axis.set_ylim(-0.14, 1.16)
    axis.axhline(0, color="#D1D5DB", linestyle="--", linewidth=0.8, zorder=0)
    axis.set_ylabel("Phenotypic activity", color="#374151", fontsize=axis_fontsize)
    axis.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    axis.tick_params(labelsize=tick_fontsize)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(left_margin, 1 - 0.27 / figure_height, title, ha="left", fontsize=title_fontsize, fontweight="bold")
    figure.text(
        0.50,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=footer_fontsize,
        color=gray,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=left_margin,
        right=0.985,
        top=1 - 0.54 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    return save_exact_figure(figure, output_dir, stem, dpi)


def plot_selected_categories(
    lipinski: list[tuple[str, pd.DataFrame]],
    clinical: list[tuple[str, pd.DataFrame]],
    collections: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render reusable Lipinski, clinical-phase, and collection panels."""
    shared = "Phenotypic activity is compound-level nMAP; labels give active compounds (%) and total compounds."
    probes_members = next(
        (len(distribution) for label, distribution in collections if label.startswith("Chemical Probes")), 0
    )
    captions = {
        "lipinski_violations": ("Compounds grouped by the number of Lipinski rule-of-five violations. " + shared),
        "clinical_phase": (
            "Repurposing Hub Phase 1, Phase 2, Phase 3, Launched, and All other categories. "
            + shared
            + " All other includes compounds without a displayed stage label."
        ),
        "pains_and_curated_collections": (
            "PAINS-alert compounds, KCGS, PKIS, PKIS2, and the combined Chemical Probes set that includes SGC, "
            "shown regardless of statistical significance. "
            + shared
            + f" All is the full cohort; Probes contains {probes_members:,} compounds."
        ),
    }
    output_paths = plot_standalone_boxplot(
        lipinski,
        output_dir,
        "lipinski_violations",
        "Lipinski violations",
        "Number of violations",
        dpi,
        figsize=(6.0, 4.5),
        caption=captions["lipinski_violations"],
    )
    output_paths.extend(
        plot_standalone_boxplot(
            clinical,
            output_dir,
            "clinical_phase",
            "Clinical phase",
            "Clinical phase",
            dpi,
            figsize=(6.0, 4.5),
            caption=captions["clinical_phase"],
        )
    )
    output_paths.extend(
        plot_standalone_boxplot(
            [
                (
                    "Probes" if label.startswith("Chemical Probes") else "All" if label == "Everything" else label,
                    distribution,
                )
                for label, distribution in collections
            ],
            output_dir,
            "pains_and_curated_collections",
            "PAINS and curated collections",
            "Category",
            dpi,
            figsize=(4.0, 4.5),
            caption=captions["pains_and_curated_collections"],
            annotation_fontsize=7.0,
            tick_fontsize=8.5,
            title_fontsize=13.0,
            left_margin=0.16,
            axis_fontsize=11.0,
            xlabel_fontsize=10.0,
            footer_fontsize=6.5,
            include_active_word=False,
        )
    )
    return output_paths, captions


def molecule_image(smiles: str, size: tuple[int, int] = (240, 110)) -> np.ndarray:
    """Draw one compact structure identity cell."""
    if smiles == "acyclic":
        image = PILImage.new("RGB", size, "white")
        ImageDraw.Draw(image).text((size[0] // 2 - 28, size[1] // 2 - 6), "ACYCLIC", fill="#6B7280")
        return np.asarray(image)
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        image = PILImage.new("RGB", size, "white")
        ImageDraw.Draw(image).text((8, size[1] // 2 - 6), "structure unavailable", fill="#6B7280")
        return np.asarray(image)
    return np.asarray(Draw.MolToImage(molecule, size=size))


def draw_horizontal_boxes(
    axis: plt.Axes,
    distributions: list[pd.DataFrame],
    labels: list[str] | None,
    *,
    tick_fontsize: float = 9.0,
    facecolors: list[str] | None = None,
    edgecolors: list[str] | None = None,
    summary_labels: list[str] | None = None,
    summary_x: float = 1.125,
    xlim: tuple[float, float] = (-0.14, 1.16),
) -> None:
    """Draw minimal horizontal compound-level nMAP boxplots."""
    if facecolors is None:
        facecolors = [NEUTRAL_BOX_FILL] * len(distributions)
    if edgecolors is None:
        edgecolors = [NEUTRAL_BOX_EDGE] * len(distributions)
    if len(facecolors) != len(distributions) or len(edgecolors) != len(distributions):
        raise ValueError("Horizontal box colors must match the number of distributions")
    if summary_labels is None:
        summary_labels = [
            f"{100 * distribution['active'].astype(bool).mean():.1f}%\nn={format_count(len(distribution))}"
            for distribution in distributions
        ]
    if len(summary_labels) != len(distributions):
        raise ValueError("Horizontal box summaries must match the number of distributions")
    positions = np.arange(1, len(distributions) + 1)
    boxplot = axis.boxplot(
        [distribution["nmap"].to_numpy(dtype=float) for distribution in distributions],
        positions=positions,
        widths=0.62,
        vert=False,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
    )
    for box, facecolor, edgecolor in zip(boxplot["boxes"], facecolors, edgecolors, strict=True):
        box.set(facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1)
    for median in boxplot["medians"]:
        median.set(color="#111827", linewidth=1.3)
    for index, edgecolor in enumerate(edgecolors):
        for artist in (
            *boxplot["whiskers"][2 * index : 2 * index + 2],
            *boxplot["caps"][2 * index : 2 * index + 2],
        ):
            artist.set(color=edgecolor, linewidth=0.9)
    for position, summary_label in zip(positions, summary_labels, strict=True):
        axis.text(
            summary_x,
            position,
            summary_label,
            ha="right",
            va="center",
            fontsize=7.4,
            color=NEUTRAL_BOX_EDGE,
            linespacing=1.05,
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(0.35, len(distributions) + 0.65)
    axis.axvline(0, color="#D1D5DB", linestyle="--", linewidth=0.9, zorder=0)
    axis.set_yticks(positions, labels if labels is not None else [""] * len(positions))
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=8.5)
    axis.tick_params(axis="y", labelsize=tick_fontsize, length=0, pad=7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.45, alpha=0.65)
    axis.set_axisbelow(True)


def validate_distribution(distribution: pd.DataFrame, row, label: str) -> None:
    """Fail when a compound-level box disagrees with its plot summary."""
    observed = {
        "n_exposed": len(distribution),
        "active_exposed": int(distribution["active"].astype(bool).sum()),
        "nmap_median": distribution["nmap"].median(),
        "nmap_q25": distribution["nmap"].quantile(0.25),
        "nmap_q75": distribution["nmap"].quantile(0.75),
    }
    for field, value in observed.items():
        expected = getattr(row, field)
        if field.startswith("nmap"):
            if not np.isclose(value, expected, rtol=0, atol=1e-12):
                raise ValueError(f"{label}: {field} mismatch: {value} != {expected}")
        elif int(value) != int(expected):
            raise ValueError(f"{label}: {field} mismatch: {value} != {expected}")


def structural_distributions_for_rows(frame: pd.DataFrame, distributions: pd.DataFrame) -> list[pd.DataFrame]:
    """Resolve and validate one compound distribution for each structural row."""
    result = []
    for row in frame.itertuples(index=False):
        distribution = distributions.loc[
            distributions["family_key"].eq(str(row.family_key))
            & distributions["source_level"].astype(str).eq(str(row.source_level))
        ]
        validate_distribution(distribution, row, f"{row.family_title} {row.rank_label}")
        result.append(distribution)
    return result


def plot_structural(
    frame: pd.DataFrame,
    distributions: pd.DataFrame,
    all_significant: pd.DataFrame,
    all_compounds: pd.DataFrame,
    maximum_rows_per_direction: int,
    fdr_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render selected structural classes as horizontal nMAP boxplots."""
    require_columns(
        frame,
        {
            "family_order",
            "family_key",
            "row_order",
            "level",
            "source_level",
            "rank_label",
            "family_title",
            "display_kind",
            "n_exposed",
            "active_exposed",
            "nmap_median",
            "nmap_q25",
            "nmap_q75",
            "estimate",
            "significant_at_5pct_fdr",
        },
        STRUCTURAL_TABLE,
    )
    require_columns(distributions, {"family_key", "source_level", "nmap", "active"}, STRUCTURAL_DISTRIBUTION_TABLE)
    require_columns(
        all_significant,
        {"family_title", "selected_for_figure", "n_tested_family"},
        STRUCTURAL_ALL_SIGNIFICANT_TABLE,
    )
    families = frame[["family_order", "family_key", "family_title", "display_kind"]].drop_duplicates()
    families = families.sort_values("family_order", kind="stable")
    if len(families) != 3 or not frame["significant_at_5pct_fdr"].astype(bool).all():
        raise ValueError(f"{STRUCTURAL_TABLE} must contain three significant structural families")
    directions = frame["rank_label"].astype(str).str[0]
    if not directions.isin({"E", "D"}).all():
        raise ValueError(f"{STRUCTURAL_TABLE} rank labels must begin with E or D")

    counts = {
        title: (int(group["n_tested_family"].iloc[0]), len(group))
        for title, group in all_significant.groupby("family_title", sort=False)
    }
    rdkit_tested, rdkit_significant = counts["RDKit functional groups"]
    brics_tested, brics_significant = counts["BRICS fragments"]
    murcko_tested, murcko_significant = counts["Murcko scaffolds"]
    caption = (
        "Compound-level activity nMAP for members of RDKit functional groups, BRICS fragments, and Murcko "
        "scaffolds. Fisher odds ratios compare members with compounds having valid structures but lacking the "
        "class; invalid structures are excluded. Benjamini-Hochberg correction is applied within each family. "
        f"At {100 * fdr_threshold:g}% FDR, {rdkit_significant}/{rdkit_tested} RDKit, "
        f"{brics_significant}/{brics_tested} BRICS, and "
        f"{murcko_significant}/{murcko_tested} Murcko classes passed. Up to {maximum_rows_per_direction} Higher "
        f"(blue) and {maximum_rows_per_direction} Lower (orange) classes per family are shown ({len(frame)} total); "
        f"`{STRUCTURAL_ALL_SIGNIFICANT_TABLE}` contains all {len(all_significant)} passing classes. All analyzed "
        "compounds is display context, not the test comparator."
    )
    panel_block_in = 13.5 * 0.915
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=10.6,
        panel_height_in=panel_block_in,
        caption_points=7.1,
    )
    figure = plt.figure(figsize=(10.6, figure_height))
    outer = figure.add_gridspec(4, 1, height_ratios=(10, 10, 8, 1.8), hspace=0.31)
    for index, family in enumerate(families.itertuples(index=False)):
        data = frame.loc[frame["family_key"].eq(family.family_key)].sort_values("row_order", kind="stable")
        nested = outer[index].subgridspec(1, 4, width_ratios=(2.35, 5.25, 1.10, 0.85), wspace=0.03)
        identity_axis = figure.add_subplot(nested[0])
        box_axis = figure.add_subplot(nested[1])
        rate_axis = figure.add_subplot(nested[2])
        estimate_axis = figure.add_subplot(nested[3])
        positions = np.arange(1, len(data) + 1)
        identity_axis.set_xlim(0, 1)
        identity_axis.set_ylim(0.35, len(data) + 0.65)
        identity_axis.invert_yaxis()
        identity_axis.set_xticks([])
        identity_axis.set_yticks([])
        identity_axis.spines[:].set_visible(False)
        identity_axis.set_title(str(family.family_title), loc="left", fontsize=12.5, fontweight="bold", pad=10)
        for position, row in zip(positions, data.itertuples(index=False), strict=True):
            if row.display_kind == "named":
                identity_axis.text(
                    0.06, position, str(row.level).removeprefix("fr_").replace("_", " "), va="center", fontsize=9.2
                )
            else:
                identity_axis.add_artist(
                    AnnotationBbox(
                        OffsetImage(molecule_image(str(row.source_level), size=(270, 100)), zoom=0.25),
                        (0.49, position),
                        frameon=False,
                    )
                )
            higher = str(row.rank_label).startswith("E")
            direction = "Higher" if higher else "Lower"
            direction_color = HIGHER_ASSOCIATION_EDGE if higher else LOWER_ASSOCIATION_EDGE
            identity_axis.text(
                0.98,
                position,
                f"{direction} {str(row.rank_label)[1:]}",
                ha="right",
                va="center",
                fontsize=7.8,
                color=direction_color,
                fontweight="bold",
            )
        resolved = structural_distributions_for_rows(data, distributions)
        higher_rows = data["rank_label"].astype(str).str.startswith("E").tolist()
        row_edges = [HIGHER_ASSOCIATION_EDGE if higher else LOWER_ASSOCIATION_EDGE for higher in higher_rows]
        row_fills = [HIGHER_ASSOCIATION_FILL if higher else LOWER_ASSOCIATION_FILL for higher in higher_rows]
        draw_horizontal_boxes(
            box_axis,
            resolved,
            labels=None,
            facecolors=row_fills,
            edgecolors=row_edges,
            summary_labels=[""] * len(data),
            xlim=(-0.14, 1.03),
        )
        box_axis.set_title("nMAP among class members", loc="left", fontsize=8.0, color=NEUTRAL_BOX_EDGE, pad=10)
        box_axis.tick_params(axis="x", labelbottom=False)

        for text_axis, heading in (
            (rate_axis, "% active, n\nclass members"),
            (estimate_axis, "Fisher OR"),
        ):
            text_axis.set_xlim(0, 1)
            text_axis.set_ylim(0.35, len(data) + 0.65)
            text_axis.invert_yaxis()
            text_axis.set_axis_off()
            text_axis.set_title(heading, fontsize=7.8, color=NEUTRAL_BOX_EDGE, pad=9)
        for position, row, distribution, edge_color in zip(
            positions,
            data.itertuples(index=False),
            resolved,
            row_edges,
            strict=True,
        ):
            rate_axis.text(
                0.5,
                position,
                f"{100 * distribution['active'].astype(bool).mean():.1f}%\nn={format_count(len(distribution))}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=NEUTRAL_BOX_EDGE,
                linespacing=1.05,
            )
            estimate_axis.text(
                0.5,
                position,
                f"{float(row.estimate):.2f}",
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color=edge_color,
            )

    context = outer[3].subgridspec(1, 4, width_ratios=(2.35, 5.25, 1.10, 0.85), wspace=0.03)
    context_label_axis = figure.add_subplot(context[0])
    context_axis = figure.add_subplot(context[1])
    context_rate_axis = figure.add_subplot(context[2])
    context_estimate_axis = figure.add_subplot(context[3])
    context_label_axis.set_axis_off()
    context_label_axis.text(
        0.06, 0.50, "All analyzed compounds", va="center", fontsize=9.2, transform=context_label_axis.transAxes
    )
    draw_horizontal_boxes(context_axis, [all_compounds], labels=None, summary_labels=[""], xlim=(-0.14, 1.03))
    context_axis.set_xlabel("Phenotypic activity (compound-level nMAP)", fontsize=10.5, labelpad=7)
    context_axis.set_title(
        "Overall context (not a statistical comparator)",
        loc="left",
        fontsize=9.0,
        color=NEUTRAL_BOX_EDGE,
        pad=8,
    )
    for text_axis in (context_rate_axis, context_estimate_axis):
        text_axis.set_xlim(0, 1)
        text_axis.set_ylim(0.35, 1.65)
        text_axis.invert_yaxis()
        text_axis.set_axis_off()
    context_rate_axis.set_title("% active, n\nall compounds", fontsize=7.8, color=NEUTRAL_BOX_EDGE, pad=8)
    context_rate_axis.text(
        0.5,
        1,
        f"{100 * all_compounds['active'].astype(bool).mean():.1f}%\nn={format_count(len(all_compounds))}",
        ha="center",
        va="center",
        fontsize=7.6,
        color=NEUTRAL_BOX_EDGE,
        linespacing=1.05,
    )
    figure.suptitle("Structural classes", x=0.055, y=0.995, ha="left", fontsize=15.5, fontweight="bold")
    figure.text(
        0.055,
        0.964,
        (
            f"All displayed classes pass family-specific {100 * fdr_threshold:g}% FDR. "
            "Tests compare class members with other valid "
            "structures lacking that class."
        ),
        ha="left",
        va="center",
        fontsize=8.5,
        color="#374151",
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=HIGHER_ASSOCIATION_FILL,
                edgecolor=HIGHER_ASSOCIATION_EDGE,
                label="Higher activity",
            ),
            Patch(facecolor=LOWER_ASSOCIATION_FILL, edgecolor=LOWER_ASSOCIATION_EDGE, label="Lower activity"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.989),
        frameon=False,
        fontsize=7.7,
        handlelength=1.5,
    )
    figure.text(
        0.52,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=NEUTRAL_BOX_EDGE,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=1 - 1.15 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    return save_figure(figure, output_dir, "structural_classes", dpi), {"structural_classes": caption}


def functional_distributions_for_rows(frame: pd.DataFrame, distributions: pd.DataFrame) -> list[pd.DataFrame]:
    """Resolve and validate one compound distribution for each functional-use row."""
    result = []
    for row in frame.itertuples(index=False):
        distribution = distributions.loc[
            distributions["section"].eq(str(row.section))
            & distributions["source_level"].astype(str).eq(str(row.source_level))
        ]
        validate_distribution(distribution, row, f"{row.section} {row.category}")
        result.append(distribution)
    return result


def plot_functional(
    frame: pd.DataFrame,
    distributions: pd.DataFrame,
    all_compounds: pd.DataFrame,
    fup_display_threshold: float,
    fdr_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render FUC and significant FUP groups as horizontal nMAP boxplots."""
    require_columns(
        frame,
        {
            "section",
            "row_order",
            "category",
            "source_level",
            "significant",
            "inference_estimate",
            "n_exposed",
            "active_exposed",
            "nmap_median",
            "nmap_q25",
            "nmap_q75",
        },
        FUNCTIONAL_TABLE,
    )
    require_columns(distributions, {"section", "source_level", "nmap", "active"}, FUNCTIONAL_DISTRIBUTION_TABLE)
    fuc = frame.loc[frame["section"].eq("FUC")].sort_values("row_order", kind="stable")
    all_fup = frame.loc[frame["section"].eq("FUP")]
    fup = all_fup.loc[all_fup["significant"].astype(bool)].sort_values("row_order", kind="stable")
    if fuc.empty or fup.empty:
        raise ValueError(f"{FUNCTIONAL_TABLE} must contain tested FUC rows and significant FUP rows")
    if fuc["significant"].astype(bool).any():
        raise ValueError("The selected functional-use design requires every displayed FUC row to be nonsignificant")
    fup_higher = fup.loc[fup["inference_estimate"] > 1].copy()
    fup_lower = fup.loc[fup["inference_estimate"] < 1].copy()
    if fup_higher.empty or fup_lower.empty or len(fup_higher) + len(fup_lower) != len(fup):
        raise ValueError("Significant FUP rows must contain non-null higher and lower continuous-model associations")

    caption = (
        "Compound-level activity nMAP by curated functional use category (FUC) and predicted functional use "
        "probability (FUP). Two-sided Fisher tests compare each FUC with other FUC-annotated compounds; missing "
        f"annotations are excluded. None of {len(fuc)} testable FUCs passed Benjamini-Hochberg "
        f"{100 * fdr_threshold:g}% FDR. FUP odds "
        "ratios are from logistic regression on continuous probabilities and are reported per interquartile-range "
        f"increase; {len(fup)} of {len(all_fup)} endpoints passed {100 * fdr_threshold:g}% FDR. "
        "FUP boxplots show compounds with "
        f"probability >= {fup_display_threshold:g}, but tests use every non-null probability. All analyzed compounds "
        "is display context, not the test comparator."
    )
    panel_block_in = 12.2 * 0.945
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=9.8,
        panel_height_in=panel_block_in,
        caption_points=7.25,
    )
    figure = plt.figure(figsize=(9.8, figure_height))
    outer = figure.add_gridspec(
        6,
        1,
        height_ratios=(1.70, len(fuc), 2.20, len(fup_higher), len(fup_lower), 1.7),
        hspace=0.38,
    )

    fuc_header = figure.add_subplot(outer[0])
    fuc_header.set_axis_off()
    fuc_header.text(
        0,
        0.98,
        "Curated functional use categories (FUC)",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        transform=fuc_header.transAxes,
    )
    fuc_header.text(
        0,
        0.10,
        (
            f"None of {len(fuc)} testable categories passed {100 * fdr_threshold:g}% FDR. "
            "Tests compare each category with other "
            "FUC-annotated compounds,\nso missing annotation is not treated as category absence."
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=NEUTRAL_BOX_EDGE,
        linespacing=1.25,
        transform=fuc_header.transAxes,
    )

    fup_header = figure.add_subplot(outer[2])
    fup_header.set_axis_off()
    fup_header.text(
        0,
        0.98,
        "Predicted functional use probabilities (FUP)",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        transform=fup_header.transAxes,
    )
    fup_header.text(
        0,
        0.62,
        "Direction and significance use continuous predicted-use probabilities\nacross all compounds with a prediction.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#374151",
        transform=fup_header.transAxes,
    )
    fup_header.text(
        0,
        0.19,
        f"Boxplots and % active/n describe only the FUP >= {fup_display_threshold:g} subset.",
        ha="left",
        va="top",
        fontsize=8.5,
        color=NEUTRAL_BOX_EDGE,
        transform=fup_header.transAxes,
    )
    fup_header.legend(
        handles=[
            Patch(
                facecolor=HIGHER_ASSOCIATION_FILL,
                edgecolor=HIGHER_ASSOCIATION_EDGE,
                label="Higher as FUP increases",
            ),
            Patch(
                facecolor=LOWER_ASSOCIATION_FILL,
                edgecolor=LOWER_ASSOCIATION_EDGE,
                label="Lower as FUP increases",
            ),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.45),
        frameon=False,
        fontsize=7.7,
        handlelength=1.5,
    )

    def draw_functional_section(
        slot,
        data: pd.DataFrame,
        *,
        title: str,
        rate_description: str,
        estimate_description: str,
        fill_color: str,
        edge_color: str,
        title_color: str,
        show_xlabels: bool = False,
    ) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
        nested = slot.subgridspec(1, 3, width_ratios=(7.0, 1.2, 1.05), wspace=0.04)
        box_axis = figure.add_subplot(nested[0])
        rate_axis = figure.add_subplot(nested[1])
        estimate_axis = figure.add_subplot(nested[2])
        resolved = functional_distributions_for_rows(data, distributions)
        draw_horizontal_boxes(
            box_axis,
            resolved,
            list(data["category"]),
            tick_fontsize=9.2,
            facecolors=[fill_color] * len(data),
            edgecolors=[edge_color] * len(data),
            summary_labels=[""] * len(data),
            xlim=(-0.14, 1.03),
        )
        box_axis.set_title(title, loc="left", fontsize=10.4, fontweight="bold", color=title_color, pad=9)
        box_axis.tick_params(axis="x", labelbottom=show_xlabels)
        if show_xlabels:
            box_axis.set_xlabel("Phenotypic activity (compound-level nMAP)", fontsize=10.2, labelpad=7)

        positions = np.arange(1, len(data) + 1)
        for text_axis, heading in ((rate_axis, rate_description), (estimate_axis, estimate_description)):
            text_axis.set_xlim(0, 1)
            text_axis.set_ylim(0.35, len(data) + 0.65)
            text_axis.invert_yaxis()
            text_axis.set_axis_off()
            text_axis.set_title(heading, fontsize=7.8, color=NEUTRAL_BOX_EDGE, pad=9)
        for position, row, distribution in zip(positions, data.itertuples(index=False), resolved, strict=True):
            rate_axis.text(
                0.5,
                position,
                f"{100 * distribution['active'].astype(bool).mean():.1f}%\nn={format_count(len(distribution))}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=NEUTRAL_BOX_EDGE,
                linespacing=1.05,
            )
            estimate_axis.text(
                0.5,
                position,
                f"{float(row.inference_estimate):.2f}",
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold",
                color=edge_color,
            )
        return box_axis, rate_axis, estimate_axis

    draw_functional_section(
        outer[1],
        fuc,
        title="nMAP among category members",
        rate_description="% active, n\ncategory members",
        estimate_description="Fisher OR",
        fill_color=NEUTRAL_BOX_FILL,
        edge_color=NEUTRAL_BOX_EDGE,
        title_color="#374151",
    )
    draw_functional_section(
        outer[3],
        fup_higher,
        title="Activity increases as predicted-use probability increases",
        rate_description=f"% active, n\nFUP >= {fup_display_threshold:g}",
        estimate_description="OR per IQR\nall predictions",
        fill_color=HIGHER_ASSOCIATION_FILL,
        edge_color=HIGHER_ASSOCIATION_EDGE,
        title_color=HIGHER_ASSOCIATION_EDGE,
    )
    draw_functional_section(
        outer[4],
        fup_lower,
        title="Activity decreases as predicted-use probability increases",
        rate_description=f"% active, n\nFUP >= {fup_display_threshold:g}",
        estimate_description="OR per IQR\nall predictions",
        fill_color=LOWER_ASSOCIATION_FILL,
        edge_color=LOWER_ASSOCIATION_EDGE,
        title_color=LOWER_ASSOCIATION_EDGE,
    )

    context = outer[5].subgridspec(1, 3, width_ratios=(7.0, 1.2, 1.05), wspace=0.04)
    context_axis = figure.add_subplot(context[0])
    context_rate_axis = figure.add_subplot(context[1])
    context_estimate_axis = figure.add_subplot(context[2])
    draw_horizontal_boxes(
        context_axis,
        [all_compounds],
        ["All analyzed compounds"],
        tick_fontsize=9.2,
        summary_labels=[""],
        xlim=(-0.14, 1.03),
    )
    context_axis.set_xlabel("Phenotypic activity (compound-level nMAP)", fontsize=10.2, labelpad=7)
    context_axis.set_title(
        "Overall context (not a statistical comparator)", loc="left", fontsize=9.0, color=NEUTRAL_BOX_EDGE, pad=8
    )
    for text_axis in (context_rate_axis, context_estimate_axis):
        text_axis.set_xlim(0, 1)
        text_axis.set_ylim(0.35, 1.65)
        text_axis.invert_yaxis()
        text_axis.set_axis_off()
    context_rate_axis.set_title("% active, n\nall compounds", fontsize=7.8, color=NEUTRAL_BOX_EDGE, pad=8)
    context_rate_axis.text(
        0.5,
        1,
        f"{100 * all_compounds['active'].astype(bool).mean():.1f}%\nn={format_count(len(all_compounds))}",
        ha="center",
        va="center",
        fontsize=7.6,
        color=NEUTRAL_BOX_EDGE,
        linespacing=1.05,
    )
    figure.suptitle("Functional use", x=0.055, y=1 - 0.14 / figure_height, ha="left", fontsize=15.5, fontweight="bold")
    figure.text(
        0.52,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=7.25,
        color=NEUTRAL_BOX_EDGE,
        linespacing=1.35,
    )
    figure.subplots_adjust(
        left=0.30,
        right=0.985,
        top=1 - 0.67 / figure_height,
        bottom=caption_fraction + 0.55 / figure_height,
    )
    return save_figure(figure, output_dir, "functional_use", dpi), {"functional_use": caption}


def select_consistency_pairs(relationships: pd.DataFrame) -> pd.DataFrame:
    """Select both tested scopes when either scope passes the global FDR threshold."""
    selected_effects = relationships.loc[
        relationships["status"].eq("tested") & relationships["global_significant_at_5pct_fdr"].astype(bool)
    ].copy()
    if selected_effects.empty:
        raise ValueError("Consistency table has no globally significant tested effect")
    selected_effects["scope_order"] = selected_effects["effect_scope"].map({"between_group": 0, "within_group": 1})
    selected_effects = selected_effects.sort_values(
        ["scope_order", "estimate_nap_per_component_sd"], ascending=[True, False], kind="stable"
    )
    selected_pairs = selected_effects[["source_id", "trait_id"]].drop_duplicates().copy()
    selected_pairs["pair_order"] = np.arange(len(selected_pairs))
    paired = relationships.merge(selected_pairs, on=["source_id", "trait_id"], how="inner", validate="many_to_one")
    if not paired["status"].eq("tested").all():
        raise ValueError("Every selected consistency pair must have a tested counterpart")
    if not paired.groupby(["source_id", "trait_id"])["effect_scope"].nunique().eq(2).all():
        raise ValueError("Every selected consistency pair must have both effect scopes")
    if not paired.groupby(["source_id", "trait_id", "effect_scope"]).size().eq(1).all():
        raise ValueError("Every selected consistency pair must have one estimate per scope")
    return paired


def compact_consistency_label(source: str, trait: str) -> str:
    """Put the intrinsic property before its annotation class."""
    source_names = {
        "MOTIVE BioKG targets": "BioKG",
        "MOTIVE OpenTargets targets": "OpenTargets",
        "Repurposing Hub targets": "Repurposing Hub",
        "ToxCast active assays": "ToxCast",
    }
    trait_names = {
        "Normalized post-treatment Cell Painting cell density": "Cell density",
        "Quantitative estimate of drug-likeness": "QED",
        "Topological polar surface area": "TPSA",
        "PAINS structural alert": "PAINS alert",
    }
    label = f"{trait_names.get(trait, trait)} | {source_names.get(source, source)}"
    return {
        "Calculated LogP | OpenTargets": "LogP | OpenTargets",
    }.get(label, label)


def format_property_anchor(value: float, row: pd.Series) -> str:
    """Format a group-mean property value in interpretable natural units."""
    if row["value_type"] == "boolean":
        return f"{100 * value:.0f}%"
    if row["unit"] == "Da":
        return f"{value:.0f} Da"
    if row["unit"] == "count":
        return f"{value:.1f}"
    if row["unit"] == "angstrom^2":
        return f"{value:.0f} A^2"
    if row["unit"] == "source-adjusted plate z-score":
        return f"{value:.2f} z"
    return f"{value:.2f}"


def representative_group_examples(
    effects: pd.DataFrame,
    group_properties: pd.DataFrame,
) -> pd.DataFrame:
    """Attach deterministic property-tail examples selected without outcome values."""
    required = {
        "annotation_source",
        "annotation_group",
        "trait_id",
        "n_property_compounds",
        "property_mean",
        "eligible_group_primary",
    }
    require_columns(group_properties, required, CONSISTENCY_GROUP_TABLE)
    eligible = group_properties.loc[group_properties["eligible_group_primary"].astype(bool)].copy()
    records: list[dict[str, object]] = []
    for _, effect in effects.iterrows():
        groups = eligible.loc[
            eligible["annotation_source"].eq(effect["source_id"]) & eligible["trait_id"].eq(effect["trait_id"])
        ].copy()
        if len(groups) != int(effect["n_groups"]):
            raise ValueError(
                f"Group audit mismatch for {effect['source_id']} / {effect['trait_id']}: "
                f"expected {int(effect['n_groups'])}, found {len(groups)}"
            )
        chosen: dict[str, object] = effect.to_dict()
        for tail, quantile in (("low", 0.10), ("high", 0.90)):
            target = float(groups["property_mean"].quantile(quantile))
            example = (
                groups.assign(distance_from_tail=(groups["property_mean"] - target).abs())
                .sort_values(
                    ["distance_from_tail", "n_property_compounds", "annotation_group"],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .iloc[0]
            )
            chosen[f"{tail}_group"] = example["annotation_group"]
            chosen[f"{tail}_property_mean"] = float(example["property_mean"])
            chosen[f"{tail}_n_compounds"] = int(example["n_property_compounds"])
        records.append(chosen)
    return pd.DataFrame.from_records(records)


def consistency_heatmap_color_limit(relationships: pd.DataFrame) -> float:
    """Return the shared symmetric color limit for consistency effect views."""
    required = {
        "trait_id",
        "effect_scope",
        "status",
        "estimate_nap_per_component_sd",
    }
    require_columns(relationships, required, CONSISTENCY_TABLE)
    heatmap_cells = relationships.loc[
        relationships["trait_id"].isin(CONSISTENCY_HEATMAP_TRAIT_IDS)
        & relationships["effect_scope"].isin(("between_group", "within_group"))
        & relationships["status"].eq("tested")
    ]
    estimates = heatmap_cells["estimate_nap_per_component_sd"].to_numpy(dtype=float)
    if not len(estimates) or not np.isfinite(estimates).all():
        raise ValueError("Every tested consistency heatmap cell must have a finite estimate")
    return max(float(np.ceil(np.abs(estimates).max() * 100) / 100), 0.01)


def consistency_effect_normalization(relationships: pd.DataFrame) -> matplotlib.colors.Normalize:
    """Return the normalization shared by the consistency heatmap and sidebar."""
    color_limit = consistency_heatmap_color_limit(relationships)
    return matplotlib.colors.Normalize(vmin=-color_limit, vmax=color_limit)


def consistency_effect_colors(relationships: pd.DataFrame, estimates: np.ndarray) -> np.ndarray:
    """Map effects to the same blue-positive, red-negative scale as the heatmap."""
    values = np.asarray(estimates, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Every displayed consistency effect must have a finite estimate")
    normalize = consistency_effect_normalization(relationships)
    return plt.colormaps[CONSISTENCY_COLORMAP](normalize(values))


def draw_consistency_sidebar(
    figure,
    relationships: pd.DataFrame,
    group_properties: pd.DataFrame,
    wrapped_caption: str,
    caption_fraction: float,
    figure_height: float,
) -> pd.DataFrame:
    """Draw significant between-group effects and named property-tail examples."""
    required = {
        "source_id",
        "source_label",
        "trait_id",
        "trait_label",
        "effect_scope",
        "status",
        "estimate_nap_per_component_sd",
        "ci_lower_nap_per_component_sd",
        "ci_upper_nap_per_component_sd",
        "global_significant_at_5pct_fdr",
        "n_groups",
        "value_type",
        "unit",
    }
    require_columns(relationships, required, CONSISTENCY_TABLE)
    paired = select_consistency_pairs(relationships)
    data = paired.loc[
        paired["effect_scope"].eq("between_group") & paired["global_significant_at_5pct_fdr"].astype(bool)
    ].copy()
    data = data.sort_values("estimate_nap_per_component_sd", kind="stable")
    data = representative_group_examples(data, group_properties)
    data["compact_label"] = [
        compact_consistency_label(source, trait)
        for source, trait in zip(data["source_label"], data["trait_label"], strict=True)
    ]

    gray = "#6B7280"
    dark = "#111827"
    grid = "#E4E7EC"
    y_values = np.arange(len(data))
    estimates = data["estimate_nap_per_component_sd"].to_numpy(dtype=float)
    lower = data["ci_lower_nap_per_component_sd"].to_numpy(dtype=float)
    upper = data["ci_upper_nap_per_component_sd"].to_numpy(dtype=float)
    effect_colors = consistency_effect_colors(relationships, estimates)

    layout = figure.add_gridspec(1, 2, width_ratios=(0.98, 2.02), wspace=0.06)
    effect_axis = figure.add_subplot(layout[0])
    group_axis = figure.add_subplot(layout[1], sharey=effect_axis)
    effect_axis.barh(
        y_values,
        estimates,
        height=0.58,
        color=effect_colors,
        edgecolor=effect_colors,
        linewidth=0.45,
    )
    for y_value, estimate, lower_bound, upper_bound, color in zip(
        y_values,
        estimates,
        lower,
        upper,
        effect_colors,
        strict=True,
    ):
        effect_axis.errorbar(
            estimate,
            y_value,
            xerr=[[estimate - lower_bound], [upper_bound - estimate]],
            color=color,
            marker="none",
            linestyle="none",
            capsize=1.8,
            linewidth=0.8,
            zorder=3,
        )
    effect_axis.axvline(0, color="#D1D5DB", linestyle="--", linewidth=0.55)
    effect_axis.set_yticks(y_values, data["compact_label"])
    effect_axis.set_xlim(-0.15, 0.16)
    effect_axis.set_xticks([-0.15, 0.0, 0.15])
    effect_axis.set_xlabel("Change in consistency per property SD", fontsize=6.2, labelpad=1.5)
    effect_axis.grid(axis="x", color=grid, linewidth=0.4)
    effect_axis.set_axisbelow(True)
    effect_axis.spines[["top", "right"]].set_visible(False)
    effect_axis.spines[["left", "bottom"]].set_linewidth(0.6)
    effect_axis.spines["left"].set_bounds(y_values[0], y_values[-1])
    effect_axis.tick_params(axis="y", labelsize=7.0, pad=2.0)
    effect_axis.tick_params(axis="x", labelsize=5.7, length=2.0, pad=1.2)

    group_axis.set_xlim(0, 1)
    group_axis.set_ylim(-0.65, len(data) - 0.35)
    group_axis.axis("off")
    blended = group_axis.get_yaxis_transform()
    for y_value, (_, row) in zip(y_values, data.iterrows(), strict=True):
        low_value = format_property_anchor(float(row["low_property_mean"]), row)
        high_value = format_property_anchor(float(row["high_property_mean"]), row)
        low_group = str(row["low_group"])
        high_group = str(row["high_group"])
        if row["source_id"] == "toxcast_assay":
            low_group = low_group.removeprefix("TOX21_").removesuffix("_ratio").replace("_", " ")
            high_group = high_group.removeprefix("TOX21_").removesuffix("_ratio").replace("_", " ")
            low_group = low_group.replace("Antagonist", "antag.")
            high_group = high_group.replace("Antagonist", "antag.")
        label = (
            f"Low: {low_group} ({low_value}, n={int(row['low_n_compounds'])})\n"
            f"High: {high_group} ({high_value}, n={int(row['high_n_compounds'])})"
        )
        group_axis.text(
            0.00,
            y_value,
            label,
            transform=blended,
            fontsize=6.1,
            color=dark,
            ha="left",
            va="center",
            linespacing=1.15,
        )

    figure.suptitle(
        "Compound properties and phenotypic consistency",
        fontsize=8.5,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=1 - 0.09 / figure_height,
    )
    figure.text(
        0.50,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=FAMILY_CAPTION_PT,
        color=gray,
        linespacing=1.2,
    )
    figure.subplots_adjust(
        left=0.415,
        right=0.995,
        top=1 - 0.46 / figure_height,
        bottom=caption_fraction + 0.35 / figure_height,
    )
    return data


def consistency_caption(relationships: pd.DataFrame, fdr_threshold: float) -> str:
    """Return the caption for the standalone consistency sidebar.

    `fdr_threshold` comes from the owning 2.43 contract, so the printed cutoff cannot drift
    from the one that set the significance flags.
    """
    tested = relationships["status"].eq("tested") & relationships["effect_scope"].eq("between_group")
    significant = tested & relationships["global_significant_at_5pct_fdr"].astype(bool)
    return (
        f"Bars show {int(significant.sum())} of {int(tested.sum())} tested between-group effects passing global "
        f"{100 * fdr_threshold:g}% "
        "Benjamini-Hochberg FDR; within-group effects are not shown. Bars give fitted changes in normalized "
        "consistency per property-component standard deviation and whiskers give pointwise 95% confidence "
        "intervals. Blue is positive and red is negative. Named groups are examples nearest the 10th and 90th "
        "property percentiles, selected without consistency values. Models adjust for activity components and "
        "group size, weight groups equally, and cluster standard errors by compound and annotation group."
    )


def plot_consistency(
    relationships: pd.DataFrame,
    group_properties: pd.DataFrame,
    fdr_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Render the selected consistency sidebar as a standalone exact-size view."""
    caption = consistency_caption(relationships, fdr_threshold)
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=4.2,
        panel_height_in=5.8,
        caption_points=FAMILY_CAPTION_PT,
    )
    figure = plt.figure(figsize=(4.2, figure_height))
    draw_consistency_sidebar(figure, relationships, group_properties, wrapped_caption, caption_fraction, figure_height)
    return save_exact_figure(figure, output_dir, "phenotypic_consistency", dpi), {"phenotypic_consistency": caption}


def alignment_jitter(jcp2022: str, target_label: str, width: float = 0.18) -> float:
    """Return deterministic horizontal jitter for one target membership."""
    digest = hashlib.sha256(f"{jcp2022}|{target_label}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return width * (2.0 * unit - 1.0)


def assign_activity_ceiling(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the memberships whose compound sits at the activity nMAP ceiling, and count them."""
    compounds = primary[["jcp2022", "activity_nmap"]].drop_duplicates()
    if compounds["jcp2022"].duplicated().any():
        raise ValueError("A compound has multiple activity values in the display cohort")
    at_ceiling = np.isclose(compounds["activity_nmap"].to_numpy(dtype=float), 1.0, rtol=0, atol=1e-12)
    if not at_ceiling.any():
        raise ValueError("No compound reaches the activity nMAP ceiling")
    ceiling = primary.loc[primary["jcp2022"].isin(compounds.loc[at_ceiling, "jcp2022"])]
    counts = {
        "total_compounds": int(len(compounds)),
        "ceiling_compounds": int(ceiling["jcp2022"].nunique()),
        "ceiling_memberships": int(len(ceiling)),
    }
    return ceiling, counts


def target_alignment_caption(
    primary: pd.DataFrame,
    compounds: pd.DataFrame,
    ceiling_counts: dict[str, int],
    spearman: dict[str, float],
    activity_threshold: float,
) -> str:
    """Return the single-paragraph human-readable caption for the alignment panels."""
    total_compounds = ceiling_counts["total_compounds"]
    ceiling_compounds = ceiling_counts["ceiling_compounds"]
    ceiling_memberships = ceiling_counts["ceiling_memberships"]
    return (
        "Activity nMAP measures same-compound replicate retrieval. Target-alignment nAP compares each compound "
        "with active reference compounds annotated to the same versus unrelated OpenTargets targets; references "
        f"pass corrected activity p < {activity_threshold:.2f} and exclude the query compound. An nAP of 0 is the "
        "random expectation. "
        "Targets contain at least five analyzed compounds and at least four active references. "
        f"Top, {len(primary)} compound-target memberships from {primary['jcp2022'].nunique()} compounds and "
        f"{primary['joint_set_id'].nunique()} distinct membership/reference sets, shown without binning. "
        f"{ceiling_compounds} of {total_compounds} compounds have activity nMAP = 1.0, accounting for "
        f"{ceiling_memberships} memberships; the right marginal shows their target-alignment nAP on the same y-axis. "
        f"Middle, the same {len(compounds)} compounds reduced to one point each at the highest nAP among their "
        "annotated targets. Top and middle share axes. Bottom, target-level mean activity nMAP and mean alignment "
        "nAP; point area gives target size. The quadratic fit adjusts for annotation degree and active-reference "
        "count, with a 95% cluster-robust interval by exact membership/reference set. Spearman rho is "
        f"{spearman['spearman_memberships']:.2f} per membership, {spearman['spearman_compounds']:.2f} per compound, "
        f"{spearman['spearman_all_target_labels']:.2f} across all target labels, and "
        f"{spearman['spearman_distinct_exact_sets']:.2f} after retaining one label per distinct "
        "membership/reference set. Target-level correlations describe between-target aggregates and are not "
        "stronger compound-level effects. Permuting members while preserving target sizes gave mean absolute rho "
        f"{spearman['null_mean_abs_rho']:.2f}; the target-level empirical p was "
        f"{spearman['null_empirical_p']:.3f} over {spearman['null_iterations']} permutations."
    )


def draw_alignment_scatter(
    scatter_axis: plt.Axes,
    ceiling_axis: plt.Axes,
    frame: pd.DataFrame,
    spearman: dict[str, float],
    unit: str,
    *,
    show_membership_count: bool,
) -> None:
    """Draw one unaggregated activity-versus-consistency panel and its ceiling marginal.

    Called once per grain. `unit` selects both the correlation to annotate and the word
    the annotation uses, so the panel states what one point is without a separate title.
    """
    ceiling, counts = assign_activity_ceiling(frame)
    scatter_axis.scatter(
        frame["activity_nmap"],
        frame["target_alignment_nap"],
        s=15,
        color="#5c677d",
        alpha=0.3,
        edgecolors="none",
    )
    scatter_axis.axhline(0, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
    scatter_axis.set_xlim(-0.03, 1.03)
    scatter_axis.set_ylabel("Phenotypic consistency\nwith target (nAP)")
    # Upper-left is the sparse corner of a positive relationship, and it is where the
    # across-target panel puts its own rho, so all three rows read the same way.
    scatter_axis.text(
        0.03,
        0.96,
        f"Spearman rho\n{spearman[f'spearman_{unit}']:.2f} ({spearman[unit]} {unit})",
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
    )
    scatter_axis.grid(color="#ededed", linewidth=0.5, zorder=0)

    ceiling_box = ceiling_axis.boxplot(
        [ceiling["target_alignment_nap"]],
        positions=[0.0],
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#b45f06", "linewidth": 1.3},
        boxprops={"edgecolor": "#5c677d", "linewidth": 0.8},
        whiskerprops={"color": "#5c677d", "linewidth": 0.8},
        capprops={"color": "#5c677d", "linewidth": 0.8},
    )
    for box in ceiling_box["boxes"]:
        box.set_facecolor("#e8edf2")
        box.set_alpha(0.85)
    ceiling_axis.scatter(
        [alignment_jitter(str(row.jcp2022), str(row.target_label)) for row in ceiling.itertuples(index=False)],
        ceiling["target_alignment_nap"],
        s=10,
        color="#5c677d",
        alpha=0.3,
        edgecolors="none",
        zorder=2,
    )
    ceiling_axis.axhline(0, color="#555555", linewidth=0.7, linestyle="--", zorder=1)
    ceiling_axis.set_xlim(-0.5, 0.5)
    # "82 / 202" read as a ratio; it was compounds over the points they contribute.
    # Per compound the second number is always equal to the first, so it is dropped.
    ceiling_label = f"1.00\nceiling\n{counts['ceiling_compounds']} cpd"
    if show_membership_count:
        ceiling_label += f"\n{counts['ceiling_memberships']} pts"
    ceiling_axis.set_xticks([0.0], [ceiling_label])
    ceiling_axis.tick_params(axis="y", labelleft=False, length=0)
    ceiling_axis.tick_params(axis="x", labelsize=6.9)
    ceiling_axis.grid(axis="y", color="#ededed", linewidth=0.5, zorder=0)


def plot_target_alignment(
    memberships: pd.DataFrame,
    compound_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    curve: pd.DataFrame,
    spearman: dict[str, float],
    activity_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], str, dict[str, int]]:
    """Draw the asymmetric target-alignment panels owned by analysis 2.46.

    Every plotted value arrives precomputed: memberships, the one-row-per-target
    summary, and the fitted across-target curve. Nothing is refit here.

    Three rows, same two quantities, three units. One point per compound-target
    membership, so a compound annotated to several targets is a vertical stripe; then
    each compound once at its best-matching target, which is the reduction the property
    figures use; then one point per target. The two scatters share both axes, so the
    only difference between them is whether a multi-target compound is drawn once or
    once per target, and the across-target row is drawn once rather than repeated in a
    second figure.
    """
    require_columns(
        memberships,
        {"jcp2022", "target_label", "activity_nmap", "target_alignment_nap", "joint_set_id", "eligible_anchor_4"},
        ALIGNMENT_MEMBERSHIP_TABLE,
    )
    require_columns(
        target_summary,
        {"target_label", "mean_activity_nmap", "mean_consistency_nap", "query_count", "joint_set_id"},
        ALIGNMENT_TARGET_TABLE,
    )
    require_columns(
        curve, {"mean_activity_nmap", "mean_consistency_fit", "ci_lower", "ci_upper"}, ALIGNMENT_CURVE_TABLE
    )
    primary = memberships.loc[memberships["eligible_anchor_4"].astype(bool)].reset_index(drop=True)
    if set(primary["target_label"]) != set(target_summary["target_label"]):
        raise ValueError("Target-alignment memberships and target summary describe different targets")
    # One row per compound at its best-matching target, carrying that target's set id
    # so the caption can still report distinct membership/reference sets.
    best = compound_summary.rename(
        columns={"best_target_label": "target_label", "best_target_alignment_nap": "target_alignment_nap"}
    ).drop(columns="joint_set_id", errors="ignore")
    reduced = best.merge(
        primary[["target_label", "joint_set_id"]].drop_duplicates(),
        on="target_label",
        how="left",
        validate="many_to_one",
    )
    if reduced["joint_set_id"].isna().any():
        raise ValueError("A best-matching target is missing from the membership table")

    _, ceiling_counts = assign_activity_ceiling(primary)
    caption = target_alignment_caption(primary, reduced, ceiling_counts, spearman, activity_threshold)
    # Scoped, not global: plt.rcParams.update leaked these into whichever figure happened
    # to render next, so a panel's appearance depended on the order of calls in main().
    rc_overrides = {
        "font.family": "DejaVu Sans",
        "font.size": 6.9,
        "axes.titlesize": 7.5,
        "axes.labelsize": 9.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc_overrides):
        # The panels are the figure. The caption is transport only: it rides along into a
        # drawing tool and is cropped off, so it never sets the width or the panel height.
        wrapped_caption = fill(
            caption, width=int((ALIGNMENT_FIGURE_WIDTH - 0.3) * 72 / (ALIGNMENT_CAPTION_POINTS * 0.645))
        )
        caption_height = len(wrapped_caption.splitlines()) * ALIGNMENT_CAPTION_POINTS * 1.18 / 72 + 0.15
        figure = plt.figure(
            figsize=(
                ALIGNMENT_FIGURE_WIDTH,
                2 * ALIGNMENT_SCATTER_HEIGHT + ALIGNMENT_PANEL_HEIGHT + caption_height,
            ),
            layout="constrained",
        )
        figure.get_layout_engine().set(w_pad=0.01, wspace=0.01, hspace=0.10)
        grid = figure.add_gridspec(
            4,
            2,
            height_ratios=[
                ALIGNMENT_SCATTER_HEIGHT,
                ALIGNMENT_SCATTER_HEIGHT,
                ALIGNMENT_PANEL_HEIGHT,
                caption_height,
            ],
            width_ratios=[1.0, 0.10],
        )
        raw_axis = figure.add_subplot(grid[0, 0])
        # Shared axes are the point: the two scatters must be comparable at a glance, so
        # the reader can attribute the difference to the reduction and nothing else.
        compound_axis = figure.add_subplot(grid[1, 0], sharex=raw_axis, sharey=raw_axis)
        raw_ceiling_axis = figure.add_subplot(grid[0, 1], sharey=raw_axis)
        compound_ceiling_axis = figure.add_subplot(grid[1, 1], sharey=compound_axis)
        relationship_axis = figure.add_subplot(grid[2, :])
        caption_axis = figure.add_subplot(grid[3, :])

        draw_alignment_scatter(raw_axis, raw_ceiling_axis, primary, spearman, "memberships", show_membership_count=True)
        draw_alignment_scatter(
            compound_axis, compound_ceiling_axis, reduced, spearman, "compounds", show_membership_count=False
        )
        raw_axis.tick_params(axis="x", labelbottom=False)
        compound_axis.set_xlabel("Compound activity nMAP")

        point_sizes = 13 + 2.8 * target_summary["query_count"].to_numpy(dtype=float)
        relationship_axis.scatter(
            target_summary["mean_activity_nmap"],
            target_summary["mean_consistency_nap"],
            s=point_sizes,
            color="#5c677d",
            alpha=0.58,
            edgecolors="white",
            linewidths=0.45,
        )
        relationship_axis.fill_between(
            curve["mean_activity_nmap"],
            curve["ci_lower"],
            curve["ci_upper"],
            color="#f2b134",
            alpha=0.24,
            linewidth=0,
        )
        relationship_axis.plot(
            curve["mean_activity_nmap"], curve["mean_consistency_fit"], color="#b45f06", linewidth=2.0
        )
        relationship_axis.axhline(0, color="#666666", linewidth=0.6, linestyle="--")
        relationship_axis.set_xlabel("Mean activity nMAP of target members")
        relationship_axis.set_ylabel("Mean phenotypic consistency\nwith target (nAP)")
        relationship_axis.text(
            0.03,
            0.96,
            "Spearman rho\n"
            f"{spearman['spearman_all_target_labels']:.2f} ({len(target_summary)} targets)\n"
            f"{spearman['spearman_distinct_exact_sets']:.2f} ({target_summary['joint_set_id'].nunique()} distinct sets)\n"
            # Without this line the row invites comparison with the two above it, which
            # is not a comparison the statistics support: averaging alone buys most of it.
            f"shuffled rho {spearman['null_mean_abs_rho']:.2f}, p {spearman['null_empirical_p']:.3f}",
            transform=relationship_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
        )
        relationship_axis.grid(color="#ededed", linewidth=0.5)

        for axis in (raw_axis, compound_axis, relationship_axis):
            axis.tick_params(axis="both", labelsize=8.6)

        caption_axis.axis("off")
        caption_axis.text(
            0,
            1,
            wrapped_caption,
            va="top",
            ha="left",
            fontsize=ALIGNMENT_CAPTION_POINTS,
            linespacing=1.18,
        )
        paths = save_figure(figure, output_dir, TARGET_ALIGNMENT_STEM, dpi)
    return paths, caption, ceiling_counts


def plot_cohort_outcome_properties(
    compound_summary: pd.DataFrame,
    property_bins: pd.DataFrame,
    compound_distributions: pd.DataFrame,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, float], dict[str, str]]:
    """Draw compound properties against one per-compound outcome for the alignment cohort.

    Deliberately the same idiom as the activity panels and the OpenTargets group
    figures: one panel per property, the per-compound outcome against it. The panel
    stays a raw scatter with an unweighted fit rather than a binned median, because
    168 compounds do not support the ten-bin median-and-IQR ribbon the activity family
    draws over 112,462.
    """
    spec = COHORT_OUTCOMES[stem]
    require_columns(
        compound_summary,
        {"jcp2022", spec["outcome"], "n_retained_targets", *(c for c, _, _ in COMPOUND_PROPERTY_TRAITS)},
        ALIGNMENT_COMPOUND_TABLE,
    )
    require_columns(
        property_bins,
        {
            "trait_order",
            "trait_id",
            "title",
            "xlabel",
            "bin",
            "x",
            "n",
            spec["rho"],
        },
        ALIGNMENT_PROPERTY_BIN_TABLE,
    )
    cohort = compound_distributions.merge(
        compound_summary[["jcp2022", spec["outcome"]]],
        on="jcp2022",
        how="inner",
        validate="one_to_one",
    )
    if len(cohort) != len(compound_summary):
        raise ValueError("Compound summary contains compounds absent from the 2.45 property distributions")
    # The categorical panels reuse the activity figure's category rules on this cohort.
    cohort = cohort.assign(nmap=cohort[spec["outcome"]])
    lipinski, clinical, collections = build_boxplot_distributions(cohort)

    values = compound_summary[spec["outcome"]].to_numpy(dtype=float)
    limits = (float(values.min()) - 0.04, float(values.max()) + 0.06)
    rho_by_trait: dict[str, float] = {}

    shared_tail = (
        "Continuous panels show individual compounds with an unweighted OLS fit and Spearman rho; category panels "
        "show compound counts."
    )
    if spec["outcome"] == "best_target_alignment_nap":
        caption = (
            f"Properties of {len(compound_summary)} OpenTargets-annotated compounds plotted against target-alignment "
            "nAP at each compound's highest-scoring annotated target. The maximum can increase with the number of "
            "retained target annotations and is not comparable to group mAP. "
            + shared_tail
            + " Null calibration is at the cohort level, not per compound."
        )
    else:
        caption = (
            f"The same {len(compound_summary)} OpenTargets-annotated compounds and properties, plotted against "
            "compound activity nMAP. Target retention required sufficient active references, so this cohort differs "
            "from the full activity screen. " + shared_tail
        )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(caption)
    figure = plt.figure(figsize=(PROPERTY_FIGURE_WIDTH_IN, figure_height))
    figure.suptitle(
        spec["title"],
        fontsize=FAMILY_SUPTITLE_PT,
        fontweight="bold",
        x=0.065,
        y=1 - 0.30 / figure_height,
        ha="left",
    )
    grid = figure.add_gridspec(
        3,
        3,
        hspace=0.57,
        wspace=0.34,
        top=1 - 0.90 / figure_height,
        bottom=caption_fraction + 0.30 / figure_height,
        left=0.10,
        right=0.99,
    )
    for index, (column, title, xlabel) in enumerate(COMPOUND_PROPERTY_TRAITS):
        block = property_bins.loc[property_bins["trait_id"].eq(column)].sort_values("bin", kind="stable")
        value_axis = figure.add_subplot(grid[divmod(index, 3)])
        value_axis.set_title(title, fontsize=FAMILY_TITLE_PT, fontweight="bold", loc="left")

        x = compound_summary[column].to_numpy(dtype=float)
        value_axis.scatter(x, values, s=6, color=FAMILY_DARK_GRAY, alpha=0.45, edgecolors="none")
        slope, intercept = np.polyfit(x, values, 1)
        fit_x = np.array([x.min(), x.max()])
        value_axis.plot(fit_x, slope * fit_x + intercept, color=FAMILY_DARK_GRAY, linewidth=0.8)
        value_axis.axhline(0, color=FAMILY_LIGHT_GRAY, linestyle="--", linewidth=0.8, zorder=0)
        rho = float(block[spec["rho"]].iloc[0])
        rho_by_trait[column] = rho
        value_axis.text(
            0.03,
            0.95,
            f"rho={rho:.2f}",
            transform=value_axis.transAxes,
            ha="left",
            va="top",
            fontsize=FAMILY_ANNOTATION_PT,
            color=FAMILY_GRAY,
        )
        value_axis.set_ylim(*limits)
        value_axis.set_xlabel(xlabel, fontsize=FAMILY_AXIS_LABEL_PT, labelpad=1.5)
        if index % 3 == 0:
            value_axis.set_ylabel(spec["outcome_label"], color=FAMILY_DARK_GRAY, fontsize=FAMILY_Y_LABEL_PT, labelpad=2)
        value_axis.tick_params(labelsize=FAMILY_TICK_PT)
        value_axis.spines[["top", "right"]].set_visible(False)

    categorical = (
        (lipinski, "Lipinski violations", "Number of violations"),
        (clinical, "Clinical phase", "Clinical development stage"),
        (collections, "PAINS/collections", "Curated collection"),
    )
    for index, (distributions, title, xlabel) in enumerate(categorical):
        axis = figure.add_subplot(grid[2, index])
        positions = np.arange(1, len(distributions) + 1)
        boxplot = axis.boxplot(
            [distribution["nmap"].to_numpy(dtype=float) for _, distribution in distributions],
            positions=positions,
            widths=0.64,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor=HIGHER_ASSOCIATION_FILL, edgecolor=FAMILY_BLUE, linewidth=0.8)
        for element in ("whiskers", "caps"):
            for artist in boxplot[element]:
                artist.set(color=FAMILY_BLUE, linewidth=0.8)
        for median in boxplot["medians"]:
            median.set(color="#111827", linewidth=1.1)
        axis.set_xticks(
            positions,
            [
                f"{COMPOUND_CATEGORY_SHORT_LABELS.get(label, label)}\n({len(distribution)})"
                for label, distribution in distributions
            ],
        )
        axis.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
        axis.axhline(0, color=FAMILY_LIGHT_GRAY, linestyle="--", linewidth=0.8, zorder=0)
        axis.set_title(title, fontsize=FAMILY_TITLE_PT, fontweight="bold", loc="left")
        axis.set_xlabel(xlabel, fontsize=FAMILY_AXIS_LABEL_PT, labelpad=1.5)
        axis.set_ylim(*limits)
        axis.tick_params(labelsize=FAMILY_CATEGORY_TICK_PT)
        axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            axis.set_ylabel(spec["outcome_label"], color=FAMILY_DARK_GRAY, fontsize=FAMILY_Y_LABEL_PT, labelpad=2)

    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=FAMILY_CAPTION_PT,
        color=FAMILY_GRAY,
        linespacing=1.35,
    )
    return save_exact_figure(figure, output_dir, stem, dpi), rho_by_trait, {stem: caption}


def align_category_slots(
    reference: list[tuple[str, pd.DataFrame]],
    available: list[tuple[str, pd.DataFrame]],
    empty: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    """Give every drawer the reference population's category slots, in its order.

    Category levels are whatever a population happens to contain: the screen reaches four
    Lipinski violations and the alignment cohort stops at two. The three drawers of a
    column share an x axis, so without this the shortest list wins and silently crops the
    screen's last two boxes. Taking the screen as the reference means a level the cohort
    lacks reads as an empty slot instead of a missing column.
    """
    lookup = dict(available)
    return [(label, lookup.get(label, empty)) for label, _ in reference]


def property_grain_paragraph(grain_audit: pd.DataFrame) -> str:
    """Summarize the compound- and target-level correlation audit."""
    audit = grain_audit.sort_values("trait_order", kind="stable")
    activity_consistency = float(audit["rho_activity_consistency_compound"].iloc[0])
    flipped = audit.loc[
        np.sign(audit["rho_property_activity_compound"]) != np.sign(audit["rho_property_consistency_compound"])
    ]
    inflation = audit["aggregation_inflation"].replace([np.inf, -np.inf], np.nan).dropna()
    cleared = audit.loc[audit["empirical_p_target"] < 0.05]
    cleared_text = (
        " and ".join(f"{row.title} (p = {row.empirical_p_target:.3f})" for row in cleared.itertuples())
        if len(cleared)
        else "none"
    )
    return (
        f"Across compounds, activity and consistency have Spearman rho = {activity_consistency:.2f}; "
        f"{len(flipped)} of {len(audit)} properties have opposite-signed correlations with the two outcomes. "
        "The median absolute target-level correlation is "
        f"{inflation.median():.1f} times its compound-level value. In size-preserving permutations, the 95th "
        "percentile of absolute target-level rho ranges from "
        f"{audit['null_p95_abs_rho_target'].min():.2f} to {audit['null_p95_abs_rho_target'].max():.2f}; target-level "
        f"empirical p < 0.05 for {cleared_text}. Full values are in "
        f"`{ALIGNMENT_GRAIN_AUDIT_TABLE}`."
    )


def plot_property_drawers(
    binned_screen: pd.DataFrame,
    compound_summary: pd.DataFrame,
    compound_distributions: pd.DataFrame,
    property_bins: pd.DataFrame,
    grain_audit: pd.DataFrame,
    analyzed_compounds: int,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Stack the three property views of one figure, one drawer per view per property.

    The three existing property figures answer the same question of the same six
    properties at three populations and outcomes. Reading them side by side means
    holding three canvases in view, so this one puts the three drawers on a shared x
    axis inside each property facet: the whole screen on top, then the alignment cohort
    scored on consistency and on activity.

    The screen drawer draws one faint point per analyzed compound rather than the binned
    curve the other property figures show, so all three drawers of a column carry the
    same mark and the reader compares clouds rather than a curve against two clouds.
    """
    validate_intrinsic_frame(binned_screen)
    require_columns(
        compound_distributions,
        {"nmap", *(c for c, _, _ in COMPOUND_PROPERTY_TRAITS)},
        COMPOUND_DISTRIBUTION_TABLE,
    )
    screen_values = compound_distributions["nmap"].to_numpy(dtype=float)
    # Fixed seed, fixed row order: the jitter is the same on every rebuild. The cohort
    # keeps the hash-based jitter it shares with the alignment figure, but 112,462
    # per-compound hashes per facet is not worth the seconds.
    rng = np.random.default_rng(0)
    # Both activities first so the outcome changes once, at the last drawer. One line
    # each: three stacked drawers leave no room for wrapped y labels.
    cohort_size = len(compound_summary)
    drawers = (
        ("screen", f"Phenotypic activity\n(nMAP):\nAll {analyzed_compounds:,}\ncompounds"),
        ("activity", f"Phenotypic activity\n(nMAP):\n{cohort_size} OT-annotated\ncompounds"),
        ("consistency", f"Phenotypic consistency\n(nAP):\n{cohort_size} OT-annotated\ncompounds"),
    )
    consistency_values = compound_summary["best_target_alignment_nap"].to_numpy(dtype=float)
    activity_values = compound_summary["activity_nmap"].to_numpy(dtype=float)
    # Headroom above each cloud so the rho label lands on blank canvas, not on points.
    drawer_limits = {
        "screen": (screen_values.min() - 0.06, screen_values.max() + 0.42),
        "activity": (activity_values.min() - 0.06, activity_values.max() + 0.62),
        "consistency": (consistency_values.min() - 0.04, consistency_values.max() + 0.34),
    }
    # The category panels reuse the activity figure's category rules on each population.
    category_sets = {"screen": build_boxplot_distributions(compound_distributions)}
    empty_frame = compound_distributions.iloc[:0]
    for drawer, outcome in (("activity", "activity_nmap"), ("consistency", "best_target_alignment_nap")):
        cohort = compound_distributions.merge(
            compound_summary[["jcp2022", outcome]], on="jcp2022", how="inner", validate="one_to_one"
        )
        category_sets[drawer] = build_boxplot_distributions(cohort.assign(nmap=cohort[outcome]))
    caption = (
        f"Top, compound activity nMAP across {analyzed_compounds:,} compounds. Middle, activity nMAP across the "
        f"{cohort_size} OpenTargets-annotated compounds in the target-alignment cohort. Bottom, target-alignment nAP "
        "for the same cohort, using each compound's highest-scoring annotated target. Points are compounds with "
        "deterministic horizontal jitter; dashed lines are unweighted least-squares fits. The full-screen fits use "
        "the 1st-99th-percentile property range, whereas cohort fits use all displayed compounds. Columns share an "
        "x-axis but rows have separate y-axes. Category panels show quartiles, 1.5-interquartile-range whiskers, and "
        f"counts; points are drawn for categories with at most {CATEGORY_DOT_LIMIT:,} compounds. Labels report "
        "Spearman rho and Benjamini-Hochberg q values within the six full-screen or twelve cohort correlations. "
        + DRAWER_EXCLUDED_NOTE
        + " "
        + property_grain_paragraph(grain_audit)
    )
    # One row of nine facets: six properties, then the three category panels. Drawers
    # are 3:2, wider than tall; neither axis is in the other's units, so the aspect
    # carries no information and squares would only make the strip taller.
    left, right, wspace, drawer_gap = 0.105, 0.995, 0.18, 0.03
    span = PROPERTY_DRAWERS_WIDTH_IN * (right - left)
    column_width = span / (9.5 + 9 * wspace)
    drawer_height = column_width * DRAWER_ASPECT
    panel_height = drawer_height * (3 + 2 * drawer_gap)

    # Twenty-seven drawers make a canvas that will be reduced to fit a page, so every
    # label is set larger than the shared property-figure type scale.
    scale = DRAWER_TYPE_SCALE
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=PROPERTY_DRAWERS_WIDTH_IN,
        panel_height_in=panel_height + 1.85,
        caption_points=FAMILY_CAPTION_PT * scale,
    )
    figure = plt.figure(figsize=(PROPERTY_DRAWERS_WIDTH_IN, figure_height))
    figure.suptitle(
        "Compound properties across three views",
        fontsize=FAMILY_SUPTITLE_PT * 1.3,
        fontweight="bold",
        x=0.065,
        y=1 - 0.45 / figure_height,
        ha="left",
    )
    grid = figure.add_gridspec(
        3,
        10,
        width_ratios=[1] * 6 + [0.5] + [1] * 3,
        hspace=drawer_gap,
        wspace=wspace,
        top=1 - 1.05 / figure_height,
        bottom=caption_fraction + 0.80 / figure_height,
        left=left,
        right=right,
    )
    property_anchors: dict[int, plt.Axes] = {}
    category_anchors: dict[int, plt.Axes] = {}
    for index, (column, title, xlabel) in enumerate(COMPOUND_PROPERTY_TRAITS):
        curve = binned_screen.loc[binned_screen["trait_id"].eq(SCREEN_TRAIT_BY_COLUMN[column])].sort_values(
            "bin", kind="stable"
        )
        block = property_bins.loc[property_bins["trait_id"].eq(column)]
        # One isolated cohort compound sets this facet's whole x window and stretches its
        # fit line into empty canvas, so it is left out of the drawing and named in the
        # caption. Excluding it is display-only: the Spearman correlations still come
        # from 2.46 over every cohort compound.
        drawn = compound_summary["jcp2022"].ne(DRAWER_EXCLUDED_COMPOUNDS.get(column, "")).to_numpy()
        x = compound_summary.loc[drawn, column].to_numpy(dtype=float)
        screen_x = compound_distributions[column].to_numpy(dtype=float)
        measured = np.isfinite(screen_x) & np.isfinite(screen_values)
        # The screen's binning window, by its outer bin edges rather than its bin centers,
        # because that window is also what 2.45 fitted its line on. Taking centers here
        # would show the fit running a half-bin past the axis at both ends.
        screen_window = (float(curve["bin_lower"].min()), float(curve["bin_upper"].max()))
        # One x window per facet, set by the cohort points and that screen window. The
        # screen's own tails run far wider on some properties, so it is drawn inside this
        # window rather than over its full range.
        pad = 0.02 * float(np.ptp(x))
        x_limits = (min(screen_window[0], x.min()) - pad, max(screen_window[1], x.max()) + pad)
        # Cohort points only, and only for drawing: rotatable bonds are integers, so 168
        # compounds land in a dozen vertical stripes. The jitter is deterministic, and
        # the fit and rho stay on the true values.
        jitter_width = 0.025 * float(np.ptp(x))
        screen_jitter = rng.uniform(-jitter_width, jitter_width, size=screen_x.shape)
        jittered = x + np.array(
            [
                alignment_jitter(jcp2022, column, width=jitter_width)
                for jcp2022 in compound_summary.loc[drawn, "jcp2022"]
            ]
        )
        axes = []
        for drawer_index, (drawer, ylabel) in enumerate(drawers):
            ink, _ = DRAWER_COLORS[drawer]
            axis = figure.add_subplot(
                grid[drawer_index, index],
                sharex=axes[0] if axes else None,
                sharey=property_anchors.get(drawer_index),
            )
            property_anchors.setdefault(drawer_index, axis)
            axis.set_box_aspect(DRAWER_ASPECT)
            axes.append(axis)
            if drawer == "screen":
                # One point per compound, as in the two drawers below. 112,462 of them
                # need a far smaller, fainter mark than 168 do, so only the size and
                # opacity scale with the population.
                axis.scatter(
                    (screen_x + screen_jitter)[measured],
                    screen_values[measured],
                    s=0.5,
                    color=ink,
                    alpha=0.05,
                    edgecolors="none",
                    rasterized=True,
                )
                fit = (float(curve["fit_slope"].iloc[0]), float(curve["fit_intercept"].iloc[0]))
                # 2.45 fits on its binning window, so the line is drawn on that window and
                # nowhere else; the screen's trimmed tails are neither fitted nor shown.
                fit_span = screen_window
                # This q says less than the cohort's. At n=112,462 every one of these is
                # p < 1e-30, including rotatable bonds at rho 0.04, so it certifies that a
                # correlation is not exactly zero and nothing more. It is shown anyway
                # because the drawer below it shows one and an absence would read as a
                # missing test rather than an uninformative one; the caption says so. The
                # inference that matters for these six traits is 2.45's odds ratio per
                # IQR, corrected across its own thirteen-trait family.
                screen_q = float(curve["rho_q_value"].iloc[0])
                label = (
                    f"rho={float(curve['rho'].iloc[0]):.2f}, {'q<0.001' if screen_q < 0.001 else f'q={screen_q:.3f}'}"
                )
            else:
                values = (consistency_values if drawer == "consistency" else activity_values)[drawn]
                axis.scatter(jittered, values, s=9, color=ink, alpha=0.45, edgecolors="none")
                suffix = "" if drawer == "consistency" else "_activity"
                fit = (
                    float(block[f"fit{suffix}_slope"].iloc[0]),
                    float(block[f"fit{suffix}_intercept"].iloc[0]),
                )
                fit_span = (x.min(), x.max())
                rho = float(block[f"rho{suffix}"].iloc[0])
                q_value = float(block[f"rho{suffix}_q_value"].iloc[0])
                shown_q = "q<0.001" if q_value < 0.001 else f"q={q_value:.3f}"
                label = f"rho={rho:.2f}, {shown_q}"
            # The one summary these drawers carry: a straight least-squares line, fitted
            # upstream like every other number here, giving the rank correlation in the
            # corner something to look at. Drawn only over the range it was fitted on, so
            # it never runs where nothing was fitted. It takes the drawer's own hue, so a
            # reader tracking one outcome down a column tracks one colour.
            slope, intercept = fit
            fit_x = np.array(fit_span)
            axis.plot(
                fit_x,
                slope * fit_x + intercept,
                color=ink,
                linewidth=1.4,
                linestyle=(0, (4, 2)),
                zorder=5,
            )
            axis.set_ylim(*drawer_limits[drawer])
            axis.axhline(0, color=FAMILY_LIGHT_GRAY, linestyle="--", linewidth=0.6, zorder=0)
            axis.text(
                0.03,
                0.94,
                label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=FAMILY_ANNOTATION_PT * scale * 0.72,
                color=FAMILY_GRAY,
            )
            # Shared y within the block, so only its leftmost column carries the scale.
            if index == 0:
                axis.set_ylabel(
                    ylabel,
                    color=ink,
                    fontsize=FAMILY_Y_LABEL_PT * scale * 0.62,
                    labelpad=8,
                    rotation=0,
                    ha="right",
                    va="center",
                )
            else:
                axis.tick_params(axis="y", labelleft=False)
            axis.tick_params(labelsize=FAMILY_TICK_PT * scale * 0.8, length=3.0, pad=2.0)
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_linewidth(0.6)
        axes[0].set_title(title, fontsize=FAMILY_TITLE_PT * 1.15, fontweight="bold", loc="left")
        for axis in axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel(xlabel, fontsize=FAMILY_AXIS_LABEL_PT * scale * 0.68, labelpad=2.5)
        axes[0].set_xlim(*x_limits)

    # The same three populations again, now over categories rather than a continuous
    # property. Boxplots here rather than the scatter-and-fit above, because a fit needs
    # an ordered x and these are levels; the quartiles are also what the earlier category
    # figures showed, so the read carries over. The compounds go on top of the box wherever
    # the category is small enough to draw them, because several cohort categories hold
    # fewer than ten compounds and a box alone would not show it.
    for index, (title, xlabel) in enumerate(CATEGORY_DRAWER_BLOCKS):
        axes = []
        for drawer_index, (drawer, ylabel) in enumerate(drawers):
            ink, tint = DRAWER_COLORS[drawer]
            axis = figure.add_subplot(
                grid[drawer_index, index + 7],
                sharex=axes[0] if axes else None,
                sharey=category_anchors.get(drawer_index),
            )
            category_anchors.setdefault(drawer_index, axis)
            axes.append(axis)
            distributions = align_category_slots(
                category_sets["screen"][index], category_sets[drawer][index], empty_frame
            )
            # Whiskers reach 1.5 IQR and outliers are not drawn, matching the category
            # boxes on the activity figure.
            drawn_slots = [slot for slot, (_, frame) in enumerate(distributions, start=1) if len(frame)]
            boxplot = axis.boxplot(
                [frame["nmap"].to_numpy(dtype=float) for _, frame in distributions if len(frame)],
                positions=drawn_slots,
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
            )
            for box in boxplot["boxes"]:
                box.set(facecolor=tint, edgecolor=ink, linewidth=0.9)
            for element in ("whiskers", "caps"):
                for artist in boxplot[element]:
                    artist.set(color=ink, linewidth=0.9)
            for median in boxplot["medians"]:
                median.set(color=ink, linewidth=1.8)
            # The compounds themselves over the box, and the count above it: a box drawn
            # on nine compounds should not look like one drawn on a hundred thousand.
            for slot, (_, frame) in enumerate(distributions, start=1):
                values = frame["nmap"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                if values.size <= CATEGORY_DOT_LIMIT:
                    axis.scatter(
                        slot + rng.uniform(-0.22, 0.22, size=values.size),
                        values,
                        s=5,
                        color=ink,
                        alpha=0.55,
                        edgecolors="none",
                        zorder=4,
                    )
                axis.text(
                    slot,
                    0.98,
                    format_count(values.size),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=FAMILY_ANNOTATION_PT * scale * 0.55,
                    color=FAMILY_GRAY,
                )
            positions = np.arange(1, len(distributions) + 1)
            axis.set_xticks(
                positions,
                [COMPOUND_CATEGORY_SHORT_LABELS.get(label, label) for label, _ in distributions],
                rotation=30,
                ha="right",
            )
            axis.set_xlim(0.4, len(distributions) + 0.6)
            low, high = drawer_limits[drawer]
            axis.set_ylim(low, low + 1.14 * (high - low))
            axis.axhline(0, color=FAMILY_LIGHT_GRAY, linestyle="--", linewidth=0.6, zorder=0)
            # These three share their own y scale, distinct from the property block
            # because the counts above the boxes need headroom the property drawers do not.
            if index:
                axis.tick_params(axis="y", labelleft=False)
            axis.tick_params(labelsize=FAMILY_TICK_PT * scale * 0.8, length=3.0, pad=2.0)
            axis.tick_params(axis="x", labelsize=FAMILY_TICK_PT * scale * (0.8 if index == 0 else 0.62))
            axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_linewidth(0.6)
        axes[0].set_title(title, fontsize=FAMILY_TITLE_PT * 1.15, fontweight="bold", loc="left")
        for axis in axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel(xlabel, fontsize=FAMILY_AXIS_LABEL_PT * scale * 0.68, labelpad=2.5)
    figure.text(
        0.5,
        0.012,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=FAMILY_CAPTION_PT * scale,
        color=FAMILY_GRAY,
        linespacing=1.35,
    )
    return save_exact_figure(figure, output_dir, PROPERTY_DRAWERS_STEM, dpi), {PROPERTY_DRAWERS_STEM: caption}


def plot_consistency_heatmap(
    relationships: pd.DataFrame,
    activity_threshold: float,
    fdr_threshold: float,
    output_dir: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, str]]:
    """Restore the full source-by-attribute between/within consistency view."""
    required = {
        "source_id",
        "source_order",
        "source_label",
        "trait_id",
        "effect_scope",
        "status",
        "estimate_nap_per_component_sd",
        "global_significant_at_5pct_fdr",
    }
    require_columns(relationships, required, CONSISTENCY_TABLE)
    heatmap = relationships.loc[
        relationships["trait_id"].isin(CONSISTENCY_HEATMAP_TRAIT_IDS)
        & relationships["effect_scope"].isin(("between_group", "within_group"))
    ].copy()
    missing_traits = sorted(set(CONSISTENCY_HEATMAP_TRAIT_IDS) - set(heatmap["trait_id"]))
    if missing_traits:
        raise ValueError(f"{CONSISTENCY_TABLE} is missing heatmap attributes: {missing_traits}")
    keys = ["source_id", "trait_id", "effect_scope"]
    if heatmap.duplicated(keys).any():
        raise ValueError(f"{CONSISTENCY_TABLE} contains duplicate heatmap cells")

    source_order = (
        heatmap[["source_order", "source_id", "source_label"]]
        .drop_duplicates()
        .sort_values("source_order", kind="stable")
    )
    if source_order["source_id"].duplicated().any():
        raise ValueError(f"{CONSISTENCY_TABLE} maps a source to multiple labels or display orders")
    source_ids = source_order["source_id"].tolist()
    source_labels = source_order["source_label"].tolist()
    expected = pd.MultiIndex.from_product(
        [source_ids, CONSISTENCY_HEATMAP_TRAIT_IDS, ("between_group", "within_group")],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(heatmap[keys])
    missing_cells = expected.difference(observed)
    if len(missing_cells):
        raise ValueError(f"{CONSISTENCY_TABLE} is missing {len(missing_cells)} heatmap cells")

    tested = heatmap["status"].eq("tested")
    significant = heatmap["global_significant_at_5pct_fdr"].fillna(False).astype(bool)
    if (significant & ~tested).any():
        raise ValueError("An untested consistency heatmap cell cannot be significant")
    color_normalization = consistency_effect_normalization(heatmap)
    caption = (
        f"Rows are {len(source_ids)} annotation sources; columns are eight structural properties, post-treatment "
        "Cell Painting cell density, and four clinical stages. Compounds pass the upstream activity filter "
        f"(corrected p < {activity_threshold:g}). Upper and lower heatmaps show between- and within-group effects, "
        "respectively, per "
        "property-component standard deviation. Blue is positive, red is negative, white dots mark global "
        f"{100 * fdr_threshold:g}% "
        "Benjamini-Hochberg FDR, and gray cells were not tested. Clinical-stage contrasts use compounds with another "
        "stage annotation as the comparator. Models adjust for activity components and log2 group size, weight "
        "groups equally, and cluster standard errors by compound and annotation group."
    )
    wrapped_caption, figure_height, caption_fraction = property_figure_caption_layout(
        caption,
        width_in=CONSISTENCY_HEATMAP_FIGURE_SIZE[0],
        panel_height_in=CONSISTENCY_HEATMAP_FIGURE_SIZE[1],
        caption_points=8.5,
    )

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(CONSISTENCY_HEATMAP_FIGURE_SIZE[0], figure_height),
            sharex=True,
            sharey=True,
        )
        image = None
        for axis, scope, panel_title in zip(
            axes,
            ("between_group", "within_group"),
            ("Between annotation groups", "Within annotation groups"),
            strict=True,
        ):
            scoped = heatmap.loc[heatmap["effect_scope"].eq(scope)].set_index(["source_id", "trait_id"])
            values = np.full((len(source_ids), len(CONSISTENCY_HEATMAP_TRAIT_IDS)), np.nan)
            significant_y: list[int] = []
            significant_x: list[int] = []
            for y_index, source_id in enumerate(source_ids):
                for x_index, trait_id in enumerate(CONSISTENCY_HEATMAP_TRAIT_IDS):
                    record = scoped.loc[(source_id, trait_id)]
                    if record["status"] == "tested":
                        values[y_index, x_index] = record["estimate_nap_per_component_sd"]
                        if bool(record["global_significant_at_5pct_fdr"]):
                            significant_y.append(y_index)
                            significant_x.append(x_index)
            palette = plt.colormaps[CONSISTENCY_COLORMAP].copy()
            palette.set_bad(CONSISTENCY_UNTESTED_COLOR)
            image = axis.imshow(
                np.ma.masked_invalid(values),
                cmap=palette,
                norm=color_normalization,
                aspect="auto",
                interpolation="nearest",
            )
            axis.scatter(
                significant_x,
                significant_y,
                s=30,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            axis.set_xticks(np.arange(len(CONSISTENCY_HEATMAP_TRAIT_IDS)))
            axis.set_xticklabels(
                [CONSISTENCY_HEATMAP_LABELS[trait_id] for trait_id in CONSISTENCY_HEATMAP_TRAIT_IDS],
                fontsize=9,
            )
            axis.set_yticks(np.arange(len(source_ids)))
            axis.set_yticklabels(source_labels, fontsize=9.5)
            axis.set_title(panel_title, fontsize=11.5, fontweight="bold")
            axis.set_xticks(np.arange(-0.5, len(CONSISTENCY_HEATMAP_TRAIT_IDS), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(source_ids), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.8)
            axis.tick_params(which="minor", bottom=False, left=False)
            axis.tick_params(axis="x", labelbottom=scope == "within_group", pad=7)

        if image is None:
            raise RuntimeError("No consistency heatmap was rendered")
        panel_top = 1 - 0.66 / figure_height
        panel_bottom = caption_fraction + 0.94 / figure_height
        colorbar_axis = figure.add_axes(
            [0.89, panel_bottom + 0.12 * (panel_top - panel_bottom), 0.018, 0.76 * (panel_top - panel_bottom)]
        )
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Change in normalized AP per 1-SD attribute component", fontsize=9)
        figure.suptitle(
            "Compound properties and phenotypic consistency",
            fontsize=13,
            fontweight="bold",
            y=1 - 0.19 / figure_height,
        )
        figure.text(
            0.5,
            0.012,
            wrapped_caption,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#444444",
            linespacing=1.35,
        )
        figure.subplots_adjust(left=0.27, right=0.86, top=panel_top, bottom=panel_bottom, hspace=0.18)
        return save_exact_figure(figure, output_dir, CONSISTENCY_HEATMAP_STEM, dpi), {CONSISTENCY_HEATMAP_STEM: caption}


@app.command()
def main(
    dataset: str = typer.Option("compound_no_source7", "--dataset", "-d"),
    functional_use_dir: Path | None = typer.Option(None, "--functional-use"),
    trait_enrichment_dir: Path | None = typer.Option(None, "--trait-enrichment"),
    property_oddsratios_dir: Path | None = typer.Option(None, "--property-oddsratios"),
    target_alignment_dir: Path | None = typer.Option(None, "--target-alignment"),
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    dpi: int = typer.Option(DEFAULT_DPI, "--dpi"),
) -> None:
    """Render the composite, modular panels, and separate figures."""
    functional_use_dir = functional_use_dir or PROCESSED_DATA_DIR / "functional-use-classification"
    trait_enrichment_dir = trait_enrichment_dir or PROCESSED_DATA_DIR / "trait-activity-enrichment" / dataset
    property_oddsratios_dir = property_oddsratios_dir or PROCESSED_DATA_DIR / "property-activity-oddsratios" / dataset
    target_alignment_dir = target_alignment_dir or PROCESSED_DATA_DIR / "opentargets-target-alignment" / dataset
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsic, property_summary = load_analysis_table(property_oddsratios_dir, INTRINSIC_TABLE)
    selected_categorical, property_summary_second = load_analysis_table(property_oddsratios_dir, CATEGORICAL_TABLE)
    if property_summary != property_summary_second:
        raise ValueError("The two 2.45 plot-ready tables were not loaded from the same summary contract")
    compound_distributions, property_summary_third = load_analysis_table(
        property_oddsratios_dir, COMPOUND_DISTRIBUTION_TABLE
    )
    if property_summary != property_summary_third:
        raise ValueError("The three 2.45 plot-ready tables were not loaded from the same summary contract")
    structural, trait_summary = load_analysis_table(trait_enrichment_dir, STRUCTURAL_TABLE)
    structural_distributions, trait_summary_distributions = load_analysis_table(
        trait_enrichment_dir, STRUCTURAL_DISTRIBUTION_TABLE
    )
    structural_all_significant, trait_summary_all_significant = load_analysis_table(
        trait_enrichment_dir, STRUCTURAL_ALL_SIGNIFICANT_TABLE
    )
    if trait_summary != trait_summary_distributions or trait_summary != trait_summary_all_significant:
        raise ValueError("The structural 2.43 tables were not loaded from the same summary contract")
    functional, functional_summary = load_analysis_table(functional_use_dir, FUNCTIONAL_TABLE)
    functional_distributions, functional_summary_distributions = load_analysis_table(
        functional_use_dir, FUNCTIONAL_DISTRIBUTION_TABLE
    )
    if functional_summary != functional_summary_distributions:
        raise ValueError("The functional-use 2.42 tables were not loaded from the same summary contract")
    consistency, trait_summary_second = load_analysis_table(trait_enrichment_dir, CONSISTENCY_TABLE)
    if trait_summary != trait_summary_second:
        raise ValueError("The two 2.43 plot-ready tables were not loaded from the same summary contract")
    consistency_groups, trait_summary_third = load_analysis_table(trait_enrichment_dir, CONSISTENCY_GROUP_TABLE)
    if trait_summary != trait_summary_third:
        raise ValueError("The 2.43 effect and group-audit tables must share one summary contract")
    alignment_memberships, alignment_summary = load_analysis_table(target_alignment_dir, ALIGNMENT_MEMBERSHIP_TABLE)
    alignment_targets, alignment_summary_targets = load_analysis_table(target_alignment_dir, ALIGNMENT_TARGET_TABLE)
    alignment_curve, alignment_summary_curve = load_analysis_table(target_alignment_dir, ALIGNMENT_CURVE_TABLE)
    alignment_compounds, alignment_summary_compounds = load_analysis_table(
        target_alignment_dir, ALIGNMENT_COMPOUND_TABLE
    )
    alignment_property_bins, alignment_summary_bins = load_analysis_table(
        target_alignment_dir, ALIGNMENT_PROPERTY_BIN_TABLE
    )
    alignment_grain_audit, alignment_summary_grain = load_analysis_table(
        target_alignment_dir, ALIGNMENT_GRAIN_AUDIT_TABLE
    )
    if any(
        alignment_summary != other
        for other in (
            alignment_summary_targets,
            alignment_summary_curve,
            alignment_summary_compounds,
            alignment_summary_bins,
            alignment_summary_grain,
        )
    ):
        raise ValueError("The six 2.46 plot-ready tables were not loaded from the same summary contract")
    # Two blocks because the two panels are at different grains; the renderer only reads them.
    # Three blocks because the three rows are at three grains; the renderer only reads them.
    alignment_null = alignment_summary["numbers"]["across_target_aggregation_null"]
    alignment_spearman = (
        alignment_summary["numbers"]["across_target_relationship"]
        | alignment_summary["numbers"]["scatter_panel_relationship"]
        | {
            "null_mean_abs_rho": alignment_null["null_mean_abs_rho"],
            "null_empirical_p": alignment_null["empirical_p_value"],
            "null_iterations": alignment_null["iterations"],
        }
    )
    validate_analysis_summaries(functional_summary, trait_summary, property_summary, dataset)
    validate_consistency_group_audit(consistency_groups, trait_summary)
    structural_class_contract = summarize_structural_class_contract(structural_all_significant, trait_summary)
    functional_use_contract = summarize_functional_use_contract(functional, functional_summary)
    consistency_effect_contract = require_number_contract(
        trait_summary,
        "consistency_property_effects",
        "Trait-enrichment",
    )
    required_consistency_contract = {"fdr_threshold", "model", "covariance", "multiple_testing"}
    missing_consistency_contract = sorted(required_consistency_contract - set(consistency_effect_contract))
    if missing_consistency_contract:
        raise ValueError(f"Trait-enrichment consistency contract is missing fields: {missing_consistency_contract}")
    consistency_fdr_threshold = float(consistency_effect_contract["fdr_threshold"])
    overall_rate = float(property_summary["numbers"]["background_active_rate"])
    activity_threshold = float(property_summary["numbers"]["activity_threshold"])

    source_paths = {
        INTRINSIC_TABLE: property_oddsratios_dir / INTRINSIC_TABLE,
        CATEGORICAL_TABLE: property_oddsratios_dir / CATEGORICAL_TABLE,
        COMPOUND_DISTRIBUTION_TABLE: property_oddsratios_dir / COMPOUND_DISTRIBUTION_TABLE,
        STRUCTURAL_TABLE: trait_enrichment_dir / STRUCTURAL_TABLE,
        STRUCTURAL_DISTRIBUTION_TABLE: trait_enrichment_dir / STRUCTURAL_DISTRIBUTION_TABLE,
        STRUCTURAL_ALL_SIGNIFICANT_TABLE: trait_enrichment_dir / STRUCTURAL_ALL_SIGNIFICANT_TABLE,
        FUNCTIONAL_TABLE: functional_use_dir / FUNCTIONAL_TABLE,
        FUNCTIONAL_DISTRIBUTION_TABLE: functional_use_dir / FUNCTIONAL_DISTRIBUTION_TABLE,
        CONSISTENCY_TABLE: trait_enrichment_dir / CONSISTENCY_TABLE,
        CONSISTENCY_GROUP_TABLE: trait_enrichment_dir / CONSISTENCY_GROUP_TABLE,
        ALIGNMENT_MEMBERSHIP_TABLE: target_alignment_dir / ALIGNMENT_MEMBERSHIP_TABLE,
        ALIGNMENT_TARGET_TABLE: target_alignment_dir / ALIGNMENT_TARGET_TABLE,
        ALIGNMENT_CURVE_TABLE: target_alignment_dir / ALIGNMENT_CURVE_TABLE,
        ALIGNMENT_COMPOUND_TABLE: target_alignment_dir / ALIGNMENT_COMPOUND_TABLE,
        ALIGNMENT_PROPERTY_BIN_TABLE: target_alignment_dir / ALIGNMENT_PROPERTY_BIN_TABLE,
        ALIGNMENT_GRAIN_AUDIT_TABLE: target_alignment_dir / ALIGNMENT_GRAIN_AUDIT_TABLE,
    }
    output_paths: list[Path] = []
    for filename, source_path in source_paths.items():
        path = output_dir / filename
        shutil.copyfile(source_path, path)
        output_paths.append(path)
    lipinski, clinical, collections = build_boxplot_distributions(compound_distributions, selected_categorical)
    # Every renderer hands back the caption it drew, so `legend.md` cannot describe a figure
    # that no longer says that.
    captions: dict[str, str] = {}

    def rendered(result: tuple[list[Path], dict[str, str]]) -> list[Path]:
        paths, drawn = result
        captions.update(drawn)
        return paths

    output_paths.extend(
        rendered(
            plot_intrinsic_properties(
                intrinsic,
                int(property_summary["numbers"]["n_compounds"]),
                output_dir,
                dpi,
            )
        )
    )
    output_paths.extend(
        rendered(
            plot_compound_properties_activity(
                intrinsic,
                lipinski,
                clinical,
                collections,
                int(property_summary["numbers"]["n_compounds"]),
                int(property_summary["numbers"]["n_active"]),
                activity_threshold,
                output_dir,
                dpi,
            )
        )
    )
    output_paths.extend(
        rendered(
            plot_structural(
                structural,
                structural_distributions,
                structural_all_significant,
                compound_distributions,
                int(structural_class_contract["maximum_rows_per_direction"]),
                float(structural_class_contract["fdr_threshold"]),
                output_dir,
                dpi,
            )
        )
    )
    output_paths.extend(rendered(plot_selected_categories(lipinski, clinical, collections, output_dir, dpi)))
    output_paths.extend(
        rendered(
            plot_functional(
                functional,
                functional_distributions,
                compound_distributions,
                float(functional_use_contract["predicted_fup"]["display_threshold"]),
                float(functional_use_contract["fdr_threshold"]),
                output_dir,
                dpi,
            )
        )
    )
    output_paths.extend(
        rendered(plot_consistency(consistency, consistency_groups, consistency_fdr_threshold, output_dir, dpi))
    )
    output_paths.extend(
        rendered(plot_consistency_heatmap(consistency, activity_threshold, consistency_fdr_threshold, output_dir, dpi))
    )
    alignment_paths, alignment_caption, alignment_ceiling = plot_target_alignment(
        alignment_memberships,
        alignment_compounds,
        alignment_targets,
        alignment_curve,
        alignment_spearman,
        activity_threshold,
        output_dir,
        dpi,
    )
    output_paths.extend(alignment_paths)
    captions[TARGET_ALIGNMENT_STEM] = alignment_caption
    cohort_rho: dict[str, dict[str, float]] = {}
    for stem in COHORT_OUTCOMES:
        paths, rho_by_trait, drawn = plot_cohort_outcome_properties(
            alignment_compounds,
            alignment_property_bins,
            compound_distributions,
            stem,
            output_dir,
            dpi,
        )
        output_paths.extend(paths)
        captions.update(drawn)
        cohort_rho[stem] = rho_by_trait
    compound_consistency_rho = cohort_rho["compound_consistency_properties"]
    output_paths.extend(
        rendered(
            plot_property_drawers(
                intrinsic,
                alignment_compounds,
                compound_distributions,
                alignment_property_bins,
                alignment_grain_audit,
                int(property_summary["numbers"]["n_compounds"]),
                output_dir,
                dpi,
            )
        )
    )
    if set(captions) != set(FIGURE_OUTPUT_STEMS):
        raise AssertionError(f"Every rendered figure needs exactly one caption: {sorted(set(captions))}")
    legend_path = output_dir / "legend.md"
    write_legend(
        legend_path,
        figure_id=FIGURE_ID,
        title="Compound properties, phenotypic activity, and consistency",
        sections=[(f"{stem}.png", SECTION_TITLES[stem], captions[stem]) for stem in FIGURE_OUTPUT_STEMS],
    )
    output_paths.append(legend_path)

    selected_pairs = select_consistency_pairs(consistency)[["source_id", "trait_id"]].drop_duplicates()
    consistency_tested = consistency["status"].eq("tested")
    consistency_significant = consistency["global_significant_at_5pct_fdr"].fillna(False).astype(bool)
    displayed_between = consistency_tested & consistency["effect_scope"].eq("between_group") & consistency_significant
    upstream = {
        "functional_use": functional_use_dir / "summary.json",
        "trait_enrichment": trait_enrichment_dir / "summary.json",
        "property_oddsratios": property_oddsratios_dir / "summary.json",
        "target_alignment": target_alignment_dir / "summary.json",
    }
    numbers = {
        "figure_id": FIGURE_ID,
        "dataset": dataset,
        "figure_exports": list(FIGURE_OUTPUT_STEMS),
        "outputs_by_role": OUTPUTS_BY_ROLE,
        "composite_members": COMPOSITE_MEMBERS,
        "selected_designs": {
            "compound_properties_activity": "3x3_activity_panels_on_the_shared_property_figure_geometry",
            "intrinsic_properties": "2x3_continuous_intrinsic_properties",
            "lipinski_violations": "standalone_boxplot",
            "structural_classes": "horizontal_boxplots_with_all_context",
            "clinical_phase": "boxplots_with_all_other_compounds",
            "curated_collections": "boxplots_with_everything_reference",
            "functional_use": "horizontal_boxplots_with_all_context",
            "consistency_relationships": "significant_between_group_with_named_property_tails",
            "consistency_heatmap": "source_by_attribute_between_and_within_rdblue",
            "target_alignment": "membership_and_per_compound_scatters_over_the_across_target_fit",
            "compound_consistency_properties": "compound_scatter_with_unweighted_ols_and_category_boxes",
            "compound_cohort_activity_properties": "same_cohort_and_layout_scored_on_activity_instead",
            "property_drawers": "three_stacked_drawers_per_property_sharing_one_x_axis",
        },
        "activity_threshold": property_summary["numbers"]["activity_threshold"],
        "background_active_rate": overall_rate,
        "analyzed_compounds": int(property_summary["numbers"]["n_compounds"]),
        "phenotypically_active_compounds": int(property_summary["numbers"]["n_active"]),
        "phenotypically_active_percent": 100 * overall_rate,
        "intrinsic_traits": int(intrinsic["trait_id"].nunique()),
        "structural_rows": int(len(structural)),
        "structural_significant_rows_full": int(len(structural_all_significant)),
        "structural_selected_rows": int(structural_all_significant["selected_for_figure"].sum()),
        "structural_class_contract": structural_class_contract,
        "clinical_phase_rows": int(selected_categorical["view"].eq("clinical_phase").sum()),
        "curated_collection_rows": int(selected_categorical["view"].eq("curated_collections").sum()),
        "chemical_probes_including_sgc_members": int(
            selected_categorical.loc[selected_categorical["trait_id"].eq("collection.chemical_probes_sgc"), "n"].item()
        ),
        "functional_rows": int(len(functional)),
        "functional_fuc_tested": int(functional["section"].eq("FUC").sum()),
        "functional_fuc_significant": int(
            (functional["section"].eq("FUC") & functional["significant"].astype(bool)).sum()
        ),
        "functional_fup_tested": int(functional["section"].eq("FUP").sum()),
        "functional_fup_significant": int(
            (functional["section"].eq("FUP") & functional["significant"].astype(bool)).sum()
        ),
        "functional_use_contract": functional_use_contract,
        "compound_consistency_compounds": int(len(alignment_compounds)),
        "compound_consistency_multi_target_compounds": int(alignment_compounds["n_retained_targets"].gt(1).sum()),
        "compound_consistency_spearman_by_property": {k: round(v, 4) for k, v in compound_consistency_rho.items()},
        "compound_cohort_activity_spearman_by_property": {
            k: round(v, 4) for k, v in cohort_rho["compound_cohort_activity_properties"].items()
        },
        "target_alignment_memberships": int(alignment_memberships["eligible_anchor_4"].astype(bool).sum()),
        "target_alignment_targets": int(len(alignment_targets)),
        "target_alignment_distinct_membership_sets": int(alignment_targets["joint_set_id"].nunique()),
        "target_alignment_ceiling_compounds": int(alignment_ceiling["ceiling_compounds"]),
        "target_alignment_ceiling_memberships": int(alignment_ceiling["ceiling_memberships"]),
        "target_alignment_spearman_all_target_labels": alignment_spearman["spearman_all_target_labels"],
        "target_alignment_spearman_distinct_exact_sets": alignment_spearman["spearman_distinct_exact_sets"],
        "consistency_source_trait_pairs": int(displayed_between.sum()),
        "consistency_selected_pairs_either_scope_global_5pct_fdr": int(len(selected_pairs)),
        "consistency_displayed_scope_estimates": int(displayed_between.sum()),
        "consistency_displayed_between_group_effects": int(displayed_between.sum()),
        "consistency_between_effects_tested": int(
            (consistency_tested & consistency["effect_scope"].eq("between_group")).sum()
        ),
        "consistency_within_effects_tested": int(
            (consistency_tested & consistency["effect_scope"].eq("within_group")).sum()
        ),
        "consistency_between_effects_significant_global_5pct_fdr": int(
            (consistency_significant & consistency["effect_scope"].eq("between_group")).sum()
        ),
        "consistency_within_effects_significant_global_5pct_fdr": int(
            (consistency_significant & consistency["effect_scope"].eq("within_group")).sum()
        ),
        "consistency_display_contract": {
            "selection": (
                "Display tested between-group effects passing global "
                f"{100 * float(consistency_effect_contract['fdr_threshold']):g}% FDR"
            ),
            "counterpart": "Within-group effects are retained in the source table but omitted from this display",
            "color": "Same symmetric RdBu normalization as the heatmap, computed from all tested fixed heatmap cells across both scopes; blue is positive and red is negative",
            "interval": "Nominal pointwise 95% confidence intervals, not multiplicity-adjusted",
            "property_tail_examples": "Nearest eligible groups to the 10th and 90th property percentiles, selected without consistency values",
            "layout": "Narrow sidebar embedded in the composite and also exported as a modular panel",
        },
        "consistency_heatmap_display_contract": {
            "attributes": list(CONSISTENCY_HEATMAP_TRAIT_IDS),
            "population_filter": f"corrected_p_value < {activity_threshold:g}",
            "effect_unit": "Fitted change in normalized AP per property-component standard deviation",
            "selection": "Display every fixed source-by-attribute cell in both effect scopes",
            "color": "RdBu with one symmetric range across both panels; blue is positive, red is negative, and gray marks untested cells",
            "fdr_threshold": float(consistency_effect_contract["fdr_threshold"]),
            "significance": (
                "White dot for global Benjamini-Hochberg q < "
                f"{float(consistency_effect_contract['fdr_threshold']):g} within effect scope"
            ),
            "model": str(consistency_effect_contract["model"]),
            "covariance": str(consistency_effect_contract["covariance"]),
            "multiple_testing": str(consistency_effect_contract["multiple_testing"]),
            "weighting": "Annotation groups are weighted equally through inverse group size",
            "clinical_stage_reference": (
                "Compounds carrying another clinical-stage annotation; compounds without one are excluded"
            ),
            "untested_cells": "Fixed cluster threshold not met",
            "layout": "Between-group heatmap above within-group heatmap with one shared colorbar",
        },
        **summarize_consistency_audit(consistency_groups, consistency),
        "non_primary_evidence_locations": {
            "descriptive_associations": "2.45 binary_trait_oddsratios.csv and multiclass_trait_oddsratios.csv",
            "high_dimensional_screens": "2.43 enrichment_activity.csv and enrichment_consistency.csv",
            "post_treatment_cell_density": "2.02b cell-count-quality analysis; density also enters the 2.43 consistency model",
            "trait_coverage": "Eligible counts in the 2.43 enrichment tables and 2.45 multiclass table",
            "consistency_group_audit": CONSISTENCY_GROUP_TABLE,
        },
        "upstream_summaries": {
            label: {"filename": path.name, "sha256": sha256_file(path)} for label, path in upstream.items()
        },
    }
    files = []
    for path in output_paths:
        if path == legend_path:
            continue
        if path.suffix in {".png", ".pdf"}:
            files.append({"path": path.name, "kind": "figure"})
        else:
            files.append({"path": path.name, "kind": "table"})
    files.append({"path": legend_path.name, "kind": "data", "caption": "Figure legend"})
    write_summary(
        output_dir / "summary.json",
        description=(
            "A manuscript composite, reusable modular panels, and separate structural, functional-use, "
            "and consistency-heatmap figures for compound properties associated with Cell Painting "
            "phenotypic activity and annotation-group consistency, rendered directly from analyses "
            "2.42, 2.43, and 2.45."
        ),
        numbers=numbers,
        files=files,
    )
    logger.success(f"Rendered {FIGURE_ID}: {output_dir}")


if __name__ == "__main__":
    app()
