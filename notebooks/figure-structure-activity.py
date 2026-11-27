"""Render the supplementary structure-activity figure directly from canonical analyses.

The renderer performs no fingerprint, profile, neighborhood, or multiple-testing
analysis. It combines the cached pair analysis with the explicitly declared active-only
notebook 2.40 outputs and notebook 2.32 scaffold-series outputs, then writes publication
figures, scientific tables, a legend, and a standard summary.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from jump_production.summaries import write_legend, write_summary
from jump_production.visualization import attach_caption, save_figure

app = typer.Typer(add_completion=False)

PUBLICATION_TYPOGRAPHY = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}


def read_standard_summary(path: Path, label: str) -> dict:
    """Read and minimally validate one standard analysis summary."""
    summary = json.loads(path.read_text())
    required = {"description", "numbers", "files"}
    if missing := required - set(summary):
        raise ValueError(f"{label} summary is missing standard envelope fields: {sorted(missing)}")
    if not isinstance(summary["description"], str) or not summary["description"]:
        raise ValueError(f"{label} summary has no description")
    if not isinstance(summary["numbers"], dict):
        raise ValueError(f"{label} summary numbers must be a mapping")
    if not isinstance(summary["files"], list):
        raise ValueError(f"{label} summary files must be a list")
    for record in summary["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"{label} summary contains an invalid file record")
    return summary


PAIR_CACHE_FILES = {
    "pairs.parquet",
    "all_pair_similarity_histogram.parquet",
    "phenotypic_cliffs.csv",
    "phenotypic_cliff_target_divergence.csv",
    "activity_cliff_pairs.parquet",
    "activity_cliff_pair_summary.csv",
    "activity_cliff_scaffold_groups.csv",
    "phenotypic_activity.csv",
    "neighbor_overlap.csv",
    "moa_coherence.csv",
}


def load_pair_cache(summary_path: Path) -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    """Load the named cache tables and validate their scientific contract."""
    summary = read_standard_summary(summary_path, "STRUCTURE pair cache")
    declared_files = {record["path"] for record in summary["files"]}
    if declared_files != PAIR_CACHE_FILES:
        raise ValueError(
            "STRUCTURE pair-cache files differ from the expected contract: "
            f"missing={sorted(PAIR_CACHE_FILES - declared_files)}, "
            f"unexpected={sorted(declared_files - PAIR_CACHE_FILES)}"
        )
    root = summary_path.parent
    missing_files = sorted(name for name in PAIR_CACHE_FILES if not (root / name).is_file())
    if missing_files:
        raise FileNotFoundError(f"STRUCTURE pair-cache files are missing: {missing_files}")

    tables = {
        "pairs": pd.read_parquet(root / "pairs.parquet"),
        "all_pair_similarity_histogram": pd.read_parquet(root / "all_pair_similarity_histogram.parquet"),
        "phenotypic_cliffs": pd.read_csv(root / "phenotypic_cliffs.csv"),
        "activity_cliff_pairs": pd.read_parquet(root / "activity_cliff_pairs.parquet"),
        "activity_cliff_pair_summary": pd.read_csv(root / "activity_cliff_pair_summary.csv"),
        "activity_cliff_scaffold_groups": pd.read_csv(root / "activity_cliff_scaffold_groups.csv"),
        "phenotypic_activity": pd.read_csv(root / "phenotypic_activity.csv"),
        "neighbor_overlap": pd.read_csv(root / "neighbor_overlap.csv"),
        "moa_coherence": pd.read_csv(root / "moa_coherence.csv"),
    }
    numbers = summary["numbers"]
    parameters = numbers.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("STRUCTURE pair-cache summary is missing parameters")

    if len(tables["pairs"]) != numbers["n_pairs"]:
        raise ValueError("Cached k-nearest-neighbor pair count differs from the summary")
    if int(tables["all_pair_similarity_histogram"]["n_pairs"].sum()) != numbers["all_possible_active_pairs"]:
        raise ValueError("Cached all-pairs histogram has the wrong denominator")
    if len(tables["phenotypic_cliffs"]) != numbers["cliff_pairs"]:
        raise ValueError("Cached itemized morphology-cliff count differs from the summary")
    if len(tables["neighbor_overlap"]) != numbers["active_compounds_analyzed"]:
        raise ValueError("Cached neighbor-overlap denominator differs from the summary")
    if len(tables["moa_coherence"]) != numbers["moa_classes_reported"]:
        raise ValueError("Cached MOA count differs from the summary")

    activity = tables["phenotypic_activity"]
    if not {"jcp2022", "below_corrected_p"} <= set(activity.columns):
        raise ValueError("Cached phenotypic activity calls are missing required columns")
    activity["jcp2022"] = activity["jcp2022"].astype(str)
    if not activity["jcp2022"].is_unique:
        raise ValueError("Cached phenotypic activity calls contain duplicate compounds")
    if activity["below_corrected_p"].isna().any():
        raise ValueError("Cached phenotypic activity calls contain missing classifications")
    if not activity["below_corrected_p"].isin([True, False]).all():
        raise ValueError("Cached phenotypic activity calls are not boolean")
    if int(activity["below_corrected_p"].eq(True).sum()) != numbers["phenotypically_active_compounds"]:
        raise ValueError("Cached active-compound count differs from the summary")

    return numbers, parameters, tables


def load_active_substructure_analysis(
    scaffold_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load the explicitly declared active-only notebook 2.40 result."""
    summary = read_standard_summary(summary_path, "active-only notebook 2.40")
    numbers = summary["numbers"]
    if numbers.get("active_only") is not True:
        raise ValueError(f"{FIGURE_ID} requires notebook 2.40 with --active-only")
    required_sections = {
        "analysis_1_scaffold_to_morphology",
        "analysis_2_morphology_to_substructure",
    }
    if missing := required_sections - set(numbers):
        raise ValueError(f"Active-only notebook 2.40 summary is missing sections: {sorted(missing)}")

    scaffold = pd.read_csv(scaffold_path)
    required_columns = {
        "scaffold",
        "n_compounds",
        "mean_cosine_sim",
        "null_mean",
        "p_value",
    }
    if missing := required_columns - set(scaffold.columns):
        raise ValueError(f"Active-only notebook 2.40 scaffold table is missing columns: {sorted(missing)}")
    if not scaffold["scaffold"].is_unique:
        raise ValueError("Active-only notebook 2.40 scaffold table has duplicate scaffolds")
    analysis_1 = numbers["analysis_1_scaffold_to_morphology"]
    if len(scaffold) != int(analysis_1["n_scaffolds_tested"]):
        raise ValueError("Notebook 2.40 scaffold denominator differs from its summary")
    if analysis_1.get("multiple_testing") != "Benjamini-Hochberg":
        raise ValueError("Notebook 2.40 must retain Benjamini-Hochberg correction")
    return (
        scaffold.sort_values("scaffold", kind="mergesort", ignore_index=True),
        numbers,
    )


def load_scaffold_cliff_series(
    scaffold_analysis_path: Path,
    cliff_scaffolds_path: Path,
    summary_path: Path,
    active_jcp: set[str],
) -> tuple[pd.DataFrame, dict, dict]:
    """Validate and expose the PhenoSeeker method-transfer series analysis.

    Notebook 2.32 analyzes all profiled compounds, not only phenotypically active
    compounds. The active counts added here make that scope visible in the public
    series table without changing the upstream reanalysis.
    """
    summary = read_standard_summary(summary_path, "notebook 2.32")["numbers"]
    parameters = summary["parameters"]
    scaffold_statistics = summary["scaffold_statistics"]
    cliff_statistics = summary["activity_cliffs"]
    reference = cliff_statistics["comparison_to_phenoseeker"]
    required_reference = {
        "paper_url",
        "release_commit",
        "released_notebook_url",
        "released_notebook_sha256",
        "released_table_url",
        "released_table_sha256",
        "reference_scaffolds_analyzed_reported",
        "reference_release_table_rows",
        "reference_cliff_scaffolds",
        "reference_profile_representation",
        "reference_scaffold_source",
        "transfer_scaffold_source",
    }
    if missing := required_reference - set(reference):
        raise ValueError(f"PhenoSeeker reference provenance is missing fields: {sorted(missing)}")
    if parameters.get("phenotypic_activity_used_for_selection") is not False:
        raise ValueError("The scaffold-series transfer must not silently select on activity status")
    analysis = pd.read_parquet(scaffold_analysis_path)
    source_series = pd.read_csv(cliff_scaffolds_path)

    required_analysis = {
        "scaffold",
        "n_compounds",
        "mean_intra_similarity",
        "mean_inter_similarity",
        "separation_score",
        "cluster_sizes",
        "min_cluster_size",
        "is_released_code_candidate",
        "is_paper_rule_candidate",
        "jcp_ids",
        "cluster_labels",
    }
    required_series = {"scaffold", "n_compounds", "separation_score", "jcp_ids"}
    if missing := required_analysis - set(analysis.columns):
        raise ValueError(f"Scaffold analysis is missing columns: {sorted(missing)}")
    if missing := required_series - set(source_series.columns):
        raise ValueError(f"Cliff-series table is missing columns: {sorted(missing)}")
    if not analysis["scaffold"].is_unique:
        raise ValueError("Scaffold analysis contains duplicate scaffolds")
    if not source_series["scaffold"].is_unique:
        raise ValueError("Cliff-series table contains duplicate scaffolds")

    n_analyzed = int(scaffold_statistics["n_scaffolds_analyzed"])
    n_paper_sensitivity = int(scaffold_statistics["n_scaffolds_paper_rule_sensitivity"])
    n_flagged = int(cliff_statistics["n_cliff_scaffolds"])
    n_flagged_paper_sensitivity = int(cliff_statistics["n_cliff_scaffolds_paper_rule_sensitivity"])
    min_compounds = int(parameters["min_compounds_per_scaffold"])
    paper_min_compounds = int(parameters["paper_text_min_compounds_per_scaffold"])
    n_clusters = int(parameters["n_clusters"])
    min_cluster_size = int(parameters["min_compounds_per_cluster"])
    separation_threshold = float(parameters["separation_threshold"])
    min_intra_similarity = float(parameters["min_intra_similarity"])
    primary_eligible = analysis["n_compounds"] >= min_compounds
    if len(analysis) != n_paper_sensitivity:
        raise ValueError("Scaffold-analysis rows differ from the paper-rule sensitivity count")
    if int(primary_eligible.sum()) != n_analyzed:
        raise ValueError("Released-code eligible series differ from the reported denominator")
    if int(analysis.loc[primary_eligible, "n_compounds"].sum()) != int(scaffold_statistics["total_compounds"]):
        raise ValueError("Scaffold-analysis compounds differ from the reported analyzed count")
    if int(analysis["n_compounds"].sum()) != int(scaffold_statistics["total_compounds_paper_rule_sensitivity"]):
        raise ValueError("Paper-rule sensitivity compounds differ from the reported count")
    if (analysis["n_compounds"] < paper_min_compounds).any():
        raise ValueError("A scaffold series falls below the paper-text minimum compound count")
    expected_paper_flagged = (analysis["separation_score"] > separation_threshold) & (
        analysis["mean_intra_similarity"] > min_intra_similarity
    )
    expected_flagged = primary_eligible & expected_paper_flagged
    if int(expected_flagged.sum()) != n_flagged:
        raise ValueError("Released-code selection differs from the reported flagged-series count")
    if int(expected_paper_flagged.sum()) != n_flagged_paper_sensitivity:
        raise ValueError("Paper-rule sensitivity differs from the reported flagged-series count")
    if not (analysis["is_released_code_candidate"] == expected_flagged).all():
        raise ValueError("Released-code candidate flags do not implement the declared rule")
    if not (analysis["is_paper_rule_candidate"] == expected_paper_flagged).all():
        raise ValueError("Paper-rule candidate flags do not implement the declared rule")
    if len(source_series) != n_flagged:
        raise ValueError("The cliff-series table is truncated or differs from the reported count")
    expected_scaffolds = set(analysis.loc[expected_flagged, "scaffold"])
    if set(source_series["scaffold"]) != expected_scaffolds:
        raise ValueError("The cliff-series table does not contain exactly the flagged scaffolds")

    if not np.allclose(
        analysis["separation_score"],
        analysis["mean_intra_similarity"] - analysis["mean_inter_similarity"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("A scaffold separation score differs from intra minus inter similarity")

    active_jcp = {str(jcp) for jcp in active_jcp}
    rows = []
    for record in analysis.to_dict("records"):
        cluster_size_value = str(record["cluster_sizes"])
        if "|" in cluster_size_value:
            cluster_sizes = sorted(int(value) for value in cluster_size_value.split("|"))
        else:
            cluster_size_map = ast.literal_eval(cluster_size_value)
            cluster_sizes = sorted(int(value) for value in cluster_size_map.values())
        if len(cluster_sizes) != n_clusters:
            raise ValueError("A scaffold series does not contain the configured number of clusters")
        if sum(cluster_sizes) != int(record["n_compounds"]):
            raise ValueError("Cluster sizes do not sum to the scaffold-series size")
        if min(cluster_sizes) < min_cluster_size:
            raise ValueError("A retained scaffold series violates the minimum cluster size")
        jcp_ids = sorted(str(value) for value in str(record["jcp_ids"]).split("|") if value)
        cluster_labels = [int(value) for value in str(record["cluster_labels"]).split("|") if value]
        if len(jcp_ids) != int(record["n_compounds"]) or len(jcp_ids) != len(set(jcp_ids)):
            raise ValueError("Scaffold-series compound IDs are incomplete or duplicated")
        if len(cluster_labels) != len(jcp_ids) or set(cluster_labels) != set(range(1, n_clusters + 1)):
            raise ValueError("Scaffold-series cluster labels are incomplete")
        n_active = sum(jcp in active_jcp for jcp in jcp_ids)
        rows.append(
            {
                "scaffold": record["scaffold"],
                "n_compounds": int(record["n_compounds"]),
                "n_phenotypically_active_compounds": int(n_active),
                "pct_phenotypically_active_compounds": float(100 * n_active / len(jcp_ids)),
                "mean_intra_morphology_cosine": float(record["mean_intra_similarity"]),
                "mean_inter_morphology_cosine": float(record["mean_inter_similarity"]),
                "separation_score": float(record["separation_score"]),
                "mean_intra_morgan_tanimoto": float(record["struct_mean_intra_similarity"]),
                "mean_inter_morgan_tanimoto": float(record["struct_mean_inter_similarity"]),
                "cluster_sizes": "|".join(str(value) for value in cluster_sizes),
                "min_cluster_size": int(min(cluster_sizes)),
                "is_released_code_candidate": bool(record["is_released_code_candidate"]),
                "is_paper_rule_candidate": bool(record["is_paper_rule_candidate"]),
                "jcp_ids": "|".join(jcp_ids),
                "cluster_labels": "|".join(str(value) for value in cluster_labels),
            }
        )
    public_series = pd.DataFrame(rows).sort_values(
        ["is_released_code_candidate", "separation_score", "scaffold"],
        ascending=[False, False, True],
        kind="mergesort",
        ignore_index=True,
    )

    configuration = {
        "scaffold_series_scope": "all compounds with valid morphology profiles; not active-only",
        "scaffold_series_profile_level": "perturbation",
        "scaffold_series_min_compounds": min_compounds,
        "scaffold_series_paper_text_min_compounds": paper_min_compounds,
        "scaffold_series_n_clusters": n_clusters,
        "scaffold_series_separation_threshold": separation_threshold,
        "scaffold_series_min_intra_similarity": min_intra_similarity,
        "scaffold_series_min_compounds_per_cluster": min_cluster_size,
        "scaffold_series_clustering_distance": parameters["clustering_distance"],
        "scaffold_series_clustering_linkage": parameters["clustering_linkage"],
        "scaffold_series_selection_role": (
            "released PhenoSeeker clustering and selection procedure transferred to "
            "JUMP scaffold memberships and CellProfiler profiles"
        ),
        "scaffold_series_phenotypic_activity_used_for_selection": False,
        "scaffold_series_reference": reference,
    }
    profile_similarity = summary["profile_similarity_distribution"]
    numbers = {
        "scaffold_series_analyzed": n_analyzed,
        "scaffold_series_compounds_analyzed": int(analysis.loc[primary_eligible, "n_compounds"].sum()),
        "scaffold_series_flagged": n_flagged,
        "scaffold_series_flagged_pct": float(100 * n_flagged / n_analyzed),
        "scaffold_series_paper_rule_analyzed": n_paper_sensitivity,
        "scaffold_series_paper_rule_flagged": n_flagged_paper_sensitivity,
        "scaffold_series_paper_rule_flagged_pct": float(100 * n_flagged_paper_sensitivity / n_paper_sensitivity),
        "scaffold_series_flagged_compounds": int(
            public_series.loc[public_series["is_released_code_candidate"], "n_compounds"].sum()
        ),
        "scaffold_series_flagged_active_compounds": int(
            public_series.loc[
                public_series["is_released_code_candidate"],
                "n_phenotypically_active_compounds",
            ].sum()
        ),
        "scaffold_series_global_cosine_mean": float(profile_similarity["mean_pairwise_cosine"]),
        "scaffold_series_global_cosine_sd": float(profile_similarity["sd_pairwise_cosine"]),
        "scaffold_series_global_cosine_mean_plus_3sd": float(profile_similarity["mean_plus_3sd"]),
        "phenoseeker_paper_series_analyzed": int(reference["reference_scaffolds_analyzed_reported"]),
        "phenoseeker_release_table_series": int(reference["reference_release_table_rows"]),
        "phenoseeker_candidate_series": int(reference["reference_cliff_scaffolds"]),
    }
    numbers["scaffold_series_flagged_active_compounds_pct"] = float(
        100 * numbers["scaffold_series_flagged_active_compounds"] / numbers["scaffold_series_flagged_compounds"]
    )
    return public_series, configuration, numbers


# Display policy owned by this rendering notebook.
STRUCTURAL_PROXY_HIGH_CUTOFF = 0.7
MORPH_LOW = 0.3
SCAFFOLD_FDR_THRESHOLD = 0.05
HISTOGRAM_BINS = 50
STRUCTURAL_PROXY_THRESHOLD_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
MORPH_THRESHOLD_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
MOA_MARKER_AREA_RANGE = (20.0, 200.0)
MOA_LABEL_OFFSET_FRACTION = 0.055
MOA_X_PADDING_MULTIPLIER = 1.35
QUADRANT_X_RIGHT_MULTIPLIER = 12.0
SCAFFOLD_ANNOTATION_POSITION = (0.97, 0.74)
DISTRIBUTION_ANNOTATION_POSITION = (0.03, 0.97)
SOURCE_FIGURE_HEIGHT_INCHES = 6.0
SMALL_PANEL_TYPOGRAPHY = {
    **PUBLICATION_TYPOGRAPHY,
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
}
FIGURE_SIZES_INCHES = {
    "similarity_map": (3.0, 3.0),
    "similar_structure_morphology": (7.0, SOURCE_FIGURE_HEIGHT_INCHES),
    "scaffold_coherence": (8.0, SOURCE_FIGURE_HEIGHT_INCHES),
    "activity_cliff_pairs": (3.0, 3.0),
    "neighbor_overlap": (7.0, SOURCE_FIGURE_HEIGHT_INCHES),
    "moa_coherence": (9.0, SOURCE_FIGURE_HEIGHT_INCHES),
    "pair_classes": (9.0, SOURCE_FIGURE_HEIGHT_INCHES),
    "thresholds": (7.5, SOURCE_FIGURE_HEIGHT_INCHES),
}
FONT_SIZES_POINTS = {"primary": 20, "secondary": 18, "dense": 16}
SIGNIFICANT_COLOR = "#0072B2"
MEAN_LINE_COLOR = "#D55E00"
RANDOM_LINE_COLOR = "#8C8C8C"
ACTIVITY_CLIFF_PAIR_COLOR = "#D55E00"
ACTIVITY_CLIFF_NULL_PERMUTATIONS = 2000
ACTIVITY_CLIFF_NULL_SEED = 20260730
ACTIVITY_CLIFF_NULL_INTERVAL = (0.025, 0.975)
ACTIVITY_CLIFF_SENSITIVITY_GRID_SIZE = 31
ACTIVITY_PAIR_STATE_ORDER = (
    "neither_active",
    "one_active_one_not_detected",
    "both_active",
)
ACTIVITY_PAIR_STATE_COLORS = {
    "neither_active": "#56B4E9",
    "one_active_one_not_detected": ACTIVITY_CLIFF_PAIR_COLOR,
    "both_active": "#0072B2",
}
QUADRANT_ORDER = ("cliffs", "reverse_cliffs", "concordant", "intermediate", "neutral")
QUADRANT_LABELS = {
    "cliffs": "Phenotypic cliffs\n(struct hi, morph lo)",
    "reverse_cliffs": "Reverse cliffs\n(struct lo, morph hi)",
    "concordant": "Concordant\n(struct hi, morph hi)",
    "intermediate": "Intermediate\n(both mid)",
    "neutral": "Neutral\n(struct lo, morph lo)",
}
QUADRANT_COLORS = {
    "cliffs": "#D55E00",
    "reverse_cliffs": "#56B4E9",
    "concordant": "#009E73",
    "intermediate": "#B8B8B8",
    "neutral": "#8C8C8C",
}


def _datasette_metadata_yaml(exact_cutoff, activity_fdr_threshold):
    activity_fdr_pct = 100 * activity_fdr_threshold
    return f"""title: JUMP structure-activity supplementary data
description: >-
  Auditable data underlying the candidate phenotypic activity-cliff analysis
  in the structure-activity supplementary figure.
databases:
  data:
    source: figure_suppl_structure
    source_url: https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore/processed/figures/figure_suppl_structure/summary.json
    tables:
      activity_cliff_scaffold_groups:
        description: >-
          One row per Bemis-Murcko scaffold represented among candidate on/off
          phenotypic activity-cliff pairs. Candidate pairs are unordered compound
          pairs with exact Morgan ECFP4 Tanimoto similarity of at least
          {exact_cutoff:g}, exactly one compound with detected phenotypic activity,
          and one assessed compound for which activity was not detected. Activity
          was detected at {activity_fdr_pct:g}% FDR using Benjamini-Hochberg
          corrected compound-level p-values. Scaffold identity is a post hoc
          annotation and was not used to select pairs. Morphological similarity
          was not used for pair classification. Counts are descriptive; a compound
          can contribute to multiple pairs.
        columns:
          scaffold: >-
            Post hoc Bemis-Murcko scaffold represented as a scaffold SMILES string.
            Scaffold membership did not determine pair eligibility.
          n_on_off_activity_cliff_pairs: >-
            Number of unordered active versus not-detected close pairs assigned to
            this scaffold. Each pair contributes once.
          n_unique_compounds: >-
            Number of distinct JCP2022 compounds participating in at least one of
            the scaffold's candidate pairs.
          n_active_compounds: >-
            Number of distinct participating compounds with phenotypic activity
            detected at {activity_fdr_pct:g}% FDR using Benjamini-Hochberg.
          n_not_detected_compounds: >-
            Number of distinct participating compounds that were assessed but did
            not meet the {activity_fdr_pct:g}% FDR activity criterion. Not detected
            should not be interpreted as evidence of biological inactivity.
          max_morgan_tanimoto: >-
            Maximum exact Morgan ECFP4 Tanimoto similarity among the scaffold's
            candidate pairs.
          median_morgan_tanimoto: >-
            Median exact Morgan ECFP4 Tanimoto similarity among the scaffold's
            candidate pairs.
          representative_jcp_i: >-
            Lexically first JCP2022 identifier in the highest-Tanimoto
            representative unordered pair. The i suffix does not encode activity
            status.
          representative_jcp_j: >-
            Lexically second JCP2022 identifier in the highest-Tanimoto
            representative unordered pair. The j suffix does not encode activity
            status.
          jcp_ids: >-
            Pipe-delimited, lexically sorted list of all distinct JCP2022
            identifiers participating in the scaffold group.
"""


# Views are named by scientific content; manuscript panel letters are applied externally.
FIGURE_ID = "figure_suppl_structure"
VIEW_ORDER = (
    "similarity_map",
    "similar_structure_morphology",
    "scaffold_coherence",
    "activity_cliff_pairs",
    "neighbor_overlap",
    "moa_coherence",
    "pair_classes",
    "thresholds",
)
# One legend section per rendered file, titled here and bodied by the caption the view carries.
SECTION_TITLES = {
    "similarity_map": "Structure-morphology relationship",
    "similar_structure_morphology": "Morphological similarity of structurally similar pairs",
    "scaffold_coherence": "Within-scaffold morphological coherence",
    "activity_cliff_pairs": "Phenotypic activity concordance and candidate cliffs",
    "neighbor_overlap": "Structure and morphology neighbor overlap",
    "moa_coherence": "Mechanism-of-action structure-morphology coherence",
    "pair_classes": "Structural k-nearest-neighbor pair classes",
    "thresholds": "Morphology-cliff threshold sensitivity",
}


def _conditional_similarity_statistics(all_pair_histogram, numbers):
    """Summarize the all-pairs histogram, ahead of drawing it, so the caption can quote it."""
    expected_pairs = numbers["all_possible_active_pairs"]
    if int(all_pair_histogram["n_pairs"].sum()) != expected_pairs:
        raise ValueError("All-pairs similarity histogram has the wrong denominator")

    morphology_centers = (all_pair_histogram["morphology_bin_left"] + all_pair_histogram["morphology_bin_right"]) / 2
    histogram = all_pair_histogram.assign(morphology_center=morphology_centers)

    def histogram_quantile(group, probability):
        ordered = group.sort_values("morphology_center")
        target = probability * ordered["n_pairs"].sum()
        index = np.searchsorted(ordered["n_pairs"].cumsum().to_numpy(), target, side="left")
        return float(ordered["morphology_center"].iloc[index])

    rows = []
    for (left, right), group in histogram.groupby(["structure_bin_left", "structure_bin_right"], sort=True):
        rows.append(
            {
                "structure_bin_left": float(left),
                "structure_bin_right": float(right),
                "structure_bin_center": float((left + right) / 2),
                "n_pairs": int(group["n_pairs"].sum()),
                "morphology_q25": histogram_quantile(group, 0.25),
                "morphology_median": histogram_quantile(group, 0.50),
                "morphology_q75": histogram_quantile(group, 0.75),
            }
        )
    summary = pd.DataFrame(rows)

    all_morphology_similar = int(histogram.loc[histogram["morphology_bin_left"] >= MORPH_LOW, "n_pairs"].sum())
    close = histogram[histogram["structure_bin_left"] >= STRUCTURAL_PROXY_HIGH_CUTOFF]
    close_pairs = int(close["n_pairs"].sum())
    close_morphology_similar = int(close.loc[close["morphology_bin_left"] >= MORPH_LOW, "n_pairs"].sum())
    overall_fraction = all_morphology_similar / expected_pairs
    close_fraction = close_morphology_similar / close_pairs
    enrichment = close_fraction / overall_fraction
    statistics = {
        "all_pair_count": int(expected_pairs),
        "structurally_similar_pair_count": close_pairs,
        "overall_morphology_similar_fraction": float(overall_fraction),
        "close_pair_morphology_similar_fraction": float(close_fraction),
        "morphology_similarity_enrichment": float(enrichment),
    }
    return summary, statistics


def _conditional_similarity(summary, statistics, caption, output_dir, dpi):
    typography = {
        "font.size": 7.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
    }
    with plt.rc_context(typography):
        figure, (axis, count_axis) = plt.subplots(
            2,
            1,
            figsize=FIGURE_SIZES_INCHES["similarity_map"],
            sharex=True,
            gridspec_kw={"height_ratios": (3.5, 1.0), "hspace": 0.08},
        )
        axis.fill_between(
            summary["structure_bin_center"],
            summary["morphology_q25"],
            summary["morphology_q75"],
            color="#9ECAE1",
            alpha=0.55,
            linewidth=0,
            label="Interquartile range",
        )
        axis.plot(
            summary["structure_bin_center"],
            summary["morphology_median"],
            color="#08519C",
            marker="o",
            linewidth=1.1,
            markersize=2.5,
            label="Median",
        )
        axis.axhline(0, color="#BDBDBD", linestyle=":", linewidth=0.6)
        axis.axhline(MORPH_LOW, color="#636363", linestyle=":", linewidth=0.7)
        axis.axvline(
            STRUCTURAL_PROXY_HIGH_CUTOFF,
            color="#636363",
            linestyle=":",
            linewidth=0.7,
        )
        axis.set_ylabel("Morphology cosine")
        axis.set_title("Morphology by structure similarity")
        axis.legend(
            frameon=False,
            loc="upper left",
            handlelength=1.7,
            borderpad=0.1,
            labelspacing=0.25,
        )
        # Under the legend, in the empty region above the band: at the lower right it covered the band.
        axis.text(
            0.035,
            0.77,
            (
                f"Morphology >= {MORPH_LOW:.2f}\n"
                f"{statistics['close_pair_morphology_similar_fraction']:.1%} at structure >= "
                f"{STRUCTURAL_PROXY_HIGH_CUTOFF:.2f}\n"
                f"{statistics['overall_morphology_similar_fraction']:.1%} overall "
                f"({statistics['morphology_similarity_enrichment']:.2f}x)"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            linespacing=1.15,
        )

        bin_width = float(summary["structure_bin_right"].iloc[0] - summary["structure_bin_left"].iloc[0])
        count_axis.bar(
            summary["structure_bin_center"],
            summary["n_pairs"],
            width=bin_width * 0.88,
            color="#BDBDBD",
        )
        count_axis.set_yscale("log")
        count_axis.set_ylabel("Pairs")
        count_axis.set_xlabel("Structure similarity")
        count_axis.set_xlim(0, 1)
        count_axis.text(
            0.99,
            0.92,
            f"{summary['n_pairs'].sum() / 1_000_000:.1f}M pairs",
            transform=count_axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
        )
        # right=0.98 cut the last tick label ("1.0") at the page edge.
        figure.subplots_adjust(left=0.19, right=0.95, bottom=0.15, top=0.92)
        attach_caption(figure, caption, caption_points=5.5)
        paths = save_figure(figure, output_dir, "similarity_map", dpi * 2)
    return paths


def _distribution(pairs, caption, output_dir, dpi):
    similar = pairs.loc[
        pairs["struct_sim"] >= STRUCTURAL_PROXY_HIGH_CUTOFF,
        "morph_sim",
    ]
    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["similar_structure_morphology"])
    axis.hist(similar, bins=HISTOGRAM_BINS, color=SIGNIFICANT_COLOR, edgecolor="white", linewidth=0.3)
    axis.axvline(MORPH_LOW, color=RANDOM_LINE_COLOR, linestyle=":", linewidth=1.2)
    axis.axvline(float(similar.mean()), color=MEAN_LINE_COLOR, linestyle="--", linewidth=1.5)
    axis.set_xlabel("Morphological similarity (cosine)")
    axis.set_ylabel("Pairs above structure cutoff")
    axis.text(
        *DISTRIBUTION_ANNOTATION_POSITION,
        f"Structure >= {STRUCTURAL_PROXY_HIGH_CUTOFF:g}\nn = {len(similar):,}\nmean = {similar.mean():.3f}",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_POINTS["primary"],
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    figure.tight_layout()
    attach_caption(figure, caption, caption_points=5.5)
    return save_figure(figure, output_dir, "similar_structure_morphology", dpi)


def _overlap(overlap, numbers, caption, output_dir, dpi):
    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["neighbor_overlap"])
    axis.hist(
        overlap["neighbor_overlap"],
        bins=HISTOGRAM_BINS,
        color=SIGNIFICANT_COLOR,
        edgecolor="white",
        linewidth=0.3,
    )
    axis.axvline(
        numbers["mean_jaccard_overlap"],
        color=MEAN_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {numbers['mean_jaccard_overlap']:.3f}",
    )
    axis.axvline(
        numbers["expected_random_overlap"],
        color=RANDOM_LINE_COLOR,
        linestyle=":",
        linewidth=1.5,
        label=f"Random = {numbers['expected_random_overlap']:.4f}",
    )
    axis.set_xlabel("Jaccard overlap (k = 50)")
    axis.set_ylabel("Active compounds")
    axis.text(
        0.97,
        0.75,
        f"n = {numbers['active_compounds_analyzed']:,}\n"
        f"{numbers['ratio_vs_random']:.1f}x random\n"
        f"non-zero: {numbers['pct_nonzero_overlap']:.1f}%",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_POINTS["primary"],
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=FONT_SIZES_POINTS["secondary"],
        frameon=False,
    )
    figure.tight_layout()
    attach_caption(figure, caption, caption_points=5.5)
    return save_figure(figure, output_dir, "neighbor_overlap", dpi)


def _moa_coherence(moa, caption, output_dir, dpi):
    moa = moa.sort_values("enrichment", ascending=True)
    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["moa_coherence"])
    y = np.arange(len(moa))
    count_span = max(moa["count"].max() - moa["count"].min(), 1)
    sizes = MOA_MARKER_AREA_RANGE[0] + (
        (moa["count"] - moa["count"].min()) / count_span * (MOA_MARKER_AREA_RANGE[1] - MOA_MARKER_AREA_RANGE[0])
    )
    axis.hlines(y, 0, moa["enrichment"], color="#CBD5E1", linewidth=1.2, zorder=1)
    axis.scatter(moa["enrichment"], y, s=sizes, color=SIGNIFICANT_COLOR, zorder=2)
    label_offset = moa["enrichment"].max() * MOA_LABEL_OFFSET_FRACTION
    for yi, (enrichment, count) in enumerate(zip(moa["enrichment"], moa["count"], strict=True)):
        axis.text(
            enrichment + label_offset,
            yi,
            f"n={count}",
            va="center",
            fontsize=FONT_SIZES_POINTS["secondary"],
            color="#555555",
        )
    axis.set_yticks(y)
    axis.set_yticklabels(moa["moa"], fontsize=FONT_SIZES_POINTS["primary"])
    axis.set_xlabel("Neighbor overlap (x random)")
    axis.set_xlim(0, moa["enrichment"].max() * MOA_X_PADDING_MULTIPLIER)
    figure.tight_layout()
    attach_caption(figure, caption, caption_points=5.5)
    return save_figure(figure, output_dir, "moa_coherence", dpi)


def _quadrants(numbers, caption, output_dir, dpi):
    quadrants = numbers["quadrants"]
    total = numbers["n_pairs"]
    table = pd.DataFrame(
        {
            "category": [QUADRANT_LABELS[key].replace("\n", " ") for key in QUADRANT_ORDER],
            "count": [quadrants[key] for key in QUADRANT_ORDER],
            "percent": [100 * quadrants[key] / total for key in QUADRANT_ORDER],
        }
    )

    def _compact_pair_count(value):
        return f"{value / 1000:.0f}k" if value >= 100_000 else f"{value:,}"

    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["pair_classes"])
    y = np.arange(len(QUADRANT_ORDER))
    axis.barh(
        y,
        [quadrants[key] for key in QUADRANT_ORDER],
        color=[QUADRANT_COLORS[key] for key in QUADRANT_ORDER],
        edgecolor="white",
        linewidth=0.5,
    )
    axis.set_xscale("log")
    axis.set_yticks(y)
    axis.set_yticklabels(
        [QUADRANT_LABELS[key] for key in QUADRANT_ORDER],
        fontsize=FONT_SIZES_POINTS["secondary"],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Number of pairs (log scale)")
    for yi, key in enumerate(QUADRANT_ORDER):
        axis.text(
            quadrants[key] * 1.15,
            yi,
            f"{_compact_pair_count(quadrants[key])} ({100 * quadrants[key] / total:.1f}%)",
            va="center",
            fontsize=FONT_SIZES_POINTS["secondary"],
        )
    axis.set_xlim(right=total * QUADRANT_X_RIGHT_MULTIPLIER)
    figure.tight_layout()
    attach_caption(figure, caption, caption_points=5.5)
    paths = save_figure(figure, output_dir, "pair_classes", dpi)
    table.to_csv(output_dir / "similarity_quadrants.csv", index=False)
    return paths


def _threshold_sensitivity(pairs, caption, output_dir, dpi):
    # Cliff count as a function of both cutoffs. The 5x5 grid is retained for the CSV, but the
    # panel draws cumulative cliff-count curves (one per structural cutoff): each curve is the
    # cumulative morphology-similarity distribution within that structural stratum, so the chosen
    # definition reads as one point on a smooth ramp rather than a magic value.
    struct_grid = list(STRUCTURAL_PROXY_THRESHOLD_GRID)
    morph_grid = list(MORPH_THRESHOLD_GRID)
    struct_sim = pairs["struct_sim"].to_numpy()
    morph_sim = pairs["morph_sim"].to_numpy()

    # Discrete grid retained for the machine-readable CSV download.
    counts = np.array([[int(((struct_sim >= s) & (morph_sim < m)).sum()) for m in morph_grid] for s in struct_grid])
    grid_df = pd.DataFrame(
        counts,
        index=[f"morgan_cosine_proxy_ge_{s:g}" for s in struct_grid],
        columns=[f"cosine_lt_{m:g}" for m in morph_grid],
    )
    grid_df.to_csv(output_dir / "threshold_sensitivity.csv")

    # Continuous cliff-count curves via the sorted per-stratum morphology CDF.
    morph_cutoffs = np.linspace(float(morph_sim.min()), 1.0, 400)
    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["thresholds"])
    curve_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(struct_grid)))
    for s, color in zip(struct_grid, curve_colors, strict=True):
        stratum_sorted = np.sort(morph_sim[struct_sim >= s])
        cumulative = np.searchsorted(stratum_sorted, morph_cutoffs, side="left")
        axis.plot(
            morph_cutoffs,
            np.maximum(cumulative, 0.5),
            color=color,
            linewidth=2.2,
            label=f"structure >= {s:g}",
        )
    chosen_sorted = np.sort(morph_sim[struct_sim >= STRUCTURAL_PROXY_HIGH_CUTOFF])
    chosen_count = int(np.searchsorted(chosen_sorted, MORPH_LOW, side="left"))
    axis.axvline(MORPH_LOW, color=RANDOM_LINE_COLOR, linestyle=":", linewidth=1.2)
    axis.plot(
        MORPH_LOW,
        chosen_count,
        marker="o",
        markersize=12,
        markerfacecolor="none",
        markeredgecolor=MEAN_LINE_COLOR,
        markeredgewidth=2.5,
        zorder=5,
    )
    axis.annotate(
        f"chosen\n{chosen_count:,} cliffs",
        (MORPH_LOW, chosen_count),
        # Upper left is above every curve; the old position sat on two of them.
        xytext=(-0.82, 2.0e4),
        fontsize=FONT_SIZES_POINTS["secondary"],
        color=MEAN_LINE_COLOR,
        arrowprops=dict(arrowstyle="->", color=MEAN_LINE_COLOR, linewidth=1.5),
    )
    axis.set_yscale("log")
    axis.set_xlabel("Morphology cutoff (cliff = cosine < x)")
    axis.set_ylabel("Cliff pairs (cumulative, log)")
    axis.set_xticks([-0.5, 0.0, 0.5, 1.0])
    # Dedicated space above the axes: every in-axes corner is crossed by a curve or the marker.
    # Lower right is below every curve; the upper left corner, used before, is crossed by four.
    axis.legend(loc="lower right", fontsize=FONT_SIZES_POINTS["dense"], frameon=False, labelspacing=0.25)
    figure.tight_layout()
    attach_caption(figure, caption, caption_points=5.5)
    return save_figure(figure, output_dir, "thresholds", dpi)


def _scaffold_coherence(scaffold, numbers, caption, output_dir, dpi):
    # Per-scaffold within-group morphological similarity (observed) vs a permutation null,
    # from notebook 2.40. Scaffold-level significance is controlled at 5% FDR (BH).
    figure, axis = plt.subplots(figsize=FIGURE_SIZES_INCHES["scaffold_coherence"])
    both = np.concatenate([scaffold["mean_cosine_sim"].to_numpy(), scaffold["null_mean"].to_numpy()])
    bins = np.linspace(float(both.min()), float(both.max()), HISTOGRAM_BINS)
    axis.hist(
        scaffold["mean_cosine_sim"],
        bins=bins,
        color=SIGNIFICANT_COLOR,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        label="Observed (within scaffold)",
    )
    axis.hist(
        scaffold["null_mean"],
        bins=bins,
        color=RANDOM_LINE_COLOR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
        label="Null (random groups)",
    )
    axis.set_xlabel("Within-group morphological similarity (cosine)")
    axis.set_ylabel("Bemis-Murcko scaffolds")
    axis.text(
        *SCAFFOLD_ANNOTATION_POSITION,
        f"{numbers['scaffold_n_tested']:,} scaffolds "
        f"(>= {numbers['scaffold_min_compounds']} active)\n"
        f"{numbers['scaffold_pct_significant_fdr005']:.1f}% significant at "
        f"{SCAFFOLD_FDR_THRESHOLD * 100:.0f}% FDR (BH)\n"
        f"{numbers['scaffold_n_clusters_enriched']}/"
        f"{numbers['scaffold_n_kmeans_clusters']} clusters enriched at 5% FDR",
        transform=axis.transAxes,
        fontsize=FONT_SIZES_POINTS["secondary"],
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    axis.legend(loc="upper right", fontsize=FONT_SIZES_POINTS["secondary"], frameon=False)
    figure.tight_layout()
    scaffold.to_csv(output_dir / "scaffold_coherence.csv", index=False)
    attach_caption(figure, caption, caption_points=5.5)
    return save_figure(figure, output_dir, "scaffold_coherence", dpi)


def _activity_cliff_threshold_sensitivity(pairs, cutoff):
    threshold_grid = np.round(
        np.linspace(cutoff, 1.0, ACTIVITY_CLIFF_SENSITIVITY_GRID_SIZE),
        2,
    )
    assessed_pairs = pairs[pairs["activity_pair_state"].isin(ACTIVITY_PAIR_STATE_ORDER)].copy()
    status_by_compound = pd.concat(
        [
            assessed_pairs[["jcp_i", "activity_status_i"]].rename(
                columns={"jcp_i": "compound", "activity_status_i": "status"}
            ),
            assessed_pairs[["jcp_j", "activity_status_j"]].rename(
                columns={"jcp_j": "compound", "activity_status_j": "status"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    status_conflicts = status_by_compound.groupby("compound")["status"].nunique()
    if (status_conflicts > 1).any():
        raise ValueError("A compound has inconsistent activity calls across close pairs")
    status_by_compound = status_by_compound.drop_duplicates("compound").set_index("compound")["status"].sort_index()

    rng = np.random.default_rng(ACTIVITY_CLIFF_NULL_SEED)
    rows = []
    for threshold in threshold_grid:
        edges = assessed_pairs[assessed_pairs["morgan_tanimoto"] >= threshold]
        if edges.empty:
            raise ValueError(f"No assessed close pairs remain at Tanimoto {threshold:g}")
        nodes = np.unique(np.concatenate([edges["jcp_i"].to_numpy(), edges["jcp_j"].to_numpy()]))
        node_index = pd.Series(np.arange(len(nodes)), index=nodes)
        edge_i = node_index.loc[edges["jcp_i"]].to_numpy()
        edge_j = node_index.loc[edges["jcp_j"]].to_numpy()
        active_labels = status_by_compound.loc[nodes].eq("active").to_numpy()
        observed_fraction = float(np.logical_xor(active_labels[edge_i], active_labels[edge_j]).mean())
        null_fractions = np.empty(ACTIVITY_CLIFF_NULL_PERMUTATIONS)
        for permutation_index in range(ACTIVITY_CLIFF_NULL_PERMUTATIONS):
            shuffled_labels = rng.permutation(active_labels)
            null_fractions[permutation_index] = np.logical_xor(shuffled_labels[edge_i], shuffled_labels[edge_j]).mean()
        n_active = int(active_labels.sum())
        n_compounds = len(active_labels)
        null_expected_fraction = float(2 * n_active * (n_compounds - n_active) / (n_compounds * (n_compounds - 1)))
        null_interval_low, null_interval_high = np.quantile(
            null_fractions,
            ACTIVITY_CLIFF_NULL_INTERVAL,
        )
        rows.append(
            {
                "minimum_exact_morgan_tanimoto": float(threshold),
                "assessed_pair_count": int(len(edges)),
                "participating_compound_count": int(n_compounds),
                "active_compound_count": n_active,
                "observed_candidate_cliff_fraction": observed_fraction,
                "null_expected_fraction": null_expected_fraction,
                "null_pointwise_2_5_fraction": float(null_interval_low),
                "null_pointwise_97_5_fraction": float(null_interval_high),
            }
        )
    sensitivity = pd.DataFrame(rows)
    sensitivity["observed_concordant_fraction"] = 1 - sensitivity["observed_candidate_cliff_fraction"]
    sensitivity["null_expected_concordant_fraction"] = 1 - sensitivity["null_expected_fraction"]
    sensitivity["null_pointwise_2_5_concordant_fraction"] = 1 - sensitivity["null_pointwise_97_5_fraction"]
    sensitivity["null_pointwise_97_5_concordant_fraction"] = 1 - sensitivity["null_pointwise_2_5_fraction"]
    return sensitivity


def _activity_cliff_pairs(pairs, pair_summary, sensitivity, numbers, configuration, caption, output_dir, dpi):
    cutoff = float(configuration["activity_cliff_pair_structural_threshold"])
    assessed_summary = (
        pair_summary[pair_summary["activity_pair_state"].isin(ACTIVITY_PAIR_STATE_ORDER)]
        .set_index("activity_pair_state")
        .reindex(ACTIVITY_PAIR_STATE_ORDER)
    )
    if assessed_summary["n_pairs"].isna().any():
        raise ValueError("Activity-pair summary is missing an assessed activity state")
    assessed_total = numbers["activity_cliff_exact_close_pairs_assessed"]
    if int(assessed_summary["n_pairs"].sum()) != assessed_total:
        raise ValueError("Rendered activity states do not partition the assessed close pairs")
    on_off_pairs = pairs[pairs["is_on_off_activity_cliff"].astype(bool)]
    if len(on_off_pairs) != numbers["activity_cliff_on_off_pairs"]:
        raise ValueError("Rendered on/off-pair count differs from the manifest")

    typography = {
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.0,
    }
    with plt.rc_context(typography):
        figure, (state_axis, sensitivity_axis) = plt.subplots(
            2,
            1,
            figsize=FIGURE_SIZES_INCHES["activity_cliff_pairs"],
            gridspec_kw={"height_ratios": (1.0, 1.55), "hspace": 0.68},
        )
        figure.suptitle(
            f"Phenotypic activity among close structural pairs\n"
            f"(Tanimoto >= {cutoff:.2f}; {assessed_total:,} assessed pairs)",
            fontsize=8.0,
            y=0.985,
        )
        state_counts = assessed_summary["n_pairs"].to_numpy(dtype=int)
        percentages = 100 * state_counts / assessed_total
        state_y = np.arange(len(ACTIVITY_PAIR_STATE_ORDER))
        bars = state_axis.barh(
            state_y,
            percentages,
            color=[ACTIVITY_PAIR_STATE_COLORS[state] for state in ACTIVITY_PAIR_STATE_ORDER],
            edgecolor="none",
            height=0.62,
        )
        state_axis.set_yticks(state_y, ["Neither active", "One active", "Both active"])
        state_axis.invert_yaxis()
        state_axis.set_xlim(0, 102)
        state_axis.set_xlabel("Assessed pairs (%)")
        state_axis.spines[["top", "right"]].set_visible(False)
        for bar, count, percentage in zip(bars, state_counts, percentages, strict=True):
            state_axis.text(
                percentage + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({percentage:.1f}%)",
                va="center",
                fontsize=6.0,
            )

        threshold_grid = sensitivity["minimum_exact_morgan_tanimoto"].to_numpy()
        observed_pct = 100 * sensitivity["observed_concordant_fraction"].to_numpy()
        null_expected_pct = 100 * sensitivity["null_expected_concordant_fraction"].to_numpy()
        null_low_pct = 100 * sensitivity["null_pointwise_2_5_concordant_fraction"].to_numpy()
        null_high_pct = 100 * sensitivity["null_pointwise_97_5_concordant_fraction"].to_numpy()
        sensitivity_axis.fill_between(
            threshold_grid,
            null_low_pct,
            null_high_pct,
            color=RANDOM_LINE_COLOR,
            alpha=0.20,
            linewidth=0,
        )
        sensitivity_axis.plot(
            threshold_grid,
            observed_pct,
            color=SIGNIFICANT_COLOR,
            linewidth=1.1,
            marker="o",
            markevery=5,
            markersize=2.4,
            label="Observed",
        )
        sensitivity_axis.plot(
            threshold_grid,
            null_expected_pct,
            color=RANDOM_LINE_COLOR,
            linewidth=1.0,
            linestyle="--",
            label="Shuffled labels",
        )
        tick_rows = sensitivity.iloc[::10]

        def compact_pair_count(value):
            if value >= 10_000:
                return f"{value / 1_000:.0f}k"
            if value >= 1_000:
                return f"{value / 1_000:.1f}k"
            return f"{value:,}"

        sensitivity_axis.set_xticks(
            tick_rows["minimum_exact_morgan_tanimoto"],
            [
                f"{threshold:.2f}\nn={compact_pair_count(pair_count)}"
                for threshold, pair_count in zip(
                    tick_rows["minimum_exact_morgan_tanimoto"],
                    tick_rows["assessed_pair_count"],
                    strict=True,
                )
            ],
        )
        sensitivity_axis.set_xlim(cutoff - 0.01, 1.01)
        sensitivity_axis.set_ylim(60, 90)
        sensitivity_axis.set_xlabel("Minimum Tanimoto")
        sensitivity_axis.set_ylabel("Concordant pairs (%)")
        sensitivity_axis.set_title("Concordance across thresholds")
        sensitivity_axis.grid(axis="y", color="#E5E7EB", linewidth=0.5, zorder=0)
        sensitivity_axis.spines[["top", "right"]].set_visible(False)
        sensitivity_axis.legend(
            frameon=False,
            loc="upper left",
            ncols=2,
            handlelength=1.6,
            columnspacing=0.9,
            borderpad=0.1,
        )
        sensitivity_axis.text(
            0.705,
            observed_pct[0] + 1.0,
            f"{observed_pct[0]:.1f}%",
            color=SIGNIFICANT_COLOR,
            fontsize=6.0,
            fontweight="bold",
        )
        sensitivity_axis.text(
            0.705,
            null_expected_pct[0] - 4.6,
            f"{null_expected_pct[0]:.1f}%",
            color=RANDOM_LINE_COLOR,
            fontsize=6.0,
            fontweight="bold",
        )
        figure.subplots_adjust(left=0.28, right=0.98, bottom=0.16, top=0.86)
        attach_caption(figure, caption, caption_points=5.5)
        return save_figure(figure, output_dir, "activity_cliff_pairs", dpi * 2)


OUTPUT_CAPTIONS = {
    "similarity_map.png": "Structure-morphology relationship",
    "similarity_map.pdf": "Structure-morphology relationship",
    "similar_structure_morphology.png": ("Morphological similarity of structurally similar active pairs"),
    "similar_structure_morphology.pdf": ("Morphological similarity of structurally similar active pairs"),
    "scaffold_coherence.png": "Within-scaffold morphological coherence",
    "scaffold_coherence.pdf": "Within-scaffold morphological coherence",
    "activity_cliff_pairs.png": ("Phenotypic activity concordance among exact-Tanimoto-close pairs"),
    "activity_cliff_pairs.pdf": ("Phenotypic activity concordance among exact-Tanimoto-close pairs"),
    "neighbor_overlap.png": "Structure and morphology neighbor overlap",
    "neighbor_overlap.pdf": "Structure and morphology neighbor overlap",
    "moa_coherence.png": "Mechanism-of-action structure-morphology coherence",
    "moa_coherence.pdf": "Mechanism-of-action structure-morphology coherence",
    "pair_classes.png": "Structural k-nearest-neighbor pair classes",
    "pair_classes.pdf": "Structural k-nearest-neighbor pair classes",
    "thresholds.png": "Morphology-cliff threshold sensitivity",
    "thresholds.pdf": "Morphology-cliff threshold sensitivity",
    "moa_coherence.csv": "Mechanism-of-action neighbor-overlap summary",
    "similarity_quadrants.csv": "Structural k-nearest-neighbor pair-class counts",
    "threshold_sensitivity.csv": "Morphology-cliff cutoff grid",
    "scaffold_coherence.csv": "Canonical active-only notebook 2.40 scaffold result",
    "phenotypic_cliffs.csv": "Itemized active-pair morphology cliffs",
    "activity_cliff_pairs.parquet": ("Global exact-Tanimoto pairs with phenotypic activity states"),
    "activity_cliff_pair_summary.csv": "Exact-pair activity-state counts",
    "activity_cliff_threshold_sensitivity.csv": (
        "Activity-call concordance and permutation null across Tanimoto thresholds"
    ),
    "activity_cliff_scaffold_groups.csv": ("Post hoc same-scaffold on-off activity-cliff groups"),
    "phenoseeker_procedure_transfer_audit.csv": ("Canonical notebook 2.32 scaffold-series procedure-transfer audit"),
    "activity_cliff_scaffold_groups__metadata.yml": ("Column annotations for post hoc same-scaffold groups"),
    "legend.md": "Figure legend",
}


def build_captions(
    *,
    numbers: dict,
    parameters: dict,
    pairs: pd.DataFrame,
    moa: pd.DataFrame,
    all_pair_statistics: dict,
    activity_sensitivity: pd.DataFrame,
) -> dict:
    """Author one caption per rendered view: drawn under it and used as its legend section."""
    mean_morph_similar = float(
        pairs.loc[
            pairs["struct_sim"] >= STRUCTURAL_PROXY_HIGH_CUTOFF,
            "morph_sim",
        ].mean()
    )
    top_moa_names = ", ".join(
        moa.sort_values(
            ["enrichment", "moa"],
            ascending=[False, True],
            kind="mergesort",
        )["moa"]
        .head(5)
        .tolist()
    )
    chosen = activity_sensitivity.iloc[0]
    concordance_pct = float(100 * chosen["observed_concordant_fraction"])
    discordance_pct = float(100 * chosen["observed_candidate_cliff_fraction"])
    null_concordance_pct = float(100 * chosen["null_expected_concordant_fraction"])
    k_pairs = int(parameters["k_pairs"])

    return {
        "similarity_map": (
            "Morphological similarity increased "
            f"with structural similarity across all {all_pair_statistics['all_pair_count']:,} "
            f"unordered pairs among {numbers['active_compounds_analyzed']:,} "
            "phenotypically active compounds. The line and shading show the median and "
            "interquartile range of Cell Painting profile cosine similarity within bins "
            "of Morgan ECFP4 cosine similarity; bars show pair counts. Pairs with "
            "structural similarity at least 0.70 were "
            f"{all_pair_statistics['morphology_similarity_enrichment']:.2f}-fold enriched "
            "for morphology similarity at least 0.30 "
            f"({all_pair_statistics['close_pair_morphology_similar_fraction']:.1%} versus "
            f"{all_pair_statistics['overall_morphology_similar_fraction']:.1%} overall). "
            "A single rho is omitted because the relationship is nonlinear and dominated "
            "by low-similarity pairs."
        ),
        "similar_structure_morphology": (
            "Distribution of "
            "morphological cosine similarity for the "
            f"{numbers['n_structurally_similar_pairs']:,} active pairs above the "
            f"Morgan-fingerprint cosine proxy cutoff of {STRUCTURAL_PROXY_HIGH_CUTOFF:g} within "
            "the symmetrized structural k-nearest-neighbor union "
            f"(mean morphology cosine {mean_morph_similar:.3f}, dashed line). Most such "
            "pairs are morphologically similar; the tail below the cosine = 0.3 "
            "threshold is the active-pair morphology-cliff set."
        ),
        "scaffold_coherence": (
            "For each of the "
            f"{numbers['scaffold_n_tested']:,} Bemis-Murcko scaffolds represented by at "
            f"least {numbers['scaffold_min_compounds']} active compounds, the mean "
            "pairwise morphological similarity among its members is compared with a "
            f"size-matched permutation null. {numbers['scaffold_pct_significant_fdr005']:.1f}% "
            "of scaffolds show greater self-similarity at 5% FDR using "
            "Benjamini-Hochberg correction. Independently, "
            f"{numbers['scaffold_n_clusters_enriched']} of "
            f"{numbers['scaffold_n_kmeans_clusters']} morphology-based KMeans clusters "
            "contain at least one over-represented Murcko scaffold at 5% FDR. This "
            "scaffold-level coherence does not rely on the pair-similarity cutoffs and "
            "is the stronger structure-morphology signal in this figure."
        ),
        "activity_cliff_pairs": (
            "Among "
            f"{numbers['activity_cliff_exact_close_pairs_assessed']:,} assessed pairs "
            "with exact Morgan Tanimoto at least 0.70, "
            f"{concordance_pct:.1f}% had concordant activity calls and "
            f"{numbers['activity_cliff_on_off_pairs']:,} ({discordance_pct:.1f}%) had "
            "exactly one compound detected as active. Activity-call concordance remained "
            "above the random-label expectation across Tanimoto thresholds; at the chosen "
            f"cutoff, the null expectation was {null_concordance_pct:.1f}% concordance. "
            "Activity was detected at 10% FDR using Benjamini-Hochberg correction."
        ),
        "neighbor_overlap": (
            "For each of the "
            f"{numbers['active_compounds_analyzed']:,} active compounds, the Jaccard "
            "overlap between its 50 nearest structural neighbors and its 50 nearest "
            "morphological neighbors is small in absolute terms: mean "
            f"{numbers['mean_jaccard_overlap']:.3f}, "
            f"median {numbers['median_jaccard_overlap']:.0f}, and "
            f"{numbers['pct_nonzero_overlap']:.1f}% nonzero overlap, versus a chance "
            f"expectation of {numbers['expected_random_overlap']:.4f}. This overlap "
            "enrichment is a weak signal and is not the basis of the figure's main claim."
        ),
        "moa_coherence": (
            "Drug Repurposing "
            "Hub mechanism-of-action classes with at least 10 active members are ranked "
            f"by the neighbor-overlap metric. The leading classes are {top_moa_names}. "
            "Because the underlying overlaps are small in absolute terms, the class "
            "identities rather than the enrichment multiples are the informative result. "
            "Dot area encodes the number of active compounds in each class."
        ),
        "pair_classes": (
            f"The {numbers['n_pairs']:,} unique unordered pairs "
            "in the union of all per-compound structural k-nearest-neighbor lists "
            f"(k = {k_pairs}) among active compounds are classified using high or low "
            "Morgan-fingerprint cosine and morphology cosine. The pair universe is the "
            "symmetrized directed-neighbor union, with every unordered pair represented "
            "once."
        ),
        "thresholds": (
            "Cumulative number of active-pair morphology "
            "cliffs as a function of morphology cosine cutoff, drawn for five structural "
            "Morgan-fingerprint cosine proxy cutoffs within the same symmetrized "
            "k-nearest-neighbor pair set. Each curve is the cumulative morphology-similarity "
            "distribution within that structural stratum. The chosen definition is structure "
            f">= {STRUCTURAL_PROXY_HIGH_CUTOFF:g} and morphology < {MORPH_LOW:g}, marking "
            f"{numbers['cliff_pairs']:,} cliff pairs. Its open marker sits on a smooth ramp, "
            "so the count is threshold-dependent and should be read as approximate."
        ),
    }


def build_figure(
    *,
    pair_cache_summary: Path,
    substructure_summary: Path,
    scaffold_coherence: Path,
    scaffold_series_summary: Path,
    scaffold_analysis: Path,
    cliff_scaffolds: Path,
    output_dir: Path,
    dpi: int,
) -> dict:
    """Validate canonical inputs and write the complete direct figure package."""
    cache_numbers, parameters, tables = load_pair_cache(pair_cache_summary)
    scaffold, substructure_numbers = load_active_substructure_analysis(
        scaffold_coherence,
        substructure_summary,
    )
    active_jcp = set(
        tables["phenotypic_activity"].loc[
            tables["phenotypic_activity"]["below_corrected_p"].eq(True),
            "jcp2022",
        ]
    )
    (
        phenoseeker_audit,
        scaffold_series_parameters,
        scaffold_series_numbers,
    ) = load_scaffold_cliff_series(
        scaffold_analysis,
        cliff_scaffolds,
        scaffold_series_summary,
        active_jcp,
    )

    pairs = tables["pairs"]
    all_pair_histogram = tables["all_pair_similarity_histogram"]
    overlap = tables["neighbor_overlap"]
    moa = tables["moa_coherence"].sort_values(
        ["enrichment", "moa"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    phenotypic_cliffs = tables["phenotypic_cliffs"]
    activity_cliff_pairs = tables["activity_cliff_pairs"]
    activity_pair_summary = tables["activity_cliff_pair_summary"]

    if not (phenotypic_cliffs["struct_sim"] >= STRUCTURAL_PROXY_HIGH_CUTOFF).all():
        raise ValueError("An active-pair morphology cliff falls below the proxy cutoff")
    if not (phenotypic_cliffs["morph_sim"] < MORPH_LOW).all():
        raise ValueError("An active-pair morphology cliff falls outside the morphology cutoff")
    exact_cutoff = float(parameters["activity_cliff_pair_structural_threshold"])
    if parameters["activity_cliff_pair_scaffold_used_for_selection"] is not False:
        raise ValueError("Scaffold membership must not select candidate on-off pairs")
    if parameters["activity_cliff_pair_morphology_similarity_used_for_pair_classification"] is not False:
        raise ValueError("Morphology must not select candidate on-off pairs")
    if len(activity_cliff_pairs) != cache_numbers["activity_cliff_exact_close_pairs"]:
        raise ValueError("Exact structural-neighbor table differs from the summary")
    if activity_cliff_pairs.duplicated(["jcp_i", "jcp_j"]).any():
        raise ValueError("Exact structural-neighbor table contains duplicate unordered pairs")
    if not (activity_cliff_pairs["jcp_i"] < activity_cliff_pairs["jcp_j"]).all():
        raise ValueError("Exact structural-neighbor pairs are not canonically ordered")
    exact_threshold = 17 * activity_cliff_pairs["fingerprint_intersection_bits"] >= 7 * (
        activity_cliff_pairs["fingerprint_bits_i"] + activity_cliff_pairs["fingerprint_bits_j"]
    )
    if not exact_threshold.all() or not (activity_cliff_pairs["morgan_tanimoto"] >= exact_cutoff).all():
        raise ValueError("An exact structural-neighbor pair falls below the cutoff")
    expected_on_off = activity_cliff_pairs["activity_pair_state"].eq("one_active_one_not_detected")
    if not activity_cliff_pairs["is_on_off_activity_cliff"].astype(bool).equals(expected_on_off):
        raise ValueError("Candidate on-off flags do not reproduce the activity-state rule")
    if int(activity_pair_summary["n_pairs"].sum()) != len(activity_cliff_pairs):
        raise ValueError("Activity-state summary does not partition the exact close pairs")

    activity_sensitivity = _activity_cliff_threshold_sensitivity(
        activity_cliff_pairs,
        exact_cutoff,
    )

    scaffold_a1 = substructure_numbers["analysis_1_scaffold_to_morphology"]
    scaffold_a2 = substructure_numbers["analysis_2_morphology_to_substructure"]
    figure_numbers = {key: value for key, value in cache_numbers.items() if key != "parameters"}
    figure_numbers.update(
        {
            "scaffold_n_tested": int(scaffold_a1["n_scaffolds_tested"]),
            "scaffold_min_compounds": int(scaffold_a1["min_compounds_per_scaffold"]),
            "scaffold_n_significant_fdr005": int(scaffold_a1["n_significant_fdr005"]),
            "scaffold_pct_significant_fdr005": float(scaffold_a1["pct_significant_fdr005"]),
            "scaffold_multiple_testing": scaffold_a1["multiple_testing"],
            "scaffold_median_cosine": float(scaffold_a1["median_cosine_sim"]),
            "scaffold_n_kmeans_clusters": int(scaffold_a2["n_clusters"]),
            "scaffold_n_clusters_enriched": int(scaffold_a2["n_clusters_with_enriched_scaffold"]),
            **scaffold_series_numbers,
        }
    )
    combined_parameters = {
        **parameters,
        **scaffold_series_parameters,
        "substructure_analysis": (
            "canonical notebook 2.40 active-only scaffold coherence and cluster-enrichment result"
        ),
        "substructure_multiple_testing": scaffold_a1["multiple_testing"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    similarity_summary, all_pair_statistics = _conditional_similarity_statistics(all_pair_histogram, figure_numbers)
    # Captions first: each is drawn under its own view and is that view's legend section.
    captions = build_captions(
        numbers=figure_numbers,
        parameters=combined_parameters,
        pairs=pairs,
        moa=moa,
        all_pair_statistics=all_pair_statistics,
        activity_sensitivity=activity_sensitivity,
    )
    if set(captions) != set(VIEW_ORDER):
        raise ValueError(f"Every rendered view needs exactly one caption: {sorted(set(captions))}")
    view_paths = {
        "similarity_map": _conditional_similarity(
            similarity_summary,
            all_pair_statistics,
            captions["similarity_map"],
            output_dir,
            dpi,
        ),
        "similar_structure_morphology": _distribution(
            pairs,
            captions["similar_structure_morphology"],
            output_dir,
            dpi,
        ),
        "scaffold_coherence": _scaffold_coherence(
            scaffold,
            figure_numbers,
            captions["scaffold_coherence"],
            output_dir,
            dpi,
        ),
        "activity_cliff_pairs": _activity_cliff_pairs(
            activity_cliff_pairs,
            activity_pair_summary,
            activity_sensitivity,
            figure_numbers,
            combined_parameters,
            captions["activity_cliff_pairs"],
            output_dir,
            dpi,
        ),
        "neighbor_overlap": _overlap(
            overlap,
            figure_numbers,
            captions["neighbor_overlap"],
            output_dir,
            dpi,
        ),
        "moa_coherence": _moa_coherence(moa, captions["moa_coherence"], output_dir, dpi),
        "pair_classes": _quadrants(figure_numbers, captions["pair_classes"], output_dir, dpi),
        "thresholds": _threshold_sensitivity(pairs, captions["thresholds"], output_dir, dpi),
    }
    if tuple(view_paths) != VIEW_ORDER:
        raise ValueError("Rendered view order differs from the scientific contract")

    moa.to_csv(
        output_dir / "moa_coherence.csv",
        index=False,
        float_format="%.17g",
    )
    # The pair cache is the source of record: copy its tables verbatim so both copies stay
    # byte-identical. A pandas read/write round trip shifts %.17g floats by one ULP and
    # re-encodes the parquet metadata.
    for name in (
        "phenotypic_cliffs.csv",
        "activity_cliff_pairs.parquet",
        "activity_cliff_pair_summary.csv",
        "activity_cliff_scaffold_groups.csv",
    ):
        shutil.copyfile(pair_cache_summary.parent / name, output_dir / name)
    activity_sensitivity.to_csv(
        output_dir / "activity_cliff_threshold_sensitivity.csv",
        index=False,
        float_format="%.10g",
    )
    phenoseeker_audit.to_csv(
        output_dir / "phenoseeker_procedure_transfer_audit.csv",
        index=False,
        float_format="%.17g",
    )
    metadata_path = output_dir / "activity_cliff_scaffold_groups__metadata.yml"
    metadata_path.write_text(
        _datasette_metadata_yaml(
            exact_cutoff,
            float(parameters["activity_corrected_p_threshold"]),
        )
    )
    legend_path = output_dir / "legend.md"
    write_legend(
        legend_path,
        figure_id=FIGURE_ID,
        title="Image-based profiles capture structure-activity relationships and reveal morphological activity cliffs",
        sections=[(f"{view_id}.png", SECTION_TITLES[view_id], captions[view_id]) for view_id in VIEW_ORDER],
    )

    generated_paths = [path for view_id in VIEW_ORDER for path in view_paths[view_id]] + [
        output_dir / "moa_coherence.csv",
        output_dir / "similarity_quadrants.csv",
        output_dir / "threshold_sensitivity.csv",
        output_dir / "scaffold_coherence.csv",
        output_dir / "phenotypic_cliffs.csv",
        output_dir / "activity_cliff_pairs.parquet",
        output_dir / "activity_cliff_pair_summary.csv",
        output_dir / "activity_cliff_threshold_sensitivity.csv",
        output_dir / "activity_cliff_scaffold_groups.csv",
        output_dir / "phenoseeker_procedure_transfer_audit.csv",
        metadata_path,
        legend_path,
    ]
    generated_names = {path.name for path in generated_paths}
    if generated_names != set(OUTPUT_CAPTIONS):
        raise ValueError(
            "Direct figure outputs differ from the scientific inventory: "
            f"missing={sorted(set(OUTPUT_CAPTIONS) - generated_names)}, "
            f"unexpected={sorted(generated_names - set(OUTPUT_CAPTIONS))}"
        )
    for path in generated_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Expected direct figure output is missing: {path}")

    figure_numbers.update(all_pair_statistics)
    chosen = activity_sensitivity.iloc[0]
    figure_numbers.update(
        {
            "parameters": combined_parameters,
            "activity_cliff_null_expected_pct_at_chosen_cutoff": float(100 * chosen["null_expected_fraction"]),
            "activity_cliff_null_pointwise_low_pct_at_chosen_cutoff": float(
                100 * chosen["null_pointwise_2_5_fraction"]
            ),
            "activity_cliff_null_pointwise_high_pct_at_chosen_cutoff": float(
                100 * chosen["null_pointwise_97_5_fraction"]
            ),
            "activity_call_concordance_pct_at_chosen_cutoff": float(100 * chosen["observed_concordant_fraction"]),
            "activity_call_null_expected_concordance_pct_at_chosen_cutoff": float(
                100 * chosen["null_expected_concordant_fraction"]
            ),
            "activity_call_null_pointwise_low_concordance_pct_at_chosen_cutoff": float(
                100 * chosen["null_pointwise_2_5_concordant_fraction"]
            ),
            "activity_call_null_pointwise_high_concordance_pct_at_chosen_cutoff": float(
                100 * chosen["null_pointwise_97_5_concordant_fraction"]
            ),
        }
    )
    return write_summary(
        output_dir / "summary.json",
        description=(
            "Active-compound structure-morphology "
            "relationships, exact-Tanimoto phenotypic activity-state pairs, "
            "active-only scaffold coherence, and the PhenoSeeker procedure-transfer "
            "audit."
        ),
        numbers=figure_numbers,
        files=[
            {
                "path": path.name,
                "caption": OUTPUT_CAPTIONS[path.name],
            }
            for path in generated_paths
        ],
    )


@app.command()
def main(
    pair_cache_summary: Path = typer.Option(
        ...,
        "--pair-cache-summary",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    substructure_summary: Path = typer.Option(
        ...,
        "--substructure-summary",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical notebook 2.40 --active-only summary.json",
    ),
    scaffold_coherence: Path = typer.Option(
        ...,
        "--scaffold-coherence",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical notebook 2.40 --active-only scaffold_coherence.csv",
    ),
    scaffold_series_summary: Path = typer.Option(
        ...,
        "--scaffold-series-summary",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical notebook 2.32 summary.json",
    ),
    scaffold_analysis: Path = typer.Option(
        ...,
        "--scaffold-analysis",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical notebook 2.32 scaffold_analysis.parquet",
    ),
    cliff_scaffolds: Path = typer.Option(
        ...,
        "--cliff-scaffolds",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Canonical notebook 2.32 cliff_scaffolds.csv",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        file_okay=False,
        help="Exact publication output directory",
    ),
    dpi: int = typer.Option(150, "--dpi", min=72),
) -> None:
    """Render the structure-activity supplementary figure without catalog infrastructure."""
    with plt.rc_context(SMALL_PANEL_TYPOGRAPHY):
        build_figure(
            pair_cache_summary=pair_cache_summary,
            substructure_summary=substructure_summary,
            scaffold_coherence=scaffold_coherence,
            scaffold_series_summary=scaffold_series_summary,
            scaffold_analysis=scaffold_analysis,
            cliff_scaffolds=cliff_scaffolds,
            output_dir=output_dir,
            dpi=dpi,
        )


if __name__ == "__main__":
    app()
