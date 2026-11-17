"""Focused tests for the OpenTargets target-group consistency figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from conftest import load_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def renderer():
    return load_notebook("figure-opentargets-group-consistency.py")


def test_membership_set_identifier_is_order_invariant(renderer):
    """Exact compound sets receive the same identifier regardless of row order."""
    first = renderer.membership_set_id(pd.Series(["JCP3", "JCP1", "JCP2"]))
    second = renderer.membership_set_id(pd.Series(["JCP2", "JCP3", "JCP1", "JCP1"]))
    assert first == second


def test_deduplication_collapses_only_exact_membership_sets(renderer):
    """The sensitivity unit keeps one representative for each exact compound set."""
    group_summary = pd.DataFrame(
        {
            "annotation_group": ["AKT1", "AKT2", "MTOR"],
            "membership_set_id": ["shared", "shared", "distinct"],
            "membership_set_targets": ["AKT1|AKT2", "AKT1|AKT2", "MTOR"],
            "mean_normalized_average_precision": [0.20, 0.20, 0.10],
            "corrected_p_value": [0.01, 0.01, 0.20],
            "consistent_at_5pct_fdr": [True, True, False],
            "n_contributing_compounds": [5, 5, 6],
            "activity_nmap_mean": [0.90, 0.90, 0.80],
        }
    )
    continuous_rows = []
    for trait_index, (trait_id, title, xlabel) in enumerate(renderer.CONTINUOUS_TRAITS):
        for group_index, row in group_summary.iterrows():
            continuous_rows.append(
                row.to_dict()
                | {
                    "trait_id": trait_id,
                    "title": title,
                    "xlabel": xlabel,
                    "property_mean": float(trait_index + group_index),
                    "quartile": group_index + 1,
                }
            )
    continuous = pd.DataFrame(continuous_rows)
    categorical = pd.DataFrame(
        {
            "annotation_group": ["AKT1", "AKT2", "MTOR"],
            "membership_set_id": ["shared", "shared", "distinct"],
            "family": "lipinski",
            "category_order": 0,
            "category": "0",
            "n_category_members": [5, 5, 6],
            "fraction_category_members": 1.0,
            "mean_normalized_average_precision": [0.20, 0.20, 0.10],
            "corrected_p_value": [0.01, 0.01, 0.20],
            "consistent_at_5pct_fdr": [True, True, False],
            "n_contributing_compounds": [5, 5, 6],
            "n_unique_compounds_in_category": 11,
        }
    )

    continuous_dedup, categorical_dedup, representatives = renderer.deduplicate_membership_sets(
        continuous, categorical, group_summary
    )

    assert set(representatives["membership_set_id"]) == {"shared", "distinct"}
    assert len(categorical_dedup) == 2
    assert continuous_dedup.groupby("trait_id")["membership_set_id"].nunique().eq(2).all()


def test_input_contract_uses_the_selected_regular_slice(renderer):
    """Owning summaries must agree with the selected consistency configuration."""
    consistency_summary = {
        "numbers": {
            "dataset": renderer.DATASET,
            "columns": renderer.COLUMNS,
            "preprocessing": renderer.ACTIVITY_PREPROCESSING,
            "filter_name": renderer.FILTER_NAME,
            "activity_params": renderer.ACTIVITY_PARAMS,
            "activity_threshold": renderer.ACTIVITY_THRESHOLD,
            "consistency_config": {
                "columns": renderer.COLUMNS,
                "preprocessing": renderer.PREPROCESSING,
                "filter_name": renderer.FILTER_NAME,
                "distance": renderer.DISTANCE,
            },
        }
    }
    property_summary = {
        "numbers": {
            "dataset": renderer.DATASET,
            "columns": renderer.COLUMNS,
            "preprocessing": renderer.ACTIVITY_PREPROCESSING,
            "filter_name": renderer.FILTER_NAME,
            "activity_params": renderer.ACTIVITY_PARAMS,
            "activity_threshold": renderer.ACTIVITY_THRESHOLD,
        }
    }
    renderer.validate_input_contracts(consistency_summary, property_summary)

    property_summary["numbers"]["activity_threshold"] = 0.05
    with pytest.raises(ValueError, match="activity thresholds changed"):
        renderer.validate_input_contracts(consistency_summary, property_summary)


def test_categorical_join_rejects_partial_compound_coverage(renderer):
    """A partial metadata join cannot silently undercount target composition."""
    scores = pd.DataFrame(
        {
            "annotation_group": ["AKT1", "AKT1"],
            "jcp2022": ["JCP1", "JCP2"],
            "normalized_average_precision": [0.1, 0.2],
        }
    )
    group_summary = pd.DataFrame(
        {
            "annotation_group": ["AKT1"],
            "membership_set_id": ["set1"],
            "n_contributing_compounds": [2],
            "mean_normalized_average_precision": [0.15],
            "activity_nmap_mean": [0.6],
            "corrected_p_value": [0.01],
            "consistent_at_5pct_fdr": [True],
        }
    )
    compounds = pd.DataFrame(
        {
            "jcp2022": ["JCP1"],
            "lipinski_violations": [0],
            "valid_molecule": [True],
            "has_pains": [False],
            "clinical_phase": [None],
            "in_kcgs": [False],
            "in_pkis": [False],
            "in_pkis2": [False],
            "in_chemical_probes_sgc": [False],
            "activity_nmap": [0.5],
        }
    )

    with pytest.raises(ValueError, match="missing 1 contributing compound IDs"):
        renderer.build_categorical_table(scores, group_summary, compounds)

    incomplete_values = pd.concat(
        [compounds, compounds.assign(jcp2022="JCP2", activity_nmap=float("nan"))],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="missing required values: activity_nmap=1"):
        renderer.build_categorical_table(scores, group_summary, incomplete_values)


def matched_inputs(renderer):
    """Return a small exact-membership fixture for matched activity contracts."""
    scores = pd.DataFrame(
        {
            "annotation_group": ["A", "A", "A", "B", "B"],
            "jcp2022": ["JCP1", "JCP2", "JCP3", "JCP1", "JCP2"],
        }
    )
    group_summary = pd.DataFrame(
        {
            "annotation_group": ["A", "B"],
            "membership_set_id": ["set_a", "set_b"],
            "n_contributing_compounds": [3, 2],
            "activity_nmap_mean": [0.7, 0.6],
        }
    )
    compounds = pd.DataFrame(
        {
            "jcp2022": ["JCP1", "JCP2", "JCP3"],
            "nmap": [0.5, 0.7, 0.9],
            "active": [True, True, True],
            "lipinski_violations": [0, 1, 2],
            "valid_molecule": [True, True, True],
            "has_pains": [False, True, False],
            "clinical_phase": ["Phase 1", None, "Launched"],
            "in_kcgs": [True, False, False],
            "in_pkis": [False, False, False],
            "in_pkis2": [False, True, False],
            "in_chemical_probes_sgc": [False, False, True],
        }
    )
    activity = pd.DataFrame(
        {
            "jcp2022": ["JCP1", "JCP2", "JCP3"],
            "activity_nmap": [0.5, 0.7, 0.9],
            "activity_corrected_p_value": [0.01, 0.02, 0.03],
            "upstream_active_flag": [True, True, True],
        }
    )
    chemical = pd.DataFrame(
        {
            "jcp2022": ["JCP1", "JCP2", "JCP3"],
            "valid_molecule": [True, True, True],
            "logp": [1.0, 2.0, 3.0],
            "molecular_weight": [300.0, 400.0, 500.0],
            "rotatable_bonds": [1.0, 2.0, 3.0],
            "qed": [0.3, 0.5, 0.7],
            "tpsa": [50.0, 70.0, 90.0],
            "fraction_csp3": [0.1, 0.3, 0.5],
        }
    )
    return scores, group_summary, compounds, activity, chemical


def test_matched_activity_uses_unique_compounds_and_exact_member_means(renderer, monkeypatch):
    """Repeated target memberships cannot overweight the compound-level view."""
    monkeypatch.setattr(renderer, "EXPECTED_UNIQUE_COMPOUNDS", 3)
    monkeypatch.setattr(renderer, "EXPECTED_MEMBERSHIPS", 5)
    scores, group_summary, compounds, activity, chemical = matched_inputs(renderer)

    matched = renderer.build_matched_compounds(scores, group_summary, compounds, activity, chemical)
    audited = renderer.augment_group_activity_audit(group_summary, scores, matched)
    bins = renderer.build_compound_activity_bins(matched)

    assert matched["jcp2022"].is_unique
    assert len(matched) == 3
    assert matched.set_index("jcp2022").loc["JCP1", "n_displayed_target_labels"] == 2
    assert audited.set_index("annotation_group").loc["A", "activity_nmap_mean"] == pytest.approx(0.7)
    assert audited.set_index("annotation_group").loc["B", "activity_nmap_mean"] == pytest.approx(0.6)
    assert audited["active_member_percent"].eq(100.0).all()
    assert bins["rate"].eq(1.0).all()


def test_matched_activity_threshold_is_strict(renderer, monkeypatch):
    """A corrected p-value equal to 0.10 is not active."""
    monkeypatch.setattr(renderer, "EXPECTED_UNIQUE_COMPOUNDS", 3)
    monkeypatch.setattr(renderer, "EXPECTED_MEMBERSHIPS", 5)
    scores, group_summary, compounds, activity, chemical = matched_inputs(renderer)
    activity.loc[activity["jcp2022"].eq("JCP3"), "activity_corrected_p_value"] = renderer.ACTIVITY_THRESHOLD

    with pytest.raises(ValueError, match="strict activity corrected p-value"):
        renderer.build_matched_compounds(scores, group_summary, compounds, activity, chemical)


def test_matched_categories_retain_empty_fixed_levels(renderer, monkeypatch):
    """Filtering preserves the fixed activity category universe, including empty levels."""
    monkeypatch.setattr(renderer, "EXPECTED_UNIQUE_COMPOUNDS", 3)
    monkeypatch.setattr(renderer, "EXPECTED_MEMBERSHIPS", 5)
    scores, group_summary, compounds, activity, chemical = matched_inputs(renderer)
    matched = renderer.build_matched_compounds(scores, group_summary, compounds, activity, chemical)

    lipinski, clinical, collections = renderer.build_matched_boxplot_distributions(matched)

    assert [label for label, _ in lipinski] == ["0", "1", "2", "3", "4"]
    assert [label for label, _ in clinical] == ["Phase 1", "Phase 2", "Phase 3", "Launched", "All other"]
    assert [label for label, _ in collections] == ["PAINS", "KCGS", "PKIS", "PKIS2", "Probes", "All"]
    assert len(dict(collections)["PKIS"]) == 0


@pytest.mark.parametrize(
    ("limits", "ticks"),
    [
        ((0.80, 1.05), (0.80, 0.90, 1.00)),
        ((0.40, 1.05), (0.40, 0.60, 0.80, 1.00)),
        ((0.40, 1.12), (0.40, 0.60, 0.80, 1.00)),
    ],
)
def test_activity_axes_use_explicit_truncated_scales(renderer, limits, ticks):
    """Trimmed activity axes must retain numeric ticks and visible break marks."""
    figure, axis = renderer.plt.subplots()
    renderer.apply_truncated_activity_y_axis(axis, limits, ticks)

    assert axis.get_ylim() == pytest.approx(limits)
    assert axis.get_yticks() == pytest.approx(ticks)
    break_lines = [line for line in axis.lines if line.get_gid() == "truncated-activity-y-axis-break"]
    assert len(break_lines) == 2
    assert all(not line.get_clip_on() for line in break_lines)
    assert [float(line.get_xdata()[0]) for line in break_lines] == pytest.approx([0.0, 0.0])
    assert [float(line.get_ydata()[0]) for line in break_lines] == pytest.approx([0.022, 0.068])
    assert all(line.get_markersize() == pytest.approx(6.0) for line in break_lines)
    assert all(line.get_linestyle() == "None" for line in break_lines)
    renderer.plt.close(figure)


def test_rule_declares_the_complete_canonical_package(renderer):
    """Snakemake tracks every figure, table, legend, and summary artifact."""
    rules = (REPOSITORY_ROOT / "rules" / "figures.smk").read_text()
    marker = "rule figure_opentargets_group_consistency:"
    assert marker in rules
    rule_block = rules.split(marker, maxsplit=1)[1].split("\n\nrule ", maxsplit=1)[0]
    for stem in renderer.FIGURE_OUTPUT_STEMS:
        assert f"{stem}.png" in rule_block
        assert f"{stem}.pdf" in rule_block
    for filename in (
        renderer.GROUP_SUMMARY_TABLE,
        renderer.CONTINUOUS_TABLE,
        renderer.CATEGORICAL_TABLE,
        renderer.MEMBERSHIP_TABLE,
        renderer.MATCHED_COMPOUND_TABLE,
        renderer.LEGEND_FILE,
        "summary.json",
    ):
        assert filename in rule_block
