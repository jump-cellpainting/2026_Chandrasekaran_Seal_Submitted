"""Focused scientific contracts for the OpenTargets target-alignment analysis 2.46."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from conftest import load_notebook
from scipy.stats import spearmanr


@pytest.fixture(scope="module")
def analysis():
    return load_notebook("2.46-ss-opentargets-target-alignment.py")


def _unit_vector_with_cosine(cosine: float) -> tuple[float, float]:
    """Return a unit vector having the requested cosine with (1, 0)."""
    return cosine, math.sqrt(1.0 - cosine**2)


def test_explicit_copairs_fixture_excludes_self_and_ambiguous_negatives(analysis):
    """Manual AP/nAP agrees after identity and multilabel exclusions.

    For query Q and target A, A1 and A2 are positives. The reference copy of
    Q is excluded by JCP2022 identity, while B1 is excluded from the negative
    set because Q also carries B. C and D are the only negatives. Their cosine
    order is C, A1, A2, D, so AP = (1/2 + 2/3) / 2 = 7/12.
    """
    query_profiles = pd.DataFrame(
        [[1.0, 0.0]],
        index=pd.Index(["Q"], name="jcp2022"),
        columns=["feature_1", "feature_2"],
    )
    reference_profiles = pd.DataFrame(
        [
            _unit_vector_with_cosine(1.0),
            _unit_vector_with_cosine(0.8),
            _unit_vector_with_cosine(0.6),
            _unit_vector_with_cosine(0.7),
            _unit_vector_with_cosine(0.9),
            _unit_vector_with_cosine(0.2),
        ],
        index=pd.Index(["Q", "A1", "A2", "B1", "C", "D"], name="jcp2022"),
        columns=query_profiles.columns,
    )
    labels_by_id = {
        "Q": ("A", "B"),
        "A1": ("A",),
        "A2": ("A",),
        "B1": ("B",),
        "C": ("C",),
        "D": ("D",),
    }

    observed = analysis.score_target_alignment(
        query_profiles,
        reference_profiles,
        labels_by_id,
        query_ids=["Q"],
        reference_ids=["Q", "A1", "A2", "B1", "C", "D"],
    )

    assert observed[["jcp2022", "target_label"]].to_dict("records") == [
        {"jcp2022": "Q", "target_label": "A"},
        {"jcp2022": "Q", "target_label": "B"},
    ]
    target_a = observed.set_index(["jcp2022", "target_label"]).loc[("Q", "A")]
    assert int(target_a["n_pos_pairs"]) == 2
    assert int(target_a["n_total_pairs"]) == 4

    expected_ap = (1.0 / 2.0 + 2.0 / 3.0) / 2.0
    assert float(target_a["average_precision"]) == pytest.approx(7.0 / 12.0)
    assert expected_ap == pytest.approx(7.0 / 12.0)

    # Independent finite-sample nAP calculation, without copairs normalization.
    positives = 2
    negatives = 2
    total = positives + negatives
    harmonic_total = sum(1.0 / rank for rank in range(1, total + 1))
    random_ap = (((positives - 1.0) / (total - 1.0)) * (total - harmonic_total) + harmonic_total) / total
    expected_nap = (expected_ap - random_ap) / (1.0 - random_ap)
    assert expected_nap == pytest.approx(-7.0 / 23.0)
    assert float(target_a["normalized_average_precision"]) == pytest.approx(expected_nap)


def _eligibility_memberships() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target, analyzed, active in (
        ("A", 5, 3),
        ("B", 5, 4),
        ("C", 5, 5),
        ("D", 4, 4),
    ):
        records.extend(
            {
                "jcp2022": f"{target}{index}",
                "target_label": target,
                "is_active": index < active,
            }
            for index in range(analyzed)
        )
    return pd.DataFrame(records)


def test_target_eligibility_is_defined_before_anchor_sensitivities(analysis):
    """A target below five analyzed members fails every anchor threshold."""
    audit = analysis.build_target_audit(_eligibility_memberships(), min_analyzed=5).set_index("target_label")

    assert audit["analyzed_members"].to_dict() == {"A": 5, "B": 5, "C": 5, "D": 4}
    assert audit["active_anchors"].to_dict() == {"A": 3, "B": 4, "C": 5, "D": 4}
    assert audit["eligible_analyzed_ge5"].to_dict() == {
        "A": True,
        "B": True,
        "C": True,
        "D": False,
    }
    assert audit["eligible_anchor_3"].to_dict() == {
        "A": True,
        "B": True,
        "C": True,
        "D": False,
    }
    assert audit["eligible_anchor_4"].to_dict() == {
        "A": False,
        "B": True,
        "C": True,
        "D": False,
    }
    assert audit["eligible_anchor_5"].to_dict() == {
        "A": False,
        "B": False,
        "C": True,
        "D": False,
    }


def test_target_audit_rejects_duplicate_memberships(analysis):
    """One compound-target membership cannot enter an audit twice."""
    memberships = _eligibility_memberships()
    memberships = pd.concat([memberships, memberships.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique by target and compound"):
        analysis.build_target_audit(memberships)


def test_exact_set_identifier_is_order_and_duplicate_invariant(analysis):
    """Exact membership and anchor sets have stable identifiers."""
    first = analysis.exact_set_id(["JCP3", "JCP1", "JCP2"])
    reordered = analysis.exact_set_id(["JCP2", "JCP3", "JCP1", "JCP1"])
    distinct = analysis.exact_set_id(["JCP1", "JCP2"])

    assert first == reordered
    assert first != distinct


def test_profile_contract_rejects_missing_and_zero_norm_profiles(analysis):
    """Every expected compound must have one finite, nonzero consensus."""
    profiles = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index(["JCP1", "JCP2"], name="jcp2022"),
    )
    analysis.validate_profile_contract(profiles, ["JCP1", "JCP2"])

    with pytest.raises(ValueError, match="Profile coverage changed: missing=1"):
        analysis.validate_profile_contract(profiles.iloc[[0]], ["JCP1", "JCP2"])

    zero_norm = profiles.copy()
    zero_norm.loc["JCP2"] = 0.0
    with pytest.raises(ValueError, match="zero-norm"):
        analysis.validate_profile_contract(zero_norm, ["JCP1", "JCP2"])


def _relationship_memberships() -> pd.DataFrame:
    """Return a deterministic full-rank contextual-model fixture."""
    rng = np.random.default_rng(20260805)
    records = []
    for target_index, target in enumerate(("A", "B", "C", "D", "E", "F")):
        anchors = 3 + target_index
        for member_index in range(8):
            activity = 0.25 + 0.07 * target_index + rng.normal(0.0, 0.08)
            degree = int(rng.integers(1, 9))
            outcome = (
                0.35 * activity
                + 0.04 * activity**2
                - 0.025 * np.log1p(degree)
                + 0.015 * np.log2(anchors)
                + rng.normal(0.0, 0.06)
            )
            records.append(
                {
                    "jcp2022": f"JCP_{target}_{member_index}",
                    "target_label": target,
                    "activity_nmap": activity,
                    "full_annotation_degree": degree,
                    "active_anchors": anchors,
                    "joint_set_id": f"set_{target}",
                    "target_alignment_nap": outcome,
                }
            )
    return pd.DataFrame(records)


def test_across_target_model_has_one_row_per_target_and_no_within_components(analysis):
    """The relationship model is exclusively between equally weighted target summaries."""
    memberships = _relationship_memberships()

    models, details = analysis.fit_across_target_relationship(
        memberships,
        outcome="target_alignment_nap",
        model_id="fixture",
    )
    assert set(models["component"]) == {
        "mean_activity_linear",
        "mean_activity_quadratic",
        "mean_log_annotation_degree",
        "log_active_anchors",
    }
    assert not models["component"].str.contains("within").any()
    prepared = details["data"]
    assert isinstance(prepared, pd.DataFrame)
    assert len(prepared) == memberships["target_label"].nunique()
    assert prepared["target_label"].is_unique
    expected = spearmanr(
        prepared["mean_activity_nmap"],
        prepared["mean_consistency_nap"],
    ).statistic
    assert analysis.across_target_spearman(memberships, "target_alignment_nap") == expected


def test_matched_nonmember_controls_are_multilabel_disjoint(analysis):
    """A closer candidate sharing any query label is never eligible as a control."""
    primary = pd.DataFrame({"jcp2022": ["QUERY"], "target_label": ["A"]})
    compounds = pd.DataFrame(
        {
            "jcp2022": ["QUERY", "ANCHOR", "SHARES_B", "DISJOINT"],
            "activity_nmap": [0.80, 0.60, 0.80, 0.75],
            "is_active": [True, True, True, True],
            "source_signature": ["source_2|source_3", "source_2", "source_2|source_3", "source_2|source_3"],
            "source_count": [2, 1, 2, 2],
            "median_cell_count": [100.0, 80.0, 100.0, 110.0],
            "cell_count_percentile": [0.50, 0.30, 0.50, 0.60],
        }
    )
    profiles = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [1.0, 0.0], [0.0, 1.0]],
        index=pd.Index(compounds["jcp2022"], name="jcp2022"),
        columns=["feature_1", "feature_2"],
    )
    labels_by_id = {
        "QUERY": ("A", "B"),
        "ANCHOR": ("A",),
        "SHARES_B": ("B",),
        "DISJOINT": ("C",),
    }

    matched = analysis.build_matched_nonmembers(
        primary,
        compounds,
        profiles,
        labels_by_id,
        active_ids={"QUERY", "ANCHOR"},
    )

    assert len(matched) == 1
    row = matched.iloc[0]
    assert row["matched_nonmember_jcp2022"] == "DISJOINT"
    assert row["matched_nonmember_jcp2022"] != "SHARES_B"
    assert int(row["shared_label_count"]) == 0
    assert bool(row["label_sets_disjoint"])
    assert row["member_labels"] == "A|B"
    assert row["matched_nonmember_labels"] == "C"


def test_exception_shortlist_ranks_only_high_consistency_extremes(analysis):
    """Dense low-consistency clusters stay aggregate rather than receiving arbitrary labels."""
    rows = [
        # HH1 is repeated by target, and HH2/HH3 share an exact set.
        ("HH1", "A1", 1.0, 1.0, "hh_set_1"),
        ("HH1", "A2", 1.0, 1.0, "hh_set_2"),
        ("HH2", "A3", 1.0, 1.0, "hh_set_3"),
        ("HH3", "A4", 1.0, 1.0, "hh_set_3"),
        ("HH4", "A5", 1.0, 1.0, "hh_set_4"),
        ("HL1", "B1", 1.0, 0.0, "hl_set_1"),
        ("HL2", "B2", 1.0, 0.0, "hl_set_2"),
        ("HL3", "B3", 1.0, 0.0, "hl_set_3"),
        ("LH1", "C1", 0.0, 1.0, "lh_set_1"),
        ("LH2", "C2", 0.0, 1.0, "lh_set_2"),
        ("LH3", "C3", 0.0, 1.0, "lh_set_3"),
        ("MID1", "D1", 0.5, 0.5, "mid_set_1"),
        ("MID2", "D2", 0.5, 0.5, "mid_set_2"),
        ("MID3", "D3", 0.5, 0.5, "mid_set_3"),
    ]
    primary = pd.DataFrame(
        rows,
        columns=[
            "jcp2022",
            "target_label",
            "activity_nmap",
            "target_alignment_nap",
            "joint_set_id",
        ],
    )

    _, exceptions, _ = analysis.assign_scientific_quadrants(primary)

    assert len(exceptions) == 6
    assert exceptions["jcp2022"].is_unique
    assert exceptions["joint_set_id"].is_unique
    assert exceptions.groupby("scientific_quadrant").size().eq(3).all()
    assert "high activity / low alignment" not in set(exceptions["scientific_quadrant"])
    assert "HH3" not in set(exceptions["jcp2022"])
    assert exceptions["exception_number"].tolist() == list(range(1, 7))


def test_quadrant_export_contains_all_category_rows_without_jargon(analysis):
    """The CSV replaces plotted examples with all categorized memberships and transparent ranks."""
    rows = []
    for index, category in enumerate(
        [
            "high activity / high alignment",
            "high activity / low alignment",
            "low activity / high alignment",
            "other",
        ]
    ):
        rows.append(
            {
                "scientific_quadrant": category,
                "jcp2022": f"JCP_{index}",
                "compound_name": f"compound_{index}",
                "target_label": f"TARGET_{index}",
                "activity_nmap": 1.0 - index / 10,
                "target_alignment_nap": 0.3 - index / 10,
                "is_active": True,
                "active_anchors": 4,
                "analyzed_members": 6,
                "membership_set_id": f"members_{index}",
                "anchor_set_id": f"references_{index}",
                "joint_set_id": f"joint_{index}",
            }
        )

    exported = analysis.build_quadrant_export(pd.DataFrame(rows))

    assert len(exported) == 3
    assert "other" not in set(exported["scientific_category"])
    assert not exported["scientific_category"].str.contains("alignment").any()
    assert "active_reference_compounds" in exported
    assert "reference_set_id" in exported
    assert not any("anchor" in column for column in exported.columns)
