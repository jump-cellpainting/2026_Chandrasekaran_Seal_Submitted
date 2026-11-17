"""Scientific unit tests for the cached structure-activity pair analysis."""

import numpy as np
import pandas as pd
import pytest
from conftest import load_notebook
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

_ANALYSIS = load_notebook("2.07-ss-structure-activity-pairs.py")

STRUCT_HIGH_THRESHOLD = _ANALYSIS.STRUCT_HIGH_THRESHOLD
TANIMOTO_THRESHOLD_DENOMINATOR = _ANALYSIS.TANIMOTO_THRESHOLD_DENOMINATOR
TANIMOTO_THRESHOLD_NUMERATOR = _ANALYSIS.TANIMOTO_THRESHOLD_NUMERATOR
classify_activity_pair_states = _ANALYSIS.classify_activity_pair_states
build_target_divergence = _ANALYSIS.build_target_divergence
parse_target_sets = _ANALYSIS.parse_target_sets
exact_tanimoto_pairs = _ANALYSIS.exact_tanimoto_pairs
summarize_activity_pair_states = _ANALYSIS.summarize_activity_pair_states
undirected_pair_indices = _ANALYSIS.undirected_pair_indices


def _expected_close_pairs(fingerprints: np.ndarray) -> dict[tuple[int, int], int]:
    similarities = 1.0 - squareform(pdist(fingerprints.astype(bool), metric="jaccard"))
    expected = {}
    for row in range(len(fingerprints)):
        for col in range(row + 1, len(fingerprints)):
            if similarities[row, col] >= 0.7:
                expected[(row, col)] = int(np.dot(fingerprints[row], fingerprints[col]))
    return expected


def test_undirected_pair_indices_unions_both_directions_and_keeps_stored_zero_edges():
    graph = csr_matrix(
        (
            [0.2, 0.2, 0.0, 0.4, 0.0],
            ([0, 1, 2, 3, 3], [1, 0, 3, 1, 3]),
        ),
        shape=(4, 4),
    )

    rows, cols = undirected_pair_indices(graph)

    assert list(zip(rows, cols, strict=True)) == [(0, 1), (1, 3), (2, 3)]


def test_exact_tanimoto_pairs_matches_scipy_across_blocks_and_boundary():
    rng = np.random.default_rng(13)
    fingerprints = rng.binomial(1, 0.25, size=(17, 32)).astype(np.uint8)
    fingerprints[:, 0] = 1
    fingerprints[0] = 0
    fingerprints[0, :10] = 1
    fingerprints[1] = 0
    fingerprints[1, :7] = 1  # exactly 7 / 10 = 0.7 against row 0
    fingerprints[2] = fingerprints[0]  # identical pair

    rows, cols, intersections = exact_tanimoto_pairs(
        fingerprints,
        use_gpu=False,
        block_size=3,
    )
    observed = {
        (int(row), int(col)): int(intersection)
        for row, col, intersection in zip(rows, cols, intersections, strict=True)
    }

    assert observed == _expected_close_pairs(fingerprints)
    assert (0, 1) in observed
    assert (0, 2) in observed
    assert np.all(cols > rows)
    assert len(observed) == len(rows)
    assert STRUCT_HIGH_THRESHOLD == TANIMOTO_THRESHOLD_NUMERATOR / TANIMOTO_THRESHOLD_DENOMINATOR


@pytest.mark.parametrize(
    ("fingerprints", "message"),
    [
        (np.array([[0, 0], [1, 0]], dtype=np.uint8), "Zero-bit fingerprints"),
        (np.array([[1.0, 0.5], [1.0, 0.0]]), "must be binary"),
    ],
)
def test_exact_tanimoto_pairs_rejects_invalid_fingerprints(fingerprints, message):
    with pytest.raises(ValueError, match=message):
        exact_tanimoto_pairs(fingerprints, use_gpu=False, block_size=1)


def test_activity_pair_states_keep_unassessed_distinct_from_not_detected():
    status_i = np.array(["active", "not_detected", "active", "not_detected", "unassessed"])
    status_j = np.array(["not_detected", "active", "active", "not_detected", "active"])

    pair_state, on_off = classify_activity_pair_states(status_i, status_j)

    assert pair_state.tolist() == [
        "one_active_one_not_detected",
        "one_active_one_not_detected",
        "both_active",
        "neither_active",
        "unassessed",
    ]
    assert on_off.tolist() == [True, True, False, False, False]


def test_activity_pair_summary_uses_assessed_and_all_pair_denominators():
    pairs = pd.DataFrame(
        {
            "activity_pair_state": [
                "neither_active",
                "one_active_one_not_detected",
                "both_active",
                "unassessed",
            ],
            "scaffold_relationship": ["same", "different", "missing", "same"],
        }
    )

    summary, assessed_pair_count = summarize_activity_pair_states(pairs)
    indexed = summary.set_index("activity_pair_state")

    assert assessed_pair_count == 3
    assert indexed["n_pairs"].to_dict() == {
        "neither_active": 1,
        "one_active_one_not_detected": 1,
        "both_active": 1,
        "unassessed": 1,
    }
    assert indexed.loc["one_active_one_not_detected", "pct_of_all_close_pairs"] == 25
    assert indexed.loc["one_active_one_not_detected", "pct_of_activity_assessed_close_pairs"] == pytest.approx(100 / 3)
    assert np.isnan(indexed.loc["unassessed", "pct_of_activity_assessed_close_pairs"])
    assert indexed.loc["one_active_one_not_detected", "different_scaffold_pairs"] == 1


def test_activity_pair_summary_handles_empty_universe():
    empty_pairs = pd.DataFrame(columns=["activity_pair_state", "scaffold_relationship"])

    summary, assessed_pair_count = summarize_activity_pair_states(empty_pairs)

    assert assessed_pair_count == 0
    assert summary["n_pairs"].eq(0).all()
    assert summary["pct_of_all_close_pairs"].isna().all()
    assert summary["pct_of_activity_assessed_close_pairs"].isna().all()


def test_parse_target_sets_splits_pipes_and_skips_missing():
    annotations = pd.DataFrame({"jcp2022": ["a", "b", "c"], "hub_targets": ["EGFR|ERBB2", None, ""]})
    assert parse_target_sets(annotations, "hub_targets") == {"a": frozenset({"EGFR", "ERBB2"})}


def test_target_divergence_compares_within_source_and_never_pools_vocabularies():
    pairs = pd.DataFrame({"jcp_i": ["a", "a", "c", "e", "g"], "jcp_j": ["b", "d", "d", "f", "h"]})
    hub = {"a": frozenset({"EGFR"}), "b": frozenset({"EGFR", "ERBB2"}), "d": frozenset({"BRD4"})}
    chembl = {
        "c": frozenset({"P00533"}),
        "e": frozenset({"P00533"}),
        "f": frozenset({"P00533", "P04626"}),
        "g": frozenset({"P1"}),
        "h": frozenset({"P2"}),
    }
    cliff = np.array([True, True, True, False, False])
    table, numbers = build_target_divergence(pairs, cliff, ~cliff, {"hub_targets": hub, "chembl_targets": chembl})
    combined = numbers["repurposing_hub_and_chembl"]
    # a-b share EGFR, a-d are disjoint, c-d mix vocabularies and are not comparable
    assert (combined["cliff_annotated_pairs"], combined["cliff_divergent_pairs"]) == (2, 1)
    # e-f share P00533, g-h are disjoint
    assert (combined["concordant_annotated_pairs"], combined["concordant_divergent_pairs"]) == (2, 1)
    assert numbers["chembl_only"]["cliff_annotated_pairs"] == 0
    assert list(table.columns[:3]) == ["contract", "sources", "group"] and len(table) == 4
    assert table.loc[table["contract"] == "chembl_only", "odds_ratio"].isna().all()
