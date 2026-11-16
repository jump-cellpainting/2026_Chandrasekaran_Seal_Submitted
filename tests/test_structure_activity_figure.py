"""Behavioral tests for the Supplementary STRUCTURE renderer's derived statistics.

The renderer owns the activity-cliff threshold sensitivity: a permutation null over the
same-versus-different activity call of structurally close pairs. That estimand is computed here
rather than in the cached `2.07` analysis, so it needs its own coverage.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPT = ROOT / "notebooks/figure-structure-activity.py"


@pytest.fixture(scope="module")
def figure_module() -> ModuleType:
    """Load the hyphenated renderer without running its CLI."""
    spec = importlib.util.spec_from_file_location("figure_structure_activity", FIGURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {FIGURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pairs(rows: list[tuple[str, str, str, str, float, str]]) -> pd.DataFrame:
    """Build a close-pair frame from (compound_i, compound_j, status_i, status_j, tanimoto, state)."""
    return pd.DataFrame(
        rows,
        columns=[
            "jcp_i",
            "jcp_j",
            "activity_status_i",
            "activity_status_j",
            "morgan_tanimoto",
            "activity_pair_state",
        ],
    )


# Four compounds, two active. Every threshold below keeps at least one edge.
#   c1 active, c2 active, c3 not detected, c4 not detected
BASE_ROWS = [
    ("c1", "c2", "active", "active", 1.00, "both_active"),
    ("c1", "c3", "active", "not_detected", 0.90, "one_active_one_not_detected"),
    ("c2", "c4", "active", "not_detected", 0.80, "one_active_one_not_detected"),
    ("c3", "c4", "not_detected", "not_detected", 0.70, "neither_active"),
]


@pytest.fixture
def sensitivity(figure_module: ModuleType) -> pd.DataFrame:
    return figure_module._activity_cliff_threshold_sensitivity(build_pairs(BASE_ROWS), 0.70)


def row_at(sensitivity: pd.DataFrame, threshold: float) -> pd.Series:
    matches = sensitivity[np.isclose(sensitivity["minimum_exact_morgan_tanimoto"], threshold)]
    assert len(matches) == 1, f"expected exactly one row at {threshold}"
    return matches.iloc[0]


def test_grid_spans_the_cutoff_to_one(figure_module: ModuleType, sensitivity: pd.DataFrame) -> None:
    assert len(sensitivity) == figure_module.ACTIVITY_CLIFF_SENSITIVITY_GRID_SIZE
    assert sensitivity["minimum_exact_morgan_tanimoto"].iloc[0] == pytest.approx(0.70)
    assert sensitivity["minimum_exact_morgan_tanimoto"].iloc[-1] == pytest.approx(1.00)
    assert sensitivity["minimum_exact_morgan_tanimoto"].is_monotonic_increasing


def test_observed_fraction_counts_discordant_pairs_above_each_threshold(sensitivity: pd.DataFrame) -> None:
    # At 0.70 all four pairs are assessed and two join an active to a non-active compound.
    loose = row_at(sensitivity, 0.70)
    assert loose["assessed_pair_count"] == 4
    assert loose["participating_compound_count"] == 4
    assert loose["active_compound_count"] == 2
    assert loose["observed_candidate_cliff_fraction"] == pytest.approx(2 / 4)

    # At 0.80 the weakest pair drops out; two of the remaining three are discordant.
    middle = row_at(sensitivity, 0.80)
    assert middle["assessed_pair_count"] == 3
    assert middle["observed_candidate_cliff_fraction"] == pytest.approx(2 / 3)

    # At 1.00 only the both-active pair survives, so nothing is discordant.
    exact = row_at(sensitivity, 1.00)
    assert exact["assessed_pair_count"] == 1
    assert exact["participating_compound_count"] == 2
    assert exact["observed_candidate_cliff_fraction"] == pytest.approx(0.0)


def test_null_expectation_matches_the_closed_form_for_random_label_assignment(sensitivity: pd.DataFrame) -> None:
    for threshold in (0.70, 0.80, 1.00):
        row = row_at(sensitivity, threshold)
        n_compounds = row["participating_compound_count"]
        n_active = row["active_compound_count"]
        expected = 2 * n_active * (n_compounds - n_active) / (n_compounds * (n_compounds - 1))
        assert row["null_expected_fraction"] == pytest.approx(expected)
        assert row["null_pointwise_2_5_fraction"] <= row["null_pointwise_97_5_fraction"]


def test_concordance_columns_are_the_complement_with_swapped_interval_bounds(sensitivity: pd.DataFrame) -> None:
    assert sensitivity["observed_concordant_fraction"].to_numpy() == pytest.approx(
        1 - sensitivity["observed_candidate_cliff_fraction"].to_numpy()
    )
    assert sensitivity["null_expected_concordant_fraction"].to_numpy() == pytest.approx(
        1 - sensitivity["null_expected_fraction"].to_numpy()
    )
    # A high candidate-cliff bound is a low concordance bound: the interval must flip, not shift.
    assert sensitivity["null_pointwise_2_5_concordant_fraction"].to_numpy() == pytest.approx(
        1 - sensitivity["null_pointwise_97_5_fraction"].to_numpy()
    )
    assert sensitivity["null_pointwise_97_5_concordant_fraction"].to_numpy() == pytest.approx(
        1 - sensitivity["null_pointwise_2_5_fraction"].to_numpy()
    )
    assert (
        sensitivity["null_pointwise_2_5_concordant_fraction"] <= sensitivity["null_pointwise_97_5_concordant_fraction"]
    ).all()


def test_permutation_null_is_deterministic_under_the_fixed_seed(figure_module: ModuleType) -> None:
    first = figure_module._activity_cliff_threshold_sensitivity(build_pairs(BASE_ROWS), 0.70)
    second = figure_module._activity_cliff_threshold_sensitivity(build_pairs(BASE_ROWS), 0.70)

    pd.testing.assert_frame_equal(first, second)


def test_row_order_does_not_change_the_estimate(figure_module: ModuleType) -> None:
    """Pair order comes from an upstream parquet scan, so it must not steer the null."""
    shuffled = build_pairs(list(reversed(BASE_ROWS)))

    baseline = figure_module._activity_cliff_threshold_sensitivity(build_pairs(BASE_ROWS), 0.70)
    reordered = figure_module._activity_cliff_threshold_sensitivity(shuffled, 0.70)

    pd.testing.assert_frame_equal(
        baseline.sort_values("minimum_exact_morgan_tanimoto").reset_index(drop=True),
        reordered.sort_values("minimum_exact_morgan_tanimoto").reset_index(drop=True),
    )


def test_conflicting_activity_calls_for_one_compound_are_rejected(figure_module: ModuleType) -> None:
    """A compound is active or not; two different calls mean the upstream join went wrong."""
    conflicting = build_pairs(
        [
            *BASE_ROWS,
            ("c3", "c1", "active", "active", 0.75, "both_active"),  # c3 was not_detected above
        ]
    )

    with pytest.raises(ValueError, match="inconsistent activity calls"):
        figure_module._activity_cliff_threshold_sensitivity(conflicting, 0.70)


def test_unassessed_pair_states_are_dropped_before_the_consistency_check(figure_module: ModuleType) -> None:
    """Pairs outside the assessed states carry no activity call and must not enter the null."""
    with_unassessed = build_pairs(
        [
            *BASE_ROWS,
            ("c3", "c1", "unknown", "unknown", 0.95, "one_missing_activity_result"),
        ]
    )

    result = figure_module._activity_cliff_threshold_sensitivity(with_unassessed, 0.70)

    pd.testing.assert_frame_equal(
        result,
        figure_module._activity_cliff_threshold_sensitivity(build_pairs(BASE_ROWS), 0.70),
    )


def test_a_threshold_with_no_remaining_pairs_fails_loudly(figure_module: ModuleType) -> None:
    """Silently dropping a grid point would leave a curve that thins out without saying so."""
    sparse = build_pairs([row for row in BASE_ROWS if row[4] <= 0.90])

    with pytest.raises(ValueError, match="No assessed close pairs remain"):
        figure_module._activity_cliff_threshold_sensitivity(sparse, 0.90)


def test_pair_cache_contract_rejects_a_summary_declaring_different_files(
    figure_module: ModuleType, tmp_path: Path
) -> None:
    """The renderer consumes analysis 2.07 by contract, so a drifted cache must not load."""
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "description": "Cached pair analysis.",
                "numbers": {},
                "files": [{"path": "pairs.parquet", "kind": "table", "caption": "pairs"}],
            }
        )
    )

    with pytest.raises(ValueError, match="differ from the expected contract"):
        figure_module.load_pair_cache(summary_path)
