"""Behavioral tests for the direct Figure 2 scientific contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPT = ROOT / "notebooks/figure-phenotypic-consistency.py"


@pytest.fixture(scope="module")
def figure_module() -> ModuleType:
    """Load the hyphenated direct-figure script without running its CLI."""
    spec = importlib.util.spec_from_file_location("figure_phenotypic_consistency", FIGURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {FIGURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_moa_memberships_preserve_pipe_delimited_labels(
    figure_module: ModuleType,
) -> None:
    """Exact labels remain distinct when one label is a substring of another."""
    metadata = pd.DataFrame(
        {
            "Metadata_JCP2022": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "Metadata_repurposing_moa": [
                "kinase inhibitor|kinase",
                "kinase inhibitor",
                "kinase",
                "",
                None,
                "MEK inhibitor|MEK inhibitor",
            ],
        }
    )

    memberships = figure_module.exact_moa_memberships(metadata, pd.Series(metadata["Metadata_JCP2022"]))

    assert list(memberships.itertuples(index=False, name=None)) == [
        ("j6", "MEK inhibitor"),
        ("j1", "kinase"),
        ("j3", "kinase"),
        ("j1", "kinase inhibitor"),
        ("j2", "kinase inhibitor"),
    ]
    assert set(memberships.loc[memberships["mechanism_of_action"] == "kinase", "jcp2022"]) == {
        "j1",
        "j3",
    }
    assert not memberships.duplicated(["jcp2022", "mechanism_of_action"]).any()


def test_rankings_use_normalized_ap_and_corrected_p_independently(
    figure_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Effect size determines rank while corrected p-values determine significance."""

    def results_for(group_type: str) -> pd.DataFrame:
        rows = []
        for index in range(22):
            rows.append(
                {
                    "group_value": f"{group_type}_{index:02d}",
                    "n_perturbations": index + 2,
                    "mean_normalized_average_precision": 0.90 - index * 0.02,
                    "corrected_p_value": 0.001 + index * 0.002,
                }
            )
        rows[0]["corrected_p_value"] = 0.80
        rows[1]["corrected_p_value"] = 1e-12
        rows[2]["corrected_p_value"] = 0.05
        rows[-1]["corrected_p_value"] = 1e-30
        return pd.DataFrame(rows)

    results = {group_type: results_for(group_type) for group_type in ("chemical_probes", "moa")}

    def fake_query(_connection, *, group_type: str, **_kwargs) -> pd.DataFrame:
        return results[group_type].copy()

    monkeypatch.setattr(figure_module, "query_consistency_results", fake_query)
    rankings = figure_module.prepare_rankings(
        object(),
        dataset="fixture",
        preprocessing="consistency_fixture",
        filter_name="all_sources",
        metric="cosine",
    )

    assert len(rankings) == 40
    for source in ("Chemical Probes targets", "Drug Repurposing Hub MOAs"):
        selected = rankings[rankings["annotation_source"] == source].reset_index(drop=True)
        assert selected["rank"].tolist() == list(range(1, 21))
        assert selected["mean_normalized_average_precision"].is_monotonic_decreasing
        assert selected.loc[0, "significant"] is np.False_
        assert selected.loc[1, "significant"] is np.True_
        assert selected.loc[2, "corrected_p_value"] == pytest.approx(0.05)
        assert selected.loc[2, "significant"] is np.False_
        assert not selected["group_value"].str.endswith("21").any()


def test_missing_activity_reasons_form_an_exact_partition(figure_module: ModuleType) -> None:
    """Each embedded profile without an activity result has exactly one documented reason."""
    well_obs = pd.DataFrame(
        {
            "Source": ["source_1"] * 8,
            "Plate": ["t1", "t2", "s1", "s2", "p1", "n1", "a1", "a2"],
            "JCP2022": [
                "target2_only",
                "target2_only",
                "single_eligible",
                "single_eligible",
                "positive_control",
                "negative_control",
                "analyzed",
                "analyzed",
            ],
        }
    )
    plates = pd.DataFrame(
        {
            "Metadata_Source": ["source_1"] * 8,
            "Metadata_Plate": ["t1", "t2", "s1", "s2", "p1", "n1", "a1", "a2"],
            "Metadata_PlateType": [
                "TARGET2",
                "TARGET2",
                "TARGET2",
                "OTHER",
                "OTHER",
                "OTHER",
                "OTHER",
                "OTHER",
            ],
        }
    )
    controls = pd.DataFrame(
        {
            "Metadata_JCP2022": ["positive_control", "negative_control"],
            "Metadata_pert_type": ["poscon", "negcon"],
        }
    )
    perturbation_ids = pd.Series(
        [
            "target2_only",
            "single_eligible",
            "positive_control",
            "negative_control",
            "analyzed",
        ]
    )
    activity = pd.DataFrame({"Metadata_JCP2022": ["analyzed"]})

    counts = figure_module.classify_missing_activity_results(well_obs, perturbation_ids, activity, plates, controls)

    assert counts == {
        "activity_missing_target2_only": 1,
        "activity_missing_insufficient_replicates": 1,
        "activity_missing_positive_controls": 1,
        "activity_missing_negative_controls": 1,
    }
    assert sum(counts.values()) == len(set(perturbation_ids) - set(activity["Metadata_JCP2022"]))


def test_cross_scaffold_retrieval_excludes_exact_shared_scaffolds(
    figure_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-MOA compound on the same exact scaffold is not an eligible partner."""
    ids = np.array(["a0", "a1", "b0", "b1"])
    memberships = pd.DataFrame(
        {
            "jcp2022": ids,
            "mechanism_of_action": ["A", "A", "B", "B"],
            "active": [True] * 4,
        }
    )
    consistency = pd.DataFrame({"mechanism_of_action": ["A", "B"]})
    annotations = pd.DataFrame(
        {
            "Metadata_JCP2022": ids,
            "Metadata_MurckoScaffold": ["shared_A", "shared_A", "b0", "b1"],
        }
    )
    perturbations = pd.DataFrame(
        {
            "jcp2022": ids,
            "umap_1": [0.0, 0.1, 1.0, 1.1],
            "umap_2": [0.0, 0.1, 1.0, 1.1],
        }
    )
    profiles = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    monkeypatch.setattr(
        figure_module,
        "load_selected_profiles",
        lambda _path, _selected_ids: (ids, profiles),
    )

    compounds, _curve, _numbers = figure_module.prepare_global_moa_retrieval(
        memberships,
        consistency,
        annotations,
        perturbations,
        Path("unused.h5ad"),
        permutations=32,
        seed=17,
        max_k=3,
    )
    indexed = compounds.set_index("jcp2022")

    assert indexed.loc["a0", "cross_scaffold_same_moa_partners"] == 0
    assert indexed.loc["a1", "cross_scaffold_same_moa_partners"] == 0
    assert pd.isna(indexed.loc["a0", "cross_scaffold_nearest_same_moa_rank"])
    assert indexed.loc["a0", "same_moa_proximity_score"] == 0.0
    assert indexed.loc["b0", "cross_scaffold_same_moa_partners"] == 1
    assert indexed.loc["b1", "cross_scaffold_same_moa_partners"] == 1


def test_seeded_multilabel_permutations_are_reproducible(
    figure_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seeded null is repeatable while complete multilabel sets remain attached."""
    ids = np.array(["j0", "j1", "j2", "j3", "j4", "j5"])
    memberships = pd.DataFrame(
        [
            ("j0", "A", True),
            ("j0", "B", True),
            ("j1", "A", True),
            ("j2", "B", True),
            ("j3", "C", True),
            ("j4", "C", True),
            ("j4", "D", True),
            ("j5", "D", True),
        ],
        columns=["jcp2022", "mechanism_of_action", "active"],
    )
    consistency = pd.DataFrame({"mechanism_of_action": ["A", "B", "C", "D"]})
    annotations = pd.DataFrame(
        {
            "Metadata_JCP2022": ids,
            "Metadata_MurckoScaffold": [f"scaffold_{index}" for index in range(len(ids))],
        }
    )
    perturbations = pd.DataFrame(
        {
            "jcp2022": ids,
            "umap_1": np.arange(len(ids), dtype=float),
            "umap_2": np.arange(len(ids), dtype=float) / 2,
        }
    )
    profiles = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        figure_module,
        "load_selected_profiles",
        lambda _path, _selected_ids: (ids, profiles),
    )

    first = figure_module.prepare_global_moa_retrieval(
        memberships,
        consistency,
        annotations,
        perturbations,
        Path("unused.h5ad"),
        permutations=128,
        seed=20260714,
        max_k=5,
    )
    second = figure_module.prepare_global_moa_retrieval(
        memberships,
        consistency,
        annotations,
        perturbations,
        Path("unused.h5ad"),
        permutations=128,
        seed=20260714,
        max_k=5,
    )
    different_seed = figure_module.prepare_global_moa_retrieval(
        memberships,
        consistency,
        annotations,
        perturbations,
        Path("unused.h5ad"),
        permutations=128,
        seed=20260715,
        max_k=5,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]
    indexed = first[0].set_index("jcp2022")
    assert indexed.loc["j0", "mechanism_of_action"] == "A|B"
    assert indexed.loc["j4", "mechanism_of_action"] == "C|D"
    pd.testing.assert_series_equal(
        first[1]["observed_coverage"],
        different_seed[1]["observed_coverage"],
    )
    assert not np.array_equal(
        first[1]["null_coverage_mean"].to_numpy(),
        different_seed[1]["null_coverage_mean"].to_numpy(),
    )


@pytest.fixture
def captions(figure_module: ModuleType) -> dict[str, str]:
    return figure_module.build_captions(
        numbers={
            "perturbation_level_profiles": 100,
            "active_perturbations": 20,
            "perturbations_without_activity_results": 4,
            "activity_missing_target2_only": 1,
            "activity_missing_insufficient_replicates": 1,
            "activity_missing_positive_controls": 1,
            "activity_missing_negative_controls": 1,
            "moa_retrieval_compounds": 42,
        },
        configuration={"moa_retrieval_permutations": 1_000},
        well_source_viewport={"included_wells": 90, "total_wells": 100, "excluded_wells": 10},
        retrieval_curve=pd.DataFrame(
            {
                "k": [1, 10],
                "observed_coverage": [0.212, 0.425],
                "null_coverage_mean": [0.013, 0.116],
            }
        ),
        global_moa_score_upper=2.5,
    )


def test_combined_moa_caption_is_self_contained(captions: dict[str, str]) -> None:
    """The combined standalone export retains the cohort, null, exclusion, and headline values."""
    caption = captions["combined_global_moa_retrieval"]
    assert "42 active, MOA-annotated compounds" in caption
    assert "same Bemis-Murcko scaffold" in caption
    assert "1,000 permutations of complete multilabel annotation sets" in caption
    assert "21.2% observed versus 1.3% permuted at k=1" in caption
    assert "42.5% versus 11.6% at k=10" in caption


def test_curated_moa_caption_distinguishes_historical_cutoff(
    figure_module: ModuleType, captions: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained class below the old cutoff must not be described as passing it now."""
    moa = "histone lysine methyltransferase inhibitor"
    monkeypatch.setattr(figure_module, "CURATED_MOAS", (moa,))
    memberships = pd.DataFrame(
        {"jcp2022": [f"j{i}" for i in range(7)], "mechanism_of_action": moa, "active": True}
    )
    evidence = pd.DataFrame([{
        "rank": 11, "mechanism_of_action": moa, "n_perturbations": 7,
        "mean_normalized_average_precision": 0.17577346031190419,
        "corrected_p_value": 0.0088726385463418003,
    }])
    _, classes = figure_module.select_curated_moas(memberships, evidence)
    assert classes.loc[0, "mean_normalized_average_precision"] == evidence.loc[0, "mean_normalized_average_precision"]
    caption = captions["selected_moa_neighborhoods"]
    assert "historically selected" in caption
    assert "cutoff is not reapplied" in caption
    assert "selected_moa_classes.csv" in caption
