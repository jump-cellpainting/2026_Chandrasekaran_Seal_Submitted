"""Focused behavioral tests for the compound-activity-association figure workflow."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import load_notebook
from PIL import Image

SYNTHETIC_SLICE = {
    "dataset": "synthetic",
    "columns": "feat_all",
    "preprocessing": "activity_no_target2",
    "filter_name": "all_sources",
    "activity_params": "default",
    "activity_threshold": 0.10,
}
SYNTHETIC_CONSISTENCY_TRAITS = (
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


def install_rdkit_stub() -> None:
    """Provide import-only RDKit shims in the lightweight test environment."""
    if "rdkit" in sys.modules:
        return
    if importlib.util.find_spec("rdkit") is not None:
        return
    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    brics = types.ModuleType("rdkit.Chem.BRICS")
    fragments = types.ModuleType("rdkit.Chem.Fragments")
    draw = types.ModuleType("rdkit.Chem.Draw")
    logger = types.SimpleNamespace(DisableLog=lambda *_: None)
    chem.MolFromSmiles = lambda value: object()
    draw.MolToImage = lambda molecule, size: Image.new("RGB", size, "white")
    chem.BRICS = brics
    chem.Fragments = fragments
    chem.Draw = draw
    rdkit.Chem = chem
    rdkit.RDLogger = logger
    sys.modules.update(
        {
            "rdkit": rdkit,
            "rdkit.Chem": chem,
            "rdkit.Chem.BRICS": brics,
            "rdkit.Chem.Fragments": fragments,
            "rdkit.Chem.Draw": draw,
        }
    )


def load_script(filename: str):
    """Import one notebook script, shimming RDKit where the test environment lacks it."""
    install_rdkit_stub()
    return load_notebook(filename)


@pytest.fixture(scope="module")
def functional_module():
    return load_script("2.42-seal-functional-use-classification.py")


@pytest.fixture(scope="module")
def structural_module():
    return load_script("2.43-seal-trait-activity-enrichment.py")


@pytest.fixture(scope="module")
def intrinsic_module():
    return load_script("2.45-seal-property-activity-oddsratios.py")


@pytest.fixture(scope="module")
def renderer_module():
    return load_script("figure-compound-activity-associations.py")


def synthetic_intrinsic_input(module, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = np.linspace(0.01, 0.99, n)
    frame = pd.DataFrame(
        {
            "Metadata_JCP2022": [f"JCP{i:04d}" for i in range(n)],
            "mean_normalized_average_precision": rng.normal(0.05, 0.08, n),
            "active": np.resize([True, False, False, True, False], n),
            "valid_mol": True,
            "pains": np.resize([True, False, False, False], n),
        }
    )
    for index, (_, column, _, _) in enumerate(module.INTRINSIC_TRAITS):
        frame[column] = values * (index + 1) + rng.normal(0, 0.03, n)
    frame["rotatable"] = np.floor(values * 8)
    frame["num_rings"] = np.floor(values * 6)
    frame["num_aromatic"] = np.floor(values * 4)
    frame["hbd"] = np.floor(values * 5)
    frame["hba"] = np.floor(values * 8)
    frame["lipinski"] = np.floor(values * 4)
    return frame


def synthetic_functional_input(module, n: int = 120) -> pd.DataFrame:
    values = np.linspace(0.01, 0.99, n)
    active = np.resize([True, False, False, True, False], n)
    frame = pd.DataFrame(
        {
            "JCP2022": [f"JCP{i:04d}" for i in range(n)],
            "functioncategory_all": np.where(np.arange(n) < n // 2, "preservative", "surfactant"),
            "is_active": active,
            "mean_normalized_average_precision": np.sin(np.arange(n) / 11) / 10,
        }
    )
    for index, column in enumerate(module.FUP_COLS):
        frame[column] = np.roll(values, index % 7)
    return frame


def synthetic_selected_categorical_input(module):
    frame = synthetic_intrinsic_input(module)
    phase = pd.Series(pd.NA, index=frame.index, dtype="string")
    phase.iloc[0:20] = "Phase 1"
    phase.iloc[20:40] = "Phase 2"
    phase.iloc[40:60] = "Phase 3"
    phase.iloc[60:80] = "Launched"
    phase.iloc[80] = "Phase 1|Phase 2"
    frame["phase"] = phase
    identifiers = frame["Metadata_JCP2022"].tolist()
    libraries = {
        "KCGS": set(identifiers[0:10]),
        "PKIS": set(identifiers[10:22]),
        "PKIS2": set(identifiers[22:36]),
        "Chemical Probes (includes SGC)": set(identifiers[36:52]),
    }
    return frame, libraries


def synthetic_structural_inputs(module):
    ids = [f"JCP{i:03d}" for i in range(100)]
    activity = pd.DataFrame(
        {
            "JCP2022": ids,
            "mean_normalized_average_precision": np.linspace(-0.1, 0.2, len(ids)),
            "is_active": [index < 20 for index in range(len(ids))],
        }
    )
    trait_maps = {}
    rows = []
    for family, *_ in module.STRUCTURAL_FIGURE_FAMILIES:
        enriched = set(ids[:10] + ids[20:30])
        depleted = set(ids[15:20] + ids[40:65])
        trait_maps[family] = {f"{family}_enriched": sorted(enriched), f"{family}_depleted": sorted(depleted)}
        for trait, members in trait_maps[family].items():
            member = activity["JCP2022"].isin(members)
            exposed_active = int((member & activity["is_active"]).sum())
            reference_active = int((~member & activity["is_active"]).sum())
            n_exposed = int(member.sum())
            n_reference = int((~member).sum())
            odds_ratio, lower, upper = module._haldane_or_ci(
                exposed_active,
                n_exposed - exposed_active,
                reference_active,
                n_reference - reference_active,
            )
            rows.append(
                {
                    "family": family,
                    "trait": trait,
                    "n_eligible": len(ids),
                    "n_missing": 0,
                    "n_with_trait": n_exposed,
                    "n_without_trait": n_reference,
                    "n_flag_with_trait": exposed_active,
                    "n_flag_without_trait": reference_active,
                    "flag_rate_with": exposed_active / n_exposed,
                    "flag_rate_background": activity["is_active"].mean(),
                    "odds_ratio": odds_ratio,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": 0.001,
                    "corrected_p_value": 0.01,
                }
            )
    return pd.DataFrame(rows), trait_maps, set(ids), activity


def synthetic_consistency_table() -> pd.DataFrame:
    rows = []
    for source_order, (source, significant_trait, significant_scope, membership_count) in enumerate(
        (
            ("source_a", "chem.logp", "between_group", 6),
            ("source_b", "derived.clinical_phase.phase_1", "within_group", 8),
        )
    ):
        for trait_order, trait in enumerate(SYNTHETIC_CONSISTENCY_TRAITS):
            clinical = trait.startswith("derived.clinical_phase.")
            n_memberships = membership_count - 2 if clinical else membership_count
            for scope, direction in (("between_group", 1), ("within_group", -1)):
                significant = trait == significant_trait and scope == significant_scope
                estimate = direction * (0.02 + 0.004 * trait_order + 0.005 * source_order)
                rows.append(
                    {
                        "source_id": source,
                        "source_order": source_order,
                        "source_label": source.replace("_", " ").title(),
                        "annotation_category": "target",
                        "trait_id": trait,
                        "trait_order": trait_order,
                        "trait_label": trait.replace("_", " ").title(),
                        "value_type": "boolean" if clinical else "continuous",
                        "unit": "indicator" if clinical else "",
                        "effect_scope": scope,
                        "status": "tested",
                        "estimate_nap_per_component_sd": estimate,
                        "standard_error_nap_per_component_sd": 0.01,
                        "ci_lower_nap_per_component_sd": estimate - 0.02,
                        "ci_upper_nap_per_component_sd": estimate + 0.02,
                        "p_value": 0.01 if significant else 0.4,
                        "q_value_global_scope": 0.02 if significant else 0.5,
                        "global_significant_at_5pct_fdr": significant,
                        "n_memberships": n_memberships,
                        "n_groups": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_intrinsic_contract_uses_equal_width_bins_numbered_by_position(intrinsic_module):
    result = intrinsic_module.build_intrinsic_activity_nmap_bins(synthetic_intrinsic_input(intrinsic_module))
    assert result["trait_id"].nunique() == 6
    # The 1-99% trim drops the extreme compounds, so bins hold nearly all of them.
    assert result.groupby("trait_id")["n"].sum().between(114, 120).all()
    assert (result["bin_lower"] < result["bin_upper"]).all()
    assert result.groupby("trait_id")["fdr_bh"].nunique().eq(1).all()
    for _, curve in result.groupby("trait_id"):
        # Bin numbers are positions in the ten-bin window, so an emptied or under-floor
        # bin leaves a gap rather than renumbering the bins above it. Position is what
        # keeps the edges monotonic across the gap.
        assert curve["bin"].between(1, 10).all()
        assert curve["bin"].is_monotonic_increasing
        assert curve["bin"].is_unique
        assert curve["bin_lower"].is_monotonic_increasing
        widths = curve["bin_upper"] - curve["bin_lower"]
        assert np.allclose(widths, widths.iloc[0])
        # Two compounds minimum, so the published standard error is never NaN.
        assert curve["n"].ge(2).all()
        assert curve["nmap_mean_se"].notna().all()


def test_functional_contract_uses_continuous_fup_inference(functional_module):
    frame = synthetic_functional_input(functional_module)
    result = functional_module.build_functional_use_activity_nmap(frame)
    row = result.loc[result["section"].eq("FUP") & result["category"].eq("Antimicrobial")].iloc[0]
    expected, *_ = functional_module.functional_logistic_or_per_iqr(frame["antimicrobial"], frame["is_active"])
    assert row["inference_estimate"] == pytest.approx(expected)
    assert row["n_exposed"] == int(frame["antimicrobial"].ge(functional_module.DEFAULT_FUP_THRESHOLD).sum())
    assert set(result["section"]) == {"FUC", "FUP"}


def test_functional_display_threshold_has_one_runtime_source(functional_module):
    frame = synthetic_functional_input(functional_module)
    threshold = 0.8
    result = functional_module.build_functional_use_activity_nmap(frame, fup_display_threshold=threshold)
    row = result.loc[result["section"].eq("FUP") & result["category"].eq("Antimicrobial")].iloc[0]
    assert row["n_exposed"] == int(frame["antimicrobial"].ge(threshold).sum())

    contract = functional_module.summarize_functional_use_activity(
        result,
        fup_display_threshold=threshold,
    )
    assert contract["fup_display_threshold"] == threshold
    assert contract["fup_display_membership"] == "Probability >= 0.8"

    displayed = result.copy()
    displayed.loc[displayed["section"].eq("FUP"), ["significant", "q_value"]] = [False, 1.0]
    selected = displayed["section"].eq("FUP") & displayed["category"].eq("Antimicrobial")
    displayed.loc[selected, ["significant", "q_value"]] = [True, 0.01]
    distributions = functional_module.build_functional_use_activity_distributions(
        frame,
        displayed,
        fup_display_threshold=threshold,
    )
    assert len(distributions.loc[distributions["section"].eq("FUP")]) == int(frame["antimicrobial"].ge(threshold).sum())


def test_consistency_fdr_threshold_has_one_runtime_source(renderer_module, structural_module, tmp_path):
    """The two consistency captions must quote the contract threshold, not a literal 5%.

    The significance flags, the summary contract, and the printed cutoff all come from
    2.43's CONSISTENCY_EFFECT_FDR_THRESHOLD, so a non-0.05 value has to reach the prose.
    """
    relationships = synthetic_consistency_table()
    threshold = 0.01

    sidebar = renderer_module.consistency_caption(relationships, threshold)
    assert "global 1% Benjamini-Hochberg FDR" in sidebar
    # "95% confidence intervals" is a different 5%, so match the FDR phrase itself.
    assert "5% Benjamini-Hochberg FDR" not in sidebar

    heatmap_paths, heatmap_captions = renderer_module.plot_consistency_heatmap(
        relationships,
        0.1,
        threshold,
        tmp_path,
        72,
    )
    heatmap = heatmap_captions[renderer_module.CONSISTENCY_HEATMAP_STEM]
    assert "white dots mark global 1% Benjamini-Hochberg FDR" in heatmap
    assert "5% Benjamini-Hochberg FDR" not in heatmap
    assert all(path.is_file() for path in heatmap_paths)

    # The upstream selection text is derived from the same constant it describes.
    assert structural_module.CONSISTENCY_DISPLAY_SELECTION.endswith(
        f"global {100 * structural_module.CONSISTENCY_EFFECT_FDR_THRESHOLD:g}% FDR"
    )


def test_selected_categorical_contract_preserves_token_membership_and_references(
    intrinsic_module,
):
    frame, libraries = synthetic_selected_categorical_input(intrinsic_module)
    result = intrinsic_module.build_selected_categorical_activity_nmap(frame, libraries)
    assert result.groupby("view").size().to_dict() == {
        "clinical_phase": 4,
        "curated_collections": 5,
    }
    phase_1 = result.loc[result["label"].eq("Phase 1")].iloc[0]
    assert phase_1["n"] == 21
    assert phase_1["n_eligible"] == 81
    assert phase_1["n"] + phase_1["reference_n"] == phase_1["n_eligible"]
    chemical_probes = result.loc[result["trait_id"].eq("collection.chemical_probes_sgc")].iloc[0]
    assert chemical_probes["n"] == 16
    assert chemical_probes["comparator"] == "Analyzed compounds outside this curated collection"


def test_compound_distribution_contract_preserves_one_row_per_compound(intrinsic_module):
    frame, libraries = synthetic_selected_categorical_input(intrinsic_module)
    result = intrinsic_module.build_compound_activity_nmap_distributions(frame, libraries)
    assert len(result) == len(frame)
    assert result["jcp2022"].is_unique
    assert result["nmap"].equals(frame["mean_normalized_average_precision"])
    assert result["in_kcgs"].sum() == len(libraries["KCGS"])


def test_structural_contract_preserves_valid_structure_reference_and_ranks(structural_module):
    enrichment, trait_maps, valid_ids, activity = synthetic_structural_inputs(structural_module)
    result = structural_module.build_structural_class_activity_nmap(enrichment, trait_maps, valid_ids, activity)
    assert result["family_key"].nunique() == 3
    assert set(result["rank_label"]) == {"E1", "D1"}
    assert result["n_eligible"].eq(len(valid_ids)).all()
    assert (result["n_exposed"] + result["n_reference"]).eq(len(valid_ids)).all()
    assert (
        result.groupby([result["family_key"], result["rank_label"].str[0]])
        .size()
        .le(structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION)
        .all()
    )


def test_consistency_group_audit_retains_one_row_per_group_and_trait(structural_module):
    member_data = pd.DataFrame(
        {
            "annotation_source": "repurposing",
            "annotation_group": "target_a",
            "jcp2022": [f"JCP{i:03d}" for i in range(5)],
            "normalized_average_precision": np.linspace(-0.1, 0.2, 5),
            "activity_nmap": np.linspace(0.0, 0.4, 5),
        }
    )
    for index, trait_id in enumerate(structural_module.CONSISTENCY_TRAIT_IDS):
        member_data[trait_id] = np.arange(5, dtype=float) + index
    group_results = pd.DataFrame(
        {
            "annotation_source": ["repurposing"],
            "annotation_source_label": ["Repurposing Hub targets"],
            "annotation_category": ["target"],
            "annotation_source_order": [0],
            "annotation_group": ["target_a"],
            "upstream_mean_normalized_average_precision": [0.05],
            "upstream_corrected_p_value": [0.2],
            "upstream_below_corrected_p": [False],
            "upstream_score_rows": [5],
            "upstream_unique_compounds": [5],
            "duplicate_score_rows": [0],
        }
    )
    result = structural_module.build_consistency_group_properties(member_data, group_results)
    assert len(result) == len(structural_module.CONSISTENCY_TRAIT_IDS)
    assert result["n_membership_compounds"].eq(5).all()
    assert result["n_property_compounds"].eq(5).all()
    assert result["eligible_group_primary"].all()
    assert result.loc[result["trait_id"].eq("chem.logp"), "trait_label"].item() == "Calculated LogP"


def test_category_slots_keep_reference_levels_the_cohort_lacks(renderer_module):
    empty = pd.DataFrame({"nmap": []})
    reference = [(label, pd.DataFrame({"nmap": [0.1]})) for label in ("0", "1", "2", "3", "4")]
    # The cohort reaches only two violations; the three drawers share an x axis, so the
    # missing levels have to survive as empty slots or they crop the screen's boxes.
    cohort = [(label, pd.DataFrame({"nmap": [0.2, 0.3]})) for label in ("0", "1", "2")]
    aligned = renderer_module.align_category_slots(reference, cohort, empty)
    assert [label for label, _ in aligned] == ["0", "1", "2", "3", "4"]
    assert [len(frame) for _, frame in aligned] == [2, 2, 2, 0, 0]
    # A level the cohort has but the reference does not is dropped, so every drawer of a
    # column keeps the same slot count.
    extra = renderer_module.align_category_slots(reference, [*cohort, ("9", pd.DataFrame({"nmap": [0.4]}))], empty)
    assert [label for label, _ in extra] == ["0", "1", "2", "3", "4"]


def test_consistency_selection_keeps_both_scopes_when_either_is_significant(renderer_module):
    paired = renderer_module.select_consistency_pairs(synthetic_consistency_table())
    assert len(paired) == 4
    assert paired.groupby(["source_id", "trait_id"])["effect_scope"].nunique().eq(2).all()
    assert paired.groupby(["source_id", "trait_id"])["global_significant_at_5pct_fdr"].sum().eq(1).all()


def test_consistency_labels_put_property_before_annotation_class(renderer_module):
    assert (
        renderer_module.compact_consistency_label(
            "MOTIVE BioKG targets",
            "Normalized post-treatment Cell Painting cell density",
        )
        == "Cell density | BioKG"
    )
    assert (
        renderer_module.compact_consistency_label(
            "MOTIVE OpenTargets targets",
            "Rotatable bonds",
        )
        == "Rotatable bonds | OpenTargets"
    )


def test_consistency_effect_colors_match_heatmap_normalization(renderer_module):
    relationships = synthetic_consistency_table()
    estimates = np.array([-0.04, 0.08])
    color_limit = renderer_module.consistency_heatmap_color_limit(relationships)
    assert color_limit == pytest.approx(0.08)
    expected = renderer_module.plt.colormaps[renderer_module.CONSISTENCY_COLORMAP](
        (estimates + color_limit) / (2 * color_limit)
    )
    actual = renderer_module.consistency_effect_colors(relationships, estimates)
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 0] > actual[0, 2]
    assert actual[1, 2] > actual[1, 0]


def test_consistency_sidebar_artists_use_heatmap_effect_colors(renderer_module):
    relationships = synthetic_consistency_table()
    negative = (
        relationships["source_id"].eq("source_b")
        & relationships["trait_id"].eq("derived.clinical_phase.phase_1")
        & relationships["effect_scope"].eq("between_group")
    )
    relationships.loc[
        negative,
        [
            "estimate_nap_per_component_sd",
            "ci_lower_nap_per_component_sd",
            "ci_upper_nap_per_component_sd",
            "global_significant_at_5pct_fdr",
        ],
    ] = [-0.04, -0.06, -0.02, True]
    nondisplayed_extreme = (
        relationships["source_id"].eq("source_a")
        & relationships["trait_id"].eq("chem.molecular_weight")
        & relationships["effect_scope"].eq("within_group")
    )
    relationships.loc[nondisplayed_extreme, "estimate_nap_per_component_sd"] = 0.193
    group_properties = pd.DataFrame(
        [
            {
                "annotation_source": source,
                "annotation_group": group,
                "trait_id": trait,
                "n_property_compounds": 6,
                "property_mean": 0.2 if trait.startswith("derived.clinical_phase.") else 1.0,
                "eligible_group_primary": True,
            }
            for source, group in (("source_a", "group_a"), ("source_b", "group_b"))
            for trait in SYNTHETIC_CONSISTENCY_TRAITS
        ]
    )
    assert renderer_module.consistency_heatmap_color_limit(relationships) == pytest.approx(0.20)

    figure = renderer_module.plt.figure(figsize=(4.2, 5.8))
    try:
        displayed = renderer_module.draw_consistency_sidebar(figure, relationships, group_properties, "", 0.0, 5.8)
        estimates = displayed["estimate_nap_per_component_sd"].to_numpy(dtype=float)
        expected_colors = renderer_module.consistency_effect_colors(relationships, estimates)
        effect_axis = figure.axes[0]
        np.testing.assert_allclose(
            np.array([bar.get_facecolor() for bar in effect_axis.patches]),
            expected_colors,
        )
        np.testing.assert_allclose(
            np.array([bar.get_edgecolor() for bar in effect_axis.patches]),
            expected_colors,
        )
        errorbars = [
            container for container in effect_axis.containers if container.__class__.__name__ == "ErrorbarContainer"
        ]
        assert len(errorbars) == len(expected_colors)
        for container, expected_color in zip(errorbars, expected_colors, strict=True):
            _, caplines, barlinecols = container.lines
            for capline in caplines:
                np.testing.assert_allclose(
                    renderer_module.matplotlib.colors.to_rgba(capline.get_color()),
                    expected_color,
                )
            for barlinecol in barlinecols:
                np.testing.assert_allclose(barlinecol.get_colors(), expected_color[np.newaxis, :])
    finally:
        renderer_module.plt.close(figure)


def write_target_alignment_contract(directory: Path, renderer_module) -> None:
    """Write a synthetic 2.46 package: memberships, target summary, and fitted curve."""
    directory.mkdir(parents=True)
    targets = ["TARGET_A", "TARGET_B", "TARGET_C"]
    rows = []
    for target_index, target in enumerate(targets):
        for member in range(6):
            rows.append(
                {
                    # Ids must fall inside the synthetic 2.45 compound universe (JCP0000-JCP0119).
                    "jcp2022": f"JCP{target_index * 6 + member:04d}",
                    "target_label": target,
                    # Every target keeps one compound exactly at the activity ceiling.
                    "activity_nmap": 1.0 if member == 0 else 0.2 + 0.1 * member,
                    "target_alignment_nap": 0.05 * target_index + 0.01 * member,
                    "joint_set_id": f"set_{target}",
                    "eligible_anchor_4": True,
                }
            )
    memberships = pd.DataFrame(rows)
    memberships.to_csv(directory / renderer_module.ALIGNMENT_MEMBERSHIP_TABLE, index=False)
    target_summary = (
        memberships.groupby("target_label", sort=True)
        .agg(
            mean_activity_nmap=("activity_nmap", "mean"),
            mean_consistency_nap=("target_alignment_nap", "mean"),
            query_count=("jcp2022", "nunique"),
            joint_set_id=("joint_set_id", "first"),
        )
        .reset_index()
    )
    target_summary.to_csv(directory / renderer_module.ALIGNMENT_TARGET_TABLE, index=False)
    grid = np.linspace(target_summary["mean_activity_nmap"].min(), target_summary["mean_activity_nmap"].max(), 16)
    pd.DataFrame(
        {
            "mean_activity_nmap": grid,
            "mean_consistency_fit": 0.1 * grid,
            "ci_lower": 0.1 * grid - 0.02,
            "ci_upper": 0.1 * grid + 0.02,
        }
    ).to_csv(directory / renderer_module.ALIGNMENT_CURVE_TABLE, index=False)
    compounds = memberships.loc[memberships.groupby("jcp2022")["target_alignment_nap"].idxmax()].copy()
    compounds = compounds.rename(
        columns={"target_label": "best_target_label", "target_alignment_nap": "best_target_alignment_nap"}
    )
    compounds["n_retained_targets"] = 1
    for index, (column, _, _) in enumerate(renderer_module.COMPOUND_PROPERTY_TRAITS):
        compounds[column] = np.linspace(1.0, 5.0, len(compounds)) * (index + 1)
    compounds["best_target_consistent"] = np.resize([True, False], len(compounds))
    compounds["activity_nmap"] = np.linspace(0.2, 1.0, len(compounds))
    compounds["activity_corrected_p_value"] = np.resize([0.01, 0.5], len(compounds))
    compounds.to_csv(directory / renderer_module.ALIGNMENT_COMPOUND_TABLE, index=False)
    bin_rows = []
    for trait_order, (column, title, xlabel) in enumerate(renderer_module.COMPOUND_PROPERTY_TRAITS):
        assignments = pd.qcut(compounds[column].rank(method="first"), 3, labels=False)
        for index, block in compounds.groupby(assignments, sort=True):
            successes = int(block["best_target_consistent"].sum())
            bin_rows.append(
                {
                    "trait_order": trait_order,
                    "trait_id": column,
                    "title": title,
                    "xlabel": xlabel,
                    "bin": int(index) + 1,
                    "x": float(block[column].median()),
                    "bin_lower": float(block[column].min()),
                    "bin_upper": float(block[column].max()),
                    "n": int(len(block)),
                    "rate": successes / len(block),
                    "lower": max(0.0, successes / len(block) - 0.2),
                    "upper": min(1.0, successes / len(block) + 0.2),
                    "nap_mean": float(block["best_target_alignment_nap"].mean()),
                    "nap_mean_se": float(block["best_target_alignment_nap"].std(ddof=1) / len(block) ** 0.5),
                    "rho": 0.5 - 0.1 * trait_order,
                    "rho_q_value": 0.02 + 0.01 * trait_order,
                    "fit_slope": 0.01 * (trait_order + 1),
                    "fit_intercept": 0.05,
                    "rate_active": 0.6,
                    "lower_active": 0.4,
                    "upper_active": 0.8,
                    "activity_nmap_mean": float(block["activity_nmap"].mean()),
                    "activity_nmap_mean_se": float(block["activity_nmap"].std(ddof=1) / len(block) ** 0.5),
                    "rho_activity": -0.3 + 0.1 * trait_order,
                    "rho_activity_q_value": 0.0005 + 0.05 * trait_order,
                    "fit_activity_slope": -0.01 * (trait_order + 1),
                    "fit_activity_intercept": 0.6,
                }
            )
    pd.DataFrame(bin_rows).to_csv(directory / renderer_module.ALIGNMENT_PROPERTY_BIN_TABLE, index=False)
    pd.DataFrame(
        [
            {
                "trait_order": order,
                "trait_id": column,
                "title": title,
                "n_compounds": len(compounds),
                "n_targets": 3,
                "rho_property_activity_compound": -0.2 + 0.05 * order,
                "rho_property_consistency_compound": 0.1 - 0.02 * order,
                "rho_activity_consistency_compound": 0.29,
                "rho_property_consistency_mediated_prediction": -0.06,
                "partial_property_consistency_given_activity": 0.15,
                "rho_property_consistency_target": 0.43,
                "aggregation_inflation": 5.6,
                "null_mean_abs_rho_target": 0.25,
                "null_p95_abs_rho_target": 0.47,
                "empirical_p_target": 0.10,
            }
            for order, (column, title, _) in enumerate(renderer_module.COMPOUND_PROPERTY_TRAITS)
        ]
    ).to_csv(directory / renderer_module.ALIGNMENT_GRAIN_AUDIT_TABLE, index=False)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "description": "synthetic test contract",
                "numbers": {
                    "across_target_relationship": {
                        "spearman_all_target_labels": 0.61,
                        "spearman_distinct_exact_sets": 0.48,
                    },
                    "scatter_panel_relationship": {
                        "memberships": 402,
                        "compounds": 168,
                        "spearman_memberships": 0.28,
                        "spearman_compounds": 0.29,
                    },
                    "across_target_aggregation_null": {
                        "iterations": 2000,
                        "null_mean_abs_rho": 0.24,
                        "empirical_p_value": 0.0024,
                    },
                },
                "files": [
                    {"path": renderer_module.ALIGNMENT_MEMBERSHIP_TABLE},
                    {"path": renderer_module.ALIGNMENT_TARGET_TABLE},
                    {"path": renderer_module.ALIGNMENT_CURVE_TABLE},
                    {"path": renderer_module.ALIGNMENT_COMPOUND_TABLE},
                    {"path": renderer_module.ALIGNMENT_PROPERTY_BIN_TABLE},
                    {"path": renderer_module.ALIGNMENT_GRAIN_AUDIT_TABLE},
                ],
            }
        )
        + "\n"
    )


def write_contract(directory: Path, filename: str, frame: pd.DataFrame, numbers: dict) -> None:
    directory.mkdir(parents=True)
    frame.to_csv(directory / filename, index=False)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "description": "synthetic test contract",
                "numbers": numbers,
                "files": [{"path": filename, "kind": "table", "caption": filename}],
            }
        )
        + "\n"
    )


def test_renderer_refuses_an_undeclared_table(renderer_module, tmp_path):
    directory = tmp_path / "analysis"
    directory.mkdir()
    pd.DataFrame({"value": [1]}).to_csv(directory / renderer_module.INTRINSIC_TABLE, index=False)
    (directory / "summary.json").write_text('{"description":"x","numbers":{},"files":[]}\n')
    with pytest.raises(ValueError, match="does not declare required table"):
        renderer_module.load_analysis_table(directory, renderer_module.INTRINSIC_TABLE)


def test_renderer_refuses_mismatched_analysis_slices(renderer_module):
    functional = {"numbers": SYNTHETIC_SLICE | {"n_canonical_activity_compounds": 120}}
    trait = {"numbers": SYNTHETIC_SLICE | {"n_activity_compounds": 120, "n_active": 48}}
    properties = {"numbers": SYNTHETIC_SLICE | {"n_compounds": 120, "n_active": 48}}
    trait["numbers"]["dataset"] = "different"
    with pytest.raises(ValueError, match="disagree on dataset"):
        renderer_module.validate_analysis_summaries(functional, trait, properties, "synthetic")


def test_direct_renderer_writes_composite_modular_and_separate_exports(
    functional_module,
    structural_module,
    intrinsic_module,
    renderer_module,
    tmp_path,
):
    intrinsic = intrinsic_module.build_intrinsic_activity_nmap_bins(synthetic_intrinsic_input(intrinsic_module))
    selected_input, libraries = synthetic_selected_categorical_input(intrinsic_module)
    selected_categorical = intrinsic_module.build_selected_categorical_activity_nmap(selected_input, libraries)
    compound_distributions = intrinsic_module.build_compound_activity_nmap_distributions(selected_input, libraries)
    functional_input = synthetic_functional_input(functional_module)
    functional = functional_module.build_functional_use_activity_nmap(functional_input)
    fup_rows = functional.index[functional["section"].eq("FUP")]
    functional.loc[fup_rows, ["significant", "q_value"]] = [False, 1.0]
    selected_fup = fup_rows[:2]
    functional.loc[selected_fup, "inference_estimate"] = [1.5, 0.5]
    functional.loc[selected_fup, ["significant", "q_value"]] = [True, 0.01]
    functional_distributions = functional_module.build_functional_use_activity_distributions(
        functional_input, functional
    )
    functional_use_activity_contract = functional_module.summarize_functional_use_activity(
        functional,
        fup_display_threshold=functional_module.DEFAULT_FUP_THRESHOLD,
    )
    enrichment, trait_maps, valid_ids, activity = synthetic_structural_inputs(structural_module)
    structural = structural_module.build_structural_class_activity_nmap(enrichment, trait_maps, valid_ids, activity)
    structural_distributions = structural_module.build_structural_class_activity_distributions(
        structural, trait_maps, valid_ids, activity
    )
    structural_all_significant = structural_module.build_structural_classes_all_significant(enrichment, structural)
    consistency = synthetic_consistency_table()
    consistency_groups = pd.DataFrame(
        [
            {
                "annotation_source": source,
                "annotation_group": group,
                "trait_id": trait,
                "n_membership_compounds": membership_count,
                "n_property_compounds": (
                    membership_count - 2 if trait.startswith("derived.clinical_phase.") else membership_count
                ),
                "property_mean": float(
                    membership_count - 2 if trait.startswith("derived.clinical_phase.") else membership_count
                ),
                "eligible_group_primary": True,
            }
            for source, group, membership_count in (
                ("source_a", "group_a", 6),
                ("source_b", "group_b", 8),
            )
            for trait in SYNTHETIC_CONSISTENCY_TRAITS
        ]
    )
    functional_dir = tmp_path / "functional"
    trait_dir = tmp_path / "trait"
    property_dir = tmp_path / "property"
    alignment_dir = tmp_path / "alignment"
    output_dir = tmp_path / "figure"
    write_target_alignment_contract(alignment_dir, renderer_module)
    write_contract(
        functional_dir,
        renderer_module.FUNCTIONAL_TABLE,
        functional,
        SYNTHETIC_SLICE
        | {
            "n_canonical_activity_compounds": 120,
            "n_with_curated_fuc_in_analysis": int(
                functional.loc[functional["section"].eq("FUC"), "n_exposed"].iloc[0]
                + functional.loc[functional["section"].eq("FUC"), "n_reference"].iloc[0]
            ),
            "n_with_fup_prediction_in_analysis": int(
                functional_input[list(functional_module.FUP_COLS)].notna().any(axis=1).sum()
            ),
            "functional_use_activity_nmap": functional_use_activity_contract,
        },
    )
    functional_distributions.to_parquet(functional_dir / renderer_module.FUNCTIONAL_DISTRIBUTION_TABLE, index=False)
    functional_contract = json.loads((functional_dir / "summary.json").read_text())
    functional_contract["files"].append({"path": renderer_module.FUNCTIONAL_DISTRIBUTION_TABLE})
    (functional_dir / "summary.json").write_text(json.dumps(functional_contract) + "\n")
    trait_dir.mkdir()
    structural.to_csv(trait_dir / renderer_module.STRUCTURAL_TABLE, index=False)
    structural_distributions.to_parquet(trait_dir / renderer_module.STRUCTURAL_DISTRIBUTION_TABLE, index=False)
    structural_all_significant.to_csv(trait_dir / renderer_module.STRUCTURAL_ALL_SIGNIFICANT_TABLE, index=False)
    consistency.to_csv(trait_dir / renderer_module.CONSISTENCY_TABLE, index=False)
    consistency_groups.to_csv(trait_dir / renderer_module.CONSISTENCY_GROUP_TABLE, index=False)
    (trait_dir / "summary.json").write_text(
        json.dumps(
            {
                "description": "synthetic test contract",
                "numbers": SYNTHETIC_SLICE
                | {
                    "n_activity_compounds": 120,
                    "n_active": 48,
                    "structural_class_activity_nmap": {
                        "selection": (
                            f"Up to {structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION} significant enriched "
                            f"and {structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION} significant depleted "
                            "relationships per family"
                        ),
                        "maximum_rows_per_direction": structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION,
                        "fdr_threshold": structural_module.STRUCTURAL_FDR_THRESHOLD,
                        "minimum_members_per_group": structural_module.STRUCTURAL_MIN_MEMBERS,
                        "minimum_contingency_cell_count": structural_module.STRUCTURAL_MIN_CONTINGENCY_CELL,
                        "test": "Two-sided Fisher exact test",
                        "multiple_testing": ("Benjamini-Hochberg separately within each structural family"),
                        "eligible_population": "Compounds with Metadata_ValidMol=true",
                        "reference_population": "Other compounds with Metadata_ValidMol=true",
                    },
                    "consistency_property_effects": {
                        "fdr_threshold": structural_module.CONSISTENCY_EFFECT_FDR_THRESHOLD,
                        "model": (
                            "Inverse-group-size contextual WLS with between- and within-group property components"
                        ),
                        "covariance": structural_module.CONSISTENCY_COVARIANCE,
                        "multiple_testing": (
                            "Benjamini-Hochberg separately across all tested effects within each scope"
                        ),
                    },
                    "consistency_group_properties": {
                        "n_rows": 26,
                        "n_groups": 2,
                        "n_traits": 13,
                        "grain": "one annotation-source group by property row",
                    },
                },
                "files": [
                    {"path": renderer_module.STRUCTURAL_TABLE},
                    {"path": renderer_module.STRUCTURAL_DISTRIBUTION_TABLE},
                    {"path": renderer_module.STRUCTURAL_ALL_SIGNIFICANT_TABLE},
                    {"path": renderer_module.CONSISTENCY_TABLE},
                    {"path": renderer_module.CONSISTENCY_GROUP_TABLE},
                ],
            }
        )
        + "\n"
    )
    property_dir.mkdir()
    intrinsic.to_csv(property_dir / renderer_module.INTRINSIC_TABLE, index=False)
    selected_categorical.to_csv(property_dir / renderer_module.CATEGORICAL_TABLE, index=False)
    compound_distributions.to_parquet(property_dir / renderer_module.COMPOUND_DISTRIBUTION_TABLE, index=False)
    (property_dir / "summary.json").write_text(
        json.dumps(
            {
                "description": "synthetic test contract",
                "numbers": SYNTHETIC_SLICE
                | {
                    "background_active_rate": 0.4,
                    "n_compounds": 120,
                    "n_active": 48,
                },
                "files": [
                    {"path": renderer_module.INTRINSIC_TABLE},
                    {"path": renderer_module.CATEGORICAL_TABLE},
                    {"path": renderer_module.COMPOUND_DISTRIBUTION_TABLE},
                ],
            }
        )
        + "\n"
    )

    renderer_module.main(
        dataset="synthetic",
        functional_use_dir=functional_dir,
        trait_enrichment_dir=trait_dir,
        property_oddsratios_dir=property_dir,
        target_alignment_dir=alignment_dir,
        output_dir=output_dir,
        dpi=30,
    )

    for stem in renderer_module.FIGURE_OUTPUT_STEMS:
        assert (output_dir / f"{stem}.png").is_file()
        assert (output_dir / f"{stem}.pdf").is_file()
    with Image.open(output_dir / "compound_properties_activity.png") as image:
        assert image.size[0] == 288
    with Image.open(output_dir / "intrinsic_properties.png") as image:
        assert image.size[0] == 288
    # Widths are the exact declared canvas; heights exceed it by the caption band, which grows
    # the canvas downward rather than taking height from the panels.
    with Image.open(output_dir / "lipinski_violations.png") as image:
        assert image.size[0] == 180
        assert image.size[1] > 135
    with Image.open(output_dir / "phenotypic_consistency.png") as image:
        assert image.size[0] == 126
        assert image.size[1] > 174
    with Image.open(output_dir / "phenotypic_consistency_heatmap.png") as image:
        assert image.size[0] == 360
        assert image.size[1] > 282
    source_directories = {
        renderer_module.INTRINSIC_TABLE: property_dir,
        renderer_module.CATEGORICAL_TABLE: property_dir,
        renderer_module.COMPOUND_DISTRIBUTION_TABLE: property_dir,
        renderer_module.STRUCTURAL_TABLE: trait_dir,
        renderer_module.STRUCTURAL_DISTRIBUTION_TABLE: trait_dir,
        renderer_module.STRUCTURAL_ALL_SIGNIFICANT_TABLE: trait_dir,
        renderer_module.FUNCTIONAL_TABLE: functional_dir,
        renderer_module.FUNCTIONAL_DISTRIBUTION_TABLE: functional_dir,
        renderer_module.CONSISTENCY_TABLE: trait_dir,
        renderer_module.CONSISTENCY_GROUP_TABLE: trait_dir,
        renderer_module.ALIGNMENT_MEMBERSHIP_TABLE: alignment_dir,
        renderer_module.ALIGNMENT_TARGET_TABLE: alignment_dir,
        renderer_module.ALIGNMENT_CURVE_TABLE: alignment_dir,
        renderer_module.ALIGNMENT_COMPOUND_TABLE: alignment_dir,
        renderer_module.ALIGNMENT_PROPERTY_BIN_TABLE: alignment_dir,
        renderer_module.ALIGNMENT_GRAIN_AUDIT_TABLE: alignment_dir,
    }
    for filename, source_directory in source_directories.items():
        assert (output_dir / filename).read_bytes() == (source_directory / filename).read_bytes()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["numbers"]["figure_exports"] == list(renderer_module.FIGURE_OUTPUT_STEMS)
    assert summary["numbers"]["outputs_by_role"] == renderer_module.OUTPUTS_BY_ROLE
    assert summary["numbers"]["composite_members"] == list(renderer_module.MODULAR_PANEL_STEMS)
    assert renderer_module.CONSISTENCY_HEATMAP_STEM in summary["numbers"]["outputs_by_role"]["separate_figure"]
    assert renderer_module.CONSISTENCY_HEATMAP_STEM not in summary["numbers"]["composite_members"]
    assert summary["numbers"]["analyzed_compounds"] == 120
    assert summary["numbers"]["phenotypically_active_compounds"] == 48
    assert summary["numbers"]["clinical_phase_rows"] == 4
    assert summary["numbers"]["curated_collection_rows"] == 5
    assert summary["numbers"]["chemical_probes_including_sgc_members"] == 16
    assert summary["numbers"]["structural_significant_rows_full"] == 6
    assert summary["numbers"]["structural_selected_rows"] == 6
    structural_contract = summary["numbers"]["structural_class_contract"]
    assert structural_contract["n_significant_total"] == len(structural_all_significant)
    assert structural_contract["n_selected"] == int(structural_all_significant["selected_for_figure"].sum())
    assert structural_contract["maximum_rows_per_direction"] == structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION
    assert structural_contract["fdr_threshold"] == structural_module.STRUCTURAL_FDR_THRESHOLD
    assert structural_contract["eligible_population"] == "Compounds with Metadata_ValidMol=true"
    assert structural_contract["reference_population"] == "Other compounds with Metadata_ValidMol=true"
    assert summary["numbers"]["functional_fuc_tested"] == 2
    assert summary["numbers"]["functional_fuc_significant"] == 0
    assert summary["numbers"]["functional_fup_significant"] == 2
    functional_contract = summary["numbers"]["functional_use_contract"]
    significant_fup = functional.loc[functional["section"].eq("FUP") & functional["significant"].astype(bool)]
    assert functional_contract["predicted_fup"]["n_significant"] == len(significant_fup)
    assert functional_contract["predicted_fup"]["n_higher"] == int(significant_fup["inference_estimate"].gt(1).sum())
    assert functional_contract["predicted_fup"]["n_lower"] == int(significant_fup["inference_estimate"].lt(1).sum())
    assert functional_contract["predicted_fup"]["display_threshold"] == functional_module.DEFAULT_FUP_THRESHOLD
    highest_rate = significant_fup.loc[significant_fup["rate_exposed"].idxmax()]
    assert functional_contract["predicted_fup"]["highest_display_active_rate"] == {
        "endpoint": highest_rate["source_level"],
        "active": int(highest_rate["active_exposed"]),
        "n": int(highest_rate["n_exposed"]),
        "active_percent": pytest.approx(float(highest_rate["rate_exposed"])),
    }
    assert summary["numbers"]["compound_consistency_compounds"] == 18
    assert set(summary["numbers"]["compound_cohort_activity_spearman_by_property"]) == {
        column for column, _, _ in renderer_module.COMPOUND_PROPERTY_TRAITS
    }
    assert set(summary["numbers"]["compound_consistency_spearman_by_property"]) == {
        column for column, _, _ in renderer_module.COMPOUND_PROPERTY_TRAITS
    }
    assert summary["numbers"]["target_alignment_memberships"] == 18
    assert summary["numbers"]["target_alignment_targets"] == 3
    assert summary["numbers"]["target_alignment_ceiling_compounds"] == 3
    assert summary["numbers"]["target_alignment_ceiling_memberships"] == 3
    assert summary["numbers"]["target_alignment_spearman_all_target_labels"] == 0.61
    legend_text = (output_dir / "legend.md").read_text()
    # Exactly one section per rendered figure, each naming its file, each bodied by the caption
    # the renderer drew.
    assert re.findall(r"^## .*\(`([^`]+)`\)$", legend_text, flags=re.MULTILINE) == [
        f"{stem}.png" for stem in renderer_module.FIGURE_OUTPUT_STEMS
    ]
    assert f"(`{renderer_module.TARGET_ALIGNMENT_STEM}.png`)" in legend_text
    section_bodies = re.findall(r"^## .*?\n\n(.*?)(?=\n\n## |\Z)", legend_text, flags=re.MULTILINE | re.DOTALL)
    assert section_bodies
    assert max(len(body.split()) for body in section_bodies) <= 250
    assert renderer_module.ALIGNMENT_GRAIN_AUDIT_TABLE in legend_text
    assert summary["numbers"]["consistency_source_trait_pairs"] == 1
    assert summary["numbers"]["consistency_displayed_scope_estimates"] == 1
    assert summary["numbers"]["consistency_displayed_between_group_effects"] == 1
    assert summary["numbers"]["consistency_between_effects_tested"] == 26
    assert summary["numbers"]["consistency_within_effects_tested"] == 26
    assert summary["numbers"]["consistency_between_effects_significant_global_5pct_fdr"] == 1
    assert summary["numbers"]["consistency_within_effects_significant_global_5pct_fdr"] == 1
    assert summary["numbers"]["consistency_group_audit_rows"] == 26
    assert summary["numbers"]["consistency_annotation_sources"] == 2
    assert summary["numbers"]["consistency_groups"] == 2
    assert summary["numbers"]["consistency_unique_memberships_total"] == 14
    assert summary["numbers"]["consistency_trait_eligible_memberships_min"] == 10
    assert summary["numbers"]["consistency_trait_eligible_memberships_max"] == 14
    assert summary["numbers"]["consistency_clinical_stage_eligible_memberships"] == 10
    assert summary["numbers"]["consistency_modeled_memberships_min"] == 10
    assert summary["numbers"]["consistency_modeled_memberships_max"] == 14
    assert summary["numbers"]["consistency_modeled_groups_min"] == 2
    assert summary["numbers"]["consistency_modeled_groups_max"] == 2
    assert (
        summary["numbers"]["consistency_heatmap_display_contract"]["fdr_threshold"]
        == structural_module.CONSISTENCY_EFFECT_FDR_THRESHOLD
    )
    declared = {item["path"]: item["kind"] for item in summary["files"]}
    assert declared["phenotypic_consistency_heatmap.png"] == "figure"
    assert declared["phenotypic_consistency_heatmap.pdf"] == "figure"
    # Captions quote the runtime thresholds and selection limits rather than literals.
    assert "probability >= 0.5" in legend_text
    assert "Compounds pass the upstream activity filter (corrected p < 0.1)" in legend_text
    assert (
        f"Up to {structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION} Higher (blue) and "
        f"{structural_module.STRUCTURAL_FIGURE_MAX_PER_DIRECTION} Lower (orange)"
    ) in legend_text
    assert all(not Path(item["path"]).is_absolute() for item in summary["files"])


def test_activity_ceiling_is_counted_by_compound_and_membership(renderer_module):
    """The ceiling marginal counts unique compounds separately from their memberships."""
    rows = [{"jcp2022": f"LOW{index:02d}", "target_label": "A", "activity_nmap": index / 20} for index in range(20)]
    rows.extend({"jcp2022": f"TOP{index:02d}", "target_label": "A", "activity_nmap": 1.0} for index in range(5))
    rows.extend({"jcp2022": f"TOP{index:02d}", "target_label": "B", "activity_nmap": 1.0} for index in range(5))

    memberships = pd.DataFrame(rows)
    ceiling, counts = renderer_module.assign_activity_ceiling(memberships)

    assert counts == {"total_compounds": 25, "ceiling_compounds": 5, "ceiling_memberships": 10}
    assert ceiling["activity_nmap"].eq(1.0).all()
    assert len(ceiling) == 10
    memberships["joint_set_id"] = memberships["target_label"]
    spearman = {
        "spearman_memberships": 0.28,
        "spearman_compounds": 0.29,
        "spearman_all_target_labels": 0.61,
        "spearman_distinct_exact_sets": 0.48,
        "null_mean_abs_rho": 0.24,
        "null_empirical_p": 0.0024,
        "null_iterations": 2000,
    }
    caption = renderer_module.target_alignment_caption(memberships, memberships.head(25), counts, spearman, 0.1)
    assert "Top, 30 compound-target memberships from 25 compounds" in caption
    assert "Middle, the same 25 compounds reduced to one point each" in caption
    assert "Bottom, target-level mean activity nMAP and mean alignment nAP" in caption
    assert "5 of 25 compounds" in caption
    assert "10 memberships" in caption
    # The three grains and the null that keeps the third from being over-read.
    assert "Spearman rho is 0.28 per membership, 0.29 per compound, 0.61 across all target labels" in caption
    assert "0.48 after retaining one label per distinct membership/reference set" in caption
    assert "mean absolute rho 0.24" in caption
    assert "empirical p was 0.002 over 2000 permutations" in caption
    # The title lives in the legend section heading, so the caption must not repeat it.
    assert renderer_module.SECTION_TITLES[renderer_module.TARGET_ALIGNMENT_STEM] not in caption
    assert "\n" not in caption
