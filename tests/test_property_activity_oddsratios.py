"""Protect the statistical and workflow contracts for analysis 2.45."""

import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from jump_production.queries import query_activity_results

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "2.45-seal-property-activity-oddsratios.py"

SPEC = importlib.util.spec_from_file_location("property_activity_oddsratios", NOTEBOOK)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def functional_use_frame(category: str, counts: tuple[int, int, int, int]) -> pd.DataFrame:
    """Build one valid 2.42-style enrichment row from a, b, c, d counts."""
    a, b, c, d = counts
    odds_ratio, ci_lower, ci_upper = MODULE._haldane_or_ci(a, b, c, d)
    _, p_value = MODULE.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return pd.DataFrame(
        [
            {
                "category": category,
                "threshold": 0.5,
                "n_eligible": a + b + c + d,
                "n_missing": 5,
                "n_high": a + b,
                "n_rest": c + d,
                "n_active_high": a,
                "n_active_rest": c,
                "odds_ratio": odds_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
                "corrected_p_value": 0.99,
            }
        ]
    )


def test_binary_odds_ratio_excludes_ineligible_missing_values():
    member = pd.Series([True, False, pd.NA], dtype="boolean")
    active = pd.Series([True, False, True], dtype=bool)
    eligible = member.notna()

    result = MODULE.odds_ratio_row(
        "assessed trait",
        "test family",
        member,
        active,
        eligible=eligible,
    )

    assert result["n_eligible"] == 2
    assert result["n_excluded"] == 1
    assert result["n_member"] == 1
    assert result["n_reference"] == 1
    assert result["odds_ratio"] == pytest.approx(9.0)
    assert result["ci_lower"] < result["odds_ratio"] < result["ci_upper"]


def test_clinical_phases_use_exact_pipe_tokens_and_family_bh():
    compounds = [f"JCP{i}" for i in range(8)]
    frame = pd.DataFrame(
        {
            "Metadata_JCP2022": compounds,
            "active": [True, False, True, False, True, False, True, False],
            "phase": [
                "Launched",
                "Phase 3",
                "Phase 2/Phase 3",
                "Phase 2",
                "Phase 1/Phase 2",
                "Phase 1",
                "Withdrawn",
                "Preclinical",
            ],
            "valid_mol": True,
            "lipinski": [0, 1, 2, 3, 0, 1, 2, 3],
            "pains": [False, True, False, True, False, True, False, True],
            "mw": [400, 600] * 4,
            "logp": [4, 6] * 4,
            "hbd": [4, 6] * 4,
            "hba": [9, 11] * 4,
            "num_rings": [0, 1] * 4,
            "scaffold": ["A", "A", "B", "B", "C", "C", "D", "E"],
        }
    )
    result = MODULE.build_binary_traits(
        frame,
        annotation_sets={"annotation": {"JCP0", "JCP1"}},
        library_sets={"library": {"JCP0", "JCP1"}},
    )
    clinical = result.loc[result["category"].eq("Clinical development")].set_index("trait")

    assert clinical.loc["Launched", "n_member"] == 1
    assert clinical.loc["Phase 3", "n_member"] == 1
    assert clinical.loc["Phase 2", "n_member"] == 1
    assert clinical.loc["Phase 1", "n_member"] == 1
    assert clinical.loc["Any clinical (Phase 1-Launched)", "n_member"] == 6
    assert clinical.loc["Withdrawn drug", "n_member"] == 1
    expected_bh = multipletests(clinical["p_value"], method="fdr_bh")[1]
    assert np.allclose(clinical["corrected_p_value"], expected_bh)


def test_multiclass_table_preserves_functional_use_eligible_population():
    _, trait_p = MODULE.fisher_exact([[5, 5], [11, 59]], alternative="two-sided")
    trait_or, trait_lo, trait_hi = MODULE._haldane_or_ci(5, 5, 11, 59)
    trait = pd.DataFrame(
        [
            {
                "family": "rdkit_fg",
                "trait": "example",
                "n_eligible": 80,
                "n_missing": 20,
                "n_with_trait": 10,
                "n_without_trait": 70,
                "n_flag_with_trait": 5,
                "n_flag_without_trait": 11,
                "n_flag_total": 16,
                "odds_ratio": trait_or,
                "ci_lower": trait_lo,
                "ci_upper": trait_hi,
                "p_value": trait_p,
                "corrected_p_value": 0.99,
            }
        ]
    )
    fup = functional_use_frame("surfactant", (5, 5, 2, 18))
    fuc = functional_use_frame("Biocide", (4, 6, 3, 17))

    result = MODULE.build_multiclass_table(
        trait,
        fup,
        fuc,
        n_total=100,
        n_active=20,
        fup_threshold=0.5,
        min_compounds=10,
    )

    fup_result = result.loc[result["family"].eq("epa_fup")].iloc[0]
    expected_or, _, _ = MODULE._haldane_or_ci(5, 5, 2, 18)
    wrong_global_or, _, _ = MODULE._haldane_or_ci(5, 5, 15, 75)
    assert fup_result["n_eligible"] == 30
    assert fup_result["n_excluded"] == 70
    assert fup_result["n_reference"] == 20
    assert fup_result["odds_ratio"] == pytest.approx(expected_or)
    assert fup_result["odds_ratio"] != pytest.approx(wrong_global_or)
    assert fup_result["corrected_p_value"] == pytest.approx(fup_result["p_value"])
    assert fup_result["membership_definition"] == "predicted functional-use probability >= 0.5"
    assert fup_result["membership_threshold"] == pytest.approx(0.5)

    fup.loc[:, "threshold"] = 0.7
    with pytest.raises(ValueError, match="threshold=0.7; expected 0.5"):
        MODULE.build_multiclass_table(
            trait,
            fup,
            fuc,
            n_total=100,
            n_active=20,
            fup_threshold=0.5,
            min_compounds=10,
        )


def test_multiclass_table_rejects_inconsistent_upstream_counts():
    _, trait_p = MODULE.fisher_exact([[5, 5], [11, 59]], alternative="two-sided")
    trait_or, trait_lo, trait_hi = MODULE._haldane_or_ci(5, 5, 11, 59)
    trait = pd.DataFrame(
        [
            {
                "family": "rdkit_fg",
                "trait": "example",
                "n_eligible": 79,
                "n_missing": 21,
                "n_with_trait": 10,
                "n_without_trait": 70,
                "n_flag_with_trait": 5,
                "n_flag_without_trait": 11,
                "n_flag_total": 16,
                "odds_ratio": trait_or,
                "ci_lower": trait_lo,
                "ci_upper": trait_hi,
                "p_value": trait_p,
                "corrected_p_value": 0.99,
            }
        ]
    )

    with pytest.raises(ValueError, match="does not sum"):
        MODULE.build_multiclass_table(
            trait,
            functional_use_frame("surfactant", (5, 5, 2, 18)),
            functional_use_frame("Biocide", (4, 6, 3, 17)),
            n_total=100,
            n_active=20,
            fup_threshold=0.5,
            min_compounds=10,
        )


def test_multiclass_table_rejects_empty_member_or_reference_group():
    trait = pd.DataFrame(
        [
            {
                "family": "rdkit_fg",
                "trait": "empty",
                "n_eligible": 80,
                "n_missing": 20,
                "n_with_trait": 0,
                "n_without_trait": 80,
                "n_flag_with_trait": 0,
                "n_flag_without_trait": 16,
                "n_flag_total": 16,
                "odds_ratio": MODULE._haldane_or_ci(0, 0, 16, 64)[0],
                "ci_lower": MODULE._haldane_or_ci(0, 0, 16, 64)[1],
                "ci_upper": MODULE._haldane_or_ci(0, 0, 16, 64)[2],
                "p_value": 1.0,
                "corrected_p_value": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="needs members and references"):
        MODULE.build_multiclass_table(
            trait,
            functional_use_frame("surfactant", (5, 5, 2, 18)),
            functional_use_frame("Biocide", (4, 6, 3, 17)),
            n_total=100,
            n_active=20,
            fup_threshold=0.5,
            min_compounds=10,
        )


def test_upstream_summary_contract_requires_the_exact_activity_slice(tmp_path):
    expected = {
        "dataset": "compound_no_source7",
        "columns": "feat_all",
        "preprocessing": "activity_no_target2",
        "filter_name": "all_sources",
        "activity_params": "default",
        "activity_threshold": 0.1,
        "fup_threshold": 0.5,
        "min_compounds": 10,
        "n_activity_compounds": 100,
        "n_active": 20,
        "skip_brics": False,
        "min_compounds_by_family": MODULE.TRAIT_MIN_COMPOUNDS,
    }
    trait_summary = tmp_path / "trait.json"
    functional_summary = tmp_path / "functional.json"
    envelope = {"description": "test", "numbers": expected, "files": []}
    trait_summary.write_text(json.dumps(envelope))
    functional_summary.write_text(json.dumps(envelope))

    MODULE.validate_upstream_contracts(
        trait_summary,
        functional_summary,
        dataset="compound_no_source7",
        columns="feat_all",
        preprocessing="activity_no_target2",
        filter_name="all_sources",
        activity_params="default",
        activity_threshold=0.1,
        fup_threshold=0.5,
        min_compounds=10,
        n_activity_compounds=100,
        n_active=20,
    )

    mismatch = {**envelope, "numbers": {**expected, "columns": "wrong_features"}}
    functional_summary.write_text(json.dumps(mismatch))
    with pytest.raises(ValueError, match="2.42 summary columns='wrong_features'"):
        MODULE.validate_upstream_contracts(
            trait_summary,
            functional_summary,
            dataset="compound_no_source7",
            columns="feat_all",
            preprocessing="activity_no_target2",
            filter_name="all_sources",
            activity_params="default",
            activity_threshold=0.1,
            fup_threshold=0.5,
            min_compounds=10,
            n_activity_compounds=100,
            n_active=20,
        )


def test_summary_envelope_validation_rejects_missing_required_key_with_extra_key(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps({"description": "test", "files": [], "extra": "value"}))

    with pytest.raises(ValueError, match="not a standard summary envelope"):
        MODULE.load_summary_numbers(path, "malformed")


def test_snakemake_rule_declares_every_scientific_dependency_and_output():
    rules = (ROOT / "rules" / "srijit.smk").read_text()
    block = rules.split("rule srijit_2_45:", maxsplit=1)[1]

    for required in (
        'copairs_db="data/processed/copairs_results.duckdb"',
        'metadata_db="data/interim/jump_metadata_augmented.duckdb"',
        "rules.srijit_2_42.output.fup_enrichment",
        "rules.srijit_2_42.output.fuc_enrichment",
        "rules.srijit_2_42.output.summary",
        "rules.srijit_2_43.output.activity_csv",
        "rules.srijit_2_43.output.summary_json",
        "binary_trait_oddsratios.csv",
        "compound_activity_nmap_distributions.parquet",
        "multiclass_trait_oddsratios.csv",
        "forest_binary_traits.png",
        "forest_multiclass_activity.pdf",
        "summary.json",
        "--columns {params.columns}",
        "--fup-threshold {params.fup_threshold}",
        "--min-compounds {params.min_compounds}",
    ):
        assert required in block
    assert "srijit_2_44" not in block


def test_trait_enrichment_pins_the_consistency_feature_slice():
    notebook = (ROOT / "notebooks" / "2.43-seal-trait-activity-enrichment.py").read_text()
    rules = (ROOT / "rules" / "srijit.smk").read_text()
    block = rules.split("rule srijit_2_43:", maxsplit=1)[1].split("rule srijit_2_44:", maxsplit=1)[0]

    assert "AND _columns = ?" in notebook
    assert 'consistency_preprocessing="consistency_no_target2"' in block
    assert 'consistency_filter="all_sources"' in block
    assert 'consistency_distance="cosine"' in block


def test_activity_query_filters_the_feature_column_slice():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE activity_results AS
        SELECT * FROM (VALUES
            ('JCP1', 0.1, 0.2, 0.3, 0.4, false, false,
             'compound_no_source7', 'feat_all', 'activity_no_target2', 'all_sources', 'default'),
            ('JCP2', 0.1, 0.2, 0.3, 0.4, false, false,
             'compound_no_source7', 'other_features', 'activity_no_target2', 'all_sources', 'default')
        ) AS t(
            Metadata_JCP2022, mean_average_precision, mean_normalized_average_precision,
            p_value, corrected_p_value, below_p, below_corrected_p,
            _dataset, _columns, _preprocessing, _filter, _activity_params
        )
        """
    )

    result = query_activity_results(con, "compound_no_source7", columns="feat_all")

    assert result["Metadata_JCP2022"].tolist() == ["JCP1"]


def test_dataset_reference_includes_only_the_consolidated_245_summary():
    rules = (ROOT / "rules" / "analysis.smk").read_text()
    manifest_block = rules.split("rule dataset_reference_manifest:", maxsplit=1)[1]
    manifest_block = manifest_block.split("rule dataset_reference:", maxsplit=1)[0]

    assert "property-activity-oddsratios" in manifest_block
    assert "functional-use-classification" not in manifest_block
    assert "trait-activity-enrichment" not in manifest_block
