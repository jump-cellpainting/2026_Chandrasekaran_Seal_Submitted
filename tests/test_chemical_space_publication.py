"""Behavioral contracts for the Figure 1 chemical-space publication paths."""


import duckdb
import pandas as pd
import pytest
from conftest import load_notebook
from PIL import Image

MODULE = load_notebook("2.04a-ss-chemical-space.py")


def test_analysis_output_paths_are_population_scoped_by_default(tmp_path):
    assert MODULE.resolve_analysis_output_dir(None, None) == MODULE.OUTPUT_DIR / "all"
    assert MODULE.resolve_analysis_output_dir(None, "compound_no_source7") == (
        MODULE.OUTPUT_DIR / "compound_no_source7"
    )
    assert MODULE.resolve_analysis_output_dir(tmp_path, None) == tmp_path
    assert MODULE.resolve_analysis_output_dir(tmp_path, "compound_no_source7") == (
        tmp_path / "compound_no_source7"
    )


def test_target_moa_coverage_uses_provider_families_and_their_union():
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE compound_metadata AS
        SELECT * FROM (VALUES
            ('JCP1', 'P12345', NULL, NULL, NULL),
            ('JCP2', NULL, 'target', NULL, NULL),
            ('JCP3', NULL, NULL, 'MOA', NULL),
            ('JCP4', NULL, NULL, NULL, 'GENE'),
            ('JCP5', NULL, NULL, NULL, NULL)
        ) AS t(
            Metadata_JCP2022,
            Metadata_Uniprot_target,
            Metadata_repurposing_target,
            Metadata_repurposing_moa,
            Metadata_chmprb_target_genes
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE motive_annotations AS
        SELECT * FROM (VALUES
            ('JCP2', 'MOTIVE2'),
            ('JCP5', 'MOTIVE5')
        ) AS t(Metadata_JCP2022, Metadata_motive_gene)
        """
    )

    result = MODULE.target_moa_coverage(connection, ["JCP1", "JCP2", "JCP3", "JCP4", "JCP5"])

    assert result["denominator"] == 5
    assert {row["id"]: row["count"] for row in result["families"]} == {
        "chembl": 1,
        "motive": 2,
        "repurposing_hub": 2,
        "chemical_probes": 1,
    }
    assert result["target_or_moa_any"]["count"] == 5


def test_publication_roster_is_exact_and_returned_in_canonical_order():
    frame = pd.DataFrame(
        {
            "Metadata_JCP2022": ["JCP2", "JCP1"],
            "value": [2, 1],
        }
    )

    ordered = MODULE.validate_and_order_publication_compounds(frame, ["JCP1", "JCP2"])

    assert ordered["Metadata_JCP2022"].tolist() == ["JCP1", "JCP2"]
    with pytest.raises(ValueError, match="1 missing and 1 extra"):
        MODULE.validate_and_order_publication_compounds(frame, ["JCP1", "JCP3"])


def test_pains_partition_keeps_invalid_structures_out_of_the_clean_denominator():
    frame = pd.DataFrame(
        {
            "Metadata_ValidMol": [True, True, False],
            "Metadata_HasPAINS": [False, True, None],
        }
    )

    result = MODULE.publication_pains_summary(frame)

    assert result == {
        "total": 3,
        "assessable": 2,
        "clean": 1,
        "flagged": 1,
        "unassessable": 1,
        "prevalence_denominator": "compounds with valid molecular structures",
        "downstream_activity_handling": "flagged but not excluded",
        "flagged_percent_of_assessable": 50.0,
        "unassessable_percent_of_total": pytest.approx(100 / 3),
    }


def test_scaffold_ranking_has_a_stable_smiles_tiebreaker():
    frame = pd.DataFrame(
        {
            "Metadata_MurckoScaffold": ["B", "A", "B", "A", "C", None, ""],
        }
    )

    top, statistics = MODULE.publication_scaffold_table(frame)

    assert top[["rank", "scaffold", "count"]].to_dict("records") == [
        {"rank": 1, "scaffold": "A", "count": 2},
        {"rank": 2, "scaffold": "B", "count": 2},
        {"rank": 3, "scaffold": "C", "count": 1},
    ]
    assert statistics["acyclic"] == 2
    assert statistics["ordering"] == "count descending, then scaffold SMILES ascending"


def test_clinical_phase_summary_separates_displayed_and_other_non_null_values():
    frame = pd.DataFrame(
        {
            "Metadata_repurposing_clinical_phase": [
                "Preclinical",
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Launched",
                "Withdrawn",
                "Phase 1/Phase 2",
                None,
            ]
        }
    )

    result = MODULE.publication_clinical_phase_summary(frame)

    assert result["denominator"] == 5
    assert result["non_null_count"] == 7
    assert result["excluded_count"] == 2
    assert result["excluded_value_counts"] == {"Phase 1/Phase 2": 1, "Withdrawn": 1}
    assert result["counts"] == {
        "Preclinical": 1,
        "Phase 1": 1,
        "Phase 2": 1,
        "Phase 3": 1,
        "Launched": 1,
    }


def test_publication_schematic_is_composited_on_white_without_changing_source(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "publication.png"
    authored = Image.new("RGBA", (2, 1))
    authored.putdata([(255, 0, 0, 255), (0, 0, 255, 0)])
    authored.save(source)

    result = MODULE.composite_publication_schematic_on_white(source, target, "The caption.", dpi=150)

    with Image.open(source) as source_image:
        assert source_image.mode == "RGBA"
        assert source_image.getpixel((1, 0)) == (0, 0, 255, 0)
    with Image.open(target) as publication_image:
        assert publication_image.mode == "RGB"
        # The schematic's own pixels are copied through untouched; the caption rides below them.
        assert publication_image.getpixel((0, 0)) == (255, 0, 0)
        assert publication_image.getpixel((1, 0)) == (255, 255, 255)
        assert publication_image.height > 1
    assert result["publication_background"] == "white"
    assert result["schematic_height_px"] == 1
    assert result["caption_band_px"] == result["height_px"] - result["schematic_height_px"] > 0
