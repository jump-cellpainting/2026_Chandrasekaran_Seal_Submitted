"""Focused tests for the exact ChEMBL 33 target-matrix reproducer."""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jump_production.downloading import chembl_targets_sqlite as generator


def _create_fixture_database(path: Path, *, release: str = "ChEMBL_33") -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE activities (
            activity_id INTEGER,
            assay_id INTEGER,
            molregno INTEGER,
            pchembl_value REAL,
            relation TEXT,
            standard_flag INTEGER,
            standard_relation TEXT,
            standard_type TEXT,
            standard_units TEXT,
            standard_value REAL,
            type TEXT,
            units TEXT,
            value REAL,
            data_validity_comment TEXT,
            potential_duplicate INTEGER
        );
        CREATE TABLE assays (
            assay_id INTEGER,
            chembl_id TEXT,
            description TEXT,
            assay_type TEXT,
            tid INTEGER,
            doc_id INTEGER,
            assay_cell_type TEXT,
            assay_organism TEXT,
            assay_subcellular_fraction TEXT,
            assay_tissue TEXT,
            confidence_score INTEGER,
            relationship_type TEXT,
            variant_id INTEGER
        );
        CREATE TABLE molecule_dictionary (
            molregno INTEGER,
            chembl_id TEXT,
            chemical_probe INTEGER,
            chirality INTEGER,
            indication_class TEXT,
            max_phase REAL,
            oral INTEGER,
            prodrug INTEGER,
            withdrawn_flag INTEGER
        );
        CREATE TABLE target_dictionary (
            tid INTEGER,
            chembl_id TEXT,
            organism TEXT,
            target_type TEXT,
            pref_name TEXT,
            tax_id INTEGER
        );
        CREATE TABLE target_components (tid INTEGER, component_id INTEGER);
        CREATE TABLE component_sequences (
            component_id INTEGER,
            accession TEXT,
            organism TEXT,
            tax_id INTEGER,
            db_source TEXT
        );
        CREATE TABLE docs (doc_id INTEGER, chembl_id TEXT);
        CREATE TABLE assay_type (assay_type TEXT, assay_desc TEXT);
        CREATE TABLE relationship_type (relationship_type TEXT, relationship_desc TEXT);
        CREATE TABLE variant_sequences (variant_id INTEGER, mutation TEXT);
        CREATE TABLE molecule_hierarchy (
            molregno INTEGER, active_molregno INTEGER, parent_molregno INTEGER
        );
        CREATE TABLE compound_structures (
            molregno INTEGER, canonical_smiles TEXT, standard_inchi_key TEXT
        );
        CREATE TABLE version (name TEXT, creation_date TEXT, comments TEXT);
        """
    )
    connection.execute(
        "INSERT INTO version VALUES (?, ?, ?)",
        (release, "2023-05-31 00:00:00.000000", "fixture"),
    )
    connection.executemany(
        "INSERT INTO molecule_dictionary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "CHEMBL1", 0, 0, None, 0, 0, 0, 0),
            (2, "CHEMBL2", 0, 0, None, 0, 0, 0, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO target_dictionary VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "CHEMBLT1", "Homo sapiens", "SINGLE PROTEIN", "Target one", 9606),
            (
                2,
                "CHEMBL_COMPLEX_ONLY",
                "Homo sapiens",
                "PROTEIN COMPLEX",
                "Complex-only target",
                9606,
            ),
            (
                3,
                "CHEMBL_SINGLE_SHARED",
                "Homo sapiens",
                "SINGLE PROTEIN",
                "Shared single target",
                9606,
            ),
            (
                4,
                "CHEMBL_COMPLEX_SHARED",
                "Homo sapiens",
                "PROTEIN COMPLEX",
                "Shared complex target",
                9606,
            ),
            (5, "CHEMBL_FAMILY", "Homo sapiens", "PROTEIN FAMILY", "Family", 9606),
            (
                6,
                "CHEMBL_TREMBL",
                "Mus musculus",
                "SINGLE PROTEIN",
                "TrEMBL target",
                10090,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO target_components VALUES (?, ?)",
        [(1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 5)],
    )
    connection.executemany(
        "INSERT INTO component_sequences VALUES (?, ?, ?, ?, ?)",
        [
            (1, "P00001", "Homo sapiens", 9606, "SWISS-PROT"),
            (2, "P00002", "Homo sapiens", 9606, "SWISS-PROT"),
            (3, "P00003", "Homo sapiens", 9606, "SWISS-PROT"),
            (4, "P00004", "Homo sapiens", 9606, "SWISS-PROT"),
            (5, "Q00005", "Mus musculus", 10090, "TREMBL"),
        ],
    )
    connection.execute("INSERT INTO docs VALUES (1, 'CHEMBLD1')")
    connection.executemany(
        "INSERT INTO assay_type VALUES (?, ?)",
        [("B", "Binding"), ("F", "Functional"), ("A", "ADME")],
    )
    connection.execute("INSERT INTO relationship_type VALUES ('D', 'Direct')")
    connection.execute("INSERT INTO variant_sequences VALUES (1, NULL)")
    connection.executemany(
        "INSERT INTO molecule_hierarchy VALUES (?, ?, ?)",
        [(1, 1, 1), (2, 2, 2)],
    )
    connection.executemany(
        "INSERT INTO compound_structures VALUES (?, ?, ?)",
        [
            (1, "CC", "AAAAAAAAAAAAAA-BBBBBBBBBB-C"),
            (2, "CCC", "CCCCCCCCCCCCCC-DDDDDDDDDD-E"),
        ],
    )
    connection.executemany(
        "INSERT INTO assays VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "CHEMBLA1", "valid", "B", 1, 1, None, None, None, None, 9, "D", 1),
            (2, "CHEMBLA2", "functional", "F", 1, 1, None, None, None, None, 6, "D", 1),
            (3, "CHEMBLA3", "low confidence", "B", 1, 1, None, None, None, None, 5, "D", 1),
            (4, "CHEMBLA4", "wrong type", "A", 1, 1, None, None, None, None, 9, "D", 1),
        ],
    )
    rows = [
        (1, 1, 1, 6.0, "=", 1, "=", "IC50", "nM", 100.0, "IC50", "nM", 100.0, None, 0),
        (2, 2, 2, None, "=", 1, "=", "Ki", "uM", 10.0, "Ki", "uM", 10.0, None, 0),
        (3, 3, 1, 9.0, "=", 1, "=", "IC50", "nM", 1.0, "IC50", "nM", 1.0, None, 0),
        (4, 4, 1, 9.0, "=", 1, "=", "IC50", "nM", 1.0, "IC50", "nM", 1.0, None, 0),
        (5, 1, 1, 9.0, "<", 1, "<", "IC50", "nM", 1.0, "IC50", "nM", 1.0, None, 0),
        (6, 1, 1, 9.0, "=", 1, "=", "IC50", "nM", 1.0, "IC50", "nM", 1.0, "Outside typical range", 0),
        (7, 1, 1, 9.0, "=", 1, "=", "IC50", "nM", 1.0, "IC50", "nM", 1.0, None, 1),
    ]
    connection.executemany(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


def _tar_info(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    return member, payload


def _write_fixture_archive(path: Path, *, database: bytes, unexpected: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo(generator.OFFICIAL_DIRECTORY_MEMBER)
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        install_member, install = _tar_info(
            generator.OFFICIAL_INSTALL_MEMBER,
            b"I" * generator.OFFICIAL_INSTALL_SIZE,
        )
        archive.addfile(install_member, io.BytesIO(install))
        database_member, database_payload = _tar_info(
            generator.OFFICIAL_DATABASE_MEMBER,
            database,
        )
        archive.addfile(database_member, io.BytesIO(database_payload))
        if unexpected is not None:
            extra_member, extra = _tar_info(unexpected, b"bad")
            archive.addfile(extra_member, io.BytesIO(extra))


def _raw_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, overrides in enumerate(rows, start=1):
        record: dict[str, object] = {column: None for column in generator.RAW_COLUMNS}
        record.update(
            {
                "JUMP_ID": f"JCP2022_{index:06d}",
                "molecule_chembl_id": f"CHEMBL{index}",
                "activity_id": index,
                "pchembl_value": 6.0,
                "standard_type": "IC50",
                "canonical_smiles": "raw",
            }
        )
        record.update(overrides)
        records.append(record)
    return pd.DataFrame(records, columns=pd.Index(generator.RAW_COLUMNS))


@pytest.mark.parametrize(
    "output_dir",
    [generator.DATA_DIR, generator.DATA_DIR / "processed" / "chembl33"],
)
def test_output_guard_rejects_mirrored_data(output_dir: Path) -> None:
    with pytest.raises(generator.SourceIntegrityError, match="outside the mirrored data"):
        generator.run_exact_reproduction(output_dir=output_dir)


def test_output_guard_follows_symlinks(tmp_path: Path) -> None:
    alias = tmp_path / "data-alias"
    alias.symlink_to(generator.DATA_DIR, target_is_directory=True)
    with pytest.raises(generator.SourceIntegrityError, match="outside the mirrored data"):
        generator.run_exact_reproduction(output_dir=alias / "generated")


def test_failure_leaves_incomplete_summary(tmp_path: Path) -> None:
    output = tmp_path / "output"
    with pytest.raises(
        generator.SourceIntegrityError,
        match="(?i)SmallWorld.*missing|source.*missing|CSV.*missing|Frozen input.*missing",
    ):
        generator.run_exact_reproduction(
            archive=tmp_path / "archive.tar.gz",
            database=tmp_path / "chembl.db",
            output_dir=output,
            reference=tmp_path / "reference.csv",
        )
    summary = json.loads((output / generator.SUMMARY_NAME).read_text())
    assert summary["status"] == "incomplete"
    assert summary["stage"].startswith("verifying")
    assert "reference" in summary["stage"]
    assert summary["error"]["type"] == "SourceIntegrityError"


def _write_pinned_csv(path: Path, frame: pd.DataFrame) -> generator.CsvSourcePin:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return generator.CsvSourcePin(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=generator._sha256(path),
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
    )


def test_verify_smallworld_mapping_checks_bytes_schema_rows_and_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jump_to_chembl.csv.gz"
    expected = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J1", "J2"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "hit_rank": [0, 1, 0],
        }
    )
    pin = _write_pinned_csv(path, expected)

    frame, metadata = generator.verify_smallworld_mapping(path, pin=pin)

    pd.testing.assert_frame_equal(frame, expected)
    assert metadata["verified"] is True

    wrong_schema = generator.CsvSourcePin(**{**pin.__dict__, "columns": ("molecule_chembl_id", "JUMP_ID", "hit_rank")})
    with pytest.raises(generator.SourceIntegrityError, match="has columns"):
        generator.verify_smallworld_mapping(path, pin=wrong_schema)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"JUMP_ID": "J1", "molecule_chembl_id": "CHEMBL1", "hit_rank": 0},
                {"JUMP_ID": "J1", "molecule_chembl_id": "CHEMBL1", "hit_rank": 1},
            ],
            "(?i)duplicate",
        ),
        (
            [{"JUMP_ID": "J1", "molecule_chembl_id": None, "hit_rank": 0}],
            "(?i)missing|null",
        ),
        (
            [
                {"JUMP_ID": "J1", "molecule_chembl_id": "CHEMBL1", "hit_rank": 0},
                {"JUMP_ID": "J1", "molecule_chembl_id": "CHEMBL2", "hit_rank": 2},
            ],
            "(?i)contiguous|rank",
        ),
        (
            [{"JUMP_ID": "J1", "molecule_chembl_id": "CHEMBL1", "hit_rank": 0.5}],
            "(?i)integer|rank",
        ),
    ],
)
def test_verify_smallworld_mapping_rejects_semantically_invalid_tables(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = tmp_path / "jump_to_chembl.csv.gz"
    frame = pd.DataFrame(rows, columns=["JUMP_ID", "molecule_chembl_id", "hit_rank"])
    pin = _write_pinned_csv(path, frame)

    with pytest.raises(generator.SourceIntegrityError, match=message):
        generator.verify_smallworld_mapping(path, pin=pin)


def test_smallworld_cache_rejects_retired_or_unexpected_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True)
    mapping_path = input_dir / "jump_to_chembl.csv.gz"
    mapping = pd.DataFrame(
        {
            "JUMP_ID": ["J1"],
            "molecule_chembl_id": ["CHEMBL1"],
            "hit_rank": [0],
        }
    )
    pin = _write_pinned_csv(mapping_path, mapping)
    monkeypatch.setattr(generator, "SMALLWORLD_MAPPING_PIN", pin)
    (input_dir / "jump_compounds.csv.gz").write_bytes(b"retired")

    with pytest.raises(generator.SourceIntegrityError, match="only the SmallWorld mapping"):
        generator.load_smallworld_mapping(output_dir)


def test_verify_jump_compound_source_checks_identity_and_unique_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compound.csv.gz"
    expected = pd.DataFrame(
        {
            "Metadata_JCP2022": ["J1", "JCP2022_999999"],
            "Metadata_InChIKey": [
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                np.nan,
            ],
            "Metadata_InChI": ["InChI=1S/CH4/h1H4", np.nan],
            "Metadata_SMILES": ["C", np.nan],
        }
    )
    pin = _write_pinned_csv(path, expected)

    frame, metadata = generator.verify_jump_compound_source(path, pin=pin)

    pd.testing.assert_frame_equal(frame, expected)
    assert metadata["verified"] is True


def test_verify_jump_compound_source_rejects_duplicate_jump_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "compound.csv.gz"
    frame = pd.DataFrame(
        {
            "Metadata_JCP2022": ["J1", "J2"],
            "Metadata_InChIKey": [
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
            ],
            "Metadata_InChI": ["InChI=1S/CH4/h1H4", "InChI=1S/C2H6/c1-2/h1-2H3"],
            "Metadata_SMILES": ["C", "CC"],
        }
    )
    frame.loc[1, "Metadata_JCP2022"] = "J1"
    pin = _write_pinned_csv(path, frame)

    with pytest.raises(generator.SourceIntegrityError, match="(?i)duplicate"):
        generator.verify_jump_compound_source(path, pin=pin)


def test_canonical_reference_gate_requires_exact_hash_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_bytes(b"a,b\n1,2\n")
    monkeypatch.setattr(generator, "CANONICAL_REFERENCE_SIZE", reference.stat().st_size)
    monkeypatch.setattr(generator, "CANONICAL_REFERENCE_SHA256", generator._sha256(reference))
    payload, metadata = generator.verify_canonical_reference(reference)
    assert payload == reference.read_bytes()
    assert metadata["verified"] is True

    reference.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(generator.SourceIntegrityError, match="SHA-256"):
        generator.verify_canonical_reference(reference)


def test_verify_archive_uses_both_size_and_sha(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"official bytes")
    metadata = generator.verify_official_archive(
        archive,
        expected_size=archive.stat().st_size,
        expected_sha256=generator._sha256(archive),
    )
    assert metadata["verified"] is True
    with pytest.raises(generator.SourceIntegrityError, match="size"):
        generator.verify_official_archive(
            archive,
            expected_size=archive.stat().st_size + 1,
            expected_sha256=generator._sha256(archive),
        )


def test_database_reuse_rejects_bytes_outside_extracted_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"pinned archive")
    database = tmp_path / "chembl_33.db"
    database.write_bytes(b"modified database")
    (tmp_path / generator.EXTRACTION_MANIFEST_NAME).write_text("{}\n")

    monkeypatch.setattr(generator, "OFFICIAL_DATABASE_SIZE", database.stat().st_size)
    monkeypatch.setattr(generator, "OFFICIAL_DATABASE_SHA256", "0" * 64)
    monkeypatch.setattr(
        generator,
        "verify_official_archive",
        lambda _path: {
            "sha256": generator._sha256(archive),
            "size_bytes": archive.stat().st_size,
        },
    )
    with pytest.raises(generator.SourceIntegrityError, match="database SHA-256"):
        generator.ensure_official_database(archive, database)


def test_safe_archive_extraction_accepts_only_pinned_members(tmp_path: Path) -> None:
    archive = tmp_path / "chembl.tar.gz"
    database_payload = b"sqlite fixture"
    _write_fixture_archive(archive, database=database_payload)
    database = tmp_path / "nested" / "chembl_33.db"
    metadata = generator.extract_official_database(
        archive,
        database,
        expected_database_size=len(database_payload),
    )
    assert database.read_bytes() == database_payload
    assert (database.parent / "INSTALL_sqlite").stat().st_size == generator.OFFICIAL_INSTALL_SIZE
    assert metadata["database_member"] == generator.OFFICIAL_DATABASE_MEMBER


@pytest.mark.parametrize("unexpected", ["../escape", "chembl_33/extra.txt"])
def test_safe_archive_extraction_rejects_unexpected_paths(
    tmp_path: Path,
    unexpected: str,
) -> None:
    archive = tmp_path / "chembl.tar.gz"
    _write_fixture_archive(archive, database=b"db", unexpected=unexpected)
    with pytest.raises(generator.SourceIntegrityError, match="Unsafe path|Unexpected member"):
        generator.extract_official_database(
            archive,
            tmp_path / "chembl.db",
            expected_database_size=2,
        )


def test_safe_archive_extraction_rejects_link_payload(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo(generator.OFFICIAL_DIRECTORY_MEMBER)
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        install, payload = _tar_info(
            generator.OFFICIAL_INSTALL_MEMBER,
            b"I" * generator.OFFICIAL_INSTALL_SIZE,
        )
        archive.addfile(install, io.BytesIO(payload))
        link = tarfile.TarInfo(generator.OFFICIAL_DATABASE_MEMBER)
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp/elsewhere"
        archive.addfile(link)
    with pytest.raises(generator.SourceIntegrityError, match="not a regular file"):
        generator.extract_official_database(
            archive_path,
            tmp_path / "chembl.db",
            expected_database_size=0,
        )


def test_database_validation_pins_release_date_and_schema(tmp_path: Path) -> None:
    database = tmp_path / "chembl.db"
    _create_fixture_database(database)
    with generator.open_chembl_database(database) as connection:
        metadata = generator.validate_database(connection)
    assert metadata["release"] == "ChEMBL_33"
    assert metadata["release_date"].startswith("2023-05-31")
    assert metadata["quick_check"] == "ok"
    assert len(metadata["required_schema_sha256"]) == 64

    wrong_release = tmp_path / "wrong.db"
    _create_fixture_database(wrong_release, release="ChEMBL_37")
    with generator.open_chembl_database(wrong_release) as connection:
        with pytest.raises(generator.SourceIntegrityError, match="Expected exactly ChEMBL_33"):
            generator.validate_database(connection)


def test_database_query_applies_historical_filters_and_uses_molecule_ids(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chembl.db"
    _create_fixture_database(database)
    mapping = pd.DataFrame(
        {
            "JUMP_ID": ["JCP2022_000001", "JCP2022_000002"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
            "hit_rank": [0, 0],
        }
    )
    with generator.open_chembl_database(database) as connection:
        activities, molecules = generator.extract_database_tables(connection, mapping)
    assert activities["activity_id"].tolist() == [1, 2]
    assert activities["molecule_chembl_id"].tolist() == ["CHEMBL1", "CHEMBL2"]
    assert molecules["metadata_source_id"].tolist() == ["CHEMBL1", "CHEMBL2"]


def test_target_mapping_is_derived_from_swiss_prot_single_and_unique_complexes(
    tmp_path: Path,
) -> None:
    database = tmp_path / "chembl.db"
    _create_fixture_database(database)
    requested = [
        "CHEMBLT1",
        "CHEMBL_COMPLEX_ONLY",
        "CHEMBL_COMPLEX_SHARED",
        "CHEMBL_FAMILY",
        "CHEMBL_TREMBL",
    ]

    with generator.open_chembl_database(database) as connection:
        observed = generator.extract_target_mapping(connection, requested)

    assert observed.columns.tolist() == ["From", "Entry", "Organism", "Target Name"]
    assert observed[["From", "Entry"]].values.tolist() == [
        ["CHEMBLT1", "P00001"],
        ["CHEMBL_COMPLEX_ONLY", "P00002"],
    ]
    assert observed["Organism"].tolist() == [
        "Homo sapiens (Human)",
        "Homo sapiens (Human)",
    ]
    assert observed["Target Name"].tolist() == ["Target one", "Complex-only target"]


def test_pchembl_reconstruction_preserves_float32_and_supported_units() -> None:
    activities = pd.DataFrame(
        {
            "standard_value": [None, 100.0, 10.0, 1000.0, 1.0],
            "standard_units": [None, "nM", chr(181) + "M", "uM", "mM"],
            "pchembl_value": [7.123456789, None, None, None, None],
            "marker": ["direct", "nm", "micro", "um", "excluded"],
        }
    )
    result = generator.calculate_missing_pchembl(activities)
    assert result["marker"].tolist() == ["direct", "nm", "micro", "um"]
    assert result.loc[0, "pchembl_value"] == pytest.approx(
        np.float32(7.123456789),
    )
    assert result["pchembl_value"].tolist()[1:] == pytest.approx([7.0, 5.0, 3.0])
    assert result["standard_value"].dtype == np.dtype("float32")


def test_pchembl_fallback_matches_row_apply_not_vectorized_float32() -> None:
    activities = pd.DataFrame(
        {
            "standard_value": [250.0],
            "standard_units": ["nM"],
            "pchembl_value": [None],
        }
    )
    result = generator.calculate_missing_pchembl(activities)
    float32_value = np.float32(250.0)
    # A mixed-type axis=1 row presents this value as a Python float, exactly as
    # the historical frame.apply calculation did.
    row_apply_value = float(-np.log10(float(float32_value) * 1e-9))
    vectorized_value = float((-np.log10(pd.Series([float32_value]) * 1e-9)).iloc[0])
    assert float(result.loc[0, "pchembl_value"]).hex() == row_apply_value.hex()
    assert row_apply_value != vectorized_value


def test_historical_metadata_association_preserves_positional_reassignment() -> None:
    activities = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J1"],
            "molecule_chembl_id": ["CHEMBL2", "CHEMBL1"],
            "activity_id": [1, 2],
            "_pchem_was_missing": [0, 0],
        }
    )
    molecules = pd.DataFrame(
        {
            "metadata_source_id": ["CHEMBL1", "CHEMBL2"],
            "molregno": [1, 2],
            "canonical_smiles": ["one", "two"],
        }
    )
    associated, audit, counts = generator.associate_historical_compound_metadata(
        activities,
        molecules,
    )
    observed = associated.set_index("molecule_chembl_id")["canonical_smiles"].to_dict()
    assert observed == {"CHEMBL2": "one", "CHEMBL1": "two"}
    assert audit["metadata_reassigned"].tolist() == [True, True]
    assert counts["metadata_reassigned_rows"] == 2
    assert counts["metadata_reassigned_jump_ids"] == 1


def test_structure_curation_is_injectable_and_removes_historical_fragments() -> None:
    raw = _raw_frame(
        [
            {"canonical_smiles": "salted"},
            {"canonical_smiles": "mixture"},
            {"canonical_smiles": "missing", "pchembl_value": None},
            {"canonical_smiles": "other", "standard_type": "IC90"},
        ]
    )
    standardized = {
        "salted": "CC.[Na+]",
        "mixture": "CC.CN",
        "missing": "CCC",
        "other": "CCCC",
    }
    calls: list[tuple[list[str], int]] = []

    def fake_standardize(smiles: list[str], jobs: int) -> list[str]:
        calls.append((smiles, jobs))
        return [standardized[value] for value in smiles]

    curated, counts = generator.curate_structures_and_activity_types(
        raw,
        jobs=3,
        standardize=fake_standardize,
    )
    assert calls == [(["salted", "mixture", "missing", "other"], 3)]
    assert curated["standard_smiles"].tolist() == ["CC"]
    assert counts == {
        "input_rows": 4,
        "unresolved_mixture_rows": 1,
        "missing_pchembl_rows": 1,
        "excluded_activity_type_rows": 1,
        "output_rows": 1,
    }


def test_jump_compounds_use_current_source_with_rank_zero_chembl_fallback() -> None:
    mapping = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J2", "J2"],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "hit_rank": [0, 0, 1],
        }
    )
    source_compounds = pd.DataFrame(
        {
            "Metadata_JCP2022": ["J1", "UNMAPPED"],
            "Metadata_InChIKey": [
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "UUUUUUUUUUUUUU-VVVVVVVVVV-W",
            ],
            "Metadata_InChI": ["source-inchi", "unmapped-inchi"],
            "Metadata_SMILES": ["source-smiles", "unmapped-smiles"],
        }
    )
    molecules = pd.DataFrame(
        {
            "metadata_source_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "canonical_smiles": ["database-one", "fallback-two", "wrong-rank"],
            "standard_inchikey": [
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
                "EEEEEEEEEEEEEE-FFFFFFFFFF-G",
            ],
        }
    )
    calls: list[tuple[list[str], int]] = []

    def fake_standardize(smiles: list[str], jobs: int) -> list[str]:
        calls.append((smiles, jobs))
        return [f"standardized-{value}" for value in smiles]

    compounds, diagnostics = generator.derive_jump_compounds(
        mapping,
        source_compounds,
        molecules,
        jobs=4,
        standardize=fake_standardize,
    )

    assert calls == [(["source-smiles", "fallback-two"], 4)]
    assert compounds.to_dict(orient="records") == [
        {
            "Metadata_JCP2022": "J1",
            "Standardized_SMILES": "standardized-source-smiles",
        },
        {
            "Metadata_JCP2022": "J2",
            "Standardized_SMILES": "standardized-fallback-two",
        },
    ]
    assert diagnostics["output_rows"] == 2
    assert diagnostics["fallback_rows"] == 1


def test_jump_compound_derivation_fails_when_rank_zero_fallback_is_unavailable() -> None:
    mapping = pd.DataFrame(
        {
            "JUMP_ID": ["J1"],
            "molecule_chembl_id": ["CHEMBL_MISSING"],
            "hit_rank": [0],
        }
    )
    source_compounds = pd.DataFrame(
        columns=[
            "Metadata_JCP2022",
            "Metadata_InChIKey",
            "Metadata_InChI",
            "Metadata_SMILES",
        ]
    )
    molecules = pd.DataFrame(columns=["metadata_source_id", "canonical_smiles", "standard_inchikey"])

    with pytest.raises(generator.SourceIntegrityError, match="(?i)fallback|missing|unresolved"):
        generator.derive_jump_compounds(
            mapping,
            source_compounds,
            molecules,
            jobs=1,
            standardize=lambda smiles, _jobs: smiles,
        )


def test_finalizer_uses_strict_threshold_support_before_structure_dedup() -> None:
    activities = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J2", "J3"],
            "target_chembl_id": ["COMPLEX_TARGET", "COMPLEX_TARGET", "OTHER"],
            "pchembl_value": [6.0, 5.0, 7.0],
        }
    )
    target_mapping = pd.DataFrame(
        {
            "From": ["COMPLEX_TARGET", "OTHER"],
            "Entry": ["P61604", "P00002"],
            "Organism": ["Homo sapiens (Human)", "Mouse"],
            "Target Name": ["HSPD1", "Other"],
        }
    )
    compounds = pd.DataFrame(
        {
            "Metadata_JCP2022": ["J1", "J2", "J3"],
            "Standardized_SMILES": ["CC", "CC", "CCC"],
        }
    )
    long_labels, wide, target_counts, summary = generator.build_annotations(
        activities,
        target_mapping,
        compounds,
        minimum_per_class=1,
    )
    assert long_labels[["JUMP_ID", "Entry", "label"]].values.tolist() == [["J1", "P61604", 1]]
    assert wide.columns.tolist() == ["JUMP_ID", "P61604", "Standardized_SMILES"]
    assert target_counts.loc[0, "UniProt_accession"] == "P61604"
    assert summary["targets"] == 1
    assert summary["active"] == 1
    assert summary["inactive"] == 0


def test_comparison_diagnostics_cover_universes_masks_labels_smiles_and_actives() -> None:
    current = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J2"],
            "P1": [1.0, np.nan],
            "P2": [0.0, 1.0],
            "Standardized_SMILES": ["CC", "CCC"],
        }
    )
    reference = pd.DataFrame(
        {
            "JUMP_ID": ["J1", "J3"],
            "P1": [0.0, 1.0],
            "P3": [1.0, np.nan],
            "Standardized_SMILES": ["CN", "CCCC"],
        }
    )
    diagnostics = generator.compare_wide_candidate(current, reference)
    assert diagnostics["compound_universe"]["current_only"] == 1
    assert diagnostics["target_universe"]["reference_only"] == 1
    assert diagnostics["measured_mask_on_common_grid"]["both_measured"] == 1
    assert diagnostics["labels_on_common_measured_coordinates"]["different"] == 1
    assert diagnostics["standardized_smiles"]["different"] == 1
    assert diagnostics["active_memberships"]["current_only"] >= 1


def test_deterministic_gzip_writer_pins_header_and_content(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    first = tmp_path / "one" / "audit.csv.gz"
    second = tmp_path / "two" / "audit.csv.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    generator._atomic_gzip_csv(frame, first)
    generator._atomic_gzip_csv(frame, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt") as handle:
        assert handle.read() == "a,b\n1,x\n2,y\n"
    raw = first.read_bytes()
    assert int.from_bytes(raw[4:8], "little") == 0
    assert b"audit.csv\0" in raw[:64]
