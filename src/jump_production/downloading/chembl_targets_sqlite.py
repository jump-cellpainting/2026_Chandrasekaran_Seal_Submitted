#!/usr/bin/env python3
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "chem-filters==1.0.0",
#   "chembl-structure-pipeline==1.2.2",
#   "dimorphite-dl==1.2.5",
#   "numpy==1.26.4",
#   "pandas==2.2.1",
#   "rdkit==2023.9.5",
#   "typer==0.21.2",
# ]
# ///
"""Exactly reproduce the published JUMP ChEMBL 33 target matrix.

Canonical production input
--------------------------
The pipeline continues to download
``data/external/targetannotations_singleproteins_jumpcompounds_all_5.csv``
from Zenodo through ``download_data.py``. That fixed file, not this local
reconstruction, is the production input. This script is a provenance audit: it
writes only below gitignored ``.cache/`` and rejects archive, database, or
output paths below ``data/`` so the audit cannot enter the S3 data mirror.

Provenance boundary
-------------------
This is an independently maintained implementation.
The public Capricho snapshot and David Araripe's later reconstruction
were used to resolve and validate historical behavior.
No Python from those packages is copied, imported, or executed here.

The reconstruction has four inputs:

* the checksum-pinned official ChEMBL 33 SQLite archive supplies activities,
  molecule structures, and ChEMBL-to-UniProt target components;
* the checksum-pinned JUMP ``data/external/compound.csv.gz`` already acquired by
  ``download_data.py`` supplies the contemporary JUMP compound roster;
* ``jump_to_chembl.csv.gz`` supplies the complete ranked exact-structure result
  saved from SmallWorld against ``chembl_31.anon``; and
* the checksum-pinned Zenodo CSV is the mandatory byte-for-byte acceptance gate.

The SmallWorld result is the only irreducible historical data artifact.
Direct ChEMBL 33 structure matching and the existing UniChem pointers recover
only subsets of its 36,488 pairs, and its 7,706 non-rank-zero rows materially
affect the result.
The exact saved bytes are published in Zenodo record 21816611
(``doi:10.5281/zenodo.21816611``) and protected here by a SHA-256 pin.

The former ``jump_compounds.csv.gz`` input is regenerated from the JUMP compound
table with the historical iterative RDKit and Dimorphite-DL standardization
procedure embedded below.
One mapped identifier absent from the current JUMP table, ``JCP2022_088778``, is
filled from its rank-zero ChEMBL 33 molecule ``CHEMBL2074916``.
The former ``chembl_target_to_uniprot.csv.gz`` input is regenerated from the
native ChEMBL 33 target component tables.
It retains Swiss-Prot single-protein components plus complex components whose
accession has no Swiss-Prot single-protein target elsewhere in the release.
The derived map has the same 3,356 pairs as the historical snapshot except that
``CHEMBL3351219 -> P24855`` replaces ``CHEMBL1293304 -> G5EF15``.
Neither accession passes the final per-class support rule, so the published
102-target matrix is unchanged.
The audit-only target summary uses ChEMBL ``pref_name`` as ``Target Name``;
it does not claim that these protein descriptions are gene symbols.

The historical output filename says ``singleproteins``, but the compatibility
rule also contains protein-complex mappings. The final 102 targets include ``P61604``
from the HSP60/HSP10 complex ``CHEMBL4106131``. The filename remains unchanged
because it is the published pipeline interface.

Compatibility procedure
-----------------------
The implementation preserves the historically consequential behavior:

1. Query ChEMBL 33 by the full ranked SmallWorld molecule identifiers.
2. Keep binding and functional assays with confidence 6-9, exact ``=``
   relations, no validity warning, and no potential-duplicate flag.
3. Fill missing pChEMBL for nM and micromolar values using the historical
   float32 conversion and row-wise calculation, then retain IC50, EC50, AC50,
   Ki, Kd, and Potency measurements.
4. Reproduce the historical positional batched-metadata association, structure
   standardization, salt/solvent cleanup, and unresolved-mixture exclusion.
5. Derive the UniProt map from ChEMBL target components and take the arithmetic
   mean by JUMP identifier and UniProt accession.
6. Label a pair active only when mean pChEMBL is strictly greater than 5.
7. Require at least 15 active and 15 inactive compounds per target before
   deduplicating JUMP-target and standardized-structure-target records.
8. Pivot and serialize the final sparse matrix deterministically.

Exact acceptance and evidence
-----------------------------
The reference itself must match ``CANONICAL_REFERENCE_SHA256``. A run completes
only when the generated wide CSV is byte-identical to that reference and all
expected row, target, and class counts pass. On failure, ``summary.json`` stays
incomplete and records universe, measured-mask, label, SMILES, and active-pair
diagnostics alongside a clearly named incomplete candidate.

A final source-derived run on 2026-08-05 produced 17,892 compounds, 102 targets,
and 68,211 measured annotations.
Its 3,100,787-byte CSV was byte-identical to the pinned Zenodo reference.

Usage
-----
Run ``direnv exec . just get-chembl-target-sources`` to acquire and verify the
reviewable sources, then ``direnv exec . just build-chembl-targets``.
The script uses the PEP 723 environment pinned above and deliberately performs
no network access itself.
The downstream production path remains Zenodo CSV -> ``data/external`` ->
``chembl.py`` -> ``chembl_protein_targets`` -> augmented DuckDB; only positive
labels are loaded into ``chembl_protein_targets``.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import re
import shutil
import sqlite3
import sys
import tarfile
import warnings
from collections import Counter
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

CHEMBL_RELEASE = "ChEMBL_33"
CHEMBL_RELEASE_DATE = "2023-05-31"
OFFICIAL_ARCHIVE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz"
OFFICIAL_ARCHIVE_SIZE = 4_735_615_261
OFFICIAL_ARCHIVE_SHA256 = "4e14ace2dc3696cd414c25cbf2370a618fd5ac028713b877725c8ae308e86e8b"
OFFICIAL_DIRECTORY_MEMBER = "chembl_33/chembl_33_sqlite"
OFFICIAL_INSTALL_MEMBER = f"{OFFICIAL_DIRECTORY_MEMBER}/INSTALL_sqlite"
OFFICIAL_DATABASE_MEMBER = f"{OFFICIAL_DIRECTORY_MEMBER}/chembl_33.db"
OFFICIAL_INSTALL_SIZE = 541
OFFICIAL_DATABASE_SIZE = 24_663_986_176
OFFICIAL_DATABASE_SHA256 = "fbc426749b425d801b9973d8522a45e04cc1a01ff8cd5217dcda5c71587b7965"
CHEMBL_ATTRIBUTION = "ChEMBL data is from https://www.ebi.ac.uk/chembl - the version of ChEMBL is chembl_33."

HISTORICAL_REFERENCE_URL = "https://github.com/David-Araripe/Capricho"
HISTORICAL_REFERENCE_COMMIT = "3888e2806a63f6d9e231856283931a93be37f4ef"
BEHAVIORAL_REFERENCE_URL = "https://github.com/David-Araripe/chembl-annotations_jump-CP"
BEHAVIORAL_REFERENCE_COMMIT = "e3d22f4b70ec2ffa53612596866a4e56df88454b"
SMALLWORLD_SOURCE_URL = "https://zenodo.org/records/21816611/files/jump_to_chembl.csv.gz"
SMALLWORLD_SOURCE_STATUS = (
    "Published unchanged in Zenodo record 21816611 (doi:10.5281/zenodo.21816611) with an exact SHA-256 pin."
)
JUMP_COMPOUND_SOURCE_URL = "https://github.com/jump-cellpainting/datasets/raw/main/metadata/compound.csv.gz"

DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "chembl-33"
DEFAULT_ARCHIVE = DEFAULT_CACHE_DIR / "chembl_33_sqlite.tar.gz"
DEFAULT_DATABASE = DEFAULT_CACHE_DIR / OFFICIAL_DATABASE_MEMBER
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "chembl-target-annotations-chembl33"
DEFAULT_REFERENCE = EXTERNAL_DATA_DIR / "targetannotations_singleproteins_jumpcompounds_all_5.csv"
DEFAULT_JUMP_COMPOUNDS = EXTERNAL_DATA_DIR / "compound.csv.gz"

EXTRACTION_MANIFEST_NAME = "chembl_33_extraction.json"
SUMMARY_NAME = "summary.json"
RAW_ACTIVITY_NAME = "chembl_activities_raw.csv.gz"
CURATED_ACTIVITY_NAME = "chembl_activities_curated.csv.gz"
ASSOCIATION_AUDIT_NAME = "metadata_association_audit.csv.gz"
LONG_ANNOTATION_NAME = "target_annotations_long.csv.gz"
TARGET_COUNTS_NAME = "target_class_counts.csv"
WIDE_MATRIX_NAME = "targetannotations_singleproteins_jumpcompounds_all_5.csv"
INCOMPLETE_CANDIDATE_NAME = "targetannotations_singleproteins_jumpcompounds_all_5.candidate_incomplete.csv"

CANONICAL_REFERENCE_SIZE = 3_100_787
CANONICAL_REFERENCE_SHA256 = "b299aabfd07a4b6685c9eea0327d762a3d20b71434c82a646a24d76c32c46480"
CANONICAL_REFERENCE_URL = (
    "https://zenodo.org/records/21800106/files/targetannotations_singleproteins_jumpcompounds_all_5.csv"
)
HISTORICAL_OUTPUT_SHA256 = {
    RAW_ACTIVITY_NAME: "2266f8392e44922d38acd5f66c36eeda84d625be10d333e0ecd4b659bf735f97",
    CURATED_ACTIVITY_NAME: "dcca68d52697b2dec4e34deca7f5d2c0c5c537f7baaa0356148a7fdbf066c7a3",
    LONG_ANNOTATION_NAME: "935e149a797e7be1f965fcfc4f55ce7870a2324750582091e06f12c18fc0421f",
    WIDE_MATRIX_NAME: CANONICAL_REFERENCE_SHA256,
}

ACTIVITY_THRESHOLD = 5.0
MINIMUM_PER_CLASS = 15
SELECTED_ACTIVITY_TYPES = ("IC50", "Potency", "Ki", "Kd", "AC50", "EC50")


@dataclass(frozen=True)
class CsvSourcePin:
    """Exact byte identity and decoded shape of one CSV source."""

    filename: str
    size_bytes: int
    sha256: str
    rows: int
    columns: tuple[str, ...]


SMALLWORLD_MAPPING_PIN = CsvSourcePin(
    filename="jump_to_chembl.csv.gz",
    size_bytes=248_071,
    sha256="50c038187761f3e8bbe998fea22525afe13c3cc01a14178ab3175335464656ac",
    rows=36_488,
    columns=("JUMP_ID", "molecule_chembl_id", "hit_rank"),
)
JUMP_COMPOUND_SOURCE_PIN = CsvSourcePin(
    filename="compound.csv.gz",
    size_bytes=8_595_537,
    sha256="8885960e92ebd99eb33699a79129f517e668d78dd94f0d7478d39c9825bd3c0a",
    rows=115_796,
    columns=("Metadata_JCP2022", "Metadata_InChIKey", "Metadata_InChI", "Metadata_SMILES"),
)
EXPECTED_SMALLWORLD_COUNTS = {
    "rows": 36_488,
    "jump_ids": 28_782,
    "chembl_ids": 36_449,
    "nonzero_rank_rows": 7_706,
}
EXPECTED_COMPOUND_DERIVATION_COUNTS = {
    "mapped_jump_ids": 28_782,
    "source_rows": 28_781,
    "fallback_rows": 1,
    "output_rows": 28_782,
}
EXPECTED_FALLBACK_MOLECULES = {"JCP2022_088778": "CHEMBL2074916"}
EXPECTED_TARGET_MAPPING_COUNTS = {
    "rows": 3_356,
    "target_chembl_ids": 3_235,
    "uniprot_accessions": 3_295,
}

EXPECTED_ACTIVITY_COUNTS = {
    "mapping_rows": 36_488,
    "sql_rows_before_pchembl_processing": 229_776,
    "raw_activity_rows": 149_982,
    "unresolved_mixture_rows": 100,
    "missing_pchembl_rows": 29,
    "excluded_activity_type_rows": 3_413,
    "curated_activity_rows": 146_440,
    "metadata_reassigned_rows": 16_734,
    "metadata_reassigned_jump_ids": 757,
}

EXPECTED_ANNOTATION_COUNTS = {
    "curated_activity_rows": 146_440,
    "mapped_activity_rows": 139_224,
    "mean_compound_target_pairs": 92_062,
    "compounds": 17_892,
    "targets": 102,
    "annotations": 68_211,
    "active": 24_029,
    "inactive": 44_182,
    "human_targets": 77,
}

REQUIRED_DATABASE_COLUMNS = {
    "activities": {
        "activity_id",
        "assay_id",
        "molregno",
        "pchembl_value",
        "relation",
        "standard_flag",
        "standard_relation",
        "standard_type",
        "standard_units",
        "standard_value",
        "type",
        "units",
        "value",
        "data_validity_comment",
        "potential_duplicate",
    },
    "assays": {
        "assay_id",
        "chembl_id",
        "description",
        "assay_type",
        "tid",
        "doc_id",
        "assay_cell_type",
        "assay_organism",
        "assay_subcellular_fraction",
        "assay_tissue",
        "confidence_score",
        "relationship_type",
        "variant_id",
    },
    "molecule_dictionary": {
        "molregno",
        "chembl_id",
        "chemical_probe",
        "chirality",
        "indication_class",
        "max_phase",
        "oral",
        "prodrug",
        "withdrawn_flag",
    },
    "target_dictionary": {"tid", "chembl_id", "organism", "target_type", "pref_name", "tax_id"},
    "target_components": {"tid", "component_id"},
    "component_sequences": {"component_id", "accession", "organism", "tax_id", "db_source"},
    "docs": {"doc_id", "chembl_id"},
    "assay_type": {"assay_type", "assay_desc"},
    "relationship_type": {"relationship_type", "relationship_desc"},
    "variant_sequences": {"variant_id", "mutation"},
    "molecule_hierarchy": {"molregno", "active_molregno", "parent_molregno"},
    "compound_structures": {"molregno", "canonical_smiles", "standard_inchi_key"},
    "version": {"name", "creation_date", "comments"},
}

RAW_COLUMNS = (
    "JUMP_ID",
    "molecule_chembl_id",
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "pchembl_value",
    "relation",
    "standard_flag",
    "standard_relation",
    "standard_type",
    "standard_units",
    "standard_value",
    "target_chembl_id",
    "target_organism",
    "type",
    "units",
    "value",
    "assay_cell_type",
    "assay_organism",
    "assay_subcellular_fraction",
    "assay_tissue",
    "assay_type_description",
    "confidence_score",
    "description",
    "document_chembl_id",
    "relationship_description",
    "relationship_type",
    "variant_sequence",
    "hierarchy_active_id",
    "hierarchy_molecule_id",
    "hierarchy_parent_id",
    "canonical_smiles",
    "standard_inchikey",
    "chemical_probe",
    "chirality",
    "indication_class",
    "max_phase",
    "oral",
    "prodrug",
    "withdrawn_flag",
)

CURATED_COLUMNS = (
    "JUMP_ID",
    "molecule_chembl_id",
    "activity_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "pchembl_value",
    "standard_flag",
    "standard_relation",
    "standard_type",
    "standard_units",
    "target_chembl_id",
    "target_organism",
    "assay_cell_type",
    "assay_organism",
    "assay_subcellular_fraction",
    "assay_tissue",
    "assay_type_description",
    "confidence_score",
    "description",
    "document_chembl_id",
    "relationship_description",
    "relationship_type",
    "variant_sequence",
    "hierarchy_active_id",
    "hierarchy_molecule_id",
    "hierarchy_parent_id",
    "canonical_smiles",
    "standard_inchikey",
    "chemical_probe",
    "chirality",
    "indication_class",
    "max_phase",
    "oral",
    "prodrug",
    "withdrawn_flag",
    "standard_smiles",
    "type",
)

ACTIVITY_SQL = """
SELECT
    jm.jump_id AS JUMP_ID,
    molecule.chembl_id AS molecule_chembl_id,
    activity.activity_id,
    assay.chembl_id AS assay_chembl_id,
    assay.description AS assay_description,
    assay.assay_type,
    activity.pchembl_value,
    activity.relation,
    activity.standard_flag,
    activity.standard_relation,
    activity.standard_type,
    activity.standard_units,
    activity.standard_value,
    target.chembl_id AS target_chembl_id,
    target.organism AS target_organism,
    activity.type,
    activity.units,
    activity.value,
    assay.assay_cell_type,
    assay.assay_organism,
    assay.assay_subcellular_fraction,
    assay.assay_tissue,
    assay_kind.assay_desc AS assay_type_description,
    assay.confidence_score,
    assay.description AS description,
    document.chembl_id AS document_chembl_id,
    relationship.relationship_desc AS relationship_description,
    assay.relationship_type,
    variant.mutation AS variant_sequence,
    activity.pchembl_value IS NULL AS _pchem_was_missing
FROM temp.jump_mapping AS jm
JOIN molecule_dictionary AS molecule
  ON molecule.chembl_id = jm.molecule_chembl_id
JOIN activities AS activity
  ON activity.molregno = molecule.molregno
JOIN assays AS assay
  ON assay.assay_id = activity.assay_id
LEFT JOIN target_dictionary AS target
  ON target.tid = assay.tid
LEFT JOIN docs AS document
  ON document.doc_id = assay.doc_id
LEFT JOIN assay_type AS assay_kind
  ON assay_kind.assay_type = assay.assay_type
LEFT JOIN relationship_type AS relationship
  ON relationship.relationship_type = assay.relationship_type
LEFT JOIN variant_sequences AS variant
  ON variant.variant_id = assay.variant_id
WHERE activity.standard_relation = '='
  AND assay.assay_type IN ('B', 'F')
  AND assay.confidence_score IN (6, 7, 8, 9)
  AND activity.data_validity_comment IS NULL
  AND activity.potential_duplicate = 0
ORDER BY jm.input_order, activity.activity_id
""".strip()

MOLECULE_SQL = """
SELECT
    molecule.chembl_id AS metadata_source_id,
    molecule.molregno,
    active_molecule.chembl_id AS hierarchy_active_id,
    molecule.chembl_id AS hierarchy_molecule_id,
    parent_molecule.chembl_id AS hierarchy_parent_id,
    structure.canonical_smiles,
    structure.standard_inchi_key AS standard_inchikey,
    molecule.chemical_probe,
    molecule.chirality,
    molecule.indication_class,
    molecule.max_phase,
    molecule.oral,
    molecule.prodrug,
    molecule.withdrawn_flag
FROM molecule_dictionary AS molecule
JOIN (
    SELECT DISTINCT molecule_chembl_id
    FROM temp.jump_mapping
) AS requested
  ON requested.molecule_chembl_id = molecule.chembl_id
LEFT JOIN molecule_hierarchy AS hierarchy
  ON hierarchy.molregno = molecule.molregno
LEFT JOIN molecule_dictionary AS active_molecule
  ON active_molecule.molregno = hierarchy.active_molregno
LEFT JOIN molecule_dictionary AS parent_molecule
  ON parent_molecule.molregno = hierarchy.parent_molregno
LEFT JOIN compound_structures AS structure
  ON structure.molregno = molecule.molregno
ORDER BY molecule.molregno
""".strip()

TARGET_MAPPING_SQL = """
WITH swiss_single_entries AS (
    SELECT DISTINCT component.accession
    FROM target_dictionary AS target
    JOIN target_components AS link
      ON link.tid = target.tid
    JOIN component_sequences AS component
      ON component.component_id = link.component_id
    WHERE component.db_source = 'SWISS-PROT'
      AND component.accession IS NOT NULL
      AND target.target_type = 'SINGLE PROTEIN'
)
SELECT DISTINCT
    target.chembl_id AS "From",
    component.accession AS "Entry",
    CASE
        WHEN component.tax_id = 9606 THEN 'Homo sapiens (Human)'
        ELSE COALESCE(component.organism, target.organism)
    END AS "Organism",
    COALESCE(target.pref_name, '') AS "Target Name"
FROM target_dictionary AS target
JOIN temp.activity_targets AS requested
  ON requested.target_chembl_id = target.chembl_id
JOIN target_components AS link
  ON link.tid = target.tid
JOIN component_sequences AS component
  ON component.component_id = link.component_id
WHERE component.db_source = 'SWISS-PROT'
  AND component.accession IS NOT NULL
  AND (
      target.target_type = 'SINGLE PROTEIN'
      OR (
          target.target_type = 'PROTEIN COMPLEX'
          AND component.accession NOT IN (SELECT accession FROM swiss_single_entries)
      )
  )
ORDER BY "From", "Entry"
""".strip()

FIXED_RESPONSE_GROUP = "JCP2022_059269"
FIXED_REQUEST_ORDER = ("CHEMBL19611", "CHEMBL304045", "CHEMBL71801")

_SALT_OR_SOLVENT_PATTERN = re.compile(
    r"^(?:"
    r"Na\+?|Cl-?|Br-?|K\+?|F-?|I-?|Ca\+\+|Mg\+\+|Zn\+\+|OH-?|"
    r"\[Na\+\]?|\[K\+\]?|\[Cl-\]?|\[Br-\]?|\[I-\]?|\[Zn\+2\]?|"
    r"CCCC\(=O\)\[O-\]|CCCCC\(=O\)\[O-\]|"
    r"C1=CC=C\(C=C1\)C\(=O\)\[O-\]|"
    r"\[O-\]\[Cl\+3\]\(\[O-\]\)\(\[O-\]\)\[O-\]|"
    r"C1=CC=NC=C1|O=CN\(C\)C|\[NO\]|"
    r"c1ccc\(\[B-\]\(c2ccccc2\)\(c2ccccc2\)c2ccccc2\)cc1|O|N"
    r")$"
)

StandardizeFunction = Callable[[list[str], int], list[str]]


class SourceIntegrityError(RuntimeError):
    """Raised when a source fails a pinned identity or safety check."""


class ExactReproductionError(RuntimeError):
    """Raised when computed results differ from the published artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, encoded)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic gzip bytes while retaining the canonical header name."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    header_name = path.name.removesuffix(".gz")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename=header_name,
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            frame.to_csv(text, index=False, lineterminator="\n")
            text.flush()
            text.detach()
    temporary.replace(path)


def _write_incomplete_summary(
    path: Path,
    *,
    stage: str,
    error: BaseException | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "description": "Exact JUMP ChEMBL 33 target-matrix reproduction",
        "status": "incomplete",
        "stage": stage,
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    if details:
        payload.update(details)
    _atomic_json(path, payload)


def _is_within(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def _guard_paths(
    *,
    output_dir: Path,
    archive: Path,
    database: Path,
    reference: Path,
    compound_source: Path | None = None,
) -> None:
    if _is_within(output_dir, DATA_DIR):
        raise SourceIntegrityError("Output directory must remain outside the mirrored data directory")
    for label, path in (("archive", archive), ("database", database)):
        if _is_within(path, DATA_DIR):
            raise SourceIntegrityError(f"ChEMBL {label} must remain outside the mirrored data directory")
    if _is_within(archive, output_dir) or _is_within(database, output_dir):
        raise SourceIntegrityError("ChEMBL archive and database must remain outside the generated package")
    if _is_within(reference, output_dir):
        raise SourceIntegrityError("Canonical reference must remain outside the generated package")
    if compound_source is not None and _is_within(compound_source, output_dir):
        raise SourceIntegrityError("JUMP compound source must remain outside the generated package")


def _verify_csv_source(
    path: Path,
    pin: CsvSourcePin,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify exact bytes before trusting the decoded table shape."""
    if path.name != pin.filename or not path.is_file():
        raise SourceIntegrityError(f"{label} is missing: {path}")
    observed_size = path.stat().st_size
    if observed_size != pin.size_bytes:
        raise SourceIntegrityError(f"{label} {pin.filename} has size {observed_size:,}; expected {pin.size_bytes:,}")
    observed_sha256 = _sha256(path)
    if observed_sha256 != pin.sha256:
        raise SourceIntegrityError(f"{label} {pin.filename} has SHA-256 {observed_sha256}; expected {pin.sha256}")
    frame = pd.read_csv(path, low_memory=False)
    observed_columns = tuple(str(column) for column in frame.columns)
    if observed_columns != pin.columns:
        raise SourceIntegrityError(f"{label} {pin.filename} has columns {observed_columns}; expected {pin.columns}")
    if len(frame) != pin.rows:
        raise SourceIntegrityError(f"{label} {pin.filename} has {len(frame):,} rows; expected {pin.rows:,}")
    return frame, {
        **asdict(pin),
        "path": str(path.resolve()),
        "role": label,
        "verified": True,
    }


def verify_smallworld_mapping(
    path: Path,
    pin: CsvSourcePin = SMALLWORLD_MAPPING_PIN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the exact SmallWorld bytes and the meaning of its ranked rows."""
    frame, metadata = _verify_csv_source(path, pin, label="SmallWorld mapping")
    if frame.isna().any().any():
        raise SourceIntegrityError("SmallWorld mapping contains null values")
    if frame.duplicated(["JUMP_ID", "molecule_chembl_id"]).any():
        raise SourceIntegrityError("SmallWorld mapping contains duplicate JUMP-ChEMBL pairs")

    numeric_ranks = pd.to_numeric(frame["hit_rank"], errors="coerce")
    if numeric_ranks.isna().any() or not numeric_ranks.eq(numeric_ranks.astype(int)).all():
        raise SourceIntegrityError("SmallWorld hit_rank must contain only integers")
    expected_ranks = frame.groupby("JUMP_ID", sort=False).cumcount()
    if not numeric_ranks.astype(int).reset_index(drop=True).equals(expected_ranks.reset_index(drop=True)):
        raise SourceIntegrityError("SmallWorld hit_rank is not contiguous in saved response order")

    counts = {
        "rows": int(len(frame)),
        "jump_ids": int(frame["JUMP_ID"].nunique()),
        "chembl_ids": int(frame["molecule_chembl_id"].nunique()),
        "nonzero_rank_rows": int(numeric_ranks.ne(0).sum()),
    }
    if pin == SMALLWORLD_MAPPING_PIN and counts != EXPECTED_SMALLWORLD_COUNTS:
        raise SourceIntegrityError(f"Unexpected SmallWorld mapping counts: {counts}")
    return frame, {**metadata, "semantic_counts": counts, "semantic_checks_passed": True}


def verify_jump_compound_source(
    path: Path,
    pin: CsvSourcePin = JUMP_COMPOUND_SOURCE_PIN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify the JUMP source already maintained by download_data.py."""
    frame, metadata = _verify_csv_source(path, pin, label="JUMP compound source")
    if frame["Metadata_JCP2022"].isna().any():
        raise SourceIntegrityError("JUMP compound source contains a null Metadata_JCP2022")
    if frame["Metadata_JCP2022"].duplicated().any():
        raise SourceIntegrityError("JUMP compound source contains duplicate Metadata_JCP2022 values")
    chemistry_columns = ["Metadata_InChIKey", "Metadata_InChI", "Metadata_SMILES"]
    missing_chemistry = set(frame.loc[frame[chemistry_columns].isna().all(axis=1), "Metadata_JCP2022"])
    if pin == JUMP_COMPOUND_SOURCE_PIN and missing_chemistry != {"JCP2022_999999"}:
        raise SourceIntegrityError(
            f"JUMP compound source has an unexpected set of all-null chemistry rows: {sorted(missing_chemistry)}"
        )
    return frame, {
        **metadata,
        "unique_jump_ids": int(frame["Metadata_JCP2022"].nunique()),
        "all_null_chemistry_jump_ids": sorted(str(value) for value in missing_chemistry),
        "semantic_checks_passed": True,
        "source_url": JUMP_COMPOUND_SOURCE_URL,
    }


def load_smallworld_mapping(output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require the cache to contain exactly the one historical artifact."""
    input_dir = output_dir / "inputs"
    if not input_dir.is_dir():
        raise SourceIntegrityError(f"SmallWorld input directory is missing: {input_dir}")
    observed = sorted(path.name for path in input_dir.iterdir())
    expected = [SMALLWORLD_MAPPING_PIN.filename]
    if observed != expected:
        raise SourceIntegrityError(
            f"Historical input cache must contain only the SmallWorld mapping; observed {observed}, expected {expected}"
        )
    return verify_smallworld_mapping(input_dir / SMALLWORLD_MAPPING_PIN.filename)


def verify_canonical_reference(reference: Path) -> tuple[bytes, dict[str, Any]]:
    if not reference.is_file():
        raise SourceIntegrityError(f"Canonical Zenodo reference is missing: {reference}")
    observed_size = reference.stat().st_size
    if observed_size != CANONICAL_REFERENCE_SIZE:
        raise SourceIntegrityError(
            f"Canonical reference has size {observed_size:,}; expected {CANONICAL_REFERENCE_SIZE:,}"
        )
    payload = reference.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != CANONICAL_REFERENCE_SHA256:
        raise SourceIntegrityError(
            f"Canonical reference SHA-256 is {observed_sha256}; expected {CANONICAL_REFERENCE_SHA256}"
        )
    return payload, {
        "path": str(reference.resolve()),
        "url": CANONICAL_REFERENCE_URL,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "verified": True,
    }


def verify_official_archive(
    archive: Path,
    *,
    expected_size: int = OFFICIAL_ARCHIVE_SIZE,
    expected_sha256: str = OFFICIAL_ARCHIVE_SHA256,
) -> dict[str, Any]:
    if not archive.is_file():
        raise SourceIntegrityError(
            f"Official ChEMBL 33 archive not found: {archive}. Download {OFFICIAL_ARCHIVE_URL} first."
        )
    observed_size = archive.stat().st_size
    if observed_size != expected_size:
        raise SourceIntegrityError(
            f"Official archive has size {observed_size:,}; expected {expected_size:,}: {archive}"
        )
    observed_sha256 = _sha256(archive)
    if observed_sha256 != expected_sha256:
        raise SourceIntegrityError(
            f"Official archive SHA-256 is {observed_sha256}; expected {expected_sha256}: {archive}"
        )
    return {
        "path": str(archive.resolve()),
        "url": OFFICIAL_ARCHIVE_URL,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "verified": True,
    }


def _validate_tar_member(member: tarfile.TarInfo) -> str:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise SourceIntegrityError(f"Unsafe path in official archive: {member.name!r}")
    normalized = member.name.rstrip("/")
    expected = {
        OFFICIAL_DIRECTORY_MEMBER,
        OFFICIAL_INSTALL_MEMBER,
        OFFICIAL_DATABASE_MEMBER,
    }
    if normalized not in expected:
        raise SourceIntegrityError(f"Unexpected member in official archive: {member.name!r}")
    if normalized == OFFICIAL_DIRECTORY_MEMBER:
        if not member.isdir():
            raise SourceIntegrityError("Expected ChEMBL archive directory is not a directory")
    elif not member.isfile():
        raise SourceIntegrityError(f"Official archive payload is not a regular file: {member.name!r}")
    return normalized


def _copy_stream_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> Path:
    source = archive.extractfile(member)
    if source is None:
        raise SourceIntegrityError(f"Could not read official archive member {member.name!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    incomplete = destination.with_suffix(destination.suffix + ".incomplete")
    with source, incomplete.open("wb") as handle:
        shutil.copyfileobj(source, handle, length=16 * 1024 * 1024)
    if incomplete.stat().st_size != member.size:
        raise SourceIntegrityError(
            f"Extracted {member.name!r} to {incomplete.stat().st_size:,} bytes; expected {member.size:,}"
        )
    return incomplete


def extract_official_database(
    archive_path: Path,
    database_path: Path,
    *,
    expected_database_size: int = OFFICIAL_DATABASE_SIZE,
) -> dict[str, Any]:
    """Stream and validate the entire tar while copying only two regular files."""
    install_path = database_path.parent / "INSTALL_sqlite"
    incomplete_files: dict[str, Path] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                normalized = _validate_tar_member(member)
                if normalized in seen:
                    raise SourceIntegrityError(f"Duplicate member in official archive: {member.name!r}")
                seen.add(normalized)
                if normalized == OFFICIAL_DIRECTORY_MEMBER:
                    continue
                expected_size = (
                    OFFICIAL_INSTALL_SIZE if normalized == OFFICIAL_INSTALL_MEMBER else expected_database_size
                )
                if member.size != expected_size:
                    raise SourceIntegrityError(
                        f"Official member {member.name!r} has size {member.size:,}; expected {expected_size:,}"
                    )
                destination = install_path if normalized == OFFICIAL_INSTALL_MEMBER else database_path
                incomplete_files[normalized] = _copy_stream_member(archive, member, destination)
    except (tarfile.TarError, OSError) as error:
        raise SourceIntegrityError(f"Could not safely extract {archive_path}: {error}") from error

    expected_members = {
        OFFICIAL_DIRECTORY_MEMBER,
        OFFICIAL_INSTALL_MEMBER,
        OFFICIAL_DATABASE_MEMBER,
    }
    if seen != expected_members:
        raise SourceIntegrityError(f"Official archive member set differs from the release pin: {sorted(seen)}")
    incomplete_files[OFFICIAL_INSTALL_MEMBER].replace(install_path)
    incomplete_files[OFFICIAL_DATABASE_MEMBER].replace(database_path)
    return {
        "database_member": OFFICIAL_DATABASE_MEMBER,
        "database_size_bytes": database_path.stat().st_size,
        "install_member": OFFICIAL_INSTALL_MEMBER,
        "install_size_bytes": install_path.stat().st_size,
    }


def ensure_official_database(archive_path: Path, database_path: Path) -> dict[str, Any]:
    """Establish byte provenance from the archive to a reusable local database."""
    archive_metadata = verify_official_archive(archive_path)
    manifest_path = database_path.parent / EXTRACTION_MANIFEST_NAME
    extracted = False
    extraction_metadata: dict[str, Any] = {}

    if not database_path.exists():
        extraction_metadata = extract_official_database(archive_path, database_path)
        extracted = True
    elif not manifest_path.is_file():
        raise SourceIntegrityError(
            "Existing ChEMBL database lacks an extraction manifest. Relocate it and let this script "
            f"extract the pinned archive: {database_path}"
        )
    if not database_path.is_file():
        raise SourceIntegrityError(f"ChEMBL database is not a regular file: {database_path}")
    if database_path.stat().st_size != OFFICIAL_DATABASE_SIZE:
        raise SourceIntegrityError(
            f"ChEMBL database has size {database_path.stat().st_size:,}; expected {OFFICIAL_DATABASE_SIZE:,}"
        )

    database_sha256 = _sha256(database_path)
    if database_sha256 != OFFICIAL_DATABASE_SHA256:
        raise SourceIntegrityError(
            f"Extracted ChEMBL database SHA-256 is {database_sha256}; expected {OFFICIAL_DATABASE_SHA256}"
        )
    expected_manifest = {
        "archive_sha256": archive_metadata["sha256"],
        "archive_size_bytes": archive_metadata["size_bytes"],
        "archive_url": OFFICIAL_ARCHIVE_URL,
        "database_member": OFFICIAL_DATABASE_MEMBER,
        "database_sha256": database_sha256,
        "database_size_bytes": OFFICIAL_DATABASE_SIZE,
    }
    if extracted:
        _atomic_json(
            manifest_path,
            {
                **expected_manifest,
                "archive_path": str(archive_path.resolve()),
                "database_path": str(database_path.resolve()),
                **extraction_metadata,
            },
        )
    else:
        try:
            observed_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SourceIntegrityError(f"Malformed extraction manifest {manifest_path}: {error}") from error
        mismatches = {
            key: (observed_manifest.get(key), expected)
            for key, expected in expected_manifest.items()
            if observed_manifest.get(key) != expected
        }
        if mismatches:
            raise SourceIntegrityError(f"Extraction manifest does not match local ChEMBL source: {mismatches}")

    return {
        "archive": archive_metadata,
        "database": {
            "path": str(database_path.resolve()),
            "member": OFFICIAL_DATABASE_MEMBER,
            "size_bytes": OFFICIAL_DATABASE_SIZE,
            "sha256": database_sha256,
            "extracted_by_generator": extracted,
            "manifest_path": str(manifest_path.resolve()),
            "verified": True,
        },
    }


@contextmanager
def open_chembl_database(database_path: Path) -> Iterator[sqlite3.Connection]:
    uri = f"{database_path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        yield connection
    finally:
        connection.close()


def validate_database(connection: sqlite3.Connection) -> dict[str, Any]:
    """Check integrity, schema, and exact ChEMBL release before scientific use."""
    quick_check = connection.execute("PRAGMA quick_check").fetchone()
    if quick_check is None or quick_check[0] != "ok":
        raise SourceIntegrityError(f"SQLite quick_check failed: {quick_check}")

    available = {
        str(name): str(kind)
        for name, kind in connection.execute(
            "SELECT name, type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    missing_tables = sorted(table for table in REQUIRED_DATABASE_COLUMNS if available.get(table) != "table")
    if missing_tables:
        raise SourceIntegrityError(f"ChEMBL database is missing required tables: {missing_tables}")

    schema: dict[str, list[str]] = {}
    missing_columns: dict[str, list[str]] = {}
    for table, required in REQUIRED_DATABASE_COLUMNS.items():
        observed = [str(row[1]) for row in connection.execute(f'PRAGMA main.table_info("{table}")').fetchall()]
        schema[table] = observed
        missing = sorted(required - set(observed))
        if missing:
            missing_columns[table] = missing
    if missing_columns:
        raise SourceIntegrityError(f"ChEMBL database is missing required columns: {missing_columns}")

    version_rows = [
        {"name": row[0], "creation_date": row[1], "comments": row[2]}
        for row in connection.execute("SELECT name, creation_date, comments FROM version ORDER BY name").fetchall()
    ]
    releases = [
        row for row in version_rows if isinstance(row["name"], str) and re.fullmatch(r"ChEMBL_\d+", row["name"])
    ]
    if len(releases) != 1 or releases[0]["name"] != CHEMBL_RELEASE:
        raise SourceIntegrityError(f"Expected exactly {CHEMBL_RELEASE}, found {[row['name'] for row in releases]}")
    release_date = str(releases[0]["creation_date"])
    if not release_date.startswith(CHEMBL_RELEASE_DATE):
        raise SourceIntegrityError(f"Expected ChEMBL 33 release date {CHEMBL_RELEASE_DATE}, found {release_date}")

    canonical_versions = json.dumps(version_rows, sort_keys=True, separators=(",", ":"), default=str)
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return {
        "release": CHEMBL_RELEASE,
        "release_date": release_date,
        "quick_check": "ok",
        "version_rows": version_rows,
        "version_table_sha256": hashlib.sha256(canonical_versions.encode()).hexdigest(),
        "required_schema_sha256": hashlib.sha256(canonical_schema.encode()).hexdigest(),
        "sqlite_runtime_version": sqlite3.sqlite_version,
        "read_only_uri_parameters": {"mode": "ro", "immutable": 1},
    }


def extract_database_tables(
    connection: sqlite3.Connection,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve filtered activity rows and compound metadata in two local queries."""
    expected_columns = SMALLWORLD_MAPPING_PIN.columns
    if tuple(str(column) for column in mapping.columns) != expected_columns:
        raise ValueError(f"Unexpected mapping columns: {mapping.columns.tolist()}")
    connection.execute("DROP TABLE IF EXISTS temp.jump_mapping")
    connection.execute(
        "CREATE TEMP TABLE jump_mapping (jump_id TEXT, molecule_chembl_id TEXT, hit_rank INTEGER, input_order INTEGER)"
    )
    records = (
        (*row, input_order)
        for input_order, row in enumerate(mapping.loc[:, list(expected_columns)].itertuples(index=False, name=None))
    )
    connection.executemany("INSERT INTO temp.jump_mapping VALUES (?, ?, ?, ?)", records)
    connection.execute("CREATE INDEX temp.idx_jump_mapping_molecule ON jump_mapping(molecule_chembl_id)")
    try:
        activities = pd.read_sql_query(ACTIVITY_SQL, connection)
        molecules = pd.read_sql_query(MOLECULE_SQL, connection)
    finally:
        connection.execute("DROP TABLE temp.jump_mapping")
    return activities, molecules


def extract_target_mapping(
    connection: sqlite3.Connection,
    target_ids: Collection[str],
) -> pd.DataFrame:
    """Derive the historical target-map semantics from native ChEMBL tables."""
    requested = sorted({str(value) for value in target_ids if pd.notna(value)})
    if not requested:
        raise SourceIntegrityError("Cannot derive a target mapping without ChEMBL target identifiers")
    connection.execute("DROP TABLE IF EXISTS temp.activity_targets")
    connection.execute("CREATE TEMP TABLE activity_targets (target_chembl_id TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO temp.activity_targets VALUES (?)", ((value,) for value in requested))
    try:
        mapping = pd.read_sql_query(TARGET_MAPPING_SQL, connection)
    finally:
        connection.execute("DROP TABLE temp.activity_targets")
    if mapping.empty:
        raise SourceIntegrityError("ChEMBL target component query returned no mappings")
    if mapping[["From", "Entry"]].isna().any().any():
        raise SourceIntegrityError("ChEMBL target component query returned a null identifier")
    if mapping.duplicated(["From", "Entry"]).any():
        raise SourceIntegrityError("ChEMBL target component query returned duplicate mappings")
    return mapping


def calculate_missing_pchembl(activities: pd.DataFrame) -> pd.DataFrame:
    """Preserve the historical float32 conversion and supported-unit ordering."""
    converted = activities.copy()
    converted["standard_value"] = converted["standard_value"].astype("float32")
    converted["pchembl_value"] = converted["pchembl_value"].astype("float32")

    retained = [converted.loc[converted["pchembl_value"].notna()].copy()]
    for unit in ("nM", chr(181) + "M", "uM"):
        rows = converted.loc[converted["pchembl_value"].isna() & converted["standard_units"].eq(unit)].copy()
        if rows.empty:
            continue

        def calculate(row: pd.Series) -> float:
            factor = 1e-9 if row["standard_units"] == "nM" else 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(-np.log10(row["standard_value"] * factor))

        rows = rows.assign(
            pchembl_value=lambda frame: frame.apply(calculate, axis=1),
        )
        retained.append(rows)
    return cast(pd.DataFrame, pd.concat(retained, ignore_index=True))


def associate_historical_compound_metadata(
    activities: pd.DataFrame,
    molecules: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Reconstruct the positional association used by the historical batched client."""
    request_order = activities.sort_values(
        ["JUMP_ID", "_pchem_was_missing", "activity_id"],
        kind="mergesort",
    )[["JUMP_ID", "molecule_chembl_id"]].drop_duplicates()
    molecule_numbers = molecules.set_index("metadata_source_id")["molregno"].to_dict()
    fixed_rank = {molecule_id: rank for rank, molecule_id in enumerate(FIXED_REQUEST_ORDER)}

    assignments: list[tuple[str, str, str]] = []
    for jump_id, group in request_order.groupby("JUMP_ID", sort=False):
        requested = group["molecule_chembl_id"].tolist()
        if jump_id == FIXED_RESPONSE_GROUP:
            unknown = sorted(set(requested) - set(FIXED_REQUEST_ORDER))
            if unknown:
                raise SourceIntegrityError(f"Unexpected molecules in fixed historical response group: {unknown}")
            requested = sorted(requested, key=fixed_rank.__getitem__)
        missing = sorted(set(requested) - set(molecule_numbers))
        if missing:
            raise SourceIntegrityError(f"Missing ChEMBL molecule metadata for {missing[:3]}")
        responses = sorted(requested, key=molecule_numbers.__getitem__)
        assignments.extend(
            (str(jump_id), str(requested_id), str(response_id))
            for requested_id, response_id in zip(requested, responses, strict=True)
        )

    association = pd.DataFrame(
        assignments,
        columns=pd.Index(["JUMP_ID", "molecule_chembl_id", "metadata_source_id"]),
    )
    association["metadata_reassigned"] = association["molecule_chembl_id"] != association["metadata_source_id"]
    associated = activities.merge(
        association,
        on=["JUMP_ID", "molecule_chembl_id"],
        validate="many_to_one",
    )
    reassigned_rows = associated["metadata_reassigned"]
    counts = {
        "assignments": int(len(association)),
        "reassigned_assignments": int(association["metadata_reassigned"].sum()),
        "metadata_reassigned_rows": int(reassigned_rows.sum()),
        "metadata_reassigned_jump_ids": int(associated.loc[reassigned_rows, "JUMP_ID"].nunique()),
    }
    associated = (
        associated.drop(columns=["metadata_reassigned"])
        .merge(
            molecules.drop(columns="molregno"),
            on="metadata_source_id",
            validate="many_to_one",
        )
        .drop(columns=["metadata_source_id", "_pchem_was_missing"])
    )
    association = association.sort_values(
        ["JUMP_ID", "molecule_chembl_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return associated, association, counts


def _standardize_with_chemfilters(smiles: list[str], jobs: int) -> list[str]:
    """Lazy import isolates the historical chemistry stack from ordinary tests."""
    try:
        from chemFilters.chem.standardizers import ChemStandardizer  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            "chem-filters 1.0.0 is required for exact structure curation. "
            "Run this file through its PEP 723 environment."
        ) from error
    standardizer = ChemStandardizer(from_smi=True, n_jobs=jobs, progress=False)
    standardized = list(standardizer(smiles))
    if len(standardized) != len(smiles):
        raise SourceIntegrityError("Structure standardizer returned an unexpected number of rows")
    return [str(value) for value in standardized]


def _standardize_jump_smiles_one(smiles: str) -> str:
    """Reproduce the historical iterative OASIS/JUMP standardization for one SMILES."""
    try:
        from dimorphite_dl.io import LoadSMIFile  # pyright: ignore[reportMissingImports]
        from dimorphite_dl.mol import Protonate  # pyright: ignore[reportMissingImports]
        from rdkit import Chem, RDLogger  # pyright: ignore[reportMissingImports]
        from rdkit.Chem import MolStandardize  # pyright: ignore[reportMissingImports]
        from rdkit.Chem.MolStandardize import (  # pyright: ignore[reportMissingImports]
            Standardizer,
            rdMolStandardize,
        )

        # The compatibility algorithm intentionally uses a deprecated RDKit API.
        # Suppress its per-row deprecation and chemistry diagnostics; failures are
        # still converted to Cannot_do and rejected by derive_jump_compounds.
        warnings.filterwarnings(
            "ignore",
            message="The class Standardizer is deprecated.*",
            category=DeprecationWarning,
        )
        RDLogger.DisableLog("rdApp.*")
        standardizer = Standardizer()
        current_smiles = smiles
        molecule = Chem.MolFromSmiles(current_smiles)
        if molecule is None:
            raise ValueError("RDKit could not parse SMILES")
        observed_smiles: Counter[str] = Counter()
        finalized = False
        standardized_smiles = current_smiles

        for _ in range(5):
            molecule = Chem.MolFromInchi(Chem.MolToInchi(molecule))
            if molecule is None:
                raise ValueError("RDKit could not complete the InChI round trip")
            molecule = rdMolStandardize.Cleanup(molecule)
            molecule = rdMolStandardize.FragmentParent(molecule)
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
            molecule = standardizer.charge_parent(molecule)
            molecule = standardizer.isotope_parent(molecule)
            molecule = standardizer.stereo_parent(molecule)
            molecule = MolStandardize.tautomer.TautomerCanonicalizer().canonicalize(molecule)
            molecule = rdMolStandardize.TautomerEnumerator().Canonicalize(molecule)
            molecule = standardizer.tautomer_parent(molecule)
            molecule = standardizer.standardize(molecule)
            standardized_smiles = str(Chem.MolToSmiles(molecule))

            if current_smiles == standardized_smiles:
                finalized = True
                break
            observed_smiles[standardized_smiles] += 1
            current_smiles = standardized_smiles
            molecule = Chem.MolFromSmiles(current_smiles)
            if molecule is None:
                raise ValueError("RDKit could not parse an intermediate SMILES")

        if not finalized:
            if not observed_smiles:
                raise ValueError("Historical standardization produced no candidate")
            standardized_smiles = observed_smiles.most_common(1)[0][0]

        protonation_args: dict[str, Any] = {
            "min_ph": 7.4,
            "max_ph": 7.4,
            "pka_precision": 0,
            "max_variants": 128,
            "label_states": False,
            "silent": True,
            "smiles_file": io.StringIO(standardized_smiles),
        }
        protonation_args["smiles_and_data"] = LoadSMIFile(
            protonation_args["smiles_file"],
            protonation_args,
        )
        protonated = [str(value).strip("\t") for value in Protonate(protonation_args)]
        if not protonated:
            raise ValueError("Dimorphite-DL produced no protonation state")
        return protonated[0]
    except Exception:
        # This sentinel is historical behavior. The caller rejects it so a modern
        # reconstruction cannot silently lose or retain an unstandardized compound.
        return "Cannot_do"


def _standardize_jump_compounds(smiles: list[str], jobs: int) -> list[str]:
    """Run the historical per-compound standardizer in deterministic input order."""
    if jobs == 1:
        return [_standardize_jump_smiles_one(value) for value in smiles]
    chunk_size = max(1, len(smiles) // (jobs * 8))
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_standardize_jump_smiles_one, smiles, chunksize=chunk_size))


def derive_jump_compounds(
    mapping: pd.DataFrame,
    source_compounds: pd.DataFrame,
    molecules: pd.DataFrame,
    *,
    jobs: int,
    standardize: StandardizeFunction | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regenerate the former frozen compound roster from maintained sources."""
    required_source = {"Metadata_JCP2022", "Metadata_InChIKey", "Metadata_SMILES"}
    missing_source_columns = sorted(required_source - set(source_compounds.columns))
    if missing_source_columns:
        raise SourceIntegrityError(f"JUMP compound source is missing columns: {missing_source_columns}")
    if source_compounds["Metadata_JCP2022"].duplicated().any():
        raise SourceIntegrityError("JUMP compound source contains duplicate identifiers")

    mapped_ids = set(mapping["JUMP_ID"].astype(str))
    source = source_compounds.loc[
        source_compounds["Metadata_JCP2022"].isin(mapped_ids),
        ["Metadata_JCP2022", "Metadata_SMILES"],
    ].copy()
    source_ids = set(source["Metadata_JCP2022"].astype(str))
    missing_ids = mapped_ids - source_ids

    molecule_columns = {"metadata_source_id", "canonical_smiles"}
    missing_molecule_columns = sorted(molecule_columns - set(molecules.columns))
    if missing_molecule_columns:
        raise SourceIntegrityError(f"ChEMBL molecule metadata is missing columns: {missing_molecule_columns}")
    ranked_fallbacks = (
        mapping.loc[mapping["JUMP_ID"].isin(missing_ids), ["JUMP_ID", "molecule_chembl_id", "hit_rank"]]
        .sort_values(["JUMP_ID", "hit_rank"], kind="mergesort")
        .merge(
            molecules[["metadata_source_id", "canonical_smiles"]],
            left_on="molecule_chembl_id",
            right_on="metadata_source_id",
            how="left",
            validate="many_to_one",
        )
    )
    fallback = (
        ranked_fallbacks.loc[ranked_fallbacks["canonical_smiles"].notna()]
        .drop_duplicates("JUMP_ID", keep="first")
        .rename(columns={"canonical_smiles": "Metadata_SMILES"})
    )
    resolved_fallback_ids = set(fallback["JUMP_ID"].astype(str))
    unresolved = sorted(missing_ids - resolved_fallback_ids)
    if unresolved:
        raise SourceIntegrityError(f"No ChEMBL structure fallback was found for mapped JUMP IDs: {unresolved[:20]}")

    fallback_manifest = {
        str(row.JUMP_ID): str(row.molecule_chembl_id)
        for row in fallback[["JUMP_ID", "molecule_chembl_id"]].itertuples(index=False)
    }
    combined = pd.concat(
        [
            source,
            fallback[["JUMP_ID", "Metadata_SMILES"]].rename(columns={"JUMP_ID": "Metadata_JCP2022"}),
        ],
        ignore_index=True,
    ).sort_values("Metadata_JCP2022", kind="mergesort")
    if set(combined["Metadata_JCP2022"].astype(str)) != mapped_ids:
        raise SourceIntegrityError("Derived compound roster does not cover the SmallWorld JUMP universe")
    if (
        combined["Metadata_SMILES"].isna().any()
        or not combined["Metadata_SMILES"].map(lambda value: isinstance(value, str) and bool(value)).all()
    ):
        raise SourceIntegrityError("Derived compound roster contains a missing or non-string SMILES")

    standardize = standardize or _standardize_jump_compounds
    source_smiles = cast(list[str], combined["Metadata_SMILES"].tolist())
    standardized = standardize(source_smiles, jobs)
    if len(standardized) != len(combined):
        raise SourceIntegrityError("JUMP compound standardizer returned an unexpected number of rows")
    failed = [index for index, value in enumerate(standardized) if value == "Cannot_do" or not value]
    if failed:
        failed_ids = combined.iloc[failed]["Metadata_JCP2022"].astype(str).tolist()
        raise SourceIntegrityError(f"Historical JUMP standardization failed for: {failed_ids[:20]}")

    compounds = combined[["Metadata_JCP2022"]].copy()
    compounds["Standardized_SMILES"] = [str(value) for value in standardized]
    compounds = compounds.reset_index(drop=True)
    diagnostics: dict[str, Any] = {
        "mapped_jump_ids": int(len(mapped_ids)),
        "source_rows": int(len(source)),
        "fallback_rows": int(len(fallback)),
        "output_rows": int(len(compounds)),
        "fallback_molecules": fallback_manifest,
        "standardization": {
            "algorithm": "historical iterative OASIS/JUMP RDKit plus Dimorphite-DL at pH 7.4",
            "failed_rows": 0,
        },
    }
    return compounds, diagnostics


def remove_salt_and_solvent_fragments(smiles: str) -> str:
    fragments = np.unique(smiles.split(".")).tolist()
    if len(fragments) == 1:
        return str(fragments[0])
    retained = [fragment for fragment in fragments if not _SALT_OR_SOLVENT_PATTERN.fullmatch(str(fragment))]
    return "." if not retained else ".".join(str(fragment) for fragment in retained)


def curate_structures_and_activity_types(
    activities: pd.DataFrame,
    *,
    jobs: int,
    standardize: StandardizeFunction | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    standardize = standardize or _standardize_with_chemfilters
    source_smiles = activities["canonical_smiles"].drop_duplicates().tolist()
    if not all(isinstance(smiles, str) for smiles in source_smiles):
        raise SourceIntegrityError("ChEMBL canonical SMILES contains a missing or non-string value")
    standardized = dict(
        zip(
            source_smiles,
            standardize(cast(list[str], source_smiles), jobs),
            strict=True,
        )
    )

    curated = activities.copy()
    curated["standard_smiles"] = (
        curated["canonical_smiles"].map(lambda value: standardized[value]).apply(remove_salt_and_solvent_fragments)
    )
    mixture_mask = curated["standard_smiles"].str.contains(".", regex=False)
    unresolved_mixture_rows = int(mixture_mask.sum())
    curated = curated.loc[~mixture_mask].reset_index(drop=True)

    missing_pchembl_rows = int(curated["pchembl_value"].isna().sum())
    curated = curated.loc[curated["pchembl_value"].notna()].reset_index(drop=True)

    selected = curated["standard_type"].isin(SELECTED_ACTIVITY_TYPES)
    excluded_activity_type_rows = int((~selected).sum())
    curated = curated.loc[selected].copy()
    curated["type"] = curated["standard_type"]
    curated = cast(pd.DataFrame, curated.loc[:, list(CURATED_COLUMNS)])
    return curated, {
        "input_rows": int(len(activities)),
        "unresolved_mixture_rows": unresolved_mixture_rows,
        "missing_pchembl_rows": missing_pchembl_rows,
        "excluded_activity_type_rows": excluded_activity_type_rows,
        "output_rows": int(len(curated)),
    }


def build_annotations(
    activities: pd.DataFrame,
    target_mapping: pd.DataFrame,
    compounds: pd.DataFrame,
    *,
    activity_threshold: float = ACTIVITY_THRESHOLD,
    minimum_per_class: int = MINIMUM_PER_CLASS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the frozen target map and historical support/dedup finalizer."""
    mapped = activities.merge(
        target_mapping,
        left_on="target_chembl_id",
        right_on="From",
        how="inner",
    )
    pair_means = (
        mapped[["JUMP_ID", "Entry", "pchembl_value"]]
        .groupby(["JUMP_ID", "Entry"])
        .mean()
        .reset_index()
        .merge(
            compounds,
            left_on="JUMP_ID",
            right_on="Metadata_JCP2022",
            how="inner",
        )
    )
    pair_means["label"] = (pair_means["pchembl_value"] > activity_threshold).astype(int)

    class_counts = pair_means.groupby("Entry")["label"].value_counts().unstack().fillna(0)
    eligible_targets = class_counts.loc[
        (class_counts[0] >= minimum_per_class) & (class_counts[1] >= minimum_per_class)
    ].index

    long_labels = pair_means.loc[
        pair_means["Entry"].isin(eligible_targets),
        ["JUMP_ID", "Standardized_SMILES", "Entry", "label"],
    ]
    long_labels = (
        long_labels.sort_values(
            ["JUMP_ID", "Entry", "Standardized_SMILES", "label"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["JUMP_ID", "Entry"])
        .drop_duplicates(["Standardized_SMILES", "Entry"])
    )

    wide_labels = (
        long_labels.pivot(index="JUMP_ID", columns="Entry", values="label")
        .reset_index()
        .merge(
            long_labels[["JUMP_ID", "Standardized_SMILES"]].drop_duplicates(),
            on="JUMP_ID",
            how="left",
        )
    )
    target_columns = wide_labels.columns[1:-1]
    output_counts = pd.DataFrame(
        {
            "inactive": (wide_labels[target_columns] == 0).sum(),
            "active": (wide_labels[target_columns] == 1).sum(),
        }
    )
    target_metadata = target_mapping.set_index("Entry").loc[target_columns, ["Organism", "Target Name"]].copy()
    target_summary = target_metadata.join(output_counts)
    target_summary.index.name = "UniProt_accession"
    target_summary = target_summary.reset_index()

    summary = {
        "activity_threshold": activity_threshold,
        "threshold_rule": f"mean pChEMBL > {activity_threshold:g}",
        "minimum_labels_per_class": minimum_per_class,
        "curated_activity_rows": int(len(activities)),
        "mapped_activity_rows": int(len(mapped)),
        "mean_compound_target_pairs": int(len(pair_means)),
        "compounds": int(wide_labels["JUMP_ID"].nunique()),
        "targets": int(len(target_columns)),
        "annotations": int(len(long_labels)),
        "active": int(long_labels["label"].sum()),
        "inactive": int((long_labels["label"] == 0).sum()),
        "human_targets": int((target_summary["Organism"] == "Homo sapiens (Human)").sum()),
    }
    return long_labels, wide_labels, target_summary, summary


def _check_expected_counts(label: str, observed: dict[str, int], expected: dict[str, int]) -> None:
    mismatches = {key: (observed.get(key), value) for key, value in expected.items() if observed.get(key) != value}
    if mismatches:
        raise ExactReproductionError(f"Unexpected {label} counts: {mismatches}")


def _output_metadata(path: Path, rows: int, columns: int) -> dict[str, Any]:
    observed_sha256 = _sha256(path)
    historical_sha256 = HISTORICAL_OUTPUT_SHA256.get(path.name)
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "columns": columns,
        "size_bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "historical_sha256": historical_sha256,
        "matches_historical_sha256": (observed_sha256 == historical_sha256 if historical_sha256 is not None else None),
    }


def _set_diagnostics(current: set[str], reference: set[str]) -> dict[str, Any]:
    current_only = sorted(current - reference)
    reference_only = sorted(reference - current)
    common = sorted(current & reference)
    return {
        "current": len(current),
        "reference": len(reference),
        "common": len(common),
        "current_only": len(current_only),
        "reference_only": len(reference_only),
        "current_only_examples": current_only[:20],
        "reference_only_examples": reference_only[:20],
        "current_sha256": hashlib.sha256("\n".join(sorted(current)).encode()).hexdigest(),
        "reference_sha256": hashlib.sha256("\n".join(sorted(reference)).encode()).hexdigest(),
    }


def compare_wide_candidate(
    current: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    """Explain any exact-gate failure without weakening byte acceptance."""
    identity_columns = {"JUMP_ID", "Standardized_SMILES"}
    for label, frame in (("current", current), ("reference", reference)):
        missing = identity_columns - set(frame.columns)
        if missing:
            raise ExactReproductionError(f"{label} matrix is missing identity columns: {sorted(missing)}")
        if frame["JUMP_ID"].duplicated().any():
            raise ExactReproductionError(f"{label} matrix contains duplicate JUMP_ID rows")

    current_ids = set(current["JUMP_ID"].astype(str))
    reference_ids = set(reference["JUMP_ID"].astype(str))
    current_targets = set(current.columns) - identity_columns
    reference_targets = set(reference.columns) - identity_columns
    common_ids = sorted(current_ids & reference_ids)
    common_targets = sorted(current_targets & reference_targets)

    current_indexed = current.set_index("JUMP_ID")
    reference_indexed = reference.set_index("JUMP_ID")
    current_grid = current_indexed.loc[common_ids, common_targets]
    reference_grid = reference_indexed.loc[common_ids, common_targets]
    current_measured = current_grid.notna()
    reference_measured = reference_grid.notna()
    both_measured = current_measured & reference_measured
    different_labels = both_measured & ~current_grid.eq(reference_grid)

    label_examples: list[dict[str, Any]] = []
    for jump_id, target in zip(*np.where(different_labels.to_numpy()), strict=True):
        row_id = common_ids[int(jump_id)]
        target_id = common_targets[int(target)]
        label_examples.append(
            {
                "JUMP_ID": row_id,
                "target": target_id,
                "current": float(current_grid.loc[row_id, target_id]),
                "reference": float(reference_grid.loc[row_id, target_id]),
            }
        )
        if len(label_examples) == 20:
            break

    current_smiles = current_indexed.loc[common_ids, "Standardized_SMILES"]
    reference_smiles = reference_indexed.loc[common_ids, "Standardized_SMILES"]
    comparable_smiles = current_smiles.notna() & reference_smiles.notna()
    different_smiles = comparable_smiles & ~current_smiles.eq(reference_smiles)
    smiles_examples = [
        {
            "JUMP_ID": str(jump_id),
            "current": str(current_smiles.loc[jump_id]),
            "reference": str(reference_smiles.loc[jump_id]),
        }
        for jump_id in different_smiles.index[different_smiles][:20]
    ]

    def active_memberships(frame: pd.DataFrame, targets: set[str]) -> set[str]:
        indexed = frame.set_index("JUMP_ID")
        memberships: set[str] = set()
        for target in sorted(targets):
            active = indexed.index[indexed[target].eq(1)]
            memberships.update(f"{jump_id}\t{target}" for jump_id in active)
        return memberships

    current_active = active_memberships(current, current_targets)
    reference_active = active_memberships(reference, reference_targets)
    return {
        "compound_universe": _set_diagnostics(current_ids, reference_ids),
        "target_universe": _set_diagnostics(current_targets, reference_targets),
        "measured_mask_on_common_grid": {
            "grid_coordinates": int(current_grid.size),
            "current_measured": int(current_measured.sum().sum()),
            "reference_measured": int(reference_measured.sum().sum()),
            "both_measured": int(both_measured.sum().sum()),
            "same_mask": int(current_measured.eq(reference_measured).sum().sum()),
            "current_only_measured": int((current_measured & ~reference_measured).sum().sum()),
            "reference_only_measured": int((~current_measured & reference_measured).sum().sum()),
        },
        "labels_on_common_measured_coordinates": {
            "same": int((both_measured & current_grid.eq(reference_grid)).sum().sum()),
            "different": int(different_labels.sum().sum()),
            "different_examples": label_examples,
        },
        "standardized_smiles": {
            "comparable": int(comparable_smiles.sum()),
            "same": int((comparable_smiles & current_smiles.eq(reference_smiles)).sum()),
            "different": int(different_smiles.sum()),
            "different_examples": smiles_examples,
        },
        "active_memberships": _set_diagnostics(current_active, reference_active),
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_versions() -> dict[str, str | None]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sqlite": sqlite3.sqlite_version,
        "chem-filters": _installed_version("chem-filters"),
        "chembl-structure-pipeline": _installed_version("chembl-structure-pipeline"),
        "dimorphite-dl": _installed_version("dimorphite-dl"),
        "rdkit": _installed_version("rdkit"),
        "typer": _installed_version("typer"),
    }


def run_exact_reproduction(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    database: Path = DEFAULT_DATABASE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reference: Path = DEFAULT_REFERENCE,
    compound_source: Path = DEFAULT_JUMP_COMPOUNDS,
    jobs: int = 8,
    standardize: StandardizeFunction | None = None,
    standardize_compounds: StandardizeFunction | None = None,
) -> dict[str, Any]:
    """Run the exact workflow and return its completed summary."""
    _guard_paths(
        output_dir=output_dir,
        archive=archive,
        database=database,
        reference=reference,
        compound_source=compound_source,
    )
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    stage = "initializing"
    failure_details: dict[str, Any] = {}
    _write_incomplete_summary(summary_path, stage=stage)

    try:
        stage = "verifying SmallWorld mapping, JUMP compounds, and canonical reference"
        mapping, smallworld_metadata = load_smallworld_mapping(output_dir)
        source_compounds, compound_source_metadata = verify_jump_compound_source(compound_source)
        reference_bytes, reference_metadata = verify_canonical_reference(reference)

        stage = "verifying and extracting official ChEMBL 33"
        source_metadata = ensure_official_database(archive, database)
        with open_chembl_database(database) as connection:
            database_metadata = validate_database(connection)
            stage = "extracting filtered ChEMBL activity data"
            extracted, molecules = extract_database_tables(connection, mapping)

        stage = "reconstructing pChEMBL and historical compound metadata"
        processed = calculate_missing_pchembl(extracted)
        associated, association_audit, association_counts = associate_historical_compound_metadata(processed, molecules)
        raw_activities = cast(pd.DataFrame, associated.loc[:, list(RAW_COLUMNS)])

        stage = "deriving the UniProt map for retained ChEMBL activity targets"
        with open_chembl_database(database) as connection:
            target_mapping = extract_target_mapping(
                connection,
                cast(Collection[str], raw_activities["target_chembl_id"].dropna().astype(str).tolist()),
            )
        target_mapping_counts = {
            "rows": int(len(target_mapping)),
            "target_chembl_ids": int(target_mapping["From"].nunique()),
            "uniprot_accessions": int(target_mapping["Entry"].nunique()),
        }
        _check_expected_counts(
            "source-derived target mapping",
            target_mapping_counts,
            EXPECTED_TARGET_MAPPING_COUNTS,
        )

        stage = "standardizing and curating structures"
        curated_activities, curation_counts = curate_structures_and_activity_types(
            raw_activities,
            jobs=jobs,
            standardize=standardize,
        )
        activity_counts = {
            "mapping_rows": int(len(mapping)),
            "sql_rows_before_pchembl_processing": int(len(extracted)),
            "raw_activity_rows": int(len(raw_activities)),
            "unresolved_mixture_rows": curation_counts["unresolved_mixture_rows"],
            "missing_pchembl_rows": curation_counts["missing_pchembl_rows"],
            "excluded_activity_type_rows": curation_counts["excluded_activity_type_rows"],
            "curated_activity_rows": int(len(curated_activities)),
            "metadata_reassigned_rows": association_counts["metadata_reassigned_rows"],
            "metadata_reassigned_jump_ids": association_counts["metadata_reassigned_jump_ids"],
        }
        _check_expected_counts("activity-preparation", activity_counts, EXPECTED_ACTIVITY_COUNTS)

        stage = "deriving and standardizing the mapped JUMP compound roster"
        compounds, compound_derivation = derive_jump_compounds(
            mapping,
            source_compounds,
            molecules,
            jobs=jobs,
            standardize=standardize_compounds,
        )
        compound_derivation_counts = {key: int(compound_derivation[key]) for key in EXPECTED_COMPOUND_DERIVATION_COUNTS}
        _check_expected_counts(
            "source-derived compound roster",
            compound_derivation_counts,
            EXPECTED_COMPOUND_DERIVATION_COUNTS,
        )
        if compound_derivation["fallback_molecules"] != EXPECTED_FALLBACK_MOLECULES:
            raise ExactReproductionError(
                f"Unexpected ChEMBL compound fallbacks: {compound_derivation['fallback_molecules']}"
            )

        raw_path = output_dir / RAW_ACTIVITY_NAME
        curated_path = output_dir / CURATED_ACTIVITY_NAME
        association_path = output_dir / ASSOCIATION_AUDIT_NAME
        stage = "writing deterministic activity audit tables"
        _atomic_gzip_csv(raw_activities, raw_path)
        _atomic_gzip_csv(curated_activities, curated_path)
        _atomic_gzip_csv(association_audit, association_path)

        # The historical procedure was two stages. Reading the curated CSV here
        # preserves its serialization and dtype boundary before group means.
        stage = "building target annotations from the serialized curated table"
        serialized_curated = pd.read_csv(curated_path, low_memory=False)
        long_labels, wide_labels, target_counts, annotation_counts = build_annotations(
            serialized_curated,
            target_mapping,
            compounds,
        )
        observed_annotation_counts = {key: int(annotation_counts[key]) for key in EXPECTED_ANNOTATION_COUNTS}
        _check_expected_counts(
            "annotation",
            observed_annotation_counts,
            EXPECTED_ANNOTATION_COUNTS,
        )

        wide_bytes = wide_labels.to_csv(index=False, lineterminator="\n").encode("utf-8")
        wide_sha256 = hashlib.sha256(wide_bytes).hexdigest()
        if wide_sha256 != CANONICAL_REFERENCE_SHA256 or wide_bytes != reference_bytes:
            candidate_path = output_dir / INCOMPLETE_CANDIDATE_NAME
            _atomic_bytes(candidate_path, wide_bytes)
            reference_frame = pd.read_csv(io.BytesIO(reference_bytes), low_memory=False)
            failure_details["reference_comparison"] = {
                "expected_sha256": CANONICAL_REFERENCE_SHA256,
                "generated_sha256": wide_sha256,
                "byte_equal": wide_bytes == reference_bytes,
                "diagnostics": compare_wide_candidate(wide_labels, reference_frame),
            }
            failure_details["incomplete_candidate"] = {
                "path": str(candidate_path.resolve()),
                "size_bytes": candidate_path.stat().st_size,
                "sha256": wide_sha256,
            }
            raise ExactReproductionError(
                "Generated wide CSV failed the canonical byte gate; detailed diagnostics "
                f"and the incomplete candidate are recorded in {summary_path}"
            )

        stage = "writing verified annotation outputs"
        long_path = output_dir / LONG_ANNOTATION_NAME
        target_path = output_dir / TARGET_COUNTS_NAME
        wide_path = output_dir / WIDE_MATRIX_NAME
        _atomic_gzip_csv(long_labels, long_path)
        _atomic_csv(target_counts, target_path)
        verified_candidate = output_dir / INCOMPLETE_CANDIDATE_NAME
        _atomic_bytes(verified_candidate, wide_bytes)
        verified_candidate.replace(wide_path)

        output_manifest = {
            path.name: _output_metadata(path, len(frame), len(frame.columns))
            for path, frame in (
                (raw_path, raw_activities),
                (curated_path, curated_activities),
                (association_path, association_audit),
                (long_path, long_labels),
                (target_path, target_counts),
                (wide_path, wide_labels),
            )
        }
        summary: dict[str, Any] = {
            "description": "Exact JUMP ChEMBL 33 target-matrix reproduction",
            "status": "complete",
            "attribution": CHEMBL_ATTRIBUTION,
            "behavioral_reference": {
                "url": BEHAVIORAL_REFERENCE_URL,
                "commit": BEHAVIORAL_REFERENCE_COMMIT,
                "role": "behavioral audit reference only; no source files are imported",
            },
            "historical_reference": {
                "url": HISTORICAL_REFERENCE_URL,
                "commit": HISTORICAL_REFERENCE_COMMIT,
                "role": "historical audit reference only; no source files are imported",
            },
            "official_source": {
                **source_metadata,
                "database_validation": database_metadata,
            },
            "historical_inputs": {
                SMALLWORLD_MAPPING_PIN.filename: smallworld_metadata,
            },
            "historical_input_provenance": {
                "source_url": SMALLWORLD_SOURCE_URL,
                "status": SMALLWORLD_SOURCE_STATUS,
            },
            "source_derived_inputs": {
                "jump_compounds": {
                    "source": compound_source_metadata,
                    "derivation": compound_derivation,
                    "replaces_historical_file": "jump_compounds.csv.gz",
                },
                "chembl_target_to_uniprot": {
                    **target_mapping_counts,
                    "selection_rule": (
                        "Swiss-Prot SINGLE PROTEIN components plus PROTEIN COMPLEX components "
                        "whose accession has no Swiss-Prot SINGLE PROTEIN target in ChEMBL 33"
                    ),
                    "replaces_historical_file": "chembl_target_to_uniprot.csv.gz",
                },
            },
            "activity_preparation": {
                **activity_counts,
                "metadata_association": association_counts,
                "curation": curation_counts,
            },
            "annotation": annotation_counts,
            "reference_gate": {
                **reference_metadata,
                "generated_sha256": wide_sha256,
                "byte_equal": True,
            },
            "software": _software_versions(),
            "outputs": output_manifest,
            "files": sorted(output_manifest),
        }
        _atomic_json(summary_path, summary)
        return summary
    except Exception as error:
        _write_incomplete_summary(
            summary_path,
            stage=stage,
            error=error,
            details=failure_details,
        )
        raise


@app.command()
def main(
    archive: Path = typer.Option(
        DEFAULT_ARCHIVE,
        help="Pinned official ChEMBL 33 SQLite archive; acquisition is separate.",
    ),
    database: Path = typer.Option(
        DEFAULT_DATABASE,
        help="Safely extracted ChEMBL 33 SQLite database.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Local package directory; the SmallWorld mapping must be its sole cached input.",
    ),
    reference: Path = typer.Option(
        DEFAULT_REFERENCE,
        help="Canonical Zenodo wide CSV used as the mandatory byte gate.",
    ),
    compound_source: Path = typer.Option(
        DEFAULT_JUMP_COMPOUNDS,
        help="Pinned JUMP compound.csv.gz already acquired by download_data.py.",
    ),
    jobs: int = typer.Option(8, min=1, help="Historical structure-standardization workers."),
) -> int:
    """Run the exact, fail-closed ChEMBL 33 reproduction."""
    summary = run_exact_reproduction(
        archive=archive,
        database=database,
        output_dir=output_dir,
        reference=reference,
        compound_source=compound_source,
        jobs=jobs,
    )
    annotation = summary["annotation"]
    typer.echo(
        "Exact ChEMBL 33 reproduction passed: "
        f"{annotation['compounds']:,} compounds, "
        f"{annotation['targets']:,} targets, "
        f"{annotation['annotations']:,} annotations"
    )
    return 0


if __name__ == "__main__":
    app()
