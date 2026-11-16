import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    import zipfile

    import duckdb
    import marimo as mo
    import pandas as pd
    import yaml

    from jump_production import config as _cfg
    from jump_production.summaries import write_summary

    REPO = _cfg.PROJ_ROOT
    ANALYSIS_CONFIG = REPO / "configs/snakemake/analysis.yaml"
    CONTRACT_PATH = REPO / "references/compound_collection_contract.json"
    COPAIRS_DB = REPO / "data/processed/copairs_results.duckdb"
    FUNCTIONAL_USE_CSV = REPO / "data/external/2026-04-06_JUMP_chems_func_use.csv"
    METADATA_DB = REPO / "data/interim/jump_metadata_augmented.duckdb"
    MOLPORT_ZIP = REPO / "data/external/molport_batch_search.zip"
    OUTPUT = REPO / "data/processed/compound-collection-facts/summary.json"

    _analysis = yaml.safe_load(ANALYSIS_CONFIG.read_text())["analysis"]
    KEY = {
        _key: _analysis[_key]
        for _key in (
            "default_dataset",
            "default_preprocessing",
            "default_filter",
            "default_activity_params",
        )
    }
    CHEMICAL_SUMMARY = REPO / (f"data/processed/chemical-space/{KEY['default_dataset']}/chemical_space_summary.json")
    return (
        CHEMICAL_SUMMARY,
        CONTRACT_PATH,
        COPAIRS_DB,
        FUNCTIONAL_USE_CSV,
        KEY,
        METADATA_DB,
        MOLPORT_ZIP,
        OUTPUT,
        duckdb,
        hashlib,
        json,
        mo,
        pd,
        write_summary,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Compound-collection facts

    Produces one dataset-specific summary envelope for the JUMP compound-collection
    claims. Computable values are derived from the canonical analyzed roster, chemical-space
    summary, metadata database, and original MolPort quote archive. A pinned numerical
    contract makes changed values fail loudly instead of silently drifting.

    EPA functional-use values are recomputed from the augmented database against the same analyzed
    roster, with the public source file recorded by checksum.
    """)
    return


@app.cell
def _(hashlib, pd, zipfile):
    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_molport_archive(path, price_cutoff_usd=200):
        with zipfile.ZipFile(path) as archive:
            batch_files = sorted(
                name for name in archive.namelist() if "/to_search/batch_" in name and name.endswith(".txt")
            )
            searched = {
                line.strip()
                for name in batch_files
                for line in archive.read(name).decode().splitlines()
                if line.strip()
            }
            result_files = sorted(
                name for name in archive.namelist() if "/search_results/Quote_" in name and name.endswith(".xls")
            )
            if not result_files:
                raise FileNotFoundError("MolPort archive contains no Quote_*.xls files")
            results = pd.concat(
                [pd.read_excel(archive.open(name)) for name in result_files],
                ignore_index=True,
            )

        hits = (
            results.groupby("Search Criteria", as_index=False, sort=True)
            .agg(
                {
                    "Molport ID": "first",
                    "Match Type": lambda values: "Perfect" if "Perfect" in values.values else "Isomer",
                    "Price, USD": "min",
                    "Supplier Name": "first",
                }
            )
            .rename(columns={"Search Criteria": "SMILES"})
        )
        raw_hit_count = len(hits)
        hits = hits.loc[hits["Price, USD"] <= price_cutoff_usd].copy()
        stats = {
            "searched_compounds": len(searched),
            "raw_hits": raw_hit_count,
            "price_filtered_hits": len(hits),
            "price_cutoff_usd": price_cutoff_usd,
            "matching": "exact SMILES against price-filtered MolPort quote results",
        }
        return hits, stats

    def flatten(mapping, prefix=""):
        flattened = {}
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(flatten(value, path))
            else:
                flattened[path] = value
        return flattened

    return flatten, load_molport_archive, sha256


@app.cell
def _(
    CHEMICAL_SUMMARY,
    CONTRACT_PATH,
    COPAIRS_DB,
    FUNCTIONAL_USE_CSV,
    KEY,
    METADATA_DB,
    MOLPORT_ZIP,
    OUTPUT,
    duckdb,
    flatten,
    json,
    load_molport_archive,
    pd,
    sha256,
    write_summary,
):
    _chemical_doc = json.loads(CHEMICAL_SUMMARY.read_text())
    _chemical = _chemical_doc["numbers"]
    _contract = json.loads(CONTRACT_PATH.read_text())
    if _contract.get("schema_version") != 1:
        raise ValueError("Unsupported compound-collection numerical contract")
    if _chemical.get("dataset_filter") != KEY["default_dataset"]:
        raise ValueError(
            "Chemical-space summary does not describe the canonical dataset: "
            f"{_chemical.get('dataset_filter')!r} != {KEY['default_dataset']!r}"
        )

    _copairs = duckdb.connect(str(COPAIRS_DB), read_only=True)
    try:
        _roster = _copairs.execute(
            """
            SELECT DISTINCT Metadata_JCP2022
            FROM activity_results
            WHERE _dataset = ?
              AND _preprocessing = ?
              AND _filter = ?
              AND _activity_params = ?
            ORDER BY Metadata_JCP2022
            """,
            list(KEY.values()),
        ).fetchdf()
    finally:
        _copairs.close()

    _n_analyzed = len(_roster)
    if _n_analyzed != int(_chemical["total_compounds"]):
        raise ValueError(
            "Canonical activity roster and chemical-space population differ: "
            f"{_n_analyzed:,} != {int(_chemical['total_compounds']):,}"
        )

    _molport_hits, _molport_stats = load_molport_archive(MOLPORT_ZIP)
    _metadata = duckdb.connect(str(METADATA_DB), read_only=True)
    try:
        _metadata.register("_compound_collection_roster", _roster)
        _metadata.register("_compound_collection_molport", _molport_hits[["SMILES"]])
        _target_or_moa = int(
            _metadata.execute(
                """
                WITH flags AS (
                    SELECT
                        cm.Metadata_JCP2022,
                        NULLIF(trim(cm.Metadata_Uniprot_target), '') IS NOT NULL AS chembl,
                        (
                            NULLIF(trim(cm.Metadata_repurposing_target), '') IS NOT NULL
                            OR NULLIF(trim(cm.Metadata_repurposing_moa), '') IS NOT NULL
                        ) AS repurposing_hub,
                        NULLIF(trim(cm.Metadata_chmprb_target_genes), '') IS NOT NULL
                            AS chemical_probes,
                        EXISTS (
                            SELECT 1
                            FROM motive_annotations motive
                            WHERE motive.Metadata_JCP2022 = cm.Metadata_JCP2022
                              AND NULLIF(trim(motive.Metadata_motive_gene), '') IS NOT NULL
                        ) AS motive
                    FROM compound_metadata cm
                    JOIN _compound_collection_roster roster USING (Metadata_JCP2022)
                )
                SELECT count(*) FILTER (
                    WHERE chembl OR repurposing_hub OR chemical_probes OR motive
                )
                FROM flags
                """
            ).fetchone()[0]
        )
        _molport_available = int(
            _metadata.execute(
                """
                SELECT count(DISTINCT roster.Metadata_JCP2022)
                    FILTER (WHERE molport.SMILES IS NOT NULL)
                FROM _compound_collection_roster roster
                JOIN compound compounds USING (Metadata_JCP2022)
                LEFT JOIN _compound_collection_molport molport
                    ON molport.SMILES = compounds.Metadata_SMILES
                """
            ).fetchone()[0]
        )
        _dual_sourced = int(
            _metadata.execute(
                """
                WITH sources AS (
                    SELECT
                        Metadata_JCP2022,
                        count(DISTINCT Metadata_Compound_Source) AS n_sources
                    FROM compound_source
                    GROUP BY Metadata_JCP2022
                )
                SELECT count(*) FILTER (WHERE n_sources >= 2)
                FROM _compound_collection_roster roster
                JOIN sources USING (Metadata_JCP2022)
                """
            ).fetchone()[0]
        )
        _fup_columns = [
            row[0]
            for row in _metadata.execute(
                """
                SELECT column_name
                FROM duckdb_columns()
                WHERE table_name = 'functional_use'
                  AND column_name LIKE 'Metadata_funcuse_%'
                  AND column_name NOT IN (
                      'Metadata_funcuse_dtxsid',
                      'Metadata_funcuse_category'
                  )
                ORDER BY column_index
                """
            ).fetchall()
        ]
        _functional_all = _metadata.execute("SELECT * FROM functional_use ORDER BY Metadata_JCP2022").fetchdf()
    finally:
        _metadata.close()

    if len(_fup_columns) != 35:
        raise ValueError(f"Expected 35 EPA FUP columns, found {len(_fup_columns)}")
    if _functional_all["Metadata_JCP2022"].duplicated().any():
        raise ValueError("functional_use must contain one row per JCP2022")

    _functional = _functional_all.merge(_roster, on="Metadata_JCP2022", how="inner", validate="one_to_one")
    _functional_provenance = _contract["functional_use_provenance"]
    _all_fuc_memberships = _functional_all["Metadata_funcuse_category"].dropna().str.split("|").explode().str.strip()
    _all_fuc_memberships = _all_fuc_memberships[_all_fuc_memberships != ""]
    _fuc_memberships = _functional["Metadata_funcuse_category"].dropna().str.split("|").explode().str.strip()
    _fuc_memberships = _fuc_memberships[_fuc_memberships != ""]
    _has_fup_all = _functional_all[_fup_columns].notna().any(axis=1)
    _has_fup = _functional[_fup_columns].notna().any(axis=1)
    _fup_counts = _functional.loc[_has_fup, _fup_columns].notna().sum(axis=1)
    _dominant = (
        _functional.loc[_has_fup, _fup_columns].idxmax(axis=1).str.removeprefix("Metadata_funcuse_").value_counts()
    )
    _top_dominant = [name.replace("_", " ").title() for name in _dominant.head(5).index]
    numbers = {
        "population": {
            "dataset": KEY["default_dataset"],
            "preprocessing": KEY["default_preprocessing"],
            "filter": KEY["default_filter"],
            "activity_params": KEY["default_activity_params"],
            "analyzed_compounds": _n_analyzed,
        },
        "chemistry": {
            "lipinski_rule_of_5_compliant": {
                "count": int(_chemical["lipinski_rule_of_5"]["rule_of_5_compliant"]["count"]),
                "percent": float(_chemical["lipinski_rule_of_5"]["rule_of_5_compliant"]["percent"]),
                "definition": "Metadata_Lipinski_Violations == 0",
            },
            "unique_murcko_scaffolds": {
                "count": int(_chemical["scaffold_diversity"]["unique_scaffolds"]),
                "definition": (
                    "Distinct non-null, non-empty Metadata_MurckoScaffold values; "
                    "acyclic compounds are excluded from the scaffold count"
                ),
            },
            "pains_flagged": {
                "count": int(_chemical["pains_structural_alerts"]["pains_flagged"]),
                "definition": "Metadata_HasPAINS is true in the canonical analyzed roster",
            },
        },
        "clinical_development": {
            "annotated_total": sum(int(value) for value in _chemical["clinical_development"].values()),
            "categories": {
                phase: int(_chemical["clinical_development"][phase])
                for phase in (
                    "Preclinical",
                    "Phase 1",
                    "Phase 2",
                    "Phase 3",
                    "Launched",
                )
            },
            "definition": (
                "Separate values of Metadata_repurposing_clinical_phase; preclinical "
                "and phase counts are not subsets of the launched count"
            ),
        },
        "annotations": {
            "target_or_moa": {
                "count": _target_or_moa,
                "percent": round(100 * _target_or_moa / _n_analyzed, 1),
                "families": [
                    "ChEMBL",
                    "Repurposing Hub target or MOA",
                    "Chemical Probes Portal",
                    "MOTIVE (all database values)",
                ],
                "definition": (
                    "Union of ChEMBL, Repurposing Hub target/MOA, Chemical Probes "
                    "Portal, and every MOTIVE gene annotation"
                ),
                "distinct_from_annotated_any": (
                    "This paper-facing target/MOA union is not the chemical-space nine-source annotated_any metric"
                ),
            }
        },
        "commercial_availability": {
            "molport": {
                "count": _molport_available,
                "percent": round(100 * _molport_available / _n_analyzed, 1),
                "definition": (
                    "Canonical analyzed compounds with an exact-SMILES match in the "
                    "MolPort quote archive after applying Price, USD <= 200"
                ),
                "raw_archive": {
                    **_molport_stats,
                    "path": "data/external/molport_batch_search.zip",
                    "sha256": sha256(MOLPORT_ZIP),
                },
            }
        },
        "dual_sourcing": {
            "count": _dual_sourced,
            "percent": round(100 * _dual_sourced / _n_analyzed, 1),
            "definition": (
                "Canonical analyzed compounds associated with at least two distinct "
                "Metadata_Compound_Source values; this operationalizes separate "
                "partner-provided samples"
            ),
        },
        "functional_use": {
            "provenance_status": _functional_provenance["provenance_status"],
            "raw_input_available": True,
            "raw_input_path": _functional_provenance["raw_input_path"],
            "raw_input_sha256": sha256(FUNCTIONAL_USE_CSV),
            "raw_input_distribution": _functional_provenance["raw_input_distribution"],
            "pipeline": _functional_provenance["pipeline"],
            "source_url": _functional_provenance["source_url"],
            "source_author": _functional_provenance["source_author"],
            "source_date": _functional_provenance["source_date"],
            "dataset": KEY["default_dataset"],
            "source7_excluded": KEY["default_dataset"] == "compound_no_source7",
            "canonical_analyzed_compounds": _n_analyzed,
            "full_delivery": {
                "compounds": int(len(_functional_all)),
                "curated_fuc_compounds": int(_functional_all["Metadata_funcuse_category"].notna().sum()),
                "curated_fuc_categories": int(_all_fuc_memberships.nunique()),
                "predicted_fup_compounds": int(_has_fup_all.sum()),
                "predicted_fup_values": int(_functional_all[_fup_columns].notna().sum().sum()),
            },
            "enrichment_analysis_compounds": int(len(_functional)),
            "curated_fuc": {
                "count": int(_functional["Metadata_funcuse_category"].notna().sum()),
                "memberships": int(len(_fuc_memberships)),
                "represented_categories": int(_fuc_memberships.nunique()),
            },
            "predicted_fup": {
                "count": int(_has_fup.sum()),
                "classes": len(_fup_columns),
                "values": int(_functional[_fup_columns].notna().sum().sum()),
                "non_null_probabilities_per_compound": {
                    "min": int(_fup_counts.min()),
                    "max": int(_fup_counts.max()),
                },
                "coverage_semantics": (
                    "Endpoint-dependent; missing FUP probabilities are excluded rather than imputed"
                ),
                "dominant_class_method": "argmax over each compound's non-null FUP probabilities",
            },
            "top_dominant_fup_classes": _top_dominant,
            "ranking_evidence_status": _functional_provenance["ranking_evidence_status"],
            "interpretation_caveat": _functional_provenance["interpretation_caveat"],
        },
        "numerical_contract": {
            "path": "references/compound_collection_contract.json",
            "sha256": sha256(CONTRACT_PATH),
            "schema_version": int(_contract["schema_version"]),
        },
        "input_provenance": {
            "chemical_space_summary": ("data/processed/chemical-space/compound_no_source7/chemical_space_summary.json"),
            "metadata_database": "data/interim/jump_metadata_augmented.duckdb",
            "copairs_database": "data/processed/copairs_results.duckdb",
            "functional_use_csv": "data/external/2026-04-06_JUMP_chems_func_use.csv",
            "database_hash_policy": (
                "Paths and canonical query key are recorded instead of byte hashes; "
                "DuckDB files with compressed strings are not byte-reproducible"
            ),
        },
    }

    _actual = flatten(numbers)
    _expected = _contract["expected_claims"]
    _mismatches = {
        claim: {"expected": expected, "actual": _actual.get(claim, "<missing>")}
        for claim, expected in _expected.items()
        if _actual.get(claim, "<missing>") != expected
    }
    if _mismatches:
        raise ValueError("Compound-collection numerical contract failed:\n" + json.dumps(_mismatches, indent=2))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary = write_summary(
        OUTPUT,
        description=(
            "Canonical compound-collection facts: chemical properties, clinical phases, "
            "target/MOA coverage, MolPort availability, sample sourcing and EPA functional use."
        ),
        numbers=numbers,
        files=[],
    )
    computed_claims = {claim: _actual[claim] for claim in _expected}
    return computed_claims, summary


@app.cell
def _(computed_claims, mo, summary):
    mo.vstack(
        [
            mo.md("## Numerical contract passed"),
            mo.ui.table(
                [{"claim": claim, "value": value} for claim, value in computed_claims.items()],
                selection=None,
            ),
            mo.md(f"**EPA functional-use provenance:** `{summary['numbers']['functional_use']['provenance_status']}`"),
        ]
    )
    return



if __name__ == "__main__":
    app.run()
