import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import pandas as pd

    from jump_production import config as _cfg
    from jump_production.summaries import write_summary

    REPO = _cfg.PROJ_ROOT
    METADATA_DB = REPO / "data/interim/jump_metadata_augmented.duckdb"
    # UniChem/MyChem community-curated ID pointers: one row per InChIKey that resolves to at least
    # one external database, one column per database (see data/external/unichem_pointers.csv header).
    UNICHEM_POINTERS = REPO / "data/external/unichem_pointers.csv"
    OUTPUT = REPO / "data/processed/unichem-mapping/summary.json"
    return METADATA_DB, OUTPUT, UNICHEM_POINTERS, duckdb, mo, pd, write_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # External-database mapping coverage (UniChem / MyChem)

        How many JUMP compounds resolve to at least one external chemical database through the
        community-curated UniChem/MyChem ID pointers, and the per-database breakdown. Compounds are
        keyed by InChIKey; the pointer file lists only InChIKeys that map, so a JUMP compound is
        "mapped" iff its InChIKey appears there.
        """
    )
    return


@app.cell
def _(METADATA_DB, OUTPUT, UNICHEM_POINTERS, duckdb, pd, write_summary):
    con = duckdb.connect(str(METADATA_DB), read_only=True)
    compounds = con.execute("SELECT Metadata_JCP2022, Metadata_InChIKey FROM compound").df()
    # Source-7-excluded population: compounds screened on any non-source-7 well. This is the
    # denominator the paper's compound-analysis narrative uses (e.g. the 112,462 analyzed compounds,
    # "source 7 excluded"), so the coverage is reported both ways to stay consistent wherever it is
    # cited. Coverage is recomputed within each population, not merely rescaled.
    non_source7 = {
        row[0]
        for row in con.execute(
            "SELECT DISTINCT w.Metadata_JCP2022 FROM well w "
            "JOIN perturbation p USING (Metadata_JCP2022) "
            "WHERE p.Metadata_perturbation_modality = 'compound' AND w.Metadata_Source <> 'source_7'"
        ).fetchall()
    }
    con.close()

    pointers = pd.read_csv(UNICHEM_POINTERS, low_memory=False)
    # Column 0 is `inchikey`; column 1 is an unnamed blank separator in the source file. Everything
    # else is one external database.
    database_columns = [
        column
        for column in pointers.columns
        if column != "inchikey" and not str(column).startswith("Unnamed")
    ]
    mapped_keys = set(pointers["inchikey"])
    compounds["mapped"] = compounds["Metadata_InChIKey"].isin(mapped_keys)

    def coverage(subset):
        total = int(len(subset))
        mapped = int(subset["mapped"].sum())
        return {
            "total_compounds": total,
            "mapped_compounds": mapped,
            "mapped_percent": round(100 * mapped / total, 2),
            "unmapped_compounds": total - mapped,
        }

    full = coverage(compounds)
    excl_source7 = coverage(compounds[compounds["Metadata_JCP2022"].isin(non_source7)])

    # Per-database compound coverage (full collection). Every pointer InChIKey belongs to exactly one
    # JUMP compound (InChIKey is 1:1 with JCP2022 in this collection), so a non-null cell counts one
    # compound.
    per_database = {
        column: int(pointers[column].notna().sum()) for column in database_columns
    }
    per_database = dict(sorted(per_database.items(), key=lambda item: item[1], reverse=True))

    numbers = {
        # Full collection (source 7 included) - the denominator issue 045 reports against.
        **full,
        # Same coverage restricted to the source-7-excluded population, to match the paper's
        # analyzed-compound framing.
        "excl_source7": excl_source7,
        "n_databases": len(database_columns),
        "databases": database_columns,
        "per_database_compounds": per_database,
        "pointer_source": "data/external/unichem_pointers.csv",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary = write_summary(
        OUTPUT,
        description=(
            f"External-database mapping coverage via community-curated UniChem/MyChem ID pointers. "
            f"Full collection: {full['mapped_compounds']:,} of {full['total_compounds']:,} JUMP "
            f"compounds ({full['mapped_percent']:.1f}%) resolve to at least one of "
            f"{len(database_columns)} external chemical databases (ChEMBL, DrugBank, PubChem, "
            f"BindingDB, CompTox, ...). Excluding source 7 to match the analyzed-compound framing: "
            f"{excl_source7['mapped_compounds']:,} of {excl_source7['total_compounds']:,} "
            f"({excl_source7['mapped_percent']:.1f}%) - the rate is essentially unchanged. Compounds "
            "are keyed by InChIKey (1:1 with JCP2022 here); per_database_compounds gives the compound "
            "count carrying an ID in each database, over the full collection."
        ),
        numbers=numbers,
    )
    return (summary,)


@app.cell
def _(mo, summary):
    import json as _json

    mo.md(f"```json\n{_json.dumps(summary['numbers'], indent=2)}\n```")
    return


if __name__ == "__main__":
    app.run()
