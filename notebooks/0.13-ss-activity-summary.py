import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import yaml

    from jump_production import config as _cfg
    from jump_production.summaries import index_files, write_summary

    REPO = _cfg.PROJ_ROOT

    # Canonical analysis key - from analysis.yaml, not hardcoded
    _analysis = yaml.safe_load(open(REPO / "configs/snakemake/analysis.yaml"))["analysis"]
    KEY = {k: _analysis[k] for k in ("default_dataset", "default_preprocessing", "default_filter", "default_activity_params")}
    KEY_VALUES = list(KEY.values())

    RES_DB = REPO / "data/processed/copairs_results.duckdb"
    return KEY, KEY_VALUES, REPO, RES_DB, duckdb, index_files, write_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Phenotypic-activity summary (canonical analyzed slice)

    Emits `data/processed/activity-summary/summary.json` in the standard envelope
    (`{description, numbers, files}`) so the dataset-reference hub views it like any other node.

    The numbers come from the copairs activity store (`copairs_results.duckdb`), which is an
    aggregate across many copairs runs - so they cannot live in any single analysis notebook's
    summary. This node gives them a home: it queries the store at the canonical key and writes
    the headline counts as an envelope, rather than the hub special-casing a DB query.
    """)
    return


@app.cell
def _(KEY, KEY_VALUES, REPO, RES_DB, duckdb, index_files, write_summary):
    import hashlib

    res = duckdb.connect(str(RES_DB), read_only=True)

    n_analyzed = int(res.execute(
        "SELECT COUNT(DISTINCT Metadata_JCP2022) FROM activity_results "
        "WHERE _dataset=? AND _preprocessing=? AND _filter=? AND _activity_params=?",
        KEY_VALUES).fetchone()[0])
    n_active = int(res.execute(
        "SELECT COUNT(*) FROM activity_results "
        "WHERE _dataset=? AND _preprocessing=? AND _filter=? AND _activity_params=? AND below_corrected_p",
        KEY_VALUES).fetchone()[0])
    res.close()

    # Canonical profile object: the exact raw parquet these numbers derive from, with its public S3
    # mirror and content md5 - so downstream tools (jump_rr, jump_hub) can point at one verified
    # artifact. Path/URL/hash are all derived from the dataset name (no hardcoded profile).
    _dataset = KEY["default_dataset"]
    _profile_path = REPO / f"data/raw/profiles/{_dataset}.parquet"
    _s3_key_prefix = "cpg0016-jump/source_all/workspace/publication_data/2025_Chandrasekaran/jump_production_datastore"
    _md5 = hashlib.md5()
    with open(_profile_path, "rb") as _fh:
        for _chunk in iter(lambda: _fh.read(1 << 20), b""):
            _md5.update(_chunk)
    profile_object = {
        "scenario": _dataset,
        "path": f"data/raw/profiles/{_dataset}.parquet",
        "s3_url": f"https://cellpainting-gallery.s3.amazonaws.com/{_s3_key_prefix}/raw/profiles/{_dataset}.parquet",
        "size_bytes": _profile_path.stat().st_size,
        "md5": _md5.hexdigest(),
    }

    numbers = {
        "dataset": KEY["default_dataset"],
        "preprocessing": KEY["default_preprocessing"],
        "filter": KEY["default_filter"],
        "activity_params": KEY["default_activity_params"],
        "analyzed_compounds": n_analyzed,
        "phenotypically_active_compounds": n_active,
        "active_rate_percent": round(100 * n_active / n_analyzed, 1),
        "profile_object": profile_object,
    }
    description = (
        "Phenotypic-activity headline for the canonical analyzed slice: compounds tested for "
        "activity (mAP vs negative controls) and the subset significant after multiple-testing "
        "correction (below_corrected_p), queried from the copairs activity store at the canonical "
        f"key ({KEY['default_dataset']} / {KEY['default_preprocessing']} / {KEY['default_filter']} / "
        f"{KEY['default_activity_params']}). Also records the canonical profile object these numbers "
        "derive from (raw parquet path, public S3 URL, size, content md5) so downstream tools can "
        "point at one verified artifact."
    )

    output_dir = REPO / "data/processed/activity-summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_summary(
        output_dir / "summary.json",
        description=description,
        numbers=numbers,
        files=index_files(output_dir),
    )
    return (summary,)


@app.cell
def _(mo, summary):
    mo.vstack([
        mo.md("## Emitted `data/processed/activity-summary/summary.json`"),
        mo.md(f"*{summary['description']}*"),
        mo.ui.table([{"key": k, "value": v} for k, v in summary["numbers"].items()], selection=None),
    ])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
