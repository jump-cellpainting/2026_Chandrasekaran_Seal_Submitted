"""Compare phenotypic activity/consistency across profiles.

Queries the copairs_results.duckdb to compare profiles pairwise.
Produces threshold curves, overlap analysis, and UpSet plots.

Usage:
    pixi run python notebooks/1.01-jfh-profile-comparison.py activity -p1 compound_DL_CPCNN_no_source7 -p2 compound_no_source7
    pixi run python notebooks/1.01-jfh-profile-comparison.py consistency_moa -p1 compound_DL_CPCNN_no_source7 -p2 compound_no_source7

Outputs to:
    data/processed/profile-comparison/{p1}_vs_{p2}/activity/
    data/processed/profile-comparison/{p1}_vs_{p2}/consistency/{grouping}/{distance}/
"""

import os
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from upsetplot import UpSet, from_indicators

from jump_production.config import DEFAULT_DPI
from jump_production.queries import query_activity_results, query_consistency_results
from jump_production.summaries import index_files, write_summary

warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot")
pd.set_option("future.no_silent_downcasting", True)

DEFAULT_DB = "data/processed/copairs_results.duckdb"


def query_results(
    db_path: str,
    analysis_type: str,
    profiles: list[str],
    preprocessing: str = "activity_no_target2",
    filter_name: str = "all_sources",
    distance: str = "cosine",
    activity_params: str = "default",
) -> dict[str, pd.DataFrame]:
    """Query results from DuckDB for given profiles.

    Args:
        db_path: Path to copairs_results.duckdb
        analysis_type: 'activity' or 'consistency_<target>'
        profiles: List of profile names to query
        preprocessing: Preprocessing config name
        filter_name: Filter config name
        distance: Distance metric for consistency queries (cosine, abs_cosine)
        activity_params: Activity params config (default, withinsource, crosssource)

    Returns:
        Dict mapping profile name to DataFrame
    """
    con = duckdb.connect(db_path, read_only=True)

    results = {}
    if analysis_type == "activity":
        for profile in profiles:
            df = query_activity_results(con, profile, preprocessing, filter_name, activity_params)
            if len(df) > 0:
                results[profile] = df
    else:
        # consistency_<target>
        target = analysis_type.removeprefix("consistency_")
        cons_preprocessing = preprocessing.replace("activity_", "consistency_")
        for profile in profiles:
            df = query_consistency_results(con, profile, cons_preprocessing, filter_name, target, distance)
            if len(df) > 0:
                results[profile] = df

    con.close()
    return results


def get_available_analysis_types(db_path: str) -> list[str]:
    """Get available analysis types from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)

    types = ["activity"]

    # Get unique consistency group types (exclude sweep runs)
    try:
        targets = con.execute(
            "SELECT DISTINCT _group_type FROM consistency_results WHERE _preprocessing NOT LIKE '%_sweep' ORDER BY _group_type"
        ).fetchall()
        types.extend([f"consistency_{t[0]}" for t in targets])
    except duckdb.CatalogException:
        pass

    con.close()
    return types


def get_significance_summary(data: dict[str, pd.DataFrame], thresholds: list[float]) -> pd.DataFrame:
    """Calculate fraction significant at each p-value threshold."""
    rows = []
    for name, df in data.items():
        if "corrected_p_value" not in df.columns:
            continue
        row = {"profile": name}
        for t in thresholds:
            row[f"p<{t}"] = (df["corrected_p_value"] < t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_threshold_curves(
    summary_df: pd.DataFrame, thresholds: list[float], title: str, output_path: str, dpi: int = DEFAULT_DPI
):
    """Plot fraction significant vs p-value threshold."""
    plt.figure(figsize=(8, 5))
    for _, row in summary_df.iterrows():
        fractions = [row[f"p<{t}"] for t in thresholds]
        plt.plot(thresholds, fractions, marker="o", label=row["profile"], linewidth=2)
    plt.xscale("log")
    plt.xlabel("Corrected p-value threshold")
    plt.ylabel("Fraction significant")
    plt.title(f"Significance by threshold: {title}")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def create_overlap_analysis(
    data: dict[str, pd.DataFrame],
    output_prefix: str,
    title: str,
    analysis_type: str,
    quiet: bool = False,
    dpi: int = DEFAULT_DPI,
) -> pd.DataFrame | None:
    """
    Create overlap analysis between two profiles.

    Produces:
    - Details CSV with per-item significance and category
    - Summary CSV with counts per category
    - UpSet plot showing overlaps
    """
    if len(data) != 2:
        print(f"Overlap analysis requires exactly 2 profiles, got {len(data)}")
        return None

    names = list(data.keys())
    dfs = list(data.values())

    # Determine key column based on analysis type
    if analysis_type == "activity":
        key_col = "Metadata_JCP2022"
    else:
        # For consistency, use the normalized group_value column
        key_col = "group_value"

    if key_col not in dfs[0].columns or "below_corrected_p" not in dfs[0].columns:
        print(f"Required columns not found (need {key_col} and below_corrected_p)")
        return None

    # Merge the two dataframes on the key column
    merged = (
        dfs[0][[key_col, "below_corrected_p"]]
        .rename(columns={"below_corrected_p": names[0]})
        .merge(
            dfs[1][[key_col, "below_corrected_p"]].rename(columns={"below_corrected_p": names[1]}),
            on=key_col,
            how="outer",
        )
    )
    merged[names[0]] = merged[names[0]].fillna(False).astype(bool)
    merged[names[1]] = merged[names[1]].fillna(False).astype(bool)

    # Vectorized category assignment
    merged["category"] = np.select(
        [
            merged[names[0]] & merged[names[1]],
            merged[names[0]] & ~merged[names[1]],
            ~merged[names[0]] & merged[names[1]],
        ],
        ["both", f"only_{names[0]}", f"only_{names[1]}"],
        default="neither",
    )

    # Save details
    merged.to_csv(f"{output_prefix}/overlap_details.csv", index=False)

    # Create and save summary with concordance
    summary = merged["category"].value_counts().reset_index()
    summary.columns = ["category", "count"]

    # Calculate concordance: (both_agree + neither_agree) / total
    counts = dict(zip(summary["category"], summary["count"]))
    total = summary["count"].sum()
    both = counts.get("both", 0)
    neither = counts.get("neither", 0)
    concordance = round((both + neither) / total, 4) if total > 0 else 0.0

    # Add concordance as a row
    concordance_row = pd.DataFrame([{"category": "concordance", "count": concordance}])
    summary = pd.concat([summary, concordance_row], ignore_index=True)
    summary.to_csv(f"{output_prefix}/overlap_summary.csv", index=False)

    # UpSet plot (exclude items not significant in either)
    sig_items = merged[merged[names[0]] | merged[names[1]]]
    if len(sig_items) > 0:
        try:
            upset_data = from_indicators(names, data=sig_items)
            fig = plt.figure(figsize=(14, 6))
            upset = UpSet(upset_data, show_counts=True)
            axes = upset.plot(fig=fig)
            axes["matrix"].tick_params(axis="y", labelsize=8)
            plt.suptitle(f"Overlap: {title}")
            plt.savefig(f"{output_prefix}/overlap_upset.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        except (AttributeError, ValueError, TypeError) as e:
            # upsetplot can fail with certain data configurations
            if not quiet:
                print(f"  Skipping UpSet plot: {e}")

    return summary


def main(
    db_path: str,
    analysis_type: str,
    profile1: str,
    profile2: str,
    output_path: str | None = None,
    quiet: bool = False,
    preprocessing: str = "activity_no_target2",
    distance: str = "cosine",
    activity_params: str = "default",
    dpi: int = DEFAULT_DPI,
):
    """Run comparison analysis between two profiles."""
    if output_path is None:
        output_path = "data/processed/profile-comparison"

    # Build hierarchical output path:
    # profile-comparison/{profile1}_vs_{profile2}/activity/{activity_params}/
    # profile-comparison/{profile1}_vs_{profile2}/consistency/{grouping}/{distance}/
    comparison_dir = f"{output_path}/{profile1}_vs_{profile2}"
    if analysis_type == "activity":
        output_dir = f"{comparison_dir}/activity/{activity_params}"
    else:
        grouping = analysis_type.removeprefix("consistency_")
        output_dir = f"{comparison_dir}/consistency/{grouping}/{distance}"

    os.makedirs(output_dir, exist_ok=True)
    output_prefix = output_dir

    # Query data from DuckDB
    data = query_results(
        db_path,
        analysis_type,
        [profile1, profile2],
        preprocessing=preprocessing,
        distance=distance,
        activity_params=activity_params,
    )

    if len(data) < 2:
        print(f"Warning: Only found data for {list(data.keys())}. Need both profiles.")
        return

    # Threshold analysis
    thresholds = [0.5, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.001, 0.0001]
    summary_df = get_significance_summary(data, thresholds)
    summary_df.to_csv(f"{output_prefix}/active_at_threshold.csv", index=False)

    # Threshold curve plot
    plot_threshold_curves(summary_df, thresholds, analysis_type, f"{output_prefix}/threshold_curve.png", dpi=dpi)

    # Overlap analysis
    overlap_summary = create_overlap_analysis(data, output_prefix, analysis_type, analysis_type, quiet=quiet, dpi=dpi)

    # Emit the standard summary envelope so the dataset-reference hub discovers this leaf.
    if overlap_summary is not None:
        cat_counts = {
            str(row["category"]): int(row["count"])
            for _, row in overlap_summary.iterrows()
            if row["category"] != "concordance"
        }
        total = sum(cat_counts.values())
        # Concordance fraction, recomputed from the real category counts (also stored as the
        # "concordance" row in overlap_summary.csv).
        agree = cat_counts.get("both", 0) + cat_counts.get("neither", 0)
        concordance_fraction = round(agree / total, 4) if total else 0.0

        def _num(v):
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, (np.floating, float)):
                return round(float(v), 4)
            return v

        write_summary(
            f"{output_prefix}/summary.json",
            description=(
                f"Profile-comparison ({analysis_type}): agreement between two profile representations "
                f"({profile1} vs {profile2}) on which items are called significant. Reports overlap "
                "counts (both / only-each / neither), the concordance fraction, and the fraction "
                "significant at each p-value threshold - used to compare Cell Painting vs deep-learning "
                "embeddings."
            ),
            numbers={
                "analysis_type": analysis_type,
                "profile1": profile1,
                "profile2": profile2,
                "distance": None if analysis_type == "activity" else distance,
                "n_items": total,
                "overlap_counts": cat_counts,
                "concordance_fraction": concordance_fraction,
                "active_at_threshold": [
                    {k: _num(v) for k, v in rec.items()} for rec in summary_df.to_dict("records")
                ],
            },
            files=index_files(Path(output_prefix)),
        )


app = typer.Typer()


@app.command()
def run(
    analysis_type: str = typer.Argument("activity", help="Type: activity or consistency_<target>"),
    profile1: str = typer.Option("compound_DL_CPCNN_no_source7", "--profile1", "-p1"),
    profile2: str = typer.Option("compound_no_source7", "--profile2", "-p2"),
    db_path: str = typer.Option(DEFAULT_DB, "--db", "-d", help="Path to copairs_results.duckdb"),
    output_path: str | None = typer.Option(None, "--output", "-o"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress verbose output"),
    preprocessing: str = typer.Option(
        "activity_no_target2", "--preprocessing", "-pp", help="Preprocessing config name"
    ),
    distance: str = typer.Option("cosine", "--distance", help="Distance metric for consistency (cosine, abs_cosine)"),
    activity_params: str = typer.Option(
        "default", "--activity-params", help="Activity params (default, withinsource, crosssource)"
    ),
    dpi: int = typer.Option(
        DEFAULT_DPI,
        "--dpi",
        help="DPI for saved figures",
    ),
):
    """Run profile phenotypic comparison analysis."""
    if not os.path.exists(db_path):
        typer.echo(f"Error: Database not found: {db_path}", err=True)
        raise typer.Exit(1)

    valid_types = get_available_analysis_types(db_path)
    if analysis_type not in valid_types:
        typer.echo(f"Error: Invalid type '{analysis_type}'. Valid: {', '.join(valid_types)}", err=True)
        raise typer.Exit(1)

    if not quiet:
        typer.echo(f"Running {analysis_type}: {profile1} vs {profile2}")
    main(
        db_path,
        analysis_type,
        profile1,
        profile2,
        output_path,
        quiet=quiet,
        preprocessing=preprocessing,
        distance=distance,
        activity_params=activity_params,
        dpi=dpi,
    )


if __name__ == "__main__":
    app()
