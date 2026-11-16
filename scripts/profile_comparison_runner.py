#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run pairwise profile comparisons for activity and consistency.

This script discovers available (group_type, distance) combinations from the
copairs results database and runs the profile comparison notebook for each.

Usage:
    pixi run python scripts/profile_comparison_runner.py \
        --db data/processed/copairs_results.duckdb \
        --script notebooks/1.01-jfh-profile-comparison.py
"""

import subprocess
import sys

import duckdb
import typer
from loguru import logger

# Profile comparison configuration (mirrors analysis.smk)
# Activity: use standard profiles (union profiles don't have activity data)
COMPARISON_PROFILES_ACTIVITY = [
    ("compound_no_source7", "compound_DL_CPCNN_no_source7"),
]
# Consistency: use union-filtered profiles (all perturbations active in either method)
COMPARISON_PROFILES_CONSISTENCY = [
    ("compound_no_source7_active_union", "compound_DL_CPCNN_no_source7_active_union"),
]


def main(
    db: str = typer.Option(..., "--db", help="Path to copairs results database"),
    script: str = typer.Option(..., "--script", help="Path to profile comparison script"),
) -> None:
    """Run pairwise profile comparisons for activity and consistency."""
    # Discover available (group_type, distance) combinations from database
    logger.info(f"Reading combinations from {db}")
    con = duckdb.connect(db, read_only=True)
    group_distance_combos = con.execute(
        "SELECT DISTINCT _group_type, _distance FROM consistency_results WHERE _preprocessing NOT LIKE '%_sweep' ORDER BY _group_type, _distance"
    ).fetchall()
    con.close()
    logger.info(f"Found {len(group_distance_combos)} (group_type, distance) combinations")

    # Run activity comparisons for standard profiles
    logger.info("Running activity comparisons (standard profiles)...")
    for p1, p2 in COMPARISON_PROFILES_ACTIVITY:
        cmd = [
            "pixi",
            "run",
            "python",
            script,
            "activity",
            "--profile1",
            p1,
            "--profile2",
            p2,
            "--db",
            db,
            "--quiet",
        ]
        logger.info(f"  activity: {p1} vs {p2}")
        subprocess.run(cmd, check=True)

    # Run consistency comparisons for union-filtered profiles
    logger.info(f"Running {len(group_distance_combos)} consistency comparisons (union profiles)...")
    for p1, p2 in COMPARISON_PROFILES_CONSISTENCY:
        for group_type, distance in group_distance_combos:
            analysis_type = f"consistency_{group_type}"
            cmd = [
                "pixi",
                "run",
                "python",
                script,
                analysis_type,
                "--profile1",
                p1,
                "--profile2",
                p2,
                "--db",
                db,
                "--quiet",
                "--preprocessing",
                "activity_prefiltered",
                "--distance",
                distance,
            ]
            logger.info(f"  {analysis_type} ({distance}): {p1} vs {p2}")
            subprocess.run(cmd, check=True)

    logger.success("Profile comparison complete")


if __name__ == "__main__":
    typer.run(main)
