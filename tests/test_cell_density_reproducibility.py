"""Well ordering must not change the floating-point source adjustment."""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd


def test_well_density_order_is_independent_of_table_insertion():
    path = Path(__file__).resolve().parents[1] / "notebooks/2.02b-ss-cell-count-quality.py"
    spec = importlib.util.spec_from_file_location("cell_density", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def query(reverse):
        with duckdb.connect() as con:
            con.execute("CREATE TABLE well (Metadata_JCP2022 VARCHAR, Metadata_Plate VARCHAR, Metadata_Well VARCHAR)")
            con.execute(
                "CREATE TABLE cell_counts (Metadata_Source VARCHAR, Metadata_Plate VARCHAR, Metadata_Well VARCHAR, Metadata_Count_Cells INTEGER)"
            )
            con.execute(
                "CREATE TABLE microscope_config_fov (Metadata_Source VARCHAR, Metadata_Well_Area_Microns2 DOUBLE)"
            )
            rows = [
                ("compound_b", "plate_b", "A01"),
                ("compound_a", "plate_a", "A02"),
                ("compound_c", "plate_a", "A01"),
            ]
            counts = [
                ("source_2", "plate_b", "A01", 20),
                ("source_1", "plate_a", "A02", 30),
                ("source_1", "plate_a", "A01", 10),
            ]
            con.executemany("INSERT INTO well VALUES (?, ?, ?)", rows[::-1] if reverse else rows)
            con.executemany("INSERT INTO cell_counts VALUES (?, ?, ?, ?)", counts[::-1] if reverse else counts)
            con.executemany("INSERT INTO microscope_config_fov VALUES (?, ?)", [("source_1", 1e6), ("source_2", 2e6)])
            return module.query_cell_density(con, aggregate=False)

    pd.testing.assert_frame_equal(query(False), query(True), check_exact=True)
