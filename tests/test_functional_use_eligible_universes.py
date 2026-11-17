import duckdb
import numpy as np
import pandas as pd
import pytest
from conftest import load_notebook

MODULE = load_notebook("2.42-seal-functional-use-classification.py")


def test_functional_use_table_is_a_required_workflow_input():
    connection = duckdb.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="functional_use table not in augmented DB"):
            MODULE.load_functional_use_data(connection)
    finally:
        connection.close()


def test_fup_enrichment_excludes_missing_predictions_from_comparator():
    observed = pd.DataFrame(
        {
            "endpoint": [0.9, 0.1, None, None],
            "is_active": [True, False, True, False],
        }
    )

    result = MODULE.compute_enrichment(observed, "endpoint", threshold=0.5, min_compounds=1).iloc[0]

    assert result["n_eligible"] == 2
    assert result["n_missing"] == 2
    assert result["n_high"] == 1
    assert result["n_rest"] == 1
    assert result["n_active_high"] == 1
    assert result["n_active_rest"] == 0
    assert result["frac_active_rest"] == 0
    assert np.isfinite(result["odds_ratio"])
    assert np.isfinite(result["log2_odds_ratio"])


def test_fuc_enrichment_uses_exact_tokens_and_only_annotated_compounds():
    observed = pd.DataFrame(
        {
            "functioncategory_all": ["Fuel", "Fuel agents", "Fuel | Other", None],
            "is_active": [True, False, False, True],
        }
    )

    result = MODULE.compute_fuc_enrichments(observed, min_compounds=1)
    fuel = result.loc[result["category"] == "Fuel"].iloc[0]

    assert fuel["n_eligible"] == 3
    assert fuel["n_missing"] == 1
    assert fuel["n_high"] == 2
    assert fuel["n_rest"] == 1
    assert fuel["n_active_high"] == 1
    assert fuel["n_active_rest"] == 0
    assert np.isfinite(fuel["odds_ratio"])
    assert np.isfinite(fuel["log2_odds_ratio"])
