import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "references/compound_collection_contract.json"


def test_compound_collection_summary_is_in_canonical_manifest():
    from snakemake.api import SnakemakeApi, OutputSettings, ResourceSettings

    with SnakemakeApi(OutputSettings()) as api:
        workflow = api.workflow(snakefile=ROOT / "Snakefile", resource_settings=ResourceSettings())._workflow
        rule = workflow.get_rule("dataset_reference_manifest")
        assert "data/processed/compound-collection-facts/summary.json" in rule.input


def test_numerical_contract_rejects_changed_computation():
    tree = ast.parse((ROOT / "notebooks/0.16-ss-compound-collection-facts.py").read_text())
    flatten = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "flatten")
    scope = {}
    exec(compile(ast.Module(body=[flatten], type_ignores=[]), "<flatten>", "exec"), scope)
    assert scope["flatten"]({"population": {"analyzed_compounds": 112462}}) == {"population.analyzed_compounds": 112462}
    # Execute the actual contract comparison block with one mutated computed value.
    start = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_actual" for t in n.targets))
    cell = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and start in n.body)
    end = next(i for i, n in enumerate(cell.body) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "_mismatches")
    block = ast.Module(body=cell.body[cell.body.index(start):end + 1], type_ignores=[])
    code = compile(block, "<numerical-contract>", "exec")
    contract = json.loads(CONTRACT_PATH.read_text())
    # Flat keys are also valid flatten inputs, allowing all expectations to be exercised.
    values = dict(contract["expected_claims"])
    env = {**scope, "json": json, "_contract": contract, "numbers": values}
    exec(code, env)
    values["population.analyzed_compounds"] += 1
    with pytest.raises(ValueError, match="numerical contract failed"):
        exec(code, env)


def test_functional_use_contract_preserves_input_and_population_caveats():
    contract = json.loads(CONTRACT_PATH.read_text())
    provenance = contract["functional_use_provenance"]
    expected = contract["expected_claims"]
    assert provenance["source_url"].endswith("/external/2026-04-06_JUMP_chems_func_use.csv")
    assert expected["functional_use.enrichment_analysis_compounds"] == 8300
    assert expected["functional_use.curated_fuc.count"] == 427
    assert expected["functional_use.predicted_fup.count"] == 7651
    assert "can overlap" in provenance["interpretation_caveat"]
