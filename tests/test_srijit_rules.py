from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_commercial_compounds_declares_copairs_database_dependency():
    rules = (ROOT / "rules" / "srijit.smk").read_text()
    block = rules.split("rule srijit_2_36:", maxsplit=1)[1].split(
        "rule srijit_2_37:", maxsplit=1
    )[0]

    assert 'copairs_db="data/processed/copairs_results.duckdb"' in block
