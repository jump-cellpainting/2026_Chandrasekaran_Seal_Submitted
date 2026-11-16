from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_profiles_to_anndata_excludes_umap_output_stems():
    rules = (ROOT / "rules" / "anndata.smk").read_text()

    assert 'dataset=r"(?!.*_(?:cosine|euclidean)_(?:all|active)_umap)[^/]+"' in rules
