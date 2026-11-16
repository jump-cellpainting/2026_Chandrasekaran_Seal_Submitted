"""Protect input bootstrapping using anonymous public inputs."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATED_PROFILE_EXCLUDES = (
    "--exclude '*_rsc*.parquet'",
    "--exclude '*_with_source7*.parquet'",
    "--exclude '*_active_union.parquet'",
)


def rendered_recipe(recipe: str, *args: str) -> str:
    result = subprocess.run(
        ["just", "--dry-run", recipe, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize("recipe", ["get-inputs"])
def test_input_bootstrap_uses_raw_profile_path_and_preserves_generated_outputs(recipe):
    rendered = rendered_recipe(recipe)
    profile_sync = next(line for line in rendered.splitlines() if "jump_production_datastore/raw/profiles/" in line)
    for exclusion in GENERATED_PROFILE_EXCLUDES:
        assert exclusion in profile_sync
    assert "jump_production_datastore/profiles/" not in rendered



def test_input_download_is_anonymous_and_deletion_capped():
    rendered = rendered_recipe("get-inputs")
    assert "--s3-env-auth" not in rendered
    assert "--max-delete" in rendered
