"""Protect input bootstrapping using anonymous public inputs."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED_PROFILE_EXCLUDES = (
    "--exclude '*_rsc*.parquet'",
    "--exclude '*_with_source7*.parquet'",
    "--exclude '*_active_union.parquet'",
)


def test_input_download_is_anonymous_capped_and_preserves_generated_profiles():
    result = subprocess.run(["just", "--dry-run", "get-inputs"], cwd=ROOT, check=True, capture_output=True, text=True)
    rendered = result.stdout + result.stderr
    profile_sync = next(line for line in rendered.splitlines() if "jump_production_datastore/raw/profiles/" in line)
    for exclusion in GENERATED_PROFILE_EXCLUDES:
        assert exclusion in profile_sync
    assert "jump_production_datastore/profiles/" not in rendered
    assert "--s3-env-auth" not in rendered
    assert "--max-delete" in rendered
