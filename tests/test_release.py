"""Release checks reject changes outside the reviewed file and content roster."""

import hashlib
import json
import subprocess

import pytest

from jump_production.release import check_release


def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("Reviewed reproduction guide.\n")
    (tmp_path / "release-files.json").write_text(json.dumps({
        "schema_version": 1,
        "files": {"README.md": hashlib.sha256((tmp_path / "README.md").read_bytes()).hexdigest()},
    }))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def test_reviewed_tree_passes_with_untracked_private_scratch(tmp_path):
    root = repository(tmp_path)
    (root / "scratch").mkdir()
    (root / "scratch/notes.md").write_text("Not part of the index or snapshot.\n")
    assert check_release(root) == 2


@pytest.mark.parametrize("change", ["extra", "changed", "staged_change", "missing", "symlink", "unmerged"])
def test_release_rejects_unreviewed_content(tmp_path, change):
    root = repository(tmp_path)
    if change == "extra":
        (root / "notes.md").write_text("Unreviewed\n")
        subprocess.run(["git", "add", "notes.md"], cwd=root, check=True)
    elif change in {"changed", "staged_change"}:
        (root / "README.md").write_text("Changed content\n")
        if change == "staged_change":
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    elif change == "missing":
        (root / "README.md").rename(root / "saved.md")
    elif change == "symlink":
        (root / "README.md").rename(root / "saved.md")
        (root / "README.md").symlink_to(root / "saved.md")
    else:
        oid = subprocess.check_output(["git", "hash-object", "README.md"], cwd=root, text=True).strip()
        subprocess.run(["git", "update-index", "--index-info"], cwd=root,
                       input=f"0 {'0' * 40}\tREADME.md\n100644 {oid} 1\tREADME.md\n", text=True, check=True)
    with pytest.raises(ValueError):
        check_release(root)
