"""Shared loader for the analysis scripts under notebooks/, whose file names are not importable."""

import importlib.util
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def load_notebook(filename: str):
    """Import one notebooks/ script as a module without running its CLI."""
    name = Path(filename).stem.replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, NOTEBOOKS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
