# Notebook execution

Build results with Snakemake before opening their notebooks.
Use `marimo run` to view a notebook with cells automatically executed, and `marimo edit` to modify it.
An editor can start with stale cells; run all cells and inspect errors before accepting changes.
Do not edit a notebook file externally while a live editor owns it.

Pure-Python notebooks with PEP 723 dependencies can run with:

```bash
direnv exec . uv run --with marimo marimo run --sandbox notebooks/dataset_reference.py --host 127.0.0.1
```

GPU notebooks importing CuPy or RAPIDS must use the pinned RAPIDS environment without `--sandbox`:

```bash
direnv exec . pixi run -e rapids python -m marimo run notebooks/2.01-ss-umap-visualization.py --host 127.0.0.1
```

Use the cheminformatics environment for scripts that require RDKit:

```bash
direnv exec . pixi run -e cheminformatics python notebooks/0.16-ss-compound-collection-facts.py
```

The full reproduction workflow caps BLAS/OpenMP threads to avoid oversubscription during scaffold cross-validation.
Use `OMP_NUM_THREADS=1` when running those analyses independently.
If uv reports too many open files, run `ulimit -n 65536` before invoking it.
