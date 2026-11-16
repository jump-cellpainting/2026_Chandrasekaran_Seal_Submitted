"""Cross-validation splitting utilities for molecular datasets.

Provides scaffold-aware k-fold splitting using precomputed Murcko scaffolds
from the metadata database, avoiding redundant RDKit computation.
"""

from collections import defaultdict

import duckdb
import numpy as np
from loguru import logger

from jump_production.config import INTERIM_DATA_DIR

METADATA_DB = INTERIM_DATA_DIR / "jump_metadata_augmented.duckdb"


def scaffold_k_fold_split(
    smiles: np.ndarray,
    k: int = 5,
) -> list[tuple[list[str], list[str]]]:
    """Generate k-fold CV splits grouping compounds by Murcko scaffold.

    Uses precomputed scaffolds from compound_metadata.Metadata_MurckoScaffold.
    Implements the same algorithm as DeepChem's ScaffoldSplitter:
    1. Group compounds by scaffold
    2. Sort groups by size (largest first)
    3. Distribute to k buckets using greedy balancing

    Args:
        smiles: Array of SMILES strings
        k: Number of folds (default: 5)

    Returns:
        List of k tuples: (train_smiles, test_smiles)

    Example:
        >>> from jump_production.modeling import scaffold_k_fold_split
        >>> for train, test in scaffold_k_fold_split(smiles, k=5):
        ...     model.fit(X[train], y[train])
    """
    # Load precomputed scaffolds from DB
    con = duckdb.connect(str(METADATA_DB), read_only=True)
    df = con.execute(
        """
        SELECT Metadata_SMILES, Metadata_MurckoScaffold
        FROM compound_metadata
        WHERE Metadata_SMILES = ANY(?)
          AND Metadata_MurckoScaffold IS NOT NULL
    """,
        [list(smiles)],
    ).df()
    con.close()

    scaffold_mapping = dict(zip(df["Metadata_SMILES"], df["Metadata_MurckoScaffold"]))
    logger.debug(f"Loaded scaffolds for {len(scaffold_mapping):,} of {len(smiles):,} compounds")

    # Group indices by scaffold
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    no_scaffold_indices: list[int] = []

    for idx, smi in enumerate(smiles):
        scaffold = scaffold_mapping.get(smi)
        if scaffold and scaffold != "":
            scaffold_to_indices[scaffold].append(idx)
        else:
            no_scaffold_indices.append(idx)

    # Sort scaffolds by group size (largest first), then alphabetically for determinism
    sorted_scaffolds = sorted(
        scaffold_to_indices.keys(),
        key=lambda s: (-len(scaffold_to_indices[s]), s),
    )

    # Initialize k buckets and distribute scaffold groups (greedy balancing)
    buckets: list[list[int]] = [[] for _ in range(k)]
    for scaffold in sorted_scaffolds:
        min_bucket = min(range(k), key=lambda i: len(buckets[i]))
        buckets[min_bucket].extend(scaffold_to_indices[scaffold])

    # Distribute compounds without scaffolds
    for idx in no_scaffold_indices:
        min_bucket = min(range(k), key=lambda i: len(buckets[i]))
        buckets[min_bucket].append(idx)

    # Generate train/test splits
    smiles_list = list(smiles)
    fold_splits = []
    for fold_idx in range(k):
        test_ids = [smiles_list[i] for i in buckets[fold_idx]]
        train_ids = [smiles_list[i] for i in sum((buckets[j] for j in range(k) if j != fold_idx), [])]
        fold_splits.append((train_ids, test_ids))

    logger.debug(f"Created {k} folds with sizes: {[len(b) for b in buckets]}")
    return fold_splits
