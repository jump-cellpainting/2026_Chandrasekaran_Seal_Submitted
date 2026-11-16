"""Modeling utilities for JUMP data analysis.

Provides utilities for model training, validation, and evaluation including:
- Cross-validation splitting (scaffold-aware, random, etc.)
"""

from jump_production.modeling.splitting import scaffold_k_fold_split

__all__ = ["scaffold_k_fold_split"]
