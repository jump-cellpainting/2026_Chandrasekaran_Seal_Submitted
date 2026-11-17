"""MitoTox permutation results must not depend on other random draws."""

import numpy as np
from conftest import load_notebook
from scipy.spatial.distance import pdist, squareform


def test_silhouette_permutations_are_repeatable():
    module = load_notebook("2.37-seal-mitotox-morphology.py")
    distances = squareform(pdist(np.random.default_rng(7).normal(size=(20, 3))))
    labels = np.repeat([0, 1], 10)
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first = module.permutation_test_silhouette(distances, labels, n_permutations=30)
        np.random.seed(2)
        second = module.permutation_test_silhouette(distances, labels, n_permutations=30)
    finally:
        np.random.set_state(state)
    assert first[:2] == second[:2]
    np.testing.assert_array_equal(first[2], second[2])
