"""Cell-count signal spans several PCs, and removing it does not improve mixing.

The primary consequence analysis used a nearly complete shared-compound
JUMP-Target-2 design. Of 302 compounds, 301 were profiled in all seven included
sources and one was profiled in six. Source 9 was excluded because its Target-2
plates had near-zero treatment activity despite strong positive-control
performance on ordinary plates, consistent with a Target-2-specific treatment
failure (see the published preprint, doi:10.1101/2023.03.23.534023). The analysis compared the saved
post-Harmony 50-PC representation with deletion of PC1, PC2, or PC50 and with
linear cell-count residualization of all 50 PCs. Normalized median iLISI and
kBET measured source mixing as defined in notebook 2.02c; cross-source retrieval
nMAP measured retrieval of the same compound from other sources. The all-well
comparison was secondary because compound composition differs by source.

PC1 retained a substantial association with cell count within source-plate
groups. All 754,741 wells joined to their compound-level median cell count
without missing values. To isolate the within-source-plate relationship, the
analysis subtracted each source-plate mean from both the PC score and log1p cell
count before calculating adjusted r. This adjustment prevents source or plate
offsets shared by both variables from creating a pooled correlation. Partial R2
is adjusted r squared: the fraction of the remaining PC variation statistically
associated with count. Across all wells, PC1 had adjusted r=0.391 (partial
R2=0.153; leave-one-source-out r=0.382-0.406). Among 37,227 Target-2 wells, PC1
had adjusted r=0.583 (partial R2=0.340). PC2 had a lower adjusted r=0.505
(partial R2=0.255), so PC1 ranked first among all 50 axes by absolute adjusted
r, although the association was not unique to PC1.

The stronger hypothesis that PC1 is essentially cell count was tested as a
concentration claim: PC1 alone should account for most adjusted log-count
variance available to a linear model of all 50 PCs, while PCs 2-50 should
account for little without it. In Target-2, PC1 alone accounted for 34.0%, PCs
2-50 jointly accounted for 69.5%, and all 50 accounted for 77.1%. PCs 2-50
therefore retained 90.2% of the full model's R2, and adding PC1 last increased
R2 by 7.6 percentage points (6.5-8.1 when each source was omitted in turn), the
largest unique increment of any PC. The corresponding all-well values were
15.3%, 26.9%, 42.1%, and 15.2 points, with 63.9% of full-model R2 retained
without PC1. Source-plate demeaning makes the PCs correlated, so single-PC R2
values are not additive; the nested-model increase defines the unique
increment. PC1 is the strongest one-dimensional readout of count, but count
information is distributed across the saved representation rather than
concentrated in PC1.

Deleting PC1 consistently worsened Target-2 source mixing. Across verified GPU
runs, normalized median iLISI fell from 0.3607-0.3611 to 0.3289-0.3293, and
kBET acceptance fell from 0.0441-0.0442 to 0.0365-0.0376. Cross-source
retrieval nMAP changed only from 0.209 to 0.211. Deleting PC2 also worsened
mixing, but less than deleting PC1; deleting PC50 left mixing near baseline.
Cell-count residualization also left mixing near baseline (iLISI
0.3603-0.3623; kBET 0.0407-0.0414), but reduced retrieval nMAP to 0.198
(-5.25%). The all-well sensitivity analysis also found no mixing benefit.

The analysis is limited to a compound-level median count covariate, repeated
across wells of each compound, and the saved 50-PC post-Harmony representation.
The joint R2 estimates are descriptive, in-sample, and weighted by wells. The
analysis contains no negative controls and cannot establish that cell count
mediates residual source separation. It also does not determine what should be
done before Harmony or in canonical high-dimensional profiles. IVF-Flat
neighbor search is approximate, so exact mixing estimates vary slightly between
runs; their direction remained stable.

Run with:
    direnv exec . scripts/gpu_lease.sh pixi run -e rapids python \
        notebooks/0.21-ss-pc1-batch-ablation.py

Set JUMP_DATA_ROOT to read data from another checkout without changing it.
Outputs: scratch/pc1-cell-count/*.csv
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import anndata as ad
import duckdb
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
from loguru import logger
from scipy import sparse
from scipy.stats import chi2, pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from jump_production.config import PROJ_ROOT

DATA_ROOT = Path(os.environ.get("JUMP_DATA_ROOT", PROJ_ROOT / "data"))
INPUT_PATH = DATA_ROOT / "interim" / "anndata" / "compound_no_source7_cosine_all_umap.h5ad"
DB_PATH = DATA_ROOT / "interim" / "jump_metadata_augmented.duckdb"
OUTPUT_DIR = PROJ_ROOT / "scratch" / "pc1-cell-count"
N_NEIGHBORS = 90
N_PERMUTATIONS = 100
KBET_SUBSAMPLE = 50_000
SEED = 42
TARGET2_REPRESENTATIONS = (
    "all_50_pcs",
    "drop_pc1",
    "drop_pc2",
    "drop_pc50",
    "cell_count_residualized_50_pcs",
)
ALL_WELL_REPRESENTATIONS = ("all_50_pcs", "drop_pc1", "cell_count_residualized_50_pcs")


def load_profiles_and_metadata(
    input_path: Path,
    db_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Load saved PCA coordinates and attach cell-count and plate metadata."""
    logger.info(f"Loading saved PCA coordinates from {input_path}")
    adata = ad.read_h5ad(input_path)
    coordinates = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    if coordinates.shape[1] != 50:
        raise ValueError(f"Expected 50 saved PCs, found {coordinates.shape[1]}")

    obs = pd.DataFrame(adata.obs).reset_index(drop=True).copy()
    obs["Source"] = obs["Source"].astype(str)
    obs["Plate"] = obs["Plate"].astype(str)
    obs["JCP2022"] = obs["JCP2022"].astype(str)
    pc1_variance_ratio = float(adata.uns["pca"]["variance_ratio"][0])
    del adata

    with duckdb.connect(str(db_path), read_only=True) as connection:
        compound = connection.sql(
            """
            SELECT
                Metadata_JCP2022 AS JCP2022,
                Metadata_median_cell_count AS median_cell_count
            FROM compound_metadata
            """
        ).fetchdf()
        plate = connection.sql(
            """
            SELECT
                Metadata_Source AS Source,
                Metadata_Plate AS Plate,
                Metadata_Batch AS Batch,
                Metadata_PlateType AS PlateType
            FROM plate
            """
        ).fetchdf()

    obs = obs.merge(compound, on="JCP2022", how="left", validate="many_to_one")
    obs = obs.merge(plate, on=["Source", "Plate"], how="left", validate="many_to_one")
    missing_plate = int(obs["PlateType"].isna().sum())
    if missing_plate:
        raise ValueError(f"Plate metadata is missing for {missing_plate:,} wells")
    if len(obs) != len(coordinates):
        raise AssertionError("Metadata join changed the number or order of observations")
    obs["log1p_median_cell_count"] = np.log1p(obs["median_cell_count"])
    return coordinates, obs, pc1_variance_ratio


def group_codes(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return stable integer codes for one or more categorical columns."""
    index = pd.MultiIndex.from_frame(frame[columns])
    return pd.factorize(index, sort=True)[0].astype(np.int32)


def demean_by_group(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Remove group means from a finite one-dimensional array."""
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=values) / counts
    return values - means[codes]


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation for two nonconstant arrays."""
    return float(pearsonr(first, second).statistic)


def adjusted_correlation(
    log_counts: np.ndarray,
    values: np.ndarray,
    frame: pd.DataFrame,
) -> float:
    """Correlate after removing source-plate means from both variables."""
    plate_codes = group_codes(frame, ["Source", "Plate"])
    return correlation(
        demean_by_group(log_counts, plate_codes),
        demean_by_group(values, plate_codes),
    )


def r2_from_moments(
    gram: np.ndarray,
    cross_product: np.ndarray,
    total_sum_squares: float,
    columns: np.ndarray,
) -> float:
    """Return no-intercept linear R2 for selected already-centered columns."""
    selected_gram = gram[np.ix_(columns, columns)]
    coefficients = np.linalg.solve(selected_gram, cross_product[columns])
    return float(cross_product[columns] @ coefficients / total_sum_squares)


def count_concentration(
    log_counts: np.ndarray,
    coordinates: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Measure how much adjusted count signal is unique to each saved PC."""
    plate_codes = group_codes(frame, ["Source", "Plate"])
    adjusted_counts = demean_by_group(log_counts, plate_codes)
    adjusted_coordinates = np.empty(coordinates.shape, dtype=float)
    for index in range(coordinates.shape[1]):
        adjusted_coordinates[:, index] = demean_by_group(coordinates[:, index].astype(float), plate_codes)

    gram = adjusted_coordinates.T @ adjusted_coordinates
    cross_product = adjusted_coordinates.T @ adjusted_counts
    total_sum_squares = float(adjusted_counts @ adjusted_counts)
    all_columns = np.arange(coordinates.shape[1])
    axis_correlations = cross_product / np.sqrt(np.diag(gram) * total_sum_squares)
    full_r2 = r2_from_moments(gram, cross_product, total_sum_squares, all_columns)
    without_pc1_r2 = r2_from_moments(gram, cross_product, total_sum_squares, all_columns[1:])
    unique_increments = np.array(
        [
            full_r2
            - r2_from_moments(
                gram,
                cross_product,
                total_sum_squares,
                np.delete(all_columns, index),
            )
            for index in all_columns
        ]
    )

    leave_one_source_out = []
    for source in sorted(frame["Source"].unique()):
        keep = frame["Source"].ne(source).to_numpy()
        kept_coordinates = adjusted_coordinates[keep]
        kept_counts = adjusted_counts[keep]
        kept_gram = kept_coordinates.T @ kept_coordinates
        kept_cross_product = kept_coordinates.T @ kept_counts
        kept_total_sum_squares = float(kept_counts @ kept_counts)
        kept_full_r2 = r2_from_moments(kept_gram, kept_cross_product, kept_total_sum_squares, all_columns)
        kept_without_pc1_r2 = r2_from_moments(
            kept_gram,
            kept_cross_product,
            kept_total_sum_squares,
            all_columns[1:],
        )
        leave_one_source_out.append(kept_full_r2 - kept_without_pc1_r2)

    unique_order = np.argsort(unique_increments)[::-1]
    return axis_correlations, {
        "pcs2_50_joint_source_plate_adjusted_r2": without_pc1_r2,
        "all_50_joint_source_plate_adjusted_r2": full_r2,
        "pcs2_50_fraction_of_all_50_count_r2": without_pc1_r2 / full_r2,
        "pc1_unique_incremental_source_plate_adjusted_r2": full_r2 - without_pc1_r2,
        "pc1_unique_increment_rank_among_50": int(np.flatnonzero(unique_order == 0)[0] + 1),
        "second_largest_pc_unique_increment_pc": int(unique_order[1] + 1),
        "second_largest_pc_unique_increment_r2": float(unique_increments[unique_order[1]]),
        "median_pc_unique_increment_r2": float(np.median(unique_increments)),
        "pc1_unique_increment_leave_one_source_out_min": float(min(leave_one_source_out)),
        "pc1_unique_increment_leave_one_source_out_max": float(max(leave_one_source_out)),
    }


def analyze_associations(
    subset: str,
    coordinates: np.ndarray,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one aggregate row plus within-source stability rows."""
    observed = frame["median_cell_count"].notna() & frame["median_cell_count"].ge(0)
    missing = int((~observed).sum())
    frame = frame.loc[observed].reset_index(drop=True)
    coordinates = coordinates[observed.to_numpy()]
    counts = frame["median_cell_count"].to_numpy(dtype=float)
    log_counts = frame["log1p_median_cell_count"].to_numpy(dtype=float)
    pc1 = coordinates[:, 0].astype(float)
    compound = (
        pd.DataFrame({"JCP2022": frame["JCP2022"], "log_count": log_counts, "pc1": pc1})
        .groupby("JCP2022", sort=True)
        .agg(log_count=("log_count", "first"), pc1=("pc1", "median"))
    )
    axis_correlations, concentration = count_concentration(log_counts, coordinates, frame)
    adjusted_r = float(axis_correlations[0])
    leave_one_source_out = []
    for source in sorted(frame["Source"].unique()):
        keep = frame["Source"].ne(source).to_numpy()
        leave_one_source_out.append(
            adjusted_correlation(
                log_counts[keep],
                pc1[keep],
                frame.loc[keep].reset_index(drop=True),
            )
        )
    rows = [
        {
            "subset": subset,
            "source": "all",
            "n_wells": len(frame),
            "n_compounds": int(frame["JCP2022"].nunique()),
            "n_sources": int(frame["Source"].nunique()),
            "n_source_plates": int(frame[["Source", "Plate"]].drop_duplicates().shape[0]),
            "n_cell_count_missing": missing,
            "pearson_pc1_raw_count": correlation(counts, pc1),
            "pearson_pc1_log1p_count": correlation(log_counts, pc1),
            "spearman_pc1_count": float(spearmanr(log_counts, pc1).statistic),
            "compound_median_pc1_pearson_log1p_count": correlation(
                compound["log_count"].to_numpy(), compound["pc1"].to_numpy()
            ),
            "compound_median_pc1_spearman_count": float(spearmanr(compound["log_count"], compound["pc1"]).statistic),
            "source_plate_adjusted_r": adjusted_r,
            "source_plate_adjusted_partial_r2": adjusted_r**2,
            "adjusted_r_leave_one_source_out_min": float(min(leave_one_source_out)),
            "adjusted_r_leave_one_source_out_max": float(max(leave_one_source_out)),
            "pc1_abs_rank_among_50_adjusted": int((np.abs(axis_correlations) >= abs(axis_correlations[0])).sum()),
            "pc2_source_plate_adjusted_r": float(axis_correlations[1]),
            "pc2_source_plate_adjusted_partial_r2": float(axis_correlations[1] ** 2),
            "pc50_source_plate_adjusted_r": float(axis_correlations[49]),
            **concentration,
        }
    ]
    for source, positions in frame.groupby("Source", sort=True).indices.items():
        indices = np.asarray(positions)
        source_frame = frame.iloc[indices].reset_index(drop=True)
        source_counts = log_counts[indices]
        source_pc1 = pc1[indices]
        source_r = adjusted_correlation(source_counts, source_pc1, source_frame)
        rows.append(
            {
                "subset": subset,
                "source": source,
                "n_wells": len(source_frame),
                "n_compounds": int(source_frame["JCP2022"].nunique()),
                "n_sources": 1,
                "n_source_plates": int(source_frame["Plate"].nunique()),
                "n_cell_count_missing": 0,
                "pearson_pc1_log1p_count": correlation(source_counts, source_pc1),
                "spearman_pc1_count": float(spearmanr(source_counts, source_pc1).statistic),
                "source_plate_adjusted_r": source_r,
                "source_plate_adjusted_partial_r2": source_r**2,
            }
        )
    return pd.DataFrame(rows)


def residualize_cell_count(coordinates: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Remove pooled linear within-plate cell-count effects from every PC.

    Only the fitted cell-count term is subtracted. Source-plate means remain, so
    this transform does not directly force the source-mixing endpoints upward.
    """
    if frame["median_cell_count"].isna().any():
        raise ValueError("Cell-count residualization requires complete cell-count data")
    log_counts = frame["log1p_median_cell_count"].to_numpy(dtype=float)
    plate_codes = group_codes(frame, ["Source", "Plate"])
    centered_counts = demean_by_group(log_counts, plate_codes)
    denominator = float(np.dot(centered_counts, centered_counts))
    coefficients = centered_counts @ coordinates / denominator
    result = coordinates.copy()
    result -= centered_counts.astype(np.float32)[:, None] * coefficients.astype(np.float32)[None, :]
    remaining_covariance = centered_counts @ result
    remaining_scale = np.sqrt(denominator * np.sum(np.square(result), axis=0))
    max_abs_correlation = float(np.max(np.abs(remaining_covariance / remaining_scale)))
    if max_abs_correlation > 1e-5:
        raise AssertionError(f"Cell-count residualization left correlation {max_abs_correlation:.2g}")
    return result


def ablate(coordinates: np.ndarray, dropped_index: int | None) -> np.ndarray:
    """Return all PCs or a view with one PC removed."""
    if dropped_index is None:
        return coordinates
    keep = np.arange(coordinates.shape[1]) != dropped_index
    result = coordinates[:, keep]
    if result.shape[1] != coordinates.shape[1] - 1:
        raise AssertionError("A one-PC ablation must remove exactly one column")
    return result


def compute_knn_graph(coordinates: np.ndarray, n_neighbors: int) -> sparse.csr_matrix:
    """Build the same cosine IVF-Flat kNN graph used by notebook 2.02c."""
    logger.info(f"Computing cosine kNN graph for {len(coordinates):,} wells x {coordinates.shape[1]} PCs")
    adata = ad.AnnData(X=np.ascontiguousarray(coordinates, dtype=np.float32))
    adata.obsm["X_pca"] = adata.X
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        metric="cosine",
        algorithm="ivfflat",
    )
    rsc.get.anndata_to_CPU(adata)
    graph = sparse.csr_matrix(adata.obsp["distances"])
    graph.data = np.ones(graph.nnz, dtype=np.int16)
    del adata
    gc.collect()
    return graph


def neighbor_label_counts(
    graph: sparse.csr_matrix,
    labels: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Count categorical neighbor labels for every graph row."""
    membership = sparse.csr_matrix(
        (np.ones(len(labels), dtype=np.int16), (np.arange(len(labels)), labels)),
        shape=(len(labels), n_labels),
    )
    return (graph @ membership).toarray()


def source_mixing_metrics(
    graph: sparse.csr_matrix,
    source_codes: np.ndarray,
    query_indices: np.ndarray,
) -> tuple[float, float]:
    """Return normalized median iLISI and the kBET acceptance rate."""
    n_sources = int(source_codes.max()) + 1
    counts = neighbor_label_counts(graph, source_codes, n_sources)
    neighbor_totals = counts.sum(axis=1)
    proportions = counts / neighbor_totals[:, None]
    lisi = 1.0 / np.square(proportions).sum(axis=1)
    ilisi_normalized = float((np.median(lisi) - 1.0) / (n_sources - 1))

    global_proportions = np.bincount(source_codes, minlength=n_sources) / len(source_codes)
    query_counts = counts[query_indices]
    expected = query_counts.sum(axis=1)[:, None] * global_proportions[None, :]
    chi_square = np.sum(
        np.divide(
            np.square(query_counts - expected),
            expected,
            out=np.zeros_like(expected, dtype=float),
            where=expected > 0,
        ),
        axis=1,
    )
    acceptance = float(np.mean(chi2.sf(chi_square, n_sources - 1) > 0.05))
    return ilisi_normalized, acceptance


def within_compound_source_permutations(
    compounds: pd.Series,
    source_codes: np.ndarray,
    n_permutations: int,
    seed: int,
) -> list[np.ndarray]:
    """Shuffle source labels within compounds while preserving their source counts."""
    groups = list(pd.Series(np.arange(len(compounds))).groupby(compounds, sort=False).agg(list))
    group_indices = [np.asarray(group, dtype=np.int32) for group in groups]
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(n_permutations):
        permuted = source_codes.copy()
        for group in group_indices:
            permuted[group] = rng.permutation(source_codes[group])
        permutations.append(permuted)
    return permutations


def normalized_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Normalize Average Precision relative to random-ranking prevalence."""
    average_precision = float(average_precision_score(y_true, scores))
    prevalence = float(np.mean(y_true))
    return (average_precision - prevalence) / (1.0 - prevalence)


def cross_source_retrieval_nmap(coordinates: np.ndarray, obs: pd.DataFrame) -> float:
    """Return mean cross-source retrieval nMAP from source-compound median profiles."""
    frame = pd.DataFrame(coordinates)
    frame.insert(0, "Source", obs["Source"].to_numpy())
    frame.insert(0, "JCP2022", obs["JCP2022"].to_numpy())
    consensus = frame.groupby(["JCP2022", "Source"], observed=True, sort=True).median()
    values = consensus.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A source-compound consensus has zero norm")
    values /= norms
    similarity = values @ values.T

    compounds = consensus.index.get_level_values("JCP2022").to_numpy()
    sources = consensus.index.get_level_values("Source").to_numpy()
    scores = []
    for query in range(len(consensus)):
        candidates = sources != sources[query]
        positives = compounds[candidates] == compounds[query]
        if not positives.any():
            continue
        scores.append(normalized_average_precision(positives, similarity[query, candidates]))
    if not scores:
        raise ValueError("No cross-source compound matches were available for retrieval")
    return float(np.mean(scores))


def summarize_null(values: list[float]) -> tuple[float, float, float]:
    """Return median and central 95 percent interval for shuffled-label metrics."""
    median, lower, upper = np.quantile(values, [0.5, 0.025, 0.975])
    return float(median), float(lower), float(upper)


def analyze_target2(
    coordinates: np.ndarray,
    obs: pd.DataFrame,
    n_neighbors: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Run the shared-compound consequence test for five representations."""
    categories = sorted(obs["Source"].unique())
    source_codes = pd.Categorical(obs["Source"], categories=categories).codes.astype(np.int16)
    if np.any(source_codes < 0):
        raise AssertionError("Every TARGET2 well must have a source code")
    compounds = pd.Series(obs["JCP2022"].to_numpy(), name="JCP2022")
    permutations = within_compound_source_permutations(compounds, source_codes, n_permutations, seed)
    query_indices = np.arange(len(obs), dtype=np.int32)
    residualized = residualize_cell_count(coordinates, obs)
    representations = {
        "all_50_pcs": coordinates,
        "drop_pc1": ablate(coordinates, 0),
        "drop_pc2": ablate(coordinates, 1),
        "drop_pc50": ablate(coordinates, 49),
        "cell_count_residualized_50_pcs": residualized,
    }
    rows = []

    for name in TARGET2_REPRESENTATIONS:
        representation = representations[name]
        graph = compute_knn_graph(representation, n_neighbors)
        ilisi, kbet = source_mixing_metrics(graph, source_codes, query_indices)
        null_metrics = [source_mixing_metrics(graph, permuted, query_indices) for permuted in permutations]
        ilisi_null = summarize_null([metric[0] for metric in null_metrics])
        kbet_null = summarize_null([metric[1] for metric in null_metrics])
        retrieval = cross_source_retrieval_nmap(representation, obs)
        rows.append(
            {
                "subset": "target2_no_source9",
                "representation": name,
                "n_wells": len(obs),
                "n_sources": len(categories),
                "n_compounds": int(obs["JCP2022"].nunique()),
                "n_dimensions": representation.shape[1],
                "iLISI_normalized": ilisi,
                "iLISI_null_median": ilisi_null[0],
                "iLISI_null_q025": ilisi_null[1],
                "iLISI_null_q975": ilisi_null[2],
                "kBET_acceptance": kbet,
                "kBET_null_median": kbet_null[0],
                "kBET_null_q025": kbet_null[1],
                "kBET_null_q975": kbet_null[2],
                "cross_source_retrieval_nMAP": retrieval,
            }
        )
        del graph
        gc.collect()

    results = pd.DataFrame(rows)
    baseline = results.loc[results["representation"] == "all_50_pcs"].iloc[0]
    for metric, null_column in (
        ("iLISI_normalized", "iLISI_null_median"),
        ("kBET_acceptance", "kBET_null_median"),
    ):
        change_column = f"{metric}_change_vs_all_50"
        results[change_column] = results[metric] - baseline[metric]
        results[f"{metric}_null_gap_closed"] = results[change_column] / (results[null_column] - baseline[metric])
    results["retrieval_change_pct"] = (
        100.0
        * (results["cross_source_retrieval_nMAP"] - baseline["cross_source_retrieval_nMAP"])
        / abs(baseline["cross_source_retrieval_nMAP"])
    )
    return results


def analyze_all_wells(
    coordinates: np.ndarray,
    obs: pd.DataFrame,
    n_neighbors: int,
    kbet_subsample: int,
    seed: int,
) -> pd.DataFrame:
    """Compare baseline, PC1 deletion, and residualization on all wells."""
    categories = sorted(obs["Source"].unique())
    source_codes = pd.Categorical(obs["Source"], categories=categories).codes.astype(np.int16)
    rng = np.random.default_rng(seed)
    n_query = min(kbet_subsample, len(obs))
    query_indices = np.sort(rng.choice(len(obs), size=n_query, replace=False)).astype(np.int32)
    residualized = residualize_cell_count(coordinates, obs)
    representations = {
        "all_50_pcs": coordinates,
        "drop_pc1": ablate(coordinates, 0),
        "cell_count_residualized_50_pcs": residualized,
    }
    rows = []

    for name in ALL_WELL_REPRESENTATIONS:
        representation = representations[name]
        graph = compute_knn_graph(representation, n_neighbors)
        ilisi, kbet = source_mixing_metrics(graph, source_codes, query_indices)
        rows.append(
            {
                "subset": "all_wells_sensitivity",
                "representation": name,
                "n_wells": len(obs),
                "n_sources": len(categories),
                "n_compounds": int(obs["JCP2022"].nunique()),
                "n_dimensions": representation.shape[1],
                "iLISI_normalized": ilisi,
                "kBET_acceptance": kbet,
            }
        )
        del graph
        gc.collect()

    return pd.DataFrame(rows)


def main() -> None:
    """Run the saved-PC1 cell-count association and consequence analysis."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coordinates, obs, pc1_variance_ratio = load_profiles_and_metadata(INPUT_PATH, DB_PATH)
    target2_mask = obs["PlateType"].eq("TARGET2") & obs["Source"].ne("source_9")
    target2_coordinates = coordinates[target2_mask.to_numpy()]
    target2_obs = obs.loc[target2_mask].reset_index(drop=True)
    logger.info(
        f"Primary subset: {len(target2_obs):,} wells, {target2_obs['JCP2022'].nunique():,} compounds, "
        f"{target2_obs['Source'].nunique()} sources, {target2_obs['Batch'].nunique()} batches"
    )
    if target2_obs["Source"].nunique() < 2 or target2_obs["JCP2022"].nunique() < 2:
        raise ValueError("TARGET2 subset lacks the cross-source structure needed for this test")

    all_association = analyze_associations(
        "all_wells",
        coordinates,
        obs,
    )
    target2_association = analyze_associations(
        "target2_no_source9",
        target2_coordinates,
        target2_obs,
    )
    associations = pd.concat([all_association, target2_association], ignore_index=True)
    associations.to_csv(OUTPUT_DIR / "associations.csv", index=False)

    primary = analyze_target2(
        target2_coordinates,
        target2_obs,
        n_neighbors=N_NEIGHBORS,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )
    sensitivity = analyze_all_wells(
        coordinates,
        obs,
        n_neighbors=N_NEIGHBORS,
        kbet_subsample=KBET_SUBSAMPLE,
        seed=SEED,
    )
    pd.concat([primary, sensitivity], ignore_index=True).to_csv(
        OUTPUT_DIR / "consequence_metrics.csv",
        index=False,
    )

    target2 = primary.set_index("representation")
    baseline = target2.loc["all_50_pcs"]
    pc1 = target2.loc["drop_pc1"]
    residualized = target2.loc["cell_count_residualized_50_pcs"]
    all_wells = sensitivity.set_index("representation")
    association = associations.loc[associations["source"].eq("all")].set_index("subset")
    logger.success(
        "PC1 explains {:.1%} of variance. Adjusted PC1-log(count) r: all {:.3f}, Target-2 {:.3f}. "
        "Target-2 baseline/drop-PC1/residualized: iLISI {:.4f}/{:.4f}/{:.4f}, "
        "kBET {:.4f}/{:.4f}/{:.4f}, retrieval nMAP {:.4f}/{:.4f}/{:.4f}. "
        "All-well iLISI {:.4f}/{:.4f}/{:.4f}, kBET {:.4f}/{:.4f}/{:.4f}.",
        pc1_variance_ratio,
        association.loc["all_wells", "source_plate_adjusted_r"],
        association.loc["target2_no_source9", "source_plate_adjusted_r"],
        baseline["iLISI_normalized"],
        pc1["iLISI_normalized"],
        residualized["iLISI_normalized"],
        baseline["kBET_acceptance"],
        pc1["kBET_acceptance"],
        residualized["kBET_acceptance"],
        baseline["cross_source_retrieval_nMAP"],
        pc1["cross_source_retrieval_nMAP"],
        residualized["cross_source_retrieval_nMAP"],
        all_wells.loc["all_50_pcs", "iLISI_normalized"],
        all_wells.loc["drop_pc1", "iLISI_normalized"],
        all_wells.loc["cell_count_residualized_50_pcs", "iLISI_normalized"],
        all_wells.loc["all_50_pcs", "kBET_acceptance"],
        all_wells.loc["drop_pc1", "kBET_acceptance"],
        all_wells.loc["cell_count_residualized_50_pcs", "kBET_acceptance"],
    )
    logger.info(f"Results written under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
